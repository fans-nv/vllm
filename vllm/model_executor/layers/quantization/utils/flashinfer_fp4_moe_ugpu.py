# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split TRTLLM-Gen NVFP4 MoE weights across the GPU's locality domains.

Each domain owns an equal share of the FC1 output channels (matching up/gate
pairs) and of the FC2 output rows. The shards are shuffled for TRTLLM-Gen and
stored in that domain's memory pool from the `flashinfer_ugpu` runtime.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    shuffle_trtllm_fp4_moe_gemm,
)
from vllm.model_executor.utils import is_weights_pre_processed
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

logger = init_logger(__name__)

# Keeps every shard's rows a multiple of the TRTLLM-Gen 128-row shuffle tile.
UGPU_CHANNEL_ALIGNMENT = 256
UGPU_NUM_PARTITIONS = 2


@dataclass
class UgpuMoeLocalization:
    """A layer's localized MoE weights and the runtime that owns their memory."""

    runtime: Any
    """`flashinfer_ugpu.UgpuRuntime`."""
    shards: Any
    """`flashinfer_ugpu.UgpuMoeShards`."""
    full_weights_released: bool


def get_ugpu_moe_localization(layer: torch.nn.Module) -> UgpuMoeLocalization | None:
    return getattr(layer, "ugpu_moe_localization", None)


def reject_pre_processed_ugpu_moe_weights() -> None:
    """Fail closed when localization is requested for weights that skip the
    kernel-format conversion (weight cache IPC loader), and so admission."""
    if envs.VLLM_MOE_UGPU_LOCALIZATION and is_weights_pre_processed():
        raise ValueError(
            "VLLM_MOE_UGPU_LOCALIZATION=1 but this NVFP4 TRTLLM-Gen MoE layer "
            "cannot be localized: its weights are pre-processed (weight cache). "
            "Unset VLLM_MOE_UGPU_LOCALIZATION to serve it with the stock kernel."
        )


def get_ugpu_moe_runtime(
    layer: "RoutedExperts",
    w13: torch.Tensor,
    w2: torch.Tensor,
    is_act_and_mul: bool,
    per_token_activation: bool,
) -> Any | None:
    """Return the locality runtime if localization is requested, else None.

    Every layer that reaches this check is a candidate: a routed MoE layer whose
    NVFP4 weights are being prepared for the FlashInfer TRTLLM-Gen backend.
    With `VLLM_MOE_UGPU_LOCALIZATION=1`, a candidate that cannot be localized
    raises instead of silently serving the stock kernel.
    """
    if not envs.VLLM_MOE_UGPU_LOCALIZATION:
        return None

    from vllm.config import get_current_vllm_config_or_none

    hidden_size = w2.size(1)
    intermediate_size = w2.size(2) * 2
    parallel_config = layer.moe_config.moe_parallel_config
    vllm_config = get_current_vllm_config_or_none()
    runtime = None
    reason = None
    if not (layer.activation.is_gated and is_act_and_mul):
        reason = "the activation is not gated"
    elif hidden_size % UGPU_CHANNEL_ALIGNMENT or (
        intermediate_size % UGPU_CHANNEL_ALIGNMENT
    ):
        reason = (
            f"hidden size {hidden_size} or intermediate size {intermediate_size} "
            f"is not a multiple of {UGPU_CHANNEL_ALIGNMENT}"
        )
    elif per_token_activation:
        reason = "per-token activation quantization is unsupported"
    elif parallel_config.use_ep or parallel_config.enable_eplb:
        reason = "expert parallelism and EPLB are unsupported"
    elif layer.moe_config.has_bias:
        reason = "expert bias is unsupported"
    elif vllm_config is None:
        reason = "no vLLM config is set"
    elif vllm_config.model_config.enable_sleep_mode:
        reason = "sleep mode is unsupported"
    elif not current_platform.is_device_capability(107):
        reason = "it requires SM107"
    else:
        runtime, reason = _load_flashinfer_ugpu(w13.device)
    if reason is not None:
        raise ValueError(
            "VLLM_MOE_UGPU_LOCALIZATION=1 but this NVFP4 TRTLLM-Gen MoE layer "
            f"cannot be localized: {reason}. Unset VLLM_MOE_UGPU_LOCALIZATION to "
            "serve it with the stock kernel."
        )
    return runtime


def _load_flashinfer_ugpu(device: torch.device) -> tuple[Any, str | None]:
    """Return the locality runtime with its operator module loaded, or the
    reason it is unusable."""
    try:
        import flashinfer_ugpu
        from flashinfer_ugpu.moe import get_module
    except ImportError as err:
        return None, f"flashinfer_ugpu is not importable ({err})"
    runtime = flashinfer_ugpu.get_runtime(device)
    if runtime is None or runtime.num_partitions != UGPU_NUM_PARTITIONS:
        return None, (
            "the device has no two-domain locality runtime "
            f"(last error: {flashinfer_ugpu.last_error(device)})"
        )
    # Load the operator now: release frees the full weights before any forward.
    try:
        get_module()
    except Exception as err:
        return None, (
            f"the flashinfer_ugpu operator module failed to load ({err!r}; "
            f"last error: {flashinfer_ugpu.last_error(device)})"
        )
    return runtime, None


def split_nvfp4_moe_weights_for_ugpu(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    partition: int,
    num_partitions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice one partition out of logical [up | gate] w13 and w2 weights.

    FC1 keeps matching up/gate channel pairs; FC2 keeps its partition's output
    rows and the full intermediate K.
    """
    intermediate = w13.size(1) // 2
    assert w13.size(1) == w13_scale.size(1) == 2 * intermediate
    assert w2.size(1) == w2_scale.size(1)
    assert intermediate % num_partitions == 0 and w2.size(1) % num_partitions == 0
    channels = intermediate // num_partitions
    rows = w2.size(1) // num_partitions
    up = slice(partition * channels, (partition + 1) * channels)
    gate = slice(intermediate + up.start, intermediate + up.stop)
    fc2 = slice(partition * rows, (partition + 1) * rows)
    return (
        torch.cat((w13[:, up], w13[:, gate]), dim=1),
        torch.cat((w13_scale[:, up], w13_scale[:, gate]), dim=1),
        w2[:, fc2].contiguous(),
        w2_scale[:, fc2].contiguous(),
    )


def _shuffled_partitions(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    num_partitions: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    hidden_size = w13.size(2) * 2
    intermediate_size = w2.size(2) * 2
    permute_indices_cache: dict[tuple[object, ...], torch.Tensor] = {}
    for p in range(num_partitions):
        w13_p, w13_scale_p, w2_p, w2_scale_p = split_nvfp4_moe_weights_for_ugpu(
            w13, w13_scale, w2, w2_scale, p, num_partitions
        )
        yield (
            *shuffle_trtllm_fp4_moe_gemm(
                w13_p,
                w13_scale_p,
                w13_p.size(1),
                hidden_size,
                is_gemm1=True,
                is_gated_activation=True,
                permute_indices_cache=permute_indices_cache,
            ),
            *shuffle_trtllm_fp4_moe_gemm(
                w2_p,
                w2_scale_p,
                w2_p.size(1),
                intermediate_size,
                is_gemm1=False,
                permute_indices_cache=permute_indices_cache,
            ),
        )


def _shard_tensors(shards: Any) -> list[tuple[torch.Tensor, ...]]:
    """Per-partition (FC1 weights, FC1 scales, FC2 weights, FC2 scales)."""
    return list(
        zip(
            shards.gemm1_weights,
            shards.gemm1_weights_scale,
            shards.gemm2_weights,
            shards.gemm2_weights_scale,
        )
    )


def prepare_ugpu_nvfp4_moe_shards(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    runtime: Any,
) -> Any:
    """Build shuffled per-partition shards of logical [up | gate] weights in
    `runtime.mem_pool(p)`."""
    from flashinfer_ugpu import UgpuMoeShards

    localized: list[list[torch.Tensor]] = [[], [], [], []]
    for p, tensors in enumerate(
        _shuffled_partitions(w13, w13_scale, w2, w2_scale, runtime.num_partitions)
    ):
        for dst, t in zip(localized, tensors):
            with torch.cuda.use_mem_pool(runtime.mem_pool(p)):
                dst.append(torch.empty_like(t))
            dst[-1].copy_(t)
    return UgpuMoeShards(
        gemm1_weights=localized[0],
        gemm1_weights_scale=localized[1],
        gemm2_weights=localized[2],
        gemm2_weights_scale=localized[3],
    )


def localize_nvfp4_moe_weights(
    layer: "RoutedExperts",
    runtime: Any,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> bool:
    """Attach or refresh the layer's shards; return True if the full weights
    are released."""
    localization = get_ugpu_moe_localization(layer)
    if localization is not None:
        # Reload: rewrite in place so captured CUDA graphs keep valid pointers.
        shards = _shard_tensors(localization.shards)
        new = _shuffled_partitions(w13, w13_scale, w2, w2_scale, len(shards))
        for old_tensors, new_tensors in zip(shards, new):
            for old, t in zip(old_tensors, new_tensors):
                assert old.shape == t.shape and old.dtype == t.dtype
                old.copy_(t)
        return localization.full_weights_released

    release = not envs.VLLM_MOE_UGPU_KEEP_FULL_WEIGHTS
    if release:
        # The locality pools cannot reuse blocks the default pool still caches.
        torch.accelerator.empty_cache()
    shards = prepare_ugpu_nvfp4_moe_shards(w13, w13_scale, w2, w2_scale, runtime)
    layer.ugpu_moe_localization = UgpuMoeLocalization(runtime, shards, release)
    logger.info_once(
        "Localized NVFP4 MoE weights across %d locality domains (%s full weights).",
        runtime.num_partitions,
        "released" if release else "kept",
    )
    return release


def has_ugpu_moe_localization(model: torch.nn.Module) -> bool:
    return any(get_ugpu_moe_localization(m) is not None for m in model.modules())
