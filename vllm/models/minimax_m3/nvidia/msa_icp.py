# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup checks and the native writer binding for MSA's ICP indexer."""

from functools import cache
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

ICP_DEVICE_PLAN_ABI = 4


@cache
def require_icp_writer() -> "type[IcpWriter]":
    """Reject stale vLLM binaries before the compound cache is bound."""
    import torch

    from vllm import _custom_ops  # noqa: F401  (registers torch.ops._C)

    requirement = (
        "msa_icp requires vLLM's fused MiniMax-M3 writer with fragment, live "
        "metadata and device-plan ABI 4. Rebuild vLLM's native extension from "
        "the companion source change."
    )
    try:
        schema = torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert.default._schema
    except AttributeError as exc:
        raise RuntimeError(requirement) from exc
    arguments = {arg.name: arg for arg in schema.arguments}
    required = {
        # The previous region-major writer used k_scale/v_scale.
        "kv_k_scale",
        "kv_v_scale",
        "index_rank",
        "index_world_size",
        "write_icp_metadata",
        "icp_query_start_loc",
        "icp_seq_lens",
        "icp_plan_abi",
    }
    if not required.issubset(arguments) or (
        arguments["icp_plan_abi"].default_value != ICP_DEVICE_PLAN_ABI
    ):
        raise RuntimeError(requirement)
    return IcpWriter


class IcpWriter:
    """Native writer launch with fixed per-layer values pre-ordered by schema.

    Calls pass only per-step values; the op is dispatched positionally and
    arguments nobody passes are None (optional) or the schema default.
    """

    def __init__(self, **fixed: Any) -> None:
        self._fixed = fixed
        self._template: list[Any] | None = None

    def _bind(self) -> list[Any]:
        import torch

        self._op = torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert.default
        arguments = self._op._schema.arguments
        self._index = {arg.name: i for i, arg in enumerate(arguments)}
        unknown = set(self._fixed) - set(self._index)
        if unknown:
            raise TypeError(f"unknown ICP writer arguments: {sorted(unknown)}")
        self._template = [
            self._fixed.get(arg.name, arg.default_value) for arg in arguments
        ]
        return self._template

    def __call__(self, **step: Any) -> None:
        values = (self._template or self._bind()).copy()
        index = self._index
        for name, value in step.items():
            values[index[name]] = value
        self._op(*values)


@cache
def require_icp_prewarm(rank: int) -> None:
    """Fail startup unless the writer is current and MSA's ICP cache is prewarmed."""
    require_icp_writer()
    from vllm.third_party.fmha_sm100.icp import prewarm

    problems = prewarm.check()
    if problems:
        raise RuntimeError(
            "ICP kernels are not fully prewarmed for this image; serving would "
            f"JIT-compile: {problems}"
        )
    logger.info("ICP_PREWARM_VERIFIED rank=%d", rank)
