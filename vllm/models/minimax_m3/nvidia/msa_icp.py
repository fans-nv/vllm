# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup admission for MSA's optional distributed indexer support."""

import importlib
import os
from collections.abc import Callable
from functools import cache
from threading import Lock
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

MSA_ICP_INTEGRATION_ABI = 1
MSA_PUBLIC_VLLM_ABI = 1
MSA_ICP_INDEXER_RUNTIME_ABI = 1
ICP_DEVICE_PLAN_ABI = 3
_msa_import_lock = Lock()


@cache
def require_msa_icp() -> Any:
    """Check MSA capabilities before loading native indexing or transport code."""
    requirement = (
        "msa_icp requires fmha_sm100.icp with "
        f"ICP_INTEGRATION_ABI={MSA_ICP_INTEGRATION_ABI} and "
        f"PUBLIC_VLLM_ABI={MSA_PUBLIC_VLLM_ABI} and "
        f"ICP_INDEXER_RUNTIME_ABI={MSA_ICP_INDEXER_RUNTIME_ABI}. "
        "Install the MSA build described in requirements/minimax_m3_icp.txt."
    )
    try:
        # Suppress canonical MSA's unrelated minfer.ops registrations. The
        # shared Q8KV4 native API registers its own isolated callbacks.
        with _msa_import_lock:
            previous = os.environ.get("MSA_REGISTER_TVM_FFI")
            os.environ["MSA_REGISTER_TVM_FFI"] = "0"
            try:
                package = importlib.import_module("fmha_sm100.icp")
            finally:
                if previous is None:
                    os.environ.pop("MSA_REGISTER_TVM_FFI", None)
                else:
                    os.environ["MSA_REGISTER_TVM_FFI"] = previous
    except ImportError as exc:
        raise RuntimeError(f"{requirement} Import failed: {exc}") from exc
    abi = getattr(package, "ICP_INTEGRATION_ABI", None)
    if type(abi) is not int or abi != MSA_ICP_INTEGRATION_ABI:
        raise RuntimeError(f"{requirement} Found ICP_INTEGRATION_ABI={abi!r}.")
    public_abi = getattr(package, "PUBLIC_VLLM_ABI", None)
    if type(public_abi) is not int or public_abi != MSA_PUBLIC_VLLM_ABI:
        raise RuntimeError(f"{requirement} Found PUBLIC_VLLM_ABI={public_abi!r}.")
    runtime_abi = getattr(package, "ICP_INDEXER_RUNTIME_ABI", None)
    if type(runtime_abi) is not int or runtime_abi != MSA_ICP_INDEXER_RUNTIME_ABI:
        raise RuntimeError(
            f"{requirement} Found ICP_INDEXER_RUNTIME_ABI={runtime_abi!r}."
        )
    return package


@cache
def require_icp_writer() -> Callable[..., None]:
    """Reject stale vLLM binaries before the compound cache is bound."""
    import torch

    from vllm import _custom_ops  # noqa: F401  (registers torch.ops._C)

    requirement = (
        "msa_icp requires vLLM's fused MiniMax-M3 writer with fragment, live "
        "metadata, device-plan ABI 3, and explicit PDL control. Rebuild vLLM's "
        "native extension from the companion source change."
    )
    try:
        schema = torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert.default._schema
    except AttributeError as exc:
        raise RuntimeError(requirement) from exc
    arguments = {arg.name: arg for arg in schema.arguments}
    required = {
        # The previous region-major writer used k_scale/v_scale. Require the
        # public prefix as well as the unchanged index device-plan ABI.
        "kv_k_scale",
        "kv_v_scale",
        "index_block_tokens",
        "index_rows_per_rank",
        "index_rank",
        "index_world_size",
        "write_icp_metadata",
        "icp_query_start_loc",
        "icp_seq_lens",
        "enable_pdl",
        "icp_plan_abi",
    }
    if not required.issubset(arguments) or (
        arguments["icp_plan_abi"].default_value != ICP_DEVICE_PLAN_ABI
    ):
        raise RuntimeError(requirement)
    return icp_kv_insert


def icp_kv_insert(*, icp_device_plan=None, **kwargs) -> None:
    """Launch the native writer with ICP arguments; mark the plan only on success."""
    import torch

    kwargs.setdefault("index_slot_mapping", None)
    if kwargs.get("index_world_size") or kwargs.get("write_icp_metadata"):
        kwargs.setdefault("icp_plan_abi", ICP_DEVICE_PLAN_ABI)
    shape = None
    if icp_device_plan is not None:
        positions = kwargs.get("icp_positions")
        if not kwargs.get("write_icp_metadata") or positions is None:
            raise ValueError(
                "icp_device_plan requires write_icp_metadata=True and icp_positions"
            )
        if icp_device_plan.abi_version != ICP_DEVICE_PLAN_ABI:
            raise RuntimeError("MiniMax-M3 native writer requires device-plan ABI 3")
        shape = dict(
            chunk_width=kwargs.get("icp_chunk_width", 0),
            num_rows=positions.numel(),
            row_begin=kwargs.get("icp_plan_row_begin", 0),
        )
        icp_device_plan.check_writer_launch(**shape)
        kwargs.update(icp_device_plan.writer_kwargs())
    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(**kwargs)
    if shape is not None:
        icp_device_plan.note_writer_launch(**shape)


@cache
def require_icp_prewarm(rank: int) -> None:
    """Fail startup unless the writer is current and MSA's ICP cache is prewarmed."""
    require_icp_writer()
    try:
        from fmha_sm100.icp import prewarm
    except ImportError as exc:
        raise RuntimeError(f"ICP prewarm verification is unavailable: {exc}") from exc
    problems = prewarm.check()
    if problems:
        raise RuntimeError(
            "ICP kernels are not fully prewarmed for this image; serving would "
            f"JIT-compile: {problems}"
        )
    logger.info("ICP_PREWARM_VERIFIED rank=%d", rank)
