# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup admission for MSA's optional distributed indexer support."""

import importlib
import os
from functools import cache
from threading import Lock
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

MSA_ICP_INTEGRATION_ABI = 1
MSA_PUBLIC_VLLM_ABI = 1
MSA_ICP_INDEXER_RUNTIME_ABI = 1
ICP_DEVICE_PLAN_ABI = 4
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
        # The previous region-major writer used k_scale/v_scale. Require the
        # public prefix as well as the unchanged index device-plan ABI.
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

    def __call__(self, *, icp_device_plan=None, **step: Any) -> None:
        """Launch; a device plan is checked first and marked only on success."""
        values = (self._template or self._bind()).copy()
        index = self._index
        for name, value in step.items():
            values[index[name]] = value
        if icp_device_plan is None:
            self._op(*values)
            return
        positions = step.get("icp_positions")
        if not step.get("write_icp_metadata") or positions is None:
            raise ValueError(
                "icp_device_plan requires write_icp_metadata=True and icp_positions"
            )
        if icp_device_plan.abi_version != ICP_DEVICE_PLAN_ABI:
            raise RuntimeError("MiniMax-M3 native writer requires device-plan ABI 4")
        shape = dict(
            chunk_width=step.get("icp_chunk_width", 0), num_rows=positions.numel()
        )
        icp_device_plan.check_writer_launch(**shape)
        for name, value in icp_device_plan.writer_kwargs().items():
            values[index[name]] = value
        self._op(*values)
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
