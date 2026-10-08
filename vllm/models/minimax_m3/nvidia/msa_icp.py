# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup admission for MSA's optional distributed indexer support."""

import importlib
import os
from collections.abc import Callable
from functools import cache
from threading import Lock
from typing import Any

MSA_ICP_INTEGRATION_ABI = 1
MSA_PUBLIC_VLLM_ABI = 1
ICP_DEVICE_PLAN_ABI = 3
_msa_import_lock = Lock()


@cache
def require_msa_icp() -> Any:
    """Check MSA capabilities before loading native indexing or transport code."""
    requirement = (
        "msa_icp requires fmha_sm100.icp with "
        f"ICP_INTEGRATION_ABI={MSA_ICP_INTEGRATION_ABI} and "
        f"PUBLIC_VLLM_ABI={MSA_PUBLIC_VLLM_ABI}. "
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
    return package


@cache
def require_icp_writer() -> Callable[..., None]:
    """Reject stale vLLM binaries before the compound cache is bound."""
    import torch

    from vllm import _custom_ops as ops

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
    return ops.fused_minimax_m3_qknorm_rope_kv_insert
