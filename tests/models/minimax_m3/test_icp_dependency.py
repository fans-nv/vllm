# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject missing or incompatible MSA ICP support before native setup."""

import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.models.minimax_m3.nvidia import msa_icp as dependency


@pytest.fixture(autouse=True)
def clear_admission_cache():
    dependency.require_msa_icp.cache_clear()
    dependency.require_icp_writer.cache_clear()
    yield
    dependency.require_msa_icp.cache_clear()
    dependency.require_icp_writer.cache_clear()


def install_package(monkeypatch, abi=1, public_abi=1):
    package: Any = ModuleType("fmha_sm100.icp")
    package.ICP_INTEGRATION_ABI = abi
    package.PUBLIC_VLLM_ABI = public_abi
    monkeypatch.setitem(sys.modules, "fmha_sm100.icp", package)
    return package


def test_missing_package_reports_the_optional_install_requirement(monkeypatch):
    monkeypatch.setitem(sys.modules, "fmha_sm100.icp", None)
    with pytest.raises(RuntimeError, match=r"requirements/minimax_m3_icp\.txt"):
        dependency.require_msa_icp()


@pytest.mark.parametrize("abi", [None, 0, 2, "1", True])
def test_incompatible_integration_abi_is_rejected(monkeypatch, abi):
    install_package(monkeypatch, abi=abi)
    with pytest.raises(RuntimeError, match="ICP_INTEGRATION_ABI"):
        dependency.require_msa_icp()


def test_compatible_package_is_checked_only_once(monkeypatch):
    package = install_package(monkeypatch)
    imported = []
    real_import = dependency.importlib.import_module

    def record_import(name):
        imported.append(name)
        return real_import(name)

    monkeypatch.setattr(dependency.importlib, "import_module", record_import)
    assert dependency.require_msa_icp() is package
    assert dependency.require_msa_icp() is package
    assert imported == ["fmha_sm100.icp"]


def test_failed_admission_is_not_cached(monkeypatch):
    install_package(monkeypatch, abi=0)
    with pytest.raises(RuntimeError, match="ICP_INTEGRATION_ABI"):
        dependency.require_msa_icp()
    package = install_package(monkeypatch)
    assert dependency.require_msa_icp() is package


@pytest.mark.parametrize("public_abi", [None, 0, 2, "1", True])
def test_old_region_major_package_is_rejected(monkeypatch, public_abi):
    install_package(monkeypatch, public_abi=public_abi)
    with pytest.raises(RuntimeError, match="PUBLIC_VLLM_ABI"):
        dependency.require_msa_icp()


@pytest.mark.parametrize("previous", [None, "0", "1"])
@pytest.mark.parametrize("failed_import", [False, True])
def test_msa_import_suppresses_global_ffi_registration_and_restores_environment(
    monkeypatch, previous, failed_import
):
    if previous is None:
        monkeypatch.delenv("MSA_REGISTER_TVM_FFI", raising=False)
    else:
        monkeypatch.setenv("MSA_REGISTER_TVM_FFI", previous)
    package = install_package(monkeypatch)

    def import_msa(name):
        assert name == "fmha_sm100.icp"
        assert os.environ["MSA_REGISTER_TVM_FFI"] == "0"
        if failed_import:
            raise ImportError("missing MSA")
        return package

    monkeypatch.setattr(dependency.importlib, "import_module", import_msa)
    if failed_import:
        with pytest.raises(RuntimeError, match="Import failed"):
            dependency.require_msa_icp()
    else:
        assert dependency.require_msa_icp() is package
    assert os.environ.get("MSA_REGISTER_TVM_FFI") == previous


@pytest.mark.parametrize("plan_abi", [None, 2, 3])
@pytest.mark.parametrize("public_prefix", [False, True])
def test_compiled_writer_schema_requires_icp_and_plan_abi(
    monkeypatch, plan_abi, public_prefix
):
    from vllm import _custom_ops as ops

    arguments = [
        SimpleNamespace(name=name, default_value=None)
        for name in (
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
        )
    ]
    if not public_prefix:
        arguments = [a for a in arguments if a.name not in ("kv_k_scale", "kv_v_scale")]
    if plan_abi is not None:
        arguments.append(SimpleNamespace(name="icp_plan_abi", default_value=plan_abi))
    native = SimpleNamespace(
        default=SimpleNamespace(_schema=SimpleNamespace(arguments=arguments))
    )
    monkeypatch.setattr(
        torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert", native, raising=False
    )
    if plan_abi == 3 and public_prefix:
        assert (
            dependency.require_icp_writer()
            is ops.fused_minimax_m3_qknorm_rope_kv_insert
        )
    else:
        with pytest.raises(RuntimeError, match="Rebuild vLLM"):
            dependency.require_icp_writer()
