# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ICP prefill/mixed steps follow r13 (vLLM 4d16b10146): every row on FMHA,
geometric rungs, a 16384-row wave ceiling, no split-KV path."""

import builtins
import importlib.util
import itertools
import sys
import types
from typing import Any

import pytest

if importlib.util.find_spec("cutlass") is None:
    _stub = types.ModuleType("vllm.models.minimax_m3.nvidia.ops.index_decode_score")
    _stub.minimax_m3_index_decode_score_cutedsl = None  # type: ignore[attr-defined]
    sys.modules.setdefault(_stub.__name__, _stub)

from vllm.models.minimax_m3.nvidia import indexer_icp as icp  # noqa: E402

HEADS_GROUP = 4
MAX_BLOCKS = icp._cdiv(1048576, 128)
BUDGET = icp.ICP_PREFILL_SCORE_SCRATCH_BUDGET_BYTES


def _waves(token_capacity):
    pytest.importorskip("fmha_sm100.icp")
    columns = icp.icp_prefill_column_rungs(MAX_BLOCKS)
    return columns, icp.icp_prefill_wave_ladder(
        columns,
        heads_group=HEADS_GROUP,
        token_capacity=token_capacity,
        budget_bytes=BUDGET,
    )


@pytest.mark.parametrize("token_capacity", [8192, 16384, 32768])
def test_wave_ceiling_is_r13s_16384(token_capacity):
    _, waves = _waves(token_capacity)
    assert max(waves.values()) <= icp.ICP_MAX_PREFILL_WAVE_TOKENS == 16384


def test_ladder_is_geometric_only():
    columns, _ = _waves(16384)
    assert columns == icp.icp_prefill_column_rungs(MAX_BLOCKS)
    assert 3264 not in columns


def test_post_r13_prefill_knobs_are_gone():
    for name in (
        "icp_prefill_rungs",
        "icp_prefill_one_launch_columns",
        "icp_prefill_band_live_blocks",
        "icp_mixed_cute_band",
        "icp_chunk_kv_splits",
        "icp_scorer_split_options",
        "icp_device_plan_store",
        "_max_kv_splits",
        "_mixed_cute_mode",
        "_mixed_decode_query_len",
        "_package_has_prefill_threshold",
    ):
        assert not hasattr(icp, name), name


@pytest.mark.parametrize("use_cute_decode", [False, True])
def test_mixed_step_has_no_cute_band(use_cute_decode):
    assert (
        icp.icp_cute_rows(
            use_cute_decode=use_cute_decode, has_prefill=True, num_tokens=99
        )
        == 0
    )
    assert (
        icp.icp_cute_rows(use_cute_decode=True, has_prefill=False, num_tokens=99) == 99
    )


def test_mixed_step_is_fmha_windows_from_row_zero():
    pytest.importorskip("fmha_sm100.icp")
    lens = [4] * 40 + [2048]
    num_tokens = list(itertools.accumulate(lens))[-1]
    windows = icp.icp_row_windows(
        num_tokens=num_tokens,
        cap=num_tokens,
        cute_rows=0,
        decode_chunk=1024,
        prefill_width=1024,
        has_prefill=True,
    )
    assert all(not w.cute and not w.decode_selector for w in windows)
    assert [w.row_begin for w in windows] == list(range(0, num_tokens, 1024))
    with pytest.raises(AssertionError):
        icp.icp_row_windows(
            num_tokens=num_tokens,
            cap=num_tokens,
            cute_rows=160,
            decode_chunk=1024,
            prefill_width=1024,
            has_prefill=True,
        )


def test_prefill_exchange_is_the_native_pack_collective():
    from vllm.models.minimax_m3.nvidia.ops import icp_dispatch

    assert not hasattr(icp_dispatch, "r13_collective_exchange_and_merge")


@pytest.mark.parametrize("problems", [[], ["fmha: sources differ"]])
def test_prewarm_startup_assertion(monkeypatch, problems):
    monkeypatch.setattr(icp, "require_icp_writer", lambda: None)
    pkg: Any = types.ModuleType("fmha_sm100.icp")
    prewarm: Any = types.ModuleType("fmha_sm100.icp.prewarm")
    prewarm.check = lambda: list(problems)
    pkg.prewarm = prewarm
    monkeypatch.setitem(sys.modules, "fmha_sm100.icp", pkg)
    monkeypatch.setitem(sys.modules, "fmha_sm100.icp.prewarm", prewarm)
    monkeypatch.setattr(icp, "_prewarm_checked", False)
    monkeypatch.delenv(icp._PREWARM_CHECK_ENV, raising=False)
    if problems:
        with pytest.raises(RuntimeError, match="not fully prewarmed"):
            icp._assert_icp_prewarm(0)
        monkeypatch.setenv(icp._PREWARM_CHECK_ENV, "warn")
    icp._assert_icp_prewarm(0)
    assert icp._prewarm_checked


@pytest.mark.parametrize("policy", [None, "fail", "warn"])
@pytest.mark.parametrize(
    "missing", ["fmha_sm100.icp.prewarm", "dependency_of_icp_prewarm"]
)
def test_missing_prewarm_verifier_requires_explicit_warning_policy(
    monkeypatch, policy, missing
):
    monkeypatch.setattr(icp, "require_icp_writer", lambda: None)
    real_import = builtins.__import__

    def import_without_verifier(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fmha_sm100.icp" and "prewarm" in fromlist:
            raise ImportError(f"No module named {missing}", name=missing)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_verifier)
    monkeypatch.setattr(icp, "_prewarm_checked", False)
    monkeypatch.delenv(icp._PREWARM_CHECK_ENV, raising=False)
    if policy is not None:
        monkeypatch.setenv(icp._PREWARM_CHECK_ENV, policy)
    if policy == "warn":
        icp._assert_icp_prewarm(0)
        assert icp._prewarm_checked
    else:
        with pytest.raises(RuntimeError, match="prewarm verification is unavailable"):
            icp._assert_icp_prewarm(0)
        assert not icp._prewarm_checked


def test_warning_only_prewarm_policy_never_admits_a_stale_writer(monkeypatch):
    def stale_writer():
        raise RuntimeError("native writer lacks device-plan ABI 3")

    monkeypatch.setattr(icp, "require_icp_writer", stale_writer)
    monkeypatch.setattr(icp, "_prewarm_checked", False)
    monkeypatch.setenv(icp._PREWARM_CHECK_ENV, "warn")
    with pytest.raises(RuntimeError, match="native writer lacks"):
        icp._assert_icp_prewarm(0)
    assert not icp._prewarm_checked


@pytest.mark.parametrize(
    ("name", "value", "refused"),
    [
        ("VLLM_MINIMAX_ICP_CUTE_SPLIT_K", "128", False),
        ("VLLM_MINIMAX_ICP_CUTE_SPLIT_K", "0", True),
        ("VLLM_MINIMAX_ICP_DECODE_CHUNK", "arena", True),
        ("VLLM_MINIMAX_ICP_MAX_KV_SPLITS", "64", True),
        ("VLLM_MINIMAX_ICP_MIN_SPLIT_TILES", "4", True),
        ("VLLM_MINIMAX_ICP_MIXED_CUTE", "auto", True),
        ("VLLM_MINIMAX_ICP_FUSED_END_WAIT", "0", True),
    ],
)
def test_removed_icp_env_is_refused(monkeypatch, name, value, refused):
    for env in icp._REMOVED_ICP_ENV:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(name, value)
    if refused:
        with pytest.raises(RuntimeError, match="removed in v13"):
            icp._refuse_removed_icp_env()
    else:
        icp._refuse_removed_icp_env()


def test_decode_profile_must_match_the_bound_exchange(monkeypatch):
    monkeypatch.setattr(icp, "_decode_profile_logged", False)
    exchange = types.SimpleNamespace(
        decode_profile={"exchange": "k5t", "merge": "network", "d3": 0, "pdl": 1}
    )
    with pytest.raises(RuntimeError, match="d3=0"):
        icp._log_decode_profile(0, exchange, 128, True)
    icp._log_decode_profile(0, exchange, 128, False)
    assert icp._decode_profile_logged
