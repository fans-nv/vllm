# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ICP prefill/mixed steps follow r13 (vLLM 4d16b10146): every row on FMHA,
geometric rungs, a 16384-row wave ceiling, no split-KV path."""

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


def _columns():
    li = pytest.importorskip("fmha_sm100.icp.local_indexer")
    return li.icp_prefill_column_rungs(
        MAX_BLOCKS,
        floor=icp.ICP_PREFILL_COLUMN_FLOOR,
        steps_per_octave=icp.ICP_PREFILL_COLUMN_STEPS_PER_OCTAVE,
    )


def _waves(token_capacity):
    li = pytest.importorskip("fmha_sm100.icp.local_indexer")
    columns = _columns()
    return columns, li.icp_prefill_wave_ladder(
        columns,
        heads_group=HEADS_GROUP,
        token_capacity=token_capacity,
        budget_bytes=BUDGET,
        max_wave=icp.ICP_MAX_PREFILL_WAVE_TOKENS,
        query_tile_tokens=icp.ICP_MSA_QUERY_TILE_TOKENS,
    )


@pytest.mark.parametrize("token_capacity", [8192, 16384, 32768])
def test_wave_ceiling_is_r13s_16384(token_capacity):
    _, waves = _waves(token_capacity)
    assert max(waves.values()) <= icp.ICP_MAX_PREFILL_WAVE_TOKENS == 16384


def test_ladder_is_geometric_only():
    columns, _ = _waves(16384)
    assert columns == _columns()
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
    li = pytest.importorskip("fmha_sm100.icp.local_indexer")
    lens = [4] * 40 + [2048]
    num_tokens = list(itertools.accumulate(lens))[-1]
    windows = li.icp_row_windows(
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
        li.icp_row_windows(
            num_tokens=num_tokens,
            cap=num_tokens,
            cute_rows=160,
            decode_chunk=1024,
            prefill_width=1024,
            has_prefill=True,
        )


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
    if problems:
        with pytest.raises(RuntimeError, match="not fully prewarmed"):
            icp._assert_icp_prewarm(0)
        assert not icp._prewarm_checked
    else:
        icp._assert_icp_prewarm(0)
        assert icp._prewarm_checked


def test_missing_prewarm_verifier_fails_startup(monkeypatch):
    monkeypatch.setattr(icp, "require_icp_writer", lambda: None)
    monkeypatch.setitem(sys.modules, "fmha_sm100.icp", types.ModuleType("icp"))
    monkeypatch.setitem(sys.modules, "fmha_sm100.icp.prewarm", None)
    monkeypatch.setattr(icp, "_prewarm_checked", False)
    with pytest.raises(RuntimeError, match="prewarm verification is unavailable"):
        icp._assert_icp_prewarm(0)
    assert not icp._prewarm_checked


def test_prewarm_check_never_admits_a_stale_writer(monkeypatch):
    def stale_writer():
        raise RuntimeError("native writer lacks device-plan ABI 3")

    monkeypatch.setattr(icp, "require_icp_writer", stale_writer)
    monkeypatch.setattr(icp, "_prewarm_checked", False)
    with pytest.raises(RuntimeError, match="native writer lacks"):
        icp._assert_icp_prewarm(0)
    assert not icp._prewarm_checked


def test_decode_profile_is_logged_once(monkeypatch):
    monkeypatch.setattr(icp, "_decode_profile_logged", False)
    exchange = types.SimpleNamespace(
        decode_profile={"exchange": "k5t", "merge": "network", "d3": 1, "pdl": 1}
    )
    icp._log_decode_profile(0, exchange, 128)
    assert icp._decode_profile_logged
    with pytest.raises(RuntimeError, match="no decode_profile"):
        monkeypatch.setattr(icp, "_decode_profile_logged", False)
        icp._log_decode_profile(0, object(), 128)


def test_prefill_full_row_controls_are_r13s():
    assert (
        icp._ICP_PREFILL_FULL_ROW_THREADS,
        icp._ICP_PREFILL_FULL_ROW_CACHED_ITEMS,
        icp._ICP_PREFILL_FOUR_WARP_FINISH_MAX_COLUMNS,
    ) == (128, 0, 128)
