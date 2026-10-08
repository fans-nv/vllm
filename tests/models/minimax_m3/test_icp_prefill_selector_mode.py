# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ICP prefill selector mode and controls: r13's full_row arm."""

import importlib.util
import sys
import types

import pytest

if importlib.util.find_spec("cutlass") is None:
    _stub = types.ModuleType("vllm.models.minimax_m3.nvidia.ops.index_decode_score")
    _stub.minimax_m3_index_decode_score_cutedsl = None  # type: ignore[attr-defined]
    sys.modules.setdefault(_stub.__name__, _stub)

from vllm.models.minimax_m3.nvidia import indexer_icp as icp  # noqa: E402


@pytest.mark.parametrize(
    "value, mode",
    [(None, "full_row"), ("FULL_ROW", "full_row"), ("bounded", "bounded")],
)
def test_prefill_selector_mode(monkeypatch, value, mode):
    if value is None:
        monkeypatch.delenv(icp._PREFILL_SELECTOR_ENV, raising=False)
    else:
        monkeypatch.setenv(icp._PREFILL_SELECTOR_ENV, value)
    assert icp._prefill_selector_mode() == mode


@pytest.mark.parametrize("value", ["threshold", "radix"])
def test_unknown_prefill_selector_mode_is_refused(monkeypatch, value):
    monkeypatch.setenv(icp._PREFILL_SELECTOR_ENV, value)
    with pytest.raises(ValueError, match="is not one of"):
        icp._prefill_selector_mode()


@pytest.mark.parametrize("columns", [128, 129, 4096])
def test_controls_are_r13s(columns):
    pytest.importorskip("fmha_sm100.icp")
    controls = icp._prefill_full_row_controls(columns)
    assert controls == (128, 0, columns <= 128)
    assert icp._prefill_selector_kwargs(controls) == dict(
        full_row_threads=128,
        full_row_cached_items=0,
        full_row_four_warp_finish=columns <= 128,
    )


def test_bounded_rollback_uses_package_defaults():
    pytest.importorskip("fmha_sm100.icp")
    assert icp._prefill_selector_kwargs(icp._ICP_PREFILL_BOUNDED_CONTROLS) == dict(
        full_row_threads=512, full_row_cached_items=3, full_row_four_warp_finish=False
    )
