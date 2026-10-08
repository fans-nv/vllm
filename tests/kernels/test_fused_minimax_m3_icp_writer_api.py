# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the one-launch MiniMax-M3 ICP writer wrapper.

The wrapper only expands a preallocated plan and dispatches an in-place op.
These tests guard ordering, failed-launch bookkeeping, old binaries and the
mutable schema. Native arithmetic/fragment checks live with the NVFP4 tests.
"""

import ast
from pathlib import Path

import pytest
import regex as re
import torch

from vllm import _custom_ops as ops


@pytest.fixture
def launch(monkeypatch):
    calls = []

    def dispatch(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert", dispatch, raising=False
    )
    return calls


def _inputs():
    tensor = torch.empty(1)
    return dict(
        qkv=tensor,
        q_norm_weight=tensor,
        k_norm_weight=tensor,
        cos_sin_cache=tensor,
        positions=tensor,
        num_heads=32,
        num_kv_heads=2,
        rotary_dim=64,
        eps=1e-6,
    )


class _Plan:
    abi_version = 3

    def __init__(self, events):
        self.events = events
        self.plane = torch.empty(2, 4, dtype=torch.int32)

    def check_writer_launch(self, **kwargs):
        self.events.append(("check", kwargs))
        if kwargs["num_rows"] > 16:
            raise ValueError("plan capacity")

    def writer_kwargs(self):
        self.events.append(("expand", None))
        return {"icp_plan_header": self.plane, "icp_plan_max_splits": 1}

    def note_writer_launch(self, **kwargs):
        self.events.append(("note", kwargs))


def test_default_writer_keeps_the_old_binary_call_prefix(launch):
    """Ordinary MSA must remain callable with the pre-ICP compiled schema."""
    ops.fused_minimax_m3_qknorm_rope_kv_insert(**_inputs())
    assert len(launch) == 1
    args, kwargs = launch[0]
    assert len(args) == 25
    assert args[5:9] == (32, 2, 64, 1e-6)
    assert args[23:25] == (None, None)  # public kv_k_scale/kv_v_scale
    assert kwargs == {}


def test_icp_plan_is_checked_dispatched_once_then_marked_current(monkeypatch):
    events: list[tuple[str, object]] = []
    plan = _Plan(events)
    positions = torch.empty(16, dtype=torch.int64)
    candidates = torch.empty(16, 4, 16, 2)

    def dispatch(*args, **kwargs):
        assert kwargs["icp_plan_header"] is plan.plane
        assert kwargs["icp_positions"] is positions
        assert kwargs["icp_candidates"] is candidates
        assert kwargs["icp_plan_max_splits"] == 1
        assert kwargs["icp_plan_abi"] == 3
        assert kwargs["enable_pdl"] is False
        events.append(("dispatch", None))

    monkeypatch.setattr(
        torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert", dispatch, raising=False
    )
    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        **_inputs(),
        index_block_tokens=128,
        index_rows_per_rank=64,
        index_rank=1,
        index_world_size=2,
        enable_pdl=False,
        write_icp_metadata=True,
        icp_positions=positions,
        icp_candidates=candidates,
        icp_chunk_width=8,
        icp_plan_row_begin=2,
        icp_device_plan=plan,
    )
    shape = dict(chunk_width=8, num_rows=16, row_begin=2)
    assert events == [
        ("check", shape),
        ("expand", None),
        ("dispatch", None),
        ("note", shape),
    ]


@pytest.mark.parametrize("failure", ["not_owner", "missing_positions", "abi", "bounds"])
def test_invalid_plan_never_dispatches_or_advances_generation(launch, failure):
    events: list[tuple[str, object]] = []
    plan = _Plan(events)
    if failure == "abi":
        plan.abi_version = 2
    count = 17 if failure == "bounds" else 16
    with pytest.raises((RuntimeError, ValueError)):
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            **_inputs(),
            index_world_size=2,
            enable_pdl=False,
            write_icp_metadata=failure != "not_owner",
            icp_positions=(
                None if failure == "missing_positions" else torch.empty(count)
            ),
            icp_device_plan=plan,
        )
    assert launch == []
    assert all(event != "note" for event, _ in events)


def test_failed_native_launch_does_not_mark_the_plan_current(monkeypatch):
    events: list[tuple[str, object]] = []
    plan = _Plan(events)

    def dispatch(*args, **kwargs):
        events.append(("dispatch", None))
        raise RuntimeError("native rejected geometry")

    monkeypatch.setattr(
        torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert", dispatch, raising=False
    )
    with pytest.raises(RuntimeError, match="native rejected geometry"):
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            **_inputs(),
            index_world_size=2,
            enable_pdl=False,
            write_icp_metadata=True,
            icp_positions=torch.empty(16),
            icp_device_plan=plan,
        )
    assert [event for event, _ in events] == ["check", "expand", "dispatch"]


def test_source_schema_marks_icp_destinations_mutable_and_advertises_abi():
    """A missing mutation annotation can discard the metadata producer."""
    bindings = (
        Path(__file__).resolve().parents[2] / "csrc/libtorch_stable/torch_bindings.cpp"
    ).read_text()
    definition = bindings.split('"fused_minimax_m3_qknorm_rope_kv_insert("', 1)[1]
    definition = definition.split(");", 1)[0]
    schema = "fused_minimax_m3_qknorm_rope_kv_insert(" + "".join(
        ast.literal_eval(part) for part in re.findall(r'"[^"\n]*"', definition)
    )
    arguments = {arg.name: arg for arg in torch._C.parse_schema(schema).arguments}
    assert arguments["icp_plan_abi"].default_value == 3
    assert arguments["index_world_size"].default_value == 0
    assert arguments["enable_pdl"].default_value is True
    for name in (
        "icp_positions",
        "icp_active",
        "icp_local_nvalid",
        "icp_local_forced",
        "icp_global_nvalid",
        "icp_forced",
        "icp_n_ordinary",
        "icp_candidates",
        "icp_qo_offsets",
        "icp_plan_segments",
        "icp_plan_work",
        "icp_plan_header",
        "icp_plan_ranges",
    ):
        assert arguments[name].alias_info.is_write, name
