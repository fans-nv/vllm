# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ICP writer launcher checks, expands and marks the device plan in order."""

import pytest
import torch

from vllm.models.minimax_m3.nvidia.msa_icp import icp_kv_insert


@pytest.fixture
def launch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.ops._C,
        "fused_minimax_m3_qknorm_rope_kv_insert",
        lambda **kwargs: calls.append(kwargs),
        raising=False,
    )
    return calls


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


def test_icp_plan_is_checked_dispatched_once_then_marked_current(monkeypatch):
    events: list[tuple[str, object]] = []
    plan = _Plan(events)
    positions = torch.empty(16, dtype=torch.int64)

    def dispatch(**kwargs):
        assert kwargs["icp_plan_header"] is plan.plane
        assert kwargs["icp_positions"] is positions
        assert kwargs["icp_plan_max_splits"] == 1
        assert kwargs["icp_plan_abi"] == 3
        assert kwargs["index_slot_mapping"] is None
        assert kwargs["enable_pdl"] is False
        events.append(("dispatch", None))

    monkeypatch.setattr(
        torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert", dispatch, raising=False
    )
    icp_kv_insert(
        index_world_size=2,
        enable_pdl=False,
        write_icp_metadata=True,
        icp_positions=positions,
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
        icp_kv_insert(
            index_world_size=2,
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

    def dispatch(**kwargs):
        events.append(("dispatch", None))
        raise RuntimeError("native rejected geometry")

    monkeypatch.setattr(
        torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert", dispatch, raising=False
    )
    with pytest.raises(RuntimeError, match="native rejected geometry"):
        icp_kv_insert(
            index_world_size=2,
            write_icp_metadata=True,
            icp_positions=torch.empty(16),
            icp_device_plan=plan,
        )
    assert [event for event, _ in events] == ["check", "expand", "dispatch"]


def test_non_owner_layers_pass_the_plan_abi_without_a_plan(launch):
    icp_kv_insert(index_world_size=2, enable_pdl=False)
    assert launch == [
        dict(
            index_world_size=2,
            enable_pdl=False,
            index_slot_mapping=None,
            icp_plan_abi=3,
        )
    ]
