# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The bound ICP writer orders arguments by schema and checks plans in order."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re
import torch

from vllm.models.minimax_m3.nvidia.msa_icp import IcpWriter

_BINDINGS = (
    Path(__file__).resolve().parents[3] / "csrc/libtorch_stable/torch_bindings.cpp"
)


def _native_schema() -> torch.FunctionSchema:
    source = _BINDINGS.read_text()
    match = re.search(
        r'"fused_minimax_m3_qknorm_rope_kv_insert\(.*?-> \(\)"\);', source, re.S
    )
    assert match is not None
    return torch._C.parse_schema("".join(re.findall(r'"([^"]*)"', match.group(0))))


class _NativeOp:
    """Records positional launches as name -> value using the real schema."""

    def __init__(self):
        self._schema = _native_schema()
        self.calls: list[dict[str, object]] = []
        self.on_call = None

    def __call__(self, *args):
        names = [arg.name for arg in self._schema.arguments]
        assert len(args) == len(names)
        self.calls.append(dict(zip(names, args)))
        if self.on_call is not None:
            self.on_call(self.calls[-1])


@pytest.fixture
def native(monkeypatch):
    op = _NativeOp()
    monkeypatch.setattr(
        torch.ops._C,
        "fused_minimax_m3_qknorm_rope_kv_insert",
        SimpleNamespace(default=op),
        raising=False,
    )
    return op


class _Plan:
    abi_version = 4

    def __init__(self, events):
        self.events = events
        self.plane = torch.empty(2, 4, dtype=torch.int32)

    def check_writer_launch(self, **kwargs):
        self.events.append(("check", kwargs))
        if kwargs["num_rows"] > 16:
            raise ValueError("plan capacity")

    def writer_kwargs(self):
        self.events.append(("expand", None))
        return {"icp_plan_header": self.plane, "icp_plan_num_heads": 4}

    def note_writer_launch(self, **kwargs):
        self.events.append(("note", kwargs))


def _layer_writer() -> IcpWriter:
    return IcpWriter(
        num_heads=8,
        num_kv_heads=1,
        rotary_dim=64,
        eps=1e-6,
        num_index_heads=1,
        block_size=128,
        kv_cache_dtype="nvfp4",
        index_rank=1,
        index_world_size=2,
    )


def _step() -> dict[str, object]:
    return {
        name: torch.empty(1)
        for name in (
            "qkv",
            "q_norm_weight",
            "k_norm_weight",
            "cos_sin_cache",
            "positions",
            "index_q_norm_weight",
            "index_k_norm_weight",
            "slot_mapping",
            "kv_cache",
            "index_cache",
            "q_out",
            "index_q_out",
            "q_fp8_out",
            "kv_k_scale",
            "kv_v_scale",
        )
    } | {"q_fp8_scale": 1.0}


def test_layer_launch_matches_the_keyword_launch_it_replaces(native):
    step = _step()
    _layer_writer()(**step)
    # The removed keyword path: its defaults table, the layer's scalars and
    # per-step values, then the plan ABI; everything else took schema defaults.
    legacy: dict[str, object] = dict(
        index_q_norm_weight=None,
        index_k_norm_weight=None,
        num_index_heads=0,
        slot_mapping=None,
        index_slot_mapping=None,
        kv_cache=None,
        index_cache=None,
        block_size=0,
        q_out=None,
        index_q_out=None,
        kv_cache_dtype="auto",
        skip_index_branch=False,
        q_fp8_out=None,
        q_fp8_scale=1.0,
        kv_k_scale=None,
        kv_v_scale=None,
    )
    legacy.update(
        dict(
            num_heads=8,
            num_kv_heads=1,
            rotary_dim=64,
            eps=1e-6,
            num_index_heads=1,
            block_size=128,
            kv_cache_dtype="nvfp4",
            index_rank=1,
            index_world_size=2,
            icp_plan_abi=4,
        )
    )
    legacy.update(step)
    expected = {
        arg.name: legacy.get(arg.name, arg.default_value)
        for arg in native._schema.arguments
    }
    assert native.calls == [expected]


def test_each_launch_passes_its_own_step_tensors(native):
    writer = _layer_writer()
    first, second = _step(), _step()
    writer(**first)
    writer(**second)
    assert native.calls[0]["qkv"] is first["qkv"]
    assert native.calls[1]["qkv"] is second["qkv"]
    assert native.calls[1]["kv_cache"] is second["kv_cache"]


def test_unknown_fixed_argument_is_rejected(native):
    with pytest.raises(TypeError, match="index_block_tokens"):
        IcpWriter(index_block_tokens=128)(**_step())
    assert native.calls == []


def test_icp_plan_is_checked_dispatched_once_then_marked_current(native):
    events: list[tuple[str, object]] = []
    plan = _Plan(events)
    positions = torch.empty(16, dtype=torch.int64)

    def dispatch(kwargs):
        assert kwargs["icp_plan_header"] is plan.plane
        assert kwargs["icp_positions"] is positions
        assert kwargs["icp_plan_num_heads"] == 4
        assert kwargs["icp_plan_abi"] == 4
        assert kwargs["index_slot_mapping"] is None
        events.append(("dispatch", None))

    native.on_call = dispatch
    _layer_writer()(
        write_icp_metadata=True,
        icp_positions=positions,
        icp_chunk_width=8,
        icp_device_plan=plan,
        **_step(),
    )
    shape = dict(chunk_width=8, num_rows=16)
    assert events == [
        ("check", shape),
        ("expand", None),
        ("dispatch", None),
        ("note", shape),
    ]


@pytest.mark.parametrize("failure", ["not_owner", "missing_positions", "abi", "bounds"])
def test_invalid_plan_never_dispatches_or_advances_generation(native, failure):
    events: list[tuple[str, object]] = []
    plan = _Plan(events)
    if failure == "abi":
        plan.abi_version = 2
    count = 17 if failure == "bounds" else 16
    with pytest.raises((RuntimeError, ValueError)):
        _layer_writer()(
            write_icp_metadata=failure != "not_owner",
            icp_positions=(
                None if failure == "missing_positions" else torch.empty(count)
            ),
            icp_device_plan=plan,
            **_step(),
        )
    assert native.calls == []
    assert all(event != "note" for event, _ in events)


def test_failed_native_launch_does_not_mark_the_plan_current(native):
    events: list[tuple[str, object]] = []
    plan = _Plan(events)

    def dispatch(kwargs):
        events.append(("dispatch", None))
        raise RuntimeError("native rejected geometry")

    native.on_call = dispatch
    with pytest.raises(RuntimeError, match="native rejected geometry"):
        _layer_writer()(
            write_icp_metadata=True,
            icp_positions=torch.empty(16),
            icp_device_plan=plan,
            **_step(),
        )
    assert [event for event, _ in events] == ["check", "expand", "dispatch"]
