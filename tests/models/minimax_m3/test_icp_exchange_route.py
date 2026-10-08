# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ICP exchange routing: NCCL for prefill/mixed, K5T for decode, no ATen glue."""

import sys
import types
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.models.minimax_m3.nvidia.ops.icp_dispatch import (
    BACKEND_K5T,
    BACKEND_NCCL,
    CLASSIC_MERGE_ENV,
    FUSED_DECODE_ENV,
    admitted_exchange_extents,
    check_slot_discipline,
    create_candidate_exchange,
    decode_merge_mode,
    fused_decode_requested,
    select_backend,
)


class _FakeTiledExchange:
    instances: list["_FakeTiledExchange"] = []

    def __init__(
        self,
        group,
        *,
        tokens,
        heads_group,
        heads_local,
        slots,
        device,
        use_pdl,
        classic_merge,
    ):
        self.kwargs = dict(
            tokens=tokens,
            heads_group=heads_group,
            heads_local=heads_local,
            slots=slots,
            use_pdl=use_pdl,
            classic_merge=classic_merge,
        )
        self.calls = []
        self.closed = False
        _FakeTiledExchange.instances.append(self)

    def provenance(self):
        return {
            "use_ack": 0,
            "symm_backend": "CUDA",
            "symm_mib": "0.00",
            "tile_tok": 4,
            "resident_cap": 848,
        }

    def exchange_and_merge(self, cand, *, out, layer_idx, forced, n_ordinary):
        self.calls.append((cand, out, layer_idx, forced, n_ordinary))
        return out

    def close(self):
        self.closed = True


class _FakeCarrier:
    def __init__(self):
        self.workspaces = {}
        self.calls = []

    def allocate_carrier_workspace(self, *, world, qchunk, h_local, device, transport):
        ws = ("workspace", world, qchunk, h_local, transport)
        self.workspaces[qchunk] = ws
        return ws

    def collective_exchange_and_merge(self, cand, **kwargs):
        self.calls.append((cand, kwargs))
        return kwargs["status"]


class _ForbidAten(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


@pytest.fixture
def stub_kernels(monkeypatch):
    pytest.importorskip("fmha_sm100.icp")
    # No `exchange` (K5) or `merge` attributes: binding either would raise.
    # These tests exercise the K5T decode route, which D3 (the default) replaces.
    monkeypatch.setenv(FUSED_DECODE_ENV, "0")
    monkeypatch.delenv(CLASSIC_MERGE_ENV, raising=False)
    monkeypatch.delenv("VLLM_MINIMAX_ICP_DECODE_PDL", raising=False)
    fake_carrier = _FakeCarrier()
    tiled: Any = types.ModuleType("fmha_sm100.icp.tiled_exchange")
    tiled.IcpTiledExchange = _FakeTiledExchange
    carrier: Any = types.ModuleType("fmha_sm100.icp.carrier")
    carrier.allocate_carrier_workspace = fake_carrier.allocate_carrier_workspace
    carrier.collective_exchange_and_merge = fake_carrier.collective_exchange_and_merge
    # Exercise the real package owner and mock only native transport leaves.
    monkeypatch.setitem(sys.modules, "fmha_sm100.icp.tiled_exchange", tiled)
    monkeypatch.setitem(sys.modules, "fmha_sm100.icp.carrier", carrier)
    _FakeTiledExchange.instances.clear()
    return _FakeTiledExchange, fake_carrier


def _exchange(capacities):
    return create_candidate_exchange(
        group=None,
        world_size=2,
        rank=0,
        num_heads_local=2,
        num_index_heads=4,
        num_sparse_layers=57,
        token_capacities=capacities,
        device=torch.device("cpu"),
    )


def test_d3_is_the_default_decode_route(monkeypatch):
    monkeypatch.delenv(FUSED_DECODE_ENV, raising=False)
    monkeypatch.delenv(CLASSIC_MERGE_ENV, raising=False)
    assert fused_decode_requested()
    assert decode_merge_mode() == "network"
    monkeypatch.setenv(FUSED_DECODE_ENV, "0")
    assert not fused_decode_requested()
    monkeypatch.setenv(CLASSIC_MERGE_ENV, "1")
    assert decode_merge_mode() == "classic"


def test_phase_alone_selects_the_route():
    assert select_backend(has_prefill=True) == BACKEND_NCCL
    assert select_backend(has_prefill=False) == BACKEND_K5T


def test_slot_discipline_for_the_model_sweep():
    pytest.importorskip("fmha_sm100.icp")
    check_slot_discipline(57, 3)
    with pytest.raises(ValueError, match="wrap"):
        check_slot_discipline(58, 3)
    with pytest.raises(ValueError, match="ACK"):
        check_slot_discipline(57, 2)


@pytest.mark.parametrize("has_prefill", [True, False])
@pytest.mark.parametrize("extent", [4, 1024, 16_384])
def test_each_phase_takes_exactly_its_route_with_no_aten(
    stub_kernels, has_prefill, extent
):
    fake_cls, fake_carrier = stub_kernels
    extents = admitted_exchange_extents(max_token_capacity=16_384)
    exchange = _exchange(extents)
    (fake,) = fake_cls.instances
    assert fake.kwargs == dict(
        tokens=16_384,
        heads_group=4,
        heads_local=2,
        slots=3,
        use_pdl=True,
        classic_merge=False,
    )
    assert set(fake_carrier.workspaces) == set(extents)
    cand = torch.zeros((extent, 4, 16, 2), dtype=torch.int32)
    forced = torch.zeros(extent, dtype=torch.int32)
    n_ord = torch.zeros(extent, dtype=torch.int32)
    out = torch.zeros((extent, 2, 16), dtype=torch.int32)
    with _ForbidAten() as mode:
        result = exchange(
            cand,
            layer_idx=7,
            has_prefill=has_prefill,
            forced=forced,
            n_ordinary=n_ord,
            out=out,
        )
    assert mode.ops == []
    assert result is out
    if has_prefill:
        assert fake.calls == []
        ((c, kw),) = fake_carrier.calls
        assert c is cand and kw["out"] is out and kw["forced"] is forced
        assert kw["n_ordinary"] is n_ord and kw["transport"] == "all_to_all"
        assert kw["workspace"] is fake_carrier.workspaces[extent]
        assert kw["status"] is exchange._status
    else:
        assert fake_carrier.calls == []
        ((c, o, layer, f, q),) = fake.calls
        assert c is cand and o is out and f is forced and q is n_ord
        assert layer == 7
    exchange.close()
    assert fake.closed


def test_unadmitted_extent_is_refused(stub_kernels):
    exchange = _exchange([4, 8, 16])
    cand = torch.zeros((12, 4, 16, 2), dtype=torch.int32)
    plane = torch.zeros(12, dtype=torch.int32)
    out = torch.zeros((12, 2, 16), dtype=torch.int32)
    with pytest.raises(ValueError, match="admitted"):
        exchange(
            cand, layer_idx=0, has_prefill=True, forced=plane, n_ordinary=plane, out=out
        )


@pytest.mark.parametrize("classic_merge", [False, True])
@pytest.mark.parametrize("use_pdl", [False, True])
def test_decode_profile_reads_the_constructed_route(
    stub_kernels, monkeypatch, classic_merge, use_pdl
):
    monkeypatch.setenv(CLASSIC_MERGE_ENV, str(int(classic_merge)))
    monkeypatch.setenv("VLLM_MINIMAX_ICP_DECODE_PDL", str(int(use_pdl)))
    exchange = _exchange([16, 32])
    (fake,) = stub_kernels[0].instances
    assert fake.kwargs["classic_merge"] is classic_merge
    assert fake.kwargs["use_pdl"] is use_pdl
    assert exchange.decode_profile == {
        "exchange": BACKEND_K5T,
        "merge": "classic" if classic_merge else "network",
        "d3": 0,
        "pdl": int(use_pdl),
    }


def test_d3_publishes_windows_then_merges_and_prefill_uses_nccl(
    stub_kernels, monkeypatch
):
    """The package owner retains D3's two-stage boundary and phase routing."""
    events: list[tuple[object, ...]] = []

    class Fused:
        def __init__(self, group, **kwargs):
            assert kwargs["use_pdl"] and kwargs["end_wait"]
            assert not kwargs["classic_merge"]

        def provenance(self):
            return {
                "use_ack": 0,
                "symm_backend": "CUDA",
                "symm_mib": "0.00",
                "tile_tok": 1,
                "resident_cap": 848,
            }

        def select_and_publish(self, scores, geometry, *, layer_idx, token_offset):
            events.append(("publish", layer_idx, token_offset))

        def merge(self, out, *, layer_idx, forced, n_ordinary):
            events.append(("merge", layer_idx))

        def close(self):
            events.append(("close",))

    fused: Any = types.ModuleType("fmha_sm100.icp.fused_exchange")
    fused.IcpFusedExchange = Fused
    monkeypatch.setitem(sys.modules, "fmha_sm100.icp.fused_exchange", fused)
    monkeypatch.setenv(FUSED_DECODE_ENV, "1")
    exchange = _exchange([4, 8])
    assert not stub_kernels[0].instances
    assert exchange.decode_profile["d3"] == 1
    cand = torch.empty((8, 4, 16, 2), dtype=torch.int32)
    plane = torch.empty(8, dtype=torch.int32)
    out = torch.empty((8, 2, 16), dtype=torch.int32)
    scores = torch.empty((4, 4, 4))
    with _ForbidAten() as mode:
        for offset in (0, 4):
            exchange.publish(scores, object(), layer_idx=2, token_offset=offset)
        for has_prefill in (False, True):
            result = exchange(
                cand,
                layer_idx=2,
                has_prefill=has_prefill,
                forced=plane,
                n_ordinary=plane,
                out=out,
            )
            assert result is out
    assert mode.ops == []
    assert events == [("publish", 2, 0), ("publish", 2, 4), ("merge", 2)]
    assert len(stub_kernels[1].calls) == 1
    exchange.close()
    exchange.close()
    assert events[-1] == ("close",) and len(events) == 4
    with pytest.raises(RuntimeError, match="closed"):
        exchange.publish(scores, object(), layer_idx=3, token_offset=0)
