# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""First-writer ownership and replay binding, without launching CUDA work."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.models.minimax_m3.nvidia.ops import icp_metadata


class NoTensorWork(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"Live metadata binding dispatched {func}")


def make_metadata(extent=4):
    # A view of a larger retained allocation, as in a small graph bucket.
    capacity = 32
    planes = {
        name: torch.empty(capacity, dtype=dtype)[:extent]
        for name, dtype in (
            ("positions", torch.int64),
            ("active", torch.bool),
            ("local_nvalid", torch.int32),
            ("local_forced", torch.int32),
            ("global_nvalid", torch.int32),
            ("forced", torch.int32),
            ("n_ordinary", torch.int32),
        )
    }
    return icp_metadata.ICPLiveMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([129], dtype=torch.int32),
        num_reqs=1,
        candidates=torch.empty(capacity, 4, 16, 2)[:extent],
        qo_offsets=torch.empty(8, 2, dtype=torch.int32),
        chunk_width=4,
        **planes,
    )


def context(live):
    return SimpleNamespace(
        attn_metadata={
            "sparse.0": SimpleNamespace(indexer=SimpleNamespace(live_metadata=live))
        }
    )


def test_owner_passes_stable_extent_views_to_the_same_native_writer(monkeypatch):
    live = make_metadata()
    monkeypatch.setattr(icp_metadata, "get_forward_context", lambda: context(live))
    monkeypatch.setattr(icp_metadata, "eager_break_during_capture", lambda fn: fn)
    launches = []
    qkv = object()
    with NoTensorWork():
        icp_metadata.launch_icp_kv_write(
            "sparse.0", True, lambda **kw: launches.append(kw), {"qkv": qkv}
        )
    assert len(launches) == 1
    launch = launches[0]
    assert launch["qkv"] is qkv
    assert launch["write_icp_metadata"] is True
    assert launch["icp_query_start_loc"] is live.query_start_loc
    assert launch["icp_positions"] is live.positions
    assert launch["icp_candidates"] is live.candidates
    assert launch["icp_positions"].shape == (4,)
    assert launch["icp_candidates"].shape == (4, 4, 16, 2)
    assert live.positions.untyped_storage().nbytes() == 32 * 8


def test_nonowner_never_binds_or_refreshes_shared_planes(monkeypatch):
    def forbidden():
        raise AssertionError("nonowner inspected forward metadata")

    monkeypatch.setattr(icp_metadata, "get_forward_context", forbidden)
    launches = []
    with NoTensorWork():
        icp_metadata.launch_icp_kv_write(
            "sparse.1", False, lambda **kw: launches.append(kw), {"qkv": "input"}
        )
    assert launches == [{"qkv": "input"}]


def test_piecewise_owner_rebinds_current_metadata_on_every_replay(monkeypatch):
    current = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(icp_metadata, "get_forward_context", lambda: current)
    replay = []

    def capture_break(fn):
        def captured(**kwargs):
            replay.append(lambda: fn(**kwargs))
            return fn(**kwargs)

        return captured

    monkeypatch.setattr(icp_metadata, "eager_break_during_capture", capture_break)
    launches = []
    writer = lambda **kw: launches.append(kw)
    icp_metadata.launch_icp_kv_write("sparse.0", True, writer, {"qkv": "captured"})
    assert launches == [{"qkv": "captured"}]
    first, second = make_metadata(4), make_metadata(16)
    for live in (first, second):
        current = context(live)
        with NoTensorWork():
            replay[0]()
        assert launches[-1]["icp_positions"] is live.positions
        assert launches[-1]["icp_candidates"] is live.candidates
    assert len(launches) == 3


def test_serving_without_native_destinations_fails_before_writer(monkeypatch):
    monkeypatch.setattr(icp_metadata, "get_forward_context", lambda: context(None))
    monkeypatch.setattr(icp_metadata, "eager_break_during_capture", lambda fn: fn)
    launches = []
    with pytest.raises(RuntimeError, match="requires native live metadata"):
        icp_metadata.launch_icp_kv_write(
            "sparse.0", True, lambda **kw: launches.append(kw), {}
        )
    assert launches == []


@pytest.mark.parametrize("wrong_metadata", [None, SimpleNamespace(indexer=None)])
def test_attention_rejects_wrong_metadata_before_index_selection(
    monkeypatch, wrong_metadata
):
    from vllm.models.minimax_m3.nvidia import sparse_attention_icp

    calls: list[dict] = []
    layer: Any = SimpleNamespace(
        layer_name="sparse.0",
        _bound_compound_cache=lambda: SimpleNamespace(index=None, main=None),
        indexer=lambda *args, **kwargs: calls.append(kwargs),
        index_world_size=2,
        tp_rank=0,
        sparse_layer_index=0,
        candidate_exchange=None,
    )
    monkeypatch.setattr(
        sparse_attention_icp,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"sparse.0": wrong_metadata}),
    )
    tensor = torch.empty(0)
    with NoTensorWork(), pytest.raises(RuntimeError, match="requires sparse metadata"):
        sparse_attention_icp.MiniMaxM3SparseICPAttention._run_attention(
            layer, tensor, None, tensor, tensor
        )
    assert calls == []


@pytest.mark.parametrize("qsl", [[0, 129], [0, 4, 5], [0, 4, 8], [0, 1]])
def test_actual_icp_builder_binds_q8kv4_decode_and_r13_prefill_csr(qsl):
    from vllm.models.minimax_m3.nvidia.sparse_attention_icp import (
        MiniMaxM3SparseICPMetadataBuilder,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata

    # Decode: views only. Prefill: the r13 CSR metadata (rebased cu_seqlens_q,
    # cu_seqlens_k cumsum into the retained zero-headed buffer) and nothing else.
    allowed = {
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.cumsum.out,
        torch.ops.aten.view.default,
    }

    class ViewsOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if any(isinstance(arg, torch.Tensor) and arg.is_meta for arg in args):
                assert func in allowed, func
            return func(*args, **(kwargs or {}))

    n = len(qsl) - 1
    cm = CommonAttentionMetadata(
        query_start_loc=torch.empty(n + 1, dtype=torch.int32, device="meta"),
        query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        seq_lens=torch.empty(n, dtype=torch.int32, device="meta"),
        num_reqs=n,
        num_actual_tokens=qsl[-1],
        max_query_len=max(b - a for a, b in zip(qsl, qsl[1:])),
        max_seq_len=8192,
        seq_lens_cpu_upper_bound=torch.full((n,), 300, dtype=torch.int32),
        block_table_tensor=torch.empty(n, 64, dtype=torch.int32, device="meta"),
        slot_mapping=torch.empty(qsl[-1], dtype=torch.int64, device="meta"),
    )
    indexer = object()
    builder = object.__new__(MiniMaxM3SparseICPMetadataBuilder)
    builder.reorder_batch_threshold = 4
    builder.q8kv4_plans = SimpleNamespace(get=lambda b, q: (b, q))
    builder._page_indptr = {64: torch.empty(9, dtype=torch.int32, device="meta")}
    builder.prefill_cu_seqlens_k = torch.zeros(9, dtype=torch.int32, device="meta")
    builder.indexer_builder = SimpleNamespace(build=lambda *a: indexer)
    with ViewsOnly():
        md = builder.build(0, cm)
    assert cm._num_computed_tokens_cache is None
    assert md.indexer is indexer
    # Prefill rows never get a Q8KV4 plan: they run r13's NVFP4 sparse forward.
    if md.num_prefills:
        p = md.prefill
        assert p.seq_lens._base is cm.seq_lens
        assert p.block_table._base is cm.block_table_tensor
        assert p.cu_seqlens_k.shape == (md.num_prefills + 1,)
        assert p.cu_seqlens_k._base is builder.prefill_cu_seqlens_k
        assert p.cu_seqlens_q.shape == (md.num_prefills + 1,)
        assert p.total_kv_blocks == 3 * md.num_prefills  # ceil(300 / 128)
        assert p.context_lens is None
    else:
        assert md.prefill is None
    if md.num_decodes:
        assert md.decode.seq_lens._base is cm.seq_lens
        assert md.decode.block_table._base is cm.block_table_tensor
