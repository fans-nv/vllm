# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public head-slot views and live main-attention inputs, with no native launch."""

from types import SimpleNamespace
from weakref import WeakValueDictionary, ref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.models.minimax_m3.common.compound_page import (
    CompoundPageLayout,
    SparseMainFormat,
    bind_compound_cache,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparsePrefillMetadata,
)
from vllm.models.minimax_m3.nvidia import msa_icp_main as main


def test_dense_nvfp4_policy_preserves_shared_sparse_cache_config():
    from vllm.config import CacheConfig
    from vllm.models.minimax_m3.nvidia.model import dense_layer_cache_config

    sparse = CacheConfig(
        cache_dtype="nvfp4", block_size=128, enable_prefix_caching=False
    )
    dense = dense_layer_cache_config(sparse)
    assert dense is not sparse
    assert dense.cache_dtype == "fp8"
    assert sparse.cache_dtype == "nvfp4"
    assert dense.block_size == sparse.block_size == 128
    assert dense.enable_prefix_caching is sparse.enable_prefix_caching is False


@pytest.mark.parametrize("cache_dtype", [None, "auto", "fp8", "bfloat16"])
def test_dense_cache_policy_preserves_other_configurations(cache_dtype):
    from vllm.config import CacheConfig
    from vllm.models.minimax_m3.nvidia.model import dense_layer_cache_config

    config = None if cache_dtype is None else CacheConfig(cache_dtype=cache_dtype)
    assert dense_layer_cache_config(config) is config


@pytest.mark.parametrize("prefix", [0, 16])
def test_compound_views_use_public_per_head_slots_without_repacking(prefix):
    pytest.importorskip("fmha_sm100.icp")
    layout = CompoundPageLayout.build(tp_size=2, main_format=SparseMainFormat.NVFP4)
    storage = torch.empty(prefix + 3 * layout.compound_span_bytes, dtype=torch.uint8)
    raw = storage[prefix:].view(3, 1, 1, layout.compound_span_bytes)
    bound = bind_compound_cache(raw, layout, 0)
    views = main.icp_main_cache_views(bound.main)
    assert layout.abi_version.endswith("compound-page.2")
    for name, offset, width in (
        ("k", 0, 64),
        ("k_sf", 8192, 8),
        ("v", 9216, 64),
        ("v_sf", 17408, 8),
    ):
        tensor = getattr(views, name)
        assert tensor.shape == (3, 2, 128, width)
        assert tensor.stride() == (45056, 18432, width, 1)
        assert tensor.storage_offset() == prefix + offset
        assert tensor.data_ptr() == raw.data_ptr() + offset
        tensor[1, 1, 127, width - 1] = 37
        assert storage[prefix + 45056 + offset + 18432 + 128 * width - 1] == 37
    assert views.k_sf_fp8.data_ptr() == views.k_sf.data_ptr()
    assert views.v_sf_fp8.data_ptr() == views.v_sf.data_ptr()
    assert bound.index.storage_offset() == prefix + 36864


class ViewsOnly(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func in {
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.view.default,
            torch.ops.aten.transpose.int,
        }, f"Main dispatch performed tensor work: {func}"
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("decode_rows,prefill_rows", [(4, 0), (0, 5), (4, 5)])
def test_main_routes_phases_with_live_calibrated_scales_and_both_query_outputs(
    monkeypatch, decode_rows, prefill_rows
):
    pytest.importorskip("fmha_sm100.icp")
    rows = decode_rows + prefill_rows
    impl = object.__new__(main.MiniMaxM3SparseICPImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 32, 2, 128
    impl.block_size, impl.topk_blocks, impl.scale = 128, 16, 0.125
    impl.bind_kv_cache(torch.empty(3, 4, 128, 72, dtype=torch.uint8))
    layer = SimpleNamespace(
        layer_name="sparse.0",
        topk_indices_buffer=torch.empty(rows, 2, 16, dtype=torch.int32),
        _k_scale=torch.tensor([0.75]),
        _v_scale=torch.tensor([1.25]),
    )
    query = torch.empty(rows, 32 * 128, dtype=torch.bfloat16)
    query_fp8 = torch.empty(rows, 32 * 128, dtype=torch.float8_e4m3fn)
    output = torch.empty_like(query)
    seq_lens = torch.tensor([301, 701], dtype=torch.int32)
    table = torch.empty(2, 8, dtype=torch.int32)
    decode = (
        main.ICPMainDecodeMetadata(
            seq_lens=seq_lens[:1],
            block_table=table[:1],
            decode_query_len=decode_rows,
            plan=object(),
            kv_indices=table[:1].view(-1),
            kv_indptr=torch.tensor([0, 8], dtype=torch.int32),
        )
        if decode_rows
        else None
    )
    prefill = (
        MiniMaxM3SparsePrefillMetadata(
            cu_seqlens_q=torch.tensor([0, prefill_rows], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 701], dtype=torch.int32),
            seq_lens=seq_lens[1:],
            context_lens=torch.tensor([701 - prefill_rows]),
            block_table=table[1:],
            max_query_len=prefill_rows,
            max_seq_len=701,
            total_kv_blocks=6,
        )
        if prefill_rows
        else None
    )
    md = main.MiniMaxM3SparseICPMetadata(
        seq_lens=seq_lens,
        max_seq_len=701,
        slot_mapping=torch.empty(rows, dtype=torch.int64),
        num_actual_tokens=rows,
        num_decodes=int(bool(decode_rows)),
        num_decode_tokens=decode_rows,
        num_prefills=int(bool(prefill_rows)),
        num_prefill_tokens=prefill_rows,
        decode=decode,
        prefill=prefill,
    )
    monkeypatch.setattr(
        main,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.layer_name: md}),
    )
    calls = []
    impl._run_decode = lambda *a, **kw: calls.append(("decode", a, kw))

    def csr(*a, **kw):
        calls.append(("csr", a, kw))
        return "rows", "indices", "schedule"

    impl._build_k2q_csr = csr
    impl._prefill_attend = lambda *a, **kw: calls.append(("prefill", a, kw))
    raw_cache = torch.empty(0)
    with ViewsOnly():
        result = impl.forward(layer, query, raw_cache, output, query_fp8=query_fp8)
    assert result is output
    assert [name for name, _, _ in calls] == (
        (["decode"] if decode_rows else [])
        + (["csr", "prefill"] if prefill_rows else [])
    )
    for name, args, kwargs in calls:
        if name == "decode":
            assert decode is not None
            assert args[0] is decode.plan
            assert args[1].data_ptr() == query_fp8.data_ptr()
            assert kwargs["seq_lens"] is decode.seq_lens
            assert kwargs["kv_indices"] is decode.kv_indices
            assert kwargs["kv_indptr"] is decode.kv_indptr
            assert kwargs["kv_global_scale"] == (layer._k_scale, layer._v_scale)
            assert kwargs["out"].shape[0] == decode_rows
        elif name == "prefill":
            assert prefill is not None
            assert args[0].data_ptr() == query[decode_rows:].data_ptr()
            assert args[0].dtype == torch.bfloat16
            assert args[5] is layer._k_scale and args[6] is layer._v_scale
            assert kwargs["page_table"] is prefill.block_table
            assert kwargs["seqused_k"] is prefill.seq_lens
            assert kwargs["schedule"] == "schedule"
            assert kwargs["out"].shape[0] == prefill_rows


def test_main_decode_plan_uses_one_streamk_launch_and_reuses_schedule(monkeypatch):
    pytest.importorskip("fmha_sm100.icp")
    from fmha_sm100 import decode_q8kv4
    from fmha_sm100.decode_q8kv4 import interface, jit

    calls = []

    def plan_decode(*, split_mode=None, **kwargs):
        calls.append((split_mode, kwargs))
        return object()

    monkeypatch.setattr(main, "require_msa_icp", lambda: None)
    monkeypatch.setattr(decode_q8kv4, "plan_decode", plan_decode)
    monkeypatch.setattr(interface, "_get_cpp", lambda: None)
    monkeypatch.setattr(jit, "get_plan_fn", lambda **kw: None)
    monkeypatch.setattr(jit, "get_fmha_fwd_variant", lambda **kw: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    plans = main.ICPMainDecodePlans(32, 2, torch.device("cpu"))
    first = plans.get(3, 4)
    assert plans.get(3, 4) is first
    assert len(calls) == 1
    assert calls[0][0] == "streamk"
    assert calls[0][1]["num_kv_splits"] == 1
    assert calls[0][1]["block_scale_shift"] == 3
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert plans.get(3, 4) is first
    with pytest.raises(RuntimeError, match="not prebuilt"):
        plans.get(4, 4)


def test_main_decode_plan_lifetime_follows_live_models_not_temporary_kv(monkeypatch):
    from vllm.models.minimax_m3.nvidia.sparse_attention_icp import (
        MiniMaxM3SparseICPAttention,
    )
    from vllm.v1.worker.gpu.shutdown import shutdown_model_resources

    def init(plans, *_args):
        # Stand in for a native DecodePlan's owned device buffers.
        plans._plans = {(1, 1): torch.empty(1)}

    monkeypatch.setattr(main.ICPMainDecodePlans, "__init__", init)
    monkeypatch.setattr(main, "_DECODE_PLANS", WeakValueDictionary())
    models = []
    for _ in range(2):
        model = torch.nn.Module()
        layer = object.__new__(MiniMaxM3SparseICPAttention)
        torch.nn.Module.__init__(layer)
        layer.impl = object.__new__(main.MiniMaxM3SparseICPImpl)
        layer.impl.decode_plans = main.get_icp_main_decode_plans(
            32, 2, torch.device("cuda:0")
        )
        model.add_module("attention", layer)
        models.append(model)
    first, second = models[0].attention, models[1].attention
    assert first.impl.decode_plans is second.impl.decode_plans
    plans_ref = ref(first.impl.decode_plans)
    schedule_ref = ref(first.impl.decode_plans._plans[(1, 1)])

    first.release_kv_cache()
    assert first.impl.decode_plans is second.impl.decode_plans
    assert schedule_ref() is not None
    shutdown_model_resources((models[0],))
    assert first.impl.decode_plans is None
    assert plans_ref() is second.impl.decode_plans
    assert schedule_ref() is not None
    shutdown_model_resources((models[1], models[1]))
    assert second.impl.decode_plans is None
    assert plans_ref() is None and schedule_ref() is None
    assert not main._DECODE_PLANS

    # Teardown remains idempotent even when model objects outlive the runner.
    shutdown_model_resources(models)
