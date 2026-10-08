# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host contracts of the ICP indexer's windows, chunking, profiles and checks.

No CUDA work: these pin the host-side row mapping and startup enumerations the
GPU plan (band split, single decode chunk, pre-warmed CuTe profiles) rests on.
"""

import functools
import importlib.util
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    # The W == 1 CuteDSL decode op is imported by the package but never run
    # here; stub it on hosts without the CUTLASS DSL.
    _stub: Any = types.ModuleType(
        "vllm.models.minimax_m3.nvidia.ops.index_decode_score"
    )
    _stub.minimax_m3_index_decode_score_cutedsl = None
    sys.modules.setdefault(_stub.__name__, _stub)

from vllm.models.minimax_m3.nvidia import indexer_icp as icp  # noqa: E402
from vllm.models.minimax_m3.nvidia.ops.icp_dispatch import (  # noqa: E402
    exchange_token_capacities,
    select_token_capacity,
)

msa_icp = pytest.importorskip("fmha_sm100.icp")

MAX_TOKENS = 16384
CAPTURE_SIZES = [1, 2, 4, *range(8, 256, 8), *range(256, 513, 16)]


# ---- item 1: the band split --------------------------------------------------


def test_pure_decode_windows_keep_the_captured_selector_extents():
    capacities = exchange_token_capacities(
        max_token_capacity=MAX_TOKENS, cudagraph_capture_sizes=CAPTURE_SIZES
    )
    for num_tokens in (1, 24, 512, 3000):
        cap = select_token_capacity(capacities, num_tokens)
        windows = icp.icp_row_windows(
            num_tokens=num_tokens,
            cap=cap,
            cute_rows=num_tokens,
            decode_chunk=1024,
            prefill_width=1024,
            has_prefill=False,
        )
        assert [w.grid_begin for w in windows] == list(range(0, num_tokens, 1024))
        for w in windows:
            assert w.cute and w.decode_selector
            assert w.select_rows == min(1024, cap - w.grid_begin)


def test_live_metadata_hands_the_device_plan_to_the_writer():
    from vllm.models.minimax_m3.nvidia.ops.icp_metadata import ICPLiveMetadata

    store = object()
    fields = dict.fromkeys(ICPLiveMetadata.__dataclass_fields__)
    fields.update(num_reqs=1, chunk_width=128, device_plan=store)
    kwargs = ICPLiveMetadata(**fields).writer_kwargs()
    assert kwargs["icp_device_plan"] is store
    assert kwargs["icp_plan_row_begin"] == 0
    fields.update(device_plan=None)
    kwargs = ICPLiveMetadata(**fields).writer_kwargs()
    assert kwargs["icp_device_plan"] is None
    assert kwargs["icp_plan_row_begin"] == 0


# ---- item 2: one decode chunk per captured size ------------------------------


@pytest.mark.parametrize("max_model_len", [131072, 262144, 524288, 1048576])
@pytest.mark.parametrize("token_capacity", [2048, 8192, 16384])
def test_decode_chunk_is_r13_fixed_budget(max_model_len, token_capacity):
    heads_group = 4
    chunk = icp.icp_decode_chunk_tokens(
        max_model_len=max_model_len,
        heads_group=heads_group,
        token_capacity=token_capacity,
    )
    assert chunk == icp.icp_outer_chunk_tokens(
        max_model_len=max_model_len,
        heads_group=heads_group,
        token_capacity=token_capacity,
        budget_bytes=icp.ICP_SCORE_SCRATCH_BUDGET_BYTES,
    )
    columns = icp._cdiv(max_model_len, 128)
    tile_bytes = icp.ICP_MSA_QUERY_TILE_TOKENS * heads_group * columns * 5
    assert chunk * heads_group * columns * 5 <= max(
        icp.ICP_SCORE_SCRATCH_BUDGET_BYTES, tile_bytes
    )
    capacities = exchange_token_capacities(
        max_token_capacity=token_capacity, cudagraph_capture_sizes=CAPTURE_SIZES
    )
    for size in [*CAPTURE_SIZES, 1024]:
        if size > token_capacity:
            continue
        cap = select_token_capacity(capacities, size)
        windows = icp.icp_row_windows(
            num_tokens=size,
            cap=cap,
            cute_rows=size,
            decode_chunk=chunk,
            prefill_width=chunk,
            has_prefill=False,
        )
        assert len(windows) == icp._cdiv(size, chunk), (size, chunk)


def test_decode_chunk_is_128_rows_at_1m():
    chunk = icp.icp_decode_chunk_tokens(
        max_model_len=1048576, heads_group=4, token_capacity=MAX_TOKENS
    )
    assert chunk == 128
    assert not hasattr(icp, "_DECODE_CHUNK_ENV")


# ---- item 3: one CuTe compile per query length, runtime windows --------------


@pytest.fixture(scope="module")
def msa_scorer_api():
    pytest.importorskip("fmha_sm100.icp.scorer.decode.icp_decode_score")
    return icp._msa_cute_decode_module()


def test_cute_window_is_the_padded_extent_and_live_requests_only():
    def window(**kw):
        return icp.icp_cute_decode_window(chunk=1024, **kw)

    # FULL graph: the padded extent and padded request count, per graph.
    assert window(query_len=1, token_begin=0, rows=24, num_reqs=24) == (0, 24, 0, 24)
    assert window(query_len=4, token_begin=0, rows=512, num_reqs=128) == (
        0,
        512,
        0,
        128,
    )
    # Mixed step: the decode prefix never reaches the prefill requests.
    assert window(query_len=4, token_begin=0, rows=8, num_reqs=5) == (0, 8, 0, 2)
    # Second chunk of a long decode band.
    assert window(query_len=1, token_begin=1024, rows=300, num_reqs=1324) == (
        1024,
        300,
        1024,
        300,
    )
    assert window(query_len=1, token_begin=1024, rows=300, num_reqs=1000) is None


def test_prewarm_compiles_one_scorer_per_query_length_without_a_window():
    calls = []
    builder = object.__new__(icp.MiniMaxM3IndexerMSAMetadataBuilder)
    builder.icp_cute_query_lens = {1, 4}
    builder.icp_cute_scorers = {}
    builder.icp_rank, builder.icp_c = 1, 2
    builder.icp_cute_factory_has_page_size = False
    builder.icp_cute_factory_has_pdl = True
    builder.icp_decode_pdl = False

    def get_icp_decode_scorer(**kw):
        calls.append(kw)
        return object()

    builder.icp_cute_module = SimpleNamespace(
        get_icp_decode_scorer=get_icp_decode_scorer
    )
    builder._prewarm_cute_decode(torch.device("cpu"))
    assert sorted(builder.icp_cute_scorers) == [1, 4]
    assert [c["query_len"] for c in calls] == [1, 4]
    for call in calls:
        assert not {"token_begin", "token_count", "request_begin"} & set(call)
        assert call["use_pdl"] is False
        assert call["split_k"] == 128  # r13's fixed split


def test_cute_launch_reaches_the_native_scan_with_runtime_window(msa_scorer_api):
    launches = []
    scorer = msa_scorer_api._PreparedIcpDecodeScorer(
        msa_scorer_api._DecodeProfile(4, 1, 2, 128),
        0,
        (10, 3),
        lambda *args: launches.append(args),
        torch,
    )
    names = ["table", "qsl", "seq", "positions", "active", "scores", "valid"]
    bound = [SimpleNamespace(name=n) for n in names]
    for begin, count in ((0, 16), (512, 64)):
        run = icp._bind_cute_launch(
            scorer.bind(
                token_begin=begin,
                token_count=count,
                request_begin=begin // 4,
                request_count=count // 4,
            ),
            *bound,
        )
        run("q", "kv", "k_pages", 1.0)
    table, qsl, seq, positions, active, scores, valid = bound
    tensors = ("q", "kv", table, scores, seq, qsl, positions, active, valid)
    # The pinned split rides along as the fifth launch scalar.
    assert launches == [
        (*tensors, 0, 16, 0, 4, 128),
        (*tensors, 512, 64, 128, 16, 128),
    ]


# ---- item 4: validators run once per step unless debugging -------------------


def _forward_fixture(debug, num_chunks=2):
    impl = object.__new__(icp.MiniMaxM3IndexerMSAImpl)
    impl.icp_c, impl.icp_rank, impl.prefix = 2, 0, "layer"
    impl.num_index_heads, impl.index_head_dim, impl.scale = 2, 128, 1.0
    impl.topk_indices_buffer = torch.zeros(8, 16, dtype=torch.int32)
    impl._icp_debug_checks = debug
    launches: list[tuple] = []
    chunks = [
        icp.MiniMaxM3IndexerICPChunk(
            plan={},
            token_begin=0,
            num_live_tokens=4,
            block_table=None,
            block_table_row_stride=16,
            block_table_row_begin=0,
            qo_offset=None,
            live_blocks=1,
            score_extent=1,
            select_extent=4,
            topk_num_valid_pages=None,
            icp_forced_column_v1=None,
            icp_active_rows=None,
            launch_score=lambda *a, i=i: launches.append(("score", i)),
            select=lambda i=i, **kw: launches.append(("select", i, kw["validate"])),
        )
        for i in range(num_chunks)
    ]
    candidates = torch.zeros(8, 4, 16, 2)
    st = SimpleNamespace(
        chunks=chunks,
        candidates=candidates,
        token_capacity=8,
        has_prefill=False,
        forced=None,
        n_ordinary=None,
        checked=False,
        exchange_candidates=candidates.view(torch.int32),
        exchange_buf=None,
        exchange_out=None,
        decode_chunk=128,
        fused_decode=False,
    )
    md = icp.MiniMaxM3IndexerMSAMetadata(
        seq_lens=None,
        max_seq_len=0,
        slot_mapping=None,
        num_actual_tokens=4,
        num_decodes=4,
        num_decode_tokens=4,
        num_prefills=0,
        num_prefill_tokens=0,
        icp=st,
    )
    return impl, md, launches


@pytest.mark.parametrize("debug", [False, True])
def test_per_layer_validators_are_gated_by_the_debug_flag(monkeypatch, debug):
    impl, md, launches = _forward_fixture(debug)
    calls = {"invocation": 0, "chunk": 0}
    real_invocation = impl._check_icp_invocation
    real_chunk = impl._check_icp_chunk

    def count_invocation(*args):
        calls["invocation"] += 1
        return real_invocation(*args)

    def count_chunk(*args):
        calls["chunk"] += 1
        return real_chunk(*args)

    monkeypatch.setattr(impl, "_check_icp_invocation", count_invocation)
    monkeypatch.setattr(impl, "_check_icp_chunk", count_chunk)
    exchanges = []

    class _Exchange:
        decode_profile = {"exchange": "k5t", "merge": "network", "d3": 0, "pdl": 1}

        def __call__(self, candidates, **kw):
            exchanges.append((kw["layer_idx"], candidates, kw["out"]))

    monkeypatch.setattr(icp, "_decode_profile_logged", False)
    layers = 5
    for layer in range(layers):
        impl._forward_icp(
            torch.zeros(4, 4 * 128),
            index_kv=torch.zeros(2, 64, 128),
            index_md=md,
            layer_idx=layer,
            candidate_exchange=_Exchange(),
        )
    per_layer = layers if debug else 1
    assert calls == {"invocation": per_layer, "chunk": 2 * per_layer}
    assert [layer for layer, _, _ in exchanges] == list(range(layers))
    # The exchange views are bound once per step, not rebuilt per layer.
    assert all(c is md.icp.exchange_candidates for _, c, _ in exchanges)
    assert all(out is exchanges[0][2] for _, _, out in exchanges)
    assert exchanges[0][2].shape[0] == md.icp.token_capacity
    for layer in range(layers):
        validate = debug or layer == 0
        assert launches[4 * layer : 4 * layer + 4] == [
            ("score", 0),
            ("select", 0, validate),
            ("score", 1),
            ("select", 1, validate),
        ]


class _FusedExchange:
    fused_decode = True
    decode_profile = {"exchange": "k5t", "merge": "network", "d3": 1, "pdl": 1}

    def __init__(self, launches):
        self.launches = launches

    def publish(self, scores, geometry, *, layer_idx, token_offset):
        self.launches.append(
            (
                "publish",
                layer_idx,
                tuple(range(token_offset, token_offset + scores.shape[0])),
                tuple(geometry.active_rows.tolist()),
            )
        )

    def __call__(self, candidates, **kw):
        self.launches.append(
            ("merge", kw["layer_idx"], candidates.shape[0], kw["out"].shape[0])
        )


def test_d3_publishes_every_window_and_padding_before_one_merge_per_layer():
    impl, md, launches = _forward_fixture(debug=False)
    st = md.icp
    assert st is not None
    st.fused_decode = st.fused_ready = True

    def bind_publish(begin, rows, active):
        geometry = msa_icp.CandidateGeometry(
            local_valid_blocks=torch.full((rows,), int(active), dtype=torch.int32),
            forced_column=torch.full((rows,), -1, dtype=torch.int32),
            active_rows=torch.full((rows,), active, dtype=torch.bool),
        )
        return functools.partial(
            icp._icp_fused_publish, torch.empty(rows, 4, 1), geometry, begin
        )

    for chunk, begin in zip(st.chunks, (0, 2), strict=True):
        chunk.token_begin = begin
        chunk.num_live_tokens = chunk.select_extent = 2
        chunk.fused_select = bind_publish(begin, 2, True)
    st.fused_padding = (bind_publish(4, 4, False),)
    exchange = _FusedExchange(launches)

    for layer in range(5):
        launches.clear()
        impl._forward_icp(
            torch.zeros(4, 4 * 128),
            index_kv=torch.zeros(2, 64, 128),
            index_md=md,
            layer_idx=layer,
            candidate_exchange=exchange,
        )
        assert launches == [
            ("score", 0),
            ("publish", layer, (0, 1), (True, True)),
            ("score", 1),
            ("publish", layer, (2, 3), (True, True)),
            ("publish", layer, (4, 5, 6, 7), (False, False, False, False)),
            ("merge", layer, st.token_capacity, st.token_capacity),
        ]
        published_rows = [
            row for event in launches if event[0] == "publish" for row in event[2]
        ]
        assert published_rows == list(range(st.token_capacity))


def test_d3_missing_fused_binding_fails_before_score_or_exchange():
    impl, md, launches = _forward_fixture(debug=False)
    st = md.icp
    assert st is not None
    st.fused_decode = True
    st.fused_ready = False
    st.fused_padding = ()

    with pytest.raises(RuntimeError, match="window without a fused selector binding"):
        impl._forward_icp(
            torch.zeros(4, 4 * 128),
            index_kv=torch.zeros(2, 64, 128),
            index_md=md,
            layer_idx=0,
            candidate_exchange=_FusedExchange(launches),
        )
    assert launches == []


def test_debug_flag_parsing(monkeypatch):
    monkeypatch.delenv(icp._DEBUG_CHECKS_ENV, raising=False)
    assert icp._debug_checks_enabled() is False
    monkeypatch.setenv(icp._DEBUG_CHECKS_ENV, "1")
    assert icp._debug_checks_enabled() is True
    monkeypatch.setenv(icp._DEBUG_CHECKS_ENV, "maybe")
    with pytest.raises(ValueError):
        icp._debug_checks_enabled()
