# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
"""ABI-3 device-plan coverage for the native MiniMax-M3 writer.

Pure CPU references compare segment slicing and greedy assignment independently.
GPU tests call the actual vLLM op with zero QKV rows and retained live metadata;
they exercise its real plan CTAs, host validation and boxed schema. No old writer,
optional MSA package, source-path shim or additional planner JIT is used.

Reference helpers are adapted from icp-kernels' MIT-licensed
tests/scorer/regression/test_icp_device_plan.py (4232bce).

Copyright (c) 2026 MiniMax
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import random

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

QO_TILE, KV_TILE, HEADS = 128, 256, 4
ICP_C, ICP_RANK, ROWS = 2, 1, 64
OPEN_END, MIN_SPLIT_TILES, TILE_PAGES = 0x7FFFFFFF, 16, 2
PLAN_ROWS = (
    "qo_segment_offsets",
    "kv_segment_offsets",
    "qo_segment_lens",
    "kv_segment_lens",
    "qo_offset",
)


def _kv_iters(qt, kv_len, offset):
    eff = min((qt + 1) * QO_TILE + offset, kv_len)
    return 0 if eff <= 0 else -(-eff // KV_TILE)


def _tile_cost(iters):
    return int(np.float32(43.0) * np.float32(iters) + np.float32(110.0))


def direct_greedy_reference(qo_lens, kv_lens, offsets, num_heads, nb):
    """plan.cuh direct_greedy, nosplit/causal/pack 1, both passes."""
    n = len(qo_lens)
    max_tiles = max((-(-q // QO_TILE) for q in qo_lens), default=0)

    def assignments():
        cost = [165] * nb
        for qt in range(max_tiles - 1, -1, -1):
            for b in range(n):
                if qt >= -(-qo_lens[b] // QO_TILE):
                    continue
                ki = _kv_iters(qt, kv_lens[b], offsets[b])
                if ki <= 0:
                    continue
                tc = _tile_cost(ki)
                h_off = 0
                while h_off < num_heads:
                    bc = min(num_heads - h_off, nb)
                    order = sorted(range(nb), key=lambda i: (cost[i], i))[:bc]
                    for rank, bucket in enumerate(order):
                        yield bucket, (qt, h_off + rank, b)
                    for bucket in order:
                        cost[bucket] += tc
                    h_off += bc

    counts = [0] * nb
    for bucket, _ in assignments():
        counts[bucket] += 1
    offs = [0] * (nb + 1)
    for i in range(nb):
        offs[i + 1] = offs[i] + counts[i]
    info: list[tuple[int, int, int]] = [(-1, -1, -1)] * offs[nb]
    fill = [0] * nb
    for bucket, item in assignments():
        info[offs[bucket] + fill[bucket]] = item
        fill[bucket] += 1
    ranges = [(offs[i], offs[i] + counts[i]) for i in range(nb)]
    return ranges, info


def _request_for_row(qsl, num_reqs, row):
    lo, hi = 0, num_reqs
    while lo < hi:
        mid = (lo + hi) >> 1
        if qsl[mid + 1] <= row:
            lo = mid + 1
        else:
            hi = mid
    return lo


def local_trip(qt, kv_len, offset, *, rank=ICP_RANK, world=ICP_C, rows=ROWS):
    """The scorer loader's OnlyScoreIcp compute_effective_end, in KV tiles."""
    gb_avail = -(-kv_len // rows)
    gb_causal = -(-((qt + 1) * QO_TILE + offset) // rows)
    lb = max(-(-(min(gb_avail, gb_causal) - rank) // world), 1)
    return -(-lb // TILE_PAGES)


def device_plan_model(
    qsl,
    seq_lens,
    num_reqs,
    num_rows,
    width,
    slot,
    num_heads,
    nb,
    *,
    max_splits=1,
    row_begin=0,
    min_split_tiles=MIN_SPLIT_TILES,
    rank=ICP_RANK,
):
    """Mirror of icpDevicePlanBlock (split pre-pass, single rank pass, scatter)."""
    t0 = max(slot * width, row_begin)
    t1 = min((slot + 1) * width, num_rows)
    live_end = qsl[num_reqs] if num_reqs > 0 else 0
    end = min(t1, live_end)
    first = n = 0
    if t0 < end:
        first = _request_for_row(qsl, num_reqs, t0)
        n = _request_for_row(qsl, num_reqs, end - 1) - first + 1
    qo_off, kv_off = [0] * (n + 1), [0] * (n + 1)
    qo_len, kv_len, causal = [0] * n, [0] * n, [0] * n
    for b in range(n):
        req = first + b
        begin, stop, seq = qsl[req], qsl[req + 1], seq_lens[req]
        lo, hi = max(begin, t0), min(stop, t1)
        qo_off[b + 1] = hi - t0
        qo_len[b] = hi - lo
        kv_len[b] = seq
        causal[b] = seq - (stop - lo)
        kv_off[b + 1] = kv_off[b] + seq
    max_tiles = max((-(-q // QO_TILE) for q in qo_len), default=0)
    tiles = [
        (qt, b)
        for qt in range(max_tiles - 1, -1, -1)
        for b in range(n)
        if qt < -(-qo_len[b] // QO_TILE)
    ]
    split = max_splits > 1
    chunk = 0
    if split:
        total = sum(local_trip(qt, kv_len[b], causal[b], rank=rank) for qt, b in tiles)
        split = 0 < len(tiles) * num_heads < nb
        if split:
            chunk = max(min_split_tiles, -(-(total * num_heads) // nb))
    cost, count = [165] * nb, [0] * nb
    scratch = []
    for qt, b in tiles:
        ki = _kv_iters(qt, kv_len[b], causal[b])
        if ki <= 0:
            continue
        trip, piece, pieces = 0, 0, 1
        if split:
            trip = local_trip(qt, kv_len[b], causal[b], rank=rank)
            pieces = min(max_splits, max(1, trip // chunk))
            piece = -(-trip // pieces)
            pieces = -(-trip // piece)
        for sp in range(pieces):
            kb = sp * piece
            ke = OPEN_END if sp == pieces - 1 else kb + piece
            tc = _tile_cost(min(kb + piece, trip) - kb) if split else _tile_cost(ki)
            h_off = 0
            while h_off < num_heads:
                bc = min(num_heads - h_off, nb)
                ranks = {}
                for tid in range(nb):
                    ranks[tid] = sum(
                        1
                        for i in range(nb)
                        if cost[i] < cost[tid] or (cost[i] == cost[tid] and i < tid)
                    )
                took: list[tuple[int, int]] = [(-1, -1)] * bc
                for tid in range(nb):
                    if ranks[tid] < bc:
                        took[ranks[tid]] = (tid, count[tid])
                for rk, (tid, pos) in enumerate(took):
                    scratch.append(((qt, h_off + rk, b), (kb, ke, sp), tid, pos))
                    count[tid] += 1
                    cost[tid] += tc
                h_off += bc
    offs = [0] * nb
    for i in range(1, nb):
        offs[i] = offs[i - 1] + count[i - 1]
    info: list[tuple[int, int, int]] = [(-1, -1, -1)] * len(scratch)
    kv_ranges: list[tuple[int, int, int]] = [(-1, -1, -1)] * len(scratch)
    for item, kv_range, bucket, pos in scratch:
        info[offs[bucket] + pos] = item
        kv_ranges[offs[bucket] + pos] = kv_range
    return {
        "header": (n, first, len(scratch), t0),
        "qo_segment_offsets": qo_off,
        "kv_segment_offsets": kv_off,
        "qo_segment_lens": qo_len,
        "kv_segment_lens": kv_len,
        "qo_offset": causal,
        "ranges": [(offs[i], offs[i] + count[i]) for i in range(nb)],
        "info": info,
        "kv_ranges": kv_ranges,
    }


def _random_batch(rng, *, exact):
    num_reqs = rng.randint(1, 12)
    q, ctx, drafts = [], [], []
    for _ in range(num_reqs):
        kind = rng.random()
        if kind < 0.35:
            q.append(rng.randint(1, 4))  # decode / verify
            ctx.append(rng.randint(0, 300000))
            drafts.append(0 if exact else rng.randint(0, 3))
        else:
            q.append(
                rng.choice(
                    [1, 63, 64, 127, 128, 129, 255, 256, 257, rng.randint(1, 3000)]
                )
            )
            ctx.append(rng.choice([0, 0, 1, 127, 128, rng.randint(0, 200000)]))
            drafts.append(0)
    qsl = [0]
    for x in q:
        qsl.append(qsl[-1] + x)
    seq = [c + x for c, x in zip(ctx, q)]
    bound = [s + d for s, d in zip(seq, drafts)]
    pad = rng.choice([0, 0, 1, 7, 64])
    return qsl, seq, bound, num_reqs, qsl[-1] + pad


def _many_segment_batch(rng, num_reqs):
    q = [rng.randint(1, 3) for _ in range(num_reqs)]
    qsl = [0]
    for x in q:
        qsl.append(qsl[-1] + x)
    seq = [rng.randint(0, 5000) + x for x in q]
    return qsl, seq, num_reqs, qsl[-1]


def _mixed_batch(rng):
    """Decode rows first (the CuTe band), then prefill rows."""
    nd_reqs = rng.randint(1, 20)
    q = [rng.randint(1, 4) for _ in range(nd_reqs)]
    q += [
        rng.choice([1, 63, 128, 129, rng.randint(1, 2500)])
        for _ in range(rng.randint(1, 5))
    ]
    ctx = [rng.randint(0, 200000) for _ in q]
    qsl = [0]
    for x in q:
        qsl.append(qsl[-1] + x)
    seq = [c + x for c, x in zip(ctx, q, strict=True)]
    return qsl, seq, len(q), qsl[-1] + rng.choice([0, 3]), qsl[nd_reqs]


def _check_native_slot(seg, work, head, ranges, dev, slot, nb):
    n, _, items, _ = dev["header"]
    assert tuple(head[slot]) == dev["header"]
    for r, key in enumerate(PLAN_ROWS):
        assert seg[slot, r, : n + 1 if r < 2 else n].tolist() == dev[key], key
    got = [(int(x) & 0xFFFFFFFF, int(x) >> 32) for x in work[slot, :nb]]
    assert got == dev["ranges"]
    assert [_unpack(x) for x in work[slot, nb : nb + items]] == dev["info"]
    kv = list(zip(*(ranges[slot, r, :items].tolist() for r in range(3))))
    assert kv == dev["kv_ranges"]


def _unpack(info):
    info = int(info) & 0xFFFFFFFFFFFFFFFF
    return (info >> 32, (info >> 16) & 0xFFFF, info & 0xFFFF)


def split_cells(plan):
    """Check a plan's KV ranges and return {(qt, h, b): covered local pages}.

    Per (qt, h, b): ranges are disjoint, contiguous from 0, cover exactly
    [0, trip) of the scorer's own trip count, and exactly one item is
    open-ended (the tail owner, which completes columns [trip, pwave)).
    """
    trips = {}
    for b in range(len(plan["qo_segment_lens"])):
        for qt in range(-(-plan["qo_segment_lens"][b] // QO_TILE)):
            trips[qt, b] = local_trip(
                qt, plan["kv_segment_lens"][b], plan["qo_offset"][b]
            )
    by_item: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for item, kv in zip(plan["info"], plan["kv_ranges"], strict=True):
        by_item.setdefault(item, []).append(kv)
    cells = {}
    for (qt, h, b), pieces in by_item.items():
        trip = trips[qt, b]
        pieces = sorted(pieces)
        assert sorted(sp for _, _, sp in pieces) == list(range(len(pieces)))
        tails = [kb for kb, ke, _ in pieces if ke == OPEN_END]
        assert len(tails) == 1, f"{(qt, h, b)}: {len(tails)} tail owners"
        assert pieces[-1][1] == OPEN_END, "the tail owner must be the last range"
        covered, cursor = set(), 0
        for kb, ke, _ in pieces:
            assert kb == cursor, f"{(qt, h, b)}: gap or overlap at {kb} vs {cursor}"
            stop = trip if ke == OPEN_END else ke
            assert kb < stop <= trip, f"{(qt, h, b)}: empty or overlong [{kb},{ke})"
            covered |= set(range(kb, stop))
            cursor = stop
        assert covered == set(range(trip))
        cells[qt, h, b] = covered
    assert set(cells) == {(qt, h, b) for (qt, b) in trips for h in range(HEADS)}
    return cells


def _host_reference(qsl, seq, num_rows, width, slot, nb, row_begin=0):
    """Slice requests directly, then use the independent sorted greedy planner."""
    t0, t1 = max(slot * width, row_begin), min((slot + 1) * width, num_rows)
    selected = [
        req for req in range(len(seq)) if max(qsl[req], t0) < min(qsl[req + 1], t1)
    ]
    first = selected[0] if selected else 0
    requests = range(first, selected[-1] + 1) if selected else range(0)
    qo_off, kv_off, qo_len, kv_len, causal = [0], [0], [], [], []
    for req in requests:
        lo, hi = max(qsl[req], t0), min(qsl[req + 1], t1)
        qo_off.append(hi - t0)
        kv_off.append(kv_off[-1] + seq[req])
        qo_len.append(hi - lo)
        kv_len.append(seq[req])
        causal.append(seq[req] - (qsl[req + 1] - lo))
    ranges, info = direct_greedy_reference(qo_len, kv_len, causal, HEADS, nb)
    return dict(
        header=(len(qo_len), first, len(info), t0),
        qo_segment_offsets=qo_off,
        kv_segment_offsets=kv_off,
        qo_segment_lens=qo_len,
        kv_segment_lens=kv_len,
        qo_offset=causal,
        ranges=ranges,
        info=info,
        kv_ranges=[(0, OPEN_END, 0)] * len(info),
    )


@pytest.mark.parametrize("width", [128, 256, 1024])
@pytest.mark.parametrize("nb", [3, 16])
def test_device_plan_reference_matches_independent_host_windows(width, nb):
    rng = random.Random(500 + width + nb)
    for _ in range(12):
        qsl, seq, num_reqs, num_rows, row_begin = _mixed_batch(rng)
        for slot in range(math.ceil(num_rows / width)):
            actual = device_plan_model(
                qsl,
                seq,
                num_reqs,
                num_rows,
                width,
                slot,
                HEADS,
                nb,
                row_begin=row_begin,
            )
            assert actual == _host_reference(
                qsl, seq, num_rows, width, slot, nb, row_begin
            )


def test_device_plan_reference_preserves_empty_requests_and_padding():
    qsl, seq = [0, 0, 64, 64, 129], [0, 300, 0, 1000]
    for slot in range(3):
        plan = device_plan_model(qsl, seq, 4, 384, 128, slot, HEADS, 16)
        assert plan == _host_reference(qsl, seq, 384, 128, slot, 16)
    assert plan["header"] == (0, 0, 0, 256)
    assert plan["ranges"] == [(0, 0)] * 16


def test_device_plan_reference_many_segments_crosses_scan_block():
    qsl, seq, num_reqs, num_rows = _many_segment_batch(random.Random(11), 700)
    largest = 0
    for slot in range(math.ceil(num_rows / 1024)):
        plan = device_plan_model(qsl, seq, num_reqs, num_rows, 1024, slot, HEADS, 16)
        largest = max(largest, plan["header"][0])
        assert plan == _host_reference(qsl, seq, num_rows, 1024, slot, 16)
    assert largest > 256


def test_device_plan_abi3_split_ranges_cover_the_unsplit_cells_once():
    qsl, seq = [0, 128], [1_000_000]
    unsplit = device_plan_model(qsl, seq, 1, 128, 128, 0, HEADS, 148)
    split = device_plan_model(qsl, seq, 1, 128, 128, 0, HEADS, 148, max_splits=64)
    assert split["header"][2] > unsplit["header"][2]
    assert split["header"][2] <= unsplit["header"][2] + 148
    assert split_cells(split) == split_cells(unsplit)


@pytest.fixture(scope="module")
def native_writer():
    if not current_platform.is_device_capability_family(100):
        pytest.skip("native device-plan checks require a built SM10x writer")
    import vllm._custom_ops  # noqa: F401 (registers the native op)

    return torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert


def _run_native(
    writer,
    qsl,
    seq,
    num_reqs,
    num_rows,
    width,
    nb,
    max_segments,
    max_work,
    max_splits=1,
    row_begin=0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the real op's metadata-only grid, including all native ABI guards."""
    slots = math.ceil(num_rows / width)

    def full(shape, dtype=torch.int32):
        return torch.full(shape, -1, dtype=dtype, device="cuda")

    weights = torch.zeros(128, dtype=torch.bfloat16, device="cuda")
    segments = full((slots, 5, max_segments + 1))
    work = full((slots, nb + 3 * max_work), torch.int64)
    header = full((slots, 4))
    ranges = full((slots, 3, 2 * max_work))
    writer(
        qkv=torch.empty(0, 41 * 128, dtype=torch.bfloat16, device="cuda"),
        q_norm_weight=weights,
        k_norm_weight=weights,
        cos_sin_cache=torch.zeros(1, 64, dtype=torch.bfloat16, device="cuda"),
        positions=full((0,), torch.int64),
        num_heads=32,
        num_kv_heads=2,
        rotary_dim=64,
        eps=1e-6,
        index_q_norm_weight=weights,
        index_k_norm_weight=weights,
        num_index_heads=4,
        slot_mapping=full((0,), torch.int64),
        index_slot_mapping=None,
        kv_cache=full((1, 4, 128, 72), torch.uint8),
        index_cache=torch.empty(1, 64, 128, dtype=torch.float8_e4m3fn, device="cuda"),
        block_size=128,
        q_out=None,
        index_q_out=None,
        kv_cache_dtype="nvfp4",
        kv_k_scale=torch.ones((), device="cuda"),
        kv_v_scale=torch.ones((), device="cuda"),
        index_block_tokens=128,
        index_rows_per_rank=64,
        index_rank=ICP_RANK,
        index_world_size=ICP_C,
        enable_pdl=False,
        write_icp_metadata=True,
        icp_query_start_loc=torch.tensor(qsl, dtype=torch.int32, device="cuda"),
        icp_seq_lens=torch.tensor(seq, dtype=torch.int32, device="cuda"),
        icp_num_reqs=num_reqs,
        icp_positions=full((num_rows,), torch.int64),
        icp_active=full((num_rows,), torch.bool),
        icp_local_nvalid=full((num_rows,)),
        icp_local_forced=full((num_rows,)),
        icp_global_nvalid=full((num_rows,)),
        icp_forced=full((num_rows,)),
        icp_n_ordinary=full((num_rows,)),
        icp_candidates=full((num_rows, 4, 16, 2), torch.float32),
        icp_qo_offsets=full((slots, max(1, max_segments))),
        icp_chunk_width=width,
        icp_plan_segments=segments,
        icp_plan_work=work,
        icp_plan_header=header,
        icp_plan_num_ctas=nb,
        icp_plan_num_heads=HEADS,
        icp_plan_ranges=ranges,
        icp_plan_max_splits=max_splits,
        icp_plan_row_begin=row_begin,
        icp_plan_tile_pages=TILE_PAGES,
        icp_plan_min_split_tiles=MIN_SPLIT_TILES,
    )
    return segments.cpu(), work.cpu(), header.cpu(), ranges.cpu()


@pytest.mark.parametrize("nb", [5, 148, 256])
@pytest.mark.parametrize("max_splits", [1, 16])
def test_native_planner_matches_reference(native_writer, nb, max_splits):
    rng = random.Random(nb)
    width, max_segments = 1024, 16
    max_work = HEADS * (math.ceil(width / QO_TILE) + max_segments) + nb
    for _ in range(6):
        qsl, seq, _, num_reqs, num_rows = _random_batch(rng, exact=True)
        result = _run_native(
            native_writer,
            qsl,
            seq,
            num_reqs,
            num_rows,
            width,
            nb,
            max_segments,
            max_work,
            max_splits,
        )
        for slot in range(result[2].shape[0]):
            expected = device_plan_model(
                qsl,
                seq,
                num_reqs,
                num_rows,
                width,
                slot,
                HEADS,
                nb,
                max_splits=max_splits,
            )
            _check_native_slot(*result, expected, slot, nb)


def test_native_planner_many_segments(native_writer):
    qsl, seq, num_reqs, num_rows = _many_segment_batch(random.Random(12), 700)
    width, nb, max_segments = 1024, 16, 1024
    max_work = HEADS * (math.ceil(width / QO_TILE) + max_segments)
    result = _run_native(
        native_writer, qsl, seq, num_reqs, num_rows, width, nb, max_segments, max_work
    )
    assert max(int(row[0]) for row in result[2]) > 256
    for slot in range(result[2].shape[0]):
        expected = device_plan_model(
            qsl, seq, num_reqs, num_rows, width, slot, HEADS, nb
        )
        _check_native_slot(*result, expected, slot, nb)


@pytest.mark.parametrize("max_splits", [1, 16])
def test_native_planner_mid_slot_windows(native_writer, max_splits):
    rng = random.Random(77 + max_splits)
    nb, width, max_segments = 148, 256, 64
    max_work = HEADS * (math.ceil(width / QO_TILE) + max_segments) + nb
    for _ in range(6):
        qsl, seq, num_reqs, num_rows, row_begin = _mixed_batch(rng)
        result = _run_native(
            native_writer,
            qsl,
            seq,
            num_reqs,
            num_rows,
            width,
            nb,
            max_segments,
            max_work,
            max_splits,
            row_begin,
        )
        for slot in range(result[2].shape[0]):
            expected = device_plan_model(
                qsl,
                seq,
                num_reqs,
                num_rows,
                width,
                slot,
                HEADS,
                nb,
                max_splits=max_splits,
                row_begin=row_begin,
            )
            _check_native_slot(*result, expected, slot, nb)


def test_native_planner_exact_capacity(native_writer):
    qsl, seq, width, nb = [0, 64, 129], [300, 1000], 256, 3
    expected = device_plan_model(qsl, seq, 2, 129, width, 0, HEADS, nb)
    result = _run_native(
        native_writer,
        qsl,
        seq,
        2,
        129,
        width,
        nb,
        expected["header"][0],
        expected["header"][2],
    )
    _check_native_slot(*result, expected, 0, nb)


def test_native_planner_long_context_split_ranges(native_writer):
    qsl, seq, width, nb = [0, 128], [1_000_000], 128, 148
    expected = device_plan_model(qsl, seq, 1, 128, width, 0, HEADS, nb, max_splits=64)
    assert expected["header"][2] > 4 * HEADS
    result = _run_native(
        native_writer,
        qsl,
        seq,
        1,
        128,
        width,
        nb,
        expected["header"][0],
        expected["header"][2],
        max_splits=64,
    )
    _check_native_slot(*result, expected, 0, nb)


@pytest.mark.parametrize(
    ("max_segments", "max_work", "message"),
    [(0, 8, "segment capacity"), (2, 0, "3 \\* max_work")],
)
def test_native_planner_rejects_empty_capacity(
    native_writer, max_segments, max_work, message
):
    # Invalid static capacities are rejected before launching a device trap.
    with pytest.raises(RuntimeError, match=message):
        _run_native(
            native_writer,
            [0, 64, 129],
            [300, 1000],
            2,
            129,
            256,
            3,
            max_segments,
            max_work,
        )
