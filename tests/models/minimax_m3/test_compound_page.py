# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU byte gates for the TP2/P128 NVFP4 compound page."""

import pytest
import torch

from vllm.models.minimax_m3.common.compound_page import (
    CompoundPageError,
    CompoundPageLayout,
    bind_compound_cache,
    compound_cache_identity,
)

PAGE_BYTES = 45_056
MAIN_BYTES = 36_864
INDEX_BYTES = 8_192


@pytest.fixture
def layout():
    return CompoundPageLayout.build(tp_size=2)


def test_page_geometry_is_the_current_packed_main_plus_owned_index(layout):
    assert layout.main_shape == (4, 128, 72)
    assert layout.main_span_bytes == MAIN_BYTES
    assert layout.index_shape == (64, 128)
    assert layout.index_span_bytes == INDEX_BYTES
    assert layout.index_offset_bytes == MAIN_BYTES
    assert layout.compound_span_bytes == PAGE_BYTES


@pytest.mark.parametrize("rank", [0, 1])
def test_shared_backing_aliases_preserve_offsets_and_neighbor_canaries(layout, rank):
    # Two layers share one buffer. The second starts after a non-page-sized
    # prefix and an explicit gap, so absolute-offset mistakes cannot pass.
    offsets = [256, 256 + 3 * PAGE_BYTES + 256]
    backing = torch.full((offsets[-1] + 3 * PAGE_BYTES + 256,), 0x71, dtype=torch.uint8)
    bound_layers = []
    for layer, offset in enumerate(offsets):
        raw = backing[offset : offset + 3 * PAGE_BYTES].view(3, 1, 1, PAGE_BYTES)
        raw.zero_()
        bound = bind_compound_cache(raw, layout, tp_rank=rank)
        bound_layers.append(bound)
        assert bound.main.shape == (3, 4, 128, 72)
        assert bound.main.stride() == (PAGE_BYTES, 9216, 72, 1)
        assert bound.index.shape == (3, 64, 128)
        assert bound.index.stride() == (PAGE_BYTES, 128, 1)
        assert bound.main.storage_offset() == offset
        assert bound.index.storage_offset() == offset + MAIN_BYTES
        assert bound.main.untyped_storage().data_ptr() == backing.data_ptr()
        assert bound.index.untyped_storage().data_ptr() == backing.data_ptr()
        for page in (2, 0, 1):
            bound.main[page].fill_(16 * (layer + 1) + page)
            bound.index[page].view(torch.uint8).fill_(128 + 16 * layer + page)

    expected = torch.full_like(backing, 0x71)
    for layer, offset in enumerate(offsets):
        for page in range(3):
            start = offset + page * PAGE_BYTES
            expected[start : start + MAIN_BYTES] = 16 * (layer + 1) + page
            expected[start + MAIN_BYTES : start + PAGE_BYTES] = 128 + 16 * layer + page
    assert torch.equal(backing, expected)
    assert compound_cache_identity(
        bound_layers[0].raw, layout
    ) != compound_cache_identity(bound_layers[1].raw, layout)


def test_raw_copy_and_zero_include_the_index_suffix(layout):
    raw = torch.zeros((4, 1, 1, PAGE_BYTES), dtype=torch.uint8)
    bound = bind_compound_cache(raw, layout, tp_rank=0)
    source = torch.arange(PAGE_BYTES, dtype=torch.int64).remainder(251).to(torch.uint8)
    raw[3, 0, 0].copy_(source)
    raw[1].copy_(raw[3])
    assert torch.equal(raw[1, 0, 0], source)
    assert torch.equal(bound.index[1].view(torch.uint8).flatten(), source[MAIN_BYTES:])
    raw[3].zero_()
    assert not bound.main[3].any()
    assert not bound.index[3].view(torch.uint8).any()
    assert torch.equal(raw[1, 0, 0], source)


@pytest.mark.parametrize("tp_size", [0, 1, 3, 4, 8])
def test_rejects_unqualified_tp_width(tp_size):
    with pytest.raises(CompoundPageError, match="TP2"):
        CompoundPageLayout.build(tp_size=tp_size)


@pytest.mark.parametrize("page_tokens", [64, 256, 512])
def test_rejects_unqualified_page_size(page_tokens):
    with pytest.raises(CompoundPageError, match="P128"):
        CompoundPageLayout.build(
            tp_size=2,
            physical_page_tokens=page_tokens,
        )


@pytest.mark.parametrize("bad_offset", [1, 8, 15])
def test_rejects_misaligned_layer_start(layout, bad_offset):
    backing = torch.zeros(PAGE_BYTES + 16, dtype=torch.uint8)
    raw = backing[bad_offset : bad_offset + PAGE_BYTES].view(1, 1, 1, PAGE_BYTES)
    with pytest.raises(CompoundPageError, match="align"):
        bind_compound_cache(raw, layout, tp_rank=0)


def test_rejects_interleaved_layers_until_that_stride_is_qualified(layout):
    raw = torch.zeros((6, 1, 1, PAGE_BYTES), dtype=torch.uint8)[::2]
    with pytest.raises(CompoundPageError, match="stride|contiguous"):
        bind_compound_cache(raw, layout, tp_rank=0)


def test_rejects_wrong_page_extent(layout):
    raw = torch.zeros((2, 1, 1, MAIN_BYTES), dtype=torch.uint8)
    with pytest.raises(CompoundPageError, match="stride"):
        bind_compound_cache(raw, layout, tp_rank=0)
