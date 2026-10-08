# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Current allocator placement, packing and merge gates for compound pages."""

from dataclasses import replace

import pytest
import torch

from vllm.models.minimax_m3.common.compound_page import (
    CompoundPageError,
    CompoundPageLayout,
    SparseMainFormat,
    bind_compound_cache,
)
from vllm.models.minimax_m3.common.compound_spec import MiniMaxM3CompoundSpec
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheTensor,
    KVQuantMode,
    UniformTypeKVCacheSpecs,
    compute_layer_kv_cache_shape_bytes,
    compute_layout_strides,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

PAGE_BYTES = 45_056


@pytest.fixture
def layout():
    return CompoundPageLayout.build(tp_size=2, main_format=SparseMainFormat.NVFP4)


@pytest.fixture
def main_spec():
    return FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
        num_head_slots=4,
        state_content_bytes=72,
    )


@pytest.fixture
def compound_spec(main_spec, layout):
    return MiniMaxM3CompoundSpec.from_main_spec(main_spec, layout)


def test_compound_spec_appends_index_bytes_after_packed_main(main_spec, compound_spec):
    assert main_spec.page_size_bytes == 36_864
    assert not main_spec.uses_raw_page_view
    assert compound_spec.uses_raw_page_view
    assert compound_spec.index_region_bytes == 8192
    assert compound_spec.page_size_bytes == PAGE_BYTES
    assert compound_spec.page_size_padded is None
    assert compute_layer_kv_cache_shape_bytes(main_spec, 3) == (3, 4, 128, 72)
    assert compute_layer_kv_cache_shape_bytes(compound_spec, 3) == (3, 1, 1, PAGE_BYTES)


def test_actual_nvfp4_backend_customization_precedes_compound_wrap(layout):
    from vllm.models.minimax_m3.nvidia.sparse_attention_msa import (
        MiniMaxM3SparseMSANvfp4Backend,
    )

    unpacked = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )
    assert unpacked.page_size_bytes == 65_536
    with pytest.raises(CompoundPageError, match="customize"):
        MiniMaxM3CompoundSpec.from_main_spec(unpacked, layout)
    packed = MiniMaxM3SparseMSANvfp4Backend.customize_spec(unpacked)
    assert packed.page_size_bytes == 36_864
    assert (
        MiniMaxM3CompoundSpec.from_main_spec(packed, layout).page_size_bytes
        == PAGE_BYTES
    )


def test_allocator_places_two_layers_at_real_offsets_without_overlap(
    compound_spec, layout
):
    blocks = 3
    layer_span = blocks * PAGE_BYTES
    # Exercise both the backing tensor's storage offset and placement offset.
    backing_offset, placement_offset, gap = 256, 256, 256
    layer_stride = layer_span + gap
    backing = torch.full(
        (backing_offset + placement_offset + 2 * layer_stride,),
        0x55,
        dtype=torch.uint8,
    )
    allocation = backing[backing_offset:].view(torch.int8)
    placement = KVCacheTensor(
        size=allocation.numel(),
        layers=["layer.0", "layer.1"],
        layer_stride=layer_stride,
        block_stride=PAGE_BYTES,
        offset=placement_offset,
    )
    views = create_kv_cache_views(
        allocation, compound_spec, blocks, KVCacheLayout.LBHNC, placement
    )
    assert compute_layout_strides(compound_spec, blocks, 2, KVCacheLayout.LBHNC) == (
        layer_span,
        PAGE_BYTES,
        PAGE_BYTES,
        PAGE_BYTES,
        1,
    )
    for layer, raw in enumerate(views):
        expected_offset = backing_offset + placement_offset + layer * layer_stride
        assert raw.dtype == torch.uint8
        assert raw.shape == (blocks, 1, 1, PAGE_BYTES)
        assert raw.storage_offset() == expected_offset
        assert raw.stride(0) == PAGE_BYTES
        bound = bind_compound_cache(raw, layout, tp_rank=0)
        bound.main.fill_(0x11 + layer)
        bound.index.view(torch.uint8).fill_(0x31 + layer)

    expected = torch.full_like(backing, 0x55)
    for layer in range(2):
        layer_start = backing_offset + placement_offset + layer * layer_stride
        for page in range(blocks):
            page_start = layer_start + page * PAGE_BYTES
            expected[page_start : page_start + 36_864] = 0x11 + layer
            expected[page_start + 36_864 : page_start + PAGE_BYTES] = 0x31 + layer
    assert torch.equal(backing, expected)


@pytest.mark.parametrize("dense_first", [False, True])
def test_dense_and_sparse_layers_keep_one_group_and_exact_byte_budget(
    compound_spec, dense_first
):
    dense = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    sparse_layers = [(f"sparse.{i}", compound_spec) for i in range(57)]
    dense_layers = [(f"dense.{i}", dense) for i in range(3)]
    specs = dict(
        dense_layers + sparse_layers if dense_first else sparse_layers + dense_layers
    )
    uniform = UniformTypeKVCacheSpecs.from_specs(specs)
    assert uniform is not None
    assert len(uniform.kv_cache_specs) == 60
    assert uniform.block_size == 128
    assert uniform.page_size_bytes == 2_764_800


def test_compound_merge_retains_the_full_contract(compound_spec):
    merged = MiniMaxM3CompoundSpec.merge([compound_spec, compound_spec])
    assert type(merged) is MiniMaxM3CompoundSpec
    assert merged == compound_spec
    assert merged.page_size_bytes == PAGE_BYTES


def test_plain_attention_merge_cannot_drop_the_index_region(main_spec, compound_spec):
    with pytest.raises(AssertionError, match="compound"):
        FullAttentionSpec.merge([compound_spec, compound_spec])
    with pytest.raises(AssertionError, match="compound"):
        FullAttentionSpec.merge([main_spec, compound_spec])
    with pytest.raises(AssertionError, match="main-only"):
        MiniMaxM3CompoundSpec.merge([compound_spec, main_spec])


@pytest.mark.parametrize("kernel_block_size", [32, 64, 256])
def test_compound_pages_cannot_be_split(compound_spec, kernel_block_size):
    with pytest.raises(ValueError, match="splitting"):
        compute_layer_kv_cache_shape_bytes(compound_spec, 3, kernel_block_size)


@pytest.mark.parametrize("cache_layout", list(KVCacheLayout)[1:])
def test_compound_pages_require_the_qualified_lbhnc_layout(compound_spec, cache_layout):
    with pytest.raises(ValueError, match="LBHNC"):
        compute_layout_strides(compound_spec, 3, 2, cache_layout)


@pytest.mark.parametrize(
    "changes",
    [
        {"page_size_padded": 45_312},
        {"index_rows_per_rank": 32},
        {"state_content_bytes": 144},
        {"num_head_slots": 2},
        {"kv_quant_mode": KVQuantMode.FP8_PER_TENSOR},
        {"block_size": 256},
    ],
)
def test_compound_spec_refuses_incompatible_packing(compound_spec, changes):
    with pytest.raises(ValueError, match="padding|packed NVFP4"):
        replace(compound_spec, **changes)


def test_compound_wrapper_cannot_be_applied_twice(compound_spec, layout):
    with pytest.raises(CompoundPageError, match="unwrapped"):
        MiniMaxM3CompoundSpec.from_main_spec(compound_spec, layout)
