# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compound KV page for MiniMax-M3 indexer context parallelism (ICP).

One physical page of ``P=128`` tokens holds this rank's packed NVFP4 main K/V
followed by its ``R=P//W`` scale-free FP8 index-key rows, under one
allocation, block-table entry, write slot and lifetime. Token ``t`` of a page
belongs to index rank ``t//R`` at local row ``t%R``; at P128 local score column
``c`` is global ranking block ``c``. Only TP2/P128/NVFP4 is integrated::

    [ main K/V [2*Hlocal, P, 72] | index keys [R, D] ]
    0                     main_span                  compound_span

``main`` and ``index`` are non-owning aliases that carry the compound page
stride; the registered raw cache stays the only owner the KV lifecycle sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim

# Recorded with GPU evidence; bump on any incompatible layout change.
COMPOUND_PAGE_ABI_VERSION = "refined-icp-v1.compound-page.2"

LOGICAL_BLOCK_TOKENS = 128
INDEX_HEAD_DIM = 128
MAIN_HEAD_DIM = 128
TOTAL_MAIN_KV_HEADS = 4


class CompoundPageError(RuntimeError):
    """A compound page or its binding violates the layout contract."""


@dataclass(frozen=True, slots=True)
class CompoundPageLayout:
    """Rank-invariant page layout; the rank lives in the binding."""

    abi_version: str
    physical_page_tokens: int
    tp_size: int
    main_num_kv_heads: int
    main_head_dim: int
    index_rows_per_rank: int
    index_head_dim: int

    @classmethod
    def build(
        cls,
        *,
        tp_size: int,
        physical_page_tokens: int = LOGICAL_BLOCK_TOKENS,
        ranking_block_tokens: int = LOGICAL_BLOCK_TOKENS,
        total_main_kv_heads: int = TOTAL_MAIN_KV_HEADS,
        main_head_dim: int = MAIN_HEAD_DIM,
        index_head_dim: int = INDEX_HEAD_DIM,
    ) -> CompoundPageLayout:
        if ranking_block_tokens != LOGICAL_BLOCK_TOKENS:
            raise CompoundPageError("ranking blocks must remain 128 tokens")
        if physical_page_tokens != LOGICAL_BLOCK_TOKENS:
            raise CompoundPageError(
                f"compound pages are P{LOGICAL_BLOCK_TOKENS} only; got "
                f"P{physical_page_tokens}. P256/R128 is not integrated."
            )
        if tp_size != 2:
            raise CompoundPageError(f"compound pages require TP2, got TP{tp_size}")
        if (
            total_main_kv_heads != TOTAL_MAIN_KV_HEADS
            or main_head_dim != MAIN_HEAD_DIM
            or index_head_dim != INDEX_HEAD_DIM
        ):
            raise CompoundPageError("compound pages require global KV4 and D128")
        return cls(
            abi_version=COMPOUND_PAGE_ABI_VERSION,
            physical_page_tokens=physical_page_tokens,
            tp_size=tp_size,
            main_num_kv_heads=total_main_kv_heads // tp_size,
            main_head_dim=main_head_dim,
            index_rows_per_rank=physical_page_tokens // tp_size,
            index_head_dim=index_head_dim,
        )

    @property
    def main_element_dim(self) -> int:
        """NVFP4 data plus inline FP8 block scales per (head slot, token)."""
        return nvfp4_kv_cache_full_dim(self.main_head_dim)

    @property
    def main_shape(self) -> tuple[int, int, int]:
        # Slot 2*h is head h's K; slot 2*h+1 is its V.
        return (
            2 * self.main_num_kv_heads,
            self.physical_page_tokens,
            self.main_element_dim,
        )

    @property
    def main_span_bytes(self) -> int:
        slots, tokens, dim = self.main_shape
        return slots * tokens * dim

    @property
    def index_shape(self) -> tuple[int, int]:
        return (self.index_rows_per_rank, self.index_head_dim)

    @property
    def index_span_bytes(self) -> int:
        return self.index_rows_per_rank * self.index_head_dim

    @property
    def main_offset_bytes(self) -> int:
        return 0

    @property
    def index_offset_bytes(self) -> int:
        return self.main_span_bytes

    @property
    def compound_span_bytes(self) -> int:
        return self.main_span_bytes + self.index_span_bytes

    def describe(self) -> str:
        return (
            f"{self.abi_version} W={self.tp_size} nvfp4 "
            f"main={self.main_span_bytes}B "
            f"index={self.index_span_bytes}B(R={self.index_rows_per_rank}) "
            f"page={self.compound_span_bytes}B"
        )


@dataclass(frozen=True, slots=True)
class BoundCompoundCache:
    """Worker-local aliases over one sparse layer's raw compound cache.

    Created once per cache lifetime and never rebound: readers key plans on
    tensor identity.
    """

    raw: torch.Tensor
    main: torch.Tensor
    index: torch.Tensor
    page_stride_bytes: int
    layout: CompoundPageLayout
    tp_rank: int
    storage_identity: tuple


def compound_cache_identity(raw: torch.Tensor, layout: CompoundPageLayout) -> tuple:
    """Identify a layer view within a possibly shared backing allocation."""
    return (
        raw.untyped_storage().data_ptr(),
        raw.storage_offset(),
        tuple(raw.shape),
        tuple(raw.stride()),
        raw.dtype,
        raw.device,
        layout,
    )


def bind_compound_cache(
    raw: torch.Tensor,
    layout: CompoundPageLayout,
    tp_rank: int,
) -> BoundCompoundCache:
    """Create the main/index aliases over a raw ``uint8[N, 1, 1, P]`` view.

    The storage may hold other layers, so the aliases use the view's actual
    offset and page stride.
    """
    if not isinstance(layout, CompoundPageLayout):
        raise CompoundPageError(
            f"layout must be CompoundPageLayout, got {type(layout)}"
        )
    if not 0 <= tp_rank < layout.tp_size:
        raise CompoundPageError(f"tp_rank {tp_rank} outside [0, {layout.tp_size})")
    if raw.dtype != torch.uint8:
        raise CompoundPageError(f"raw compound page must be uint8, got {raw.dtype}")
    if raw.ndim != 4 or raw.shape[1] != 1 or raw.shape[2] != 1:
        raise CompoundPageError(
            f"raw compound page must be [N,1,1,P], got {tuple(raw.shape)}"
        )
    if not raw.is_contiguous():
        raise CompoundPageError("raw compound page must be contiguous")
    num_pages, _, _, page_bytes = raw.shape
    page_stride = raw.stride(0)
    base_offset = raw.storage_offset()
    expected = layout.compound_span_bytes
    if page_bytes != expected or page_stride != expected:
        raise CompoundPageError(
            f"page extent/stride {page_bytes}/{page_stride}B does not match layout "
            f"{layout.describe()} (expected {expected}B)"
        )
    if num_pages < 1 or raw.stride(-1) != 1:
        raise CompoundPageError("compound cache needs nonempty, dense byte pages")
    if base_offset % 16 or raw.data_ptr() % 16 or page_stride % 16:
        raise CompoundPageError("compound page bases and strides must be 16B aligned")
    storage_end = base_offset + (num_pages - 1) * page_stride + page_bytes
    if storage_end > raw.untyped_storage().nbytes():
        raise CompoundPageError(
            f"compound view ends at {storage_end}B, beyond its backing storage"
        )

    main = torch.as_strided(
        raw,
        size=(num_pages, *layout.main_shape),
        stride=_page_strides(layout.main_shape, page_stride),
        storage_offset=base_offset + layout.main_offset_bytes,
    )
    index = torch.as_strided(
        raw.view(torch.float8_e4m3fn),
        size=(num_pages, *layout.index_shape),
        stride=_page_strides(layout.index_shape, page_stride),
        storage_offset=base_offset + layout.index_offset_bytes,
    )
    return BoundCompoundCache(
        raw=raw,
        main=main,
        index=index,
        page_stride_bytes=page_stride,
        layout=layout,
        tp_rank=tp_rank,
        storage_identity=compound_cache_identity(raw, layout),
    )


def _page_strides(inner_shape: tuple[int, ...], page_stride: int) -> tuple[int, ...]:
    """Row-major strides for ``[N, *inner_shape]`` with the compound page stride."""
    strides = [1]
    for dim in reversed(inner_shape[1:]):
        strides.append(strides[-1] * dim)
    strides.reverse()
    return (page_stride, *strides)
