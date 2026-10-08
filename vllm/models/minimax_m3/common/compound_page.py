# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compound KV page for MiniMax-M3 indexer context parallelism (indexer DCP).

One physical page of ``P`` tokens holds this rank's main K/V and its compact
index-key rows under one allocation, block-table entry, write slot and lifetime.
Ranking always uses logical ``B=128`` blocks. The integrated profile is
``P128/R64`` at TP2 with plain NVFP4 main K/V.
``P256/R128`` is not integrated in this path and is gated off at construction:
it needs MSA subpage readers and a selector route that emits the
non-contiguous ranking ids ``2*column + rank``.

With ``W`` equal to TP and indexer CP, each rank stores ``R=P//W`` index rows,
``Hlocal=4//W`` main KV heads, and head dimension 128. Physical-page token
``t`` belongs to index rank ``t//R`` at local row ``t%R``. The compact local
index view starts at row zero on every rank; rank never offsets its base.
At P128 every rank holds a fragment of every logical block, so its local score
column ``c`` maps to global logical block ``c`` -- the identity.

Page interior, main region first at the layer's shared-storage offset::

    [ main K/V .................. | index keys ....... ]
    0                    main_span   index_offset   compound_span

The main region reproduces the layout the main attention backend already
expects.  For NVFP4 that is the byte-budget carrier
``[2*Hlocal, P, nvfp4_kv_cache_full_dim(D)]`` whose physical content is per-head
``[K_data | K_scale | V_data | V_scale]`` slots, matching public vLLM;
for FP8 E4M3 it is
``[Hlocal, P, 2*D]``.  The index region is ``[R, D]`` of scale-free FP8 E4M3.

The registered cache stays the raw byte owner; ``main`` and ``index`` are
non-owning aliases into it.  Both carry the **compound** page stride, not a
stride recomputed from their own extent -- that difference is the whole point,
and recomputing it silently reads the neighbouring page.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch

from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim

# Frozen ABI marker.  Bump with any incompatible layout change; the value is
# recorded alongside GPU evidence so a result can be tied to a layout.
COMPOUND_PAGE_ABI_VERSION = "refined-icp-v1.compound-page.2"

LOGICAL_BLOCK_TOKENS = 128
INDEX_HEAD_DIM = 128
MAIN_HEAD_DIM = 128
TOTAL_MAIN_KV_HEADS = 4


class SparseMainFormat(str, enum.Enum):
    """Main K/V storage format of one sparse layer, resolved once at startup."""

    NVFP4 = "nvfp4"
    FP8_E4M3 = "fp8_e4m3"


class CompoundPageError(RuntimeError):
    """Raised when a compound page or its binding violates the layout contract.

    Deliberately not ``AssertionError``: these checks must survive ``python -O``
    and must not be confused with a test assertion.
    """


def icp_local_read_bounds(
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    physical_page_tokens: int,
    world_size: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive readable score prefixes and local forced columns on device.

    ``nvalid`` is the EXACT number of score columns the ABI2 decode scorer
    writes for each row, not an upper bound on it. With
    ``U = max(0, min(seq_lens, positions + 1))`` the scorer's per-row store
    guard admits column ``b`` iff ``P*b + R*rank < U``, which counts to
    ``U//P + (U%P > rank*R)`` -- the expression below, term for term. The
    scorer leaves every other cell of the score plane undefined, so this must
    stay equal to that guard; a bound that merely dominates it would let the
    selector rank uninitialised memory.

    Args:
        positions: int64 absolute token positions, on device.
        seq_lens: int32 per-token materialised history length, on device.
        physical_page_tokens: ``P``; 128 is the only integrated value.
        world_size: ``W``, the indexer context-parallel degree.
        rank: this rank's index in ``[0, W)``.

    Returns:
        ``(nvalid, forced_column)``, both int32 and on ``positions``' device.

    Raises:
        CompoundPageError: if ``physical_page_tokens`` is not 128.

    """
    if physical_page_tokens != LOGICAL_BLOCK_TOKENS:
        raise CompoundPageError(
            f"icp_local_read_bounds is P{LOGICAL_BLOCK_TOKENS} only; got "
            f"P{physical_page_tokens}. At P256 a local column would denote "
            "global ranking id `2*column + rank`, which the selector cannot "
            "emit -- it produces a contiguous prefix."
        )
    if world_size != 2 or rank not in (0, 1):
        raise CompoundPageError("ICP read bounds require TP2 and rank 0 or 1")
    rows = physical_page_tokens // world_size
    visible = torch.minimum(seq_lens, positions + 1).clamp_min(0)
    nvalid = visible // physical_page_tokens + (
        visible % physical_page_tokens > rank * rows
    )
    # P128 identity: local score column `c` IS global ranking block `c`, so
    # the forced column is just the position's own block when this rank holds
    # a readable fragment of it.
    column = positions // LOGICAL_BLOCK_TOKENS
    owns = (column >= 0) & (column < nvalid)
    forced = torch.where(owns, column, -1)
    return nvalid.to(torch.int32), forced.to(torch.int32)


@dataclass(frozen=True, slots=True)
class CompoundPageLayout:
    """Immutable, rank-invariant description of one sparse layer's page.

    Rank-invariant on purpose: every rank of a TP group has a byte-identical
    layout and differs only in *which* index rows and main heads it holds.  The
    worker rank lives in the binding (:class:`BoundCompoundCache`), never here,
    so this object can be compared, hashed and logged across ranks.
    """

    abi_version: str
    physical_page_tokens: int
    tp_size: int
    main_format: SparseMainFormat
    main_num_kv_heads: int
    main_head_dim: int
    index_rows_per_rank: int
    index_head_dim: int

    @classmethod
    def build(
        cls,
        *,
        tp_size: int,
        main_format: SparseMainFormat,
        physical_page_tokens: int = LOGICAL_BLOCK_TOKENS,
        ranking_block_tokens: int = LOGICAL_BLOCK_TOKENS,
        total_main_kv_heads: int = TOTAL_MAIN_KV_HEADS,
        main_head_dim: int = MAIN_HEAD_DIM,
        index_head_dim: int = INDEX_HEAD_DIM,
    ) -> CompoundPageLayout:
        if ranking_block_tokens != LOGICAL_BLOCK_TOKENS:
            raise CompoundPageError("ranking blocks must remain 128 tokens")
        # P128 is asserted by the layout itself, not only by its caller:
        # `nvidia/model.py` refuses a non-128 `--block-size` earlier, but this
        # constructor is public and a layout that exists at P256 is a P256
        # configuration.
        if physical_page_tokens != LOGICAL_BLOCK_TOKENS:
            raise CompoundPageError(
                f"compound pages are P{LOGICAL_BLOCK_TOKENS} only; got "
                f"P{physical_page_tokens}. P256/R128 is not integrated: it "
                "needs MSA subpage readers and a selector able to emit the "
                "non-contiguous ranking ids `2*column + rank`."
            )
        if tp_size != 2:
            raise CompoundPageError(f"compound pages require TP2, got TP{tp_size}")
        if SparseMainFormat(main_format) is not SparseMainFormat.NVFP4:
            raise CompoundPageError("compound pages require plain NVFP4 main K/V")
        if (
            total_main_kv_heads != TOTAL_MAIN_KV_HEADS
            or main_head_dim != MAIN_HEAD_DIM
            or index_head_dim != INDEX_HEAD_DIM
        ):
            raise CompoundPageError("compound pages require global KV4 and D128")
        if physical_page_tokens % tp_size:
            raise CompoundPageError(
                f"index rows do not partition: block tokens "
                f"{physical_page_tokens} % W {tp_size} != 0"
            )
        if total_main_kv_heads % tp_size:
            raise CompoundPageError(
                f"main KV heads do not shard without replication: "
                f"{total_main_kv_heads} % W {tp_size} != 0"
            )
        return cls(
            abi_version=COMPOUND_PAGE_ABI_VERSION,
            physical_page_tokens=physical_page_tokens,
            tp_size=tp_size,
            main_format=SparseMainFormat(main_format),
            main_num_kv_heads=total_main_kv_heads // tp_size,
            main_head_dim=main_head_dim,
            index_rows_per_rank=physical_page_tokens // tp_size,
            index_head_dim=index_head_dim,
        )

    # -- derived geometry -------------------------------------------------

    @property
    def ranking_block_tokens(self) -> int:
        return LOGICAL_BLOCK_TOKENS

    @property
    def main_element_dim(self) -> int:
        """Trailing bytes per (head, token) of the main region.

        NVFP4 packs fp4 data plus its inline fp8 block scales into
        ``nvfp4_kv_cache_full_dim(D)`` bytes per side; FP8 E4M3 is one byte per
        element and carries **no** scale payload for this model.
        """
        if self.main_format is SparseMainFormat.NVFP4:
            return nvfp4_kv_cache_full_dim(self.main_head_dim)
        return self.main_head_dim

    @property
    def main_shape(self) -> tuple[int, int, int]:
        """Per-page main shape, matching the backend's existing carrier."""
        if self.main_format is SparseMainFormat.NVFP4:
            # Slot 2*h is head h's K; slot 2*h+1 is its V.
            return (
                2 * self.main_num_kv_heads,
                self.physical_page_tokens,
                self.main_element_dim,
            )
        # K and V packed into the content dim.
        return (
            self.main_num_kv_heads,
            self.physical_page_tokens,
            2 * self.main_element_dim,
        )

    @property
    def main_span_bytes(self) -> int:
        span = 1
        for dim in self.main_shape:
            span *= dim
        return span

    @property
    def index_shape(self) -> tuple[int, int]:
        return (self.index_rows_per_rank, self.index_head_dim)

    @property
    def index_span_bytes(self) -> int:
        # Scale-free FP8 E4M3: one byte per element, no scale payload.
        return self.index_rows_per_rank * self.index_head_dim

    @property
    def main_offset_bytes(self) -> int:
        return 0

    @property
    def index_offset_bytes(self) -> int:
        # The index region starts where main ends and ``compound_span_bytes``
        # is their sum, so the regions are disjoint and page-covering by
        # construction. The load-bearing guards are the page-stride and
        # allocation-size checks in ``bind_compound_cache``.
        return self.main_span_bytes

    @property
    def compound_span_bytes(self) -> int:
        return self.main_span_bytes + self.index_span_bytes

    def describe(self) -> str:
        return (
            f"{self.abi_version} W={self.tp_size} "
            f"{self.main_format.value} main={self.main_span_bytes}B "
            f"index={self.index_span_bytes}B(R={self.index_rows_per_rank}) "
            f"page={self.compound_span_bytes}B"
        )


@dataclass(frozen=True, slots=True)
class BoundCompoundCache:
    """Worker-local binding of one sparse layer's compound cache.

    ``raw`` is the registered owner and the only tensor the KV lifecycle
    (zeroing, COW, prefix reuse, CPU offload) ever sees.  ``main`` and ``index``
    are non-owning aliases created once at startup and never rebound, because
    the readers cache plans keyed on tensor identity.
    """

    raw: torch.Tensor
    main: torch.Tensor
    index: torch.Tensor
    page_stride_bytes: int
    layout: CompoundPageLayout
    tp_rank: int
    storage_identity: tuple

    @property
    def num_pages(self) -> int:
        return self.raw.shape[0]

    def owns_index_row(self, token_offset: int) -> bool:
        """Whether this rank stores the index key for ``token_offset``.

        Clause C1: offset ``t`` belongs to rank ``t // R`` at row ``t % R``.
        Non-ownership suppresses only the index *store*; every rank still writes
        its main K/V and still produces all four index queries.
        """
        return token_offset // self.layout.index_rows_per_rank == self.tp_rank


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
    """Create the stable main/index aliases over a raw compound allocation.

    ``raw`` is the complete ``uint8[N, 1, 1, P]`` layer view created by the
    shared allocator. Its storage may include other layers, so aliases use
    its actual offset and page stride.
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

    main_shape = (num_pages, *layout.main_shape)
    main_stride = _row_major_strides(layout.main_shape, page_stride)
    main = torch.as_strided(
        raw,
        size=main_shape,
        stride=main_stride,
        storage_offset=base_offset + layout.main_offset_bytes,
    )

    index_shape = (num_pages, *layout.index_shape)
    index_stride = _row_major_strides(layout.index_shape, page_stride)
    index = torch.as_strided(
        raw.view(torch.float8_e4m3fn),
        size=index_shape,
        stride=index_stride,
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


def _row_major_strides(
    inner_shape: tuple[int, ...], page_stride: int
) -> tuple[int, ...]:
    """Strides for ``[N, *inner_shape]`` where the page stride is the compound
    page, not the region's own extent."""
    strides = [1]
    for dim in reversed(inner_shape[1:]):
        strides.append(strides[-1] * dim)
    strides.reverse()
    return (page_stride, *strides)
