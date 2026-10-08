# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute-only MiniMax M3 ICP indexer.

The parent attention layer owns the compound cache and passes its index-key
fragment and metadata to the indexer. Shared metadata and builder setup come
from the existing indexer; this module never creates a side cache.
"""

from typing import Any, ClassVar

import torch
from torch import nn

from vllm.config.attention import IndexerKVDType
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend as _IndexerBackend,
)
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerMetadata as MiniMaxM3IndexerMetadata,
)
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerMetadataBuilder as MiniMaxM3IndexerMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend

logger = init_logger(__name__)


def indexer_kv_torch_dtype(indexer_kv_dtype: IndexerKVDType) -> torch.dtype:
    """Storage dtype of one index key, resolved once at construction."""
    if indexer_kv_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    if indexer_kv_dtype == "bf16":
        return torch.bfloat16
    raise NotImplementedError(
        f"indexer_kv_dtype={indexer_kv_dtype!r} is not a MiniMax M3 index-key "
        "storage format (only 'bf16' or 'fp8'/'fp8_e4m3')."
    )


class MiniMaxM3IndexerBackend(_IndexerBackend):
    """Index-key backend (key-only).

    On the compound-page path only ``get_builder_cls`` is consulted: the parent
    sparse layer drives that builder as a sub-builder, and the cache shape and
    stride order below describe nothing this backend owns.
    """

    @staticmethod
    def get_impl_cls() -> type[Any]:
        # Compute-only nn.Module, selected directly rather than instantiated by
        # the generic Attention layer; the backend only supplies sub-metadata.
        return MiniMaxM3IndexerImpl

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerMetadataBuilder"]:
        from vllm.models.minimax_m3.nvidia.indexer_icp import (
            MiniMaxM3IndexerMSAMetadataBuilder,
        )

        return MiniMaxM3IndexerMSAMetadataBuilder

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        # num_blocks is the outermost physical dim (get_kv_cache_shape below is
        # (num_blocks, block_size, head_size) with stride order (0, 1, 2)), so
        # the side cache tolerates a non-contiguous block dim and a padded page
        # can be read through a strided view. The base implementation would
        # return False here only because get_kv_cache_stride_order raises
        # NotImplementedError for the layered variant (M3 has no cross-layer
        # KV blocks), which is not evidence against block-stride indexing.
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # M3 does not use cross-layer (per-layer-stacked) KV blocks.
            raise NotImplementedError
        return (0, 1, 2)


class MiniMaxM3IndexerImpl(nn.Module):
    """Abstract base for the indexer kernel impls. **Compute only.**

    The impl owns no cache, no spec and no builder. It reports which backend
    supplies its metadata shape via ``indexer_backend_cls``, and the parent
    attention layer's metadata builder drives that backend's builder as a
    sub-builder; ``forward`` is handed the index-key view and the metadata it
    produced. The ICP implementation writes the shared top-k buffer.
    """

    # Selects the metadata builder the parent layer's builder must drive.
    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerBackend

    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 0,
        score_type: str = "max",
        indexer_kv_dtype: IndexerKVDType = "bf16",
        topk_indices_buffer: torch.Tensor | None = None,
        icp_c: int = 1,
        icp_rank: int = 0,
    ) -> None:
        super().__init__()
        if icp_c != 2 or icp_rank not in (0, 1):
            raise ValueError("MiniMax M3 ICP requires two indexer ranks")
        # Fragment ownership, from the parent layer and never re-derived: the
        # impl's plans, page lists and head routing are all keyed on it, so it
        # has to be the same (W, rank) the layer stores its index rows with.
        self.icp_c = icp_c
        self.icp_rank = icp_rank
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.score_type = score_type
        self.num_index_heads = num_index_heads
        self.index_head_dim = index_head_dim
        self.indexer_kv_dtype = indexer_kv_dtype
        # Resolved once here, never re-read from a mutable config in forward.
        self.index_kv_torch_dtype = indexer_kv_torch_dtype(indexer_kv_dtype)
        self.prefix = prefix
        # Shared, stable-address top-k output buffer (set by the model for the
        # cudagraph-safe MSA impl); None -> impl allocates fresh (eager).
        self.topk_indices_buffer = topk_indices_buffer

    def forward(
        self,
        index_query: torch.Tensor,
        *,
        index_kv: torch.Tensor,
        index_md: "MiniMaxM3IndexerMetadata | None",
        icp_c: int = 1,
        icp_rank: int = 0,
        layer_idx: int = 0,
        candidate_exchange: object | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return ``(decode_topk, prefill_topk)``; implemented per kernel impl.

        Args:
            index_query: The layer's index queries, ``[num_tokens, H*D]``. At
                ``icp_c > 1`` ``H`` is the whole group's index-head count: every
                rank scores all of them and routes head ``h`` to its owner.
            index_kv: THIS RANK'S index-key fragment, ``[num_pages, R, D]``
                with ``R = 128 // icp_c``. Not a gathered block: scoring is
                sharded too, so the block scores produced here are partial
                maxima that the candidate merge later resolves.
            index_md: Indexer sub-metadata from the parent layer's builder, or
                ``None`` on the memory-profiling run (caches unbound).
            icp_c: Indexer context-parallel degree ``W``.
            icp_rank: This rank within the indexer group; with ``icp_c`` it
                reconstructs each fragment row's global position,
                ``128*b + icp_rank*R + j``.
            layer_idx: This sparse layer's index, forwarded to the exchange
                (which uses it to pick a transport slot).
            candidate_exchange: The parent layer's bound candidate exchange,
                called exactly once per invocation at ``W > 1``. Opaque here:
                this module implements no transport and no merge.

        """
        raise NotImplementedError


def select_indexer_impl_cls(
    *,
    topk_blocks: int,
    indexer_kv_dtype: IndexerKVDType = "bf16",
) -> type[MiniMaxM3IndexerImpl]:
    """Select the scorer for the admitted ICP path; no generic fallback."""
    if indexer_kv_dtype in ("mxfp4", "nvfp4"):
        raise NotImplementedError(
            f"indexer_kv_dtype={indexer_kv_dtype!r} needs the (not-yet-added) "
            "CuteDSL indexer impl."
        )
    is_sm100 = (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    )
    use_msa = (
        is_sm100
        and topk_blocks == 16
        and indexer_kv_dtype in ("bf16", "fp8", "fp8_e4m3")
    )
    if use_msa:
        # Lazy import so AMD / non-SM100 never import fmha_sm100.
        from vllm.models.minimax_m3.nvidia.indexer_icp import (
            MiniMaxM3IndexerMSAImpl,
        )

        logger.info_once(
            "MiniMax M3 indexer: selected MSA (fmha_sm100 score + top-k) "
            "[topk_blocks=%d, indexer_kv_dtype=%s]",
            topk_blocks,
            indexer_kv_dtype,
        )
        return MiniMaxM3IndexerMSAImpl
    raise NotImplementedError(
        "MiniMax M3 ICP requires the SM100-family MSA indexer with topk=16 "
        f"and FP8 index keys; got topk={topk_blocks}, "
        f"indexer_kv_dtype={indexer_kv_dtype!r}"
    )


def select_indexer_builder_cls(
    *,
    topk_blocks: int,
    indexer_kv_dtype: IndexerKVDType = "bf16",
) -> type[MiniMaxM3IndexerMetadataBuilder]:
    """The metadata builder matching the impl ``select_indexer_impl_cls`` picks.

    Driven from the *parent* layer's builder as a sub-builder, never registered
    on its own: with the index keys inside the parent's page there is no second
    KV-cache group to hang an independent builder off. Resolving it through the
    same selector, from the same two inputs, is what keeps the metadata the
    parent hands over in step with the impl that consumes it.
    """
    impl_cls = select_indexer_impl_cls(
        topk_blocks=topk_blocks, indexer_kv_dtype=indexer_kv_dtype
    )
    return impl_cls.indexer_backend_cls.get_builder_cls()


class MiniMaxM3Indexer(nn.Module):
    """Indexer module held by the attention layer (like ``DeepseekV4Indexer``).

    Picks the kernel impl in ``__init__`` (``select_indexer_impl_cls``) and
    delegates ``forward``. Holds no cache: the parent layer supplies the
    index-key view and the sub-metadata on every call.
    """

    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 0,
        score_type: str = "max",
        indexer_kv_dtype: IndexerKVDType = "bf16",
        topk_indices_buffer: torch.Tensor | None = None,
        icp_c: int = 1,
        icp_rank: int = 0,
    ) -> None:
        super().__init__()
        impl_cls = select_indexer_impl_cls(
            topk_blocks=topk_blocks,
            indexer_kv_dtype=indexer_kv_dtype,
        )
        self.impl = impl_cls(
            num_kv_heads=num_kv_heads,
            scale=scale,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
            num_index_heads=num_index_heads,
            index_head_dim=index_head_dim,
            prefix=prefix,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            score_type=score_type,
            indexer_kv_dtype=indexer_kv_dtype,
            topk_indices_buffer=topk_indices_buffer,
            icp_c=icp_c,
            icp_rank=icp_rank,
        )

    @property
    def num_index_heads(self) -> int:
        return self.impl.num_index_heads

    @property
    def index_kv_torch_dtype(self) -> torch.dtype:
        """Storage dtype of an index key; also the dtype of ``index_q``."""
        return self.impl.index_kv_torch_dtype

    def forward(
        self,
        index_query: torch.Tensor,
        *,
        index_kv: torch.Tensor,
        index_md: MiniMaxM3IndexerMetadata | None,
        icp_c: int = 1,
        icp_rank: int = 0,
        layer_idx: int = 0,
        candidate_exchange: object | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.impl(
            index_query,
            index_kv=index_kv,
            index_md=index_md,
            icp_c=icp_c,
            icp_rank=icp_rank,
            layer_idx=layer_idx,
            candidate_exchange=candidate_exchange,
        )
