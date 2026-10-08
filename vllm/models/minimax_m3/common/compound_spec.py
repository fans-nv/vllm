# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-specific compound-page specification for MiniMax-M3 ICP."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Self

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

if TYPE_CHECKING:
    from vllm.models.minimax_m3.common.compound_page import CompoundPageLayout


@dataclass(frozen=True, kw_only=True)
class MiniMaxM3CompoundSpec(FullAttentionSpec):
    """One full-attention lifetime for packed main KV and rank-local index keys."""

    index_rows_per_rank: int
    index_head_dim: int

    def __post_init__(self):
        super().__post_init__()
        if self.page_size_padded is not None:
            raise ValueError("MiniMax-M3 compound pages do not support page padding")
        if (
            self.block_size != 128
            or self.num_kv_heads != 2
            or self.head_size != 128
            or self.head_size_v != 128
            or self.dtype != torch.uint8
            or self.kv_quant_mode != KVQuantMode.NVFP4
            or self.num_head_slots != 4
            or self.state_content_bytes != 72
            or self.tokens_per_state != 1
            or self.index_rows_per_rank != 64
            or self.index_head_dim != 128
        ):
            raise ValueError("MiniMax-M3 compound pages require TP2/P128 packed NVFP4")

    @property
    def uses_raw_page_view(self) -> bool:
        return True

    @property
    def index_region_bytes(self) -> int:
        return self.index_rows_per_rank * self.index_head_dim

    @property
    def unpadded_page_size_bytes(self) -> int:
        return super().unpadded_page_size_bytes + self.index_region_bytes

    @classmethod
    def from_main_spec(
        cls, main_spec: FullAttentionSpec, layout: CompoundPageLayout
    ) -> Self:
        """Append index bytes after the attention backend publishes main packing."""
        from vllm.models.minimax_m3.common.compound_page import CompoundPageError

        if type(main_spec) is not FullAttentionSpec:
            raise CompoundPageError(
                "Compound pages need one unwrapped main attention spec"
            )
        if main_spec.page_size_bytes != layout.main_span_bytes:
            raise CompoundPageError(
                f"main spec budgets {main_spec.page_size_bytes}B, but the layout "
                f"needs {layout.main_span_bytes}B; customize the NVFP4 spec first"
            )
        spec = cls(
            **{
                field.name: getattr(main_spec, field.name)
                for field in fields(main_spec)
            },
            index_rows_per_rank=layout.index_rows_per_rank,
            index_head_dim=layout.index_head_dim,
        )
        if (
            spec.block_size != layout.physical_page_tokens
            or spec.num_kv_heads != layout.main_num_kv_heads
            or spec.head_size != layout.main_head_dim
            or spec.page_size_bytes != layout.compound_span_bytes
        ):
            raise CompoundPageError(f"compound spec does not match {layout.describe()}")
        return spec

    @classmethod
    def from_compound_layout(cls, layout: CompoundPageLayout, **kwargs) -> Self:
        """Compatibility factory with the current backend's packed main fields."""
        from vllm.models.minimax_m3.common.compound_page import (
            CompoundPageError,
            SparseMainFormat,
        )

        if layout.main_format is not SparseMainFormat.NVFP4:
            raise CompoundPageError("MiniMax-M3 compound pages require plain NVFP4")
        main_spec = FullAttentionSpec(
            block_size=layout.physical_page_tokens,
            num_kv_heads=layout.main_num_kv_heads,
            head_size=layout.main_head_dim,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.NVFP4,
            num_head_slots=2 * layout.main_num_kv_heads,
            state_content_bytes=layout.main_element_dim,
            **kwargs,
        )
        return cls.from_main_spec(main_spec, layout)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, cls) for spec in specs), (
            "Compound pages cannot merge with main-only attention specs."
        )
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All compound pages in a group must have the same geometry."
        )
        return copy.deepcopy(specs[0])
