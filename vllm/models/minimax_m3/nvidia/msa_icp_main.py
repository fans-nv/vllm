# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ICP main attention using MSA's shared kernels and public head-slot cache."""

from dataclasses import dataclass
from typing import Any, cast
from weakref import WeakValueDictionary

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerMetadata
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
)
from vllm.models.minimax_m3.nvidia.msa_icp import require_msa_icp
from vllm.v1.attention.backend import AttentionLayer


@dataclass
class ICPMainDecodeMetadata(MiniMaxM3SparseDecodeMetadata):
    plan: Any
    kv_indices: torch.Tensor
    kv_indptr: torch.Tensor


@dataclass
class MiniMaxM3SparseICPMetadata(MiniMaxM3SparseMetadata):
    indexer: MiniMaxM3IndexerMetadata | None = None


class ICPMainDecodePlans:
    """Shape-only schedules shared by all sparse layers on one device."""

    def __init__(self, num_heads: int, num_kv_heads: int, device: torch.device):
        require_msa_icp()
        from fmha_sm100.decode_q8kv4 import interface, jit, plan_decode

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.device = device
        self._plan_decode = plan_decode
        self._plans: dict[tuple[int, int], Any] = {}
        # The planner and both direct/stream-K kernel variants must be loaded
        # before graph capture. The shared MSA JIT owns locking and cache keys.
        interface._get_cpp()
        jit.get_plan_fn(device=device)
        for split_kv in (False, True):
            jit.get_fmha_fwd_variant(
                topk=16,
                split_kv=split_kv,
                gqa_ratio=num_heads // num_kv_heads,
                device=device,
                block_scale_shift=3,
            )

    def get(self, batch: int, query_len: int):
        key = (batch, query_len)
        plan = self._plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"ICP main decode plan {key} was not prebuilt")
            plan = self._plan_decode(
                batch_size=batch,
                q_len_per_req=query_len,
                topk=16,
                device=self.device,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                num_kv_splits=1,
                split_mode="streamk",
                block_scale_shift=3,
            )
            self._plans[key] = plan
        return plan

    def prebuild(self, config: VllmConfig) -> None:
        max_query_len = (
            1
            if config.speculative_config is None
            else 1 + (config.speculative_config.num_speculative_tokens or 0)
        )
        for query_len in range(1, max_query_len + 1):
            for tokens in config.compilation_config.cudagraph_capture_sizes or ():
                batch = (tokens + query_len - 1) // query_len
                if 0 < batch <= config.scheduler_config.max_num_seqs:
                    self.get(batch, query_len)


def get_icp_main_decode_plans(
    num_heads: int, num_kv_heads: int, device: torch.device
) -> ICPMainDecodePlans:
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.accelerator.current_device_index())
    key = (num_heads, num_kv_heads, device)
    plans = _DECODE_PLANS.get(key)
    if plans is None:
        plans = ICPMainDecodePlans(num_heads, num_kv_heads, device)
        _DECODE_PLANS[key] = plans
    return plans


# Layers and metadata builders own schedules; this registry only shares them.
# A process-global strong cache would retain CUDA buffers after runner shutdown.
_DECODE_PLANS: WeakValueDictionary[
    tuple[int, int, torch.device], ICPMainDecodePlans
] = WeakValueDictionary()


@dataclass(frozen=True)
class ICPMainCacheViews:
    k: torch.Tensor
    v: torch.Tensor
    k_sf: torch.Tensor
    v_sf: torch.Tensor
    k_sf_fp8: torch.Tensor
    v_sf_fp8: torch.Tensor


def icp_main_cache_views(kv_cache: torch.Tensor) -> ICPMainCacheViews:
    """Bind MSA's existing per-head-slot views, preserving the parent stride."""
    from fmha_sm100.nvfp4_kv import nvfp4_head_slot_views

    k, k_sf, v, v_sf = nvfp4_head_slot_views(kv_cache[:, 0::2], kv_cache[:, 1::2])
    return ICPMainCacheViews(
        k,
        v,
        k_sf,
        v_sf,
        k_sf.view(torch.float8_e4m3fn),
        v_sf.view(torch.float8_e4m3fn),
    )


class MiniMaxM3SparseICPImpl(MiniMaxM3SparseImpl):
    """Decode uses Q8KV4; prefill uses shared NVFP4 CSR forward/combine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        require_msa_icp()
        from fmha_sm100.decode_q8kv4 import run_decode
        from fmha_sm100.icp.attention.nvfp4_prefill import (
            build_k2q_csr,
            sparse_atten_nvfp4_kv_func,
        )

        self._run_decode = run_decode
        self._build_k2q_csr = build_k2q_csr
        self._prefill_attend = sparse_atten_nvfp4_kv_func
        self._views: ICPMainCacheViews | None = None
        device = torch.device("cuda", torch.accelerator.current_device_index())
        decode_plans = get_icp_main_decode_plans(
            self.num_heads, self.num_kv_heads, device
        )
        decode_plans.prebuild(get_current_vllm_config())
        self.decode_plans: ICPMainDecodePlans | None = decode_plans

    def should_use_msa_decode(self, layer_name: str) -> bool:
        return True

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self._views = icp_main_cache_views(kv_cache)

    def release_kv_cache(self) -> None:
        self._views = None

    def shutdown_model_resources(self) -> None:
        """Release this layer's schedules after graph and stream teardown."""
        self.decode_plans = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        query_fp8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return output
        main = metadata[layer.layer_name]  # type: ignore[attr-defined]
        if not isinstance(main, MiniMaxM3SparseICPMetadata):
            raise RuntimeError("ICP main attention requires compound metadata")
        views = self._views
        if views is None:
            raise RuntimeError("ICP main cache views must be bound before forward")
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        rows, nd = main.num_actual_tokens, main.num_decode_tokens
        out = output[:rows].view(-1, self.num_heads, self.head_size)
        # The ICP producer writes both outputs. Keep its BF16 query for
        # prefill without an FP8 round trip or another device launch.
        q = query[:rows].view(-1, self.num_heads, self.head_size)
        k_global = layer._k_scale
        v_global = layer._v_scale
        if main.num_decodes:
            assert query_fp8 is not None
            decode = cast(ICPMainDecodeMetadata, main.decode)
            self._run_decode(
                decode.plan,
                query_fp8[:nd].view(-1, self.num_heads, self.head_size),
                (views.k, views.v),
                kv_cache_sf=(views.k_sf_fp8, views.v_sf_fp8),
                seq_lens=decode.seq_lens,
                kv_indices=decode.kv_indices,
                kv_indptr=decode.kv_indptr,
                topk_indices=topk[:nd],
                sm_scale=self.scale,
                kv_global_scale=(k_global, v_global),
                out=out[:nd],
            )
        if main.num_prefills:
            prefill = main.prefill
            assert prefill is not None
            row_ptr, q_indices, schedule = self._build_k2q_csr(
                topk[nd:rows].transpose(0, 1),
                prefill.cu_seqlens_q,
                prefill.cu_seqlens_k,
                self.block_size,
                total_k=0,
                max_seqlen_k=prefill.max_seq_len,
                max_seqlen_q=prefill.max_query_len,
                total_rows=prefill.total_kv_blocks,
                qhead_per_kv=self.num_heads // self.num_kv_heads,
                return_schedule=True,
            )
            self._prefill_attend(
                q[nd:],
                views.k,
                views.v,
                views.k_sf,
                views.v_sf,
                k_global,
                v_global,
                row_ptr,
                q_indices,
                self.topk_blocks,
                cu_seqlens_q=prefill.cu_seqlens_q,
                cu_seqlens_k=prefill.cu_seqlens_k,
                max_seqlen_q=prefill.max_query_len,
                max_seqlen_k=prefill.max_seq_len,
                blk_kv=self.block_size,
                causal=True,
                softmax_scale=self.scale,
                page_table=prefill.block_table,
                seqused_k=prefill.seq_lens,
                schedule=schedule,
                out=out[nd:],
            )
        return output
