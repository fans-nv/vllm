# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 TP2 indexer metadata over rank-local index-key fragments (ICP).

MSA's ``IcpIndexerRuntime`` owns the scorer/selector capacity, plans each step
and launches each layer; this builder only adapts vLLM's attention metadata.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
)
from vllm.models.minimax_m3.nvidia.msa_icp import require_icp_writer, require_msa_icp
from vllm.models.minimax_m3.nvidia.ops.icp_metadata import ICPLiveMetadata
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@dataclass
class MiniMaxM3IndexerMSAMetadata(MiniMaxM3IndexerMetadata):
    """One step's ICP plan plus the views the first sparse KV writer fills."""

    live_metadata: ICPLiveMetadata | None = None
    icp: Any = None  # fmha_sm100.icp.indexer_runtime.IcpStep
    runtime: Any = None  # the builder's IcpIndexerRuntime


class MiniMaxM3IndexerMSAMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """Plan retained ICP decode profiles and per-step eager prefill windows."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # Prefill plans and exchange extents are per-step eager choices.
    cudagraph_decode_phase_only: ClassVar[bool] = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if get_tensor_model_parallel_world_size() != 2:
            raise ValueError("MiniMax M3 ICP requires TP2")
        require_msa_icp()
        # The writer is vLLM's; MSA's prewarm manifest covers only its kernels.
        require_icp_writer()
        from fmha_sm100.icp.indexer_runtime import (
            IcpIndexerConfig,
            IcpIndexerRuntime,
        )

        model_config = vllm_config.model_config
        sparse_cfg = model_config.hf_text_config.sparse_attention_config
        self.runtime = IcpIndexerRuntime(
            IcpIndexerConfig(
                world_size=2,
                rank=get_tensor_model_parallel_rank(),
                num_index_heads=self.num_index_heads,
                index_head_dim=int(sparse_cfg["sparse_index_dim"]),
                max_model_len=model_config.max_model_len,
                max_num_batched_tokens=(
                    vllm_config.scheduler_config.max_num_batched_tokens
                ),
                max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
                max_decode_query_len=self.max_decode_query_len,
                page_tokens=kv_cache_spec.block_size,
                cudagraph_capture_sizes=tuple(
                    vllm_config.compilation_config.cudagraph_capture_sizes or ()
                ),
            ),
            device=device,
            logger=logger,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerMSAMetadata:
        cm = common_attn_metadata
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                cm, decode_threshold=self.reorder_batch_threshold, require_uniform=True
            )
        )
        assert num_decodes + num_prefills == cm.num_reqs
        assert num_decode_tokens + num_prefill_tokens == cm.num_actual_tokens
        assert cm.positions is not None
        # Host planning only: no device read and no allocation.
        step = self.runtime.plan_step(
            block_table=cm.block_table_tensor,
            query_start_loc=cm.query_start_loc,
            query_start_loc_cpu=cm.query_start_loc_cpu,
            seq_lens=cm.seq_lens,
            seq_lens_cpu_upper_bound=cm.seq_lens_cpu_upper_bound,
            is_prefilling=cm.is_prefilling,
            num_reqs=cm.num_reqs,
            num_tokens=cm.num_actual_tokens,
        )
        live = step.live
        return MiniMaxM3IndexerMSAMetadata(
            seq_lens=cm.seq_lens,
            max_seq_len=cm.max_seq_len,
            slot_mapping=cm.slot_mapping,
            num_actual_tokens=cm.num_actual_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            live_metadata=ICPLiveMetadata(
                query_start_loc=cm.query_start_loc,
                seq_lens=cm.seq_lens,
                num_reqs=cm.num_reqs,
                positions=live.positions,
                active=live.active,
                local_nvalid=live.local_nvalid,
                local_forced=live.local_forced,
                global_nvalid=live.global_nvalid,
                forced=live.forced,
                n_ordinary=live.n_ordinary,
                candidates=live.candidates,
                qo_offsets=live.qo_offsets,
                chunk_width=live.chunk_width,
                device_plan=live.device_plan,
            ),
            icp=step,
            runtime=self.runtime,
        )
