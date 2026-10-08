# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in ICP adapter over the current NVFP4 Q8KV4 main-attention path.

The parent sparse layer remains the ICP-OFF implementation.
"""

from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MinimaxM3QKVParallelLinearWithIndexer,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.minimax_m3.common.compound_page import (
    BoundCompoundCache,
    CompoundPageError,
    CompoundPageLayout,
    SparseMainFormat,
    bind_compound_cache,
    compound_cache_identity,
)
from vllm.models.minimax_m3.common.compound_spec import MiniMaxM3CompoundSpec
from vllm.models.minimax_m3.common.indexer_icp import MiniMaxM3Indexer
from vllm.models.minimax_m3.common.ops.sparse_attn import SPARSE_BLOCK_SIZE
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseMetadataBuilder,
    MiniMaxM3SparsePrefillMetadata,
)
from vllm.models.minimax_m3.nvidia.indexer_icp import (
    ICP_PRODUCER_ENABLE_PDL,
    MiniMaxM3IndexerMSAMetadataBuilder,
    _assert_icp_prewarm,
)
from vllm.models.minimax_m3.nvidia.model import (
    MiniMAXGemmaRMSNorm,
    _sparse_attention_layer_ids,
)
from vllm.models.minimax_m3.nvidia.msa_icp import (
    ICP_DEVICE_PLAN_ABI,
    require_icp_writer,
    require_msa_icp,
)
from vllm.models.minimax_m3.nvidia.msa_icp_main import (
    ICPMainDecodeMetadata,
    MiniMaxM3SparseICPImpl,
    MiniMaxM3SparseICPMetadata,
    get_icp_main_decode_plans,
)
from vllm.models.minimax_m3.nvidia.ops.icp_dispatch import (
    admitted_exchange_extents,
    create_candidate_exchange,
)
from vllm.models.minimax_m3.nvidia.ops.icp_metadata import launch_icp_kv_write
from vllm.models.minimax_m3.nvidia.sparse_attention_msa import (
    MiniMaxM3SparseMSANvfp4Backend,
)
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionLayer, CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

_FRAGMENT_PRODUCER_WARMED: set[tuple[int, ...]] = set()


def validate_icp_config(vllm_config: VllmConfig, config: PreTrainedConfig) -> None:
    parallel = vllm_config.parallel_config
    cache = vllm_config.cache_config
    attention = vllm_config.attention_config
    sparse = config.sparse_attention_config
    expected = {
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
    }
    for name, value in expected.items():
        if getattr(parallel, name) != value:
            raise ValueError(f"msa_icp requires {name}={value}")
    if parallel.enable_expert_parallel:
        raise ValueError("msa_icp does not support expert parallelism")
    if cache.block_size != 128 or cache.cache_dtype != "nvfp4":
        raise ValueError("msa_icp requires P128 and plain nvfp4 main KV")
    if attention.minimax_m3_msa_decode_backend != "cutlass":
        raise ValueError("msa_icp requires minimax_m3_msa_decode_backend=cutlass")
    if attention.indexer_kv_dtype != "fp8":
        raise ValueError("msa_icp requires indexer_kv_dtype=fp8")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("msa_icp requires Model Runner V2")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("msa_icp requires BF16 model activations")
    if torch.cuda.get_device_capability() != (10, 7):
        raise ValueError("msa_icp requires Rubin SM107")
    if (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        sparse["sparse_num_index_heads"],
        sparse["sparse_index_dim"],
        sparse["sparse_topk_blocks"],
        sparse["sparse_block_size"],
    ) != (64, 4, 128, 4, 128, 16, 128):
        raise ValueError("msa_icp requires Q64/KV4/Hindex4/D128/TopK16/B128")
    if (sparse.get("sparse_init_block", 0), sparse.get("sparse_local_block", 0)) != (
        0,
        1,
    ):
        raise ValueError("msa_icp requires zero initial and one current forced block")
    if sparse.get("sparse_score_type", "max") != "max":
        raise ValueError("msa_icp requires maximum block scores")
    if vllm_config.model_config.max_model_len > 1048576:
        raise ValueError("msa_icp supports at most 8192 P128 score columns")
    if vllm_config.model_config.enable_sleep_mode:
        raise ValueError("msa_icp does not support sleep mode")
    if (
        vllm_config.offload_config.uva.cpu_offload_gb > 0
        or vllm_config.offload_config.prefetch.offload_group_size > 0
    ):
        raise ValueError("msa_icp does not support model-weight CPU offload")
    transfer = vllm_config.kv_transfer_config
    if transfer is not None:
        # Compound pages must travel through the qualified whole-page CPU path.
        # Check the direct connector, not has_connector(), which includes children
        # of MultiConnector and cannot establish the child's buffer lifecycle.
        if (
            transfer.kv_connector != "SimpleCPUOffloadConnector"
            or transfer.kv_connector_module_path is not None
            or transfer.kv_role != "kv_both"
            or transfer.kv_connector_extra_config.get("kv_offload_backend", "cpu")
            != "cpu"
        ):
            raise ValueError(
                "msa_icp only admits the built-in SimpleCPUOffloadConnector "
                "with kv_role=kv_both and kv_offload_backend=cpu"
            )
        if not cache.enable_prefix_caching:
            raise ValueError("msa_icp CPU offload requires prefix caching")
    spec = vllm_config.speculative_config
    if spec is not None:
        if spec.method not in ("eagle", "eagle3") or spec.kv_cache_dtype != "fp8":
            raise ValueError("msa_icp requires an FP8 Eagle draft cache")
        if (spec.num_speculative_tokens or 0) > 3:
            raise ValueError("msa_icp supports Q1-Q4 target verification")


def uniform_decode_query_len(
    common_attn_metadata: CommonAttentionMetadata,
    num_decodes: int,
    num_decode_tokens: int,
    *,
    decode_only: bool,
) -> int:
    """Per-request query length of the decode rows, which
    ``split_decodes_and_prefills(require_uniform=True)`` keeps uniform.

    ``decode_only`` (a decode-only step of a builder that avoids per-step host
    work) takes it from ``max_query_len`` with no host tensor math; otherwise
    it is read from, and checked against, the host query-start offsets.
    """
    if decode_only:
        decode_query_len = common_attn_metadata.max_query_len
    else:
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
        decode_query_len = int(query_lens_cpu[0].item())
        assert decode_query_len > 0
        assert torch.all((query_lens_cpu == decode_query_len) | (query_lens_cpu == 0))
    assert num_decode_tokens == num_decodes * decode_query_len
    return decode_query_len


class MiniMaxM3SparseICPMetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    cudagraph_decode_phase_only = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        config = vllm_config.model_config.hf_text_config
        self.q8kv4_plans = get_icp_main_decode_plans(
            config.num_attention_heads
            // vllm_config.parallel_config.tensor_parallel_size,
            kv_cache_spec.num_kv_heads,
            device,
        )
        self._page_indptr: dict[int, torch.Tensor] = {}
        self._max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.prefill_cu_seqlens_k = torch.zeros(
            vllm_config.scheduler_config.max_num_seqs + 1,
            dtype=torch.int32,
            device=device,
        )
        self.indexer_builder = MiniMaxM3IndexerMSAMetadataBuilder(
            kv_cache_spec, layer_names, vllm_config, device
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        # Q8KV4 decode reads request lengths and block tables directly; prefill
        # rows get only the CSR metadata the NVFP4 sparse forward reads.
        cm = common_attn_metadata
        nd, np, nt_d, nt_p = split_decodes_and_prefills(
            cm, decode_threshold=self.reorder_batch_threshold, require_uniform=True
        )
        decode = None
        if nd:
            q_len = uniform_decode_query_len(cm, nd, nt_d, decode_only=np == 0)
            block_table = cm.block_table_tensor[:nd]
            if not block_table.is_contiguous():
                raise ValueError("ICP main decode requires a contiguous block table")
            stride = block_table.stride(0)
            indptr = self._page_indptr.get(stride)
            if indptr is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "ICP page-table offsets were not bound before capture"
                    )
                indptr = (
                    torch.arange(
                        self._max_num_seqs + 1,
                        dtype=torch.int32,
                        device=block_table.device,
                    )
                    * stride
                )
                self._page_indptr[stride] = indptr
            decode = ICPMainDecodeMetadata(
                seq_lens=cm.seq_lens[:nd],
                block_table=block_table,
                decode_query_len=q_len,
                plan=self.q8kv4_plans.get(nd, q_len),
                kv_indices=block_table.view(-1),
                kv_indptr=indptr[: nd + 1],
            )
        prefill = None
        if np:
            # r13 prefill attend (sparse_atten_nvfp4_kv_func) consumes the MSA
            # CSR metadata; built as MiniMaxM3SparseMetadataBuilder.build does.
            seq_lens_cpu = cm.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            total_kv_blocks = (
                (
                    (seq_lens_cpu[nd : cm.num_reqs] + SPARSE_BLOCK_SIZE - 1)
                    // SPARSE_BLOCK_SIZE
                )
                .sum()
                .item()
            )
            kv_lens = cm.seq_lens[nd : cm.num_reqs]
            # Retained, zero-headed: only [1 : np + 1] is written per step.
            cu_seqlens_k = self.prefill_cu_seqlens_k[: np + 1]
            torch.cumsum(kv_lens, dim=0, out=cu_seqlens_k[1:])
            prefill = MiniMaxM3SparsePrefillMetadata(
                cu_seqlens_q=(cm.query_start_loc[nd:] - nt_d).to(torch.int32),
                cu_seqlens_k=cu_seqlens_k,
                seq_lens=kv_lens,
                # Unused by the NVFP4 sparse forward; not materialised.
                context_lens=None,  # type: ignore[arg-type]
                block_table=cm.block_table_tensor[nd : cm.num_reqs],
                max_query_len=cm.max_query_len,
                max_seq_len=cm.max_seq_len,
                total_kv_blocks=total_kv_blocks,
            )
        metadata = MiniMaxM3SparseICPMetadata(
            seq_lens=cm.seq_lens,
            max_seq_len=cm.max_seq_len,
            slot_mapping=cm.slot_mapping,
            num_actual_tokens=cm.num_actual_tokens,
            num_decodes=nd,
            num_decode_tokens=nt_d,
            num_prefills=np,
            num_prefill_tokens=nt_p,
            decode=decode,
            prefill=prefill,
        )
        metadata.indexer = self.indexer_builder.build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        return metadata


class MiniMaxM3SparseICPBackend(MiniMaxM3SparseMSANvfp4Backend):
    @staticmethod
    def get_builder_cls():
        return MiniMaxM3SparseICPMetadataBuilder


class MiniMaxM3SparseICPAttention(nn.Module, AttentionLayerBase):
    """Block-sparse attention layer with the lightning-indexer branch.

    This is a merged attention layer: it owns the projections (qkv + index
    q/k), per-head QK norms and RoPE, *and* the attention-backend wiring that a
    generic ``Attention`` layer would normally provide. It owns the compound
    KV cache and the compute-only lightning indexer (``MiniMaxM3Indexer``).

    The index branch (index_{q,k}_proj + index_{q,k}_norm) feeds the sparse
    top-k block selection. M3 always disables the index value/output
    projections (``sparse_disable_index_value`` set for every sparse layer), so
    ``index_{v,o}_proj`` are never created.
    """

    # Registered or assigned by set_default_quant_scales at construction.
    _q_scale: torch.Tensor
    _k_scale: torch.Tensor
    _k_scale_cpu: torch.Tensor
    _v_scale: torch.Tensor
    _v_scale_cpu: torch.Tensor
    _q_scale_float: float
    _k_scale_float: float
    _v_scale_float: float
    _prob_scale: torch.Tensor

    def __init__(
        self,
        config: PreTrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        validate_icp_config(get_current_vllm_config(), config)
        require_msa_icp()
        _assert_icp_prewarm(get_tensor_model_parallel_rank())
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        # The candidate exchange picks its transport slot as
        # `sparse_layer_index % slots`, and its slot-discipline check is stated
        # over the number of SPARSE layers -- so this is the ordinal within the
        # sparse sweep, not the decoder layer id, which skips the dense layers
        # and would let two consecutive sparse launches share one slot. A layer
        # forced sparse outside the config's own set (the native MTP block) sits
        # after that sweep, which is what makes the binder's layer count -- and
        # therefore its slot-discipline check -- see it.
        sparse_layer_ids = sorted(_sparse_attention_layer_ids(config))
        if layer_id in sparse_layer_ids:
            self.sparse_layer_index = sparse_layer_ids.index(layer_id)
        else:
            self.sparse_layer_index = len(sparse_layer_ids)

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Sparse "index" branch dims.
        sparse_cfg = config.sparse_attention_config
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]

        # Single fused projection: q, k, v, index_q, index_k in one GEMM.
        # W (= TP size, see `index_world_size` below) and the TP rank are
        # passed explicitly because C7 replicates the index-Q head group across
        # the W ICP ranks -- each rank scores all the group's index-query heads
        # against its own R = 128/W key rows -- while q/k/v stay sharded over
        # the full TP group. A defaulted W would silently narrow this rank's
        # queries instead of failing.
        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            self.total_idx_heads,
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            index_world_size=tp_size,
            tp_rank=get_tensor_model_parallel_rank(),
        )
        # The projection is the single source of truth for the index-Q width:
        # the `[num_tokens, index_q_size]` buffer below and the producer's
        # `num_index_heads` must be exactly what the GEMM emits.
        self.num_idx_heads = self.qkv_proj.num_index_heads
        self.index_q_size = self.num_idx_heads * self.idx_head_dim
        # reduce_results=False: the attention all-reduce is fused with the
        # following post_attention_layernorm (GemmaRMSNorm) in the decoder layer
        # via fused_allreduce_gemma_rms_norm.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK norm (qk_norm_type == "per_head", use_gemma_norm == True).
        self.q_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Partial RoPE: rotary_dim == head_dim * partial_rotary_factor.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.index_q_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.index_k_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.index_rotary_emb = self.rotary_emb

        # Attention-backend wiring.
        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        # MiniMax-M3 sparse attention owns its KV-cache insert/read path instead
        # of wrapping the generic Attention module. Keep the same runtime scale
        # attributes so FP8 KV reads can honor vLLM's per-layer descale contract.
        self.calculate_kv_scales = False
        set_default_quant_scales(self, register_buffer=True)
        # Index-key storage format (--attention-config '{"indexer_kv_dtype":
        # ...}'). refined-icp-v1 fixes it at scale-free FP8 E4M3: the index
        # region of a compound page is budgeted at exactly one byte per element
        # with no scale payload, so a 2-byte bf16 key does not fit the page the
        # KV-cache manager allocated.
        self.indexer_kv_dtype = vllm_config.attention_config.indexer_kv_dtype
        if self.indexer_kv_dtype not in ("fp8", "fp8_e4m3"):
            raise CompoundPageError(
                f"{self.layer_name}: the index keys live inside this layer's KV "
                f"page as scale-free FP8 E4M3 (1 byte/element), but "
                f"indexer_kv_dtype={self.indexer_kv_dtype!r} was requested. "
                'Pass --attention-config \'{"indexer_kv_dtype": "fp8"}\'.'
            )

        # ---- compound KV page (refined-icp-v1 clause C1) -------------------
        # One physical page per cache-config block carries this rank's main K/V
        # and the index-key rows it owns. W (= TP size) is the indexer
        # context-parallel degree; `CompoundPageLayout` owns the geometry.
        self.tp_rank = get_tensor_model_parallel_rank()
        self.index_world_size = tp_size
        self.main_format = SparseMainFormat.NVFP4
        physical_page_tokens = vllm_config.cache_config.block_size
        if sparse_cfg["sparse_block_size"] != 128:
            raise CompoundPageError("MiniMax sparse ranking requires B128")
        if physical_page_tokens != 128:
            raise CompoundPageError(
                "Compound pages require P128; got "
                f"P{physical_page_tokens}. P256/R128 is not integrated."
            )
        self.compound_layout = CompoundPageLayout.build(
            tp_size=tp_size,
            main_format=self.main_format,
            ranking_block_tokens=sparse_cfg["sparse_block_size"],
            physical_page_tokens=physical_page_tokens,
            total_main_kv_heads=self.total_num_kv_heads,
            main_head_dim=self.head_dim,
            index_head_dim=self.idx_head_dim,
        )
        self.index_rows_per_rank = self.compound_layout.index_rows_per_rank
        # Bind non-owning main/index aliases once per cache lifetime.
        self._bound: BoundCompoundCache | None = None
        # Transport seam. C1 shards the scoring as well as the store, so this
        # rank's block scores are partial maxima; the true score is the maximum
        # over the W ranks, resolved by merging 8-byte candidate records rather
        # than by gathering keys. Local Top-16 per index head goes in, merged
        # block ids come out. Until an exchange is bound, W > 1 is refused at
        # bind time rather than selecting from partial scores alone. This layer
        # implements no scoring, selection, merge or transport itself.
        self.candidate_exchange: Callable[[torch.Tensor], torch.Tensor] | None = None

        # Shared top-k buffer: the indexer writes the selected blocks into it and
        # the attend impl reads them back (so nothing crosses the eager break as a
        # Python value, which would freeze at capture).
        self.topk_indices_buffer = topk_indices_buffer
        msa_decode_backend = vllm_config.attention_config.minimax_m3_msa_decode_backend
        # Admission requires the public CUTLASS selection for this ICP route.
        self.attn_backend = MiniMaxM3SparseICPBackend
        self.impl: MiniMaxM3SparseICPImpl = MiniMaxM3SparseICPImpl(  # type: ignore[assignment]
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
            msa_decode_backend=msa_decode_backend,
        )
        # Compute only: no side cache, no spec, no builder of its own. The
        # bound index alias and the indexer sub-metadata this layer's own
        # builder produced are handed in on every call.
        self.indexer = MiniMaxM3Indexer(
            num_kv_heads=self.num_kv_heads,
            scale=self.scaling,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
            num_index_heads=self.num_idx_heads,
            index_head_dim=self.idx_head_dim,
            prefix=self.layer_name,
            init_blocks=sparse_cfg.get("sparse_init_block", 0),
            local_blocks=sparse_cfg.get("sparse_local_block", 0),
            score_type=sparse_cfg.get("sparse_score_type", "max"),
            indexer_kv_dtype=self.indexer_kv_dtype,
            topk_indices_buffer=topk_indices_buffer,
            # Fragment ownership, explicit and identical to the producer's: the
            # indexer's score plans, page lists and head routing are keyed on
            # it, so it must be the (W, rank) this layer stores its rows with.
            icp_c=self.index_world_size,
            icp_rank=self.tp_rank,
        )

        # Resolve the compiled operator and cross-repository plan ABI before
        # cache binding or graph capture. Both caches use this single writer.
        self._fused_indexer_kv_write = require_icp_writer()
        from fmha_sm100.icp.scorer.prefill import api as msa_api

        if msa_api.ICP_DEVICE_PLAN_ABI_VERSION != ICP_DEVICE_PLAN_ABI:
            raise RuntimeError("msa_icp requires MSA device-plan ABI 3")
        self._producer_kv_cache_dtype = (
            "nvfp4" if self.main_format is SparseMainFormat.NVFP4 else "fp8_e4m3"
        )
        self._warmup_sparse_producer()

        # Register the main K/V cache so the KV-cache manager allocates it.
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache

    @torch.no_grad()
    def _warmup_sparse_producer(self) -> None:
        """Load the native ICP specialization before capture, with no live cache."""
        if not torch.cuda.is_available():
            return
        key = (
            torch.accelerator.current_device_index(),
            self.num_heads,
            self.num_kv_heads,
            self.num_idx_heads,
            self.rotary_emb.rotary_dim,
        )
        if key in _FRAGMENT_PRODUCER_WARMED:
            return
        device = torch.device("cuda", key[0])
        dtype = torch.bfloat16
        row = (self.num_heads + 2 * self.num_kv_heads + self.num_idx_heads + 1) * 128
        norm_weight = torch.zeros(128, device=device, dtype=dtype)
        self._fused_indexer_kv_write(
            qkv=torch.zeros((1, row), device=device, dtype=dtype),
            q_norm_weight=norm_weight,
            k_norm_weight=norm_weight,
            cos_sin_cache=torch.zeros(
                (1, self.rotary_emb.rotary_dim), device=device, dtype=dtype
            ),
            positions=torch.zeros(1, device=device, dtype=torch.int64),
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            rotary_dim=self.rotary_emb.rotary_dim,
            eps=self.q_norm.variance_epsilon,
            index_q_norm_weight=norm_weight,
            index_k_norm_weight=norm_weight,
            num_index_heads=self.num_idx_heads,
            slot_mapping=torch.full((1,), -1, device=device, dtype=torch.int64),
            block_size=self.compound_layout.physical_page_tokens,
            q_out=torch.empty((1, self.q_size), device=device, dtype=dtype),
            index_q_out=torch.empty(
                (1, self.index_q_size),
                device=device,
                dtype=self.indexer.index_kv_torch_dtype,
            ),
            q_fp8_out=torch.empty(
                (1, self.q_size), device=device, dtype=torch.float8_e4m3fn
            ),
            kv_cache_dtype="nvfp4",
            index_block_tokens=self.compound_layout.physical_page_tokens,
            index_rows_per_rank=self.index_rows_per_rank,
            index_rank=self.tp_rank,
            index_world_size=self.index_world_size,
            enable_pdl=ICP_PRODUCER_ENABLE_PDL,
        )
        torch.accelerator.synchronize(device)
        _FRAGMENT_PRODUCER_WARMED.add(key)

    def bind_candidate_exchange(
        self, exchange: "Callable[[torch.Tensor], torch.Tensor]"
    ) -> None:
        """Bind the one candidate exchange for this sparse-layer invocation.

        The explicit seam for the transport stream. It is handed this rank's
        completed query-major local candidates ``[Qexchange, 4, 16, 2]`` -- 8
        bytes per record, an fp32 score and an int32 global block id -- and must
        return the merged selected ids ``[Qcapacity, Hlocal, 16]``, ascending
        valid prefix then -1.

        Exactly one exchange per invocation, after every bounded query chunk has
        finished: NCCL for any invocation containing prefill, K5T/D3 without ACK
        for pure decode and target verification. At ``W == 1`` no
        exchange is needed because this rank's partial maxima are already the
        true block scores.

        Nothing here implements scoring, selection, merge or transport.
        """
        self.candidate_exchange = exchange

    def release_kv_cache(self) -> None:
        """Drop KV aliases after dependent graphs/streams have drained."""
        self._bound = None
        self.impl.release_kv_cache()
        # The model-scoped peer window has an independent distributed lifetime.
        # A temporary KV profiling pool must never close it on only one rank.

    def shutdown_model_resources(self) -> None:
        """Drop final model ownership without disturbing other live models."""
        self.impl.shutdown_model_resources()

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind aliases once at allocation, outside model/graph execution."""
        identity = compound_cache_identity(kv_cache, self.compound_layout)
        if self._bound is not None:
            if (
                self._bound.raw is not kv_cache
                or self._bound.storage_identity != identity
            ):
                raise CompoundPageError(
                    "ICP cache changed without release_kv_cache after draining graphs"
                )
        else:
            if self.candidate_exchange is None:
                raise CompoundPageError("ICP candidate exchange is not bound")
            self._bound = bind_compound_cache(
                kv_cache, self.compound_layout, self.tp_rank
            )
            self.impl.bind_kv_cache(self._bound.main)
        self.kv_cache = kv_cache

    def _bound_compound_cache(self) -> BoundCompoundCache:
        bound = self._bound
        if bound is None:
            raise CompoundPageError("ICP cache aliases must be bound before forward")
        # Allocation-time binding validates the full storage signature. Keep
        # only an object guard in the compiled producer path: data_ptr/storage
        # inspection here would force a TorchDynamo graph break.
        if bound.raw is not self.kv_cache:
            raise CompoundPageError(
                "ICP cache identity changed during its live lifetime"
            )
        return bound

    def get_attn_backend(self) -> type[MiniMaxM3SparseBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        if vllm_config.cache_config.block_size != 128:
            raise CompoundPageError("ICP requires logical page size 128")
        main = FullAttentionSpec(
            block_size=128,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
        customized = self.attn_backend.customize_spec(main)
        assert isinstance(customized, FullAttentionSpec)
        return MiniMaxM3CompoundSpec.from_main_spec(customized, self.compound_layout)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Single fused projection emitting [q | k | v | index_q | index_k].
        qkv, _ = self.qkv_proj(hidden_states)

        # Horizontally-fused per-head Gemma QK-norm + partial NeoX RoPE on the
        # main (q/k) and index (index_q/index_k) branches, all read straight out
        # of the single fused ``qkv`` tensor (the "5 results").  Once the paged
        # caches are bound the kernel also inserts k/v and the index key into
        # them; the initial memory-profiling run (caches unbound, no slot_mapping)
        # short-circuits to zeros below. k/v and index_k are rewritten in place
        # inside qkv (and scatter-inserted into the caches); q and index_q are
        # de-interleaved
        # straight into the dedicated contiguous ``q``/``index_q`` buffers below.

        cos_sin_cache = self.rotary_emb.cos_sin_cache
        rotary_dim = self.rotary_emb.rotary_dim
        eps = self.q_norm.variance_epsilon
        num_tokens = qkv.shape[0]

        fwd_slot_mapping = get_forward_context().slot_mapping
        if (
            not isinstance(fwd_slot_mapping, dict)
            or self.layer_name not in fwd_slot_mapping
        ):
            # Memory-profiling run: caches not yet bound, slot_mapping is empty.
            return qkv.new_zeros((num_tokens, self.hidden_size))

        # ONE slot map. Main K/V and the index rows share this layer's page, so
        # they page identically: for a valid slot `s`, `page = s // B` and
        # `u = s % B`; every rank writes its local main KV heads at `(page, u)`
        # and the single rank `u // R` writes index row `u % R`.
        slot_mapping = fwd_slot_mapping[self.layer_name]
        bound = self._bound_compound_cache()
        q = qkv.new_empty((num_tokens, self.q_size))
        # CUDAGRAPH CAPTURE/REPLAY CONTRACT -- do NOT make this conditional on
        # per-step metadata.  This allocation sits inside the captured region,
        # while its consumer (`impl.forward`, reached through the
        # `@eager_break_during_capture` on `_run_attention` below) runs in an
        # eager break that is replayed with CAPTURE-TIME arguments while
        # re-reading the LIVE forward context.  If `should_use_msa_decode`
        # varied per step, a shape captured as "not CUTLASS" would bake in
        # `query_fp8=None` and could still be routed to the CUTLASS branch at
        # replay.  `should_use_msa_decode` is static by contract for exactly
        # this reason -- see MiniMaxM3SparseImpl.should_use_msa_decode.
        use_msa_decode = self.impl.should_use_msa_decode(self.layer_name)
        query_fp8 = (
            torch.empty(
                (num_tokens, self.q_size),
                dtype=torch.float8_e4m3fn,
                device=qkv.device,
            )
            if use_msa_decode
            else None
        )
        # index_q carries the index-key storage dtype (scale-free FP8 E4M3);
        # the fused producer emits fp8 directly into an e4m3 buffer.
        index_q = qkv.new_empty(
            (num_tokens, self.index_q_size),
            dtype=self.indexer.index_kv_torch_dtype,
        )
        # One launch: Gemma QK-norm -> partial-NeoX RoPE -> bf16 Q + unscaled
        # E4M3 Q -> index-Q -> this rank's index-K row -> packed NVFP4 main K/V.
        # Named arguments bind the extended vLLM operator's ICP contract.
        #
        # `index_cache` is the 3-D fragment view `[num_pages, R, 128]`, whose
        # `stride(0)` is the *compound* page stride and so spans the main
        # region too. A 2-D flat cache is rejected outright.
        launch_icp_kv_write(
            self.layer_name,
            self.sparse_layer_index == 0,
            self._fused_indexer_kv_write,
            dict(
                qkv=qkv,
                q_norm_weight=self.q_norm.weight,
                k_norm_weight=self.k_norm.weight,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                rotary_dim=rotary_dim,
                eps=eps,
                index_q_norm_weight=self.index_q_norm.weight,
                index_k_norm_weight=self.index_k_norm.weight,
                num_index_heads=self.num_idx_heads,
                slot_mapping=slot_mapping,
                kv_cache=bound.main,
                index_cache=bound.index,
                block_size=self.compound_layout.physical_page_tokens,
                q_out=q,
                index_q_out=index_q,
                kv_cache_dtype=self._producer_kv_cache_dtype,
                q_fp8_out=query_fp8,
                # A plain Python float that MUST be 1.0 -- the kernel is
                # specialised to a unit Q scale and hard-checks it, so a checkpoint
                # carrying a real Q scale fails loudly instead of quantizing wrongly.
                q_fp8_scale=self._q_scale_float,
                # 1-element fp32 CUDA tensors, not the `_float` scalars: reading a
                # host scalar would be a D2H sync inside a captured region.
                kv_k_scale=self._k_scale,
                kv_v_scale=self._v_scale,
                # Fragment ownership (refined-icp-v1 C1). Explicit, never inferred:
                # the index view's -2 dim is R, not B, so inferring B from it would
                # divide the parent slot by R and put every page but the first in
                # the wrong place, in bounds and silently.
                index_block_tokens=self.compound_layout.physical_page_tokens,
                index_rows_per_rank=self.index_rows_per_rank,
                index_rank=self.tp_rank,
                index_world_size=self.index_world_size,
                # Must stay False while the CuTe scorer is PDL (see indexer_icp).
                enable_pdl=ICP_PRODUCER_ENABLE_PDL,
            ),
        )

        output = torch.empty_like(q)
        attn_output = self._run_attention(q, query_fp8, index_q, output)
        output, _ = self.o_proj(attn_output)
        return output

    @eager_break_during_capture
    def _run_attention(
        self,
        query: torch.Tensor,
        query_fp8: torch.Tensor | None,
        index_query: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        # Single eager break around both: their split-K kernels read per-request
        # metadata and can't be captured into a cudagraph. The indexer writes its
        # top-k into the shared ``topk_indices_buffer``; the attend reads it back.
        #
        # The indexer holds no cache and no metadata of its own: both come from
        # this layer. ``attn_metadata`` has exactly one entry for this layer --
        # there is no second key for an index cache, because there is no index
        # cache.
        bound = self._bound_compound_cache()
        attn_metadata = get_forward_context().attn_metadata
        index_md = None
        if isinstance(attn_metadata, dict):
            layer_metadata = attn_metadata[self.layer_name]
            if not isinstance(layer_metadata, MiniMaxM3SparseICPMetadata):
                raise RuntimeError(
                    f"{self.layer_name}: ICP attention requires sparse metadata"
                )
            index_md = layer_metadata.indexer
        # This rank's own R-row fragment, not a gathered block: the scorer is
        # fragment-aware and reconstructs global positions as
        # `128*b + rank*R + j` from (icp_c, icp_rank). At W > 1 the block scores
        # it produces are partial maxima, so the selection is only complete
        # after `candidate_exchange` -- which the indexer calls exactly once,
        # after every bounded query chunk. This layer still implements no
        # scoring, selection, merge or transport.
        self.indexer(
            index_query,
            index_kv=bound.index,
            index_md=index_md,
            icp_c=self.index_world_size,
            icp_rank=self.tp_rank,
            layer_idx=self.sparse_layer_index,
            candidate_exchange=self.candidate_exchange,
        )
        # No transport status is read here. A check that synchronises the
        # device, copies a CUDA value into Python or waits on a CPU status gate
        # is forbidden on the per-step submission path, and moving it out of
        # graph capture does not satisfy that rule, so there is no step-boundary
        # check, no deferred checker and no downstream consumer gate. Reported
        # GPU/peer failures remain the existing CUDA/NCCL/worker handling's.
        # The main impl reads the main *region*, never the raw compound page:
        # past `main_span_bytes` lie this rank's index rows.
        # The impl consumes this layer's cache/scale fields. This merged layer
        # has a model-facing forward signature rather than Attention.forward.
        return self.impl.forward(
            cast(AttentionLayer, self),
            query,
            bound.main,
            output,
            query_fp8=query_fp8,
        )


def bind_model_candidate_exchange(self, vllm_config: VllmConfig) -> None:
    """Construct this process's one ICP candidate exchange and bind it.

    Clause C1 shards the index store across ``W = TP`` ranks, so every
    block score a sparse layer computes is a partial maximum and the
    selection is only complete after the candidate merge. Until an exchange
    is bound the layers refuse to serve (`_bound_compound_cache`), so this
    is what makes ``W > 1`` startable at all.

    All sparse layers share one exchange, separated by ``layer_idx``
    (`sparse_layer_index`), which K5T turns into a slot as
    ``layer_idx % slots``; `create_candidate_exchange` checks the slot-reuse
    discipline against the participants counted here. At ``W == 1`` nothing
    is constructed: this rank's partial maxima are already the true block
    scores, and K5T's window cannot form a symmetric-memory group alone.

    The process group is vLLM's existing TP group; the model contract fixes
    ``TP == indexer CP == W`` and puts all K5T peers in it, with vLLM
    supplying the group. TP/CP subgroups are an explicit non-goal.
    """
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return
    sparse_layers = [
        layer.self_attn
        for layer in self.layers[self.start_layer : self.end_layer]
        if isinstance(getattr(layer, "self_attn", None), MiniMaxM3SparseICPAttention)
    ]
    if not sparse_layers:
        return
    buf = self.topk_indices_buffer
    assert buf is not None
    # The merged ids are written straight into the shared top-k buffer, so
    # the exchange's output geometry IS that buffer's: T rows and this
    # rank's H_local index heads. Read off it rather than recomputed, so
    # the two cannot drift (the indexer asserts the same identity).
    token_capacity, num_heads_local = buf.shape[0], buf.shape[1]
    # The admitted exchange extents: both bands' ladders merged, because
    # one instance serves both and validates the presented extent against a
    # single admitted set. Merging cannot move either band's own selection;
    # what it buys is one retained carrier workspace per rung of either
    # ladder. Derived from the config alone (refined-icp-v1 C4: the shape
    # must be fixed across ranks), and the indexer derives the same lists
    # from the same functions, so the two cannot drift.
    token_capacities = admitted_exchange_extents(
        max_token_capacity=vllm_config.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            vllm_config.compilation_config.cudagraph_capture_sizes or ()
        ),
    )
    assert token_capacities[-1] == token_capacity, (
        f"the shared top-k buffer holds {token_capacity} rows but the "
        f"largest admitted exchange extent is {token_capacities[-1]}; the "
        "merged ids are written straight into that buffer, so its row "
        "count is the global capacity."
    )
    exchange = create_candidate_exchange(
        group=get_tp_group().device_group,
        world_size=world_size,
        rank=get_tensor_model_parallel_rank(),
        num_heads_local=num_heads_local,
        # C7: every rank scores ALL of the group's index-query heads and
        # routes head h to rank h // H_local, so the gathered head axis is
        # rank-major and W * H_local wide.
        num_index_heads=world_size * num_heads_local,
        # The participants in the slot rotation, counted -- not assumed.
        # A layer forced sparse outside the config's own set changes the
        # count, and the slot-discipline check must see the real one.
        num_sparse_layers=len(sparse_layers),
        token_capacities=token_capacities,
        device=buf.device,
    )
    self._icp_candidate_exchange = exchange
    for layer in sparse_layers:
        layer.bind_candidate_exchange(exchange)
