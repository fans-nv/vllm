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
from vllm.logger import init_logger
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
    bind_compound_cache,
    compound_cache_identity,
)
from vllm.models.minimax_m3.common.compound_spec import MiniMaxM3CompoundSpec
from vllm.models.minimax_m3.common.ops.sparse_attn import SPARSE_BLOCK_SIZE
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseMetadataBuilder,
    MiniMaxM3SparsePrefillMetadata,
)
from vllm.models.minimax_m3.nvidia.indexer_icp import (
    MiniMaxM3IndexerMSAMetadata,
    MiniMaxM3IndexerMSAMetadataBuilder,
)
from vllm.models.minimax_m3.nvidia.model import (
    MiniMAXGemmaRMSNorm,
    _sparse_attention_layer_ids,
)
from vllm.models.minimax_m3.nvidia.msa_icp import (
    ICP_DEVICE_PLAN_ABI,
    require_icp_prewarm,
    require_icp_writer,
    require_msa_icp,
)
from vllm.models.minimax_m3.nvidia.msa_icp_main import (
    ICPMainDecodeMetadata,
    MiniMaxM3SparseICPImpl,
    MiniMaxM3SparseICPMetadata,
    get_icp_main_decode_plans,
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

logger = init_logger(__name__)

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
        # Only the qualified direct whole-page CPU connector, not MultiConnector.
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
    """Uniform decode query length; host-checked unless the step is decode-only."""
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
    """Sparse attention layer that owns the ICP compound cache and indexer."""

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
        require_icp_prewarm(get_tensor_model_parallel_rank())
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        # Ordinal within the sparse sweep: the exchange slot is index % slots.
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

        sparse_cfg = config.sparse_attention_config
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]

        # One fused GEMM; index-Q is replicated across the W ICP ranks.
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
        # The projection fixes the index-Q width the producer and buffers use.
        self.num_idx_heads = self.qkv_proj.num_index_heads
        self.index_q_size = self.num_idx_heads * self.idx_head_dim
        # The all-reduce is fused into the following post-attention norm.
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
        # Runtime scale attributes honour vLLM's per-layer KV descale contract.
        self.calculate_kv_scales = False
        set_default_quant_scales(self, register_buffer=True)
        # The compound page budgets one byte per index-key element (FP8 E4M3).
        self.indexer_kv_dtype = vllm_config.attention_config.indexer_kv_dtype
        if self.indexer_kv_dtype not in ("fp8", "fp8_e4m3"):
            raise CompoundPageError(
                f"{self.layer_name}: the index keys live inside this layer's KV "
                f"page as scale-free FP8 E4M3 (1 byte/element), but "
                f"indexer_kv_dtype={self.indexer_kv_dtype!r} was requested. "
                'Pass --attention-config \'{"indexer_kv_dtype": "fp8"}\'.'
            )

        # One page per block: this rank's main K/V plus the index rows it owns.
        self.tp_rank = get_tensor_model_parallel_rank()
        self.index_world_size = tp_size
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
            ranking_block_tokens=sparse_cfg["sparse_block_size"],
            physical_page_tokens=physical_page_tokens,
            total_main_kv_heads=self.total_num_kv_heads,
            main_head_dim=self.head_dim,
            index_head_dim=self.idx_head_dim,
        )
        # Bind non-owning main/index aliases once per cache lifetime.
        self._bound: BoundCompoundCache | None = None
        # Merges this rank's partial-max candidates; serving is refused until bound.
        self.candidate_exchange: Callable[[torch.Tensor], torch.Tensor] | None = None

        # Shared top-k buffer: nothing crosses the eager break as a Python value.
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
        from fmha_sm100.icp.indexer_runtime import IcpIndexerRuntime

        IcpIndexerRuntime.check_installed()

        # Resolve the writer and the MSA plan ABI before binding or capture.
        self._fused_indexer_kv_write = require_icp_writer()(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            rotary_dim=self.rotary_emb.rotary_dim,
            eps=self.q_norm.variance_epsilon,
            num_index_heads=self.num_idx_heads,
            block_size=self.compound_layout.physical_page_tokens,
            kv_cache_dtype="nvfp4",
            index_rank=self.tp_rank,
            index_world_size=self.index_world_size,
        )
        from fmha_sm100.icp.scorer.prefill import api as msa_api

        if msa_api.ICP_DEVICE_PLAN_ABI_VERSION != ICP_DEVICE_PLAN_ABI:
            raise RuntimeError("msa_icp requires MSA device-plan ABI 4")
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
            index_q_norm_weight=norm_weight,
            index_k_norm_weight=norm_weight,
            slot_mapping=torch.full((1,), -1, device=device, dtype=torch.int64),
            q_out=torch.empty((1, self.q_size), device=device, dtype=dtype),
            index_q_out=torch.empty(
                (1, self.index_q_size),
                device=device,
                dtype=torch.float8_e4m3fn,
            ),
            q_fp8_out=torch.empty(
                (1, self.q_size), device=device, dtype=torch.float8_e4m3fn
            ),
        )
        torch.accelerator.synchronize(device)
        _FRAGMENT_PRODUCER_WARMED.add(key)

    def bind_candidate_exchange(
        self, exchange: "Callable[[torch.Tensor], torch.Tensor]"
    ) -> None:
        """Bind the model's one candidate exchange (called once per invocation)."""
        self.candidate_exchange = exchange

    def release_kv_cache(self) -> None:
        """Drop KV aliases after dependent graphs/streams have drained."""
        self._bound = None
        self.impl.release_kv_cache()
        # The model-scoped exchange outlives a temporary profiling KV pool.

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
        # Identity check only: storage inspection here would break the Dynamo graph.
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

        # The fused producer normalises, ropes and inserts K/V and the index key.

        num_tokens = qkv.shape[0]

        fwd_slot_mapping = get_forward_context().slot_mapping
        if (
            not isinstance(fwd_slot_mapping, dict)
            or self.layer_name not in fwd_slot_mapping
        ):
            # Memory-profiling run: caches not yet bound, slot_mapping is empty.
            return qkv.new_zeros((num_tokens, self.hidden_size))

        # One slot map: rank u // R writes index row u % R of the shared page.
        slot_mapping = fwd_slot_mapping[self.layer_name]
        bound = self._bound_compound_cache()
        q = qkv.new_empty((num_tokens, self.q_size))
        # Must not depend on per-step metadata: the eager break replays capture args.
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
        index_q = qkv.new_empty(
            (num_tokens, self.index_q_size),
            dtype=torch.float8_e4m3fn,
        )
        # index_cache is [num_pages, R, 128] with the compound page stride.
        launch_icp_kv_write(
            self.layer_name,
            self.sparse_layer_index == 0,
            self._fused_indexer_kv_write,
            dict(
                qkv=qkv,
                q_norm_weight=self.q_norm.weight,
                k_norm_weight=self.k_norm.weight,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                positions=positions,
                index_q_norm_weight=self.index_q_norm.weight,
                index_k_norm_weight=self.index_k_norm.weight,
                slot_mapping=slot_mapping,
                kv_cache=bound.main,
                index_cache=bound.index,
                q_out=q,
                index_q_out=index_q,
                q_fp8_out=query_fp8,
                # Must be 1.0: the kernel is specialised to a unit Q scale.
                q_fp8_scale=self._q_scale_float,
                # Device scalars: a host read would sync inside the captured region.
                kv_k_scale=self._k_scale,
                kv_v_scale=self._v_scale,
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
        # One eager break: both read per-request metadata and cannot be captured.
        bound = self._bound_compound_cache()
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            layer_metadata = attn_metadata[self.layer_name]
            if not isinstance(layer_metadata, MiniMaxM3SparseICPMetadata):
                raise RuntimeError(
                    f"{self.layer_name}: ICP attention requires sparse metadata"
                )
            index_md = cast(MiniMaxM3IndexerMSAMetadata, layer_metadata.indexer)
            # This rank's R-row fragment; selection completes after one exchange.
            index_md.runtime.run_layer(
                index_md.icp,
                index_query,
                bound.index,
                layer_idx=self.sparse_layer_index,
                exchange=self.candidate_exchange,
                topk_buffer=self.topk_indices_buffer,
                scale=self.scaling,
            )
        # No transport status is read: no sync on the submission path.
        return self.impl.forward(
            cast(AttentionLayer, self),
            query,
            bound.main,
            output,
            query_fp8=query_fp8,
        )


def bind_model_candidate_exchange(self, vllm_config: VllmConfig) -> None:
    """Construct this process's one ICP candidate exchange and bind it.

    Every sparse layer shares it, separated by ``sparse_layer_index``. Its
    extents are both bands' admitted ladders, derived from the config alone so
    every rank agrees; the merged ids land in the shared top-k buffer.
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
    require_msa_icp()
    from fmha_sm100.icp.candidate_exchange import CandidateExchange

    # The serving policy is the factory default: D3, PDL, network merge.
    exchange = CandidateExchange.for_tp_group(
        group=get_tp_group().device_group,
        world_size=world_size,
        rank=get_tensor_model_parallel_rank(),
        topk_buffer=self.topk_indices_buffer,
        num_sparse_layers=len(sparse_layers),
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            vllm_config.compilation_config.cudagraph_capture_sizes or ()
        ),
    )
    logger.info(
        "MiniMax-M3 ICP candidate exchange: layers=%d %s",
        len(sparse_layers),
        exchange.decode_provenance,
    )
    self._icp_candidate_exchange = exchange
    for layer in sparse_layers:
        layer.bind_candidate_exchange(exchange)
