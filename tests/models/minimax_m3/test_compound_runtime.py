# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compound pages through current grouping, allocation and binding lifetimes."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig
from vllm.models.minimax_m3.common.compound_page import (
    CompoundPageLayout,
    SparseMainFormat,
    bind_compound_cache,
    compound_cache_identity,
)
from vllm.models.minimax_m3.common.compound_spec import MiniMaxM3CompoundSpec
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.utils import (
    KVBlockZeroer,
    allocate_kv_cache,
    bind_kv_cache_to_layers,
    clear_layer_kv_caches,
    copy_kv_cache_blocks_inplace,
)


class _Builder:
    requires_block_table_width = False

    def __init__(self, spec, *_args, **_kwargs):
        self.cudagraph_decode_phase_only = spec.uses_raw_page_view

    def get_cudagraph_support(self, *_args):
        return AttentionCGSupport.UNIFORM_BATCH

    def set_kernel_block_size(self, block_size):
        assert block_size == 128


class _Backend:
    @staticmethod
    def full_cls_name():
        return (__name__, "_Backend")

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [128]

    @staticmethod
    def get_builder_cls():
        return _Builder


class _Layer:
    num_heads = 32
    kv_sharing_target_layer_name = None

    def __init__(self, layout=None):
        self.layout = layout
        self.bound = None
        self.kv_cache = torch.tensor([])

    @staticmethod
    def get_attn_backend():
        return _Backend

    def bind_kv_cache(self, raw):
        if self.layout is not None:
            if self.bound is not None and self.bound.storage_identity != (
                compound_cache_identity(raw, self.layout)
            ):
                raise RuntimeError("cache lifetime must be released before rebinding")
            self.bound = bind_compound_cache(raw, self.layout, tp_rank=0)
        self.kv_cache = raw

    def release_kv_cache(self):
        self.bound = None


def test_dense_draft_compound_group_keeps_raw_lifetime_and_attention(monkeypatch):
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    layout = CompoundPageLayout.build(tp_size=2, main_format=SparseMainFormat.NVFP4)
    compound = MiniMaxM3CompoundSpec.from_compound_layout(layout)
    dense = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    specs = {
        "model.layers.0.attn": dense,
        "model.layers.1.attn": compound,
        "model.layers.2.attn": compound,
        "draft.layers.0.attn": dense,
    }
    config = SimpleNamespace(
        cache_config=CacheConfig(block_size=128),
        attention_config=SimpleNamespace(hisparse_config=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(use_ubatching=False),
        speculative_config=None,
    )
    config.cache_config.kv_cache_layout = "LBHNC"
    groups = get_kv_cache_groups(config, specs)
    assert len(groups) == 1
    assert isinstance(groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)
    bytes_per_block = 2 * (65_536 + 45_056)
    kv_config = get_kv_cache_config_from_groups(
        config, groups, available_memory=3 * bytes_per_block
    )
    assert kv_config.num_blocks == 3
    assert {tensor.size for tensor in kv_config.kv_cache_tensors} == {
        3 * bytes_per_block
    }
    layers = {
        name: _Layer(layout if spec.uses_raw_page_view else None)
        for name, spec in specs.items()
    }
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda _config, _type, names=None: {
            name: layers[name] for name in names or layers
        },
    )
    attn_groups, support, kernel_sizes = attn_utils.init_attn_backend(
        kv_config, config, torch.device("cpu")
    )
    assert kernel_sizes == [128]
    assert {name for group in attn_groups[0] for name in group.layer_names} == set(
        specs
    )
    assert support.decode_phase_only
    assert all(group.kv_cache_spec.uses_slot_mapping for group in attn_groups[0])

    caches = allocate_kv_cache(
        kv_config, torch.device("cpu"), KVCacheLayout.LBHNC, kernel_sizes
    )
    assert len({cache.untyped_storage().data_ptr() for cache in caches.values()}) == 1
    bind_kv_cache_to_layers(caches, layers)
    zeroer = KVBlockZeroer(
        torch.device("cpu"),
        attn_groups_iter=iter(attn_groups[0]),
        kernel_block_sizes=kernel_sizes,
        static_forward_context=layers,
        num_blocks=3,
    )
    assert zeroer._meta is not None
    assert sorted((zeroer._meta[2] * 4).tolist()) == [45_056, 45_056, 65_536, 65_536]
    for name in ("model.layers.1.attn", "model.layers.2.attn"):
        raw = caches[name]
        assert raw.shape == (3, 1, 1, 45_056)
        assert raw.storage_offset() > 0
        raw[0].fill_(0x17)
        bound = layers[name].bound
        assert bound is not None
        bound.index[0].view(torch.uint8).fill_(0x91)
    before = {name: cache.clone() for name, cache in caches.items()}
    copy_kv_cache_blocks_inplace(caches.values(), 3, [(0, 2)])
    for name, cache in caches.items():
        assert torch.equal(cache[2], before[name][0])
        assert torch.equal(cache[1], before[name][1])

    final_caches = allocate_kv_cache(
        kv_config, torch.device("cpu"), KVCacheLayout.LBHNC, kernel_sizes
    )
    with pytest.raises(RuntimeError, match="released before rebinding"):
        bind_kv_cache_to_layers(final_caches, layers)
    clear_layer_kv_caches(layers.values())
    assert all(
        layer.bound is None and layer.kv_cache.numel() == 0 for layer in layers.values()
    )
    bind_kv_cache_to_layers(final_caches, layers)
    assert all(layer.kv_cache is final_caches[name] for name, layer in layers.items())
