# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Only the direct whole-page CPU connector is admitted for compound KV."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import KVTransferConfig
from vllm.models.minimax_m3.nvidia.sparse_attention_icp import validate_icp_config


@pytest.fixture
def configs(monkeypatch):
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (10, 7))
    runtime = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(
            block_size=128, cache_dtype="nvfp4", enable_prefix_caching=True
        ),
        attention_config=SimpleNamespace(
            indexer_kv_dtype="fp8", minimax_m3_msa_decode_backend="cutlass"
        ),
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            max_model_len=8192, enable_sleep_mode=False, dtype=torch.bfloat16
        ),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )
    model = SimpleNamespace(
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        sparse_attention_config={
            "sparse_num_index_heads": 4,
            "sparse_index_dim": 128,
            "sparse_topk_blocks": 16,
            "sparse_block_size": 128,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
        },
    )
    return runtime, model


def _transfer(**overrides):
    values = dict(kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both")
    values.update(overrides)
    return KVTransferConfig(**values)


@pytest.mark.parametrize("backend", [None, "cpu"])
@pytest.mark.parametrize("lazy", [False, True])
def test_direct_cpu_path_admits_eager_and_lazy(configs, backend, lazy):
    runtime, model = configs
    extra = {"lazy_offload": lazy, "cpu_bytes_to_use": 8 << 30}
    if backend is not None:
        extra["kv_offload_backend"] = backend
    runtime.kv_transfer_config = _transfer(kv_connector_extra_config=extra)
    validate_icp_config(runtime, model)


@pytest.mark.parametrize(
    "overrides",
    [
        {"kv_connector": "NixlConnector"},
        {"kv_connector": "OffloadingConnector"},
        {
            "kv_connector": "MultiConnector",
            "kv_connector_extra_config": {
                "connectors": [
                    {"kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both"}
                ]
            },
        },
        {"kv_connector_module_path": "custom.connector"},
        {"kv_role": "kv_producer"},
        {"kv_role": "kv_consumer"},
        {"kv_connector_extra_config": {"kv_offload_backend": "disk"}},
        {"kv_connector_extra_config": {"kv_offload_backend": "unknown"}},
        {"kv_connector_extra_config": {"kv_offload_backend": None}},
    ],
)
def test_unqualified_connectors_fail_closed(configs, overrides):
    runtime, model = configs
    runtime.kv_transfer_config = _transfer(**overrides)
    with pytest.raises(ValueError, match="built-in SimpleCPUOffloadConnector"):
        validate_icp_config(runtime, model)


def test_cpu_offload_requires_prefix_cache_but_connector_free_does_not(configs):
    runtime, model = configs
    runtime.cache_config.enable_prefix_caching = False
    validate_icp_config(runtime, model)
    runtime.kv_transfer_config = _transfer()
    with pytest.raises(ValueError, match="requires prefix caching"):
        validate_icp_config(runtime, model)


@pytest.mark.parametrize(
    ("section", "name", "value", "message"),
    [
        ("parallel_config", "tensor_parallel_size", 1, "tensor_parallel_size=2"),
        ("parallel_config", "decode_context_parallel_size", 2, "parallel_size=1"),
        ("cache_config", "block_size", 64, "P128"),
        ("cache_config", "cache_dtype", "fp8", "plain nvfp4"),
        ("model_config", "enable_sleep_mode", True, "sleep mode"),
        ("model_config", "dtype", torch.float16, "BF16"),
        ("attention_config", "minimax_m3_msa_decode_backend", "triton", "cutlass"),
    ],
)
def test_cpu_connector_does_not_relax_compound_requirements(
    configs, section, name, value, message
):
    runtime, model = configs
    runtime.kv_transfer_config = _transfer()
    setattr(getattr(runtime, section), name, value)
    with pytest.raises(ValueError, match=message):
        validate_icp_config(runtime, model)
