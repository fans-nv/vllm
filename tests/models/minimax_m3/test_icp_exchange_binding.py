# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model binds one D3/PDL/network-merge exchange to every sparse ICP layer."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.minimax_m3.nvidia import sparse_attention_icp as icp_attn


def test_one_exchange_with_the_serving_policy_reaches_every_sparse_layer(
    monkeypatch,
):
    pytest.importorskip("fmha_sm100.icp")
    import fmha_sm100.icp.candidate_exchange as owner

    seen = {}

    class Recorder:
        decode_provenance: dict = {}

        @classmethod
        def for_tp_group(cls, **kwargs):
            seen.update(kwargs)
            return cls()

    monkeypatch.setattr(owner, "CandidateExchange", Recorder)
    monkeypatch.setattr(icp_attn, "require_msa_icp", lambda: None)
    monkeypatch.setattr(icp_attn, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(icp_attn, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        icp_attn, "get_tp_group", lambda: SimpleNamespace(device_group="tp")
    )
    bound = []

    class Layer(icp_attn.MiniMaxM3SparseICPAttention):
        def __init__(self):  # noqa: D107 - skip the real layer construction
            pass

        def bind_candidate_exchange(self, exchange):
            bound.append(exchange)

    layers = [SimpleNamespace(self_attn=Layer()) for _ in range(3)]
    layers.insert(1, SimpleNamespace(self_attn=object()))
    model = SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=len(layers),
        topk_indices_buffer=torch.empty(16384, 2, 16, dtype=torch.int32),
    )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 4, 512]),
    )
    icp_attn.bind_model_candidate_exchange(model, config)
    # The serving policy (D3, PDL, network merge) is the factory default.
    assert not {"fused_decode", "use_pdl", "classic_merge"} & set(seen)
    assert seen["num_sparse_layers"] == 3
    assert (seen["world_size"], seen["rank"], seen["group"]) == (2, 1, "tp")
    assert seen["topk_buffer"] is model.topk_indices_buffer
    assert seen["max_num_batched_tokens"] == 16384
    assert list(seen["cudagraph_capture_sizes"]) == [1, 4, 512]
    assert len(bound) == 3 and all(e is model._icp_candidate_exchange for e in bound)


def test_factory_defaults_are_the_serving_policy():
    owner = pytest.importorskip("fmha_sm100.icp.candidate_exchange")
    import inspect

    params = inspect.signature(owner.CandidateExchange.for_tp_group).parameters
    assert params["use_pdl"].default is True
    # D3 and the network merge are fixed policy, not knobs.
    assert not {"fused_decode", "classic_merge"} & set(params)
