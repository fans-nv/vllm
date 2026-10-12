# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2 index-Q replication preserves the actual checkpoint rows of each group."""

import pytest
import torch

import vllm.model_executor.parameter as parameters
from vllm.model_executor.layers.linear import MinimaxM3QKVParallelLinearWithIndexer

HIDDEN, DIM = 32, 8
HEADS = {"q": 8, "k": 4, "v": 4, "index_q": 4, "index_k": 1}


@pytest.fixture(autouse=True)
def no_global_tp_group(monkeypatch):
    monkeypatch.setattr(parameters, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameters, "get_tensor_model_parallel_world_size", lambda: 1)


def checkpoint(group):
    tag = list(HEADS).index(group) + 1
    row = torch.arange(HEADS[group] * DIM, dtype=torch.float32)
    col = torch.arange(HIDDEN, dtype=torch.float32)
    return tag * 1_000_000 + row[:, None] * 100 + col[None, :]


def make_layer(rank, **kwargs):
    return MinimaxM3QKVParallelLinearWithIndexer(
        HIDDEN,
        DIM,
        8,
        4,
        4,
        DIM,
        bias=False,
        prefix="qkv_proj",
        tp_rank=rank,
        tp_size=2,
        **kwargs,
    )


def load(layer, loader):
    if loader == "v2":
        param = layer.weight
        param.data.fill_(float("nan"))
        fn = layer.weight_loader_v2
    else:
        param = torch.nn.Parameter(
            torch.full_like(layer.weight, float("nan")), requires_grad=False
        )
        param.output_dim = 0
        fn = layer.weight_loader
    for group in HEADS:
        fn(param, checkpoint(group), group)
    return param.detach()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("icp_width", [1, 2])
@pytest.mark.parametrize("loader", ["v2", "legacy"])
def test_index_query_replication_preserves_main_qkv_shards(rank, icp_width, loader):
    layer = make_layer(rank, index_world_size=icp_width)
    actual = load(layer, loader)
    expected_groups = []
    for group, heads in HEADS.items():
        full = checkpoint(group)
        replicated = group == "index_k" or (group == "index_q" and icp_width == 2)
        expected_groups.append(full if replicated else full.chunk(2, dim=0)[rank])
    expected = torch.cat(expected_groups, dim=0)
    assert layer.num_heads == 4
    assert layer.num_kv_heads == 2
    assert layer.num_index_heads == (4 if icp_width == 2 else 2)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rank", [0, 1])
def test_omitted_icp_width_keeps_the_parent_projection_and_loaded_rows(rank):
    default = make_layer(rank)
    explicit_off = make_layer(rank, index_world_size=1)
    assert default.output_sizes == explicit_off.output_sizes
    torch.testing.assert_close(
        load(default, "v2"), load(explicit_off, "v2"), rtol=0, atol=0
    )


def test_rank1_h4_replication_is_observable_against_parent_sharding():
    off = make_layer(1, index_world_size=1)
    on = make_layer(1, index_world_size=2)
    off_rows = load(off, "v2")[8 * DIM : 10 * DIM]
    on_rows = load(on, "v2")[8 * DIM : 12 * DIM]
    assert torch.equal(off_rows, checkpoint("index_q")[2 * DIM :])
    assert torch.equal(on_rows, checkpoint("index_q"))
    assert not torch.equal(on_rows[: 2 * DIM], off_rows)
