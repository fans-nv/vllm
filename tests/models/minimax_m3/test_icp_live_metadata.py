# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host phase/extent contracts; exact live geometry is tested in native Q1."""

import pytest

from vllm.models.minimax_m3.nvidia.ops.icp_dispatch import (
    BACKEND_K5T,
    BACKEND_NCCL,
    exchange_token_capacities,
    select_backend,
    select_token_capacity,
)


def test_deployment_q4_rows_use_1024_eager_rung_above_capture512():
    pytest.importorskip("fmha_sm100.icp")
    capacities = exchange_token_capacities(
        max_token_capacity=16_384, cudagraph_capture_sizes=[1, 4, 128, 512]
    )
    assert select_token_capacity(capacities, 128 * 4) == 512
    assert select_token_capacity(capacities, 129 * 4) == 1024
    assert select_token_capacity(capacities, 256 * 4) == 1024
    assert select_token_capacity(capacities, 1) == 4
    assert select_token_capacity(capacities, 16_384) == 16_384
    with pytest.raises(ValueError, match="exceed"):
        select_token_capacity(capacities, 16_385)


def test_actual_phase_selects_transport_independently_of_query_length():
    assert select_backend(has_prefill=True) == BACKEND_NCCL
    assert select_backend(has_prefill=False) == BACKEND_K5T
