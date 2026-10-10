# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host phase/extent contracts; exact live geometry is tested in native Q1."""

import pytest


def test_deployment_q4_rows_use_1024_eager_rung_above_capture512():
    plan = pytest.importorskip("fmha_sm100.icp.exchange_plan")
    capacities = plan.exchange_token_capacities(
        max_token_capacity=16_384, cudagraph_capture_sizes=[1, 4, 128, 512]
    )
    select_token_capacity = plan.select_token_capacity
    assert select_token_capacity(capacities, 128 * 4) == 512
    assert select_token_capacity(capacities, 129 * 4) == 1024
    assert select_token_capacity(capacities, 256 * 4) == 1024
    assert select_token_capacity(capacities, 1) == 4
    assert select_token_capacity(capacities, 16_384) == 16_384
    with pytest.raises(ValueError, match="exceed"):
        select_token_capacity(capacities, 16_385)
