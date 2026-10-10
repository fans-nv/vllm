# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the MiniMax-M3 writer schema extended for ICP.

The ordinary wrapper keeps the pre-ICP call; ICP callers bind the native op
directly. Native arithmetic/fragment checks live with the NVFP4 tests.
"""

import ast
from pathlib import Path

import pytest
import regex as re
import torch

from vllm import _custom_ops as ops


@pytest.fixture
def launch(monkeypatch):
    calls = []

    def dispatch(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert", dispatch, raising=False
    )
    return calls


def _inputs():
    tensor = torch.empty(1)
    return dict(
        qkv=tensor,
        q_norm_weight=tensor,
        k_norm_weight=tensor,
        cos_sin_cache=tensor,
        positions=tensor,
        num_heads=32,
        num_kv_heads=2,
        rotary_dim=64,
        eps=1e-6,
    )


def test_default_writer_keeps_the_old_binary_call_prefix(launch):
    """Ordinary MSA must remain callable with the pre-ICP compiled schema."""
    ops.fused_minimax_m3_qknorm_rope_kv_insert(**_inputs())
    assert len(launch) == 1
    args, kwargs = launch[0]
    assert len(args) == 25
    assert args[5:9] == (32, 2, 64, 1e-6)
    assert args[23:25] == (None, None)  # public kv_k_scale/kv_v_scale
    assert kwargs == {}


def test_source_schema_marks_icp_destinations_mutable_and_advertises_abi():
    """A missing mutation annotation can discard the metadata producer."""
    bindings = (
        Path(__file__).resolve().parents[2] / "csrc/libtorch_stable/torch_bindings.cpp"
    ).read_text()
    definition = bindings.split('"fused_minimax_m3_qknorm_rope_kv_insert("', 1)[1]
    definition = definition.split(");", 1)[0]
    schema = "fused_minimax_m3_qknorm_rope_kv_insert(" + "".join(
        ast.literal_eval(part) for part in re.findall(r'"[^"\n]*"', definition)
    )
    arguments = {arg.name: arg for arg in torch._C.parse_schema(schema).arguments}
    assert arguments["icp_plan_abi"].default_value == 3
    assert arguments["index_world_size"].default_value == 0
    assert arguments["enable_pdl"].default_value is True
    for name in (
        "icp_positions",
        "icp_active",
        "icp_local_nvalid",
        "icp_local_forced",
        "icp_global_nvalid",
        "icp_forced",
        "icp_n_ordinary",
        "icp_candidates",
        "icp_qo_offsets",
        "icp_plan_segments",
        "icp_plan_work",
        "icp_plan_header",
        "icp_plan_ranges",
    ):
        assert arguments[name].alias_info.is_write, name
