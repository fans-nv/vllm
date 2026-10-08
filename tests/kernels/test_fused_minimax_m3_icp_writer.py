# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ICP-specific oracles for the public per-head MiniMax-M3 cache layout.

The ordinary writer's independent numerical tests stay in the adjacent
qknorm/rope suite. These cases deliberately exercise ICP's different index-Q
rounding and NVFP4 arithmetic, without using another fused writer as an oracle.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

D, P, R, HQ, HKV, HI = 128, 128, 64, 32, 2, 4
MAIN_BYTES = 2 * HKV * P * 72
PAGE_BYTES = MAIN_BYTES + R * D

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="ICP writer needs the compiled SM10x extension",
)


def _case(n, dtype, rank, scales=(1.0, 1.0)):
    pages = max(1, (n + P - 1) // P)
    offset, stride = 32, PAGE_BYTES + 64
    storage = torch.full(
        (offset + pages * stride + 32,), 0xA5, dtype=torch.uint8, device="cuda"
    )
    main = storage.as_strided((pages, 2 * HKV, P, 72), (stride, P * 72, 72, 1), offset)
    index = storage.view(torch.float8_e4m3fn).as_strided(
        (pages, R, D), (stride, D, 1), offset + MAIN_BYTES
    )
    row = torch.arange(n, dtype=torch.int64, device="cuda")
    slots = (pages - 1 - row // P) * P + row % P
    if n:
        slots[-1] = -1
    weights = torch.zeros(D, dtype=dtype, device="cuda")
    cos_sin = torch.cat((torch.ones(32), torch.zeros(32))).to("cuda", dtype)
    kwargs = dict(
        qkv=torch.ones(n, (HQ + 2 * HKV + HI + 1) * D, dtype=dtype, device="cuda"),
        q_norm_weight=weights,
        k_norm_weight=weights,
        cos_sin_cache=cos_sin[None],
        positions=torch.zeros(n, dtype=torch.int64, device="cuda"),
        num_heads=HQ,
        num_kv_heads=HKV,
        rotary_dim=64,
        eps=0.0,
        index_q_norm_weight=weights,
        index_k_norm_weight=weights,
        num_index_heads=HI,
        slot_mapping=slots,
        kv_cache=main,
        index_cache=index,
        block_size=P,
        q_out=torch.empty(n, HQ * D, dtype=dtype, device="cuda"),
        q_fp8_out=torch.empty(n, HQ * D, dtype=torch.float8_e4m3fn, device="cuda"),
        index_q_out=torch.empty(n, HI * D, dtype=torch.float8_e4m3fn, device="cuda"),
        kv_cache_dtype="nvfp4",
        kv_k_scale=torch.tensor(scales[0], dtype=torch.float32, device="cuda"),
        kv_v_scale=torch.tensor(scales[1], dtype=torch.float32, device="cuda"),
        index_block_tokens=P,
        index_rows_per_rank=R,
        index_rank=rank,
        index_world_size=2,
        enable_pdl=False,
    )
    return kwargs, storage, offset, stride


def _nvfp4_control_bytes(x, global_scale):
    """Independent mathematical oracle for controlled, non-midpoint inputs.

    ICP has no minimum block-scale floor and preserves signed zero. The input
    groups below use exact powers of two or saturated values, away from places
    where approximate reciprocal instructions could change a rounding tie.
    """
    blocks = x.float().unflatten(-1, (8, 16))
    sf = (blocks.abs().amax(-1, keepdim=True) / (6 * global_scale)).clamp(max=448)
    sf = sf.to(torch.float8_e4m3fn).float()
    inverse = torch.where(sf != 0, 1 / (sf * global_scale), 0.0)
    q = blocks * inverse
    midpoints = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    magnitude = sum(
        (q.abs() >= m) if i % 2 else (q.abs() > m) for i, m in enumerate(midpoints)
    ).to(torch.uint8)
    codes = (magnitude | (torch.signbit(q).to(torch.uint8) << 3)).flatten(-2)
    return codes[..., 0::2] | (codes[..., 1::2] << 4), sf.squeeze(-1).to(
        torch.float8_e4m3fn
    ).view(torch.uint8)


@pytest.mark.parametrize("n", [129, 1023, 1024])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.5, 2.0)])
def test_icp_owns_only_its_index_fragment_in_public_head_slots(n, rank, dtype, scales):
    """Check whole compound bytes, main replication, both owners and padding."""
    kwargs, storage, offset, stride = _case(n, dtype, rank, scales)
    expected = storage.clone()
    # Distinct heads/groups expose accidental region-major addressing. Include
    # tiny scale underflow, signed zeros and block-scale saturation.
    levels = torch.tensor(
        [-6, -4, -3, -2, -1.5, -1, -0.5, -0.0, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
        device="cuda",
    )
    amplitudes = torch.tensor([2**-12, 0.5, 1, 2, 8, 32, 256, 2048], device="cuda")
    value = (amplitudes[:, None] * levels).flatten().to(dtype)
    values = kwargs["qkv"][:, (HQ + HKV) * D : (HQ + 2 * HKV) * D].view(n, HKV, D)
    values[:, 0] = value
    values[:, 1] = -value
    sides = (torch.ones_like(values), values.clone())
    slots = kwargs["slot_mapping"]
    valid = slots >= 0
    page, token = slots[valid] // P, slots[valid] % P
    group = torch.arange(8, device="cuda")
    for side, (x, scale) in enumerate(zip(sides, scales)):
        data, sf = _nvfp4_control_bytes(x[valid], scale)
        for head in range(HKV):
            base = offset + page * stride + (2 * head + side) * P * 72
            expected[
                base[:, None] + token[:, None] * 64 + torch.arange(64, device="cuda")
            ] = data[:, head]
            t = token[:, None]
            sf_offset = (
                t * 8 + group
                if side == 0
                else ((t // 4) * 4 + group // 2) * 8 + (group % 2) * 4 + t % 4
            )
            expected[base[:, None] + P * 64 + sf_offset] = sf[:, head]
    owned = slots[(slots >= 0) & ((slots % P) // R == rank)]
    index_bytes = torch.ones(D, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    addresses = offset + (owned // P) * stride + MAIN_BYTES + (owned % R) * D
    expected[addresses[:, None] + torch.arange(D, device="cuda")] = index_bytes
    ops.fused_minimax_m3_qknorm_rope_kv_insert(**kwargs)
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    for name in ("q_out", "q_fp8_out", "index_q_out"):
        torch.testing.assert_close(
            kwargs[name].float(), torch.ones_like(kwargs[name].float()), rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_icp_index_q_keeps_direct_fp32_rounding_and_both_main_q_outputs(dtype):
    """An exact FP8 midpoint after model rounding distinguishes both contracts."""
    kwargs, *_ = _case(2, dtype, 0)
    delta = torch.finfo(dtype).eps / 4
    kwargs["cos_sin_cache"][:, :32] = 1.0625
    kwargs["cos_sin_cache"][:, 32:] = -delta
    # No caches: also exercise the profiling/warmup specialization.
    kwargs["kv_cache"] = kwargs["index_cache"] = None
    kwargs["kv_k_scale"] = kwargs["kv_v_scale"] = None
    ops.fused_minimax_m3_qknorm_rope_kv_insert(**kwargs)
    iq = kwargs["index_q_out"].view(2, HI, D).float()
    q = kwargs["q_out"].view(2, HQ, D).float()
    q8 = kwargs["q_fp8_out"].view(2, HQ, D).float()
    assert torch.all(iq[..., :32] == 1.125)
    assert torch.all(q[..., :32] == 1.0625)
    assert torch.all(q8[..., :32] == 1.0)


def test_icp_direct_index_fp8_preserves_nan_class():
    kwargs, *_ = _case(2, torch.bfloat16, 0)
    kwargs["qkv"][0, 0] = float("nan")
    kwargs["qkv"][0, (HQ + 2 * HKV) * D] = float("nan")
    kwargs["qkv"][0, -D] = float("nan")
    ops.fused_minimax_m3_qknorm_rope_kv_insert(**kwargs)
    assert torch.isnan(kwargs["q_fp8_out"][0, :D].float()).all()
    assert torch.isnan(kwargs["index_q_out"][0, :D].float()).all()
    assert torch.isnan(kwargs["index_cache"][0, 0].float()).all()


@pytest.mark.parametrize(
    "change,message",
    [
        ({"index_world_size": 1}, "TP2/Q32/KV2/I4/P128/R64"),
        ({"index_rows_per_rank": 128}, "TP2/Q32/KV2/I4/P128/R64"),
        ({"index_rank": 2}, "TP2/Q32/KV2/I4/P128/R64"),
        ({"enable_pdl": True}, "enable_pdl=False"),
        ({"q_fp8_scale": 0.5}, "unit q_fp8_scale"),
    ],
)
def test_icp_native_admission_rejects_before_writing_cache(change, message):
    kwargs, storage, *_ = _case(2, torch.bfloat16, 0)
    before = storage.clone()
    kwargs.update(change)
    with pytest.raises(RuntimeError, match=message):
        ops.fused_minimax_m3_qknorm_rope_kv_insert(**kwargs)
    torch.testing.assert_close(storage, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name",
    [
        "qkv",
        "q_norm_weight",
        "k_norm_weight",
        "index_q_norm_weight",
        "index_k_norm_weight",
        "q_out",
        "q_fp8_out",
        "index_q_out",
        "kv_cache",
    ],
)
def test_icp_rejects_misaligned_offset_views_before_any_store(name):
    """Contiguity alone does not make packed loads and stores aligned."""
    kwargs, storage, *_ = _case(2, torch.bfloat16, 0)
    before = storage.clone()
    value = kwargs[name]
    shifted = torch.empty(value.numel() + 1, dtype=value.dtype, device="cuda")[1:]
    shifted = shifted.view(value.shape)
    shifted.copy_(value)
    kwargs[name] = shifted
    with pytest.raises(RuntimeError, match="aligned"):
        ops.fused_minimax_m3_qknorm_rope_kv_insert(**kwargs)
    torch.testing.assert_close(storage, before, rtol=0, atol=0)
