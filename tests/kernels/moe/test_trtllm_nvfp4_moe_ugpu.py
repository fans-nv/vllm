# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""uGPU localization of the TRTLLM-Gen NVFP4 MoE: shard layout, admission,
dispatch, reload and residency.

Host tests replace `flashinfer_ugpu` and the FlashInfer operators with
recorders. GPU tests need FlashInfer on CUDA; the `real_runtime` tests also
need a `flashinfer_ugpu` locality runtime.
"""

import contextlib
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.config
import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts import trtllm_nvfp4_moe
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsModular,
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.quantization.utils import (
    flashinfer_fp4_moe_ugpu as ugpu,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_nvfp4_moe_layer_for_fi_or_cutlass,
    prepare_static_weights_for_trtllm_fp4_moe,
    reorder_w1w3_to_w3w1,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

H = 6144
# Rank-local intermediate sizes at MoE TP2 and TP4.
INTERMEDIATE = [pytest.param(1536, id="tp2"), pytest.param(768, id="tp4")]
E = 4
TOPK = 4
UGPU_ROUTED_OP = "trtllm_fp4_block_scale_routed_moe_ugpu"
UGPU_MONOLITHIC_OP = "trtllm_fp4_block_scale_moe_ugpu"

SHARD_FIELDS = (
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm2_weights",
    "gemm2_weights_scale",
)
# Stock operator arguments that the shards replace.
WEIGHT_KWARGS = SHARD_FIELDS

requires_flashinfer_cuda = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and current_platform.has_device_capability(100)
        and has_flashinfer()
    ),
    reason="Requires FlashInfer on a Blackwell-family GPU",
)


def _checkpoint(i: int, e: int = E, device: str = "cpu") -> SimpleNamespace:
    """Random NVFP4 bytes in checkpoint order: w1 = gate, w3 = up."""
    g = torch.Generator().manual_seed(i)

    def rand(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g).to(device)

    return SimpleNamespace(
        gate=rand(e, i, H // 2),
        up=rand(e, i, H // 2),
        gate_scale=rand(e, i, H // 16),
        up_scale=rand(e, i, H // 16),
        w2=rand(e, H, i // 2),
        w2_scale=rand(e, H, i // 16),
    )


def _logical_w13(ck) -> tuple[torch.Tensor, torch.Tensor]:
    """w13 after vLLM's gate/up reorder: [up | gate]."""
    return reorder_w1w3_to_w3w1(
        torch.cat([ck.gate, ck.up], dim=1).contiguous(),
        torch.cat([ck.gate_scale, ck.up_scale], dim=1).contiguous(),
    )


@pytest.mark.parametrize("num_partitions", [2, 4])
@pytest.mark.parametrize("i", INTERMEDIATE)
def test_split_pairs_up_gate_channels(i: int, num_partitions: int):
    ck = _checkpoint(i)
    w13, w13_scale = _logical_w13(ck)
    channels, rows = i // num_partitions, H // num_partitions
    parts = [
        ugpu.split_nvfp4_moe_weights_for_ugpu(
            w13, w13_scale, ck.w2, ck.w2_scale, p, num_partitions
        )
        for p in range(num_partitions)
    ]
    for p, (fc1, fc1_scale, fc2, fc2_scale) in enumerate(parts):
        ch = slice(p * channels, (p + 1) * channels)
        assert torch.equal(fc1, torch.cat([ck.up[:, ch], ck.gate[:, ch]], dim=1))
        assert torch.equal(
            fc1_scale, torch.cat([ck.up_scale[:, ch], ck.gate_scale[:, ch]], dim=1)
        )
        # FC2 keeps the full K = I.
        assert fc2.shape == (E, rows, i // 2)
        assert torch.equal(fc2, ck.w2[:, p * rows : (p + 1) * rows])
        assert torch.equal(fc2_scale, ck.w2_scale[:, p * rows : (p + 1) * rows])
    up = torch.cat([fc1[:, :channels] for fc1, *_ in parts], dim=1)
    gate = torch.cat([fc1[:, channels:] for fc1, *_ in parts], dim=1)
    assert torch.equal(torch.cat([up, gate], dim=1), w13)


def _prepare_layer(layer, ck, per_token_activation: bool = False):
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

    ones = torch.ones(ck.w2.size(0), device=ck.w2.device)
    return prepare_nvfp4_moe_layer_for_fi_or_cutlass(
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        layer,
        torch.cat([ck.gate, ck.up], dim=1),
        torch.cat([ck.gate_scale, ck.up_scale], dim=1).view(torch.float8_e4m3fn),
        *(ones, ones, ck.w2, ck.w2_scale.view(torch.float8_e4m3fn), ones, ones),
        is_act_and_mul=True,
        per_token_activation=per_token_activation,
    )


def _layer(**overrides) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.activation = MoEActivation.SWIGLUOAI_UNINTERLEAVE
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            use_ep=overrides.get("ep", False),
            enable_eplb=overrides.get("eplb", False),
            ep_rank=0,
        ),
        has_bias=overrides.get("bias", False),
        hidden_dim=H,
        hidden_dim_unpadded=None,
        intermediate_size_per_partition=None,
    )
    return layer


def _install_ugpu(
    monkeypatch, runtime=None, *, module_error: Any = None, **ops
) -> types.ModuleType:
    module: Any = types.ModuleType("flashinfer_ugpu")
    module.get_runtime = lambda device: runtime
    module.last_error = lambda device: "driver error" if runtime is None else None
    module.UgpuMoeShards = SimpleNamespace
    for name, fn in ops.items():
        setattr(module, name, fn)
    moe: Any = types.ModuleType("flashinfer_ugpu.moe")

    def get_module():
        if module_error is not None:
            raise module_error

    moe.get_module = get_module
    module.moe = moe
    monkeypatch.setitem(sys.modules, "flashinfer_ugpu", module)
    monkeypatch.setitem(sys.modules, "flashinfer_ugpu.moe", moe)
    return module


@pytest.fixture
def admitted_env(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_UGPU_LOCALIZATION", "1")
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda c, device_id=0: c == 107
    )
    config = SimpleNamespace(model_config=SimpleNamespace(enable_sleep_mode=False))
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    return config


# Rejection reason expected for each case; None means admitted.
ADMISSION_CASES = {
    "tp2": None,
    "tp4": None,
    "unaligned": "not a multiple of 256",
    "per_token": "per-token",
    "ep": "expert parallelism",
    "eplb": "expert parallelism",
    "bias": "bias",
    "sleep": "sleep mode",
    "no_config": "no vLLM config",
    "not_sm107": "SM107",
    "no_package": "not importable",
    "no_runtime": "two-domain locality runtime .last error: driver error",
    "one_domain": "two-domain locality runtime",
    "no_op_module": "operator module failed to load .*no .so",
}


def test_defaults_are_stock_and_release(monkeypatch, admitted_env):
    monkeypatch.delenv("VLLM_MOE_UGPU_LOCALIZATION")
    monkeypatch.delenv("VLLM_MOE_UGPU_KEEP_FULL_WEIGHTS", raising=False)
    assert not envs.VLLM_MOE_UGPU_KEEP_FULL_WEIGHTS
    _install_ugpu(monkeypatch, SimpleNamespace(num_partitions=2))
    w13 = torch.empty((E, 2 * 1536, H // 2), dtype=torch.uint8)
    w2 = torch.empty((E, H, 1536 // 2), dtype=torch.uint8)
    assert ugpu.get_ugpu_moe_runtime(_layer(), w13, w2, True, False) is None


@pytest.mark.parametrize("case", ADMISSION_CASES)
def test_admission(monkeypatch, admitted_env, case: str):
    """An explicitly requested localization that cannot be admitted fails the
    load instead of serving the stock kernel."""
    runtimes = {"no_runtime": None, "one_domain": SimpleNamespace(num_partitions=1)}
    runtime = runtimes.get(case, SimpleNamespace(num_partitions=2))
    module_error = RuntimeError("no .so") if case == "no_op_module" else None
    _install_ugpu(monkeypatch, runtime, module_error=module_error)
    if case == "sleep":
        admitted_env.model_config.enable_sleep_mode = True
    if case == "no_config":
        monkeypatch.setattr(
            vllm.config, "get_current_vllm_config_or_none", lambda: None
        )
    if case == "not_sm107":
        monkeypatch.setattr(
            current_platform, "is_device_capability", lambda c, device_id=0: False
        )
    if case == "no_package":
        monkeypatch.setitem(sys.modules, "flashinfer_ugpu", None)
    i = {"tp4": 768, "unaligned": 1536 + 128}.get(case, 1536)
    w13 = torch.empty((E, 2 * i, H // 2), dtype=torch.uint8)
    w2 = torch.empty((E, H, i // 2), dtype=torch.uint8)
    layer = _layer(**{case: True})
    reason = ADMISSION_CASES[case]
    if reason is None:
        got = ugpu.get_ugpu_moe_runtime(layer, w13, w2, True, False)
        assert got is runtime
        return
    with pytest.raises(ValueError, match=reason):
        ugpu.get_ugpu_moe_runtime(layer, w13, w2, True, case == "per_token")


@pytest.mark.parametrize("keep_full", ["1", "0"])
@pytest.mark.parametrize("per_token", [False, True])
def test_prepare_layer_localizes_before_shuffle(
    monkeypatch, admitted_env, keep_full: str, per_token: bool
):
    """Shards are cut from the logical [up | gate] weights; release returns
    zero-width placeholders and per-token quantization fails before building
    shards."""
    from vllm.model_executor.layers.quantization.utils import flashinfer_fp4_moe

    monkeypatch.setenv("VLLM_MOE_UGPU_KEEP_FULL_WEIGHTS", keep_full)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    runtime = SimpleNamespace(num_partitions=2)
    _install_ugpu(monkeypatch, runtime)
    built = []

    def prepare_shards(w13, w13_scale, w2, w2_scale, rt):
        built.append((w13.clone(), rt))
        return SimpleNamespace()

    monkeypatch.setattr(ugpu, "prepare_ugpu_nvfp4_moe_shards", prepare_shards)
    monkeypatch.setattr(
        flashinfer_fp4_moe,
        "prepare_static_weights_for_trtllm_fp4_moe",
        lambda w13, w2, s13, s2, **kw: (w13, s13, w2, s2),
    )
    ck = _checkpoint(1536)
    layer = _layer()
    if per_token:
        with pytest.raises(ValueError, match="per-token"):
            _prepare_layer(layer, ck, per_token)
        assert not built and ugpu.get_ugpu_moe_localization(layer) is None
        return
    out = _prepare_layer(layer, ck, per_token)
    localization = ugpu.get_ugpu_moe_localization(layer)
    [(w13, rt)] = built
    assert rt is runtime and localization.runtime is runtime
    assert torch.equal(w13, torch.cat([ck.up, ck.gate], dim=1))
    assert localization.full_weights_released == (keep_full == "0")
    expected_rows = 0 if keep_full == "0" else 2 * 1536
    assert [out[k].size(1) for k in (0, 1)] == [expected_rows] * 2


def test_reload_rewrites_shards_in_place(monkeypatch):
    shards = SimpleNamespace(
        **{name: [torch.zeros(3), torch.zeros(3)] for name in SHARD_FIELDS}
    )
    layer = _layer()
    layer.ugpu_moe_localization = ugpu.UgpuMoeLocalization(object(), shards, True)
    old = [t for name in SHARD_FIELDS for t in getattr(shards, name)]
    monkeypatch.setattr(
        ugpu,
        "_shuffled_partitions",
        lambda *args: iter([(torch.full((3,), p + 1.0),) * 4 for p in range(2)]),
    )
    assert ugpu.localize_nvfp4_moe_weights(layer, object(), *[None] * 4)
    assert ugpu.get_ugpu_moe_localization(layer).shards is shards
    for name in SHARD_FIELDS:
        for p, t in enumerate(getattr(shards, name)):
            assert t is old[SHARD_FIELDS.index(name) * 2 + p]
            assert torch.all(t == p + 1)


def test_kernel_format_reload_is_rejected():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    model = torch.nn.Sequential(_layer())
    model[0].ugpu_moe_localization = ugpu.UgpuMoeLocalization(None, None, True)
    runner = SimpleNamespace(get_model=lambda: model, lora_config=None)
    with pytest.raises(NotImplementedError, match="checkpoint_format=True"):
        GPUModelRunner.reload_weights(
            runner, weights_iterator=iter([]), is_checkpoint_format=False
        )


def _moe_config(e: int, routing: RoutingMethodType, **kwargs) -> FusedMoEConfig:
    """MiniMax-M3-shaped MoE config at MoE TP2."""
    return FusedMoEConfig(
        num_experts=e,
        experts_per_token=TOPK,
        hidden_dim=H,
        intermediate_size=1536,
        num_local_experts=e,
        num_logical_experts=e,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        routing_method=routing,
        swiglu_limit=7.0,
        **({"device": "cpu"} | kwargs),
    )


def _make_experts(monkeypatch, experts_cls, localization=None, deferred=False, **q):
    """Construct the experts and run their post-load hook on a fake layer."""
    e, device = q.get("e", E), q.get("device", "cpu")
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: device)
    # Deferred finalize needs TP > 1 to take effect.
    monkeypatch.setattr(
        FusedMoEConfig,
        "use_deferred_moe_finalize",
        property(lambda self: self.defer_moe_finalize),
    )
    moe_config = _moe_config(
        e,
        q.get("routing", RoutingMethodType.Renormalize),
        device=device,
        max_num_tokens=8192,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        defer_moe_finalize=deferred,
    )
    moe_config.intermediate_size_per_partition = 1536
    ones = torch.ones(e, device=device)
    no_scale = torch.zeros(1, dtype=torch.uint8)
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=q.get("g1_alphas", ones),
        g2_alphas=q.get("g2_alphas", ones),
        a1_gscale=ones,
        a2_gscale=ones,
        w1_scale=q.get("w1_scale", no_scale),
        w2_scale=q.get("w2_scale", no_scale),
        is_scale_swizzled=False,
    )
    experts = experts_cls(moe_config=moe_config, quant_config=quant_config)
    layer = _layer()
    layer.w13_weight_scale_2 = quant_config.g1_alphas
    layer.w2_weight_scale_2 = quant_config.g2_alphas
    layer.w13_input_scale = layer.w2_input_scale = ones
    if localization is not None:
        layer.ugpu_moe_localization = localization
    experts.process_weights_after_loading(layer)
    return experts


def _fake_localization() -> ugpu.UgpuMoeLocalization:
    return ugpu.UgpuMoeLocalization(object(), SimpleNamespace(), True)


class _Recorder:
    def __init__(self, result=None):
        self.calls: list[dict] = []
        self.result = result

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _dispatch(
    monkeypatch,
    experts_cls,
    localized: bool,
    deferred: bool,
    tokens=3,
    install_ugpu: bool = True,
):
    flashinfer = pytest.importorskip("flashinfer")
    stock, local = _Recorder(), _Recorder()
    if experts_cls is TrtLlmNvFp4ExpertsModular:
        stock_name, local_name = "trtllm_fp4_block_scale_routed_moe", UGPU_ROUTED_OP
    else:
        stock_name, local_name = "trtllm_fp4_block_scale_moe", UGPU_MONOLITHIC_OP
        # The stock op's return: [output], or the three deferred-finalize tensors.
        stock.result = local.result = (
            [
                torch.zeros((tokens * TOPK, H), dtype=torch.bfloat16),
                torch.zeros((tokens, TOPK)),
                torch.arange(tokens * TOPK, dtype=torch.int32),
            ]
            if deferred
            else [torch.zeros((tokens, H), dtype=torch.bfloat16)]
        )
    monkeypatch.setattr(flashinfer.fused_moe, stock_name, stock)
    if install_ugpu:
        _install_ugpu(monkeypatch, **{local_name: local})
    monkeypatch.setattr(trtllm_nvfp4_moe, "activation_to_flashinfer_int", lambda a: 7)
    localization = _fake_localization() if localized else None
    experts = _make_experts(monkeypatch, experts_cls, localization, deferred)
    common = dict(
        hidden_states=torch.zeros((tokens, H // 2), dtype=torch.uint8),
        w1=torch.zeros((E, 0 if localized else 3072, H // 2), dtype=torch.uint8),
        w2=torch.zeros((E, 0 if localized else H, 768), dtype=torch.uint8),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        global_num_experts=E,
        expert_map=None,
        a1q_scale=torch.zeros((tokens, H // 16), dtype=torch.uint8),
        apply_router_weight_on_input=False,
    )
    if experts_cls is TrtLlmNvFp4ExpertsModular:
        result = experts.apply(
            output=torch.empty((tokens, H), dtype=torch.bfloat16),
            topk_weights=torch.full((tokens, TOPK), 1.0 / TOPK),
            topk_ids=torch.zeros((tokens, TOPK), dtype=torch.int32),
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            **common,
        )
    else:
        result = experts.apply(
            router_logits=torch.zeros((tokens, E)),
            e_score_correction_bias=torch.zeros(E),
            routed_scaling_factor=2.0,
            **common,
        )
    return experts, stock, local, result


CASES = [
    (TrtLlmNvFp4ExpertsModular, False),
    (TrtLlmNvFp4ExpertsMonolithic, False),
    (TrtLlmNvFp4ExpertsMonolithic, True),
]


@pytest.mark.parametrize("experts_cls,deferred", CASES)
def test_dispatch_contract(monkeypatch, experts_cls, deferred):
    """Localized dispatch passes the stock arguments with the four weight
    tensors replaced by shards and runtime; default-off dispatch is stock."""
    calls = {}
    for localized in (False, True):
        experts, stock, local, result = _dispatch(
            monkeypatch, experts_cls, localized, deferred
        )
        assert not (stock if localized else local).calls
        [calls[localized]] = (local if localized else stock).calls
        call = calls[localized]
        assert call["gemm1_clamp_limit"] is experts.gemm1_clamp_limit
        assert call["do_finalize"] is not deferred
        assert isinstance(result, UnfinalizedMoEOutput) == deferred
    assert calls[True]["shards"] is experts.ugpu_moe.shards
    assert calls[True]["runtime"] is experts.ugpu_moe.runtime
    assert set(calls[False]) - set(calls[True]) == set(WEIGHT_KWARGS)
    assert set(calls[True]) - set(calls[False]) == {"shards", "runtime"}
    for k, v in calls[False].items():
        if k not in WEIGHT_KWARGS and not isinstance(v, (torch.Tensor, tuple)):
            assert calls[True][k] == v, k
    if experts_cls is TrtLlmNvFp4ExpertsModular:
        # Unpacked precomputed routing: (int32 expert ids, weights).
        for call in calls.values():
            topk_ids, topk_weights = call["topk_ids"]
            assert topk_ids.dtype == torch.int32
            assert topk_ids.shape == topk_weights.shape


class _ImportRecorder:
    """Meta-path finder that records any attempt to import flashinfer_ugpu."""

    def __init__(self):
        self.attempts: list[str] = []

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "flashinfer_ugpu":
            self.attempts.append(name)
        return None


def test_default_off_binds_stock_ops_without_importing_ugpu(monkeypatch, admitted_env):
    """With both switches unset, a layer that would otherwise be admitted is
    prepared and dispatched by the stock FlashInfer path, and flashinfer_ugpu
    is never imported."""
    import subprocess

    from vllm.model_executor.layers.quantization.utils import flashinfer_fp4_moe

    monkeypatch.delenv("VLLM_MOE_UGPU_LOCALIZATION")
    monkeypatch.delenv("VLLM_MOE_UGPU_KEEP_FULL_WEIGHTS", raising=False)
    assert envs.VLLM_MOE_UGPU_LOCALIZATION is False
    assert envs.VLLM_MOE_UGPU_KEEP_FULL_WEIGHTS is False
    for name in ("flashinfer_ugpu", "flashinfer_ugpu.moe"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    guard = _ImportRecorder()
    monkeypatch.setattr(sys, "meta_path", [guard, *sys.meta_path])

    stock_shuffles = []

    def stock_shuffle(w13, w2, s13, s2, **kw):
        stock_shuffles.append(kw)
        return w13, s13, w2, s2

    monkeypatch.setattr(
        flashinfer_fp4_moe, "prepare_static_weights_for_trtllm_fp4_moe", stock_shuffle
    )
    layer = _layer()
    out = _prepare_layer(layer, _checkpoint(1536))
    assert len(stock_shuffles) == 1
    assert ugpu.get_ugpu_moe_localization(layer) is None
    assert [out[k].size(1) for k in (0, 1)] == [2 * 1536] * 2

    for experts_cls, deferred in CASES:
        experts, stock, local, _ = _dispatch(
            monkeypatch, experts_cls, False, deferred, install_ugpu=False
        )
        assert experts.ugpu_moe is None and experts._ugpu_moe_fn is None
        [call] = stock.calls
        assert set(WEIGHT_KWARGS) <= set(call)
        assert not {"shards", "runtime"} & set(call)

    assert not guard.attempts
    assert "flashinfer_ugpu" not in sys.modules

    # A fresh interpreter importing the MoE modules does not load it either.
    script = (
        "import sys\n"
        "import vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe\n"
        "import vllm.model_executor.layers.quantization.utils."
        "flashinfer_fp4_moe_ugpu\n"
        "assert 'flashinfer_ugpu' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("localize", ["1", "0"])
def test_pre_processed_weights_are_rejected(monkeypatch, localize: str):
    """Weight-cache (pre-processed) weights skip the conversion that admits a
    layer, so a requested localization must fail instead of serving stock."""
    from vllm.model_executor.utils import weights_already_processed

    monkeypatch.setenv("VLLM_MOE_UGPU_LOCALIZATION", localize)
    with weights_already_processed():
        if localize == "0":
            _make_experts(monkeypatch, TrtLlmNvFp4ExpertsModular)
            return
        with pytest.raises(ValueError, match="pre-processed"):
            _make_experts(monkeypatch, TrtLlmNvFp4ExpertsModular)


def test_released_weights_problem_size(monkeypatch):
    _install_ugpu(monkeypatch, **{UGPU_ROUTED_OP: _Recorder()})
    experts = _make_experts(
        monkeypatch, TrtLlmNvFp4ExpertsModular, _fake_localization()
    )
    w1 = torch.empty((E, 0, H // 2), dtype=torch.uint8)
    topk_ids = torch.zeros((3, TOPK), dtype=torch.int32)
    a1 = torch.empty((3, H // 2), dtype=torch.uint8)
    assert experts.moe_problem_size(a1, w1, w1, topk_ids)[2] == 2 * 1536


def _m3_routing_method() -> RoutingMethodType:
    """The routing type of MiniMax-M3's router: sigmoid + bias, renormalize,
    routed_scaling_factor 2.0."""
    from vllm.model_executor.layers.fused_moe.router.router_factory import (
        create_fused_moe_router,
    )

    router = create_fused_moe_router(
        top_k=TOPK,
        global_num_experts=128,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.0,
        e_score_correction_bias=torch.zeros(128),
    )
    return router.routing_method_type


@pytest.mark.parametrize(
    "routing,experts_cls,op",
    [
        # MiniMax-M3 once #58880 maps scaled sigmoid + bias to MiniMax2.
        (None, TrtLlmNvFp4ExpertsMonolithic, UGPU_MONOLITHIC_OP),
        (RoutingMethodType.Unspecified, TrtLlmNvFp4ExpertsModular, UGPU_ROUTED_OP),
    ],
)
def test_routing_selects_experts_and_localized_op(
    monkeypatch, routing, experts_cls, op
):
    """M3 routing selects the monolithic class and the localized monolithic op;
    Unspecified routing keeps the modular class and the localized routed op."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        select_nvfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )

    if routing is None:
        routing = _m3_routing_method()
        assert routing == RoutingMethodType.MiniMax2
    for cls in (TrtLlmNvFp4ExpertsModular, TrtLlmNvFp4ExpertsMonolithic):
        monkeypatch.setattr(cls, "_supports_current_device", staticmethod(lambda: True))
    config = _moe_config(
        128, routing, router_logits_dtype=torch.float32, moe_backend="flashinfer_trtllm"
    )
    backend, selected = select_nvfp4_moe_backend(config, kNvfp4Static, kNvfp4Dynamic)
    assert backend == NvFp4MoeBackend.FLASHINFER_TRTLLM
    assert selected is experts_cls
    recorder = _Recorder()
    _install_ugpu(monkeypatch, **{op: recorder})
    experts = _make_experts(
        monkeypatch, selected, _fake_localization(), routing=routing
    )
    assert experts.ugpu_moe is not None
    assert experts._ugpu_moe_fn is recorder


@pytest.mark.parametrize("localized", [True, False])
def test_autotune_warmup_tunes_localized_op(monkeypatch, tmp_path, localized):
    """The vLLM FlashInfer autotune warmup must reach the localized operator
    under autotune(True) with the serving shards and arguments."""
    import vllm.distributed.parallel_state as parallel_state
    import vllm.utils.flashinfer as fi_utils
    from vllm.model_executor.warmup import kernel_warmup

    tuning = {"on": False}

    @contextlib.contextmanager
    def autotune(tune_mode=True, skip_ops=None, tuning_buckets=None):
        tuning["on"] = tune_mode
        yield
        tuning["on"] = False

    tuner = SimpleNamespace(load_configs=lambda path: None, save_configs=lambda p: None)
    fake_autotuner: Any = types.ModuleType("flashinfer.autotuner")
    fake_autotuner.AutoTuner = SimpleNamespace(get=lambda: tuner)
    fake_autotuner.set_autotune_process_group = lambda group: None
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake_autotuner)
    monkeypatch.setattr(fi_utils, "autotune", autotune)
    world = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        cpu_group=None,
        broadcast_object=lambda obj, src: obj,
        barrier=lambda: None,
    )
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: world)
    monkeypatch.setattr(
        kernel_warmup, "resolve_flashinfer_autotune_file", lambda r: tmp_path / "a"
    )
    monkeypatch.delenv("VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS", raising=False)

    tuned: list[tuple[bool, dict]] = []

    def run(num_tokens, **kwargs):
        _, stock, local, _ = _dispatch(
            monkeypatch, TrtLlmNvFp4ExpertsModular, localized, False, num_tokens
        )
        tuned.extend((tuning["on"], call) for call in local.calls)

    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        vllm_config=SimpleNamespace(
            attention_config=SimpleNamespace(hisparse_config=None),
            cache_config=SimpleNamespace(use_replayssm=False),
            kernel_config=SimpleNamespace(linear_backend="auto"),
        ),
        get_model=lambda: torch.nn.Module(),
        _dummy_run=run,
    )
    kernel_warmup.flashinfer_autotune(runner)
    if not localized:
        assert not tuned
        return
    [(tune_mode, call)] = tuned
    assert tune_mode and call["hidden_states"].shape[0] == 64
    assert call["do_finalize"] is True and "enable_pdl" not in call
    assert call["tune_max_num_tokens"] == 8192


def test_localization_switches_change_compile_factors(monkeypatch):
    """Localized arms get their own autotune cache directory."""

    def factors(localize: str, keep: str) -> str:
        monkeypatch.setenv("VLLM_MOE_UGPU_LOCALIZATION", localize)
        monkeypatch.setenv("VLLM_MOE_UGPU_KEEP_FULL_WEIGHTS", keep)
        return str(envs.compile_factors())

    assert len({factors("0", "1"), factors("1", "1"), factors("1", "0")}) == 3


@requires_flashinfer_cuda
@pytest.mark.parametrize("i", INTERMEDIATE)
@torch.inference_mode()
def test_shard_shuffle_is_row_slice_of_full_shuffle(i: int):
    """Each shard's GEN shuffle equals its rows of the full-tensor shuffle; the
    coupled full-tensor helper cannot produce the FC2 shard shape."""
    ck = _checkpoint(i, device="cuda")
    w13, w13_scale = _logical_w13(ck)
    full = prepare_static_weights_for_trtllm_fp4_moe(
        w13, ck.w2, w13_scale, ck.w2_scale, H, i, E, True
    )
    parts = list(ugpu._shuffled_partitions(w13, w13_scale, ck.w2, ck.w2_scale, 2))
    for full_t, rows, shard_ts in zip(full, (i, i, H // 2, H // 2), zip(*parts)):
        full_bytes = full_t.view(torch.uint8)
        for p, shard in enumerate(shard_ts):
            expected = full_bytes[:, p * rows : (p + 1) * rows]
            assert torch.equal(shard.view(torch.uint8), expected)

    fc1, fc1_scale, fc2, fc2_scale = ugpu.split_nvfp4_moe_weights_for_ugpu(
        w13, w13_scale, ck.w2, ck.w2_scale, 0, 2
    )
    coupled = prepare_static_weights_for_trtllm_fp4_moe(
        fc1, fc2, fc1_scale, fc2_scale, H, i // 2, E, True
    )
    assert coupled[2].shape != parts[0][2].shape == (E, H // 2, i // 2)


@pytest.fixture
def real_runtime():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    flashinfer_ugpu = pytest.importorskip("flashinfer_ugpu")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    runtime = flashinfer_ugpu.get_runtime(device)
    if runtime is None:
        pytest.skip(f"No locality runtime: {flashinfer_ugpu.last_error(device)}")
    return runtime


def _domains(runtime, t: torch.Tensor) -> tuple[int, int]:
    flat = t.view(torch.uint8).reshape(-1)
    return runtime.pointer_domain(flat[:1]), runtime.pointer_domain(flat[-1:])


@pytest.mark.parametrize("keep_full", ["1", "0"])
@torch.inference_mode()
def test_real_runtime_placement_and_residency(
    monkeypatch, admitted_env, real_runtime, keep_full
):
    """TP2 at 128 experts: every shard byte range lies in its partition's
    domain, and release leaves only the shards allocated."""
    monkeypatch.setenv("VLLM_MOE_UGPU_KEEP_FULL_WEIGHTS", keep_full)
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    allocated_before = torch.accelerator.memory_allocated()
    ck = _checkpoint(1536, e=128, device="cuda")
    layer = _layer()
    out = _prepare_layer(layer, ck)
    del ck
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()

    shards = ugpu.get_ugpu_moe_localization(layer).shards
    shard_bytes = 0
    for name in SHARD_FIELDS:
        for p, t in enumerate(getattr(shards, name)):
            domain = real_runtime.domain_ids[p]
            assert _domains(real_runtime, t) == (domain, domain), (name, p)
            shard_bytes += t.nbytes
    ordinary = torch.empty(16, dtype=torch.uint8, device="cuda")
    assert _domains(real_runtime, ordinary) == (-1, -1)
    full_bytes = sum(out[k].nbytes for k in (0, 1, 4, 5))
    assert full_bytes == (shard_bytes if keep_full == "1" else 0)
    allocated = torch.accelerator.memory_allocated() - allocated_before
    assert shard_bytes + full_bytes <= allocated < shard_bytes + full_bytes + (8 << 20)


M3_ROUTED_SCALE = 2.0


def _real_problem(runtime, num_tokens: int) -> SimpleNamespace:
    """M3 TP2 shapes at 128 experts: full stock weights, their localized shards
    (and the two partitions swapped), NVFP4 input and M3-style router inputs."""
    import flashinfer
    from flashinfer_ugpu import UgpuMoeShards
    from flashinfer_ugpu.moe import get_module

    try:
        get_module()
    except Exception as err:
        pytest.skip(f"flashinfer_ugpu operator module unavailable: {err}")

    e, i = 128, 1536
    torch.manual_seed(num_tokens)
    ck = _checkpoint(i, e=e, device="cuda")
    for name in ("gate_scale", "up_scale", "w2_scale"):
        valid = torch.empty(getattr(ck, name).shape, device="cuda").uniform_(0.5, 2)
        setattr(ck, name, valid.to(torch.float8_e4m3fn).view(torch.uint8))
    w13, w13_scale = _logical_w13(ck)
    full = prepare_static_weights_for_trtllm_fp4_moe(
        w13, ck.w2, w13_scale, ck.w2_scale, H, i, e, True
    )
    shards = ugpu.prepare_ugpu_nvfp4_moe_shards(
        w13, w13_scale, ck.w2, ck.w2_scale, runtime
    )
    swapped = UgpuMoeShards(
        **{name: list(reversed(getattr(shards, name))) for name in SHARD_FIELDS}
    )
    x = torch.randn((num_tokens, H), device="cuda", dtype=torch.bfloat16)
    x_gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
    x_fp4, x_sf = flashinfer.fp4_quantize(x, x_gs, 16, False, False)
    return SimpleNamespace(
        e=e,
        i=i,
        num_tokens=num_tokens,
        full=full,
        shards=shards,
        swapped=swapped,
        x=x,
        x_fp4=x_fp4,
        x_sf=x_sf,
        quant=dict(
            e=e,
            device="cuda",
            g1_alphas=torch.full((e,), 1.0 / x_gs.item(), device="cuda") / H,
            g2_alphas=torch.full((e,), 1.0 / i, device="cuda"),
            w1_scale=full[1],
            w2_scale=full[3],
        ),
        # M3 router: fp32 logits and an fp32 score-correction bias.
        router_logits=torch.randn((num_tokens, e), device="cuda"),
        bias=0.1 * torch.randn(e, device="cuda"),
    )


def _run_experts(monkeypatch, p, experts_cls, arm_shards=None, runtime=None, **inputs):
    """One stock (arm_shards=None) or localized forward of vLLM's experts; the
    localized arm runs on zero-width w1/w2 and must call its op exactly once."""
    localization = None
    w1, w2 = p.full[0], p.full[2]
    if arm_shards is not None:
        localization = ugpu.UgpuMoeLocalization(runtime, arm_shards, True)
        w1, w2 = (t.new_empty((p.e, 0, t.size(2))) for t in (w1, w2))
    routing = (
        RoutingMethodType.MiniMax2
        if experts_cls is TrtLlmNvFp4ExpertsMonolithic
        else RoutingMethodType.Renormalize
    )
    experts = _make_experts(
        monkeypatch, experts_cls, localization, routing=routing, **p.quant
    )
    calls: list[int] = []
    if localization is not None:
        op = experts._ugpu_moe_fn
        assert op is not None

        def spy(op=op, calls=calls, **kwargs):
            calls.append(1)
            return op(**kwargs)

        experts._ugpu_moe_fn = spy
    common = dict(
        hidden_states=p.x_fp4,
        w1=w1,
        w2=w2,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        global_num_experts=p.e,
        expert_map=None,
        a1q_scale=p.x_sf,
        apply_router_weight_on_input=False,
    )
    if experts_cls is TrtLlmNvFp4ExpertsMonolithic:
        out = experts.apply(
            router_logits=p.router_logits,
            e_score_correction_bias=p.bias,
            routed_scaling_factor=M3_ROUTED_SCALE,
            **common,
        )
    else:
        out = torch.full((p.num_tokens, H), float("nan"), device="cuda").bfloat16()
        experts.apply(
            output=out,
            topk_weights=inputs["topk_weights"],
            topk_ids=inputs["topk_ids"].to(torch.int32).contiguous(),
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            **common,
        )
    assert len(calls) == (localization is not None)
    return out


def _check_parity(stock, local, swapped):
    assert torch.isfinite(stock).all() and stock.abs().sum() > 0
    stock, local, swapped = (t.view(torch.int16) for t in (stock, local, swapped))
    assert torch.equal(stock, local)
    assert (stock != swapped).float().mean() > 0.5


@pytest.mark.parametrize("num_tokens", [1, 64, 257])
@torch.inference_mode()
def test_real_modular_dispatch_matches_stock(monkeypatch, real_runtime, num_tokens):
    """The vLLM modular experts on released, localized shards reproduce the stock
    routed kernel on the full weights bit for bit; swapping the two partitions'
    shards must not."""
    p = _real_problem(real_runtime, num_tokens)
    topk_ids = torch.rand((num_tokens, p.e), device="cuda").topk(TOPK, dim=-1)[1]
    routes = dict(
        topk_weights=torch.full((num_tokens, TOPK), 1.0 / TOPK, device="cuda"),
        topk_ids=topk_ids,
    )
    outs = [
        _run_experts(
            monkeypatch, p, TrtLlmNvFp4ExpertsModular, s, real_runtime, **routes
        )
        for s in (None, p.shards, p.swapped)
    ]
    _check_parity(*outs)


@pytest.mark.parametrize("num_tokens", [1, 64, 257])
@torch.inference_mode()
def test_real_monolithic_dispatch_matches_stock(monkeypatch, real_runtime, num_tokens):
    """M3 after #58880: vLLM's monolithic experts with MiniMax2 routing and
    routed_scaling_factor 2.0 on released, localized shards reproduce the stock
    FlashInfer monolithic op bit for bit; swapping the partitions must not."""
    p = _real_problem(real_runtime, num_tokens)
    outs = [
        _run_experts(monkeypatch, p, TrtLlmNvFp4ExpertsMonolithic, s, real_runtime)
        for s in (None, p.shards, p.swapped)
    ]
    _check_parity(*outs)


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).norm()
    return (diff / expected.float().norm()).item()


def _norm_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float().norm() / expected.float().norm()).item()


@pytest.mark.parametrize("num_tokens", [1, 64, 257])
@torch.inference_mode()
def test_real_minimax2_localized_op_matches_stock(
    monkeypatch, real_runtime, num_tokens
):
    """Operator level, with the exact arguments vLLM passes for M3 (MiniMax2,
    routed_scaling_factor 2.0): the localized monolithic op equals the same
    module's single-stream stock mode bit for bit, and agrees with the image
    FlashInfer trtllm_fp4_block_scale_moe."""
    import flashinfer
    import flashinfer_ugpu
    from flashinfer_ugpu.moe import choose_tactic

    p = _real_problem(real_runtime, num_tokens)
    image_op = flashinfer.fused_moe.trtllm_fp4_block_scale_moe
    captured: list[dict] = []

    def capture(**kwargs):
        captured.append(kwargs)
        return image_op(**kwargs)

    monkeypatch.setattr(flashinfer.fused_moe, "trtllm_fp4_block_scale_moe", capture)
    image = _run_experts(monkeypatch, p, TrtLlmNvFp4ExpertsMonolithic)
    [kwargs] = captured
    assert kwargs["routing_method_type"] == RoutingMethodType.MiniMax2
    assert kwargs["routed_scaling_factor"] == M3_ROUTED_SCALE
    args = {k: v for k, v in kwargs.items() if k not in WEIGHT_KWARGS}
    full_shards = flashinfer_ugpu.UgpuMoeShards(
        **{name: [kwargs[name]] for name in SHARD_FIELDS}
    )
    tactic = choose_tactic(num_tokens, TOPK, H, p.i, p.e, list(real_runtime.sm_counts))
    op = flashinfer_ugpu.trtllm_fp4_block_scale_moe_ugpu
    local = op(**args, shards=p.shards, runtime=real_runtime, tactic=tactic)[0]
    same = op(**args, shards=full_shards, runtime=real_runtime, tactic=tactic, _mode=1)[
        0
    ]
    assert torch.isfinite(local).all() and local.abs().sum() > 0
    assert torch.equal(local.view(torch.int16), same.view(torch.int16))
    ratio = _norm_ratio(local, image)
    assert 0.9 < ratio < 1.1, f"localized/image output norm ratio {ratio:.3f}"
    assert _rel_err(local, image) < 0.05


@pytest.mark.parametrize("num_tokens", [1, 64, 257])
@torch.inference_mode()
def test_real_minimax2_routed_scale_matches_reference(
    monkeypatch, real_runtime, num_tokens
):
    """The stock monolithic op with MiniMax2 routing must apply
    routed_scaling_factor 2.0: it matches vLLM's fused_topk_bias routing (scale
    2.0) through the stock modular path, and not the unscaled reference."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )

    p = _real_problem(real_runtime, num_tokens)
    mono = _run_experts(monkeypatch, p, TrtLlmNvFp4ExpertsMonolithic)

    def reference(scale: float) -> torch.Tensor:
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=p.x,
            gating_output=p.router_logits,
            scoring_func="sigmoid",
            e_score_correction_bias=p.bias,
            topk=TOPK,
            renormalize=True,
            routed_scaling_factor=scale,
        )
        return _run_experts(
            monkeypatch,
            p,
            TrtLlmNvFp4ExpertsModular,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    scaled, unscaled = reference(M3_ROUTED_SCALE), reference(1.0)
    assert torch.isfinite(mono).all() and mono.abs().sum() > 0
    # Control: the reference itself responds to the scale.
    assert abs(_norm_ratio(unscaled, scaled) - 1 / M3_ROUTED_SCALE) < 0.01
    ratio = _norm_ratio(mono, scaled)
    assert 0.9 < ratio < 1.1, (
        f"monolithic/reference output norm ratio {ratio:.3f}; about 0.5 means "
        "this FlashInfer build ignores routed_scaling_factor for NVFP4 MiniMax2"
    )
    assert _rel_err(mono, scaled) < 0.05
    assert _rel_err(mono, unscaled) > 0.4
