# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 TP2 indexer over rank-local index-key fragments.

Pure decode uses the CuTe H4 scorer; any batch containing prefill uses FMHA
OnlyScoreIcp for every row. Bounded score windows produce local candidates,
then one exchange per sparse layer merges them into the shared top-k buffer.
Native msa_icp imports stay lazy so this module is import-safe without it.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.compound_page import (
    INDEX_HEAD_DIM,
)
from vllm.models.minimax_m3.common.indexer_icp import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
)
from vllm.models.minimax_m3.nvidia.msa_icp import require_icp_writer, require_msa_icp
from vllm.models.minimax_m3.nvidia.ops.icp_dispatch import (
    CAND_K as _DISPATCH_CAND_K,
)
from vllm.models.minimax_m3.nvidia.ops.icp_dispatch import (
    FUSED_DECODE_ENV,
    ICP_PREFILL_EXTENT_STEPS_PER_OCTAVE,
    admitted_exchange_extents,
    exchange_token_capacities,
    fused_decode_requested,
    prefill_exchange_token_capacities,
    select_token_capacity,
)
from vllm.models.minimax_m3.nvidia.ops.icp_dispatch import (
    QCAPACITY_ALIGNMENT as _DISPATCH_QCAPACITY_ALIGNMENT,
)
from vllm.models.minimax_m3.nvidia.ops.icp_metadata import ICPLiveMetadata
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from fmha_sm100.icp.local_indexer import ICPRowWindow

logger = init_logger(__name__)

# Page size == sparse block size == index-K block; fmha tile id == M3 block id.
PAGE_SIZE = 128

# ---- refined-icp-v1: fragment-aware scoring at W > 1 ------------------------
# Rank r of W stores R = 128 // W index-key rows of EVERY logical 128-token
# block, so the block scores it produces are PARTIAL maxima; the true block
# score is the maximum across ranks, recovered by merging 8-byte candidate
# records. Everything below produces this rank's candidates; the exchange
# resolves them.
#
# The INNER MSA tile, in query rows. `fmha_sm100.api._qo_tile_size_for` pins the
# Q tile to 128 whenever ``icp_c > 1`` and the ICP arm re-asserts it at launch:
# at qo_tile 256 one compute tile spans two compound pages and the loader
# gathers only one. This constant names that pin; it is NOT the outer chunk.
ICP_MSA_QUERY_TILE_TOKENS = 128

# The OUTER chunk -- query rows per bounded scoring/selection call -- is NOT a
# constant. The score wave is ``[chunk, H_group, N]`` with
# ``N = cdiv(max_model_len, 128)``, so the chunk is what keeps the scratch
# bounded: a full ``[T, H_group, N]`` query-by-context matrix is forbidden
# (ABI W3). A prefill call contains many internal Q128 tiles, so the outer size
# follows memory and parallelism rather than the inner tile;
# `icp_outer_chunk_tokens` derives it per deployment from the scratch budget
# below, floored at the inner tile and rounded to a multiple of it.

# Device bytes the DECODE score scratch may occupy: one fp32 score plane plus
# one uint8 validity plane, both ``[chunk, H_group, N]`` (r13's sizing; 128 rows
# at 1M context). The chunk is a startup constant, which is what the captured
# decode profiles (`icp_select_extents`, `icp_decode_plans`,
# `icp_decode_workspaces`) require.
ICP_SCORE_SCRATCH_BUDGET_BYTES = 32 * 1024 * 1024

# Hard ceiling on the derived outer chunk, independent of the budget: the outer
# chunk also sets the plan's work decomposition, so it is bounded by the
# qualified per-chunk query-length envelope and not by the memory budget alone.
ICP_MAX_OUTER_CHUNK_TOKENS = 1024

# ---- the PREFILL wave -------------------------------------------------------
# The eager prefill band sizes its score plane from the invocation's admitted
# LIVE column extent rather than from the capacity `N = cdiv(max_model_len,
# 128)`. Under a capacity-wide plane the per-row cost is fixed at the longest
# context the deployment admits, which drives the wave down to the 128-row
# inner-tile floor and serialises a prefill into one scorer+selector launch per
# floor-sized wave per sparse layer.
#
# Why a per-invocation extent is admissible here and is not for decode:
#
#  * The prefill band is EAGER. `cudagraph_decode_phase_only` is True, so any
#    prefill in the batch excludes FULL decode capture (a pure-decode
#    invocation may also run eagerly -- the implication is one-way). Nothing
#    about a prefill launch is baked into a graph, and the decode plane and
#    decode plans keep their capacity shapes untouched.
#  * It is RANK-INVARIANT. The extent comes from `seq_lens_cpu_upper_bound`,
#    scheduler metadata identical on every TP rank, and the phase comes from
#    `is_prefilling`, which C4 already requires to be identical. Two ranks
#    therefore resolve the same wave.
#  * The wave is query rows per scorer launch; the exchange EXTENT is candidate
#    rows per selector call and per exchange, has its own ladder
#    (`prefill_exchange_token_capacities`), and is reached once after every wave
#    has finished. Do not size one from the other.
#
# The admitted extents are a startup-enumerated LADDER, not a free per-step
# number: every (columns, wave) pair, its `PrefillPlan`, its score/validity view
# and its selector scratch view are built in `__init__`, and a build only picks
# a rung. Nothing is allocated, and no device value is read, on the submission
# path. :func:`icp_prefill_column_rungs` owns the fill and its trade.
ICP_PREFILL_COLUMN_FLOOR = 4
ICP_PREFILL_COLUMN_STEPS_PER_OCTAVE = 2

# Device bytes the PREFILL score + validity scratch may occupy, i.e. the size of
# the retained arena every rung's view is carved from. The rung the invocation
# lands on -- never its exact column count -- sizes the plane, so the reach this
# budget buys is quantised to the ladder; the un-quantised division is not the
# reach. The configuration-specific reach is in the column-rung report.
#
# It is allocated once per metadata builder per rank, NOT per sparse layer, and
# after the memory-profiling window has closed, so it comes out of the headroom
# between `gpu_memory_utilization` and 100% and not out of the KV pool. Re-check
# it at high utilization.
#
# If it cannot be allocated that is a STARTUP failure naming the bytes and the
# knob (`_icp_allocate_score_planes`), never a silent degradation to a
# floor-width wave.
ICP_PREFILL_SCORE_SCRATCH_BUDGET_BYTES = 1024 * 1024 * 1024

# Hard ceiling on the derived PREFILL wave, the counterpart of
# :data:`ICP_MAX_OUTER_CHUNK_TOKENS` for the eager band. Unlike that one it is
# not a measurement envelope: `_qo_tile_size_for` pins the INNER tile at 128
# whenever ``icp_c > 1`` regardless of the outer width, so a wider wave only
# adds whole Q128 tiles to one plan.
#
# What bounds it instead is the wave-invariance gate -- the final Top-16
# candidate set must be IDENTICAL across wave widths on frozen inputs
# in the kernel package, compared through an
# int32 view because an invalid record's id bits are NaN as a float -- plus the
# scheduler's token budget. r13's fixed ceiling.
ICP_MAX_PREFILL_WAVE_TOKENS = 16384

# `auto` (default): resolve the wave per invocation from the admitted live
# column extent, as above. An integer pins ONE wave width at FULL capacity
# columns. Pinning is a measurement control for the profiling lane, not a
# serving mode.
_PREFILL_WAVE_ENV = "VLLM_MINIMAX_ICP_PREFILL_WAVE"
# Override for :data:`ICP_PREFILL_SCORE_SCRATCH_BUDGET_BYTES`, in MiB.
_PREFILL_SCORE_MIB_ENV = "VLLM_MINIMAX_ICP_PREFILL_SCORE_MIB"
# Greppable startup markers for the resolved wave ladder and for the eager
# prefill exchange-extent ladder. They are separate quantities -- query rows per
# scorer launch versus candidate rows per selector call and per exchange -- and
# so must stay distinct literals. External capture harnesses grep these exact
# strings and count their occurrences; neither the marker text nor how often it
# is emitted may change.
_ICP_PREFILL_WAVE_MARKER = "ICP_PREFILL_WAVE_SCHEMA"
_ICP_PREFILL_EXTENT_MARKER = "ICP_PREFILL_EXTENT_SCHEMA"


def _icp_wave_from_budget(
    *,
    columns: int,
    heads_group: int,
    token_capacity: int,
    budget_bytes: int,
    max_wave: int,
) -> int:
    from fmha_sm100.icp.local_indexer import _icp_wave_from_budget as wave_from_budget

    return wave_from_budget(
        columns=columns,
        heads_group=heads_group,
        token_capacity=token_capacity,
        budget_bytes=budget_bytes,
        max_wave=max_wave,
        query_tile_tokens=ICP_MSA_QUERY_TILE_TOKENS,
    )


def icp_prefill_column_rungs(
    max_blocks: int,
    *,
    floor: int | None = None,
    steps_per_octave: int | None = None,
) -> tuple[int, ...]:
    from fmha_sm100.icp.local_indexer import icp_prefill_column_rungs as column_rungs

    if floor is None:
        floor = ICP_PREFILL_COLUMN_FLOOR
    if steps_per_octave is None:
        steps_per_octave = ICP_PREFILL_COLUMN_STEPS_PER_OCTAVE
    return column_rungs(max_blocks, floor=floor, steps_per_octave=steps_per_octave)


def _prefill_wave_override() -> int | None:
    """The pinned prefill wave, or ``None`` for the derived ladder."""
    import os  # noqa: PLC0415

    raw = os.environ.get(_PREFILL_WAVE_ENV, "auto").strip().lower()
    if raw in ("", "auto"):
        return None
    try:
        wave = int(raw)
    except ValueError:
        raise ValueError(
            f"{_PREFILL_WAVE_ENV}={raw!r} must be `auto` or an integer number "
            "of query rows"
        ) from None
    tile = ICP_MSA_QUERY_TILE_TOKENS
    if wave < tile or wave % tile or wave > ICP_MAX_PREFILL_WAVE_TOKENS:
        raise ValueError(
            f"{_PREFILL_WAVE_ENV}={wave} must be a multiple of the inner MSA "
            f"tile {tile} in [{tile}, {ICP_MAX_PREFILL_WAVE_TOKENS}]. The inner "
            "tile is pinned by `_qo_tile_size_for` at icp_c > 1 and a wave that "
            "is not a whole number of tiles would leave a partial tile in the "
            "last plan."
        )
    return wave


def _prefill_score_budget_bytes() -> int:
    """Prefill score + validity arena bytes, from the env or the constant."""
    import os  # noqa: PLC0415

    raw = os.environ.get(_PREFILL_SCORE_MIB_ENV, "").strip()
    if not raw:
        return ICP_PREFILL_SCORE_SCRATCH_BUDGET_BYTES
    try:
        mib = int(raw)
    except ValueError:
        raise ValueError(
            f"{_PREFILL_SCORE_MIB_ENV}={raw!r} must be an integer MiB count"
        ) from None
    if mib < 1:
        raise ValueError(f"{_PREFILL_SCORE_MIB_ENV}={mib} must be >= 1 MiB")
    return mib * 1024 * 1024


def _icp_allocate_score_planes(
    *,
    elems: int,
    device,
    budget_bytes: int,
    heads_group: int,
    columns: int,
):
    """The retained fp32 score arena and uint8 validity arena.

    The arena is a startup constant and every rung's view is carved from it, so
    there is no smaller shape to fall back to: an allocation that cannot be
    satisfied ends STARTUP with the bytes and the knobs named rather than
    silently degrading to a floor-width wave.

    Args:
        elems: Cells in each arena, i.e. ``max`` over every band's and every
            rung's plane.
        device: The allocation's device.
        budget_bytes: The configured prefill budget, for the message.
        heads_group: ``H_group``, for the message.
        columns: ``max_blocks_per_req``, for the message.

    Returns:
        ``(scores, valid)``, flat and contiguous.

    Raises:
        RuntimeError: If either arena cannot be allocated.

    """
    want = elems * (4 + 1)
    try:
        scores = torch.empty(elems, dtype=torch.float32, device=device)
        valid = torch.empty(elems, dtype=torch.uint8, device=device)
    except (RuntimeError, MemoryError) as exc:
        raise RuntimeError(
            "MiniMax-M3 ICP indexer: the prefill score scratch could not be "
            f"allocated. It needs {want} bytes ({want / 2**30:.2f} GiB) on "
            f"{device}: one fp32 score arena and one uint8 validity arena, each "
            f"{elems} cells, holding the widest of the prefill wave ladder's "
            f"[wave, {heads_group}, <= {columns}] planes under a "
            f"{budget_bytes // 2**20} MiB budget, once per rank and NOT per "
            "sparse layer. It is allocated after the memory-profiling window "
            "closes, so it comes out of the headroom between "
            "gpu_memory_utilization and 100% and not out of the KV pool: lower "
            "--gpu-memory-utilization, --max-num-batched-tokens or "
            f"--max-model-len, or lower the budget with {_PREFILL_SCORE_MIB_ENV}"
            " -- which BUYS FEWER one-launch columns and is a deliberate "
            "performance decision, not a workaround. Failing here is "
            "deliberate: a wave that silently collapses to one 128-row inner "
            "tile is the defect this replaces."
        ) from exc
    return scores, valid


def icp_outer_chunk_tokens(
    *,
    max_model_len: int,
    heads_group: int,
    token_capacity: int,
    physical_page_tokens: int = PAGE_SIZE,
    budget_bytes: int | None = None,
    max_chunk: int | None = None,
) -> int:
    from fmha_sm100.icp.local_indexer import icp_outer_chunk_tokens as outer_chunk

    if budget_bytes is None:
        budget_bytes = ICP_SCORE_SCRATCH_BUDGET_BYTES
    if max_chunk is None:
        max_chunk = ICP_MAX_OUTER_CHUNK_TOKENS
    return outer_chunk(
        max_model_len=max_model_len,
        heads_group=heads_group,
        token_capacity=token_capacity,
        physical_page_tokens=physical_page_tokens,
        budget_bytes=budget_bytes,
        max_chunk=max_chunk,
        query_tile_tokens=ICP_MSA_QUERY_TILE_TOKENS,
    )


def icp_decode_chunk_tokens(
    *,
    max_model_len: int,
    heads_group: int,
    token_capacity: int,
    physical_page_tokens: int = PAGE_SIZE,
) -> int:
    from fmha_sm100.icp.local_indexer import icp_decode_chunk_tokens as decode_chunk

    return decode_chunk(
        max_model_len=max_model_len,
        heads_group=heads_group,
        token_capacity=token_capacity,
        physical_page_tokens=physical_page_tokens,
        budget_bytes=ICP_SCORE_SCRATCH_BUDGET_BYTES,
        max_chunk=ICP_MAX_OUTER_CHUNK_TOKENS,
        query_tile_tokens=ICP_MSA_QUERY_TILE_TOKENS,
    )


def icp_prefill_wave_ladder(
    columns: tuple[int, ...],
    *,
    heads_group: int,
    token_capacity: int,
    budget_bytes: int,
) -> dict[int, int]:
    from fmha_sm100.icp.local_indexer import icp_prefill_wave_ladder as wave_ladder

    return wave_ladder(
        columns,
        heads_group=heads_group,
        token_capacity=token_capacity,
        budget_bytes=budget_bytes,
        max_wave=ICP_MAX_PREFILL_WAVE_TOKENS,
        query_tile_tokens=ICP_MSA_QUERY_TILE_TOKENS,
    )


def icp_prefill_rung(columns: tuple[int, ...], live_blocks: int) -> int:
    from fmha_sm100.icp.local_indexer import icp_prefill_rung as prefill_rung

    return prefill_rung(columns, live_blocks)


def icp_prefill_arena_elems(waves: dict[int, int], heads_group: int) -> int:
    from fmha_sm100.icp.local_indexer import icp_prefill_arena_elems as arena_elems

    return arena_elems(waves, heads_group)


# ---- CuTe decode scorer windows ---------------------------------------------
# MSA's launch ABI 2 takes the token window and request range as runtime launch
# scalars, so one compile per query length serves every window. The window is
# the band's rows from the chunk origin, which under a FULL graph is the PADDED
# extent (`num_actual_tokens`) and so a constant of the graph; the request range
# is the requests those rows can belong to, capped at the metadata's own
# request count, so it never names a slot the runner's views do not hold.


def icp_cute_decode_window(
    *, query_len: int, token_begin: int, rows: int, chunk: int, num_reqs: int
) -> tuple[int, int, int, int] | None:
    from fmha_sm100.icp.local_indexer import icp_cute_decode_window as decode_window

    return decode_window(
        query_len=query_len,
        token_begin=token_begin,
        rows=rows,
        chunk=chunk,
        num_reqs=num_reqs,
    )


def __getattr__(name: str):
    if name == "ICPRowWindow":
        from fmha_sm100.icp.local_indexer import ICPRowWindow

        return ICPRowWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def icp_row_windows(
    *,
    num_tokens: int,
    cap: int,
    cute_rows: int,
    decode_chunk: int,
    prefill_width: int,
    has_prefill: bool,
) -> list["ICPRowWindow"]:
    from fmha_sm100.icp.local_indexer import icp_row_windows as row_windows

    return row_windows(
        num_tokens=num_tokens,
        cap=cap,
        cute_rows=cute_rows,
        decode_chunk=decode_chunk,
        prefill_width=prefill_width,
        has_prefill=has_prefill,
    )


def _bind_cute_launch(
    scorer,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    active_rows: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
) -> Callable[..., None]:
    from fmha_sm100.icp.local_indexer import bind_cute_launch

    return bind_cute_launch(
        scorer,
        block_table,
        query_start_loc,
        seq_lens,
        positions,
        active_rows,
        scores,
        valid,
    )


def _bind_fmha_launch(
    fmha: Callable[..., None],
    plan: dict,
    row_begin: int,
    rows: int,
    **kwargs,
) -> Callable[..., None]:
    from fmha_sm100.icp.local_indexer import bind_fmha_launch

    return bind_fmha_launch(fmha, plan, row_begin, rows, **kwargs)


# Per-layer host validators -- the CuTe launch check, the chunk contract
# assertions and the invocation shape checks -- run on the first sparse layer
# of each step only. `1` runs them on every layer and chunk. Startup checks and
# the per-step builder checks always run.
_DEBUG_CHECKS_ENV = "VLLM_MINIMAX_ICP_DEBUG_CHECKS"


def _debug_checks_enabled() -> bool:
    import os  # noqa: PLC0415

    raw = os.environ.get(_DEBUG_CHECKS_ENV, "0").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    raise ValueError(f"{_DEBUG_CHECKS_ENV}={raw!r} must be 0 or 1")


# Startup marker of the r13 prefill/mix path.
_ICP_PREFILL_PATH_MARKER = "ICP_PREFILL_PATH"
# One line with the effective decode profile, for the end-to-end qualification.
_ICP_DECODE_PROFILE_MARKER = "ICP_DECODE_PROFILE"
# `fail` (default): a prewarmed MSA ICP cache that does not cover the
# installed sources stops startup; `warn` logs and continues.
_PREWARM_CHECK_ENV = "VLLM_MINIMAX_ICP_PREWARM_CHECK"
_prewarm_checked = False
_decode_profile_logged = False
# Knobs of features v13 deleted, with the value that IS v13's behaviour (None:
# no value is v13's). Any other setting is refused, never silently ignored.
_REMOVED_ICP_ENV = {
    "VLLM_MINIMAX_ICP_MAX_KV_SPLITS": "1",
    "VLLM_MINIMAX_ICP_MIN_SPLIT_TILES": None,
    "VLLM_MINIMAX_ICP_MIXED_CUTE": "off",
    "VLLM_MINIMAX_ICP_DECODE_CHUNK": "fixed",
    "VLLM_MINIMAX_ICP_CUTE_SPLIT_K": "128",
    "VLLM_MINIMAX_ICP_FUSED_END_WAIT": "1",
}


def _refuse_removed_icp_env() -> None:
    import os  # noqa: PLC0415

    bad = {
        name: os.environ[name]
        for name, keep in _REMOVED_ICP_ENV.items()
        if name in os.environ and os.environ[name].strip().lower() != keep
    }
    if bad:
        raise RuntimeError(
            f"ICP: {sorted(bad)} select features removed in v13 ({bad}); unset "
            "them (v13 behaviour: "
            + ", ".join(f"{k}={v}" for k, v in _REMOVED_ICP_ENV.items() if v)
            + ")."
        )


def _log_decode_profile(rank: int, exchange: object, chunk: int, d3: bool) -> None:
    """Once per process: the decode profile of the BOUND exchange."""
    global _decode_profile_logged
    if _decode_profile_logged:
        return
    profile = getattr(exchange, "decode_profile", None)
    if profile is None:
        raise RuntimeError("ICP: the bound candidate exchange has no decode_profile")
    if profile["d3"] != int(d3):
        raise RuntimeError(
            f"ICP: the bound exchange has d3={profile['d3']} but the indexer "
            f"metadata was built with d3={int(d3)}; set {FUSED_DECODE_ENV} "
            "identically for every process."
        )
    logger.info(
        "%s rank=%d split_k=%d chunk=%d exchange=%s merge=%s d3=%d pdl=%d",
        _ICP_DECODE_PROFILE_MARKER,
        rank,
        ICP_CUTE_SPLIT_K,
        chunk,
        profile["exchange"],
        profile["merge"],
        profile["d3"],
        profile["pdl"],
    )
    _decode_profile_logged = True


def _assert_icp_prewarm(rank: int) -> None:
    """Startup assertion: every ICP kernel is in the image's keyed cache."""
    global _prewarm_checked
    if _prewarm_checked:
        return
    # The producer is an AOT vLLM op; MSA's manifest covers the JIT kernels.
    # A warning-only cache policy must not admit a stale compound-page writer.
    require_icp_writer()
    import os  # noqa: PLC0415

    mode = os.environ.get(_PREWARM_CHECK_ENV, "fail").strip().lower() or "fail"
    if mode not in ("fail", "warn"):
        raise ValueError(f"{_PREWARM_CHECK_ENV}={mode!r} must be fail or warn")
    try:
        from fmha_sm100.icp import prewarm  # noqa: PLC0415
    except ImportError as exc:
        message = f"ICP prewarm verification is unavailable: {exc}"
        if mode == "fail":
            raise RuntimeError(message) from exc
        logger.warning("ICP_PREWARM_PROBLEM rank=%d %s", rank, message)
        _prewarm_checked = True
        return
    problems = prewarm.check()
    if problems and mode == "fail":
        raise RuntimeError(
            "ICP kernels are not fully prewarmed for this image; serving would "
            f"JIT-compile ({_PREWARM_CHECK_ENV}=warn to continue): {problems}"
        )
    for problem in problems:
        logger.warning("ICP_PREWARM_PROBLEM rank=%d %s", rank, problem)
    if not problems:
        logger.info("ICP_PREWARM_VERIFIED rank=%d", rank)
    _prewarm_checked = True


# `0` launches the decode scorer, selector and K5T without PDL.
_DECODE_PDL_ENV = "VLLM_MINIMAX_ICP_DECODE_PDL"
# The fused ICP producer (sparse_attention_icp.py) must stay a non-PDL launch:
# it is the full barrier that makes the CuTe scorer's pre-wait reads of
# GPU-produced step metadata (Model Runner V2) safe.
ICP_PRODUCER_ENABLE_PDL = False


def _env_flag(name: str, default: bool) -> bool:
    import os  # noqa: PLC0415

    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    raise ValueError(f"{name}={raw!r} must be 0 or 1")


def _decode_pdl_enabled() -> bool:
    return _env_flag(_DECODE_PDL_ENV, True)


def _icp_fused_publish(scores, geometry, token_offset, exchange, layer_idx):
    """D3: one decode window's selector, publishing into the owners' windows."""
    exchange.publish(scores, geometry, layer_idx=layer_idx, token_offset=token_offset)


def icp_cute_rows(*, use_cute_decode: bool, has_prefill: bool, num_tokens: int) -> int:
    """Leading rows the CuTe scorer takes: a pure-decode step only, as r13."""
    return num_tokens if use_cute_decode and not has_prefill else 0


_MSA_API_MODULE = "fmha_sm100.icp.scorer.prefill.api"
_CUTE_DECODE_MODULE = "fmha_sm100.icp.scorer.decode.icp_decode_score"


def _msa_api() -> Any:
    """Load the canonical ICP prefill scorer API."""
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module(_MSA_API_MODULE)
    except ImportError as exc:
        raise ImportError(
            f"msa_icp requires {_MSA_API_MODULE}; install MSA with ICP support: {exc}"
        ) from exc


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _numel(shape) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _msa_cute_decode_module() -> Any:
    """Load the canonical ICP decode scorer with its live-prefix launch ABI."""
    import importlib

    try:
        module = importlib.import_module(_CUTE_DECODE_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == _CUTE_DECODE_MODULE:
            return None
        raise
    if module.ICP_DECODE_SCORE_ABI_VERSION != 2:
        raise RuntimeError("MiniMax ICP decode requires MSA live-prefix ABI 2")
    # Launch ABI 3: window and split_k are runtime launch scalars (split_k=0
    # picks per request count) and the factory takes `use_pdl`.
    if getattr(module, "ICP_DECODE_SCORE_LAUNCH_ABI_VERSION", 1) != 3:
        raise RuntimeError(
            "MiniMax ICP decode requires MSA's per-batch-split launch ABI 3"
        )
    return module


# Greppable engagement markers, so that "did the CuTe scorer run?" is answered
# by runtime evidence rather than by the launch command: on an unadmitted
# architecture MSA's `supports_icp_decode_score` returns False silently, so
# intent and outcome diverge with nothing in the log.
_ICP_CUTE_STATUS_MARKER = "ICP_CUTE_DECODE_STATUS"
_ICP_CUTE_ENGAGED_MARKER = "ICP_CUTE_DECODE_ENGAGED"

_CUTE_DECODE_MODE_ENV = "VLLM_MINIMAX_ICP_CUTE_DECODE"
# `strict` (default): at ICP2 an unusable CuTe route is a startup failure.
# `auto`: warn and fall back to the FMHA scorer. `off`: do not probe at all.
_CUTE_DECODE_MODES = ("strict", "auto", "off")

# r13's fixed CuTe decode split; MSA's per-request-count `auto_split_k` (0) is
# never selected.
ICP_CUTE_SPLIT_K = 128


def _cute_decode_mode() -> str:
    """`strict` (hard-fail, default), `auto` (fall back to FMHA), or `off`."""
    import os  # noqa: PLC0415

    mode = os.environ.get(_CUTE_DECODE_MODE_ENV, "strict").strip().lower()
    if mode not in _CUTE_DECODE_MODES:
        raise ValueError(
            f"{_CUTE_DECODE_MODE_ENV}={mode!r} is not one of {_CUTE_DECODE_MODES}"
        )
    return mode


# Which selector arm the decode band runs. Greppable, because the arm is a
# capture-time constant and the two arms serve identically: without a log line
# an A/B cannot tell which one produced its numbers.
_ICP_DECODE_SELECTOR_MARKER = "ICP_DECODE_SELECTOR"

_DECODE_SELECTOR_ENV = "VLLM_MINIMAX_ICP_DECODE_SELECTOR"
# `full_row` (default): the exact full-row decode Top-16 arm. `bounded`: the arm
# this integration shipped before, kept reachable as the rollback path.
# These are not a fallback pair. An unusable `full_row` is a STARTUP failure,
# because the bounded arm serves normally and a silent demotion would quietly
# measure the kernel the mode exists to replace.
_DECODE_SELECTOR_MODES = ("bounded", "full_row")

# The full-row arm covers capacities up to this column count; above it the
# package's own dispatch hands the work to the bounded partitioned selector.
# That fallback is correct but silent, so admission is refused here instead.
# Now that `full_row` is the default, that refusal is the DEFAULT behaviour above
# the ceiling: P128 at max_model_len 1048576 is exactly 8192 columns, so any
# larger context refuses at startup until the arm covers it. `bounded` serves it.
_ICP_FULL_ROW_MAX_COLUMNS = 8192


def _decode_selector_mode() -> str:
    """`full_row` (default) or `bounded`."""
    import os  # noqa: PLC0415

    mode = os.environ.get(_DECODE_SELECTOR_ENV, "full_row").strip().lower()
    if mode not in _DECODE_SELECTOR_MODES:
        raise ValueError(
            f"{_DECODE_SELECTOR_ENV}={mode!r} is not one of {_DECODE_SELECTOR_MODES}"
        )
    return mode


# Which selector arm the PREFILL band runs, and under which tuning controls.
# Greppable for the same reason the decode marker is: both arms serve
# identically, so without a log line an A/B cannot tell which one produced its
# numbers -- and here the controls matter as much as the arm, because
# `RADIX_FULL_ROW` at the package's default `(512, 3)` is a different kernel
# configuration from the one that was measured.
_ICP_PREFILL_SELECTOR_MARKER = "ICP_PREFILL_SELECTOR"

_PREFILL_SELECTOR_ENV = "VLLM_MINIMAX_ICP_PREFILL_SELECTOR"
# `full_row` (default): the tuned full-row prefill arm, as r13. `bounded`: the
# arm this integration shipped before, kept reachable as the rollback path. An
# unusable `full_row` is a STARTUP failure, as on the decode band.
_PREFILL_SELECTOR_MODES = ("bounded", "full_row")

# Above this column count the package's own dispatch hands full-row work to the
# bounded partitioned selector (PREFILL_SELECTION.md, "Controls and dispatch":
# `Full-row, N > 8192: bounded fallback`). That fallback is correct but silent,
# so admission is refused here instead of reporting an arm we did not run.
_ICP_PREFILL_FULL_ROW_MAX_COLUMNS = 8192

# The measured caller recipe, not the package defaults. `PREFILL_SELECTION.md`
# ("Controls and dispatch") selects explicit FULL_ROW at 128 threads and
# cache0 for every qualified rung, with the four-warp finish enabled only at
# the host-known small-capacity condition below. The package ships (512, 3,
# False); passing that here would run a configuration nothing measured.
_ICP_PREFILL_FULL_ROW_THREADS = 128
_ICP_PREFILL_FULL_ROW_CACHED_ITEMS = 0
# finishTrue won only for the paired history-zero capacities; at 150K and 500K
# (N=1180..1300 and N=3916..4036) finishFalse won at both L2 controls.
_ICP_PREFILL_FOUR_WARP_FINISH_MAX_COLUMNS = 128

# What the package itself defaults to, used verbatim on the `bounded` rollback:
# `_prefill_full_row_config` rejects non-default controls unless the arm is
# `RADIX_FULL_ROW`, so the rollback path must hand back the defaults exactly.
_ICP_PREFILL_BOUNDED_CONTROLS = (512, 3, False)


def _prefill_selector_mode() -> str:
    """`full_row` (default) or `bounded`."""
    import os  # noqa: PLC0415

    mode = os.environ.get(_PREFILL_SELECTOR_ENV, "full_row").strip().lower()
    if mode not in _PREFILL_SELECTOR_MODES:
        raise ValueError(
            f"{_PREFILL_SELECTOR_ENV}={mode!r} is not one of {_PREFILL_SELECTOR_MODES}"
        )
    return mode


def _prefill_full_row_controls(columns: int) -> tuple[int, int, bool]:
    from fmha_sm100.icp.local_indexer import prefill_full_row_controls

    return prefill_full_row_controls(
        columns,
        threads=_ICP_PREFILL_FULL_ROW_THREADS,
        cached_items=_ICP_PREFILL_FULL_ROW_CACHED_ITEMS,
        four_warp_finish_max_columns=_ICP_PREFILL_FOUR_WARP_FINISH_MAX_COLUMNS,
    )


def _prefill_selector_kwargs(controls: tuple[int, int, bool]) -> dict:
    from fmha_sm100.icp.local_indexer import prefill_selector_kwargs

    return prefill_selector_kwargs(controls)


def _select_arm_enum():
    """The package's ``SelectArm``, imported lazily like every other
    ``fmha_sm100.icp`` symbol here so import order stays unchanged."""
    from fmha_sm100.icp.candidates import SelectArm  # noqa: PLC0415

    return SelectArm


def _device_arch_str(device) -> str:
    """`major``minor` for the log, or `unknown` off-GPU. Never raises."""
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:  # noqa: BLE001 - diagnostics only
        return "unknown"
    return f"{major}{minor}"


# Forward-side engagement evidence. Host-side only: a replaying CUDA graph
# re-executes no Python, so this counts the invocations that were RECORDED (and
# any eager ones), not graph replays. It distinguishes "the CuTe callable was
# bound into the decode graph" from "the FMHA scorer was".
_ICP_CUTE_ENGAGEMENTS: dict[int, int] = {}


def _note_cute_engagement(rank: int, chunk) -> None:
    seen = _ICP_CUTE_ENGAGEMENTS.get(rank, 0)
    _ICP_CUTE_ENGAGEMENTS[rank] = seen + 1
    if seen == 0:
        logger.info(
            "%s rank=%d token_begin=%d num_live_tokens=%d score_extent=%d scorer=%r",
            _ICP_CUTE_ENGAGED_MARKER,
            rank,
            chunk.token_begin,
            chunk.num_live_tokens,
            chunk.score_extent,
            chunk.cute_scorer,
        )


def _index_head_dim(vllm_config: VllmConfig) -> int:
    """This model's index head dimension, for the CuTe admission probe.

    Resolved as `common/indexer` resolves the sparse block, and defaulted to
    the compound page's own `INDEX_HEAD_DIM` when the model config does not
    name it -- the layout uses the same default, so the two cannot disagree.
    """
    hf_config = vllm_config.model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    sparse_cfg = getattr(text_config, "sparse_attention_config", None) or {}
    try:
        return int(sparse_cfg["sparse_index_dim"])
    except (KeyError, TypeError, ValueError):
        return INDEX_HEAD_DIM


def _uniform_decode_query_len(query_start_loc_cpu, has_prefill: bool) -> int:
    if has_prefill or int(query_start_loc_cpu[0]) != 0:
        return 0
    lengths = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    live = lengths[lengths > 0]
    if not live.numel():
        return 0
    # V2 packs live requests before zero-length capture padding. Other layouts
    # keep the FMHA route rather than deriving the wrong fixed request range.
    if not torch.equal(lengths[: live.numel()], live):
        return 0
    query_len = int(live[0])
    return query_len if query_len <= 4 and bool((live == query_len).all()) else 0


# The ICP scorer's INPUT ABI, versioned separately from `_FMHA_ICP_SCORE_ABI`
# on the output side because the two waves move independently. This version
# hands the kernel the ordinary rectangular block table and lets it derive its
# own bound from the exact device sequence length
# (`DIRECT_TABLE_CONTRACT.md` §5.1).
_MSA_DIRECT_TABLE_ABI_VERSION = 1

# The planner parameters that carry it. All three, not one representative: the
# row ORIGIN is what makes "batch index is table row" true for chunk `k > 0` as
# well as chunk 0. A planner taking the base and the pitch but not the origin
# would address every chunk as if it were the first -- in bounds, wrong tenant,
# no fault.
_MSA_DIRECT_TABLE_PLAN_ARGS = frozenset(
    {
        "icp_block_table",
        "icp_block_table_row_stride",
        "icp_block_table_row_begin",
    }
)


def _msa_has_direct_table() -> bool:
    """Whether the installed ``fmha_sm100`` accepts the ordinary block table.

    DERIVED, not declared. An overlay whose Python half advertises the mode
    while its kernel half does not carry it still ACCEPTS the call, leaves the
    table pointer null, and takes the PACKED path -- on a plan that has no
    packed list, because this builder no longer produces one.

    Two independent facts are required, because either alone is satisfiable by
    a half-migrated overlay: the planner's own signature must take the table,
    and the package's kernel-half probe (``_FMHA_HAS_ICP_DIRECT_TABLE``, itself
    derived from the overlay sources) must agree. The probe is assumed true only
    when the attribute is absent entirely, i.e. for a pre-probe package rather
    than a failing one.

    Returns:
        True when a direct-table plan can be built and executed.

    """
    import inspect  # noqa: PLC0415

    api = _msa_api()
    impl = getattr(api, "_fmha_sm100_plan_impl", None)
    if impl is None:
        return False
    try:
        params = inspect.signature(impl).parameters
    except (TypeError, ValueError):
        return False
    if not params.keys() >= _MSA_DIRECT_TABLE_PLAN_ARGS:
        return False
    return bool(getattr(api, "_FMHA_HAS_ICP_DIRECT_TABLE", False))


class MiniMaxM3IndexerMSABackend(MiniMaxM3IndexerBackend):
    """Indexer side-cache backend selecting the MSA builder."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerMSAMetadataBuilder"]:
        return MiniMaxM3IndexerMSAMetadataBuilder


@dataclass
class MiniMaxM3IndexerICPChunk:
    """One bounded query chunk of a ``W > 1`` invocation.

    Carries a prepared CuTe decode callable or an FMHA fallback plan, plus
    per-row selection geometry. Physical score column ``c`` maps to logical
    B128 block ``begin + stride*c``; at the integrated P128 profile that is
    begin=0/stride=1, i.e. the identity.
    """

    plan: dict | None  # absent for the prepared CuTe decode route
    token_begin: int  # first invocation row of this chunk
    num_live_tokens: int  # rows actually scored; the rest are inactive padding
    # int32 [max_num_reqs, row_stride], the ORDINARY block table, passed WHOLE
    # and never as a slice: it is the runner's persistent per-group table at
    # storage offset 0, so its `data_ptr` is a startup constant and a captured
    # graph keeps addressing the bytes the next step's `prepare_inputs` writes.
    # Row `i` holds request `i`'s physical compound pages in logical order; this
    # rank's compact index rows occupy `[0, R)` of each local page, so rank
    # changes logical positions and never a byte offset.
    #
    # I-1: the row's CAPACITY is an address pitch and never a bound. Columns
    # past a request's live block count hold physical page ids left by evicted
    # requests -- valid addresses belonging to another tenant -- so the bound
    # comes from exact device sequence lengths and causal positions, which both
    # the CuTe kernel and the FMHA plan consume natively.
    block_table: torch.Tensor
    # Host startup constant: the table's row pitch, i.e. `block_table.stride(0)`.
    # Permitted under I-3 precisely because it is an address pitch fixed at
    # allocation, not a live extent.
    block_table_row_stride: int
    # Host per-graph constant: the table row of this chunk's plan batch index 0,
    # i.e. `table_row = block_table_row_begin + batch_index`. I-6/§5.2 state the
    # mapping as the bare identity, which is chunk 0 (and so every single-chunk
    # decode graph), where this is 0.
    #
    # It is a legal HOST scalar in both bands: prefill is eager, and on the
    # decode band the query length is uniform by `require_uniform`, so the first
    # request of the chunk starting at token `t0` is `t0 // query_len` -- a
    # constant of the graph, not of the live request count. The TABLE POINTER is
    # unaffected; it is still the whole table at offset 0, which is what I-6
    # protects.
    block_table_row_begin: int
    # int32 [num_requests_in_chunk], the EXACT absolute position of each
    # request's first row in this chunk. Device-resident and device-derived,
    # because the host has no exact sequence length (see ``_build_icp``); it
    # overrides the plan's own upper-bound offsets at launch through
    # ``_fmha_sm100(q_offset_override=...)``, so the causal mask is exact even
    # though the plan was sized from a bound.
    qo_offset: torch.Tensor
    live_blocks: int  # host bound on the reachable columns of any row here
    # P2: the advertised column extent for this chunk -- Pwave, the FMHA plan's
    # `max_k_tiles`. It is NOT the score plane's row stride: the plane's width
    # is a per-INVOCATION constant (the capacity `icp_max_blocks` on the decode
    # band, the admitted live-column rung on the eager prefill band), the
    # selector's `blocks` is `scores.size(2)`, and its partition ladder is a
    # pure function of that. Only the extent moves per chunk; the column-to-CTA
    # map does not.
    #
    # Bounded so the producer/consumer agreement survives. The selector reads
    # columns `[0, min(local_valid_blocks[t], blocks))` of row `t`, where
    # `local_valid_blocks` is `icp_nvalid`, derived from the EXACT device
    # positions. `live_blocks` is that same causal expression evaluated at the
    # chunk's largest HOST-BOUND position, and the bound dominates the exact
    # position (`seq_lens_cpu_upper_bound >= seq_lens`), so
    # `score_extent >= icp_nvalid[t]` for every row here: every column the
    # selector consumes is written. CuTe ABI2 writes the exact device prefix
    # only and never initializes unread tails.
    #
    # Equal to `icp_max_blocks` on the decode band; see `_build_icp`.
    score_extent: int
    # Rows the SELECTOR runs on for this chunk: `min(chunk, extent - t0)`,
    # where `extent` is the invocation's exchange extent. It is NOT the outer
    # chunk, which is a prefill call size; pinning the decode selector to it
    # would launch selection over `icp_chunk` rows for a 16-row decode graph.
    # `select_decode_candidates` requires `scores.shape[0]` to EQUAL its plan's
    # `token_capacity`, so this also picks the plan (`decode_plans`). Every term
    # is a constant of the graph, which is what C4 requires.
    select_extent: int
    topk_num_valid_pages: torch.Tensor  # int32 [select_extent]
    icp_forced_column_v1: torch.Tensor  # int32 [select_extent]
    icp_active_rows: torch.Tensor  # bool [select_extent]
    icp_scan_block_begin_v1: int = 0
    icp_global_block_stride_v1: int = 1
    cute_scorer: Callable[..., None] | None = None
    # True only when `topk_num_valid_pages` is the RANK-AWARE bound
    # `icp_local_read_bounds` produces, i.e. `U//P + (U%P > rank*R)` clamped by
    # `seq_lens`. The no-init ABI2 scorer writes exactly that prefix and leaves
    # the tail of a `torch.empty` plane undefined, so the selector's read bound
    # must equal the scorer's write bound -- not merely dominate it.
    #
    # The other producer in this file, `build`'s non-ICP `num_valid_pages`
    # (`positions // PAGE_SIZE + 1`), is neither rank-aware nor clamped by
    # `seq_lens` and would over-state the prefix. It is unreachable from here,
    # but the two tensors travel under the same metadata field name, so the
    # containment is asserted at the point of use rather than inferred.
    rank_aware_nvalid: bool = False
    # Bound once per step by the builder, so a sparse layer only launches:
    # `launch_score(index_q, index_kv, k_pages, sm_scale)` runs the producer,
    # `select()` the selector over the chunk's candidate rows.
    geometry: object | None = None
    launch_score: Callable[..., None] | None = None
    select: Callable[..., object] | None = None
    # D3: `fused_select(exchange, layer_idx)` replaces `select` on pure-decode
    # steps; None when the step or the deployment does not take D3.
    fused_select: Callable[[Any, int], None] | None = None
    # `(window-bound scorer, block_table, query_start_loc, seq_lens, positions,
    # active_rows, scores, valid)` of a CuTe chunk, for MSA's `validate`.
    cute_check_args: tuple | None = None
    # Set once the per-layer validators have run for this step.
    checked: bool = False


@dataclass
class MiniMaxM3IndexerICPState:
    """The ``W > 1`` path's per-forward state; the buffers are the builder's.

    One exchange per sparse-layer invocation consumes ``candidates`` whole over
    ``token_capacity`` rows, so the planes here are sized at that extent and not
    at the live token count. ``fmha_sm100.icp`` merges a row with ``forced == -1``
    and ``n_ordinary == 0`` to all ``-1``, so the rows past the live count yield
    only invalid candidates and cannot change a live result -- but they still
    occupy the selected execution and transport extent, which is why that extent
    is chosen per invocation rather than left at global capacity.

    ``token_capacity`` is the *selected profile*, not the process's global
    capacity: the smallest admitted exchange extent that holds this
    invocation's padded rows. It is a constant of the cudagraph (the padded
    extent is) and equal on every rank (the ladder is config-derived), which is
    what refined-icp-v1 C4 requires of the exchange shape.
    """

    chunks: list[MiniMaxM3IndexerICPChunk]
    # fp32 [chunk_width, H_group, N] scratch, a startup-built view of the
    # builder's arena. On the decode band N is the capacity `icp_max_blocks`,
    # which a capture requires; on the eager prefill band it is the rung that
    # covers this invocation's admitted live column extent, which is what lets
    # `chunk_width` be the whole invocation instead of one 128-row inner tile.
    scores: torch.Tensor
    valid: torch.Tensor  # uint8 [chunk_width, H_group, N], the ABI W4 plane
    candidates: torch.Tensor  # fp32 [Qcapacity_padded, H_group, 16, 2]
    forced: torch.Tensor  # int32 [Qcapacity], C3
    n_ordinary: torch.Tensor  # int32 [Qcapacity], C3
    prefill_plan: object  # msa_icp PrefillPlan, any prefix of its capacity
    # The prefill selector's retained partitioned scratch, fp32
    # [chunk_width, H_group, partitions, 16, 2], sized at construction from
    # `PrefillPlan.scan_extent` (a startup constant). A call takes the
    # `[:extent]` prefix. Pure scratch -- written before read -- so retaining it
    # cannot change an answer; it only stops the package allocating one per
    # chunk per layer.
    prefill_partials: torch.Tensor
    # One DecodePlan + CandidateWorkspace per admitted selection extent,
    # keyed by it. The decode selector is captured at a fixed shape and refuses
    # a shorter prefix, so the extent has to BE a plan rather than a slice.
    decode_plans: dict[int, object]
    decode_workspaces: dict[int, object]
    has_prefill: bool
    token_capacity: int
    chunk_width: int
    # The decode selector arm, resolved at startup by `_init_icp` and carried
    # here because the package fixes its dispatch before graph capture. Never
    # derived per invocation.
    decode_arm: object
    # The prefill selector arm (a startup constant) and the tuning controls for
    # the rung `prefill_plan` names, as `(threads, cached_items,
    # four_warp_finish)`. The controls are per-rung rather than per-startup
    # because the four-warp finish is a small-capacity choice; on the `bounded`
    # rollback they are the package defaults, which is what that arm requires.
    prefill_arm: object
    prefill_controls: tuple[int, int, bool]
    query_start_loc: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    # Both sliced to [num_actual_tokens], matching index_q: the scorer binds
    # all three to one symbol.
    positions: torch.Tensor | None = None
    # uint8, nonzero = live. The CuTe scorer's ABI takes bytes; this is a
    # retype of the builder's bool plane, not a second buffer.
    active_rows: torch.Tensor | None = None
    # Rows ``[0, cute_rows)`` are CuTe-scored; the rest take FMHA.
    cute_rows: int = 0
    # Startup constants, for the bound-exchange profile check.
    decode_chunk: int = 0
    fused_decode: bool = False
    checked: bool = False
    # int32 view of ``candidates[:token_capacity]``, the exchange's input.
    exchange_candidates: torch.Tensor | None = None
    # ``topk_indices_buffer[:token_capacity]`` of ``exchange_buf``.
    exchange_buf: torch.Tensor | None = None
    exchange_out: torch.Tensor | None = None
    # D3: every decode window carries `fused_select`, and `fused_padding`
    # publishes the extent's rows no window selects (all inactive), so the
    # merge sees every row of `[0, token_capacity)` once per layer.
    fused_ready: bool = False
    fused_padding: tuple = ()


@dataclass
class MiniMaxM3IndexerMSAMetadata(MiniMaxM3IndexerMetadata):
    """Live ICP geometry and retained per-invocation workspaces."""

    step_n_valid: torch.Tensor | None = None
    live_metadata: ICPLiveMetadata | None = None
    # Rank-local count of blocks with a causally visible index row, [total_q].
    # This is the bound consumed by CandidateGeometry.
    topk_num_valid_pages: torch.Tensor | None = None
    # refined-icp-v1 selection geometry over the whole invocation, under the
    # versioned names CandidateGeometry duck-types.
    icp_forced_column_v1: torch.Tensor | None = None
    icp_active_rows: torch.Tensor | None = None
    icp_scan_block_begin_v1: int = 0
    # The fragment-aware path's per-forward state.
    icp: MiniMaxM3IndexerICPState | None = None


class MiniMaxM3IndexerMSAMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """Build retained ICP decode profiles and per-step eager prefill plans."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # Prefill (NCCL exchange) stays out of FULL capture because its plans,
    # wave rung and exchange extent are per-step eager choices (see the
    # prefill-wave notes above); decode (K5T/D3) is captured.
    cudagraph_decode_phase_only: ClassVar[bool] = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # W (= TP) is the indexer context-parallel degree, exactly as the parent
        # sparse layer derives it (nvidia/model.py: index_world_size = tp_size).
        self.icp_c = get_tensor_model_parallel_world_size()
        self.icp_rank = get_tensor_model_parallel_rank()
        self.icp_physical_page_tokens = kv_cache_spec.block_size
        # Startup bounds. Every destination below is sized from these, never
        # from a per-step quantity and never from anything the device knows.
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_blocks_per_req = _cdiv(
            vllm_config.model_config.max_model_len, self.icp_physical_page_tokens
        )
        if self.icp_c != 2:
            raise ValueError("MiniMax M3 ICP requires TP2")
        self._init_icp(vllm_config, device)

    def _init_icp(self, vllm_config: VllmConfig, device: torch.device) -> None:
        """Allocate the fragment-aware path's retained capacity, once.

        Everything sized here is capacity: the selector's row stride ``N``, the
        bounded chunk, the exchange's token capacity. The decode workspace in
        particular must be obtained here and never per call -- it is retained
        allocation plus a JIT build.
        """
        msa_icp = require_msa_icp()
        from fmha_sm100.icp.abi import QCAPACITY_ALIGNMENT  # noqa: PLC0415
        from fmha_sm100.icp.candidates import (  # noqa: PLC0415
            CANDIDATE_MAPPING_ABI_VERSION,
            DecodePlan,
            PrefillPlan,
            allocate_workspace,
        )

        # Refuses a stale or pre-C4 msa_icp build; both failure modes it
        # names are silent at the call site.
        expected_abis = (
            "refined-icp-v1.abi.1",
            "refined-icp-v1.k2.2",
            "refined-icp-v1.C4",
            2,
        )
        actual_abis = (
            msa_icp.abi.ABI_VERSION,
            msa_icp.abi.K2_ABI_VERSION,
            msa_icp.abi.K2_CARRIER_VERSION,
            CANDIDATE_MAPPING_ABI_VERSION,
        )
        if actual_abis != expected_abis:
            raise RuntimeError(f"ICP package ABI mismatch: {actual_abis!r}")
        msa_icp.abi.assert_compatible()

        # ...and an fmha_sm100 that still wants the packed page list. Checked at
        # STARTUP, because the alternative is a plan carrying a `block_table`
        # key the planner ignores while its `kv_page_indptr` stays None: the
        # kernel would then read a null page directory, and the builder has
        # nothing left to hand it.
        if not _msa_has_direct_table():
            raise RuntimeError(
                f"{self.__class__.__name__}: W={self.icp_c} needs an "
                "fmha_sm100 that accepts the ordinary block table: the planner "
                f"must take {sorted(_MSA_DIRECT_TABLE_PLAN_ARGS)} and derive "
                "this rank's fragment bound from the exact device "
                "`kv_segment_lens` (floor(L/128) + ((L mod 128) > rank*R)). "
                "The installed one still wants a per-rank packed page list, "
                "which this builder no longer produces -- the second page "
                "directory was removed so prefix caching, COW, eviction and "
                "offload run on one lifecycle."
            )
        # ...and it must be the version this builder was written against. The
        # capability probe says the symbols exist; this says they still mean
        # what they meant. A version that redefined, say, the row origin from a
        # row index to a byte offset would pass the probe and address the wrong
        # tenant.
        direct_version = getattr(_msa_api(), "_FMHA_ICP_DIRECT_TABLE_ABI_VERSION", None)
        if direct_version != _MSA_DIRECT_TABLE_ABI_VERSION:
            raise RuntimeError(
                f"{self.__class__.__name__}: this fmha_sm100 advertises "
                f"direct-table input ABI version {direct_version}; this "
                f"builder emits version {_MSA_DIRECT_TABLE_ABI_VERSION} "
                "(block table base + row pitch + per-chunk row origin + exact "
                "device KV lengths)."
            )

        physical_page_tokens = self.icp_physical_page_tokens
        # P128/R64 only: this vLLM route is not integrated for any other
        # compound-page size. A P256/R128 layout would give each rank whole
        # alternating B128 blocks, an affine global id -- which this file's own
        # `icp_global_block_stride` contract can already express, so the gate is
        # about what is integrated and qualified here, not about a missing
        # capability. Gate it rather than carry a second, half-live layout.
        if physical_page_tokens != 128:
            raise ValueError(
                "ICP supports P128 compound pages only; got "
                f"P{physical_page_tokens}. P256/R128 is not integrated."
            )
        self.icp_rows = physical_page_tokens // self.icp_c
        # Identity at P128: local score column `c` IS global B128 block `c`.
        self.icp_global_block_stride = 1
        self.icp_scan_block_begin = 0
        # H_group: every rank scores ALL of the group's index-query heads and
        # routes head h to rank h // H_local (ABI G5/C9). Computed before the
        # CuTe probe, which must be asked about the REAL head envelope.
        self.icp_heads_group = self.num_index_heads * self.icp_c
        self.icp_index_head_dim = _index_head_dim(vllm_config)

        self.icp_cute_module = _msa_cute_decode_module() if self.icp_c == 2 else None
        self.icp_cute_query_lens: set[int] = set()
        self.icp_cute_scorers: dict[int, Any] = {}
        self.icp_cute_factory_has_page_size = False
        self.icp_cute_factory_has_pdl = False
        self._icp_cute_dtypes_checked = False
        self.icp_cute_mode = _cute_decode_mode()
        # Why the route is or is not available, in one greppable string: the
        # startup half of the engagement evidence, the forward half being the
        # `_ICP_CUTE_ENGAGED_MARKER` line emitted at first invocation.
        self.icp_cute_status = "disabled: icp_c != 2"
        if self.icp_c == 2 and self.icp_cute_mode == "off":
            self.icp_cute_status = f"disabled: {_CUTE_DECODE_MODE_ENV}=off"
        elif self.icp_c == 2 and self.icp_cute_module is None:
            self.icp_cute_status = (
                "unavailable: fmha_sm100.icp_decode_score is not installed "
                "(this image predates the MSA CuTe decode scorer)"
            )
        elif self.icp_cute_module is not None:
            import inspect

            factory_params = inspect.signature(
                self.icp_cute_module.get_icp_decode_scorer
            ).parameters
            self.icp_cute_factory_has_page_size = "page_size" in factory_params
            self.icp_cute_factory_has_pdl = "use_pdl" in factory_params
            # Ask about the REAL envelope: omitting `num_heads`/`head_dim` lets
            # MSA's defaults answer for a model that has neither, so startup
            # would admit a profile `validate()` then rejects at first
            # invocation -- for a decode-only model, during cudagraph capture.
            # A decode row's query length never exceeds the reorder threshold,
            # so every admitted length is one `_prewarm_cute_decode` compiles.
            self.icp_cute_query_lens = {
                q
                for q in range(1, min(4, self.max_decode_query_len) + 1)
                if self.icp_cute_module.supports_icp_decode_score(
                    query_len=q,
                    world_size=self.icp_c,
                    page_size=physical_page_tokens,
                    num_heads=self.icp_heads_group,
                    head_dim=self.icp_index_head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
            }
            if self.icp_cute_query_lens:
                self.icp_cute_status = (
                    f"engaged: query_lens={sorted(self.icp_cute_query_lens)} "
                    f"split_k={ICP_CUTE_SPLIT_K}"
                )
            else:
                # `supports_icp_decode_score` returns False and never raises,
                # so every rejection reason -- architecture, head envelope,
                # dtype, page size -- arrives here as an empty set. Name them
                # all, since the message is the only diagnosis available.
                self.icp_cute_status = (
                    "unavailable: MSA refused every query_len "
                    f"1..{min(4, self.max_decode_query_len)} for "
                    f"world_size={self.icp_c} page_size={physical_page_tokens} "
                    f"num_heads={self.icp_heads_group} "
                    f"head_dim={self.icp_index_head_dim} "
                    f"arch=sm_{_device_arch_str(device)}"
                )
        # WARNING, not INFO, when the route is unavailable: the fallback serves
        # normally, so a silent demotion is indistinguishable from success.
        _log = logger.info if self.icp_cute_query_lens else logger.warning
        _log(
            "%s rank=%d %s",
            _ICP_CUTE_STATUS_MARKER,
            self.icp_rank,
            self.icp_cute_status,
        )
        # Default `strict`: at ICP2 an unusable CuTe route is a startup failure,
        # because the demoted run still serves and every number it produces
        # belongs to the FMHA `OnlyScoreIcp` scorer instead. `auto` accepts the
        # fallback deliberately, `off` disables the route.
        if (
            self.icp_c == 2
            and self.icp_cute_mode == "strict"
            and not self.icp_cute_query_lens
        ):
            raise RuntimeError(
                "MiniMax ICP CuTe decode scorer is required at ICP2 but is "
                f"not usable: {self.icp_cute_status}. Continuing would fall "
                "back to the FMHA OnlyScoreIcp scorer and serve normally "
                "while measuring the kernel this route exists to replace. "
                f"Set {_CUTE_DECODE_MODE_ENV}=auto to accept that fallback "
                f"deliberately, or {_CUTE_DECODE_MODE_ENV}=off to disable the "
                "route."
            )
        # P128 only, so the score/table column count and the global B128 id
        # space are the same quantity and share one name.
        self.icp_max_blocks = _cdiv(
            vllm_config.model_config.max_model_len, physical_page_tokens
        )
        # The decode selector arm, resolved ONCE at startup. The owning package
        # fixes the arm's dispatch before graph capture, so this cannot be a
        # per-invocation choice; it is carried on the metadata state and read at
        # the call site.
        self.icp_decode_selector_mode = _decode_selector_mode()
        self.icp_decode_arm = _select_arm_enum().RADIX_BOUNDED
        if self.icp_decode_selector_mode == "full_row":
            arm = getattr(_select_arm_enum(), "RADIX_FULL_ROW", None)
            if arm is None:
                raise RuntimeError(
                    "full_row (the default decode selector mode) requires an "
                    "fmha_sm100.icp package that defines "
                    "SelectArm.RADIX_FULL_ROW; the installed one does not. "
                    "Bump the package, or set "
                    f"{_DECODE_SELECTOR_ENV}=bounded to roll back."
                )
            if self.icp_max_blocks > _ICP_FULL_ROW_MAX_COLUMNS:
                raise RuntimeError(
                    "full_row (the default decode selector mode) admits up to "
                    f"{_ICP_FULL_ROW_MAX_COLUMNS} B128 columns, but this "
                    f"deployment's capacity is {self.icp_max_blocks} "
                    f"(max_model_len={vllm_config.model_config.max_model_len}, "
                    f"page={physical_page_tokens}). The package would dispatch "
                    "to the bounded partitioned selector with no log line, so "
                    "the run would report an arm it did not use. Set "
                    f"{_DECODE_SELECTOR_ENV}=bounded to serve this capacity."
                )
            self.icp_decode_arm = arm
        logger.info(
            "%s rank=%d arm=%s mode=%s capacity=%d columns",
            _ICP_DECODE_SELECTOR_MARKER,
            self.icp_rank,
            self.icp_decode_arm.name,
            self.icp_decode_selector_mode,
            self.icp_max_blocks,
        )
        # D3 publishes from the full-row selector's epilogue; no other arm has
        # one. The exchange reads the same env, so a mismatch cannot start.
        _refuse_removed_icp_env()
        self.icp_fused_decode = fused_decode_requested()
        if self.icp_fused_decode and self.icp_decode_selector_mode != "full_row":
            raise RuntimeError(
                f"D3 ({FUSED_DECODE_ENV}, default 1) needs the full_row decode "
                f"selector; got {self.icp_decode_selector_mode!r}. Set "
                f"{FUSED_DECODE_ENV}=0 to serve this selector over K5T."
            )
        # The PREFILL selector arm, resolved ONCE at startup for the same reason
        # the decode arm is -- except that prefill is eager, so what is fixed
        # here is the recipe, not a capture. The CONTROLS still vary per rung
        # (the four-warp finish is a small-capacity choice), so only the arm and
        # the mode are startup constants; `_build_icp` derives the controls from
        # the rung it selects.
        self.icp_prefill_selector_mode = _prefill_selector_mode()
        self.icp_prefill_arm = _select_arm_enum().RADIX_BOUNDED
        if self.icp_prefill_selector_mode == "full_row":
            arm = getattr(_select_arm_enum(), "RADIX_FULL_ROW", None)
            if arm is None:
                raise RuntimeError(
                    "full_row (the default prefill selector mode) requires an "
                    "fmha_sm100.icp package that defines "
                    "SelectArm.RADIX_FULL_ROW; the installed one does not. "
                    "Bump the package, or set "
                    f"{_PREFILL_SELECTOR_ENV}=bounded to roll back."
                )
            if self.icp_max_blocks > _ICP_PREFILL_FULL_ROW_MAX_COLUMNS:
                raise RuntimeError(
                    "full_row (the default prefill selector mode) admits up to "
                    f"{_ICP_PREFILL_FULL_ROW_MAX_COLUMNS} B128 columns, but "
                    f"this deployment's capacity is {self.icp_max_blocks} "
                    f"(max_model_len={vllm_config.model_config.max_model_len}, "
                    f"page={physical_page_tokens}). The package would dispatch "
                    "to the bounded partitioned selector with no log line, so "
                    "the run would report an arm it did not use. Set "
                    f"{_PREFILL_SELECTOR_ENV}=bounded to serve this capacity."
                )
            self.icp_prefill_arm = arm
        if self.icp_prefill_selector_mode == "full_row":
            controls = (
                f"threads={_ICP_PREFILL_FULL_ROW_THREADS} "
                f"cached_items={_ICP_PREFILL_FULL_ROW_CACHED_ITEMS} "
                "four_warp_finish=columns<="
                f"{_ICP_PREFILL_FOUR_WARP_FINISH_MAX_COLUMNS}"
            )
        else:
            controls = "package defaults"
        logger.info(
            "%s rank=%d arm=%s mode=%s capacity=%d columns %s",
            _ICP_PREFILL_SELECTOR_MARKER,
            self.icp_rank,
            self.icp_prefill_arm.name,
            self.icp_prefill_selector_mode,
            self.icp_max_blocks,
            controls,
        )
        # The exchange's admitted extents, from the config alone -- the same
        # call `nvidia/model.py:_bind_candidate_exchange` makes, so the ladder
        # the builder selects from IS the ladder the windows were built for.
        # The mirror of the ABI's row alignment is cross-checked against the
        # real constant here, where both are in scope.
        assert _DISPATCH_QCAPACITY_ALIGNMENT == QCAPACITY_ALIGNMENT, (
            "icp_dispatch.QCAPACITY_ALIGNMENT "
            f"({_DISPATCH_QCAPACITY_ALIGNMENT}) has drifted from "
            f"fmha_sm100.icp.abi.QCAPACITY_ALIGNMENT ({QCAPACITY_ALIGNMENT})."
        )
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.icp_exchange_capacities = exchange_token_capacities(
            max_token_capacity=max_num_batched_tokens,
            cudagraph_capture_sizes=(
                vllm_config.compilation_config.cudagraph_capture_sizes or ()
            ),
        )
        # Qcapacity is the exchange's GLOBAL token capacity and must match the
        # shared top-k buffer the model allocated (nvidia/model.py pads by 4,
        # which is ABI G12); the impl asserts that identity. A single
        # invocation exchanges only the smallest admitted extent that holds its
        # padded rows, which is what keeps a 16-row decode step off the
        # scheduler's maximum.
        self.icp_token_capacity = self.icp_exchange_capacities[-1]
        assert self.icp_token_capacity == (
            _cdiv(max_num_batched_tokens, QCAPACITY_ALIGNMENT) * QCAPACITY_ALIGNMENT
        )
        # ---- the EAGER PREFILL exchange extents ------------------------------
        # The list above is the CAPTURE ladder, whose rungs are the FULL decode
        # buckets; it is sparse between the largest bucket and the token budget,
        # so a prefill just past that bucket would exchange -- and select over --
        # the whole budget. The eager band gets its own geometric fill instead.
        #
        # It may, where decode may not, because `cudagraph_decode_phase_only` is
        # True: any prefill in the batch excludes FULL decode capture, so there
        # is no capture shape to respect and no replay for the extent to be
        # constant across. What C4 still requires is rank-equality, and this has
        # it for the same reason the capture ladder does -- the rungs are a pure
        # function of `max_num_batched_tokens` and a module constant, and the
        # selection input is `num_actual_tokens`, which is scheduler metadata.
        #
        # `icp_exchange_capacities` above is UNTOUCHED, and with it every
        # decode-side quantity derived from it (`icp_select_extents`,
        # `icp_decode_plans`, `icp_decode_workspaces`). The two bands select out
        # of their own lists; only the shared exchange instance sees the union,
        # which is what `nvidia/model.py` builds it from.
        self.icp_prefill_exchange_capacities = prefill_exchange_token_capacities(
            max_token_capacity=max_num_batched_tokens,
        )
        assert self.icp_prefill_exchange_capacities[-1] == self.icp_token_capacity
        self.icp_admitted_exchange_extents = admitted_exchange_extents(
            max_token_capacity=max_num_batched_tokens,
            cudagraph_capture_sizes=(
                vllm_config.compilation_config.cudagraph_capture_sizes or ()
            ),
        )
        # Every extent either band can resolve is one the shared instance
        # admits. Both sides derive it from the same functions and the same
        # config, so this can only fail if one of them stops doing that -- and
        # the failure it guards is a `ValueError` deep in `_check_candidates`
        # on the first prefill step of a serving run, not at startup.
        assert set(self.icp_exchange_capacities) <= set(
            self.icp_admitted_exchange_extents
        )
        assert set(self.icp_prefill_exchange_capacities) <= set(
            self.icp_admitted_exchange_extents
        )
        # ---- the prefill wave ladder ----------------------------------------
        # One rung per admitted score-column extent; the build picks the
        # smallest rung that covers the invocation's host-bounded live extent.
        # `_prefill_wave_override` collapses the ladder to a single pinned width
        # at FULL capacity columns, which is the profiling lane's control arm
        # (128 reproduces the shipped schedule exactly).
        self.icp_prefill_budget_bytes = _prefill_score_budget_bytes()
        self.icp_prefill_pinned_wave = _prefill_wave_override()
        self.icp_decode_pdl = _decode_pdl_enabled()
        if self.icp_prefill_pinned_wave is None:
            self.icp_prefill_columns = icp_prefill_column_rungs(self.icp_max_blocks)
            self.icp_prefill_waves = icp_prefill_wave_ladder(
                self.icp_prefill_columns,
                heads_group=self.icp_heads_group,
                token_capacity=self.icp_token_capacity,
                budget_bytes=self.icp_prefill_budget_bytes,
            )
        else:
            # A pin collapses the ladder to ONE width at FULL capacity columns.
            # A pin wider than the invocation is REFUSED, not clamped, so a
            # sweep arm cannot be mislabelled with a width it never ran.
            budget_rows = (
                _cdiv(self.icp_token_capacity, ICP_MSA_QUERY_TILE_TOKENS)
                * ICP_MSA_QUERY_TILE_TOKENS
            )
            if self.icp_prefill_pinned_wave > budget_rows:
                raise ValueError(
                    f"{_PREFILL_WAVE_ENV}={self.icp_prefill_pinned_wave} is "
                    f"wider than this deployment's {budget_rows}-row token "
                    f"budget (padded capacity {self.icp_token_capacity}). The "
                    "override exists to PIN A NARROWER wave for the profiling "
                    "lane; a wider one cannot be scheduled."
                )
            self.icp_prefill_columns = (self.icp_max_blocks,)
            self.icp_prefill_waves = {self.icp_max_blocks: self.icp_prefill_pinned_wave}
        self.icp_prefill_max_wave = max(self.icp_prefill_waves.values())
        self.icp_prefill_min_wave = min(self.icp_prefill_waves.values())

        # The DECODE outer chunk (`icp_decode_chunk_tokens`). It is a
        # startup constant: `icp_select_extents`, `icp_decode_plans` and
        # `icp_decode_workspaces` are derived from it and ARE the captured
        # decode profiles.
        prefill_elems = icp_prefill_arena_elems(
            self.icp_prefill_waves, self.icp_heads_group
        )
        self.icp_chunk = icp_decode_chunk_tokens(
            max_model_len=vllm_config.model_config.max_model_len,
            heads_group=self.icp_heads_group,
            token_capacity=self.icp_token_capacity,
            physical_page_tokens=physical_page_tokens,
        )
        split_sizes = sorted(
            size
            for size in (vllm_config.compilation_config.cudagraph_capture_sizes or ())
            if min(size, self.icp_token_capacity) > self.icp_chunk
        )
        if split_sizes:
            logger.warning(
                "MiniMax-M3 ICP indexer (W=%d rank=%d): the %d-row decode chunk "
                "is smaller than captured decode sizes %s, which run several "
                "scorer + selector launches per sparse layer.",
                self.icp_c,
                self.icp_rank,
                self.icp_chunk,
                split_sizes,
            )

        # Rounded up so the last chunk's candidate slice is a whole chunk, for
        # EVERY admitted stride: the decode chunk and each prefill wave.
        cap_padded = max(
            _cdiv(self.icp_token_capacity, width) * width
            for width in {self.icp_chunk, *self.icp_prefill_waves.values()}
        )

        # One DEVICE plan slot per chunk position, allocated once
        # (ICP_DEVICE_PLAN_ABI 1). The first sparse KV writer derives every
        # slot's OnlyScoreIcp work plan from the exact device `query_start_loc`
        # and `seq_lens` in the same launch as the live metadata; a host build
        # only sizes and views a slot. The slots cover every chunk POSITION
        # either band can reach -- `ceil(icp_token_capacity / width)` over the
        # decode chunk and the NARROWEST prefill wave -- so
        # `icp_plan_store.slot(...)` is in range by construction.
        self.icp_chunk_slots = max(
            _cdiv(self.icp_token_capacity, self.icp_chunk),
            _cdiv(self.icp_token_capacity, self.icp_prefill_min_wave),
        )
        self.icp_plan_slots = self.icp_chunk_slots
        # The widest admitted stride bounds a plan's tokens and, with
        # max_num_seqs, its requests.
        pool_chunk = max(self.icp_chunk, self.icp_prefill_max_wave)
        # r13 planned each chunk on the host with num_kv_splits=1; this MSA
        # refuses host-staged direct-table plans (api.py `_fmha_sm100_plan_impl`),
        # so the same unsplit direct_greedy plan is derived by the writer.
        self.icp_plan_store = _msa_api().IcpDevicePlanStore(
            device=device,
            num_slots=self.icp_plan_slots,
            max_segments=min(pool_chunk, self.max_num_reqs),
            num_heads=self.icp_heads_group,
            max_chunk_tokens=pool_chunk,
            max_kv_splits=1,
        )
        logger.info(
            "%s rank=%d r13 prefill/mix: device_plan num_kv_splits=%d plan_slots=%d "
            "prefill_selector=%s mixed_steps=fmha_only rungs=geometric "
            "max_wave=%d exchange=native_pack+all_to_all+k2 | decode: "
            "decode chunk %d rows (fixed) %s=%s %s=%d %s=%d %s=%d",
            _ICP_PREFILL_PATH_MARKER,
            self.icp_rank,
            self.icp_plan_store.max_kv_splits,
            self.icp_plan_store.num_slots,
            self.icp_prefill_selector_mode,
            self.icp_prefill_max_wave,
            self.icp_chunk,
            _CUTE_DECODE_MODE_ENV,
            self.icp_cute_mode,
            _DEBUG_CHECKS_ENV,
            int(_debug_checks_enabled()),
            _DECODE_PDL_ENV,
            int(self.icp_decode_pdl),
            FUSED_DECODE_ENV,
            int(self.icp_fused_decode),
        )
        _assert_icp_prewarm(self.icp_rank)
        if self.icp_decode_pdl and ICP_PRODUCER_ENABLE_PDL:
            raise RuntimeError(
                "ICP decode PDL requires the fused producer to be a non-PDL "
                "launch: the CuTe scorer reads step metadata before its wait"
            )

        # ---- one score/validity arena, carved per band ----------------------
        # One explicitly shared arena with a fixed decode view. The decode view
        # sits at offset 0 with shape `[icp_chunk, H_group, icp_max_blocks]`, so
        # its address is as fixed as a dedicated allocation's -- which is what a
        # capture requires of it.
        #
        # Prefill and mixed invocations use only FMHA rung views; pure decode
        # uses the CuTe view. Every window is a
        # scorer launch followed by the selector launch that consumes it, all
        # on the current stream, and no window reads a cell another window
        # wrote: each selector reads only rows its own scorer just wrote, and
        # only the exact live prefix of each (ABI2; the FMHA producer
        # completes its advertised rectangle). Decode's CuTe scorer, decode
        # selector and K5T are PDL launches (`VLLM_MINIMAX_ICP_DECODE_PDL`),
        # but each executes griddepcontrol.wait before its first store and
        # before reading anything its predecessor wrote; the scorer's only
        # pre-wait reads are step inputs written before the forward (by GPU
        # kernels under Model Runner V2; safe only because the fused producer
        # ahead of the scorer is non-PDL, ICP_PRODUCER_ENABLE_PDL) and index-K
        # fragments that end before this step's new tokens. A wait returns
        # only after the
        # predecessor grid completed, and every predecessor waited on its own,
        # so a later scorer still cannot overwrite a plane an earlier selector
        # is reading. Prefill and mixed selectors stay non-PDL. The selectors'
        # partials scratch aliases the same way.
        decode_plane = (self.icp_chunk, self.icp_heads_group, self.icp_max_blocks)
        decode_elems = self.icp_chunk * self.icp_heads_group * self.icp_max_blocks
        self.icp_score_arena, self.icp_valid_arena = _icp_allocate_score_planes(
            elems=max(decode_elems, prefill_elems),
            device=device,
            budget_bytes=self.icp_prefill_budget_bytes,
            heads_group=self.icp_heads_group,
            columns=self.icp_max_blocks,
        )
        # CuTe ABI2 leaves tails undefined; selection reads exact live prefixes.
        self.icp_scores = self.icp_score_arena[:decode_elems].view(decode_plane)
        self.icp_valid = self.icp_valid_arena[:decode_elems].view(decode_plane)
        # The native first sparse writer invalidates inactive rows within this
        # invocation's exchange extent before selection or communication.
        self.icp_candidates = torch.empty(
            (cap_padded, self.icp_heads_group, 16, 2),
            dtype=torch.float32,
            device=device,
        )
        self.icp_positions = torch.zeros(cap_padded, dtype=torch.int64, device=device)
        self.icp_active = torch.zeros(cap_padded, dtype=torch.bool, device=device)
        # The SAME bytes, retyped once, for MSA's CuTe decode scorer: its ABI
        # takes the activity plane as `uint8` and tvm-ffi rejects a `torch.bool`
        # tensor on DLPack dtype code alone. `view` is a retype, not a copy, so
        # the selector (bool) and the scorer (uint8) read one buffer and cannot
        # disagree. Built here, never per forward: the build path may not
        # allocate, and a captured graph holds the capture address.
        self.icp_active_bytes = self.icp_active.view(torch.uint8)
        self.icp_nvalid = torch.zeros(cap_padded, dtype=torch.int32, device=device)
        self.icp_global_nvalid = torch.zeros(
            cap_padded, dtype=torch.int32, device=device
        )
        self.icp_forced_column = torch.full(
            (cap_padded,), -1, dtype=torch.int32, device=device
        )
        # Merge and selector forcing share the first writer's exact positions.
        # Retain all destinations for FULL-graph replay.
        self.icp_forced = torch.full(
            (cap_padded,), -1, dtype=torch.int32, device=device
        )
        self.icp_n_ordinary = torch.zeros(cap_padded, dtype=torch.int32, device=device)

        # ---- fixed-capacity destinations, sized from startup constants ------
        # There is deliberately NO page pool here: the W > 1 route consumes the
        # ordinary block table directly, so this rank has no second page
        # directory that could disagree with the first about eviction, COW,
        # prefix reuse or offload (DIRECT_TABLE_CONTRACT.md §7). The read bound
        # moves with it, from a host upper bound to the exact device length the
        # kernel evaluates (I-1).
        #
        # `icp_max_blocks == max_blocks_per_req` is load-bearing: the local
        # score plane and the ordinary table share one physical-page column
        # axis. Logical ranking IDs additionally apply
        # `icp_scan_block_begin + icp_global_block_stride * column`.
        assert self.icp_max_blocks == self.max_blocks_per_req
        # Per-(chunk, request) absolute causal offsets, SLOT-MAJOR because a
        # running counter over the surviving chunks hands chunk `k` a different
        # start as soon as an earlier chunk's live request count changes, so a
        # captured graph would replay chunk `k`'s plan against another chunk's
        # offsets. Row `slot_index` is at a startup-constant offset, so the
        # pointer is fixed for the process's lifetime and the per-step write
        # goes INTO it.
        #
        # A chunk holds at most `min(chunk, max_num_seqs)` requests -- each
        # contributes at least one of its <= `chunk` tokens -- which is what
        # bounds the row. `pool_chunk` is the widest admitted stride, so this
        # equals the pool's own `max_segments` and the two cannot drift.
        self.icp_reqs_per_slot = min(pool_chunk, self.max_num_reqs)
        self.icp_qo_offset = torch.zeros(
            (self.icp_chunk_slots, self.icp_reqs_per_slot),
            dtype=torch.int32,
            device=device,
        )
        # `(data_ptr, row_stride)` of the block table, latched on the first
        # build. The direct-table path treats both as startup constants, so
        # this is where that claim is enforced rather than assumed; see
        # `_build_icp`.
        self._icp_table_identity: tuple[int, int] | None = None

        plan_kwargs = dict(
            icp_degree=self.icp_c,
            icp_rank=self.icp_rank,
            num_heads_local=self.num_index_heads,
            max_local_blocks=self.icp_max_blocks,
        )
        # Prefill takes any prefix of its capacity, so ONE plan per admitted
        # column rung covers every wave it will ever be handed at that rung.
        #
        # `max_local_blocks` is per-rung and not the capacity, because
        # `select_prefill_candidates` requires the score tensor's trailing
        # extent to EQUAL it -- it is the row stride and nothing else -- so a
        # narrowed plane needs a plan naming the same width. It also narrows
        # `PrefillPlan.scan_extent`, the selector's launch geometry.
        #
        # What a narrower rung changes is the ROW PITCH, `scores.size(2)`: the
        # arena footprint and the address span per query row shrink with it.
        # Pitch, launched work and valid reads are three different quantities;
        # below the selector's narrowest ladder arm a narrower rung does not
        # reduce the launched partitions or items at all, and no latency claim
        # follows from a buffer-width ratio.
        #
        # It is SOUND only because the rung dominates every row's read bound.
        # The kernel clamps `valid = min(local_valid_blocks[t], blocks)`, so a
        # rung below a row's `icp_nvalid` would truncate that row silently; the
        # build asserts `chunk_live_blocks <= columns` for every chunk, and
        # `chunk_live_blocks` is the same causal expression as `icp_nvalid`
        # evaluated at the host UPPER bound, which dominates the exact device
        # value (see `MiniMaxM3IndexerICPChunk.score_extent`).
        self.icp_prefill_plans = {
            columns: PrefillPlan(
                token_capacity=wave,
                **{**plan_kwargs, "max_local_blocks": columns},
            )
            for columns, wave in self.icp_prefill_waves.items()
        }
        # The capacity rung, kept under the shipped name for callers that mean
        # "the plan that covers any history".
        self.icp_prefill_plan = self.icp_prefill_plans[self.icp_max_blocks]
        # Decode does NOT take a prefix: `select_decode_candidates` requires
        # `scores.shape[0] == plan.token_capacity`, so the decode selector is
        # bound to the GRAPH BUCKET -- one plan per selection extent the
        # exchange ladder can produce. Coupling it to the outer chunk instead
        # would launch selection over `icp_chunk` rows for every decode graph,
        # however few rows that graph carries.
        self.icp_select_extents = sorted(
            {
                min(self.icp_chunk, capacity - t0)
                for capacity in self.icp_exchange_capacities
                for t0 in range(0, capacity, self.icp_chunk)
            }
        )
        self.icp_decode_plans = {
            extent: DecodePlan(
                token_capacity=extent, use_pdl=self.icp_decode_pdl, **plan_kwargs
            )
            for extent in self.icp_select_extents
        }
        self.icp_decode_workspaces = {
            extent: allocate_workspace(plan, device)
            for extent, plan in self.icp_decode_plans.items()
        }
        # The PREFILL selector's partitioned scratch, retained (P4).
        #
        # `select_prefill_candidates` allocates this per call when it is not
        # given one -- once per chunk per sparse layer. It is pure scratch:
        # every used slot is written before it is read, so a retained buffer and
        # a fresh one give the same answer, and `retained[:T]` is the package's
        # supported way to serve a shorter call. Passing `partials=` is an
        # allocation change only; the scan extent is still
        # `PrefillPlan.scan_extent` and no host value is derived from a device
        # tensor.
        #
        # Sized from STARTUP BOUNDS only, one shape per admitted rung: the rows
        # are that rung's wave (a call's wave is a prefix of it) and the
        # partition count comes from that rung's `scan_extent`, itself a startup
        # constant. It is never the live extent of a step. Above one CTA per row
        # the tensor is zero-width, which is legal and still passed.
        #
        # One arena carved per rung, for the same reason the score plane is: the
        # rungs are mutually exclusive within an invocation, so the peak is the
        # maximum over rungs and not the sum.
        prefill_partial_shapes = {
            columns: plan.partial_shape(
                self.icp_prefill_waves[columns], plan.scan_extent
            )
            for columns, plan in self.icp_prefill_plans.items()
        }
        self.icp_prefill_partials_arena = torch.empty(
            max(_numel(shape) for shape in prefill_partial_shapes.values()),
            dtype=torch.float32,
            device=device,
        )
        # Every per-rung view is built HERE, at startup. A build does a dict
        # lookup and nothing else: no allocation, no reshape, no device read.
        self.icp_prefill_views = {
            columns: (
                self.icp_score_arena[
                    : self.icp_prefill_waves[columns] * self.icp_heads_group * columns
                ].view(self.icp_prefill_waves[columns], self.icp_heads_group, columns),
                self.icp_valid_arena[
                    : self.icp_prefill_waves[columns] * self.icp_heads_group * columns
                ].view(self.icp_prefill_waves[columns], self.icp_heads_group, columns),
                self.icp_prefill_partials_arena[
                    : _numel(prefill_partial_shapes[columns])
                ].view(prefill_partial_shapes[columns]),
            )
            for columns in self.icp_prefill_columns
        }
        # The capacity rung's scratch, under the shipped name.
        self.icp_prefill_partials = self.icp_prefill_views[self.icp_max_blocks][2]

        logger.info(
            "MiniMax-M3 ICP indexer retained capacity (W=%d rank=%d): decode "
            "chunk %d rows, prefill wave <= %d rows, %d plan slots (inner MSA "
            "tile pinned at %d), score+validity arena %.1f MiB, page directory "
            "NONE (direct-table input ABI v%d: "
            "the ordinary block table is consumed directly, %d columns per "
            "request), slot-major "
            "causal offsets %d x %d, exchange extents %s of a %d-row capacity, "
            "decode selector extents %s, retained prefill selector partials "
            "%.1f MiB. Every one of these is a startup constant; nothing "
            "here is sized from a step.",
            self.icp_c,
            self.icp_rank,
            self.icp_chunk,
            self.icp_prefill_max_wave,
            self.icp_plan_store.num_slots,
            ICP_MSA_QUERY_TILE_TOKENS,
            (self.icp_score_arena.numel() * 4 + self.icp_valid_arena.numel()) / 2**20,
            _MSA_DIRECT_TABLE_ABI_VERSION,
            self.icp_max_blocks,
            self.icp_qo_offset.shape[0],
            self.icp_reqs_per_slot,
            self.icp_exchange_capacities,
            self.icp_token_capacity,
            self.icp_select_extents,
            self.icp_prefill_partials_arena.numel() * 4 / 2**20,
        )
        # The prefill ladder, condensed to the rungs where the wave changes, so
        # that a wave collapsing back to the inner tile is a LOGGED fact. The
        # full COLUMN ladder is printed alongside it under the SAME marker,
        # because the condensed form collapses every rung sharing a wave and
        # would otherwise hide the floor. External harnesses count occurrences
        # of the marker, so this must stay one record rather than gaining a
        # second marker literal.
        column_ratio = max(
            (
                hi / lo
                for lo, hi in zip(
                    self.icp_prefill_columns, self.icp_prefill_columns[1:]
                )
            ),
            default=1.0,
        )
        ladder, previous = [], None
        for columns in self.icp_prefill_columns:
            wave = self.icp_prefill_waves[columns]
            if wave != previous:
                ladder.append(
                    f"<={columns} cols -> wave {wave} "
                    f"({_cdiv(self.icp_token_capacity, wave)} calls/layer)"
                )
                previous = wave
        one_launch = (
            _cdiv(self.icp_token_capacity, ICP_MSA_QUERY_TILE_TOKENS)
            * ICP_MSA_QUERY_TILE_TOKENS
        )
        self.icp_prefill_one_launch_columns = max(
            (
                columns
                for columns in self.icp_prefill_columns
                if self.icp_prefill_waves[columns] >= one_launch
            ),
            default=0,
        )
        arena_bytes = self.icp_score_arena.numel() * 4 + self.icp_valid_arena.numel()
        logger.info(
            "%s (W=%d rank=%d): %s, %d MiB score budget, %d admitted column "
            "rungs, %d bytes (%.4f GiB) of score+validity arena actually "
            "retained. Column ladder %s: floor %d columns == %d tokens, %d "
            "steps/octave, worst-case column over-run %.2fx. Resolved ladder "
            "at a %d-row invocation: %s. One "
            "OnlyScoreIcp call per sparse layer holds up to %d B128 columns == "
            "%d tokens of live history.",
            _ICP_PREFILL_WAVE_MARKER,
            self.icp_c,
            self.icp_rank,
            (
                f"pinned wave {self.icp_prefill_pinned_wave} at full capacity "
                "columns (MEASUREMENT CONTROL, not a serving mode)"
                if self.icp_prefill_pinned_wave is not None
                else "wave derived per invocation from the admitted live column extent"
            ),
            self.icp_prefill_budget_bytes // 2**20,
            len(self.icp_prefill_columns),
            arena_bytes,
            arena_bytes / 2**30,
            list(self.icp_prefill_columns),
            self.icp_prefill_columns[0],
            self.icp_prefill_columns[0] * physical_page_tokens,
            ICP_PREFILL_COLUMN_STEPS_PER_OCTAVE,
            column_ratio,
            self.icp_token_capacity,
            "; ".join(ladder),
            self.icp_prefill_one_launch_columns,
            self.icp_prefill_one_launch_columns * physical_page_tokens,
        )
        if self.icp_prefill_pinned_wave is not None:
            logger.warning(
                "%s (W=%d rank=%d): the prefill wave is PINNED at %d rows, so a "
                "%d-row prefill runs %d serial scorer+selector waves per sparse "
                "layer whatever its history is. %s is a measurement control for "
                "the profiling lane and must not be set on a serving "
                "deployment.",
                _ICP_PREFILL_WAVE_MARKER,
                self.icp_c,
                self.icp_rank,
                self.icp_prefill_pinned_wave,
                self.icp_token_capacity,
                _cdiv(self.icp_token_capacity, self.icp_prefill_pinned_wave),
                _PREFILL_WAVE_ENV,
            )
        if self.icp_prefill_one_launch_columns < self.icp_max_blocks:
            logger.warning(
                "%s (W=%d rank=%d): a %d-row invocation whose live history "
                "exceeds %d B128 columns cannot be scored in one wave at a %d "
                "MiB budget; the %d-column capacity rung degrades to %d waves "
                "per sparse layer. This is bounded and LOGGED, not silent -- a "
                "wave that quietly falls back to one %d-row inner tile is the "
                "defect this ladder replaces. One launch at capacity would "
                "need %d MiB; raise it with %s.",
                _ICP_PREFILL_WAVE_MARKER,
                self.icp_c,
                self.icp_rank,
                self.icp_token_capacity,
                self.icp_prefill_one_launch_columns,
                self.icp_prefill_budget_bytes // 2**20,
                self.icp_max_blocks,
                _cdiv(self.icp_token_capacity, self.icp_prefill_min_wave),
                ICP_MSA_QUERY_TILE_TOKENS,
                _cdiv(
                    one_launch * self.icp_heads_group * self.icp_max_blocks * 5,
                    2**20,
                ),
                _PREFILL_SCORE_MIB_ENV,
            )
        # The eager prefill EXCHANGE EXTENT ladder, a separate record from the
        # wave schema above because it is a separate quantity: candidate rows
        # per selector call and per exchange, not query rows per scorer launch.
        #
        # `bytes_per_row` is the carrier payload per rank per sparse layer:
        # H_group index heads x 16 candidate records x (fp32 score bits, int32
        # id). The over-run figure is the largest ratio between neighbouring
        # rungs, which is what a row count landing just above a rung pays.
        bytes_per_row = self.icp_heads_group * _DISPATCH_CAND_K * 2 * 4
        prefill_extents = self.icp_prefill_exchange_capacities
        worst_ratio = max(
            (hi / (lo + 1) for lo, hi in zip(prefill_extents, prefill_extents[1:])),
            default=1.0,
        )
        decode_only = set(self.icp_exchange_capacities)
        logger.info(
            "%s (W=%d rank=%d): %d admitted eager-prefill exchange extents, "
            "%d steps/octave, %d B alignment: %s. Carrier payload %d B per row "
            "per rank per sparse layer, so those extents span %d B .. %d B; "
            "the shared exchange retains one carrier workspace per extent of "
            "the UNION of both ladders (%d extents, %d rows, %.1f MiB of "
            "send+recv carriers) against %d extents / %d rows / %.1f MiB for "
            "the capture ladder alone. Worst-case extent over-run %.2fx. The "
            "DECODE ladder is unchanged and still %s; a decode invocation "
            "never selects out of the prefill list.",
            _ICP_PREFILL_EXTENT_MARKER,
            self.icp_c,
            self.icp_rank,
            len(prefill_extents),
            ICP_PREFILL_EXTENT_STEPS_PER_OCTAVE,
            QCAPACITY_ALIGNMENT,
            prefill_extents,
            bytes_per_row,
            prefill_extents[0] * bytes_per_row,
            prefill_extents[-1] * bytes_per_row,
            len(self.icp_admitted_exchange_extents),
            sum(self.icp_admitted_exchange_extents),
            2 * sum(self.icp_admitted_exchange_extents) * bytes_per_row / 2**20,
            len(decode_only),
            sum(decode_only),
            2 * sum(decode_only) * bytes_per_row / 2**20,
            worst_ratio,
            self.icp_exchange_capacities,
        )
        self._prewarm_cute_decode(device)

    def _prewarm_cute_decode(self, device: torch.device) -> None:
        """Compile the CuTe scorer for every admitted query length, at startup.

        MSA compiles once per ``(query_len, rank, world_size, split_k)``; every
        window is a launch argument, so nothing compiles on the submission path.
        """
        import time  # noqa: PLC0415

        module = self.icp_cute_module
        start = time.monotonic()
        for query_len in sorted(self.icp_cute_query_lens):
            assert module is not None
            kwargs = dict(
                query_len=query_len,
                rank=self.icp_rank,
                world_size=self.icp_c,
                split_k=ICP_CUTE_SPLIT_K,
                device=device,
            )
            if self.icp_cute_factory_has_page_size:
                kwargs["page_size"] = self.icp_physical_page_tokens
            if self.icp_cute_factory_has_pdl:
                kwargs["use_pdl"] = self.icp_decode_pdl
            self.icp_cute_scorers[query_len] = module.get_icp_decode_scorer(**kwargs)
        if self.icp_cute_scorers:
            logger.info(
                "%s rank=%d prewarmed %d CuTe scorers (query_lens=%s) in %.1f s",
                _ICP_CUTE_STATUS_MARKER,
                self.icp_rank,
                len(self.icp_cute_scorers),
                sorted(self.icp_cute_scorers),
                time.monotonic() - start,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerMSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        positions = common_attn_metadata.positions
        assert positions is not None

        return self._build_icp(
            common_attn_metadata,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
        )

    def _build_icp(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        *,
        num_reqs: int,
        num_tokens: int,
        num_decodes: int,
        num_decode_tokens: int,
        num_prefills: int,
        num_prefill_tokens: int,
    ) -> MiniMaxM3IndexerMSAMetadata:
        """Plan one fragment-aware invocation: chunk plans + row geometry.

        Built once per forward and shared by every sparse layer.

        **Nothing here reads a device value.** Two quantities must stay
        separate:

        *Extents* -- how many blocks a row can reach, how far the scorer's work
        decomposition runs, how wide the score wave is -- come from
        ``seq_lens_cpu_upper_bound`` and the startup constants
        ``max_model_len`` / ``max_num_batched_tokens`` / ``max_num_seqs``. That
        field is exact for prefill rows and optimistic by the rejected-draft
        count on async Eagle3 decode rows, and every use here is safe to
        over-state: a wider decomposition still covers the work, and the
        selector reads only ``min(local_valid_blocks, N)`` columns per row.

        *Causality and the live bound* are NOT taken from that field. A query
        row's absolute position decides which keys its diagonal block can see,
        and the sequence length decides how far along a block-table row the
        kernel may read -- columns past the live end hold physical pages
        belonging to evicted requests. The V2 runner has no exact host sequence
        length, so both come from ``common_attn_metadata.seq_lens``: the offset
        is handed to the scorer per launch as ``q_offset_override`` and the
        length is written over the plan's own bound-derived
        ``kv_segment_lens``. The plan is sized by a bound, then masked and
        bounded by the exact value.
        """
        from fmha_sm100.icp.candidates import live_blocks  # noqa: PLC0415

        _fmha_sm100_plan = _msa_api()._fmha_sm100_plan

        c, rank, rows = self.icp_c, self.icp_rank, self.icp_rows
        block_table = common_attn_metadata.block_table_tensor
        assert num_tokens <= self.icp_token_capacity

        # ---- host extents (SIZING ONLY) -------------------------------------
        qsl_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1].to(
            torch.int64
        )
        q_begin, q_end = qsl_cpu[:-1], qsl_cpu[1:]

        # `num_tokens` is `num_actual_tokens`, which under a FULL cudagraph is
        # the PADDED graph extent and not the live count (the padding requests
        # have query length 0, which `split_decodes_and_prefills` admits as
        # decodes so `num_decodes` matches the captured size). The live count is
        # the last cumulative query offset, which `prepare_inputs` replicates
        # into every padded entry of `query_start_loc_np`, so this is exact in
        # both modes and still a host value.
        #
        # The rows in `[live, padded)` belong to no request and carry whatever
        # positions the previous step left in the runner's buffer. Marking them
        # ACTIVE would have the selector rank them from stale scores; an
        # inactive row is also what keeps a stale NaN away from the selector's
        # C5 check.
        live_tokens = int(q_end[-1]) if num_reqs > 0 else 0
        assert 0 <= live_tokens <= num_tokens

        # ---- the phase, from actual prefill state ---------------------------
        # NOT from the token count, the query length, Q > 1 or a size threshold:
        # a verification batch is decode-shaped with Q > 1 and a chunked prefill
        # can be small. It must also be identical on every rank, which the
        # per-request flag is and a local shape is not.
        is_prefilling = common_attn_metadata.is_prefilling
        if is_prefilling is None:
            raise RuntimeError(
                "refined-icp-v1 needs CommonAttentionMetadata.is_prefilling to "
                "route the candidate exchange: the phase decides the transport "
                "and must be identical on every rank, so it cannot be inferred "
                "from this rank's token count or query length."
            )
        has_prefill = bool(is_prefilling[:num_reqs].any().item())

        # The EXCHANGE extent: the smallest extent of THIS BAND's ladder that
        # holds the PADDED rows. It is also the selector's bound -- every
        # chunk's `select_extent` is `min(width, cap - t0)` -- so it decides
        # both the transport payload and how many rows the Top-16 ranks.
        #
        # Padded, not live: under a FULL cudagraph the padded extent is a
        # constant of the graph and the live count is not, and C4 requires the
        # exchange shape to be fixed across replays and equal on every rank.
        #
        # The phase bound above picks WHICH ladder -- decode's capture sizes, or
        # the eager prefill fill built in `_init_icp`, which also records why
        # each is rank-equal. A lookup over a short Python list: nothing is
        # allocated and no device value is read.
        cap = select_token_capacity(
            self.icp_prefill_exchange_capacities
            if has_prefill
            else self.icp_exchange_capacities,
            num_tokens,
        )

        # ---- the CuTe band ---------------------------------------------------
        # Rows `[0, cute_rows)` take the CuTe H4 scorer: on a pure-decode step
        # the whole invocation (or nothing), on any prefill step none (r13).
        decode_query_len = _uniform_decode_query_len(qsl_cpu, has_prefill)
        kv_bound_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if kv_bound_cpu is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: refined-icp-v1 sizes the scorer's "
                "work decomposition and the score wave from "
                "CommonAttentionMetadata.seq_lens_cpu_upper_bound. Without it "
                "the only remaining source is a device->host copy of "
                "`seq_lens`, which the accepted policy forbids on the "
                "submission path."
            )
        kv_ub = kv_bound_cpu[:num_reqs].to(torch.int64)
        use_cute_decode = decode_query_len in self.icp_cute_query_lens
        cute_rows = icp_cute_rows(
            use_cute_decode=use_cute_decode,
            has_prefill=has_prefill,
            num_tokens=num_tokens,
        )
        use_cute_decode = cute_rows > 0
        if use_cute_decode and not self._icp_cute_dtypes_checked:
            # MSA's validator raises on any dtype mismatch, but only at
            # INVOCATION -- for a decode-only model, first reached during
            # cudagraph capture, far from the cause. `positions` and
            # `active_rows` are this builder's own retained buffers; these two
            # come straight from `CommonAttentionMetadata` and are checked
            # nowhere else here.
            for _name, _tensor, _want in (
                ("query_start_loc", common_attn_metadata.query_start_loc, torch.int32),
                ("seq_lens", common_attn_metadata.seq_lens, torch.int32),
            ):
                if _tensor.dtype != _want:
                    raise RuntimeError(
                        f"MiniMax ICP CuTe decode needs {_name} as {_want}; "
                        f"CommonAttentionMetadata supplied {_tensor.dtype}."
                    )
            self._icp_cute_dtypes_checked = True

        # The first sparse KV writer fills these retained views, including all
        # inactive exchange rows. Binding these views submits no GPU work.
        forced = self.icp_forced[:cap]
        n_ordinary = self.icp_n_ordinary[:cap]

        # ---- host extents (SIZING ONLY): `kv_ub` and `q_end_list` above ----

        # ---- the ordinary block table, consumed directly --------------------
        # No packing pass: MSA reads the table itself
        # (DIRECT_TABLE_CONTRACT.md §2.1/§7). These are the three I-1/I-3/I-6
        # preconditions the direct read needs, asserted rather than assumed,
        # because every way of getting them wrong is silent:
        #
        #  * storage offset 0. `block_table_tensor` is the runner's persistent
        #    per-group table, passed unsliced, so its `data_ptr` does not move
        #    with the live request count and a FULL graph captured at a small
        #    request count keeps addressing the bytes a larger step writes. A
        #    view starting at the first live request would look identical and
        #    replay wrong.
        #  * unit column stride, so physical-page columns have ordinary addressing.
        #  * a row pitch at least `icp_max_blocks` wide, so every physical page
        #    a request can name has a column. The pitch is CAPACITY and never a
        #    bound (I-1): columns past a request's live block count hold page
        #    ids from evicted requests, which read cleanly and belong to
        #    somebody else. The live bound is the exact device length installed
        #    on the plan below.
        assert block_table.dim() == 2, block_table.shape
        assert block_table.storage_offset() == 0, (
            "MiniMax-M3 indexer: the block table must be passed at storage "
            f"offset 0, got {block_table.storage_offset()}. I-6: the base "
            "pointer has to be invariant to the live request count, or a "
            "captured graph replays against another step's first request."
        )
        table_row_stride = block_table.stride(0)
        assert block_table.stride(1) == 1, (
            "MiniMax-M3 indexer: the block table's column stride must be 1, so "
            f"physical columns are contiguous; got {block_table.stride(1)}."
        )
        assert table_row_stride >= self.icp_max_blocks, (
            f"MiniMax-M3 indexer: the block table's row pitch "
            f"{table_row_stride} is narrower than the {self.icp_max_blocks} "
            "physical pages cdiv(max_model_len, physical_page_tokens) admits; a "
            "request can name has no column."
        )
        # The pitch and the base are HOST constants only because they are
        # startup constants (I-3). Pinning them across builds makes that
        # checkable rather than assumed after the first build.
        table_identity = (block_table.data_ptr(), table_row_stride)
        if self._icp_table_identity is None:
            self._icp_table_identity = table_identity
        elif self._icp_table_identity != table_identity:
            raise RuntimeError(
                "MiniMax-M3 indexer: the block table moved between builds "
                f"({self._icp_table_identity} -> {table_identity}). The direct "
                "consumption path passes its base pointer and row pitch to the "
                "scorer as startup constants; a table that is reallocated or "
                "re-viewed per step makes every captured graph stale."
            )

        # ---- the invocation's wave, from its admitted live column extent ----
        # PREFILL ONLY. Decode keeps `icp_chunk` and the capacity-wide plane:
        # its selection extents and graph profiles are derived from that chunk
        # at startup and a captured shape may not move.
        #
        # The max of `kv_ub` over the FMHA band's requests is the longest
        # history any FMHA row can reach, on the host. It is the same quantity
        # `chunk_live_blocks` is derived from below, evaluated over the whole
        # band rather than one chunk, so the selected rung dominates every FMHA
        # chunk's advertised extent and therefore every such row's exact device
        # `icp_nvalid` -- which is what makes narrowing the plane safe rather
        # than a silent truncation (the selector clamps
        # `valid = min(nvalid, blocks)`). As r13, the max is over ALL requests,
        # decode rows of a mixed step included (they are FMHA-scored too).
        if has_prefill:
            invocation_live_blocks = min(
                live_blocks(
                    max(1, int(kv_ub.max()) if num_reqs else 1), page_size=PAGE_SIZE
                ),
                self.icp_max_blocks,
            )
            score_columns = icp_prefill_rung(
                self.icp_prefill_columns, invocation_live_blocks
            )
            width = self.icp_prefill_waves[score_columns]
            score_plane, valid_plane, prefill_partials = self.icp_prefill_views[
                score_columns
            ]
            prefill_plan = self.icp_prefill_plans[score_columns]
        else:
            score_columns = self.icp_max_blocks
            width = self.icp_chunk
            score_plane, valid_plane = self.icp_scores, self.icp_valid
            prefill_partials = self.icp_prefill_partials
            prefill_plan = self.icp_prefill_plan
        # The controls belong to the RUNG, so they are derived from the same
        # `score_columns` that chose the plan and the plane -- never from a
        # chunk's live extent, which is a device-bounded quantity. On the
        # decode band this is unused (`select_decode_candidates` takes none of
        # it) but still consistent, so the state never carries a stale pairing.
        prefill_controls = (
            _prefill_full_row_controls(score_columns)
            if self.icp_prefill_selector_mode == "full_row"
            else _ICP_PREFILL_BOUNDED_CONTROLS
        )

        # ---- one scorer + selector per window, bound once per step ---------
        # `icp_row_windows` splits the invocation into windows on the decode
        # grid (pure decode, CuTe) or the wave grid (any prefill step, FMHA).
        # Each window's producer and selector launches are bound here, so
        # a sparse layer only launches them. A window's slot is its POSITION on
        # its band's grid and never a counter over surviving windows, so plan
        # pointers do not move with the batch's shape.
        from fmha_sm100.icp.candidates import (  # noqa: PLC0415
            CandidateGeometry,
            select_decode_candidates,
            select_prefill_candidates,
        )

        fmha = _msa_api()._fmha_sm100
        chunks: list[MiniMaxM3IndexerICPChunk] = []
        for window in icp_row_windows(
            num_tokens=num_tokens,
            cap=cap,
            cute_rows=cute_rows,
            decode_chunk=self.icp_chunk,
            prefill_width=width,
            has_prefill=has_prefill,
        ):
            t0, begin, t1 = window.grid_begin, window.row_begin, window.row_end
            # Selector rows, from `begin`: a pinned decode-plan extent on the
            # decode band (enumerated at startup as `icp_select_extents`),
            # otherwise any prefix the eager prefill entry point accepts.
            extent = window.select_rows
            geometry = CandidateGeometry(
                local_valid_blocks=self.icp_nvalid[begin : begin + extent],
                forced_column=self.icp_forced_column[begin : begin + extent],
                active_rows=self.icp_active[begin : begin + extent],
                scan_block_begin=self.icp_scan_block_begin,
                global_block_stride=self.icp_global_block_stride,
            )
            out = self.icp_candidates[begin : begin + extent]
            if window.cute:
                launch_window = icp_cute_decode_window(
                    query_len=decode_query_len,
                    token_begin=t0,
                    rows=cute_rows - t0,
                    chunk=self.icp_chunk,
                    num_reqs=num_reqs,
                )
                if launch_window is None:
                    # No request reaches these rows: capture padding only,
                    # which stays invalid candidates.
                    continue
                window_rows, request_begin = launch_window[1], launch_window[2]
                # One compiled scan per query length; the window is bound as
                # launch scalars (a FULL graph records its padded extent).
                scorer = self.icp_cute_scorers[decode_query_len].bind(
                    token_begin=launch_window[0],
                    token_count=window_rows,
                    request_begin=request_begin,
                    request_count=launch_window[3],
                )
                table = block_table
                query_start_loc = common_attn_metadata.query_start_loc
                cute_seq_lens = common_attn_metadata.seq_lens
                cute_scores = self.icp_scores[:window_rows]
                cute_valid = self.icp_valid[:window_rows]
                positions = self.icp_positions[:num_tokens]
                active_rows = self.icp_active_bytes[:num_tokens]
                fused_select = None
                if window.decode_selector:
                    select = functools.partial(
                        select_decode_candidates,
                        self.icp_scores[:extent],
                        geometry,
                        plan=self.icp_decode_plans[extent],
                        workspace=self.icp_decode_workspaces[extent],
                        out=out,
                        arm=self.icp_decode_arm,
                    )
                    if self.icp_fused_decode:
                        fused_select = functools.partial(
                            _icp_fused_publish,
                            self.icp_scores[:extent],
                            geometry,
                            begin,
                        )
                chunks.append(
                    MiniMaxM3IndexerICPChunk(
                        plan=None,
                        token_begin=t0,
                        num_live_tokens=t1 - t0,
                        block_table=block_table,
                        block_table_row_stride=table_row_stride,
                        block_table_row_begin=request_begin,
                        qo_offset=self.icp_qo_offset[0, :0],
                        live_blocks=self.icp_max_blocks,
                        score_extent=self.icp_max_blocks,
                        select_extent=extent,
                        topk_num_valid_pages=geometry.local_valid_blocks,
                        icp_forced_column_v1=geometry.forced_column,
                        icp_active_rows=geometry.active_rows,
                        icp_scan_block_begin_v1=self.icp_scan_block_begin,
                        icp_global_block_stride_v1=self.icp_global_block_stride,
                        cute_scorer=scorer,
                        rank_aware_nvalid=True,
                        geometry=geometry,
                        launch_score=_bind_cute_launch(
                            scorer,
                            table,
                            query_start_loc,
                            cute_seq_lens,
                            positions,
                            active_rows,
                            cute_scores,
                            cute_valid,
                        ),
                        select=select,
                        fused_select=fused_select,
                        cute_check_args=(
                            scorer,
                            table,
                            query_start_loc,
                            cute_seq_lens,
                            positions,
                            active_rows,
                            cute_scores,
                            cute_valid,
                        ),
                    )
                )
                continue
            # ---- r13 FMHA chunk (4d16b10146 indexer_msa.py:2445-2633), unsplit
            # except: window/slot names, the v10 bound launch/select closures.
            lo = q_begin.clamp(t0, t1)
            hi = q_end.clamp(t0, t1)
            sel = (hi - lo > 0).nonzero(as_tuple=True)[0]
            n_sel = int(sel.numel())
            if n_sel == 0:
                continue
            slot_index = window.slot
            first, last = int(sel[0]), int(sel[-1])
            assert last - first + 1 == n_sel
            if n_sel > self.icp_reqs_per_slot:
                raise RuntimeError(
                    f"MiniMax-M3 indexer: chunk {slot_index} covers {n_sel} "
                    f"requests but a slot row holds {self.icp_reqs_per_slot}."
                )
            # The writer's exact slot-major causal offsets (row origin 0).
            qo_offset = self.icp_qo_offset[slot_index, :n_sel]
            kv_sel = kv_ub[sel]
            qo_offset_bound = kv_sel - q_end[sel] + lo[sel]
            last_position_bound = int((kv_sel - q_end[sel] + hi[sel] - 1).max())
            chunk_live_blocks = min(
                live_blocks(last_position_bound + 1, page_size=PAGE_SIZE),
                self.icp_max_blocks,
            )
            plan = _fmha_sm100_plan(
                (hi[sel] - lo[sel]).to(torch.int32),
                kv_sel.to(torch.int32),  # GLOBAL kv lengths; the mask needs them
                self.icp_heads_group,
                num_kv_heads=1,
                qo_offset=qo_offset_bound.to(torch.int32),
                page_size=rows,  # R: one page IS one rank fragment (ABI G13)
                output_maxscore=True,
                causal=True,
                num_kv_splits=1,
                icp_c=c,
                icp_rank=rank,
                icp_block_table=block_table,
                icp_block_table_row_stride=table_row_stride,
                icp_block_table_row_begin=first,
                device_plan=self.icp_plan_store.slot(slot_index, chunk_width=width),
            )
            score_extent = chunk_live_blocks if has_prefill else self.icp_max_blocks
            assert 1 <= score_extent <= self.icp_max_blocks
            assert score_extent <= score_columns, (
                f"chunk {slot_index} advertises {score_extent} score columns "
                f"but the invocation's plane is {score_columns} wide; the "
                "wave rung must cover every chunk's live extent."
            )
            plan["max_k_tiles"] = score_extent
            live = min(t1, live_tokens) - t0
            fused_select = None
            if window.decode_selector:
                select = functools.partial(
                    select_decode_candidates,
                    score_plane[:extent],
                    geometry,
                    plan=self.icp_decode_plans[extent],
                    workspace=self.icp_decode_workspaces[extent],
                    out=out,
                    arm=self.icp_decode_arm,
                )
                if self.icp_fused_decode:
                    fused_select = functools.partial(
                        _icp_fused_publish,
                        score_plane[:extent],
                        geometry,
                        begin,
                    )
            else:
                select = functools.partial(
                    select_prefill_candidates,
                    score_plane[:extent],
                    geometry,
                    plan=prefill_plan,
                    live_blocks=chunk_live_blocks,
                    out=out,
                    partials=prefill_partials[:extent],
                    arm=self.icp_prefill_arm,
                    **_prefill_selector_kwargs(prefill_controls),
                )
            chunks.append(
                MiniMaxM3IndexerICPChunk(
                    plan=plan,
                    token_begin=t0,
                    num_live_tokens=live,
                    block_table=block_table,
                    block_table_row_stride=table_row_stride,
                    block_table_row_begin=first,
                    qo_offset=qo_offset,
                    live_blocks=chunk_live_blocks,
                    score_extent=score_extent,
                    select_extent=extent,
                    topk_num_valid_pages=geometry.local_valid_blocks,
                    icp_forced_column_v1=geometry.forced_column,
                    icp_active_rows=geometry.active_rows,
                    geometry=geometry,
                    launch_score=_bind_fmha_launch(
                        fmha,
                        plan,
                        t0,
                        live,
                        kv_indices=block_table,
                        max_score=score_plane[:live, :, :score_extent],
                        valid_score=valid_plane[:live, :, :score_extent],
                        q_offset_override=qo_offset,
                        icp_c=c,
                        icp_rank=rank,
                    ),
                    select=select,
                    fused_select=fused_select,
                )
            )

        fused_ready = False
        fused_padding: list = []
        if self.icp_fused_decode and not has_prefill:
            fused_ready = all(chunk.fused_select is not None for chunk in chunks)
            covered = sorted(
                (chunk.token_begin, chunk.token_begin + chunk.select_extent)
                for chunk in chunks
            )
            gaps: list[tuple[int, int]] = []
            at = 0
            for cover_begin, cover_end in covered:
                if cover_begin > at:
                    gaps.append((at, cover_begin))
                at = max(at, cover_end)
            if at < cap:
                gaps.append((at, cap))
            for gap_begin, gap_end in gaps:
                for p0 in range(gap_begin, gap_end, self.icp_chunk):
                    pad_rows = min(self.icp_chunk, gap_end - p0)
                    # Rows no window selects are inactive (the first writer
                    # invalidates them), so the selector reads no score and
                    # publishes (-inf, -1) records.
                    pad_geometry = CandidateGeometry(
                        local_valid_blocks=self.icp_nvalid[p0 : p0 + pad_rows],
                        forced_column=self.icp_forced_column[p0 : p0 + pad_rows],
                        active_rows=self.icp_active[p0 : p0 + pad_rows],
                        scan_block_begin=self.icp_scan_block_begin,
                        global_block_stride=self.icp_global_block_stride,
                    )
                    fused_padding.append(
                        functools.partial(
                            _icp_fused_publish,
                            self.icp_scores[:pad_rows],
                            pad_geometry,
                            p0,
                        )
                    )

        icp = MiniMaxM3IndexerICPState(
            chunks=chunks,
            scores=score_plane,
            valid=valid_plane,
            candidates=self.icp_candidates,
            forced=forced,
            n_ordinary=n_ordinary,
            prefill_plan=prefill_plan,
            prefill_partials=prefill_partials,
            decode_plans=self.icp_decode_plans,
            decode_workspaces=self.icp_decode_workspaces,
            has_prefill=has_prefill,
            token_capacity=cap,
            chunk_width=width,
            decode_arm=self.icp_decode_arm,
            prefill_arm=self.icp_prefill_arm,
            prefill_controls=prefill_controls,
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            # Sliced to the invocation, like `index_query[:num_tokens]` at the
            # call site and like `icp_nvalid`/`icp_active` below. The scorer's
            # compiled signature binds index_q, positions and active_rows to a
            # single symbol, so a capacity-length plane is rejected at bind
            # time. A prefix slice allocates nothing and starts at offset 0, so
            # `data_ptr()` is the retained buffer's and a captured graph sees
            # the same address it saw at capture.
            positions=self.icp_positions[:num_tokens],
            # uint8 retype of `icp_active`, same storage: this field feeds only
            # the CuTe scorer, whose ABI takes bytes. The selector's own
            # activity plane (`icp_active_rows` below) stays bool.
            active_rows=self.icp_active_bytes[:num_tokens],
            cute_rows=cute_rows,
            # `view`, not a conversion: an id above 2**24 does not survive a
            # float cast, and both carrier words are bitcast (ABI C1/C2).
            exchange_candidates=self.icp_candidates[:cap].view(torch.int32),
            fused_ready=fused_ready,
            fused_padding=tuple(fused_padding),
            decode_chunk=self.icp_chunk,
            fused_decode=self.icp_fused_decode,
        )
        return MiniMaxM3IndexerMSAMetadata(
            step_n_valid=self.icp_global_nvalid[:num_tokens],
            live_metadata=ICPLiveMetadata(
                query_start_loc=common_attn_metadata.query_start_loc,
                seq_lens=common_attn_metadata.seq_lens,
                num_reqs=num_reqs,
                positions=self.icp_positions[:cap],
                active=self.icp_active[:cap],
                local_nvalid=self.icp_nvalid[:cap],
                local_forced=self.icp_forced_column[:cap],
                global_nvalid=self.icp_global_nvalid[:cap],
                forced=forced,
                n_ordinary=n_ordinary,
                candidates=self.icp_candidates[:cap],
                qo_offsets=self.icp_qo_offset[: _cdiv(cap, width)],
                chunk_width=width,
                device_plan=(
                    self.icp_plan_store
                    if any(chunk.plan is not None for chunk in chunks)
                    else None
                ),
            ),
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            topk_num_valid_pages=self.icp_nvalid[:num_tokens],
            icp_forced_column_v1=self.icp_forced_column[:num_tokens],
            icp_active_rows=self.icp_active[:num_tokens],
            icp=icp,
        )


class MiniMaxM3IndexerMSAImpl(MiniMaxM3IndexerImpl):
    """Score rank-local fragments, then select and exchange block candidates."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerMSABackend

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._icp_debug_checks = _debug_checks_enabled()
        from fmha_sm100.icp.scorer.prefill import api  # noqa: PLC0415

        # The ICP score wave's output ABI. Version 1 signalled emptiness in band
        # with -inf, which is a legal representable score, so a consumer written
        # against it cannot tell "scored, maximum is -inf" from "past the end".
        if api._FMHA_ICP_SCORE_ABI_VERSION != 2:
            raise RuntimeError(
                f"{self.prefix}: this fmha_sm100 advertises ICP score ABI "
                f"version {api._FMHA_ICP_SCORE_ABI_VERSION}; refined-icp-v1 "
                "requires version 2, the out-of-band uint8 validity plane."
            )
        # Derived from the overlay's kernel half, not declared: a Python half
        # that advertises the plane over a kernel half that ignores it hands
        # back a buffer the kernel never wrote, and uint8 has no spare value
        # that would make that visible.
        if not api._FMHA_HAS_ICP_VALIDITY_PLANE:
            raise RuntimeError(
                f"{self.prefix}: this fmha_sm100 overlay's kernel half does "
                "not carry ptr_ValidScore, so the ICP score wave's validity "
                "plane would come back unwritten."
            )

    def forward(
        self,
        index_query: torch.Tensor,
        *,
        index_kv: torch.Tensor,
        index_md: MiniMaxM3IndexerMetadata | None,
        icp_c: int = 1,
        icp_rank: int = 0,
        layer_idx: int = 0,
        candidate_exchange: Any = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if index_md is None:
            return None, None  # profiling run; caches unbound
        assert (icp_c, icp_rank) == (self.icp_c, self.icp_rank), (
            f"{self.prefix}: the fragment geometry this impl was built for "
            f"(W={self.icp_c}, rank={self.icp_rank}) is not the one the layer "
            f"passed (W={icp_c}, rank={icp_rank}); the plans, the page lists "
            "and the head routing are all keyed on it."
        )
        return self._forward_icp(
            index_query,
            index_kv=index_kv,
            index_md=index_md,
            layer_idx=layer_idx,
            candidate_exchange=candidate_exchange,
        )

    def _check_icp_invocation(self, index_query, md, buf) -> None:
        """Per-invocation shape contracts; see ``_forward_icp`` for gating."""
        assert isinstance(md, MiniMaxM3IndexerMSAMetadata)
        st = md.icp
        assert st is not None
        assert buf is not None
        assert st.token_capacity <= buf.shape[0], (
            f"{self.prefix}: the shared top-k buffer holds {buf.shape[0]} rows "
            f"but this invocation's exchange extent is {st.token_capacity}; "
            "the merged ids are written straight into a prefix of it, so the "
            "buffer must be at least as tall as the extent."
        )
        heads_group = st.candidates.shape[1]
        head_dim = self.index_head_dim
        if index_query.shape[-1] != heads_group * head_dim:
            raise RuntimeError(
                f"{self.prefix}: refined-icp-v1 scores ALL {heads_group} of the "
                "group's index-query heads on every rank and routes head h to "
                f"rank h // {self.num_index_heads} (ABI G5/C9), but this rank's "
                f"index_query carries {index_query.shape[-1] // head_dim} head"
                "(s). The fused QKV projection still shards index_q like the KV "
                "heads; it has to replicate the index-Q group across the W ICP "
                "ranks (MinimaxM3QKVParallelLinearWithIndexer) before W > 1 can "
                "serve."
            )

    def _check_icp_chunk(self, chunk, index_q, index_kv) -> None:
        """Per-chunk launch contracts; see ``_forward_icp`` for gating."""
        if chunk.launch_score is None or chunk.select is None:
            raise RuntimeError(
                f"{self.prefix}: an ICP chunk reached the forward without its "
                "bound launches; the metadata builder binds them once per step."
            )
        if chunk.cute_scorer is None:
            return
        # The no-init ABI2 contract: the scorer writes only the exact live
        # prefix of a `torch.empty` plane, so the selector's read bound must be
        # the rank-aware bound equal to that prefix.
        assert chunk.rank_aware_nvalid, (
            "CuTe decode chunk carries a non-rank-aware topk_num_valid_pages; "
            "the no-init scorer leaves tails undefined and the selector would "
            "rank uninitialised memory."
        )
        # MSA's own host metadata check for exactly the launch the forward
        # makes unchecked (`launch`), with the same bound window.
        scorer, *tensors = chunk.cute_check_args
        scorer.validate(index_q, index_kv, *tensors)

    def _forward_icp(
        self,
        index_query: torch.Tensor,
        *,
        index_kv: torch.Tensor,
        index_md: MiniMaxM3IndexerMetadata,
        layer_idx: int,
        candidate_exchange: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Score this rank's fragment, select locally, exchange exactly once.

        Every launch argument was bound by the builder once per step; a layer
        only launches. The per-layer validators run on the first sparse layer
        of a step, or on every layer under ``VLLM_MINIMAX_ICP_DEBUG_CHECKS=1``.
        """
        md = index_md
        assert isinstance(md, MiniMaxM3IndexerMSAMetadata)
        st = md.icp
        assert st is not None
        if candidate_exchange is None:
            raise RuntimeError(
                f"{self.prefix}: at W={self.icp_c} local candidates cover only "
                "this rank's index ownership. Global selection is complete "
                "after the candidate merge. Bind the exchange "
                "with `bind_candidate_exchange` before serving."
            )
        buf = self.topk_indices_buffer
        assert buf is not None
        debug = self._icp_debug_checks
        if debug or not st.checked:
            self._check_icp_invocation(index_query, md, buf)
            st.checked = True

        heads_group = st.candidates.shape[1]
        num_tokens = md.num_actual_tokens
        index_q = index_query[:num_tokens].view(-1, heads_group, self.index_head_dim)
        # `unsqueeze`, not `view`: the index region of a compound page carries
        # the *compound* page stride, so `view` would reject it. FMHA takes
        # [num_pages, 1, R, D]; CuTe the rank-three view and its stride.
        k_pages = index_kv.unsqueeze(1)

        # CuTe ABI2 writes only each active row's exact readable prefix and the
        # selector reads only that prefix; the FMHA producer completes its
        # advertised extent. Both store raw unscaled QK maxima, and each
        # window's selector consumes only rows its own producer just wrote.
        # D3: pure decode publishes from the selectors and the exchange below
        # only merges. A D3 exchange with a step that cannot publish every
        # row would time out, so that is refused here instead.
        _log_decode_profile(
            self.icp_rank, candidate_exchange, st.decode_chunk, st.fused_decode
        )
        fused = not st.has_prefill and getattr(
            candidate_exchange, "fused_decode", False
        )
        if fused and not st.fused_ready:
            raise RuntimeError(
                f"{self.prefix}: the D3 exchange is bound but this decode step "
                "has a window without a fused selector binding; build the "
                f"metadata with the same {FUSED_DECODE_ENV} on every rank."
            )
        for chunk in st.chunks:
            # The selectors' own shape validation rides the same gate: the
            # first layer of a step validates, later layers take the
            # pre-validated native launch.
            validate = debug or not chunk.checked
            if validate:
                self._check_icp_chunk(chunk, index_q, index_kv)
                chunk.checked = True
            if chunk.cute_scorer is not None:
                _note_cute_engagement(self.icp_rank, chunk)
            assert chunk.launch_score is not None and chunk.select is not None
            chunk.launch_score(index_q, index_kv, k_pages, self.scale)
            if fused:
                assert chunk.fused_select is not None
                chunk.fused_select(candidate_exchange, layer_idx)
            else:
                chunk.select(validate=validate)
        if fused:
            for publish in st.fused_padding:
                publish(candidate_exchange, layer_idx)

        # ONE exchange, after every chunk has finished -- never inside the loop.
        # `view`, not a conversion: an id above 2**24 does not survive a float
        # cast, and both carrier words are bitcast by contract (ABI C1/C2).
        #
        # The extent is the builder's selected profile, not the global capacity.
        # Rows above it are never written here and nothing reads them: the
        # attend slices `[:num_actual_tokens]`, which the profile bounds by
        # construction.
        # Both views are bound once per step: the candidates by the builder,
        # the output on the first layer that sees the shared top-k buffer.
        if st.exchange_buf is not buf:
            st.exchange_buf = buf
            st.exchange_out = buf[: st.token_capacity]
        candidate_exchange(
            st.exchange_candidates,
            layer_idx=layer_idx,
            has_prefill=st.has_prefill,
            forced=st.forced,
            n_ordinary=st.n_ordinary,
            out=st.exchange_out,
        )

        # The attend reads ``buf`` directly; this return is vestigial.
        return None, None
