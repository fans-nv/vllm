# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve vLLM policy for the package-owned ICP candidate transport.

Imports of the optional kernel package remain lazy. Extent planning, retained
workspaces and D3/K5T/NCCL submission belong to ``fmha_sm100.icp``; this adapter
resolves environment flags and logs the admitted model configuration.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.models.minimax_m3.nvidia.msa_icp import require_msa_icp

if TYPE_CHECKING:
    from fmha_sm100.icp.candidate_exchange import CandidateExchange

logger = init_logger(__name__)

# Mirrors checked against the package during opted-in indexer initialization.
CAND_K = 16
QCAPACITY_ALIGNMENT = 4
ICP_PREFILL_EXTENT_STEPS_PER_OCTAVE = 2
ICP_EXCHANGE_SLOTS = 3
ACK_FREE_SLOTS = 3
BACKEND_NCCL = "nccl"
BACKEND_K5T = "k5t"

FUSED_DECODE_ENV = "VLLM_MINIMAX_ICP_DECODE_FUSED_EXCHANGE"
CLASSIC_MERGE_ENV = "VLLM_MINIMAX_ICP_CLASSIC_MERGE"


def _flag(name: str, default: bool) -> bool:
    import os  # noqa: PLC0415

    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    raise ValueError(f"{name}={raw!r} must be 0 or 1")


def fused_decode_requested() -> bool:
    """Whether the decode route is D3 rather than K5T (startup constant)."""
    return _flag(FUSED_DECODE_ENV, True)


def decode_merge_mode() -> str:
    """`network` (shuffle-network merge, default) or `classic` (32-round)."""
    return "classic" if _flag(CLASSIC_MERGE_ENV, False) else "network"


def select_backend(has_prefill: bool) -> str:
    """NCCL (native pack + all_to_all + K2) for prefill/mixed, K5T for decode."""
    return BACKEND_NCCL if has_prefill else BACKEND_K5T


def exchange_token_capacities(
    *, max_token_capacity: int, cudagraph_capture_sizes: Iterable[int] = ()
) -> list[int]:
    from fmha_sm100.icp.exchange_plan import exchange_token_capacities as plan

    return plan(
        max_token_capacity=max_token_capacity,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )


def prefill_exchange_token_capacities(
    *, max_token_capacity: int, steps_per_octave: int | None = None
) -> list[int]:
    from fmha_sm100.icp.exchange_plan import prefill_exchange_token_capacities as plan

    return plan(
        max_token_capacity=max_token_capacity, steps_per_octave=steps_per_octave
    )


def admitted_exchange_extents(
    *,
    max_token_capacity: int,
    cudagraph_capture_sizes: Iterable[int] = (),
    steps_per_octave: int | None = None,
) -> list[int]:
    from fmha_sm100.icp.exchange_plan import admitted_exchange_extents as plan

    return plan(
        max_token_capacity=max_token_capacity,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
        steps_per_octave=steps_per_octave,
    )


def select_token_capacity(capacities: Sequence[int], num_tokens: int) -> int:
    from fmha_sm100.icp.exchange_plan import select_token_capacity as select

    return select(capacities, num_tokens)


def check_slot_discipline(num_sparse_layers: int, slots: int) -> None:
    from fmha_sm100.icp.exchange_plan import check_slot_discipline as check

    check(num_sparse_layers, slots)


def create_candidate_exchange(
    *,
    group: dist.ProcessGroup,
    world_size: int,
    rank: int,
    num_heads_local: int,
    num_index_heads: int,
    num_sparse_layers: int,
    token_capacities: Sequence[int],
    device: torch.device,
) -> CandidateExchange:
    """Construct the package owner after resolving vLLM's decode policy."""
    require_msa_icp()
    from fmha_sm100.icp.candidate_exchange import CandidateExchange

    exchange = CandidateExchange(
        group=group,
        world_size=world_size,
        rank=rank,
        num_heads_local=num_heads_local,
        num_index_heads=num_index_heads,
        num_sparse_layers=num_sparse_layers,
        token_capacities=token_capacities,
        device=device,
        fused_decode=fused_decode_requested(),
        use_pdl=_flag("VLLM_MINIMAX_ICP_DECODE_PDL", True),
        classic_merge=decode_merge_mode() == "classic",
        slots=ICP_EXCHANGE_SLOTS,
    )
    provenance = exchange.decode_provenance
    bytes_per_row = num_index_heads * CAND_K * 2 * 4
    route = "D3 select/publish + merge" if exchange.fused_decode else "K5T push/merge"
    logger.info("MiniMax-M3 ICP decode exchange route: %s (%s)", route, provenance)
    logger.info(
        "MiniMax-M3 ICP candidate exchange admitted: prefill/mixed -> NCCL "
        "(native pack + all_to_all + K2), decode -> %s; one final merge per "
        "sparse layer, no ACK, slots=%d, CUDA symmetric-memory backend. "
        "W=%d rank=%d H_group=%d H_local=%d layers=%d. ONE window at T=%d "
        "(%s MiB, tile=%d tokens, resident CTA cap=%d); admitted extents "
        "(tokens -> payload bytes/layer): %s. Retained NCCL carriers: %.2f MiB.",
        route,
        exchange.slots,
        world_size,
        rank,
        num_index_heads,
        num_heads_local,
        num_sparse_layers,
        exchange.token_capacity,
        provenance["symm_mib"],
        provenance["tile_tok"],
        provenance["resident_cap"],
        ", ".join(f"{t} -> {t * bytes_per_row}" for t in token_capacities),
        2 * sum(token_capacities) * bytes_per_row / 2**20,
    )
    return exchange
