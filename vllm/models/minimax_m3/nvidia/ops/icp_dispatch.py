# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve vLLM policy for the package-owned ICP candidate transport.

Imports of the optional kernel package remain lazy. Extent planning, retained
workspaces and D3/K5T/NCCL submission belong to ``fmha_sm100.icp``; this adapter
fixes the serving policy (D3, PDL, network merge) and logs the configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.models.minimax_m3.nvidia.msa_icp import require_msa_icp

if TYPE_CHECKING:
    from fmha_sm100.icp.candidate_exchange import CandidateExchange

logger = init_logger(__name__)

CAND_K = 16
ICP_EXCHANGE_SLOTS = 3


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
    """Construct the package owner with the qualified decode policy."""
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
        fused_decode=True,
        use_pdl=True,
        classic_merge=False,
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
