# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable views written by the first sparse layer's native KV producer."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context

if TYPE_CHECKING:
    from vllm.models.minimax_m3.nvidia.msa_icp_main import MiniMaxM3SparseICPMetadata


@dataclass(frozen=True)
class ICPLiveMetadata:
    """One invocation's views; construction and binding perform no device work.

    Row outputs cover the selected exchange extent, including its padding.
    Their storage belongs to the metadata builder and outlives graph replay.
    The native producer derives every value from exact device query offsets
    and sequence lengths in the same launch that writes compound KV pages.
    """

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    num_reqs: int
    positions: torch.Tensor
    active: torch.Tensor
    local_nvalid: torch.Tensor
    local_forced: torch.Tensor
    global_nvalid: torch.Tensor
    forced: torch.Tensor
    n_ordinary: torch.Tensor
    candidates: torch.Tensor
    qo_offsets: torch.Tensor
    chunk_width: int
    # The `IcpDevicePlanStore` whose slots this invocation's FMHA plans view;
    # None when no chunk takes FMHA.
    device_plan: object | None = None

    def writer_kwargs(self) -> dict[str, Any]:
        return dict(
            write_icp_metadata=True,
            icp_query_start_loc=self.query_start_loc,
            icp_seq_lens=self.seq_lens,
            icp_num_reqs=self.num_reqs,
            icp_positions=self.positions,
            icp_active=self.active,
            icp_local_nvalid=self.local_nvalid,
            icp_local_forced=self.local_forced,
            icp_global_nvalid=self.global_nvalid,
            icp_forced=self.forced,
            icp_n_ordinary=self.n_ordinary,
            icp_candidates=self.candidates,
            icp_qo_offsets=self.qo_offsets,
            icp_chunk_width=self.chunk_width,
            # Every FMHA window starts at its slot origin (no CuTe band).
            icp_plan_row_begin=0,
            icp_device_plan=self.device_plan,
        )


def launch_icp_kv_write(
    layer_name: str,
    is_metadata_owner: bool,
    writer: Callable[..., Any],
    kwargs: dict[str, Any],
) -> None:
    """Fuse live geometry into the first sparse writer, including graph replay.

    FULL capture records this native launch with stable metadata addresses.
    PIECEWISE capture has no attention metadata; its first sparse writer is an
    eager segment so replay binds the current prefill/mixed invocation. Other
    sparse writers keep the metadata-free native specialization.
    """
    if not is_metadata_owner:
        writer(**kwargs)
        return

    @eager_break_during_capture
    def launch_owner(**writer_args: Any) -> None:
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            writer(**writer_args)
            return
        layer_metadata = cast("MiniMaxM3SparseICPMetadata", metadata[layer_name])
        indexer = layer_metadata.indexer
        live = getattr(indexer, "live_metadata", None)
        if not isinstance(live, ICPLiveMetadata):
            raise RuntimeError("ICP KV writer requires native live metadata")
        writer(**writer_args, **live.writer_kwargs())

    launch_owner(**kwargs)
