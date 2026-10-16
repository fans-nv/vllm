# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The first sparse layer's KV write also writes the step's ICP live metadata."""

from collections.abc import Callable
from typing import Any

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context


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
        step = getattr(metadata[layer_name], "icp", None)
        if step is None:
            raise RuntimeError("ICP KV writer requires the step's ICP plan")
        writer(**writer_args, **step.writer_args)

    launch_owner(**kwargs)
