# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def shutdown_model_resources(models: Iterable[nn.Module | None]) -> None:
    """Close model-owned resources after draining graphs, before TP teardown.

    This is deliberately separate from temporary KV profiling-cache release.
    Shared modules run the optional hook once even if both model roots own them.
    """
    seen: set[int] = set()
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            close = getattr(module, "shutdown_model_resources", None)
            if callable(close):
                close()


def free_before_shutdown(vllm_config: VllmConfig) -> None:
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
    from vllm.v1.worker.workspace import reset_workspace_manager

    cache_config = vllm_config.cache_config
    cache_config.num_gpu_blocks = None

    compilation_config = vllm_config.compilation_config
    compilation_config.static_forward_context.clear()

    _ROPE_DICT.clear()
    reset_workspace_manager()
