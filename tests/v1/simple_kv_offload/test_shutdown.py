# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for DMA queue shutdown and registered host-buffer ownership."""

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.simple_kv_offload import copy_backend
from vllm.v1.simple_kv_offload import worker as worker_module
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker


def test_shutdown_drains_unpublished_queue_then_device_work(monkeypatch):
    """A queued copy and a submitted copy must both finish before shutdown returns."""
    copy_entered = threading.Event()
    copy_release = threading.Event()
    sync_entered = threading.Event()
    sync_release = threading.Event()
    shutdown_done = threading.Event()
    copied = []
    synced = []
    backend = DmaCopyBackend()

    class Stream:
        def synchronize(self):
            assert backend._thread is not None
            assert not backend._thread.is_alive()
            sync_entered.set()
            assert sync_release.wait(timeout=5)
            synced.append(self)

    class Event:
        def record(self, stream):
            self.stream = stream

    def copy(src, dst, params):
        if src == [0]:
            copy_entered.set()
            assert copy_release.wait(timeout=5)
        copied.append((src, dst))

    def shutdown():
        backend.shutdown()
        shutdown_done.set()

    monkeypatch.setattr(copy_backend, "build_params", lambda *a, **kw: object())
    monkeypatch.setattr(copy_backend, "copy_blocks", copy)
    monkeypatch.setattr(
        copy_backend.current_platform, "set_device", lambda device: None
    )
    monkeypatch.setattr(copy_backend.torch, "Event", Event)
    load, store = Stream(), Stream()
    backend.init({}, {}, torch.device("cpu"), load, store)
    events: list[tuple[int, torch.Event]] = []
    closer = threading.Thread(target=shutdown)
    try:
        backend.launch_copy([0], [1], True, 0, events)
        assert copy_entered.wait(timeout=5)
        backend.launch_copy([2], [3], False, 1, events)
        assert events == []
        closer.start()
        assert not shutdown_done.wait(timeout=0.05)
        copy_release.set()
        assert sync_entered.wait(timeout=5)
        assert copied == [([0], [1]), ([2], [3])]
        assert [event_idx for event_idx, _ in events] == [0, 1]
        assert not shutdown_done.is_set()
        with pytest.raises(RuntimeError, match="after DMA shutdown"):
            backend.launch_copy([4], [5], True, 2, events)
        sync_release.set()
        closer.join(timeout=5)
        assert not closer.is_alive()
        assert shutdown_done.is_set()
        assert synced == [load, store]
        backend.shutdown()
    finally:
        copy_release.set()
        sync_release.set()
        if closer.ident is not None:
            closer.join(timeout=5)
        backend.shutdown()


@pytest.mark.parametrize("failure", [None, "drain", "unregister"])
def test_connector_shutdown_keeps_owners_until_drain_and_unregistration(
    monkeypatch, failure
):
    """Cleanup retries retain owners and unregister each successful pin once."""
    worker = SimpleCPUOffloadWorker(None, None, cpu_capacity_bytes=64)
    worker.num_cpu_blocks = 2
    worker.gpu_kv_caches = {
        "main": torch.zeros((2, 8), dtype=torch.int8),
        "index": torch.zeros((2, 4), dtype=torch.int8),
    }
    transitions = []
    registered = set()
    unregistered: list[int] = []
    should_fail = failure

    class Backend:
        def init(self, *args):
            pass

        def shutdown(self):
            nonlocal should_fail
            assert worker.gpu_kv_caches is not None
            assert worker.cpu_kv_caches is not None
            if should_fail == "drain":
                should_fail = None
                raise RuntimeError("drain failed")
            transitions.append("drain")

    def pin(tensor):
        registered.add(tensor.data_ptr())

    def unpin(tensor):
        nonlocal should_fail
        assert "events" in transitions
        assert worker.gpu_kv_caches is not None
        assert worker.cpu_kv_caches is not None
        if should_fail == "unregister" and unregistered:
            should_fail = None
            raise RuntimeError("unregister failed")
        unregistered.append(tensor.data_ptr())

    monkeypatch.setattr(worker_module, "PIN_MEMORY", True)
    monkeypatch.setattr(worker_module, "pin_tensor", pin)
    monkeypatch.setattr(worker_module, "unpin_tensor", unpin)
    monkeypatch.setattr(worker_module, "DmaCopyBackend", Backend)
    worker._init_cpu_mode(worker.gpu_kv_caches, 12, torch.device("cpu"))
    worker._load_events.append(
        (3, SimpleNamespace(synchronize=lambda: transitions.append("events")))
    )
    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    connector.worker_handler = worker
    if failure is not None:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            connector.shutdown()
        assert worker.gpu_kv_caches is not None
        assert worker.cpu_kv_caches is not None
        assert worker._registered_cpu_tensors
        assert not worker._shutdown_complete
    connector.shutdown()
    connector.shutdown()
    assert set(unregistered) == registered
    assert len(unregistered) == len(registered) == 2
    assert worker._registered_cpu_tensors == []
    assert worker.gpu_kv_caches is None
    assert worker.cpu_kv_caches is None
    assert worker._backend is None
    assert worker._load_events == []
    with pytest.raises(RuntimeError, match="after CPU offload shutdown"):
        worker.bind_connector_metadata(SimpleCPUOffloadMetadata())
    with pytest.raises(RuntimeError, match="after CPU offload shutdown"):
        worker.register_kv_caches({})


def test_partial_cpu_registration_is_unregistered_at_shutdown(monkeypatch):
    """Initialization failures must preserve the successfully registered owners."""
    worker = SimpleCPUOffloadWorker(None, None, cpu_capacity_bytes=64)
    worker.num_cpu_blocks = 2
    gpu = {name: torch.zeros((2, 4), dtype=torch.int8) for name in ("main", "index")}
    pins: list[int] = []
    unpins = []

    def pin(tensor):
        if pins:
            raise RuntimeError("registration failed")
        pins.append(tensor.data_ptr())

    monkeypatch.setattr(worker_module, "PIN_MEMORY", True)
    monkeypatch.setattr(worker_module, "pin_tensor", pin)
    monkeypatch.setattr(
        worker_module, "unpin_tensor", lambda tensor: unpins.append(tensor.data_ptr())
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        worker._init_cpu_mode(gpu, 8, torch.device("cpu"))
    assert len(worker._registered_cpu_tensors) == 1
    worker.shutdown()
    assert unpins == pins
