# SPDX-License-Identifier: Apache-2.0
import ctypes
import hashlib
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_kv_cache import HostKVCache
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_kv_layout import (
    ComponentSpec,
    KVObjectLayout,
    ObjectRef,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_token_resolver import (
    HostTokenResolver,
    object_ref_from_wire,
    object_ref_to_wire,
)


class Allocation:
    def __init__(self, size):
        self.data = bytearray(size)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.data))
        self.closed = False

    @property
    def buffer(self):
        return memoryview(self.data)

    def close(self):
        self.closed = True


class Backend:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def get_host_buffers(self, keys):
        self.calls += 1
        return [self.payload for _ in keys]


def object_fixture(start=128):
    # Two layers, distinct row lengths and bytes, plus real header padding.
    blobs = [bytes(range(16)), bytes(range(40, 48)), bytes(range(80, 96)), bytes(range(120, 128))]
    offsets = [64]
    for blob in blobs:
        offsets.append(offsets[-1] + len(blob))
    payload = bytearray(struct.pack("<6Q", 4, *offsets) + bytes(16) + b"".join(blobs))
    specs = tuple(
        ComponentSpec(
            f"layer{index // 2}", "latent" if index % 2 == 0 else "rope", index, "bf16", len(blob) // 4, len(blob) // 4
        )
        for index, blob in enumerate(blobs)
    )
    layout = KVObjectLayout("model-revision", 0, "replicated-main", specs)
    ref = ObjectRef(
        "key1",
        layout,
        "immutable-1",
        start,
        4,
        len(payload),
        tuple(map(len, blobs)),
        payload_digest=hashlib.sha256(b"".join(blobs)).hexdigest(),
    )
    return payload, ref


def test_owned_copy_offset_and_same_object_hit():
    payload, ref = object_fixture()
    backend = Backend(payload)
    cache = HostKVCache(backend, Allocation, 4096)
    lease = cache.acquire([ref])[0]
    span = lease.resolve(ref, "layer1", "latent", 130)
    assert ctypes.string_at(span.address, span.nbytes) == bytes(range(88, 92))
    assert lease.address != ctypes.addressof(ctypes.c_char.from_buffer(payload))
    payload[:] = bytes(len(payload))  # SDK ownership/storage can disappear after copy.
    lease.release()
    with cache.acquire([ref])[0] as second:
        row = second.resolve(ref, "layer1", "rope", 131)
        assert ctypes.string_at(row.address, row.nbytes) == bytes([126, 127])
    assert backend.calls == 1
    with pytest.raises(RuntimeError, match="released"):
        _ = span.address
    cache.close()


@pytest.mark.parametrize("field,offset,value", [("count", 0, 99), ("header", 8, 8), ("end", 40, 999)])
def test_corrupt_offset_table_is_never_published(field, offset, value):
    payload, ref = object_fixture()
    struct.pack_into("<Q", payload, offset, value)
    cache = HostKVCache(Backend(payload), Allocation, 4096)
    with pytest.raises(ValueError):
        cache.acquire([ref])
    assert cache.used_bytes == 0
    cache.close()


def test_same_size_blob_swap_fails_payload_identity():
    payload, ref = object_fixture()
    payload[64:80], payload[88:104] = payload[88:104], payload[64:80]
    cache = HostKVCache(Backend(payload), Allocation, 4096)
    with pytest.raises(ValueError, match="payload identity"):
        cache.acquire([ref])


def test_pinned_entry_budget_and_conflicting_versions():
    payload, ref = object_fixture()
    cache = HostKVCache(Backend(payload), Allocation, len(payload))
    lease = cache.acquire([ref])[0]
    with pytest.raises(RuntimeError, match="active"):
        cache.clear_when_idle()
    with pytest.raises(ValueError, match="identity"):
        cache.acquire([replace(ref, content_identity="another-version")])
    with pytest.raises(MemoryError):
        cache.acquire([replace(ref, key="second-key")])
    lease.release()
    cache.clear_when_idle()
    assert cache.used_bytes == 0


def test_single_flight_and_failure_wake_waiters():
    payload, ref = object_fixture()
    entered, proceed = threading.Event(), threading.Event()

    class BlockingBackend(Backend):
        def get_host_buffers(self, keys):
            entered.set()
            assert proceed.wait(5)
            return super().get_host_buffers(keys)

    backend = BlockingBackend(payload)
    cache = HostKVCache(backend, Allocation, 4096)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(cache.acquire, [ref])
        assert entered.wait(5)
        second = pool.submit(cache.acquire, [ref])
        proceed.set()
        leases = first.result(5) + second.result(5)
    assert backend.calls == 1
    for lease in leases:
        lease.release()
    cache.close()


def test_missing_and_allocation_failure_do_not_consume_budget():
    payload, ref = object_fixture()
    cache = HostKVCache(Backend(None), Allocation, 4096)
    with pytest.raises(RuntimeError, match="missing"):
        cache.acquire([ref])
    assert cache.used_bytes == 0
    cache.close()


def test_request_ranges_and_wire_identity():
    payload, ref = object_fixture(start=0)
    assert object_ref_from_wire(object_ref_to_wire(ref)) == ref
    wire = object_ref_to_wire(ref)
    wire["layout"]["components"][0]["layer_name"] = "swapped-layer"
    with pytest.raises(ValueError, match="layout_id"):
        object_ref_from_wire(wire)
    # Resolver registers single-layer producer objects; core cache permits multi-layer objects.
    single = replace(ref.layout, components=ref.layout.components[:2])
    ref = replace(ref, layout=single, blob_sizes=ref.blob_sizes[:2], payload_digest=None)
    resolver = HostTokenResolver("model-revision")
    resolver.register("request1", "layer0", [ref])
    assert resolver.resolve("request1", "layer0", 2) == ref
    with pytest.raises(ValueError):
        resolver.resolve("request1", "layer0", 4)
    with pytest.raises(ValueError, match="replaced"):
        resolver.register("request1", "layer0", [replace(ref, key="other")])
    resolver.discard("request1")
    with pytest.raises(RuntimeError, match="not published"):
        resolver.resolve("request1", "layer0", 2)


def test_failed_single_flight_wakes_waiter_and_retry_is_clean():
    payload, ref = object_fixture()
    entered, proceed = threading.Event(), threading.Event()

    class BrokenBackend(Backend):
        def get_host_buffers(self, keys):
            entered.set()
            assert proceed.wait(5)
            raise RuntimeError("injected Get failure")

    cache = HostKVCache(BrokenBackend(payload), Allocation, 4096)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(cache.acquire, [ref])
        assert entered.wait(5)
        second = pool.submit(cache.acquire, [ref])
        proceed.set()
        for future in (first, second):
            with pytest.raises(RuntimeError):
                future.result(5)
    assert cache.used_bytes == 0
    cache.close()
