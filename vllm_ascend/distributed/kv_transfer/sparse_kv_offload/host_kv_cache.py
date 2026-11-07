# SPDX-License-Identifier: Apache-2.0
"""Owned host copies of immutable Yuanrong objects, with single-flight loading."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_kv_layout import (
    UINT64_MAX,
    ComponentView,
    ObjectRef,
    validate_composed_buffer,
)


class HostAllocation(Protocol):
    @property
    def address(self) -> int: ...
    @property
    def buffer(self) -> memoryview: ...
    def close(self) -> None: ...


class HostBufferBackend(Protocol):
    def get_host_buffers(self, keys: list[str]) -> Sequence[object | None]: ...


@dataclass
class _Entry:
    ref: ObjectRef
    allocation: HostAllocation
    components: dict[tuple[str, str], ComponentView]
    pins: int = 0


@dataclass
class _Flight:
    ref: ObjectRef
    event: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass(frozen=True)
class HostSpan:
    lease: HostEntryLease
    offset: int
    nbytes: int

    @property
    def address(self) -> int:
        address = self.lease.address + self.offset
        if address > UINT64_MAX or self.nbytes > UINT64_MAX - address:
            raise ValueError("host pointer arithmetic overflow")
        return address

    def read_bytes(self) -> bytes:
        """Validation-only snapshot; the live lease pins the source throughout."""
        self.lease._check_live()
        return bytes(self.lease._entry.allocation.buffer[self.offset : self.offset + self.nbytes])


class HostEntryLease:
    def __init__(self, cache: HostKVCache, entry: _Entry):
        self._cache = cache
        self._entry = entry
        self._released = False

    @property
    def address(self) -> int:
        self._check_live()
        return self._entry.allocation.address

    def _check_live(self) -> None:
        if self._released:
            raise RuntimeError("host entry lease has been released")

    def resolve(self, ref: ObjectRef, layer_name: str, component: str, token: int) -> HostSpan:
        self._check_live()
        if ref.key != self._entry.ref.key or ref.cache_identity != self._entry.ref.cache_identity:
            raise ValueError("host cache object identity mismatch")
        try:
            view = self._entry.components[(layer_name, component)]
        except KeyError as exc:
            raise ValueError("layer/component is absent from the object layout") from exc
        offset, size = view.token_range(ref.row_for(token), ref.raw_size)
        span = HostSpan(self, offset, size)
        _ = span.address
        return span

    def release(self) -> None:
        with self._cache._lock:
            if not self._released:
                self._entry.pins -= 1
                self._released = True

    def __enter__(self) -> HostEntryLease:
        self._check_live()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class HostKVCache:
    """One cache scope per model/store identity; no eviction of live entries."""

    def __init__(
        self,
        backend: HostBufferBackend,
        allocator: Callable[[int], HostAllocation],
        capacity_bytes: int,
        *,
        max_object_bytes: int | None = None,
    ):
        if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int) or capacity_bytes <= 0:
            raise ValueError("host cache capacity must be positive")
        if max_object_bytes is not None and (
            isinstance(max_object_bytes, bool) or not isinstance(max_object_bytes, int) or max_object_bytes <= 0
        ):
            raise ValueError("max object size must be positive")
        self._backend = backend
        self._allocator = allocator
        self._capacity = capacity_bytes
        self._max_object = capacity_bytes if max_object_bytes is None else min(capacity_bytes, max_object_bytes)
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._pending: dict[str, _Flight] = {}
        self._used_bytes = 0
        self._reserved_bytes = 0
        self._closed = False
        self._stats = {
            "host_hits": 0,
            "get_calls": 0,
            "host_copy_bytes": 0,
            "get_seconds": 0.0,
            "host_copy_seconds": 0.0,
        }

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    @property
    def backend_put_calls(self) -> int | None:
        return getattr(self._backend, "put_calls", None)

    def snapshot_stats(self) -> dict:
        with self._lock:
            return {**self._stats, "owned_bytes": self._used_bytes, "ready_objects": len(self._entries)}

    def acquire(self, refs: Sequence[ObjectRef]) -> list[HostEntryLease]:
        identities: dict[str, tuple] = {}
        for ref in refs:
            previous = identities.setdefault(ref.key, ref.cache_identity)
            if previous != ref.cache_identity:
                raise ValueError("conflicting object identities in one cache request")
        leases: list[HostEntryLease] = []
        try:
            for ref in refs:
                leases.append(self._acquire_one(ref))
        except BaseException:
            for lease in leases:
                lease.release()
            raise
        return leases

    def _acquire_one(self, ref: ObjectRef) -> HostEntryLease:
        with self._lock:
            if self._closed:
                raise RuntimeError("host cache is closed")
            entry = self._entries.get(ref.key)
            if entry is not None:
                if entry.ref.cache_identity != ref.cache_identity:
                    raise ValueError("host cache object identity mismatch")
                entry.pins += 1
                self._stats["host_hits"] += 1
                return HostEntryLease(self, entry)
            flight = self._pending.get(ref.key)
            owns_load = flight is None
            if flight is None:
                if (
                    ref.raw_size > self._max_object
                    or ref.raw_size > self._capacity - self._used_bytes - self._reserved_bytes
                ):
                    raise MemoryError("host KV cache admission budget exceeded")
                flight = _Flight(ref)
                self._pending[ref.key] = flight
                self._reserved_bytes += ref.raw_size
            elif flight.ref.cache_identity != ref.cache_identity:
                raise ValueError("conflicting in-flight object identity")
        if not owns_load:
            flight.event.wait()
            if flight.error is not None:
                raise RuntimeError("host KV object load failed") from flight.error
            return self._acquire_one(ref)

        allocation: HostAllocation | None = None
        try:
            with self._lock:
                self._stats["get_calls"] += 1
            get_start = time.perf_counter()
            result = self._backend.get_host_buffers([ref.key])
            get_seconds = time.perf_counter() - get_start
            if result is None or len(result) != 1 or result[0] is None:
                raise RuntimeError("Host Get returned a missing object or malformed result")
            allocation = self._allocator(ref.raw_size)
            if allocation.address <= 0 or allocation.address > UINT64_MAX - ref.raw_size:
                raise ValueError("invalid owned host allocation address")
            destination = allocation.buffer
            if not destination.c_contiguous or destination.readonly or destination.nbytes != ref.raw_size:
                raise ValueError("allocator must return exactly sized writable contiguous host storage")
            destination = destination.cast("B")
            source = result[0]
            sdk_address = getattr(source, "data_ptr", None)
            if callable(sdk_address) and sdk_address() == allocation.address:
                raise ValueError("Host cache allocation aliases SDK storage")
            copy_start = time.perf_counter()
            copy_into = getattr(source, "copy_into", None)
            if callable(copy_into):
                if getattr(source, "nbytes", None) != ref.raw_size:
                    raise ValueError("SDK buffer size mismatch")
                copy_into(destination)
            else:
                # A buffer-protocol exporter keeps its SDK owner/read guard alive during this copy.
                source_view = memoryview(source)
                try:
                    if not source_view.c_contiguous or source_view.nbytes != ref.raw_size:
                        raise ValueError("SDK buffer must be contiguous and exactly sized")
                    destination[:] = source_view.cast("B")
                finally:
                    source_view.release()
            copy_seconds = time.perf_counter() - copy_start
            components = validate_composed_buffer(destination, ref)
            entry = _Entry(ref, allocation, {(v.layer_name, v.component): v for v in components}, pins=1)
            with self._lock:
                self._entries[ref.key] = entry
                self._stats["get_seconds"] += get_seconds
                self._stats["host_copy_seconds"] += copy_seconds
                self._stats["host_copy_bytes"] += ref.raw_size
                self._used_bytes += ref.raw_size
                self._reserved_bytes -= ref.raw_size
                del self._pending[ref.key]
                flight.event.set()
            return HostEntryLease(self, entry)
        except BaseException as exc:
            if allocation is not None:
                # Preserve the original error and always wake same-key waiters.
                with contextlib.suppress(Exception):
                    allocation.close()
            with self._lock:
                self._reserved_bytes -= ref.raw_size
                self._pending.pop(ref.key, None)
                flight.error = exc
                flight.event.set()
            raise

    def discard_when_idle(self, keys: set[str]) -> None:
        with self._lock:
            if any(key in self._pending or (key in self._entries and self._entries[key].pins) for key in keys):
                raise RuntimeError("cannot discard Host objects with active loads or leases")
            entries = [self._entries.pop(key) for key in keys if key in self._entries]
            self._used_bytes -= sum(entry.ref.raw_size for entry in entries)
        for entry in entries:
            entry.allocation.close()

    def clear_when_idle(self) -> None:
        self._clear(close=False)

    def close(self) -> None:
        self._clear(close=True)

    def _clear(self, *, close: bool) -> None:
        with self._lock:
            if self._pending or any(entry.pins for entry in self._entries.values()):
                raise RuntimeError("cannot release host KV with active loads or leases")
            entries = list(self._entries.values())
            self._entries.clear()
            self._used_bytes = 0
            self._closed |= close
        errors = []
        for entry in entries:
            try:
                entry.allocation.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("failed to release owned host KV allocation") from errors[0]
