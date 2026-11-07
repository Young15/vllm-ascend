# SPDX-License-Identifier: Apache-2.0
"""Owned slices of the original MemFabric Host allocation."""

from __future__ import annotations

import threading


class HostArenaAllocation:
    def __init__(self, arena: HostKVArena, offset: int, size: int, reserved: int):
        self._arena = arena
        self._offset = offset
        self._size = size
        self._reserved = reserved
        self._closed = False

    @property
    def address(self) -> int:
        if self._closed:
            raise RuntimeError("Host allocation was released")
        return self._arena.address + self._offset

    @property
    def buffer(self) -> memoryview:
        _ = self.address
        return self._arena.buffer[self._offset : self._offset + self._size]

    def close(self) -> None:
        self._arena.release(self)


class HostKVArena:
    """A fixed owner keeps registered addresses stable until all DMA completes."""

    ALIGNMENT = 64

    def __init__(self, tensor):
        self._owner = tensor
        self.address = int(tensor.data_ptr())
        self.buffer = memoryview(tensor.numpy()).cast("B")
        self._free = [(0, self.buffer.nbytes)]
        self._lock = threading.Lock()

    def allocate(self, size: int) -> HostArenaAllocation:
        if size <= 0:
            raise ValueError("Host allocation size must be positive")
        reserved = (size + self.ALIGNMENT - 1) // self.ALIGNMENT * self.ALIGNMENT
        with self._lock:
            for index, (offset, available) in enumerate(self._free):
                if available >= reserved:
                    self._free[index : index + 1] = (
                        [(offset + reserved, available - reserved)] if available > reserved else []
                    )
                    return HostArenaAllocation(self, offset, size, reserved)
        raise MemoryError("MemFabric Host KV arena is exhausted")

    def release(self, allocation: HostArenaAllocation) -> None:
        with self._lock:
            if allocation._closed:
                return
            allocation._closed = True
            ranges = sorted([*self._free, (allocation._offset, allocation._reserved)])
            merged: list[tuple[int, int]] = []
            for start, length in ranges:
                if merged and merged[-1][0] + merged[-1][1] == start:
                    previous, count = merged.pop()
                    merged.append((previous, count + length))
                else:
                    merged.append((start, length))
            self._free = merged
