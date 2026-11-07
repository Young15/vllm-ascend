# SPDX-License-Identifier: Apache-2.0
import ctypes
import struct
from types import SimpleNamespace
from zlib import adler32

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_kv_allocator import HostKVArena
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_kv_cache import HostKVCache
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_kv_publisher import HostKVPublisher
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_kv_runtime import HostKVRuntime


class SDKBuffer:
    def __init__(self, data):
        self.data = data
        self.nbytes = len(data)

    def copy_into(self, destination):
        destination[:] = self.data


class Store:
    def __init__(self):
        self.data = {}
        self.get_count = 0
        self.put_count = 0

    def put_host_cache_objects(self, keys, addresses, sizes):
        self.put_count += len(keys)
        for key, addrs, lengths in zip(keys, addresses, sizes):
            blobs = [ctypes.string_at(addr, size) for addr, size in zip(addrs, lengths)]
            offsets = [128]  # Deliberately differs from example/document header alignment.
            for blob in blobs:
                offsets.append(offsets[-1] + len(blob))
            header = struct.pack("<4Q", 2, *offsets)
            self.data[key] = header + bytes(128 - len(header)) + b"".join(blobs)

    def get_host_buffers(self, keys):
        self.get_count += len(keys)
        return [SDKBuffer(self.data[key]) if key in self.data else None for key in keys]


def make_runtime():
    store = Store()
    publisher = HostKVPublisher(store, "model-and-weights-v1", "generation1", 0, 4)
    k = torch.arange(24, dtype=torch.bfloat16).reshape(3, 4, 1, 2)
    rope = (torch.arange(12, dtype=torch.bfloat16) + 100).reshape(3, 4, 1, 1)
    metadata = SimpleNamespace(main_tensor_count=2, block_len=[16, 8], block_size_scale=[1, 1])
    # Logical blocks 0/1 map to physical pages 2/0. Last prompt block has only 2 valid tokens.
    objects = publisher.publish("external", "layer0", [2, 0], 0, 6, (k, rope), metadata)
    arena = HostKVArena(torch.empty(4096, dtype=torch.uint8))
    cache = HostKVCache(store, arena.allocate, 4096)
    runtime = HostKVRuntime(cache, "model-and-weights-v1")
    runtime.register_request("internal", "external")
    runtime.register_objects("external", "layer0", objects, (4, 2))
    hot_k, hot_rope = torch.zeros(1, 8, 1, 2, dtype=torch.bfloat16), torch.zeros(1, 8, 1, 1, dtype=torch.bfloat16)
    manager = SimpleNamespace(
        topk_buffer_size=8,
        lru_req_ids_cpu=torch.tensor([adler32(b"internal")]),
        lru_stable_prefix_lens_cpu=torch.tensor([6]),
        lru_topk_indices_cpu=torch.tensor([[1, 4, 6, -1]]),
        _get_offload_layer_id=lambda name: 0,
        addr_k_bases=[hot_k.data_ptr()],
        addr_v_bases=[hot_rope.data_ptr()],
        token_size_bytes_k=4,
        token_size_bytes_v=2,
        gvas_buffer_cpu=torch.zeros(8, dtype=torch.int64),
        addr_buffer_cpu=torch.zeros(8, dtype=torch.int64),
        size_buffer_cpu=torch.zeros(8, dtype=torch.int32),
        num_tokens_buffer_cpu=torch.zeros(1, dtype=torch.int32),
        topk_buffers_k=[hot_k],
        topk_buffers_v=[hot_rope],
    )
    return runtime, manager, store, k, rope


def copy_descriptors(manager):
    for index in range(int(manager.num_tokens_buffer_cpu[0])):
        ctypes.memmove(
            int(manager.addr_buffer_cpu[index]),
            int(manager.gvas_buffer_cpu[index]),
            int(manager.size_buffer_cpu[index]),
        )


def test_prefill_to_owned_host_to_resident_slots_and_reuse():
    runtime, manager, store, k, rope = make_runtime()
    store.get_count = 0  # Separate Prefill publication checks from Decode misses.
    current_k, current_rope = torch.tensor([[77, 78]], dtype=torch.bfloat16), torch.tensor([[79]], dtype=torch.bfloat16)
    runtime.remember_current("layer0", current_k, current_rope)
    runtime.mask_unavailable_tail(manager, "layer0", 1)
    runtime.build_descriptors(
        manager, "layer0", torch.tensor([3]), torch.tensor([[1, 4, 6]]), torch.tensor([[2, 3, 7]]), 1
    )
    assert store.get_count == 2
    assert int(manager.num_tokens_buffer_cpu[0]) == 4  # Two history rows, two components each.
    with pytest.raises(RuntimeError, match="active"):
        runtime.cache.clear_when_idle()
    copy_descriptors(manager)
    runtime.complete("layer0")
    assert torch.equal(manager.topk_buffers_k[0][0, 2], k[2, 1])
    assert torch.equal(manager.topk_buffers_k[0][0, 3], k[0, 0])
    assert torch.equal(manager.topk_buffers_v[0][0, 2], rope[2, 1])
    assert torch.equal(manager.topk_buffers_k[0][0, 7], current_k)
    store.data.clear()  # Reuse must come from owned Host bytes even if remote disappears.
    runtime.build_descriptors(manager, "layer0", torch.tensor([1]), torch.tensor([[2]]), torch.tensor([[5]]), 1)
    copy_descriptors(manager)
    runtime.complete("layer0")
    assert store.get_count == 2
    assert torch.equal(manager.topk_buffers_k[0][0, 5], k[2, 2])
    runtime.build_descriptors(manager, "layer0", torch.tensor([0]), torch.tensor([[0]]), torch.tensor([[0]]), 1)
    assert int(manager.num_tokens_buffer_cpu[0]) == 0
    runtime.complete("layer0")
    assert store.get_count == 2 and store.put_count == 2
    runtime.discard_request("internal", "external")
    assert runtime.cache.used_bytes == 0


def test_old_generated_tail_excluded_before_lru_even_if_hbm_cached():
    runtime, manager, *_ = make_runtime()
    manager.lru_stable_prefix_lens_cpu.fill_(8)
    manager.lru_topk_indices_cpu = torch.tensor([[1, 6, 7, 8]])
    runtime.mask_unavailable_tail(manager, "layer0", 1)
    assert manager.lru_topk_indices_cpu.tolist() == [[1, -1, -1, 8]]


def test_pd_row_layout_mismatch_and_partial_prompt_bounds():
    runtime, manager, *_ = make_runtime()
    with pytest.raises(ValueError, match="covered"):
        runtime.resolver.resolve("internal", "layer0", 6)
    with pytest.raises(RuntimeError, match="registered"):
        runtime.request_for(123)
    runtime.failed = True
    with pytest.raises(RuntimeError, match="restart"):
        runtime.remember_current("layer0", None, None)


def test_arena_release_coalesces_and_old_allocation_cannot_be_read():
    arena = HostKVArena(torch.empty(256, dtype=torch.uint8))
    a, b = arena.allocate(65), arena.allocate(65)
    with pytest.raises(MemoryError):
        arena.allocate(1)
    a.close()
    b.close()
    whole = arena.allocate(256)
    assert whole.address == arena.address
    with pytest.raises(RuntimeError, match="released"):
        _ = a.address


def test_h2d_verification_detects_wrong_hot_slot_before_attention():
    runtime, manager, *_ = make_runtime()
    runtime.verify_h2d = True
    runtime.build_descriptors(manager, "layer0", torch.tensor([1]), torch.tensor([[1]]), torch.tensor([[2]]), 1)
    copy_descriptors(manager)
    runtime.verify_loaded_rows()
    assert runtime.verified_h2d_bytes == 6
    runtime.complete("layer0")
    runtime.build_descriptors(manager, "layer0", torch.tensor([1]), torch.tensor([[1]]), torch.tensor([[2]]), 1)
    copy_descriptors(manager)
    manager.topk_buffers_k[0][0, 2].zero_()
    with pytest.raises(RuntimeError, match="attention must not run"):
        runtime.verify_loaded_rows()
    assert runtime.failed
    with pytest.raises(RuntimeError, match="active"):
        runtime.cache.close()
