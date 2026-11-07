# SPDX-License-Identifier: Apache-2.0
"""Yuanrong source resolution for the existing resident LRU/sparse_copy path."""

from __future__ import annotations

import threading
from zlib import adler32

import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_kv_cache import HostKVCache
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_token_resolver import (
    HostTokenResolver,
    object_ref_from_wire,
)


class HostKVRuntime:
    def __init__(self, cache: HostKVCache, namespace: str, *, verify_h2d: bool = False):
        self.cache = cache
        self.resolver = HostTokenResolver(namespace)
        self._requests: dict[int, str] = {}
        self._external: dict[str, str] = {}
        self._lock = threading.Lock()
        self._leases = []
        self._current_kv: dict[str, tuple] = {}
        self.verify_h2d = verify_h2d
        self.verified_h2d_bytes = 0
        self._checks = []
        self.failed = False
        self.completed_layers = 0
        self.h2d_bytes = 0
        self._pending_h2d_bytes = 0

    def register_request(self, request: str, external: str) -> None:
        identity = adler32(request.encode())
        with self._lock:
            if self._requests.get(identity, request) != request or self._external.get(external, request) != request:
                raise ValueError("Host KV request identity collision")
            self._requests[identity] = request
            self._external[external] = request

    def register_objects(self, external: str, layer: str, objects: list[dict], row_bytes: tuple[int, int]) -> None:
        refs = [object_ref_from_wire(value) for value in objects]
        for ref in refs:
            components = ref.layout.components
            if tuple((c.component, c.dtype, c.row_bytes, c.row_stride) for c in components) != (
                ("latent", "bf16", row_bytes[0], row_bytes[0]),
                ("rope", "bf16", row_bytes[1], row_bytes[1]),
            ):
                raise ValueError("Prefill/Decode main KV layout mismatch")
        with self._lock:
            request = self._external.get(external)
            if request is None:
                raise RuntimeError("Host KV objects arrived before request registration")
            self.resolver.register(request, layer, refs)

    def request_for(self, identity: int) -> str:
        with self._lock:
            request = self._requests.get(identity)
        if request is None:
            raise RuntimeError("Decode row has no registered Host KV request")
        return request

    def remember_current(self, layer: str, k, rope) -> None:
        if self.failed:
            raise RuntimeError("Host KV runtime failed; restart the worker")
        self._current_kv[layer] = (k, rope)

    def mask_unavailable_tail(self, manager, layer: str, num_rows: int) -> None:
        if self.failed:
            raise RuntimeError("Host KV runtime failed; restart the worker")
        for row in range(num_rows):
            request = self.request_for(int(manager.lru_req_ids_cpu[row]))
            prompt_end = self.resolver.prompt_end(request, layer)
            current = int(manager.lru_stable_prefix_lens_cpu[row])
            selected = manager.lru_topk_indices_cpu[row]
            # Generated history is discarded; current KV remains available for this forward.
            selected.masked_fill_((selected >= prompt_end) & (selected != current), -1)

    def build_descriptors(self, manager, layer: str, counts, tokens, slots, rows: int) -> None:
        if self._leases:
            raise RuntimeError("previous Host KV DMA was not completed")
        needed = []
        current_moves = []
        for row in range(rows):
            request = self.request_for(int(manager.lru_req_ids_cpu[row]))
            current = int(manager.lru_stable_prefix_lens_cpu[row])
            for column in range(int(counts[row])):
                token, slot = int(tokens[row, column]), int(slots[row, column])
                if not 0 <= slot < manager.topk_buffer_size:
                    raise ValueError("LRU returned an invalid HBM slot")
                linear_slot = row * manager.topk_buffer_size + slot
                if token == current:
                    current_moves.append((row, linear_slot))
                else:
                    needed.append((self.resolver.resolve(request, layer, token), token, linear_slot))
        unique = {ref.key: ref for ref, _, _ in needed}
        self._leases = self.cache.acquire(list(unique.values()))
        by_key = dict(zip(unique, self._leases))
        layer_id = manager._get_offload_layer_id(layer)
        descriptor = 0
        for ref, token, slot in needed:
            for component, base, row_bytes in (
                ("latent", manager.addr_k_bases[layer_id], manager.token_size_bytes_k),
                ("rope", manager.addr_v_bases[layer_id], manager.token_size_bytes_v),
            ):
                span = by_key[ref.key].resolve(ref, layer, component, token)
                if span.nbytes != row_bytes:
                    raise ValueError("Host row length does not match the HBM component")
                manager.gvas_buffer_cpu[descriptor] = span.address
                manager.addr_buffer_cpu[descriptor] = base + slot * row_bytes
                manager.size_buffer_cpu[descriptor] = row_bytes
                if self.verify_h2d:
                    buffer = (manager.topk_buffers_k if component == "latent" else manager.topk_buffers_v)[layer_id]
                    self._checks.append((span, buffer, slot))
                descriptor += 1
        self._pending_h2d_bytes = sum(int(size) for size in manager.size_buffer_cpu[:descriptor])
        manager.num_tokens_buffer_cpu.fill_(descriptor)
        if current_moves:
            values = self._current_kv.get(layer)
            if values is None:
                raise RuntimeError("current Decode KV is not available")
            sources = torch.tensor([row for row, _ in current_moves], device=values[0].device, dtype=torch.long)
            targets = torch.tensor([slot for _, slot in current_moves], device=values[0].device, dtype=torch.long)
            for value, buffer in zip(values, (manager.topk_buffers_k[layer_id], manager.topk_buffers_v[layer_id])):
                buffer.view(-1, buffer.shape[-1]).index_copy_(
                    0, targets, value.reshape(-1, buffer.shape[-1]).index_select(0, sources)
                )

    def verify_loaded_rows(self) -> None:
        groups = {}
        for span, buffer, slot in self._checks:
            groups.setdefault(id(buffer), (buffer, []))[1].append((span, slot))
        for buffer, rows in groups.values():
            indices = torch.tensor([slot for _, slot in rows], dtype=torch.long, device=buffer.device)
            actual = (
                buffer.view(-1, buffer.shape[-1]).index_select(0, indices).view(torch.uint8).cpu().numpy().tobytes()
            )
            expected = b"".join(span.read_bytes() for span, _ in rows)
            if actual != expected:
                self.failed = True
                raise RuntimeError("Yuanrong Host KV H2D byte validation failed; attention must not run")
            self.verified_h2d_bytes += len(expected)
        self._checks.clear()

    def complete(self, layer: str) -> None:
        # Caller must establish actual sparse_copy completion, not descriptor-copy completion.
        self.h2d_bytes += self._pending_h2d_bytes
        self._pending_h2d_bytes = 0
        self.completed_layers += 1
        for lease in self._leases:
            lease.release()
        self._leases.clear()
        self._current_kv.pop(layer, None)

    def snapshot_stats(self) -> dict:
        return {
            **self.cache.snapshot_stats(),
            "completed_decode_layers": self.completed_layers,
            "h2d_bytes": self.h2d_bytes,
            "decode_put_calls": self.cache.backend_put_calls,
            "verified_h2d_bytes": self.verified_h2d_bytes,
        }

    def discard_request(self, request: str, external: str) -> None:
        with self._lock:
            keys = self.resolver.discard(request)
            self.cache.discard_when_idle(keys)
            self._requests.pop(adler32(request.encode()), None)
            self._external.pop(external, None)
