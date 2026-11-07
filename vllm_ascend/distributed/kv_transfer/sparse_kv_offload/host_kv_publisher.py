# SPDX-License-Identifier: Apache-2.0
"""Publish stable prompt block objects before sending PD readiness."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence

import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_kv_layout import (
    ComponentSpec,
    KVObjectLayout,
    ObjectRef,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_token_resolver import object_ref_to_wire


class HostKVPublisher:
    def __init__(self, backend, namespace: str, generation: str, group_id: int, block_size: int):
        if not namespace or not generation or block_size <= 0:
            raise ValueError("Host KV publication requires namespace, generation and block size")
        self.backend = backend
        self.namespace = namespace
        self.generation = generation
        self.group_id = group_id
        self.block_size = block_size

    def publish(
        self,
        request: str,
        layer: str,
        blocks: Sequence[int],
        start: int,
        valid_tokens: int,
        tensors: Sequence,
        main_metadata,
    ) -> list[dict]:
        if len(tensors) != 2 or main_metadata.main_tensor_count != 2:
            raise ValueError("Yuanrong SFA publication requires latent and RoPE components")
        specs, bases, strides = [], [], []
        for index, (name, tensor) in enumerate(zip(("latent", "rope"), tensors)):
            if not tensor.is_contiguous() or str(tensor.dtype) != "torch.bfloat16":
                raise ValueError("Yuanrong SFA publication requires contiguous BF16 component tensors")
            stride = main_metadata.block_len[index] * main_metadata.block_size_scale[index]
            if stride % self.block_size:
                raise ValueError("SFA component block stride does not contain complete rows")
            row_bytes = stride // self.block_size
            specs.append(ComponentSpec(layer, name, index, "bf16", row_bytes, row_bytes))
            bases.append(int(tensor.data_ptr()))
            strides.append(stride)
        layout = KVObjectLayout(
            self.namespace, self.group_id, "replicated-main", tuple(specs), byte_order=sys.byteorder
        )
        refs = []
        for offset, physical in enumerate(blocks):
            token_start = (start + offset) * self.block_size
            count = min(self.block_size, valid_tokens - token_start)
            if count <= 0 or physical < 0:
                raise ValueError("publication range exceeds computed prompt tokens")
            addresses, sizes = [], []
            digest = hashlib.sha256()
            for tensor, spec, base, stride in zip(tensors, specs, bases, strides):
                size = count * spec.row_bytes
                if physical * stride + size > tensor.numel() * tensor.element_size():
                    raise ValueError("publication exceeds Prefill KV allocation")
                addresses.append(base + physical * stride)
                sizes.append(size)
                # Independent P-side reference also detects equal-sized component swaps.
                raw = tensor.view(-1).view(torch.uint8)
                digest.update(raw[physical * stride : physical * stride + size].cpu().numpy().tobytes())
            identity = hashlib.sha256(
                json.dumps(
                    [self.generation, request, layout.layout_id, token_start, count], separators=(",", ":")
                ).encode()
            ).hexdigest()
            key = "sfa-host-v2-" + identity
            self.backend.put_host_cache_objects([key], [addresses], [sizes])
            # The Worker controls composed-header alignment. Read the actual size;
            # never infer raw_size from a presumed 32/64-byte header.
            buffers = self.backend.get_host_buffers([key])
            if len(buffers) != 1 or buffers[0] is None:
                raise RuntimeError("published Yuanrong prompt object is not readable")
            raw_size = buffers[0].nbytes
            ref = ObjectRef(
                key, layout, identity, token_start, count, raw_size, tuple(sizes), payload_digest=digest.hexdigest()
            )
            refs.append(object_ref_to_wire(ref))
        return refs
