# SPDX-License-Identifier: Apache-2.0
"""Request-scoped logical-token mappings, independent of recyclable block IDs."""

from __future__ import annotations

import bisect
import threading
from dataclasses import asdict

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.host_kv_layout import (
    ComponentSpec,
    KVObjectLayout,
    ObjectRef,
)


def object_ref_to_wire(ref: ObjectRef) -> dict:
    result = asdict(ref)
    result["layout_id"] = ref.layout.layout_id
    return result


def object_ref_from_wire(value: dict) -> ObjectRef:
    value = dict(value)
    expected = value.pop("layout_id")
    layout = dict(value.pop("layout"))
    layout["components"] = tuple(ComponentSpec(**item) for item in layout["components"])
    descriptor = KVObjectLayout(**layout)
    if expected != descriptor.layout_id:
        raise ValueError("PD object layout_id mismatch")
    return ObjectRef(layout=descriptor, **value)


class HostTokenResolver:
    """Control thread registers ranges; forward thread takes immutable snapshots."""

    def __init__(self, model_revision: str):
        self.model_revision = model_revision
        self._lock = threading.Lock()
        self._requests: dict[str, dict[str, tuple[ObjectRef, ...]]] = {}

    def register(self, request: str, layer: str, refs: list[ObjectRef]) -> None:
        if not request or not refs:
            raise ValueError("request and object ranges are required")
        for ref in refs:
            if ref.layout.model_revision != self.model_revision:
                raise ValueError("PD model/store namespace mismatch")
            if ref.logical_positions is not None:
                raise ValueError("this PD adapter requires contiguous token ranges")
            if {c.layer_name for c in ref.layout.components} != {layer}:
                raise ValueError("PD layer/component identity mismatch")
        with self._lock:
            layers = self._requests.setdefault(request, {})
            by_start = {ref.token_start: ref for ref in layers.get(layer, ())}
            for ref in refs:
                previous = by_start.get(ref.token_start)
                if previous is not None and previous != ref:
                    raise ValueError("immutable prompt object range was replaced")
                by_start[ref.token_start] = ref
            ordered = tuple(by_start[start] for start in sorted(by_start))
            end = 0
            for ref in ordered:
                if ref.token_start != end:
                    raise ValueError("prompt object ranges overlap or leave a gap")
                end += ref.token_count
            layers[layer] = ordered

    def snapshot(self, request: str, layer: str) -> tuple[ObjectRef, ...]:
        with self._lock:
            refs = self._requests.get(request, {}).get(layer)
            if not refs:
                raise RuntimeError("prompt KV was not published for this request/layer")
            return refs

    def resolve(self, request: str, layer: str, token: int) -> ObjectRef:
        refs = self.snapshot(request, layer)
        index = bisect.bisect_right([ref.token_start for ref in refs], token) - 1
        if index < 0:
            raise ValueError("token precedes the published prompt")
        ref = refs[index]
        ref.row_for(token)
        return ref

    def prompt_end(self, request: str, layer: str) -> int:
        refs = self.snapshot(request, layer)
        return refs[-1].token_start + refs[-1].token_count

    def discard(self, request: str) -> set[str]:
        with self._lock:
            layers = self._requests.pop(request, {})
        return {ref.key for refs in layers.values() for ref in refs}
