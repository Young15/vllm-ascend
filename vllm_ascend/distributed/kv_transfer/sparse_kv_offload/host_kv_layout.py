# SPDX-License-Identifier: Apache-2.0
"""Producer-described layouts for composed Yuanrong objects."""

from __future__ import annotations

import bisect
import hashlib
import json
import struct
from dataclasses import asdict, dataclass

UINT64_MAX = (1 << 64) - 1
OFFSET_FIELD_BYTES = 8
COMPOSED_CODEC = "yuanrong_composed_v1"


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {UINT64_MAX}]")
    return value


@dataclass(frozen=True)
class ComponentSpec:
    layer_name: str
    component: str
    blob_index: int
    dtype: str
    row_bytes: int
    row_stride: int

    def __post_init__(self) -> None:
        if not self.layer_name or not self.component or not self.dtype:
            raise ValueError("component layer, name and dtype must be nonempty")
        _integer("blob_index", self.blob_index)
        _integer("row_bytes", self.row_bytes, 1)
        _integer("row_stride", self.row_stride, self.row_bytes)


@dataclass(frozen=True)
class KVObjectLayout:
    model_revision: str
    group_id: int
    shard_id: str
    components: tuple[ComponentSpec, ...]
    byte_order: str = "little"
    codec: str = COMPOSED_CODEC
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.codec != COMPOSED_CODEC or self.byte_order not in ("little", "big"):
            raise ValueError("unsupported KV object layout encoding")
        if not self.model_revision or not self.shard_id:
            raise ValueError("model revision and shard identity are required")
        _integer("group_id", self.group_id)
        object.__setattr__(self, "components", tuple(self.components))
        if not self.components:
            raise ValueError("layout requires components")
        if sorted(c.blob_index for c in self.components) != list(range(len(self.components))):
            raise ValueError("blob indices must cover the object exactly once")
        identities = [(c.layer_name, c.component) for c in self.components]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate layer/component identity")

    @property
    def layout_id(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ObjectRef:
    key: str
    layout: KVObjectLayout
    content_identity: str
    token_start: int
    token_count: int
    raw_size: int
    blob_sizes: tuple[int, ...]
    logical_positions: tuple[int, ...] | None = None
    payload_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.key or not self.content_identity:
            raise ValueError("object key and immutable content identity are required")
        _integer("token_start", self.token_start)
        _integer("token_count", self.token_count, 1)
        _integer("raw_size", self.raw_size, 1)
        object.__setattr__(self, "blob_sizes", tuple(self.blob_sizes))
        if len(self.blob_sizes) != len(self.layout.components):
            raise ValueError("blob sizes do not match the layout")
        for size in self.blob_sizes:
            _integer("blob_size", size, 1)
        if self.raw_size < (len(self.blob_sizes) + 2) * OFFSET_FIELD_BYTES + sum(self.blob_sizes):
            raise ValueError("raw size cannot contain the composed object")
        if self.logical_positions is not None:
            positions = tuple(self.logical_positions)
            for value in positions:
                _integer("logical position", value)
            if len(positions) != self.token_count or tuple(sorted(set(positions))) != positions:
                raise ValueError("logical positions must be sorted, unique and cover every row")
            if positions[0] != self.token_start:
                raise ValueError("token_start must match the first logical position")
            object.__setattr__(self, "logical_positions", positions)
        elif self.token_start + self.token_count > UINT64_MAX:
            raise ValueError("token range overflows uint64")
        if self.payload_digest is not None:
            if len(self.payload_digest) != 64 or any(c not in "0123456789abcdef" for c in self.payload_digest):
                raise ValueError("payload_digest must be a lowercase SHA-256 digest")
        for spec in self.layout.components:
            required = (self.token_count - 1) * spec.row_stride + spec.row_bytes
            if required > self.blob_sizes[spec.blob_index]:
                raise ValueError("component rows exceed the declared blob size")

    @property
    def cache_identity(self) -> tuple:
        return (
            self.layout.layout_id,
            self.content_identity,
            self.raw_size,
            self.blob_sizes,
            self.token_count,
            self.payload_digest,
        )

    def row_for(self, token: int) -> int:
        _integer("logical token", token)
        if self.logical_positions is None:
            row = token - self.token_start
            if 0 <= row < self.token_count:
                return row
        else:
            # Binary search keeps lookup independent of the context length.
            row = bisect.bisect_left(self.logical_positions, token)
            if row < len(self.logical_positions) and self.logical_positions[row] == token:
                return row
        raise ValueError("token is not covered by this object")


@dataclass(frozen=True)
class ComponentView:
    layer_name: str
    component: str
    byte_offset: int
    byte_length: int
    row_bytes: int
    row_stride: int
    row_count: int

    def token_range(self, row: int, object_size: int) -> tuple[int, int]:
        _integer("row", row)
        if row >= self.row_count:
            raise ValueError("token row is outside the component")
        relative = row * self.row_stride
        if relative > self.byte_length or self.row_bytes > self.byte_length - relative:
            raise ValueError("token row exceeds component bytes")
        offset = self.byte_offset + relative
        if offset > object_size or self.row_bytes > object_size - offset:
            raise ValueError("token row exceeds object bytes")
        return offset, self.row_bytes


def validate_composed_buffer(raw: memoryview, ref: ObjectRef) -> tuple[ComponentView, ...]:
    """Validate raw bytes without copying payload or inventing component semantics."""
    if not raw.c_contiguous:
        raise ValueError("composed buffer must be contiguous")
    raw = raw.cast("B")
    if raw.nbytes != ref.raw_size or raw.nbytes < OFFSET_FIELD_BYTES:
        raise ValueError("composed buffer size mismatch")
    prefix = "<" if ref.layout.byte_order == "little" else ">"
    count = struct.unpack_from(prefix + "Q", raw)[0]
    if count != len(ref.layout.components):
        raise ValueError("composed buffer component count mismatch")
    header_bytes = (count + 2) * OFFSET_FIELD_BYTES
    if header_bytes > raw.nbytes:
        raise ValueError("truncated composed buffer offset table")
    offsets = struct.unpack_from(prefix + "Q" * (count + 1), raw, OFFSET_FIELD_BYTES)
    if offsets[0] < header_bytes or offsets[-1] != raw.nbytes:
        raise ValueError("invalid composed buffer header or end offset")
    for index, size in enumerate(ref.blob_sizes):
        if not offsets[index] <= offsets[index + 1] <= raw.nbytes:
            raise ValueError("composed buffer offsets are not monotonic or in bounds")
        if offsets[index + 1] - offsets[index] != size:
            raise ValueError("composed buffer blob length mismatch")
    if ref.payload_digest is not None:
        # Producer and consumer hash valid blobs, excluding alignment padding.
        digest = hashlib.sha256()
        for index in range(count):
            digest.update(raw[offsets[index] : offsets[index + 1]])
        if digest.hexdigest() != ref.payload_digest:
            raise ValueError("object payload identity mismatch")
    return tuple(
        ComponentView(
            spec.layer_name,
            spec.component,
            offsets[spec.blob_index],
            ref.blob_sizes[spec.blob_index],
            spec.row_bytes,
            spec.row_stride,
            ref.token_count,
        )
        for spec in ref.layout.components
    )
