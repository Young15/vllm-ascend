# SPDX-License-Identifier: Apache-2.0
import ctypes
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import LayerMetadata, SendTask
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread import ConsumerReadState, MembPullReadThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.send_thread import MembPullSendingThread, ProducerSendState


@pytest.mark.parametrize("host_history", [True, False])
def test_pd_preserves_indexer_transfer_and_only_skips_main_in_host_mode(host_history):
    source_k = torch.arange(24, dtype=torch.bfloat16).reshape(3, 4, 1, 2)
    source_rope = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4, 1, 1)
    source_index = (torch.arange(12, dtype=torch.bfloat16) + 100).reshape(3, 4, 1, 1)
    dest_k, dest_rope, dest_index = (
        torch.zeros_like(source_k),
        torch.zeros_like(source_rope),
        torch.zeros_like(source_index),
    )
    state = ConsumerReadState(
        num_blocks=3,
        tp_size=1,
        layer_metadata={"layer0": object()},
        main_name_to_idx={"layer0": 0},
        cpu_pools=[(dest_k, dest_rope)],
        main_gva_bases=[(dest_k.data_ptr(), dest_rope.data_ptr())],
        main_block_lens=[(16, 8)],
        indexer_tensors=[dest_index],
        indexer_scale_tensors=[None],
        dest_blocks_by_req={"request": ([1, 2], [2, 1])},
        get_offload_layer_id=lambda name: 0,
        block_size=4,
        host_kv_runtime=object() if host_history else None,
    )
    transfers = []

    def read(session, destinations, sources, sizes):
        transfers.extend(zip(destinations, sources, sizes))
        for dst, src, size in zip(destinations, sources, sizes):
            ctypes.memmove(dst, src, size)
        return 0

    reader = MembPullReadThread(0, 1, SimpleNamespace(batch_transfer_sync_read=read), state)
    metadata = {
        "layer0": {
            "base_addrs": [source_k.data_ptr(), source_rope.data_ptr(), source_index.data_ptr()],
            "block_len": [16, 8, 8],
            "block_size_scale": [1, 1, 1],
            "main_tensor_count": 2,
            "has_indexer": True,
        }
    }
    reader._do_read_batch("layer0", [("request", [2, 0], [1, 0], 0, 0)], p_session="peer", p_layer_meta=metadata)
    assert torch.equal(dest_index[2], source_index[1])
    assert torch.equal(dest_index[1], source_index[0])
    if host_history:
        assert not dest_k.count_nonzero() and not dest_rope.count_nonzero()
        assert sum(size for _, _, size in transfers) == 16
    else:
        assert torch.equal(dest_k[1], source_k[2])
        assert torch.equal(dest_rope[2], source_rope[0])
        assert sum(size for _, _, size in transfers) == 64


def test_pd_publication_completes_before_ready_notification():
    events = []
    wire_object = {"key": "published-key"}
    publisher = SimpleNamespace(publish=Mock(side_effect=lambda *args: events.append("published") or [wire_object]))
    state = ProducerSendState(
        0,
        {"layer0": LayerMetadata([0, 0], [10, 20], [16, 8], [1, 1])},
        "session",
        0,
        0,
        (4,),
        {0: (0,)},
        host_publisher=publisher,
        main_tensors={"layer0": (object(), object())},
    )
    sender = MembPullSendingThread(ready_event=threading.Event(), state=state)
    dealer = SimpleNamespace(send=lambda payload: events.append(msgspec.msgpack.decode(payload)))
    sender._ensure_peer_dealer = lambda *a: dealer
    sender._send_mf_meta = lambda *a: None
    meta = SimpleNamespace(
        local_block_ids=[[2]],
        remote_cache_tokens=0,
        local_transed_tokens=0,
        local_computed_tokens=3,
        prompt_len=3,
        chunk_finish=True,
        remote_host="localhost",
        remote_port=1,
        remote_engine_id="decode",
        group_member_idx=0,
        tp_ratio=1,
    )
    try:
        sender._process_send_task(
            SendTask({"request": meta}, layer_idx=0, layer_name="layer0"), msgspec.msgpack.Encoder()
        )
    finally:
        sender._persist_ctx.destroy(linger=0)
    assert events[0] == "published"
    assert events[1][3][0][5] == [wire_object]
    assert events[1][4] == ["request"]


def test_failed_publication_never_announces_ready():
    state = ProducerSendState(
        0,
        {"layer0": LayerMetadata([0, 0], [10, 20], [16, 8], [1, 1])},
        "session",
        0,
        0,
        (4,),
        {0: (0,)},
        host_publisher=SimpleNamespace(publish=Mock(side_effect=RuntimeError("store failed"))),
        main_tensors={"layer0": (object(), object())},
    )
    sender = MembPullSendingThread(ready_event=threading.Event(), state=state)
    connect = Mock(side_effect=AssertionError("must not announce ready"))
    sender._ensure_peer_dealer = connect
    meta = SimpleNamespace(
        local_block_ids=[[2]],
        remote_cache_tokens=0,
        local_transed_tokens=0,
        local_computed_tokens=3,
        prompt_len=3,
        chunk_finish=True,
        remote_host="localhost",
        remote_port=1,
        remote_engine_id="decode",
        group_member_idx=0,
        tp_ratio=1,
    )
    try:
        with pytest.raises(RuntimeError, match="store failed"):
            sender._process_send_task(
                SendTask({"request": meta}, layer_idx=0, layer_name="layer0"), msgspec.msgpack.Encoder()
            )
    finally:
        sender._persist_ctx.destroy(linger=0)
    connect.assert_not_called()
