# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import sparse_kv_offload_manager as module
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    allocate_kv_cache_tensors_for_sparse_kv_offload,
    plan_sparse_kv_offload_memory,
)


def test_yuanrong_allocation_never_allocates_dense_host_or_main_hbm():
    allocator = Mock(side_effect=AssertionError("dense allocation is forbidden"))
    with patch.object(module, "empty_aligned_int8_cpu_tensors", allocator):
        tensors = allocate_kv_cache_tensors_for_sparse_kv_offload(
            1024, 128, 64, 0, False, allocator, allocate_dense_host=False
        )
    assert tensors == (None, None, None, None, 1152)


def test_new_token_kv_bypasses_d2h_but_retains_current_row():
    manager = object.__new__(SparseKVOffloadManager)
    manager.host_kv_runtime = SimpleNamespace(remember_current=Mock())
    manager._offload_new_kv_on_current_stream = Mock(side_effect=AssertionError("D2H is forbidden"))
    k, rope = torch.tensor([[1.0]]), torch.tensor([[2.0]])
    manager.offload_new_kv("layer0", torch.tensor([6]), None, None, None, None, k, rope)
    manager.host_kv_runtime.remember_current.assert_called_once_with("layer0", k, rope)
    with pytest.raises(RuntimeError, match="eager Decode"):
        manager.offload_new_kv("layer0", torch.tensor([6]), None, None, None, None, k, rope, capturing=True)


def test_source_resolution_occurs_after_native_lru_without_dense_address_builder():
    manager = object.__new__(SparseKVOffloadManager)
    manager.topk, manager.topk_buffer_size, manager.max_model_len, manager.lru_workspace_threads = 4, 8, 128, 1
    manager.offload_layer_names = ["layer0"]
    events = []
    manager.host_kv_runtime = SimpleNamespace(build_descriptors=Mock(side_effect=lambda *a: events.append("resolve")))
    native = SimpleNamespace(
        sparse_kv_lru_resident_compact=Mock(side_effect=lambda *a: events.append("lru")),
        sparse_kv_compute_lru_resident_addrs=Mock(side_effect=AssertionError("dense addressing")),
    )
    args = [1, torch.tensor([0]), torch.zeros((1, 4)), torch.zeros((1, 4))] + [0] * 28
    # The final tuple field is layer_id. The unchanged native planner receives the original 19 pointer/shape args.
    with patch.object(module, "_sparse_kv_ops", return_value=native):
        manager._onload_topk_kv_cpu(tuple(args))
    assert events == ["lru", "resolve"]
    native.sparse_kv_compute_lru_resident_addrs.assert_not_called()


def test_yuanrong_memory_plan_limits_logical_blocks_by_device_and_workload():
    class Spec:
        block_size = 4
        page_size_bytes = 64

        def __init__(self, host):
            self.store_on_host = host

        def max_memory_usage_bytes(self, config):
            return 640

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        additional_config={
            "sparse_kv_offload_config": {"history_source": "yuanrong_host_cache", "host_cache_capacity_bytes": 1024}
        },
    )
    result = plan_sparse_kv_offload_memory({"main": Spec(True), "indexer": Spec(False)}, config, 64000, 1024, False)
    assert result.final_num_blocks == 21
    assert result.planned_host_bytes == 1024
    assert result.host_alignment_reserve_bytes == 0


@pytest.mark.parametrize("local_failure", [True, False])
def test_host_get_failure_stops_all_tp_ranks_before_attention(local_failure):
    manager = object.__new__(SparseKVOffloadManager)
    manager.tp_size = 2
    manager.tp_group = SimpleNamespace(cpu_group=object())
    manager.host_kv_runtime = SimpleNamespace(mask_unavailable_tail=Mock(), failed=False)
    manager._onload_topk_kv_cpu = Mock(side_effect=RuntimeError("Get failed") if local_failure else None)
    with (
        patch.object(torch.distributed, "all_reduce", side_effect=lambda status, **kw: status.fill_(1)),
        pytest.raises(RuntimeError, match="attention must not run"),
    ):
        manager._prepare_host_kv_onload((), "layer0", 1)
    assert manager.host_kv_runtime.failed
