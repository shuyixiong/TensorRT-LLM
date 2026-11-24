# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.prefill_batch_wait import PrefillBatchWait
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def test_sparse_work_drains_previous_results_before_starting_deadline() -> None:
    waiter = PrefillBatchWait(1.0, 0.5)
    assert waiter.should_wait([[553, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]], 1024, 0)
    assert waiter.started_ns is None
    # Time spent finishing the previous GPU batch does not consume intake time.
    assert waiter.should_wait([[553, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], 1024, 100_000_000)
    assert waiter.started_ns == 100_000_000


def test_new_arrivals_release_batch_before_timeout() -> None:
    waiter = PrefillBatchWait(1.0, 0.5)
    assert waiter.should_wait([[553, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], 1024, 0)
    assert not waiter.should_wait([[553, 0, 0, 0, 0, 500], [553, 0, 0, 0, 0, 0]], 1024, 500)
    assert waiter.started_ns is None


def test_coordinator_deadline_releases_an_unfillable_batch() -> None:
    votes = [[553, 0, 0, 0, 0, 1_000_000], [0, 0, 0, 0, 0, 0]]
    for local_now in (100, 10_000_000_000):
        waiter = PrefillBatchWait(1.0, 0.5)
        assert not waiter.should_wait(votes, 1024, local_now)
        assert waiter.timeout_releases == 1


@pytest.mark.parametrize("field", [1, 2, 4])
def test_decode_continuations_and_shutdown_bypass_waiting(field: int) -> None:
    waiter = PrefillBatchWait(1.0, 0.5)
    votes = [[553, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]]
    votes[1][field] = 1
    assert not waiter.should_wait(votes, 1024, 0)


def test_empty_fleet_does_not_start_a_wait() -> None:
    waiter = PrefillBatchWait(1.0, 0.5)
    assert not waiter.should_wait([[0, 0, 0, 1, 0, 0]], 1024, 0)


def test_drain_only_releases_as_soon_as_previous_results_are_consumed() -> None:
    waiter = PrefillBatchWait(0, 0.5, drain_only=True)
    assert waiter.should_wait([[553, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]], 1024, 0)
    assert not waiter.should_wait([[553, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], 1024, 1)
    assert waiter.started_ns is None
    assert waiter.drain_passes == 1
    assert waiter.wait_passes == 0


def test_zero_timeout_without_drain_only_is_disabled() -> None:
    waiter = PrefillBatchWait(0, 0.5)
    assert not waiter.should_wait([[553, 0, 0, 1, 0, 0]], 2048, 0)


def test_full_ready_batch_keeps_overlap_in_drain_only_mode() -> None:
    waiter = PrefillBatchWait(0, 0.5, drain_only=True)
    assert not waiter.should_wait([[1024, 0, 0, 1, 0, 0] for _ in range(32)], 1024, 0)
    assert waiter.drain_passes == 0


@pytest.mark.parametrize("timeout,ratio", [(-1, 0.5), (float("nan"), 0.5), (1, 0), (1, 1.1)])
def test_invalid_wait_configuration_is_rejected(timeout: float, ratio: float) -> None:
    with pytest.raises(ValueError):
        PrefillBatchWait(timeout, ratio)


def test_adaptive_deadline_shrinks_as_the_batch_fills() -> None:
    waiter = PrefillBatchWait(10.0, 0.5, adaptive=True)
    assert waiter.should_wait([[256, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], 1024, 0)
    sparse_limit = waiter.current_limit_ns
    assert waiter.should_wait([[768, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]], 1024, 1)
    assert waiter.current_limit_ns < sparse_limit
    assert waiter.current_limit_ns == int(10e6 * (1 - 768 / 1024))


def test_disabled_wait_does_not_add_a_collective(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TLLM_ADP_PREFILL_BATCH_WAIT_MS", raising=False)
    monkeypatch.delenv("TLLM_ADP_PREFILL_BATCH_WAIT_RATIO", raising=False)
    monkeypatch.delenv("TLLM_ADP_PREFILL_BATCH_WAIT_DRAIN_ONLY", raising=False)
    executor = object.__new__(PyExecutor)
    executor.is_warmup = False
    executor.dist = Mock()
    request = object()

    assert executor._coalesce_adp_context_requests([request], []) == [request]
    executor.dist.tp_allgather_int64.assert_not_called()


def test_drain_only_is_enabled_with_zero_intake_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLLM_ADP_PREFILL_BATCH_WAIT_MS", "0")
    monkeypatch.setenv("TLLM_ADP_PREFILL_BATCH_WAIT_RATIO", "0.5")
    monkeypatch.setenv("TLLM_ADP_PREFILL_BATCH_WAIT_DRAIN_ONLY", "1")
    votes = [[553, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]]
    executor = SimpleNamespace(
        is_warmup=False,
        disable_overlap_scheduler=False,
        _is_kv_manager_v2=True,
        is_encoder_decoder=False,
        previous_batch=object(),
        is_shutdown=False,
        control_requests=[],
        max_num_tokens=1024,
        dist=SimpleNamespace(
            pp_size=1,
            cp_size=1,
            tp_allgather_int64=Mock(return_value=SimpleNamespace(tolist=lambda: votes)),
        ),
    )
    request = SimpleNamespace(
        is_attention_dp_dummy=False, context_chunk_size=553, is_first_context_chunk=True
    )
    assert PyExecutor._coalesce_adp_context_requests(executor, [request], []) == []
    executor.previous_batch = None
    votes[0][3] = votes[1][3] = 0
    assert PyExecutor._coalesce_adp_context_requests(executor, [request], []) == [request]
    assert executor._adp_prefill_batch_wait.wait_passes == 0
