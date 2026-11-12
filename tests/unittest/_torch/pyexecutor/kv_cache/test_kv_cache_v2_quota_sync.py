# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for KVCacheManagerV2._sync_quota_across_ranks.

The cost model below has the shape of a hybrid Mamba manager with block reuse and a
prompt-end snapshot: zero tokens cost only the live states, while one token adds one
snapshot slot per resident sequence. Sizes are those of a DEP16 Nemotron Ultra rank at
max batch 64 (64 live + 2 dummy + 64 snapshot state slots).
"""

import math

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

GIB = 1 << 30
UTIL = 0.95
STATE_BYTES = 407_957_000  # one Mamba state slot (SSM + conv) per rank
TOKEN_BYTES = 12_288  # attention bytes per token per rank
LIVE_SLOTS = 66
SNAPSHOT_SLOTS = 64


class _StepCostModel:
    """Stand-in for a manager whose quota for 1 token jumps above the quota for 0."""

    _max_tokens_for_gpu_quota = KVCacheManagerV2._max_tokens_for_gpu_quota
    _gpu_quota_for_max_tokens = KVCacheManagerV2._gpu_quota_for_max_tokens

    def __init__(self, token_bytes=TOKEN_BYTES):
        self.token_bytes = token_bytes

    def _fixed_region_quota(self):
        return 0

    def _get_quota_from_max_tokens(self, max_tokens):
        slots = LIVE_SLOTS + (SNAPSHOT_SLOTS if max_tokens > 0 else 0)
        return slots * STATE_BYTES + max_tokens * self.token_bytes

    def _get_max_tokens_from_quota(self, quota):
        if self._get_quota_from_max_tokens(1) > quota:
            return 0
        return (quota - (LIVE_SLOTS + SNAPSHOT_SLOTS) * STATE_BYTES) // self.token_bytes


def _sync(model, quota, peer_tokens=None, peer_quota=None):
    """Run the sync on one rank against one peer; MIN over the two ranks."""
    calls = []

    def allreduce_min(value):
        calls.append(value)
        if len(calls) == 1:
            peer = value if peer_tokens is None else peer_tokens
        else:
            peer = value if peer_quota is None else peer_quota
        return min(value, peer)

    return KVCacheManagerV2._sync_quota_across_ranks(model, quota, UTIL, allreduce_min), calls


def test_equal_ranks_keep_quota_when_one_token_does_not_fit():
    quota = int(47.16 * GIB)
    model = _StepCostModel()
    assert model._get_max_tokens_from_quota(int(quota * UTIL)) == 0
    synced, calls = _sync(model, quota)
    # Converting 0 tokens back to bytes would keep only the live states (~26.4 GiB).
    assert synced == quota
    assert len(calls) == 1


def test_equal_ranks_keep_quota_when_tokens_fit():
    quota = 80 * GIB
    synced, _ = _sync(_StepCostModel(), quota)
    assert synced == quota


def test_smaller_peer_lowers_quota_to_its_token_capacity():
    quota = 80 * GIB
    model = _StepCostModel()
    local_tokens = model._get_max_tokens_from_quota(int(quota * UTIL))
    peer_tokens = local_tokens // 2
    synced, _ = _sync(model, quota, peer_tokens=peer_tokens)
    expected = int(math.ceil(model._get_quota_from_max_tokens(peer_tokens) / UTIL))
    assert synced == expected < quota


def test_peer_without_token_capacity_syncs_bytes():
    quota = 80 * GIB
    peer_quota = int(47.16 * GIB)
    synced, calls = _sync(_StepCostModel(), quota, peer_tokens=0, peer_quota=peer_quota)
    assert synced == peer_quota
    assert len(calls) == 2


def test_quota_never_increases():
    quota = int(47.16 * GIB)
    synced, _ = _sync(_StepCostModel(), quota, peer_tokens=0, peer_quota=100 * GIB)
    assert synced == quota


@pytest.mark.parametrize("peer_tokens", [float("inf")])
def test_infinite_capacity_keeps_quota(peer_tokens):
    class _SwaOnly(_StepCostModel):
        def _get_max_tokens_from_quota(self, quota):
            return float("inf")

    quota = 30 * GIB
    synced, _ = _sync(_SwaOnly(), quota, peer_tokens=peer_tokens)
    assert synced == quota
