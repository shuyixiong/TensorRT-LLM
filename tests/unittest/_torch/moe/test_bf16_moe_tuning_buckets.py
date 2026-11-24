# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AutoTuner buckets of the BF16 TRTLLM-Gen MoE runner."""

import pytest
import torch

trtllm_gen_custom_ops = pytest.importorskip("tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops")
Bf16MoERunner = trtllm_gen_custom_ops.Bf16MoERunner


def _spec(ep_size, use_dp, tune_max_num_tokens=8192):
    (spec,) = Bf16MoERunner.get_dynamic_tensor_specs(
        ep_size, tune_max_num_tokens=tune_max_num_tokens, use_dp=use_dp
    )
    return spec


@pytest.mark.parametrize("use_dp,ep_size", [(False, 1), (True, 16)])
def test_decode_token_counts_get_multiple_of_8_buckets(use_dp, ep_size):
    spec = _spec(ep_size, use_dp)
    buckets = set(spec.gen_tuning_buckets)
    scale = ep_size if use_dp else 1
    # Exact below 8, a multiple of 8 up to 128.
    for per_rank, bucket in [(1, 1), (7, 7), (8, 8), (26, 24), (28, 24), (32, 32), (127, 120), (128, 128)]:
        mapped = spec.map_to_tuning_buckets(per_rank * scale)
        assert mapped == bucket
        assert mapped in buckets


def test_large_token_counts_keep_power_of_2_buckets():
    spec = _spec(1, use_dp=False, tune_max_num_tokens=16384)
    assert spec.map_to_tuning_buckets(129) == 128
    assert spec.map_to_tuning_buckets(1000) == 512
    assert spec.map_to_tuning_buckets(8192) == 8192
    assert spec.map_to_tuning_buckets(10000) == 16384
    buckets = set(spec.gen_tuning_buckets)
    assert {128, 256, 512, 8192, 16384} <= buckets
    assert 136 not in buckets


def _dispatched_dummy(monkeypatch, num_rows, local_num_experts, hidden):
    from tensorrt_llm._torch.autotuner import AutoTuner, TuningConfig

    monkeypatch.setattr(AutoTuner.get(), "is_tuning_mode", True)
    ids = torch.zeros(num_rows, 22, dtype=torch.int32)
    return trtllm_gen_custom_ops.prepare_dummy_topk_and_hook(
        topk_weights=torch.ones(num_rows, 22, dtype=torch.bfloat16),
        topk_ids=ids,
        hidden_states=hidden,
        routing_logits=None,
        routing_method_type=0,
        base_tuning_config=TuningConfig(),
        top_k=22,
        num_experts=512,
        local_num_experts=local_num_experts,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=None,
        local_expert_offset=0,
        use_dp=True,
        tune_on_dispatched_rows=True,
    )


@pytest.mark.parametrize("local_num_experts", [16, 32])
def test_dep_tuning_profiles_the_dispatched_buffer(monkeypatch, local_num_experts):
    """Bucket b is profiled on ep_size * b rows whose distinct picks span all experts."""
    ep_size = 512 // local_num_experts
    real = torch.zeros(ep_size * 8, 64, dtype=torch.bfloat16)
    logits, weights, ids, config = _dispatched_dummy(monkeypatch, real.shape[0], local_num_experts, real)
    assert logits is None
    assert ids.shape == (real.shape[0], 22)

    bucket = torch.zeros(4, 64, dtype=torch.bfloat16)
    inputs = config.inputs_pre_hook([None, None, bucket, None, None, weights, ids])
    hidden, ids = inputs[2], inputs[-1]
    assert hidden.shape == (4 * ep_size, 64)
    assert inputs[-2].shape == ids.shape == (4 * ep_size, 22)
    assert all(len(set(row.tolist())) == 22 for row in ids)
    local_fraction = (ids < local_num_experts).float().mean().item()
    assert abs(local_fraction - local_num_experts / 512) < 0.05

    # The call's own (already dispatched) rows are not grown again.
    inputs = config.inputs_pre_hook([None, None, real, None, None, weights, ids])
    assert inputs[2] is real and inputs[-1].shape == (real.shape[0], 22)
