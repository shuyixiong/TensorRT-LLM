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
