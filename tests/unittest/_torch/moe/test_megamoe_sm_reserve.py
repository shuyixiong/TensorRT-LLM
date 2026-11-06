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

"""MegaMoE CuteDSL grid sizing honors ``TRTLLM_CUTEDSL_SM_RESERVE``."""

import sys
import types

import pytest

from tensorrt_llm._torch.cute_dsl_utils import reserved_max_active_clusters, sm_reserve

megamoe_op = pytest.importorskip("tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op")

pytestmark = pytest.mark.cpu_only

_DEVICE_SMS = 152
_SM_RESERVE_ENV = "TRTLLM_CUTEDSL_SM_RESERVE"


@pytest.fixture
def occupancy_queries(monkeypatch):
    """Replace ``cutlass.utils.HardwareInfo`` with a device that applies the reserve.

    The fake mirrors the ``cute_dsl_utils`` hook: the full-device cluster count
    less the clusters covering ``sm_reserve()`` SMs. It records every query so
    tests can tell cache hits from fresh queries.
    """
    queries = []

    class FakeHardwareInfo:
        def get_max_active_clusters(self, cluster_size: int) -> int:
            queries.append(cluster_size)
            return reserved_max_active_clusters(
                _DEVICE_SMS // cluster_size, cluster_size, sm_reserve()
            )

    utils = types.ModuleType("cutlass.utils")
    utils.HardwareInfo = FakeHardwareInfo
    cutlass = types.ModuleType("cutlass")
    cutlass.utils = utils
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.utils", utils)
    monkeypatch.delenv(_SM_RESERVE_ENV, raising=False)
    megamoe_op._max_active_clusters_for_reserve.cache_clear()
    yield queries
    megamoe_op._max_active_clusters_for_reserve.cache_clear()


def test_reserve_set_after_first_query_is_not_hidden_by_cache(monkeypatch, occupancy_queries):
    assert megamoe_op._max_active_clusters(2, 100) == 76

    monkeypatch.setenv(_SM_RESERVE_ENV, "8")
    assert megamoe_op._max_active_clusters(2, 100) == 72

    monkeypatch.delenv(_SM_RESERVE_ENV)
    assert megamoe_op._max_active_clusters(2, 100) == 76
    # The last query is served from the cache entry of the first.
    assert occupancy_queries == [2, 2]


@pytest.mark.parametrize(
    "reserve, expected",
    [(None, (76, None, None, 152)), ("4", (74, None, None, 148)), ("8", (72, None, None, 144))],
    ids=["no_reserve", "reserve_4", "reserve_8"],
)
def test_launch_configuration_leaves_reserved_sms_free(
    monkeypatch, occupancy_queries, reserve, expected
):
    if reserve is not None:
        monkeypatch.setenv(_SM_RESERVE_ENV, reserve)
    assert megamoe_op._launch_cluster_configuration([2, 1, 1], None, 100) == expected
