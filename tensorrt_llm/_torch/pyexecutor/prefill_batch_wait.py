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
"""Bounded, rank-consistent coalescing of first prefill chunks."""

import math


class PrefillBatchWait:
    """Hold sparse first chunks while allowing the previous batch to drain.

    Each vote is ``[tokens, real_decode, continuation, previous_batch,
    stopping, elapsed_ns]``. Only the first rank's elapsed time controls the
    deadline, so differences between host clocks cannot split the decision.
    Drain-only mode releases immediately after prior results are consumed;
    it never starts an additional intake timer.
    """

    def __init__(
        self,
        timeout_ms: float,
        token_ratio: float,
        *,
        drain_only: bool = False,
        adaptive: bool = False,
    ) -> None:
        if not math.isfinite(timeout_ms) or timeout_ms < 0:
            raise ValueError(
                "Prefill batch wait timeout must be finite and nonnegative"
            )
        if not math.isfinite(token_ratio) or not 0 < token_ratio <= 1:
            raise ValueError("Prefill batch wait token ratio must be in (0, 1]")
        self.timeout_ns = int(timeout_ms * 1e6)
        self.token_ratio = token_ratio
        self.drain_only = drain_only
        self.adaptive = adaptive
        self.started_ns: int | None = None
        self.current_limit_ns: int | None = None
        self.last_ready_tokens = 0
        self.last_target_tokens = 0.0
        self.last_limit_ns = 0
        self.last_elapsed_ns = 0
        self.last_reason = "uninitialized"
        self.drain_passes = 0
        self.wait_passes = 0
        self.timeout_releases = 0

    def elapsed_ns(self, now_ns: int) -> int:
        return 0 if self.started_ns is None else max(0, now_ns - self.started_ns)

    def should_wait(self, votes: list[list[int]], max_tokens: int, now_ns: int) -> bool:
        """Return one fleet-wide decision using already gathered readiness."""
        tokens = sum(row[0] for row in votes)
        bypass = any(row[1] or row[2] or row[4] for row in votes)
        target = self.token_ratio * max_tokens * len(votes)
        self.last_ready_tokens = tokens
        self.last_target_tokens = target
        self.last_elapsed_ns = votes[0][5]
        self.last_limit_ns = 0
        if (
            (self.timeout_ns == 0 and not self.drain_only)
            or tokens == 0
            or bypass
            or tokens >= target
        ):
            self.last_reason = (
                "disabled"
                if self.timeout_ns == 0 and not self.drain_only
                else "empty"
                if tokens == 0
                else "bypass"
                if bypass
                else "full"
            )
            self.started_ns = None
            self.current_limit_ns = None
            return False
        if any(row[3] for row in votes):
            # Complete and publish previous results before starting the intake
            # deadline: closed-loop clients need those results to send more work.
            self.started_ns = None
            self.current_limit_ns = None
            self.last_reason = "drain_previous"
            self.drain_passes += 1
            return True
        if self.drain_only:
            self.started_ns = None
            self.current_limit_ns = None
            self.last_reason = "drained_release"
            return False
        proposed_limit_ns = (
            int(self.timeout_ns * (1.0 - tokens / target))
            if self.adaptive
            else self.timeout_ns
        )
        self.current_limit_ns = (
            proposed_limit_ns
            if self.current_limit_ns is None
            else min(self.current_limit_ns, proposed_limit_ns)
        )
        self.last_limit_ns = self.current_limit_ns
        if votes[0][5] >= self.current_limit_ns:
            self.started_ns = None
            self.current_limit_ns = None
            self.last_reason = "deadline_release"
            self.timeout_releases += 1
            return False
        if self.started_ns is None:
            self.started_ns = now_ns
        self.last_reason = "accumulate"
        self.wait_passes += 1
        return True
