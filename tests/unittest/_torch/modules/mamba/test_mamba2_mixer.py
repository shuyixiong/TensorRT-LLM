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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

mamba2_mixer_mod = pytest.importorskip("tensorrt_llm._torch.modules.mamba.mamba2_mixer")
Mamba2Mixer = mamba2_mixer_mod.Mamba2Mixer


def test_mamba2_mixer_post_load_weights_caches_derived_state():
    mixer = Mamba2Mixer.__new__(Mamba2Mixer)
    nn.Module.__init__(mixer)
    mixer.norm = SimpleNamespace(is_nvfp4=False)
    mixer.A = torch.tensor([1.0, 2.0])
    mixer.dt_bias = torch.tensor([3.0, 4.0])
    mixer.D = torch.tensor([5.0, 6.0])
    mixer.head_dim = 2
    mixer.d_state = 3

    mixer.post_load_weights()

    assert mixer._A_expanded.shape == (2, 2, 3)
    assert mixer._dt_bias_expanded.shape == (2, 2)
    assert mixer._D_expanded.shape == (2, 2)
    assert not hasattr(mixer, "_weights_transformed")


def _split_decode_xbc(xbc, nheads, head_dim, ngroups, d_state):
    x, B, C = torch.split(xbc, [nheads * head_dim, ngroups * d_state, ngroups * d_state], dim=-1)
    return (x.view(x.shape[0], nheads, head_dim), B.view(B.shape[0], ngroups, d_state),
            C.view(C.shape[0], ngroups, d_state))


def test_row_aligned_or_contiguous_keeps_aligned_strided_views():
    row_aligned_or_contiguous = mamba2_mixer_mod._row_aligned_or_contiguous
    # Conv output rows of 4 heads x 8 + 2 x (1 group x 8) bf16 values: 96-byte rows.
    xbc = torch.randn(3, 48, dtype=torch.bfloat16)
    for view in _split_decode_xbc(xbc, 4, 8, 1, 8):
        assert not view.is_contiguous()
        out = row_aligned_or_contiguous(view)
        assert out.data_ptr() == view.data_ptr()
        assert out.stride() == view.stride()

    # A row stride that is not a multiple of 16 bytes forces a copy.
    odd = torch.randn(3, 52, dtype=torch.bfloat16)[:, :32].view(3, 4, 8)
    out = row_aligned_or_contiguous(odd)
    assert out.is_contiguous()
    assert torch.equal(out, odd)

    # Inner dims that are not packed force a copy.
    gapped = torch.randn(3, 4, 16, dtype=torch.bfloat16)[:, :, :8]
    assert row_aligned_or_contiguous(gapped).is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("batch", [1, 2, 8, 32])
def test_flashinfer_ssu_strided_decode_views_match_contiguous(batch):
    fi_mamba = pytest.importorskip("flashinfer.mamba")
    row_aligned_or_contiguous = mamba2_mixer_mod._row_aligned_or_contiguous
    nheads, head_dim, ngroups, d_state, slots = 32, 64, 2, 128, 64
    torch.manual_seed(0)
    xbc = torch.randn(batch, nheads * head_dim + 2 * ngroups * d_state, device="cuda",
                      dtype=torch.bfloat16)
    dt = torch.randn(batch, nheads, 1, device="cuda").expand(batch, nheads, head_dim)
    A = (-torch.rand(nheads, 1, 1, device="cuda")).expand(nheads, head_dim, d_state)
    D = torch.rand(nheads, 1, device="cuda").expand(nheads, head_dim)
    dt_bias = torch.rand(nheads, 1, device="cuda").expand(nheads, head_dim)
    indices = torch.randperm(slots, device="cuda")[:batch].to(torch.int32)
    state0 = torch.randn(slots, nheads, head_dim, d_state, device="cuda")

    results = []
    for strided in (True, False):
        x, B, C = _split_decode_xbc(xbc, nheads, head_dim, ngroups, d_state)
        if strided:
            x, B, C = (row_aligned_or_contiguous(t) for t in (x, B, C))
            assert batch == 1 or not x.is_contiguous()
        else:
            x, B, C = x.contiguous(), B.contiguous(), C.contiguous()
        state = state0.clone()
        out = torch.empty(batch, nheads, head_dim, device="cuda", dtype=torch.bfloat16)
        fi_mamba.selective_state_update(state, x, dt, A, B, C, D, z=None, dt_bias=dt_bias,
                                        dt_softplus=True, state_batch_indices=indices, out=out)
        results.append((out, state))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
