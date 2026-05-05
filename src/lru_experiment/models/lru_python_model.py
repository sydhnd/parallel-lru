import math

import torch
import torch.nn as nn


class LruPythonModel(nn.Module):

    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()

        self.hidden_size = hidden_size

        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(torch.rand(hidden_size) * 0.9 + 0.1))
        )
        self.theta_log = nn.Parameter(
            torch.log(torch.pi * torch.rand(hidden_size))
        )

        # Complex input projection B.
        self.B_re = nn.Parameter(
            torch.randn(hidden_size, in_features) / math.sqrt(in_features)
        )
        self.B_im = nn.Parameter(
            torch.randn(hidden_size, in_features) / math.sqrt(in_features)
        )

        # Complex output projection C.
        self.C_re = nn.Parameter(
            torch.randn(out_features, hidden_size) / math.sqrt(hidden_size)
        )
        self.C_im = nn.Parameter(
            torch.randn(out_features, hidden_size) / math.sqrt(hidden_size)
        )

    def get_complex_params(self):
        decay = torch.exp(-torch.exp(self.nu_log))
        phase = torch.exp(self.theta_log)

        lam = torch.complex(
            decay,
            torch.zeros_like(decay),
        ) * torch.exp(1j * phase)

        B = torch.complex(self.B_re, self.B_im)
        C = torch.complex(self.C_re, self.C_im)

        return lam, B, C

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        lam, B, C = self.get_complex_params()

        # Complex hidden state 
        state = torch.zeros(
            batch_size,
            self.hidden_size,
            dtype=torch.complex64,
            device=x.device,
        )

        # Sequential Python loop version.
        # The ROCm/C++ version can replace this part later.
        for t in range(seq_len):
            u_t = x[:, t, :].to(torch.complex64)
            state = lam * state + torch.matmul(u_t, B.t())

        # only care the result of last state
        # Sequence-to-one output using only the final state.
        # dropping intermediate states saves O(L*H) memory.
        return torch.real(torch.matmul(state, C.t()))