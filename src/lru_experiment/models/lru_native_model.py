import math

import torch
import torch.nn as nn
from lru_experiment.native.lru_scan import FastLruScan

class LruNativeModel(nn.Module):
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

        # Complex number hidden state
        state = torch.zeros(
            batch_size,
            self.hidden_size,
            dtype=torch.complex64,
            device=x.device,
        )

        Bx = torch.matmul(x.to(torch.complex64), B.t()) 

        # change from batch_size, squseq_len, hidden_size
        # to batch_size, hidden_size, squseq_len
        Bx = Bx.permute(0, 2, 1).contiguous()

        scanned_Bx = FastLruScan.apply(lam, Bx)

        # final state last element of sequence 
        # memory optimization O(L*H)
        final_state = scanned_Bx[:, :, -1]

        final_prediction = torch.real(torch.matmul(final_state, C.t()))
        
        return final_prediction   
