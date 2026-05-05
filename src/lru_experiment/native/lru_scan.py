import torch

import lru_scan_forward
import lru_scan_backward 

# PyTorch autograd wrapper 
class FastLruScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lambda_diag, input_scan):
        output_scan = lru_scan_forward.forward(lambda_diag, input_scan)
        ctx.save_for_backward(lambda_diag, input_scan, output_scan)
        return output_scan

    @staticmethod
    def backward(ctx, grad_output):
        lambda_diag, input_scan, output_scan = ctx.saved_tensors

        grad_input, grad_lambda = lru_scan_backward.backward(
            lambda_diag,
            output_scan,
            grad_output.contiguous(),
        )
        
        return grad_lambda, grad_input