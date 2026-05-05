#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <torch/types.h>

__global__ void lru_backward_kernel(
    const hipFloatComplex* lam, 
    const hipFloatComplex* out, 
    const hipFloatComplex* grad_out,
    hipFloatComplex* grad_x, 
    hipFloatComplex* grad_lam_out,
    int batch_size, int hidden_size, int seq_len) {
    
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int h = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (b < batch_size && h < hidden_size) {
        hipFloatComplex running_dx = make_hipFloatComplex(0.0f, 0.0f);
        hipFloatComplex lam_conj = hipConjf(lam[h]);
        hipFloatComplex total_grad_lam = make_hipFloatComplex(0.0f, 0.0f);

        for (int t = seq_len - 1; t >= 0; --t) {
            int idx = (b * hidden_size * seq_len) + (h * seq_len) + t;
            
            running_dx = hipCaddf(grad_out[idx], hipCmulf(lam_conj, running_dx));
            grad_x[idx] = running_dx;

            if (t > 0) {
                int prev_idx = (b * hidden_size * seq_len) + (h * seq_len) + (t - 1);
                total_grad_lam = hipCaddf(total_grad_lam, hipCmulf(running_dx, hipConjf(out[prev_idx])));
            }
        }
        grad_lam_out[b * hidden_size + h] = total_grad_lam;
    }
}

// launcher
std::vector<torch::Tensor> lru_backward_launcher(
    torch::Tensor lam, torch::Tensor out, torch::Tensor grad_out) {
    
    TORCH_CHECK(out.is_contiguous() && grad_out.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(out.scalar_type() == torch::kComplexFloat, "complex64 only")
    
    int batch_size = out.size(0);
    int hidden_size = out.size(1);
    int seq_len = out.size(2);
    
    auto grad_x = torch::zeros_like(out);
    auto grad_lam_out = torch::zeros({batch_size, hidden_size}, out.options());

    constexpr int TILE_SIZE = 16; 
    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const dim3 blocks((batch_size + 15) / TILE_SIZE, (hidden_size + 15) / TILE_SIZE);

    hipLaunchKernelGGL(lru_backward_kernel, blocks, threads, 0, 0,
        (const hipFloatComplex*)lam.data_ptr<c10::complex<float>>(),
        (const hipFloatComplex*)out.data_ptr<c10::complex<float>>(),
        (const hipFloatComplex*)grad_out.data_ptr<c10::complex<float>>(),
        (hipFloatComplex*)grad_x.data_ptr<c10::complex<float>>(),
        (hipFloatComplex*)grad_lam_out.data_ptr<c10::complex<float>>(),
        batch_size, hidden_size, seq_len
    );

    return {grad_x, grad_lam_out.sum(0)};
}