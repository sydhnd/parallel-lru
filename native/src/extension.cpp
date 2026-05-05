#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

void lru_forward_launcher(hipFloatComplex* lamda_ptr, hipFloatComplex* u_ptr, int batch_size, int hidden_size, int seq_len, hipStream_t stream);

torch::Tensor lru_foward(torch::Tensor lamda, torch::Tensor u) {
    TORCH_CHECK(u.is_cuda() && lamda.is_cuda(), "inputs must be on GPU");
    TORCH_CHECK(u.scalar_type() == torch::kComplexFloat, "u must be complex64");
    TORCH_CHECK(lamda.scalar_type() == torch::kComplexFloat, "lamda must be complex64");

    int batch_size = u.size(0);
    int hidden_size = u.size(1);
    int seq_len = u.size(2);

    auto lamda_gpu = lamda.contiguous();
    auto u_gpu = u.contiguous().clone();

    hipStream_t stream = at::hip::getCurrentHIPStream();

    auto lamda_ptr = reinterpret_cast<hipFloatComplex*>(lamda_gpu.data_ptr<c10::complex<float>>());
    auto u_ptr = reinterpret_cast<hipFloatComplex*>(u_gpu.data_ptr<c10::complex<float>>());

    lru_forward_launcher(lamda_ptr, u_ptr, batch_size, hidden_size, seq_len, stream);

    return u_gpu;
}

// foward need to be same signature to tourch api
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lru_foward, "LRU forward");
}