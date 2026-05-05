#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> lru_backward_launcher(
    torch::Tensor lamda, 
    torch::Tensor out, 
    torch::Tensor grad_out);

std::vector<torch::Tensor> lru_backward(
    torch::Tensor lamda, 
    torch::Tensor out, 
    torch::Tensor grad_out) {
    TORCH_CHECK(lamda.is_cuda() && out.is_cuda() && grad_out.is_cuda(), "inputs must be on GPU");
    TORCH_CHECK(out.scalar_type() == torch::kComplexFloat, "complex64 only");

    auto lamda_c = lamda.contiguous();
    auto out_c = out.contiguous();
    auto grad_out_c = grad_out.contiguous();

    return lru_backward_launcher(lamda_c, out_c, grad_out_c);
}

// backward need to be same signature to tourch api
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &lru_backward, "LRU backward pass");
}