#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hipcub/hipcub.hpp>

struct ScanTuple {
    hipFloatComplex lambda;
    hipFloatComplex value;
};

 
struct CustomCombine {
    __device__ __forceinline__
    ScanTuple operator()(const ScanTuple& left, const ScanTuple& right) const {
        ScanTuple res;
        res.lambda = hipCmulf(right.lambda, left.lambda);
        res.value  = hipCaddf(hipCmulf(right.lambda, left.value), right.value);
        return res;
    }
};

// hardcode block size to 512 due to physical limit of my card
__global__ void lru_forward_kernel(hipFloatComplex* lam, hipFloatComplex* u, int seq_len, int hidden_size) {
    typedef hipcub::BlockScan<ScanTuple, 512> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int tid = threadIdx.x;
    int batch_hidden_idx = blockIdx.x;
    int hidden_idx = batch_hidden_idx % hidden_size;

    hipFloatComplex* my_u = u + (batch_hidden_idx * seq_len);
    hipFloatComplex my_lam = lam[hidden_idx];

    ScanTuple thread_data;

    if (tid < seq_len) {
        thread_data.lambda = my_lam;
        thread_data.value = my_u[tid];
    } else {
        thread_data = {make_hipFloatComplex(1.0f, 0.0f), make_hipFloatComplex(0.0f, 0.0f)};
    }
    __syncthreads();

    BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, CustomCombine());

    if (tid < seq_len) {
        my_u[tid] = thread_data.value;
    }
}

// launcher
void lru_forward_launcher(hipFloatComplex* lam_ptr, hipFloatComplex* u_ptr, int batch_size, int hidden_size, int seq_len, hipStream_t stream) {
    int grid_size = batch_size * hidden_size;
    int threads = 512;
    
    hipLaunchKernelGGL(lru_forward_kernel, dim3(grid_size), dim3(threads), 0, stream, lam_ptr, u_ptr, seq_len, hidden_size);
}