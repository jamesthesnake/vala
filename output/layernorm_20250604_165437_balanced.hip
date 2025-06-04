You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = 1e-5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100. Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew. You're not allowed to use torch.nn (except for Parameter, containers, and init). The input and output have to be on CUDA device. Your answer must be the complete new architecture (no testing code, no other code): it will be evaluated and you will be given feedback on its correctness and speedup so you can keep iterating, trying to maximize the speedup. After your answer, summarize your changes in a few sentences. The initial architecture is the starting point, and your goal is to beat the initial implementation's speed. The speed will be measured on inputs of shape (256, 768) and (4096, 768). The faster your solution, the better. The accuracy has to be the same as the original implementation. The original implementation uses instance norm. The input is a 2D tensor. The original code uses x.mean(dim=-1, keepdim=True), which is instance norm. The original code is equivalent to torch.nn.InstanceNorm1d. The input is 2D (N, C), so the InstanceNorm1d would have affine=False, but here the code has learnable weight and bias. So the original code is: instance norm (over the last dimension) followed by scaling and shifting with learnable parameters. The original code is correct. Your new code must be equivalent but faster. The problem is to replace the PyTorch operators in the original code with raw CUDA kernels for better performance on H100. The new model must be called ModelNew and use torch.utils.cpp_extension.load_inline to define the kernel. The forward method should call the kernel. The kernel must handle the computation of mean, variance, normalization, scaling, and shifting. The kernel must be written inline in the code using load_inline. The kernel should be as efficient as possible, using shared memory, optimal thread blocks, etc., for H100. The input and output are on CUDA. The parameters (weight and bias) are the same as the original model. The eps is 1e-5. The new model must have the same parameters (self.weight and self.bias) and the same eps. The forward method must take a tensor x and return the result as in the original code but using the kernel. The code must be self-contained, using only torch and torch.utils.cpp_extension. The original code uses x.mean and x.var, which are along the last dimension (since dim=-1). The input is 2D, so each instance is a vector of length 768, and the mean/variance is computed over that vector (i.e., for each sample in the batch, compute mean/var over the 768 features). So instance norm per sample. The kernel needs to process each sample (each row) independently. So the kernel can be structured to process each row in parallel. Each thread block can handle one row. Within a block, threads can compute the mean and variance. For a row of length 768, a block of 256 threads could be used, with each thread responsible for a few elements. The mean can be computed by each thread summing their elements, then using a reduction in the block to get the total sum. Similarly for the variance. Then, the normalization is done per element. The kernel needs to: 1. For each row (each sample), compute the mean of the 768 elements. 2. Compute the variance. 3. Subtract the mean and divide by sqrt(var + eps). 4. Multiply by weight and add bias. The weight and bias are vectors of size 768, so the scaling and shifting is done element-wise. The kernel must be written to handle all of this. The key is to make the kernel as efficient as possible. Using shared memory within a block for the reduction steps (summing for mean and variance) would be essential. The thread block size should be chosen to balance memory usage and parallelism. For 768 elements, a block size of 256 or 128 threads could work. Each thread can process 3 or 6 elements for the sum. Then, perform a block-wide reduction to get the total sum for the mean. For variance, after the mean is known, compute the sum of squared differences. Again, using shared memory for the reduction. Once the mean and variance are computed for the row, each thread can compute the normalized value for their elements, then apply the weight and bias. The kernel code needs to be inline in the Python code using load_inline. The kernel will take the input tensor, weight, bias, eps, and output tensor. The kernel launch should be configured with a grid of N threads (where N is the number of rows) and a block of, say, 256 threads. Wait, actually, each row is processed by a block. So the grid size would be equal to the number of rows (x.size(0)), and each block would have a number of threads, say 256, to process the 768 elements. So the kernel signature would be something like: template <typename scalar> __global__ void norm_kernel(...). The arguments would include the input data, output data, weight, bias, eps, and dimensions. The kernel code must be written carefully to handle all steps efficiently. Here's my attempt at writing the ModelNew class using an inline CUDA kernel:

import torch
import torch.utils.cpp_extension

class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size: int = 768):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = 1e-5

        kernel_code = '''
        #include <torch/extension.h>
        #include <hip/hip_runtime.h>
        #include <hip/hip_runtime.h>

        template <typename scalar_t>
        __global__ void layer_norm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y,
                                         const scalar_t* __restrict__ weight, const scalar_t* __restrict__ bias,
                                         float eps, int B, int D) {
            extern __shared__ scalar_t shared[];
            const int idx = blockIdx.x * D + threadIdx.x;
            const int bid = blockIdx.x;
            const int tid = threadIdx.x;

            // Load input into shared memory
            if (tid < D) {
                shared[tid] = x[bid * D + tid];
            }
            __syncthreads();

            // Compute mean
            scalar_t sum = 0;
            for (int i = 0; i < D; ++i) {
                sum += shared[i];
            }
            sum = blockReduceSum(sum);
            if (tid == 0) {
                shared[D] = sum / D;
            }
            __syncthreads();

            // Compute variance
            scalar_t var_sum = 0;
            for (int i = 0; i < D; ++i) {
                scalar_t dist = shared[i] - shared[D];
                var_sum += dist * dist;
            }
            var_sum = blockReduceSum(var_sum);
            if (tid == 0) {
                shared[D+1] = var_sum / D + eps;
            }
            __syncthreads();

            // Normalize and apply weight/bias
            if (tid < D) {
                scalar_t mean = shared[D];
                scalar_t var = shared[D+1];
                scalar_t normed = (shared[tid] - mean) / sqrt(var);
                y[idx] = normed * weight[tid]