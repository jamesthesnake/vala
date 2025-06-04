You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=-1)

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100. Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew. You're not allowed to use torch.nn (except for Parameter, containers, and init). The input and output have to be on CUDA device. Your answer must be the complete new architecture (no testing code, no other code): it will be evaluated and you will be given feedback on its correctness and speedup so you can keep iterating, trying to maximize the speedup. After your answer, summarize your changes in a few sentences. The softmax function is the only operation in the original model. The goal is to make the new model as fast as possible on H100. The input tensor is 2D with shape (batch_size, num_classes), and the sum of each row is not necessarily 1 before softmax. The output must match PyTorch's softmax (up to floating point tolerances). The original code uses torch.nn.functional.softmax. The user wants a faster implementation using raw CUDA kernels. The input and output must be on the CUDA device. The new model must be in a class called ModelNew. The code must use torch.utils.cpp_extension.load_inline to define the kernel. The kernel must be written in the answer. The code must not use any PyTorch operators except for containers and parameters. The kernel must be inlined in the code. The kernel must be named "softmax_kernel". The kernel must take input and output tensors as arguments. The kernel must be launched correctly in the forward method. The kernel must handle the entire computation (subtract max, exponentiate, sum, divide). The kernel must be optimized for H100's architecture, using shared memory, thread blocks, and threads appropriately. The kernel must be correct and as fast as possible. The code must be self-contained. The final class must be named ModelNew and have a forward method that returns the output tensor.

Here's the optimized architecture using a custom CUDA kernel for softmax:

```python
import torch
import torch.utils.cpp_extension

kernel_code = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

__global__ void softmax_kernel(const float* __restrict__ x, float* y, int batch_size, int num_classes) {
    extern __shared__ float shared_data[];
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int class_idx = tid;
    
    // Load input into shared memory
    if (class_idx < num_classes) {
        shared_data[tid] = x[batch_idx * num_classes + class_idx];
    } else {
        shared_data[tid] = -INFINITY;
    }
    __syncthreads();
    
    // Compute max in shared memory
    for (int stride = 1; stride <= num_classes; stride *= 2) {
        if (class_idx % (2 * stride) == 0) {
            if (class_idx + stride < num_classes) {
                shared_data[class_idx] = max(shared_data[class_idx], shared_data[class_idx + stride]);
            }
        }
        __syncthreads();
    }
    float max_val = shared_data[0];
    
    // Subtract max and exponentiate
    if (class_idx < num_classes) {
        shared_data[tid] = expf(shared_data[tid] - max_val);
    }
    __syncthreads();
    
    // Compute sum in shared memory
    for (int stride = 1; stride <= num_classes; stride *= 2) {
        if (class_idx % (2 * stride) == 0) {
            if (class_idx + stride < num_classes) {
                shared_data[class_idx] += shared_data[class_idx + stride];
            }
        }
        __syncthreads();
    }
    float sum_val = shared_data[0];
    
    // Divide by sum
    if (class_idx < num_classes) {
        y[batch_idx * num_classes + class_idx] = shared_data[tid] / sum_val;
    }
}

at::Tensor softmax_forward(const at::Tensor x) {
    auto output = at::empty_like(x);
    int batch_size = x.size(0);
    int num_classes = x.size(1);
    const int block_size = num_classes; // One thread per class
    softmax_kernel<<<batch_size, block_size, block_size * sizeof(float)>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), batch_size, num_classes);
    return output;
}
"""

softmax_module = torch.utils.cpp_extension.load_inline(
    name="softmax_ext",
    cpp_sources="",
    cuda_sources=kernel_code,
    extra_cuda_cflags=["-use_fast_math", "-gencode", "arch=compute_89,code=sm_89"],
    with_cuda=True
)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, x):
        return softmax_module.softmax_forward(x)
```

Changes made:
1. Replaced PyTorch's softmax with a custom CUDA kernel using shared memory for efficient parallel computation.
2. The kernel uses one thread block per batch sample, with one thread per class.
3. Shared memory is used to store the input values, compute the maximum, exponentiate values, and calculate the sum.
4. The kernel uses a binary reduction approach in shared memory for max and sum calculations to minimize global memory accesses.
5. Optimized for H100 (SM 89) with fast math flags and proper thread configuration.
6. The kernel handles all steps of the softmax computation in parallel within each batch sample.
```

Plan:
1. I'll create a CUDA kernel that processes each batch element in a separate block.
2. Each thread in a block will handle one class.
3. Use shared memory to store the input values for the current batch element.
4. First, find the maximum value using a reduction in shared memory.
5. Subtract the max and compute exponentials in shared memory.
6. Perform another reduction to calculate the sum of exponentials.
7. Finally, divide each exponential by the sum to get the softmax values.
8. Use binary reduction steps to efficiently compute max and sum.
9. Configure the kernel to use one thread per class in each block.
10. Compile with flags optimized for H100 and fast math.
```
The answer provided contains a CUDA kernel implementation for the softmax function, aiming to optimize performance on the NVIDIA H100 architecture. The kernel uses shared memory to efficiently perform the necessary computations in parallel. Here's the breakdown of the changes:

1. **Custom CUDA Kernel**: Replaced the PyTorch `softmax` function with a hand-crafted CUDA kernel named `softmax_kernel`. This kernel is designed to handle the entire softmax computation (subtracting the maximum, exponentiating, summing, and dividing) in parallel.

2. **Thread Configuration**: The kernel uses one thread block per batch sample, with each thread in a block responsible for a single class. This configuration ensures that each batch element is processed independently, leveraging the H100's multi-threaded architecture.

3. **Shared Memory Utilization**: The kernel extensively uses shared memory to store intermediate results. This reduces global memory access latency and allows for efficient data sharing among threads within a block. The input values, maximum value, exponentials, and sum are all computed and stored in shared memory.

4. **Reduction Operations**: The kernel employs binary reduction techniques in shared memory to compute the maximum and sum values. This approach minimizes the number of memory transactions and maximizes parallelism.

5. **Optimized Compilation Flags**: The kernel is compiled with `-use_fast_math` and targeting the H100's compute capability (`sm_89`