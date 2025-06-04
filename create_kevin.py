#!/usr/bin/env python3
"""
Kevin-32B Proper Usage with Correct Input Formats
"""

import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer, AutoConfig
import transformers.modeling_utils
import time

# Apply initialization bypass
transformers.modeling_utils.PreTrainedModel.post_init = lambda self: self.init_weights()
transformers.modeling_utils.PreTrainedModel._backward_compatibility_gradient_checkpointing = lambda self: None

print("Kevin-32B CUDA Optimization Examples")
print("="*50)

# Load model
print("\nLoading Kevin-32B...")
config = AutoConfig.from_pretrained("cognition-ai/Kevin-32B")
if config.sliding_window is None:
    config.sliding_window = False

model = Qwen2ForCausalLM.from_pretrained(
    "cognition-ai/Kevin-32B",
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Fix vocab size
if len(tokenizer) < model.config.vocab_size:
    diff = model.config.vocab_size - len(tokenizer)
    tokenizer.add_tokens([f"<unused{i}>" for i in range(diff)])
    model.resize_token_embeddings(len(tokenizer))

print("✓ Model ready!")

def generate_cuda_optimization(prompt, max_tokens=1500):
    """Generate CUDA optimization with proper format"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    
    print(f"\nGenerating optimization...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    end_time = time.time()
    print(f"Generated in {end_time - start_time:.1f} seconds")
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# =============================================================================
# Example 1: Matrix Multiplication (Primary Use Case)
# =============================================================================
print("\n" + "="*50)
print("Example 1: Matrix Multiplication → CUDA")
print("="*50)

matmul_input = """You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100 (e.g. shared memory, kernel fusion, warp primitives, vectorization,...). Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew. You're not allowed to use torch.nn (except for Parameter, containers, and init). The input and output have to be on CUDA device. Your answer must be the complete new architecture (no testing code, no other code): it will be evaluated and you will be given feedback on its correctness and speedup so you can keep iterating, trying to maximize the speedup. After your answer, summarize your changes in a few sentences."""

result = generate_cuda_optimization(matmul_input)
print("\nGenerated ModelNew:")
print(result)

# Save result
with open("kevin_matmul_optimized.py", "w") as f:
    f.write("# Kevin-32B Generated Matrix Multiplication CUDA Optimization\n\n")
    f.write(result)

# =============================================================================
# Example 2: Softmax Operation
# =============================================================================
print("\n" + "="*50)
print("Example 2: Softmax → CUDA")
print("="*50)

softmax_input = """You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=-1)

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100. Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew."""

result = generate_cuda_optimization(softmax_input, max_tokens=1000)
print("\nGenerated Softmax CUDA Implementation:")
print(result[:1000] + "..." if len(result) > 1000 else result)

# =============================================================================
# Example 3: Fused Linear + GELU
# =============================================================================
print("\n" + "="*50)
print("Example 3: Fused Linear + GELU → CUDA")
print("="*50)

fused_input = """You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # Linear layer followed by GELU activation
        output = torch.nn.functional.linear(x, weight, bias)
        return torch.nn.functional.gelu(output)

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100. Try to fuse operations where possible."""

result = generate_cuda_optimization(fused_input, max_tokens=1200)
print("\nGenerated Fused Kernel:")
print(result[:1000] + "..." if len(result) > 1000 else result)

# =============================================================================
# Helper Function for Custom PyTorch Operations
# =============================================================================

def optimize_my_model(pytorch_code, description="", target_gpu="H100"):
    """Helper to optimize any PyTorch model with Kevin-32B"""
    
    prompt = f"""You are given the following architecture:

{pytorch_code}

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA {target_gpu}. {description}Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew. You're not allowed to use torch.nn (except for Parameter, containers, and init). The input and output have to be on CUDA device. Your answer must be the complete new architecture."""
    
    return generate_cuda_optimization(prompt)

# Example: Optimize a custom operation
print("\n" + "="*50)
print("Custom Example: Your Own PyTorch Code")
print("="*50)

custom_pytorch = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_size: int):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom computation: Linear + ReLU + Normalization
        out = torch.matmul(x, self.weight) + self.bias
        out = torch.relu(out)
        return out / torch.norm(out, dim=-1, keepdim=True)"""

print("Optimizing custom PyTorch model...")
custom_result = optimize_my_model(
    custom_pytorch,
    description="Focus on fusing all operations into a single kernel. ",
    target_gpu="A100"
)

print("\nGenerated Custom CUDA Optimization:")
print(custom_result[:800] + "..." if len(custom_result) > 800 else custom_result)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*50)
print("Kevin-32B Usage Summary")
print("="*50)
print("""
Kevin-32B is specifically designed to convert PyTorch nn.Module architectures
into optimized CUDA implementations. 

Key inputs it expects:
1. Complete PyTorch nn.Module class with __init__ and forward methods
2. The exact prompt format starting with "You are given the following architecture:"
3. Specific optimization instructions (target GPU, techniques to use)
4. Request for "ModelNew" as the output class name

The model will generate:
- A complete ModelNew class using torch.utils.cpp_extension.load_inline
- Raw CUDA kernels optimized for the target GPU
- Proper memory management and kernel launching code
- Summary of optimizations applied

Use the optimize_my_model() helper function to easily convert your PyTorch code!
""")
