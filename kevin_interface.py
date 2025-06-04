#!/usr/bin/env python3
"""
Interface for Kevin-32B model - handles all Kevin interactions
"""

import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer, AutoConfig
import transformers.modeling_utils
import time

# Apply initialization bypass
transformers.modeling_utils.PreTrainedModel.post_init = lambda self: self.init_weights()
transformers.modeling_utils.PreTrainedModel._backward_compatibility_gradient_checkpointing = lambda self: None

class KevinInterface:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load Kevin-32B model and tokenizer"""
        print("Loading Kevin-32B...")
        
        # Load config
        config = AutoConfig.from_pretrained("cognition-ai/Kevin-32B")
        if config.sliding_window is None:
            config.sliding_window = False
        
        # Load model
        self.model = Qwen2ForCausalLM.from_pretrained(
            "cognition-ai/Kevin-32B",
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Fix vocab size
        if len(self.tokenizer) < self.model.config.vocab_size:
            diff = self.model.config.vocab_size - len(self.tokenizer)
            self.tokenizer.add_tokens([f"<unused{i}>" for i in range(diff)])
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        print("✓ Kevin ready!")
    
    def generate_cuda_optimization(self, prompt, max_tokens=1500):
        """Generate CUDA optimization with Kevin"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        end_time = time.time()
        print(f"Generated in {end_time - start_time:.1f} seconds")
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def optimize_pytorch_model(self, pytorch_code, target_gpu="H100"):
        """Helper to optimize any PyTorch model"""
        prompt = f"""You are given the following architecture:

{pytorch_code}

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA {target_gpu}. Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew. You're not allowed to use torch.nn (except for Parameter, containers, and init). The input and output have to be on CUDA device. Your answer must be the complete new architecture (no testing code, no other code): it will be evaluated and you will be given feedback on its correctness and speedup so you can keep iterating, trying to maximize the speedup. After your answer, summarize your changes in a few sentences."""
        
        return self.generate_cuda_optimization(prompt)
