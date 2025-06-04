#!/usr/bin/env python3
"""
Main pipeline for PyTorch → Kevin CUDA → AMD HIP conversion
"""

import json
import time
from typing import Dict, List
from kevin_interface import KevinInterface
from reasoning_extractor import ReasoningExtractor
from amd_translator import ReasoningBasedTranslator

class AMDOptimizationPipeline:
    def __init__(self):
        self.kevin = KevinInterface()
        self.extractor = ReasoningExtractor()
        self.translator = ReasoningBasedTranslator()
    
    def process_model(self, pytorch_code: str, model_name: str = "model") -> Dict:
        """Complete pipeline: PyTorch → Kevin → AMD versions"""
        
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")
        
        # Step 1: Get Kevin's CUDA optimization
        print("\n1. Getting Kevin's NVIDIA optimization...")
        kevin_output = self.kevin.optimize_pytorch_model(pytorch_code)
        
        # Step 2: Extract optimization patterns
        print("\n2. Extracting optimization reasoning...")
        optimizations, summary = self.extractor.extract_optimization_reasoning(kevin_output)
        
        print(f"   Detected optimizations: {[k for k, v in optimizations.items() if v]}")
        
        # Step 3: Generate reasoning trace
        print("\n3. Generating reasoning trace...")
        reasoning_trace = self.extractor.generate_reasoning_trace(kevin_output, optimizations)
        
        # Step 4: Create AMD versions
        print("\n4. Generating AMD versions...")
        amd_versions = self.translator.translate_with_reasoning(kevin_output, optimizations)
        
        # Step 5: Predict performance
        performance_predictions = self._predict_performance(optimizations, reasoning_trace)
        
        return {
            "model_name": model_name,
            "original_pytorch": pytorch_code,
            "kevin_cuda": kevin_output,
            "kevin_summary": summary,
            "optimizations": optimizations,
            "reasoning_trace": reasoning_trace,
            "amd_versions": amd_versions,
            "performance_predictions": performance_predictions,
            "timestamp": time.time()
        }
    
    def _predict_performance(self, optimizations: Dict, trace: List) -> Dict:
        """Predict AMD performance based on optimizations"""
        predictions = {}
        
        # Base predictions
        base_speedup = 1.0
        
        # Adjust based on optimizations
        if optimizations.get('uses_shared_memory'):
            base_speedup *= 1.5
        if optimizations.get('uses_vectorization'):
            base_speedup *= 1.3
        if optimizations.get('uses_tiling'):
            base_speedup *= 1.4
        
        # Apply AMD efficiency factor (typically 85% of NVIDIA)
        amd_factor = 0.85
        
        predictions['conservative'] = {
            'speedup': base_speedup * 0.5 * amd_factor,
            'confidence': 95,
            'bottleneck': 'Not optimized for AMD architecture'
        }
        
        predictions['balanced'] = {
            'speedup': base_speedup * amd_factor,
            'confidence': 75,
            'bottleneck': 'Memory bandwidth' if optimizations.get('uses_shared_memory') else 'Compute'
        }
        
        predictions['aggressive'] = {
            'speedup': base_speedup * 1.2 * amd_factor,
            'confidence': 45,
            'bottleneck': 'Experimental - may not compile on all AMD GPUs'
        }
        
        return predictions
    
    def generate_training_example(self, result: Dict) -> Dict:
        """Format result as training example"""
        return {
            "instruction": "Convert this CUDA kernel to AMD HIP with multiple risk levels",
            "input": result['kevin_cuda'],
            "reasoning_trace": [
                f"Step {step['step']}: {step['action']} - {step['reasoning']}"
                for step in result['reasoning_trace']
            ],
            "output": {
                level: version['code']
                for level, version in result['amd_versions'].items()
            },
            "metadata": {
                "optimizations": result['optimizations'],
                "performance_predictions": result['performance_predictions']
            }
        }

def format_reasoning_trace(trace: List[Dict]) -> str:
    """Format reasoning trace for display"""
    formatted = []
    for step in trace:
        text = f"Step {step['step']}: {step['action']}\n"
        text += f"  Observation: {step['observation']}\n"
        text += f"  Reasoning: {step['reasoning']}"
        if 'adaptation' in step:
            text += f"\n  Adaptation: {step['adaptation']}"
        formatted.append(text)
    return "\n\n".join(formatted)
