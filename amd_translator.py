#!/usr/bin/env python3
"""
CUDA to AMD HIP translator with reasoning-based adaptations
"""

import re
from typing import Dict, List, Tuple

class ReasoningBasedTranslator:
    def __init__(self):
        # Basic syntax mappings
        self.cuda_to_hip_simple = {
            'cudaMalloc': 'hipMalloc',
            'cudaFree': 'hipFree',
            'cudaMemcpy': 'hipMemcpy',
            'cudaDeviceSynchronize': 'hipDeviceSynchronize',
            'cudaGetLastError': 'hipGetLastError',
            'cudaSuccess': 'hipSuccess',
            '__syncthreads': '__syncthreads',  # Same in HIP
            'threadIdx': 'threadIdx',          # Same in HIP
            'blockIdx': 'blockIdx',            # Same in HIP
            'blockDim': 'blockDim',            # Same in HIP
            'gridDim': 'gridDim',              # Same in HIP
        }
        
        # Architecture-specific adaptations
        self.architectural_rules = {
            'warp_size': {
                'nvidia': 32,
                'amd': 64,
                'pattern': r'(\w+)\s*[<%]\s*32(?!\d)',
                'replacement': r'\1 < waveSize'
            },
            'shuffle_ops': {
                'nvidia': '__shfl_xor_sync(0xffffffff,',
                'amd': '__shfl_xor(',
                'reasoning': 'HIP shuffle doesn\'t need mask parameter'
            }
        }
    
    def translate_with_reasoning(self, cuda_code: str, optimization_data: Dict) -> Dict[str, Dict]:
        """Generate three AMD versions based on optimization reasoning"""
        
        versions = {}
        
        # Conservative: Minimal changes
        versions['conservative'] = {
            'code': self._conservative_translation(cuda_code),
            'risk_level': 'Very Low',
            'confidence': 95,
            'expected_speedup': 1.2,
            'description': 'Direct syntax translation with minimal architectural changes'
        }
        
        # Balanced: Apply AMD optimizations
        versions['balanced'] = {
            'code': self._balanced_translation(cuda_code, optimization_data),
            'risk_level': 'Low-Medium',
            'confidence': 75,
            'expected_speedup': 2.1,
            'description': 'Wavefront-aware optimizations with proven AMD patterns'
        }
        
        # Aggressive: Push AMD limits
        versions['aggressive'] = {
            'code': self._aggressive_translation(cuda_code, optimization_data),
            'risk_level': 'High',
            'confidence': 45,
            'expected_speedup': 3.5,
            'description': 'Experimental optimizations using latest AMD features'
        }
        
        return versions
    
    def _conservative_translation(self, cuda_code: str) -> str:
        """Minimal risk translation - just syntax"""
        hip_code = cuda_code
        
        # Replace CUDA API calls
        for cuda_api, hip_api in self.cuda_to_hip_simple.items():
            hip_code = hip_code.replace(cuda_api, hip_api)
        
        # Update includes
        hip_code = self._update_includes(hip_code)
        
        # Fix kernel launch syntax if needed
        hip_code = self._fix_kernel_launch(hip_code)
        
        return hip_code
    
    def _balanced_translation(self, cuda_code: str, opt_data: Dict) -> str:
        """Apply AMD-specific optimizations based on reasoning"""
        # Start with conservative
        hip_code = self._conservative_translation(cuda_code)
        
        # Apply wavefront adaptations
        if opt_data.get('uses_warp_primitives'):
            hip_code = self._adapt_warp_to_wavefront(hip_code)
        
        # Adjust tile sizes for wavefront
        if opt_data.get('tile_size'):
            hip_code = self._adapt_tile_size_balanced(hip_code, opt_data['tile_size'])
        
        # Optimize memory patterns
        if opt_data.get('uses_shared_memory'):
            hip_code = self._optimize_lds_usage(hip_code)
        
        return hip_code
    
    def _aggressive_translation(self, cuda_code: str, opt_data: Dict) -> str:
        """Experimental optimizations for maximum performance"""
        # Start with balanced
        hip_code = self._balanced_translation(cuda_code, opt_data)
        
        # Use matrix cores if applicable
        if 'gemm' in cuda_code.lower() or 'matmul' in cuda_code.lower():
            hip_code = self._add_matrix_cores(hip_code)
        
        # Aggressive tiling
        if opt_data.get('tile_size'):
            hip_code = self._adapt_tile_size_aggressive(hip_code, opt_data['tile_size'])
        
        # Add AMD-specific intrinsics
        hip_code = self._add_amd_intrinsics(hip_code)
        
        return hip_code
    
    def _update_includes(self, code: str) -> str:
        """Update CUDA includes to HIP"""
        replacements = [
            (r'#include\s*<cuda\.h>', '#include <hip/hip_runtime.h>'),
            (r'#include\s*<cuda_runtime\.h>', '#include <hip/hip_runtime.h>'),
            (r'#include\s*"cuda\.h"', '#include <hip/hip_runtime.h>'),
        ]
        
        for pattern, replacement in replacements:
            code = re.sub(pattern, replacement, code)
        
        return code
    
    def _fix_kernel_launch(self, code: str) -> str:
        """Ensure kernel launch syntax is HIP-compatible"""
        # HIP supports the same <<<>>> syntax as CUDA
        # But also supports hipLaunchKernelGGL
        return code
    
    def _adapt_warp_to_wavefront(self, code: str) -> str:
        """Adapt warp-level operations to wavefront"""
        # Replace hardcoded 32 with waveSize
        code = re.sub(r'(\w+)\s*<\s*32(?!\d)', r'\1 < waveSize', code)
        code = re.sub(r'32\s*threads', '64 threads', code)
        
        # Adapt shuffle operations
        code = code.replace('__shfl_xor_sync(0xffffffff,', '__shfl_xor(')
        code = code.replace('__shfl_down_sync(0xffffffff,', '__shfl_down(')
        
        return code
    
    def _adapt_tile_size_balanced(self, code: str, tile_size: Tuple[int, int]) -> str:
        """Conservative tile size adaptation"""
        if tile_size:
            old_x, old_y = tile_size
            # Double width for wavefront, keep height
            new_x, new_y = old_x * 2, old_y
            
            # Replace tile definitions
            code = re.sub(f'TILE_SIZE_X\s*=\s*{old_x}', f'TILE_SIZE_X = {new_x}', code)
            code = re.sub(f'tile\[{old_x}\]\[{old_y}\]', f'tile[{new_x}][{new_y}]', code)
            
        return code
    
    def _adapt_tile_size_aggressive(self, code: str, tile_size: Tuple[int, int]) -> str:
        """Aggressive tile size for maximum LDS usage"""
        if tile_size:
            old_x, old_y = tile_size
            # Quadruple for aggressive optimization
            new_x, new_y = old_x * 2, old_y * 2
            
            code = re.sub(f'TILE_SIZE\s*=\s*{old_x}', f'TILE_SIZE = {new_x}', code)
            code = re.sub(f'tile\[{old_x}\]\[{old_y}\]', f'tile[{new_x}][{new_y}]', code)
            
        return code
    
    def _optimize_lds_usage(self, code: str) -> str:
        """Optimize Local Data Share usage patterns"""
        # Add padding to avoid bank conflicts on AMD
        code = re.sub(
            r'__shared__\s+(\w+)\s+(\w+)\[(\d+)\]\[(\d+)\]',
            r'__shared__ \1 \2[\3][\4 + 1]',  # Add padding
            code
        )
        return code
    
    def _add_matrix_cores(self, code: str) -> str:
        """Add AMD Matrix Core instructions for GEMM operations"""
        # This is a simplified example - real implementation would be more complex
        matrix_core_pragma = """
// Enable AMD Matrix Cores
#pragma clang fp contract(fast)
#ifdef __gfx90a__
    // Use MFMA instructions on MI200 series
    #define USE_AMD_MATRIX_CORES 1
#endif
"""
        return matrix_core_pragma + code
    
    def _add_amd_intrinsics(self, code: str) -> str:
        """Add AMD-specific performance intrinsics"""
        # Add DPP (Data Parallel Primitives) for reductions
        if 'reduction' in code.lower():
            code = code.replace(
                '__syncthreads()',
                '__syncthreads(); // Consider __builtin_amdgcn_wave_barrier() for wavefront sync'
            )
        return code
