#!/usr/bin/env python3
"""
Extract optimization reasoning from Kevin's output
"""

import re
from typing import Dict, Tuple, List, Any

class ReasoningExtractor:
    def __init__(self):
        self.optimization_patterns = {
            'shared_memory': [r'__shared__', r'shared memory', r'smem'],
            'vectorization': [r'float4', r'float2', r'vectorized', r'vector'],
            'warp_primitives': [r'__shfl', r'warpReduce', r'__ballot', r'warp'],
            'tiling': [r'TILE_SIZE', r'tile\[', r'tiling'],
            'fusion': [r'fused', r'fusion', r'combined'],
            'unrolling': [r'#pragma unroll', r'unroll', r'unrolled'],
            'coalescing': [r'coalesced', r'coalescing', r'memory access']
        }
    
    def extract_optimization_reasoning(self, kevin_output: str) -> Tuple[Dict, str]:
        """Extract optimization techniques and reasoning from Kevin's output"""
        
        # Find summary section
        summary = self._extract_summary(kevin_output)
        
        # Extract optimization flags
        optimizations = self._detect_optimizations(kevin_output)
        
        # Extract specific parameters
        optimizations.update({
            'tile_size': self._extract_tile_size(kevin_output),
            'block_size': self._extract_block_size(kevin_output),
            'memory_pattern': self._analyze_memory_pattern(kevin_output),
            'fusion_strategy': 'fused' in summary.lower() if summary else False,
        })
        
        return optimizations, summary
    
    def _extract_summary(self, output: str) -> str:
        """Extract Kevin's summary section"""
        # Look for common summary markers
        markers = [
            "summarize your changes",
            "Summary:",
            "In summary",
            "I've optimized",
            "The optimizations include"
        ]
        
        summary_start = -1
        for marker in markers:
            idx = output.lower().find(marker.lower())
            if idx != -1:
                summary_start = idx
                break
        
        if summary_start == -1:
            # Try to find last paragraph
            paragraphs = output.split('\n\n')
            if paragraphs:
                return paragraphs[-1]
        
        return output[summary_start:] if summary_start != -1 else ""
    
    def _detect_optimizations(self, code: str) -> Dict[str, bool]:
        """Detect which optimization techniques are used"""
        results = {}
        
        for opt_name, patterns in self.optimization_patterns.items():
            results[f'uses_{opt_name}'] = any(
                re.search(pattern, code, re.IGNORECASE) 
                for pattern in patterns
            )
        
        return results
    
    def _extract_tile_size(self, code: str) -> Tuple[int, int]:
        """Extract tile dimensions from code"""
        # Common patterns
        patterns = [
            r'TILE_SIZE\s*=\s*(\d+)',
            r'TILE_DIM\s*=\s*(\d+)',
            r'tile\[(\d+)\]\[(\d+)\]',
            r'TILE_SIZE_X\s*=\s*(\d+).*TILE_SIZE_Y\s*=\s*(\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            if matches:
                if isinstance(matches[0], tuple):
                    return (int(matches[0][0]), int(matches[0][1]))
                else:
                    size = int(matches[0])
                    return (size, size)
        
        return None
    
    def _extract_block_size(self, code: str) -> Tuple[int, int, int]:
        """Extract thread block dimensions"""
        patterns = [
            r'blockDim\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)',
            r'dim3\s+blockDim\((\d+),\s*(\d+)\)',
            r'<<<.*,\s*(\d+)\s*>>>',
            r'BLOCK_SIZE\s*=\s*(\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            if matches:
                if len(matches[0]) == 3:
                    return tuple(int(x) for x in matches[0])
                elif len(matches[0]) == 2:
                    return (int(matches[0][0]), int(matches[0][1]), 1)
                else:
                    size = int(matches[0])
                    return (size, 1, 1)
        
        return None
    
    def _analyze_memory_pattern(self, code: str) -> str:
        """Analyze memory access pattern"""
        if 'coalesced' in code.lower():
            return 'coalesced'
        elif 'strided' in code.lower():
            return 'strided'
        elif '__shared__' in code:
            return 'shared_memory_optimized'
        else:
            return 'unknown'
    
    def generate_reasoning_trace(self, cuda_kernel: str, optimizations: Dict) -> List[Dict]:
        """Generate step-by-step reasoning for AMD adaptation"""
        trace = []
        
        # Step 1: Analyze NVIDIA optimizations
        opt_list = [k.replace('uses_', '') for k, v in optimizations.items() 
                   if k.startswith('uses_') and v]
        
        trace.append({
            "step": 1,
            "action": "Analyze NVIDIA optimizations",
            "observation": f"Kevin optimized using: {', '.join(opt_list)}",
            "reasoning": "These target NVIDIA's 32-thread warps and SM architecture"
        })
        
        # Step 2: Identify AMD differences
        trace.append({
            "step": 2,
            "action": "Identify AMD architectural differences",
            "observation": "AMD GPUs have: 64-thread wavefronts, 64KB LDS per CU, different cache hierarchy",
            "reasoning": "Must adapt thread-level parallelism and memory usage patterns"
        })
        
        # Step 3: Specific adaptations
        step_num = 3
        
        if optimizations.get('uses_warp_primitives'):
            trace.append({
                "step": step_num,
                "action": "Adapt warp primitives to wavefront",
                "observation": "NVIDIA uses 32-thread warp shuffle operations",
                "reasoning": "AMD wavefronts are 64 threads, need different shuffle patterns",
                "adaptation": "Replace warp-level operations with wavefront-aware equivalents"
            })
            step_num += 1
        
        if optimizations.get('tile_size'):
            tile_x, tile_y = optimizations['tile_size']
            trace.append({
                "step": step_num,
                "action": "Adapt tile dimensions",
                "observation": f"Kevin used {tile_x}x{tile_y} tiles for 32-thread warps",
                "reasoning": "AMD's 64-thread wavefronts can process more data per wave",
                "adaptation": f"Consider {tile_x*2}x{tile_y} tiles for better wavefront utilization"
            })
            step_num += 1
        
        if optimizations.get('uses_shared_memory'):
            trace.append({
                "step": step_num,
                "action": "Optimize LDS (Local Data Share) usage",
                "observation": "Kevin used shared memory for data reuse",
                "reasoning": "AMD has 64KB LDS per CU with different bank conflict patterns",
                "adaptation": "Adjust padding and access patterns for AMD's LDS banking"
            })
            step_num += 1
        
        return trace
