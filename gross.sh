#!/bin/bash
# Multi-Turn RL KernelSWiRL Setup Script
# Fast setup for hackathon - gets you generating in minutes

set -e

echo "🚀 Setting up Multi-Turn RL KernelSWiRL"
echo "====================================="

# Create project structure
echo "📁 Creating RL project structure..."
mkdir -p rl_kernelswirl/{src,configs,data,outputs,scripts}
mkdir -p rl_kernelswirl/src/{rl_engine,generators,hardware,validators}
mkdir -p rl_kernelswirl/data/{kernels,trajectories,rewards}
mkdir -p rl_kernelswirl/outputs/{trajectories,checkpoints,visualizations}

cd rl_kernelswirl

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Create requirements.txt with RL dependencies
cat > requirements.txt << 'EOF'
# Core ML/RL
torch>=2.0.0
numpy>=1.24.0
gymnasium>=0.29.0

# LLM Integration  
openai>=1.0.0
anthropic>=0.18.0
google-generativeai>=0.3.0
vllm>=0.5.0

# Data Processing
pandas>=2.0.0
jsonlines>=4.0.0
pyyaml>=6.0.0

# Visualization
matplotlib>=3.7.0
plotly>=5.17.0
streamlit>=1.28.0
wandb>=0.16.0

# Development
rich>=13.0.0
typer>=0.9.0
python-dotenv>=1.0.0
tqdm>=4.66.0
EOF

echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create RL configuration
echo "⚙️ Creating RL configuration..."
cat > configs/rl_config.yaml << 'EOF'
# Multi-Turn RL Configuration

rl_settings:
  algorithm: "dqn"  # Fast for hackathon
  epsilon_start: 0.9
  epsilon_end: 0.1
  epsilon_decay: 0.995
  gamma: 0.95
  learning_rate: 0.001
  batch_size: 32
  memory_size: 10000
  
action_space:
  # Optimization actions
  optimizations:
    - "apply_tiling"
    - "use_shared_memory"
    - "optimize_memory_access"
    - "vectorize_operations"
    - "use_tensor_cores"
    - "reduce_divergence"
    - "improve_occupancy"
    - "fuse_kernels"
    
  # Exploration actions  
  exploration:
    - "try_alternative"
    - "backtrack"
    - "increase_tile_size"
    - "decrease_tile_size"
    - "change_parallelization"
    
  # Hardware-specific actions
  hardware_specific:
    cuda:
      - "use_wmma"
      - "optimize_for_sm80"
      - "use_async_copy"
    hip:
      - "use_mfma" 
      - "optimize_for_wavefront64"
      - "use_lds"
    sycl:
      - "use_sub_groups"
      - "optimize_for_xe"
      - "use_slm"

reward_structure:
  # Performance rewards
  speedup_2x: 1.0
  speedup_1_5x: 0.7
  speedup_1_2x: 0.4
  no_improvement: 0.0
  slower: -0.5
  
  # Compilation rewards
  compiles_correctly: 0.2
  syntax_error: -1.0
  runtime_error: -0.8
  
  # Learning rewards
  teaches_new_concept: 0.3
  demonstrates_pitfall: 0.4
  shows_hardware_diff: 0.5
  
  # Diversity rewards
  unique_approach: 0.2
  redundant_approach: -0.1

hardware_specs:
  cuda:
    warp_size: 32
    max_threads: 1024
    has_tensor_cores: true
  hip:
    wavefront_size: 64
    max_threads: 1024
    has_matrix_cores: true
  sycl:
    subgroup_size: 32
    max_threads: 1024
    has_xmx: true

trajectory_settings:
  max_turns: 8
  min_turns: 3
  early_stop_reward: 0.95
  failure_exploration_rate: 0.2
  
generation_targets:
  trajectories_per_kernel: 50
  kernels_to_process: 20
  total_trajectories: 1000
  
model_settings:
  primary_model: "gpt-4o-mini"  # Fast and cheap
  enhancement_model: "claude-3-5-sonnet-20241022"
  local_model: "Qwen/Qwen2.5-Coder-7B-Instruct"  # Smaller for speed
EOF

# Create the main RL engine
echo "🧠 Creating RL engine..."
cat > src/rl_engine/kernel_rl.py << 'EOF'
"""
Multi-Turn RL Engine for Kernel Optimization Trajectories
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque, defaultdict
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import yaml
import json
from pathlib import Path

@dataclass
class State:
    """RL State representation"""
    kernel_features: Dict[str, float]  # Kernel characteristics
    optimization_history: List[str]    # Applied optimizations
    performance_metrics: Dict[str, float]  # Current performance
    hardware: str                      # Target hardware
    turn: int                         # Current turn number
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for NN"""
        features = []
        
        # Kernel features (normalized)
        features.extend([
            self.kernel_features.get('memory_intensity', 0),
            self.kernel_features.get('compute_intensity', 0),
            self.kernel_features.get('parallelism', 0),
            self.kernel_features.get('data_reuse', 0)
        ])
        
        # One-hot encode last 3 optimizations
        opt_encoding = [0] * 30  # 10 possible opts * 3 history
        for i, opt in enumerate(self.optimization_history[-3:]):
            if opt in OPTIMIZATION_TO_IDX:
                opt_encoding[i * 10 + OPTIMIZATION_TO_IDX[opt]] = 1
        features.extend(opt_encoding)
        
        # Performance metrics
        features.extend([
            self.performance_metrics.get('speedup', 1.0),
            self.performance_metrics.get('memory_efficiency', 0.5),
            self.performance_metrics.get('occupancy', 0.5)
        ])
        
        # Hardware encoding
        hw_encoding = [0] * 4  # 4 hardware types
        hw_map = {'cuda': 0, 'hip': 1, 'sycl': 2, 'metal': 3}
        if self.hardware in hw_map:
            hw_encoding[hw_map[self.hardware]] = 1
        features.extend(hw_encoding)
        
        # Turn number (normalized)
        features.append(self.turn / 10.0)
        
        return np.array(features, dtype=np.float32)

@dataclass
class Action:
    """Action in the RL environment"""
    optimization: str
    parameters: Dict[str, any]
    
class DQN(nn.Module):
    """Deep Q-Network for action selection"""
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class KernelOptimizationRL:
    """Multi-turn RL for kernel optimization trajectory generation"""
    
    def __init__(self, config_path: str = "configs/rl_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.setup_action_space()
        self.setup_dqn()
        self.memory = deque(maxlen=self.config['rl_settings']['memory_size'])
        self.epsilon = self.config['rl_settings']['epsilon_start']
        
        # Track generation statistics
        self.stats = defaultdict(int)
        self.trajectory_history = []
        
    def setup_action_space(self):
        """Initialize action space from config"""
        self.actions = []
        
        # Add optimization actions
        for opt in self.config['action_space']['optimizations']:
            self.actions.append(Action(opt, {}))
            
        # Add exploration actions
        for exp in self.config['action_space']['exploration']:
            self.actions.append(Action(exp, {}))
            
        self.action_to_idx = {a.optimization: i for i, a in enumerate(self.actions)}
        global OPTIMIZATION_TO_IDX
        OPTIMIZATION_TO_IDX = self.action_to_idx
        
    def setup_dqn(self):
        """Initialize DQN"""
        state_size = 42  # Based on State.to_vector() size
        action_size = len(self.actions)
        
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), 
            lr=self.config['rl_settings']['learning_rate']
        )
        
    def select_action(self, state: State) -> Action:
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(self.actions)
        else:
            # Exploitation: best Q-value
            state_vector = torch.FloatTensor(state.to_vector()).unsqueeze(0)
            q_values = self.q_network(state_vector)
            action_idx = q_values.argmax().item()
            return self.actions[action_idx]
            
    def calculate_reward(self, 
                        state: State, 
                        action: Action, 
                        result: Dict[str, any]) -> float:
        """Calculate reward based on action result"""
        reward = 0.0
        rewards = self.config['reward_structure']
        
        # Performance reward
        speedup = result.get('speedup', 1.0)
        if speedup >= 2.0:
            reward += rewards['speedup_2x']
        elif speedup >= 1.5:
            reward += rewards['speedup_1_5x']
        elif speedup >= 1.2:
            reward += rewards['speedup_1_2x']
        elif speedup < 1.0:
            reward += rewards['slower']
            
        # Compilation reward
        if result.get('compiles', False):
            reward += rewards['compiles_correctly']
        elif result.get('syntax_error', False):
            reward += rewards['syntax_error']
        elif result.get('runtime_error', False):
            reward += rewards['runtime_error']
            
        # Learning value reward
        if self.is_novel_approach(state, action):
            reward += rewards['teaches_new_concept']
            
        if result.get('demonstrates_pitfall', False):
            reward += rewards['demonstrates_pitfall']
            
        # Hardware-specific bonus
        if state.hardware != 'cuda' and result.get('hardware_specific', False):
            reward += rewards['shows_hardware_diff']
            
        return reward
        
    def is_novel_approach(self, state: State, action: Action) -> bool:
        """Check if this optimization approach is novel"""
        # Simple novelty: not in recent history
        return action.optimization not in state.optimization_history[-3:]
        
    def generate_trajectory(self, 
                          kernel_code: str, 
                          kernel_name: str,
                          hardware: str = "cuda") -> Dict[str, any]:
        """Generate a complete optimization trajectory using RL"""
        
        # Initialize state
        state = State(
            kernel_features=self.analyze_kernel(kernel_code),
            optimization_history=[],
            performance_metrics={'speedup': 1.0},
            hardware=hardware,
            turn=0
        )
        
        trajectory = {
            'kernel_name': kernel_name,
            'kernel_code': kernel_code,
            'hardware': hardware,
            'steps': []
        }
        
        # Multi-turn optimization loop
        for turn in range(self.config['trajectory_settings']['max_turns']):
            # Select action
            action = self.select_action(state)
            
            # Execute action (would call LLM here)
            result = self.execute_action(state, action, kernel_code)
            
            # Calculate reward
            reward = self.calculate_reward(state, action, result)
            
            # Store step in trajectory
            trajectory['steps'].append({
                'turn': turn,
                'action': action.optimization,
                'state_features': state.kernel_features,
                'result': result,
                'reward': reward,
                'accumulated_speedup': result.get('speedup', 1.0)
            })
            
            # Update state
            next_state = self.update_state(state, action, result)
            
            # Store experience
            self.memory.append((
                state.to_vector(),
                self.action_to_idx[action.optimization],
                reward,
                next_state.to_vector(),
                result.get('terminal', False)
            ))
            
            # Train DQN
            if len(self.memory) > self.config['rl_settings']['batch_size']:
                self.train_step()
                
            state = next_state
            state.turn = turn + 1
            
            # Early stopping
            if reward > self.config['trajectory_settings']['early_stop_reward']:
                break
                
        # Decay epsilon
        self.epsilon *= self.config['rl_settings']['epsilon_decay']
        self.epsilon = max(self.epsilon, self.config['rl_settings']['epsilon_end'])
        
        # Calculate trajectory quality
        trajectory['total_reward'] = sum(s['reward'] for s in trajectory['steps'])
        trajectory['final_speedup'] = trajectory['steps'][-1]['accumulated_speedup']
        
        self.trajectory_history.append(trajectory)
        self.stats['trajectories_generated'] += 1
        
        return trajectory
        
    def analyze_kernel(self, kernel_code: str) -> Dict[str, float]:
        """Extract features from kernel code"""
        features = {}
        
        # Simple heuristics for demo
        code_lower = kernel_code.lower()
        
        # Memory intensity (global memory accesses)
        memory_accesses = code_lower.count('[') + code_lower.count('*')
        features['memory_intensity'] = min(memory_accesses / 50.0, 1.0)
        
        # Compute intensity (arithmetic operations)
        compute_ops = sum(code_lower.count(op) for op in ['+', '-', '*', '/', 'fma'])
        features['compute_intensity'] = min(compute_ops / 30.0, 1.0)
        
        # Parallelism (thread/block references)
        parallel_refs = sum(code_lower.count(ref) for ref in ['threadidx', 'blockidx', 'blockdim'])
        features['parallelism'] = min(parallel_refs / 10.0, 1.0)
        
        # Data reuse potential
        features['data_reuse'] = 0.5  # Would need deeper analysis
        
        return features
        
    def execute_action(self, state: State, action: Action, kernel_code: str) -> Dict[str, any]:
        """Execute optimization action - would call LLM in real implementation"""
        
        # For hackathon demo, simulate results based on action
        result = {
            'compiles': True,
            'speedup': 1.0,
            'code': kernel_code,  # Would be modified code
            'explanation': f"Applied {action.optimization} optimization"
        }
        
        # Simulate different optimization impacts
        optimization_impacts = {
            'apply_tiling': {'speedup': 1.5, 'helps_memory': True},
            'use_shared_memory': {'speedup': 2.0, 'helps_memory': True},
            'use_tensor_cores': {'speedup': 4.0, 'requires': 'tensor_core_hw'},
            'vectorize_operations': {'speedup': 1.3, 'helps_compute': True},
            'optimize_memory_access': {'speedup': 1.4, 'helps_memory': True},
            'try_alternative': {'speedup': 0.9, 'exploration': True},
            'backtrack': {'speedup': 1.0, 'terminal': True}
        }
        
        if action.optimization in optimization_impacts:
            impact = optimization_impacts[action.optimization]
            
            # Apply impact
            current_speedup = state.performance_metrics.get('speedup', 1.0)
            result['speedup'] = current_speedup * impact.get('speedup', 1.0)
            
            # Hardware-specific adjustments
            if state.hardware == 'cuda' and 'tensor_core' in action.optimization:
                result['hardware_specific'] = True
            elif state.hardware == 'hip' and 'wavefront' in str(impact):
                result['hardware_specific'] = True
                
            # Simulate failures occasionally
            if random.random() < 0.1:
                result['compiles'] = False
                result['syntax_error'] = True
                result['speedup'] = 0.0
                
        return result
        
    def update_state(self, state: State, action: Action, result: Dict) -> State:
        """Update state based on action result"""
        new_state = State(
            kernel_features=state.kernel_features.copy(),
            optimization_history=state.optimization_history + [action.optimization],
            performance_metrics={
                'speedup': result.get('speedup', state.performance_metrics['speedup']),
                'memory_efficiency': state.performance_metrics.get('memory_efficiency', 0.5),
                'occupancy': state.performance_metrics.get('occupancy', 0.5)
            },
            hardware=state.hardware,
            turn=state.turn + 1
        )
        
        return new_state
        
    def train_step(self):
        """Train DQN on batch from memory"""
        if len(self.memory) < self.config['rl_settings']['batch_size']:
            return
            
        batch_size = self.config['rl_settings']['batch_size']
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.config['rl_settings']['gamma'] * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.stats['training_steps'] += 1
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'stats': dict(self.stats)
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.stats = defaultdict(int, checkpoint['stats'])

# Quick test
if __name__ == "__main__":
    rl = KernelOptimizationRL()
    
    test_kernel = """
__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""
    
    trajectory = rl.generate_trajectory(test_kernel, "matmul_test", "cuda")
    print(f"Generated trajectory with {len(trajectory['steps'])} steps")
    print(f"Total reward: {trajectory['total_reward']:.2f}")
    print(f"Final speedup: {trajectory['final_speedup']:.2fx}")
EOF

# Create trajectory generator with LLM integration
echo "🤖 Creating LLM-integrated generator..."
cat > src/generators/rl_trajectory_generator.py << 'EOF'
"""
RL-driven trajectory generator with LLM integration
"""

import asyncio
import json
from typing import Dict, List, Any
from pathlib import Path
import yaml
import openai
from rich.console import Console

from src.rl_engine.kernel_rl import KernelOptimizationRL, State, Action

console = Console()

class RLTrajectoryGenerator:
    """Generate optimization trajectories using RL + LLM"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.rl_engine = KernelOptimizationRL()
        self.model = model
        self.optimization_templates = self.load_templates()
        
    def load_templates(self) -> Dict[str, str]:
        """Load optimization templates"""
        return {
            "apply_tiling": """Apply tiling optimization to this kernel:
{kernel_code}

Current state: {state_desc}
Hardware: {hardware}

Generate:
1. Tiled version of the kernel
2. Explanation of tile size choice
3. Expected performance impact""",

            "use_shared_memory": """Optimize using shared memory:
{kernel_code}

Current optimizations: {history}
Hardware: {hardware}

Generate:
1. Kernel using shared memory
2. Bank conflict analysis
3. Memory access pattern improvement""",

            "use_tensor_cores": """Optimize for tensor cores:
{kernel_code}

Hardware: {hardware} (supports tensor cores: {has_tensor_cores})

Generate:
1. Tensor core enabled kernel
2. Data layout requirements
3. Performance expectations""",

            "try_alternative": """Current approach not working well. Try alternative:
{kernel_code}

Failed attempts: {history}
Current performance: {performance}

Suggest and implement a different optimization approach."""
        }
        
    async def execute_llm_action(self, 
                                state: State, 
                                action: Action, 
                                kernel_code: str) -> Dict[str, Any]:
        """Execute action using LLM"""
        
        # Build prompt
        template = self.optimization_templates.get(
            action.optimization,
            "Apply {action} optimization to the kernel."
        )
        
        prompt = template.format(
            kernel_code=kernel_code,
            state_desc=self.describe_state(state),
            hardware=state.hardware,
            history=", ".join(state.optimization_history),
            performance=f"{state.performance_metrics['speedup']:.1f}x",
            has_tensor_cores=state.hardware in ['cuda', 'hip'],
            action=action.optimization
        )
        
        try:
            # Call LLM
            response = await self.call_llm(prompt)
            
            # Parse response
            result = self.parse_llm_response(response, action)
            
            # Validate generated code
            result['compiles'] = self.validate_syntax(result.get('code', ''))
            
            # Estimate performance (would run profiler in real system)
            result['speedup'] = self.estimate_speedup(state, action, result)
            
            return result
            
        except Exception as e:
            console.print(f"[red]LLM error: {e}[/red]")
            return {
                'success': False,
                'error': str(e),
                'speedup': 1.0,
                'compiles': False
            }
            
    async def call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        # For hackathon, using OpenAI for speed
        client = openai.AsyncOpenAI()
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert GPU kernel optimization engineer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
        
    def parse_llm_response(self, response: str, action: Action) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        result = {
            'raw_response': response,
            'action': action.optimization,
            'code': '',
            'explanation': '',
            'expected_impact': ''
        }
        
        # Extract code blocks
        code_blocks = []
        in_code = False
        current_block = []
        
        for line in response.split('\n'):
            if '```' in line:
                if in_code and current_block:
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                in_code = not in_code
            elif in_code:
                current_block.append(line)
                
        if code_blocks:
            result['code'] = code_blocks[0]
            
        # Extract sections (simple heuristic)
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if 'explanation' in line.lower() and i + 1 < len(lines):
                result['explanation'] = lines[i + 1]
            elif 'performance' in line.lower() and i + 1 < len(lines):
                result['expected_impact'] = lines[i + 1]
                
        return result
        
    def validate_syntax(self, code: str) -> bool:
        """Basic syntax validation"""
        if not code:
            return False
            
        # Check for basic kernel structure
        required = ['__global__', 'void', '{', '}']
        return all(req in code for req in required)
        
    def estimate_speedup(self, state: State, action: Action, result: Dict) -> float:
        """Estimate speedup based on optimization and response"""
        base_speedup = state.performance_metrics['speedup']
        
        # Optimization-specific multipliers
        multipliers = {
            'apply_tiling': 1.5,
            'use_shared_memory': 2.0,
            'use_tensor_cores': 4.0,
            'vectorize_operations': 1.3,
            'optimize_memory_access': 1.4
        }
        
        if action.optimization in multipliers:
            # Check if LLM thinks it will work
            if 'significant' in result.get('expected_impact', '').lower():
                return base_speedup * multipliers[action.optimization]
            elif 'moderate' in result.get('expected_impact', '').lower():
                return base_speedup * (multipliers[action.optimization] * 0.7)
            else:
                return base_speedup * 1.1
                
        return base_speedup
        
    def describe_state(self, state: State) -> str:
        """Generate human-readable state description"""
        desc = f"Memory intensity: {state.kernel_features['memory_intensity']:.1f}, "
        desc += f"Compute intensity: {state.kernel_features['compute_intensity']:.1f}, "
        desc += f"Current speedup: {state.performance_metrics['speedup']:.1f}x"
        
        if state.optimization_history:
            desc += f", Applied: {', '.join(state.optimization_history[-3:])}"
            
        return desc
        
    async def generate_rl_trajectory(self,
                                   kernel_code: str,
                                   kernel_name: str,
                                   hardware: str = "cuda") -> Dict[str, Any]:
        """Generate trajectory using RL + LLM"""
        
        console.print(f"[cyan]Generating RL trajectory for {kernel_name} on {hardware}[/cyan]")
        
        # Let RL engine handle the multi-turn loop
        # But override execute_action to use LLM
        original_execute = self.rl_engine.execute_action
        self.rl_engine.execute_action = lambda s, a, k: asyncio.run(
            self.execute_llm_action(s, a, k)
        )
        
        trajectory = self.rl_engine.generate_trajectory(
            kernel_code, kernel_name, hardware
        )
        
        # Restore original
        self.rl_engine.execute_action = original_execute
        
        # Enhance trajectory with LLM explanations
        trajectory['enhanced_steps'] = []
        for step in trajectory['steps']:
            enhanced = step.copy()
            
            # Add reasoning
            enhanced['reasoning'] = await self.generate_step_reasoning(step)
            
            # Add hardware notes
            if hardware != 'cuda':
                enhanced['hardware_notes'] = await self.generate_hardware_notes(
                    step, hardware
                )
                
            trajectory['enhanced_steps'].append(enhanced)
            
        return trajectory
        
    async def generate_step_reasoning(self, step: Dict) -> str:
        """Generate reasoning for why this step was taken"""
        prompt = f"""Explain why applying '{step['action']}' optimization makes sense given:
- Current speedup: {step.get('accumulated_speedup', 1.0):.1f}x
- Reward received: {step['reward']:.2f}

Provide a brief technical explanation (2-3 sentences)."""

        response = await self.call_llm(prompt)
        return response.strip()
        
    async def generate_hardware_notes(self, step: Dict, hardware: str) -> str:
        """Generate hardware-specific notes"""
        prompt = f"""For {hardware} hardware, what specific considerations apply to '{step['action']}'?
Focus on differences from CUDA. Be concise."""

        response = await self.call_llm(prompt)
        return response.strip()

# Batch trajectory generator
class BatchRLGenerator:
    """Generate many trajectories efficiently"""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.generators = [RLTrajectoryGenerator() for _ in range(num_workers)]
        
    async def generate_batch(self, 
                           kernels: List[Tuple[str, str]], 
                           hardware_list: List[str],
                           trajectories_per_kernel: int = 10) -> List[Dict]:
        """Generate trajectories in parallel"""
        
        tasks = []
        
        for kernel_code, kernel_name in kernels:
            for hardware in hardware_list:
                for i in range(trajectories_per_kernel):
                    generator = self.generators[
                        len(tasks) % self.num_workers
                    ]
                    
                    task = generator.generate_rl_trajectory(
                        kernel_code,
                        f"{kernel_name}_v{i}",
                        hardware
                    )
                    tasks.append(task)
                    
        console.print(f"[green]Generating {len(tasks)} trajectories in parallel...[/green]")
        
        trajectories = await asyncio.gather(*tasks)
        
        return trajectories

if __name__ == "__main__":
    # Test the generator
    generator = RLTrajectoryGenerator()
    
    test_kernel = """
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""
    
    trajectory = asyncio.run(
        generator.generate_rl_trajectory(test_kernel, "vector_add_test", "cuda")
    )
    
    print(f"Generated trajectory with {len(trajectory['steps'])} steps")
    print(f"Final speedup: {trajectory.get('final_speedup', 1.0):.1f}x")
    
    # Save sample
    with open("outputs/sample_rl_trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2)
EOF

# Create monitoring dashboard
echo "📊 Creating RL monitoring dashboard..."
cat > src/utils/rl_monitor.py << 'EOF'
"""
Real-time monitoring for RL trajectory generation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path
import numpy as np

st.set_page_config(
    page_title="RL KernelSWiRL Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_trajectories(output_dir: Path, limit: int = 100):
    """Load recent trajectories"""
    trajectories = []
    
    for f in sorted(output_dir.glob("*.json"), 
                   key=lambda x: x.stat().st_mtime, 
                   reverse=True)[:limit]:
        try:
            with open(f, 'r') as file:
                trajectories.append(json.load(file))
        except:
            pass
            
    return trajectories

def main():
    st.title("🧠 RL KernelSWiRL Monitor")
    st.markdown("Real-time monitoring of multi-turn RL trajectory generation")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        output_dir = Path(st.text_input("Output Directory", "outputs/trajectories"))
        refresh_rate = st.slider("Refresh Rate (s)", 5, 60, 10)
        show_rewards = st.checkbox("Show Reward Details", True)
        
    # Load data
    trajectories = load_trajectories(output_dir)
    
    if not trajectories:
        st.warning("No trajectories found. Start generating!")
        return
        
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trajectories", len(trajectories))
        
    with col2:
        avg_reward = np.mean([t.get('total_reward', 0) for t in trajectories])
        st.metric("Avg Total Reward", f"{avg_reward:.2f}")
        
    with col3:
        avg_speedup = np.mean([t.get('final_speedup', 1.0) for t in trajectories])
        st.metric("Avg Final Speedup", f"{avg_speedup:.1f}x")
        
    with col4:
        avg_steps = np.mean([len(t.get('steps', [])) for t in trajectories])
        st.metric("Avg Steps/Trajectory", f"{avg_steps:.1f}")
        
    # Charts
    st.header("📈 Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Reward distribution
        rewards = [t.get('total_reward', 0) for t in trajectories]
        fig = px.histogram(
            rewards, nbins=20,
            title="Total Reward Distribution",
            labels={"value": "Total Reward", "count": "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Speedup vs Steps
        data = []
        for t in trajectories:
            steps = len(t.get('steps', []))
            speedup = t.get('final_speedup', 1.0)
            data.append({'Steps': steps, 'Speedup': speedup})
            
        df = pd.DataFrame(data)
        fig = px.scatter(
            df, x='Steps', y='Speedup',
            title="Speedup vs Trajectory Length",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # Action usage
    st.header("🎮 Action Analysis")
    
    action_counts = {}
    action_rewards = {}
    
    for t in trajectories:
        for step in t.get('steps', []):
            action = step.get('action', 'unknown')
            reward = step.get('reward', 0)
            
            action_counts[action] = action_counts.get(action, 0) + 1
            if action not in action_rewards:
                action_rewards[action] = []
            action_rewards[action].append(reward)
            
    # Calculate average rewards
    action_data = []
    for action, counts in action_counts.items():
        avg_reward = np.mean(action_rewards[action])
        action_data.append({
            'Action': action,
            'Count': counts,
            'Avg Reward': avg_reward
        })
        
    df_actions = pd.DataFrame(action_data).sort_values('Count', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df_actions.head(10), x='Count', y='Action',
            orientation='h', title="Most Used Actions"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        fig = px.bar(
            df_actions.sort_values('Avg Reward', ascending=False).head(10),
            x='Avg Reward', y='Action',
            orientation='h', title="Highest Reward Actions",
            color='Avg Reward', color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # Hardware comparison
    st.header("🖥️ Hardware Comparison")
    
    hw_data = {}
    for t in trajectories:
        hw = t.get('hardware', 'unknown')
        if hw not in hw_data:
            hw_data[hw] = {'speedups': [], 'rewards': []}
        hw_data[hw]['speedups'].append(t.get('final_speedup', 1.0))
        hw_data[hw]['rewards'].append(t.get('total_reward', 0))
        
    if hw_data:
        hw_comparison = []
        for hw, data in hw_data.items():
            hw_comparison.append({
                'Hardware': hw,
                'Avg Speedup': np.mean(data['speedups']),
                'Avg Reward': np.mean(data['rewards']),
                'Count': len(data['speedups'])
            })
            
        df_hw = pd.DataFrame(hw_comparison)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Avg Speedup',
            x=df_hw['Hardware'],
            y=df_hw['Avg Speedup'],
            yaxis='y',
            offsetgroup=1
        ))
        fig.add_trace(go.Bar(
            name='Avg Reward',
            x=df_hw['Hardware'],
            y=df_hw['Avg Reward'],
            yaxis='y2',
            offsetgroup=2
        ))
        
        fig.update_layout(
            title='Hardware Performance Comparison',
            yaxis=dict(title='Avg Speedup', side='left'),
            yaxis2=dict(title='Avg Reward', overlaying='y', side='right')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    # Recent trajectories
    st.header("📋 Recent Trajectories")
    
    recent = []
    for t in trajectories[:10]:
        recent.append({
            'Kernel': t.get('kernel_name', 'unknown'),
            'Hardware': t.get('hardware', 'unknown'),
            'Steps': len(t.get('steps', [])),
            'Final Speedup': f"{t.get('final_speedup', 1.0):.1f}x",
            'Total Reward': f"{t.get('total_reward', 0):.2f}"
        })
        
    st.dataframe(pd.DataFrame(recent), use_container_width=True)
    
    # Trajectory explorer
    if st.checkbox("Show Trajectory Explorer"):
        st.header("🔍 Trajectory Explorer")
        
        selected_idx = st.selectbox(
            "Select trajectory",
            range(len(trajectories)),
            format_func=lambda x: f"{trajectories[x].get('kernel_name', 'unknown')} - {trajectories[x].get('hardware', 'unknown')}"
        )
        
        if selected_idx is not None:
            traj = trajectories[selected_idx]
            
            # Show steps
            for i, step in enumerate(traj.get('steps', [])):
                with st.expander(f"Step {i+1}: {step.get('action', 'unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Reward", f"{step.get('reward', 0):.2f}")
                    with col2:
                        st.metric("Speedup", f"{step.get('accumulated_speedup', 1.0):.1f}x")
                    with col3:
                        st.metric("Turn", step.get('turn', i))
                        
                    if 'result' in step:
                        st.json(step['result'])
                        
    # Auto refresh
    if st.button("🔄 Refresh Now"):
        st.rerun()
        
    # Add countdown
    st.empty().text(f"Auto refresh in {refresh_rate} seconds...")
    
if __name__ == "__main__":
    main()
EOF

# Create sample kernels
echo "📝 Creating sample kernels..."
cat > data/kernels/samples.py << 'EOF'
"""Sample kernels for RL trajectory generation"""

SAMPLE_KERNELS = {
    "matmul_naive": """__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}""",

    "reduction": """__global__ void reduce_sum(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata[];
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}""",

    "convolution": """__global__ void conv2d(float* input, float* kernel, float* output,
                              int width, int height, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_k = ksize / 2;
        
        for (int ky = -half_k; ky <= half_k; ky++) {
            for (int kx = -half_k; kx <= half_k; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    sum += input[iy * width + ix] * 
                           kernel[(ky + half_k) * ksize + (kx + half_k)];
                }
            }
        }
        output[y * width + x] = sum;
    }
}""",

    "stencil": """__global__ void stencil_3d(float* input, float* output,
                                int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if (x < nx-1 && y < ny-1 && z < nz-1) {
        int idx = z * nx * ny + y * nx + x;
        
        output[idx] = 0.1667f * (
            input[idx] +
            input[idx - 1] + input[idx + 1] +
            input[idx - nx] + input[idx + nx] +
            input[idx - nx*ny] + input[idx + nx*ny]
        );
    }
}""",

    "scan": """__global__ void scan(float* input, float* output, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int offset = 1;
    
    temp[2*tid] = input[2*tid];
    temp[2*tid+1] = input[2*tid+1];
    
    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    if (tid == 0) temp[n - 1] = 0;
    
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    output[2*tid] = temp[2*tid];
    output[2*tid+1] = temp[2*tid+1];
}"""
}

def get_kernel_list():
    """Get list of (code, name) tuples"""
    return [(code, name) for name, code in SAMPLE_KERNELS.items()]
EOF

# Create main execution script
echo "🎮 Creating main execution script..."
cat > run_rl_generation.py << 'EOF'
#!/usr/bin/env python3
"""
Main script to run RL-based trajectory generation
"""

import asyncio
import json
from pathlib import Path
import typer
from rich.console import Console
from rich.progress import track
from datetime import datetime

from src.generators.rl_trajectory_generator import BatchRLGenerator
from data.kernels.samples import get_kernel_list

app = typer.Typer()
console = Console()

@app.command()
def generate(
    num_trajectories: int = typer.Option(100, help="Number of trajectories to generate"),
    hardware: str = typer.Option("cuda,hip,sycl", help="Comma-separated hardware targets"),
    output_dir: str = typer.Option("outputs/trajectories", help="Output directory"),
    num_workers: int = typer.Option(4, help="Parallel workers"),
    trajectories_per_kernel: int = typer.Option(5, help="Trajectories per kernel/hardware combo")
):
    """Generate RL-optimized trajectories"""
    
    console.print("[bold green]🚀 Starting RL Trajectory Generation[/bold green]")
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    hardware_list = hardware.split(',')
    kernels = get_kernel_list()
    
    # Calculate actual number of trajectories
    total = len(kernels) * len(hardware_list) * trajectories_per_kernel
    console.print(f"Generating {total} trajectories ({len(kernels)} kernels × {len(hardware_list)} hardware × {trajectories_per_kernel} variants)")
    
    # Initialize generator
    generator = BatchRLGenerator(num_workers=num_workers)
    
    # Generate trajectories
    start_time = datetime.now()
    
    async def run_generation():
        trajectories = await generator.generate_batch(
            kernels, 
            hardware_list,
            trajectories_per_kernel
        )
        
        # Save trajectories
        console.print(f"\n[cyan]Saving {len(trajectories)} trajectories...[/cyan]")
        
        stats = {
            'total_reward': 0,
            'total_speedup': 0,
            'by_hardware': {},
            'by_action': {}
        }
        
        for i, traj in enumerate(track(trajectories, description="Saving")):
            # Save trajectory
            filename = f"{traj['kernel_name']}_{traj['hardware']}_{i:04d}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(traj, f, indent=2)
                
            # Update stats
            stats['total_reward'] += traj.get('total_reward', 0)
            stats['total_speedup'] += traj.get('final_speedup', 1.0)
            
            hw = traj['hardware']
            if hw not in stats['by_hardware']:
                stats['by_hardware'][hw] = {'count': 0, 'reward': 0, 'speedup': 0}
            stats['by_hardware'][hw]['count'] += 1
            stats['by_hardware'][hw]['reward'] += traj.get('total_reward', 0)
            stats['by_hardware'][hw]['speedup'] += traj.get('final_speedup', 1.0)
            
            # Track action usage
            for step in traj.get('steps', []):
                action = step.get('action', 'unknown')
                if action not in stats['by_action']:
                    stats['by_action'][action] = {'count': 0, 'reward': 0}
                stats['by_action'][action]['count'] += 1
                stats['by_action'][action]['reward'] += step.get('reward', 0)
                
        return trajectories, stats
        
    trajectories, stats = asyncio.run(run_generation())
    
    # Print summary
    duration = (datetime.now() - start_time).total_seconds()
    
    console.print("\n[bold green]✅ Generation Complete![/bold green]")
    console.print(f"Time: {duration:.1f} seconds ({len(trajectories)/duration:.1f} trajectories/sec)")
    console.print(f"Average reward: {stats['total_reward']/len(trajectories):.2f}")
    console.print(f"Average speedup: {stats['total_speedup']/len(trajectories):.1f}x")
    
    # Hardware breakdown
    console.print("\n[bold]Hardware Breakdown:[/bold]")
    for hw, data in stats['by_hardware'].items():
        count = data['count']
        console.print(f"  {hw}: {count} trajectories, "
                     f"avg reward {data['reward']/count:.2f}, "
                     f"avg speedup {data['speedup']/count:.1f}x")
        
    # Top actions
    console.print("\n[bold]Top Actions by Usage:[/bold]")
    sorted_actions = sorted(stats['by_action'].items(), 
                          key=lambda x: x[1]['count'], 
                          reverse=True)[:5]
    for action, data in sorted_actions:
        console.print(f"  {action}: {data['count']} uses, "
                     f"avg reward {data['reward']/data['count']:.2f}")

@app.command()
def monitor():
    """Launch monitoring dashboard"""
    console.print("[bold yellow]🎯 Launching RL monitoring dashboard...[/bold yellow]")
    import subprocess
    subprocess.run(["streamlit", "run", "src/utils/rl_monitor.py"])

@app.command()
def quickstart():
    """Quick generation for testing"""
    console.print("[bold cyan]🏃 Running quick start generation...[/bold cyan]")
    
    # Generate just 10 trajectories for testing
    generate(
        num_trajectories=10,
        hardware="cuda",
        trajectories_per_kernel=2,
        num_workers=2
    )

if __name__ == "__main__":
    app()
EOF

chmod +x run_rl_generation.py

# Create .env template
cat > .env.template << 'EOF'
# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_AI_API_KEY=your_google_key_here

# Model Selection
PRIMARY_MODEL=gpt-4o-mini
ENHANCEMENT_MODEL=claude-3-5-sonnet-20241022
LOCAL_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct

# RL Settings
RL_LEARNING_RATE=0.001
RL_EPSILON_DECAY=0.995
RL_BATCH_SIZE=32
EOF

# Create quick test script
cat > test_rl_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test RL setup is working"""

import sys
from pathlib import Path

def test_imports():
    """Test all imports work"""
    try:
        import torch
        print("✅ PyTorch installed")
        
        import gymnasium
        print("✅ Gymnasium installed")
        
        from src.rl_engine.kernel_rl import KernelOptimizationRL
        print("✅ RL engine loads")
        
        from src.generators.rl_trajectory_generator import RLTrajectoryGenerator
        print("✅ Generator loads")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_rl_engine():
    """Test RL engine works"""
    try:
        from src.rl_engine.kernel_rl import KernelOptimizationRL
        
        rl = KernelOptimizationRL()
        
        # Test trajectory generation (without LLM)
        kernel = """__global__ void test(float* a) { a[0] = 1.0f; }"""
        trajectory = rl.generate_trajectory(kernel, "test", "cuda")
        
        print(f"✅ Generated trajectory with {len(trajectory['steps'])} steps")
        print(f"   Total reward: {trajectory['total_reward']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ RL engine error: {e}")
        return False

def main():
    print("🧪 Testing RL KernelSWiRL Setup\n")
    
    if not test_imports():
        return 1
        
    if not test_rl_engine():
        return 1
        
    print("\n✅ All tests passed! Ready for RL trajectory generation.")
    print("\nQuick start: python run_rl_generation.py quickstart")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_rl_setup.py

# Final message
echo ""
echo "✅ RL KernelSWiRL Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Copy and configure .env:"
echo "   cp .env.template .env"
echo "   # Add your API keys"
echo ""
echo "2. Test the setup:"
echo "   source venv/bin/activate"
echo "   python test_rl_setup.py"
echo ""
echo "3. Quick test (10 trajectories):"
echo "   python run_rl_generation.py quickstart"
echo ""
echo "4. Full generation:"
echo "   python run_rl_generation.py generate --num-trajectories 1000"
echo ""
echo "5. Monitor progress:"
echo "   python run_rl_generation.py monitor"
echo ""
echo "📚 The RL engine will learn which optimizations work best for different kernels!"
