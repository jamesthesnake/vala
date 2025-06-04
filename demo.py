#!/usr/bin/env python3
"""
Main demo script for AMD optimization pipeline
"""

import json
import os
from datetime import datetime
from pipeline import AMDOptimizationPipeline, format_reasoning_trace
from test_models import TEST_MODELS
from typing import Dict,List
def save_results(results: Dict, output_dir: str = "output"):
    """Save results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{results['model_name']}_{timestamp}"
    
    # Save complete result as JSON
    with open(f"{output_dir}/{base_name}_complete.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save AMD versions as separate files
    for level, version in results['amd_versions'].items():
        with open(f"{output_dir}/{base_name}_{level}.hip", "w") as f:
            f.write(version['code'])
    
    # Save reasoning trace
    with open(f"{output_dir}/{base_name}_reasoning.txt", "w") as f:
        f.write(format_reasoning_trace(results['reasoning_trace']))
    
    print(f"\nResults saved to {output_dir}/{base_name}_*")

def display_results(results: Dict):
    """Display results in readable format"""
    print("\n" + "="*60)
    print(f"Results for {results['model_name']}")
    print("="*60)
    
    print("\n--- Kevin's Summary ---")
    print(results['kevin_summary'])
    
    print("\n--- Reasoning Trace ---")
    print(format_reasoning_trace(results['reasoning_trace']))
    
    print("\n--- AMD Versions ---")
    for level, version in results['amd_versions'].items():
        print(f"\n{level.upper()} VERSION:")
        print(f"  Risk: {version['risk_level']}")
        print(f"  Confidence: {version['confidence']}%")
        print(f"  Expected Speedup: {version['expected_speedup']}x")
        print(f"  Description: {version['description']}")
        
        # Show first 500 chars of code
        code_preview = version['code'][:500] + "..." if len(version['code']) > 500 else version['code']
        print(f"\n  Code Preview:\n{code_preview}")
    
    print("\n--- Performance Predictions ---")
    for level, pred in results['performance_predictions'].items():
        print(f"\n{level.capitalize()}:")
        print(f"  Speedup: {pred['speedup']:.2f}x")
        print(f"  Confidence: {pred['confidence']}%")
        print(f"  Bottleneck: {pred['bottleneck']}")

def generate_training_dataset(results_list: List[Dict], output_file: str = "amd_training_data.jsonl"):
    """Generate training dataset from results"""
    pipeline = AMDOptimizationPipeline()
    
    with open(output_file, "w") as f:
        for result in results_list:
            training_example = pipeline.generate_training_example(result)
            f.write(json.dumps(training_example) + "\n")
    
    print(f"\nGenerated {len(results_list)} training examples in {output_file}")

def main():
    """Run the complete demo"""
    print("AMD Optimization Pipeline Demo")
    print("="*60)
    
    # Initialize pipeline
    pipeline = AMDOptimizationPipeline()
    
    # Process test models
    results = []
    
    for model_name, pytorch_code in TEST_MODELS.items():
        print(f"\nProcessing {model_name}...")
        
        try:
            result = pipeline.process_model(pytorch_code, model_name)
            results.append(result)
            
            # Display results
            display_results(result)
            
            # Save results
            save_results(result)
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    # Generate training dataset
    if results:
        generate_training_dataset(results)
    
    print("\n" + "="*60)
    print("Demo complete!")
    print(f"Processed {len(results)} models successfully")

if __name__ == "__main__":
    main()
