#!/bin/bash

# Run the AMD optimization demo

echo "Starting AMD Optimization Pipeline Demo"
echo "======================================"

# Check if Kevin model is accessible
python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('cognition-ai/Kevin-32B')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Cannot access Kevin-32B model"
    echo "Please ensure you have access to cognition-ai/Kevin-32B on HuggingFace"
    exit 1
fi

# Create output directory
mkdir -p output
mkdir -p logs

# Run the demo
python3 demo.py 2>&1 | tee logs/demo_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Demo complete! Check the output/ directory for results"
