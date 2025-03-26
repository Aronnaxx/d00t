#!/bin/bash
# run.sh - Script to run the Isaac Lab robot integration demo with UV package manager

# Create directories if they don't exist
mkdir -p assets logs

# Check if robot USD exists, create it if not
if [ ! -f "assets/robot.usd" ]; then
    echo "Robot USD not found, creating it..."
    uv run create-robot
fi

# Check if ONNX model exists
if [ ! -f "model.onnx" ]; then
    echo "ONNX model will be created automatically when running main.py"
fi

# Run the main simulation
echo "Starting Isaac Lab robot simulation..."
uv run main "$@" 