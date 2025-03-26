#!/bin/bash
# setup.sh - Script to set up the Isaac Lab robot integration project with UV

echo "Setting up Isaac Lab robot integration project..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV package manager not found. Please install it first:"
    echo "curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/astral-sh/uv/main/install.sh | sh"
    exit 1
fi

# Create directories
mkdir -p assets logs

# Install dependencies using UV
echo "Installing dependencies with UV..."
uv pip install -e . --extra-index-url https://pypi.nvidia.com

# Create the robot USD file
echo "Creating robot USD file..."
uv run create-robot

echo "Setup complete! You can now run the simulation with:"
echo "  ./run.sh"
echo "Or with UV directly:"
echo "  uv run main" 