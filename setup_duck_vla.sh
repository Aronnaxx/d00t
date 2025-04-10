#!/bin/bash
# Duck VLA Setup Script
# This script sets up the Duck VLA system with open_duck_playground integration

set -e  # Exit on error

echo "Setting up Duck VLA system..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -sSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc  # Reload shell configuration
fi

# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install required packages
echo "Installing required packages..."
uv pip install -U ollama transformers torch numpy pillow opencv-python

# Check if open_duck_playground submodule exists
if [ ! -d "submodules/open_duck_playground" ]; then
    echo "Setting up open_duck_playground submodule..."
    mkdir -p submodules
    
    # Clone the repository if git is available
    if command -v git &> /dev/null; then
        git clone https://github.com/open-duck/open-duck-playground.git submodules/open_duck_playground
    else
        echo "Git not found. Please manually download and extract open-duck-playground to submodules/open_duck_playground"
        echo "https://github.com/open-duck/open-duck-playground"
        exit 1
    fi
fi

# Install the open_duck_playground package
echo "Installing open_duck_playground..."
cd submodules/open_duck_playground
uv pip install -e .
cd "$SCRIPT_DIR"

# Pull the moondream model in Ollama
echo "Pulling moondream model in Ollama..."
ollama pull moondream

# Create onnx directory if it doesn't exist
mkdir -p duck_vla/onnx
if [ ! -f "duck_vla/onnx/README.md" ]; then
    echo "Creating README for ONNX models directory..."
    echo "# ONNX Models Directory

Place your ONNX model files (.onnx) in this directory. 
The Duck VLA system will use the first .onnx file it finds in this directory.

You can download pre-trained models from the Open Duck GitHub repository.
" > duck_vla/onnx/README.md
fi

# Display information about Ollama
echo ""
echo "Please ensure Ollama is installed and the models you wish to use are available."
echo "You can install Ollama from: https://ollama.com/download"
echo ""
echo "For example, to install the moondream model for vision capabilities:"
echo "    ollama pull moondream"
echo ""

# Display success message and usage instructions
echo ""
echo "Duck VLA setup complete!"
echo ""
echo "To run the Duck VLA system:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Run the simulation: uv run run_duck_sim.py"
echo ""
echo "For additional options:"
echo "  uv run run_duck_sim.py --help"
echo "" 