#!/usr/bin/env python
"""
Open Duck MuJoCo Inference Runner

This script runs the MuJoCo inference for the duck directly using UV.
It will launch the MuJoCo simulation window with the duck moving based
on a pre-trained ONNX model.

Usage:
    uv run run_mujoco_duck.py [--debug] 
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("mujoco_runner")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Open Duck MuJoCo Inference Runner"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        help="Path to ONNX model file (will look for one if not specified)"
    )
    
    return parser.parse_args()

def find_onnx_model(playground_path):
    """Find an ONNX model in the playground directory."""
    # Look for ONNX models in the playground directory
    onnx_files = list(playground_path.glob("**/*.onnx"))
    
    if not onnx_files:
        # Check if there's an onnx file in the models directory
        model_dir = Path(__file__).parent.absolute() / "duck_vla" / "models"
        if model_dir.exists():
            onnx_files = list(model_dir.glob("**/*.onnx"))
    
    if not onnx_files:
        return None
    
    # Return the first one found
    return str(onnx_files[0])

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Get workspace directory (repo root)
    workspace_dir = Path(__file__).parent.absolute()
    logger.debug(f"Workspace directory: {workspace_dir}")
    
    # Get the playground submodule path
    playground_path = workspace_dir / "submodules" / "open_duck_playground"
    logger.debug(f"Playground path: {playground_path}")
    
    # Check if playground exists
    if not playground_path.exists():
        logger.error(f"Playground directory not found at {playground_path}")
        return 1
    
    # Set working directory to the playground path
    os.chdir(playground_path)
    logger.debug(f"Changed working directory to: {playground_path}")
    
    # Get ONNX model path
    onnx_model_path = args.onnx_model
    if not onnx_model_path:
        onnx_model_path = find_onnx_model(playground_path)
        if not onnx_model_path:
            logger.error("No ONNX model found. Please provide one with --onnx-model")
            return 1
    
    logger.info(f"Using ONNX model: {onnx_model_path}")
    
    # Run the MuJoCo inference
    cmd_args = [
        "uv", 
        "run", 
        "playground/open_duck_mini_v2/mujoco_infer.py",
        "-o", 
        onnx_model_path
    ]
    
    # Log the command we're about to run
    logger.info(f"Running command: {' '.join(cmd_args)}")
    
    # Run the command
    try:
        process = subprocess.run(
            cmd_args,
            check=True
        )
        return process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, terminating...")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 