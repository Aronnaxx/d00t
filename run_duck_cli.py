#!/usr/bin/env python
"""
Duck VLA CLI Runner

This script is a convenience wrapper to run the Duck VLA system
with CLI control and Ollama vision system.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("duck_cli_runner")

def main():
    """Main entry point to run Duck VLA with CLI control."""
    # Get workspace directory (repo root)
    workspace_dir = Path(__file__).parent.absolute()
    
    # Get the playground submodule path
    playground_path = workspace_dir / "submodules" / "open_duck_playground"
    
    # Check if playground exists
    if not playground_path.exists():
        logger.error(f"Playground directory not found at {playground_path}")
        logger.error("Please run ./setup_duck_vla.sh to set up the environment")
        return 1
    
    # Check if ollama is running
    try:
        import ollama
        client = ollama.Client()
        client.list()  # Will fail if server is not running
        logger.info("Ollama server is running")
    except Exception as e:
        logger.error(f"Ollama server error: {e}")
        logger.error("Please make sure Ollama is installed and running (ollama serve)")
        return 1
    
    # Check if moondream model is available
    try:
        models = client.list()
        model_names = [m['name'].split(':')[0] for m in models.get('models', [])]
        if 'moondream' not in model_names:
            logger.warning("Moondream model not found in Ollama")
            logger.info("Pulling moondream model...")
            client.pull('moondream')
    except Exception as e:
        logger.error(f"Error checking/pulling moondream model: {e}")
        return 1
    
    # Setup the command to run Duck VLA with CLI, no audio, simulated mode
    cmd_args = [
        "python", "-m", "duck_vla.run_duck", 
        "--simulate",    # Use simulation mode
        "--no-audio",    # Disable audio input
    ]
    
    # Log the command we're about to run
    logger.info(f"Running command: {' '.join(cmd_args)}")
    
    # Set up environment with PYTHONPATH including the playground
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{playground_path}:{env.get('PYTHONPATH', '')}"
    
    # Run the command
    try:
        process = subprocess.run(
            cmd_args,
            env=env,
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