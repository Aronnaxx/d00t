#!/usr/bin/env python
"""
Duck VLA Simulation Runner

This script runs the Duck VLA simulation by:
1. Setting up the Python path to include the Open Duck Playground 
2. Running the simulation with UV (the fast package installer and runner)

Usage:
    uv run run_duck_sim.py [--debug] [--no-audio] [--no-camera]
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("duck_sim_runner")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Duck VLA Simulation Runner"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    parser.add_argument(
        "--no-audio", 
        action="store_true", 
        help="Disable audio input/output"
    )
    parser.add_argument(
        "--no-camera", 
        action="store_true", 
        help="Disable camera input"
    )
    
    return parser.parse_args()

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
    
    # Setup the command to run
    cmd_args = ["python", "-m", "duck_vla.run_duck", "--simulate"]
    
    # Add additional arguments
    if args.debug:
        cmd_args.append("--debug")
    if args.no_audio:
        cmd_args.append("--no-audio")
    if args.no_camera:
        cmd_args.append("--no-camera")
    
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