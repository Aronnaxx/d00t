#!/usr/bin/env python
"""
Open Duck Playground Runner

This script runs the Open Duck Playground directly using UV.
It will launch the MuJoCo simulation environment showing the duck.

Usage:
    uv run run_playground.py [--debug]
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
logger = logging.getLogger("playground_runner")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Open Duck Playground Runner"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
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
    
    # Set working directory to the playground path
    os.chdir(playground_path)
    logger.debug(f"Changed working directory to: {playground_path}")
    
    # Run the playground directly
    cmd_args = ["uv", "run", "playground/open_duck_mini_v2/runner.py"]
    
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