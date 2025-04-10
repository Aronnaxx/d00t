#!/usr/bin/env python
"""
Setup Ollama Models for Duck VLA

This script ensures that the required Ollama models are available
for use with the Duck VLA system.
"""

import logging
import sys
import argparse
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ollama_setup")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup Ollama Models for Duck VLA"
    )
    parser.add_argument(
        "--model", 
        type=str,
        default="moondream",
        help="Ollama model to use (default: moondream)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the setup script."""
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    model_name = args.model
    logger.info(f"Setting up Ollama model: {model_name}")
    
    # Check if Ollama is installed
    try:
        import ollama
        logger.info("Ollama package found")
    except ImportError:
        logger.error("Ollama package not found. Installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
            logger.info("Ollama package installed successfully")
            import ollama
        except Exception as e:
            logger.error(f"Failed to install Ollama package: {e}")
            return 1
    
    # Check if Ollama server is running
    try:
        client = ollama.Client()
        # Just make a simple call to check connectivity
        client.list()
        logger.info("Ollama server is running")
    except Exception as e:
        logger.error(f"Ollama server error: {e}")
        logger.error("Please make sure the Ollama server is running (ollama serve)")
        return 1
    
    # Pull the model
    try:
        logger.info(f"Pulling Ollama model '{model_name}'...")
        model_exists = False
        
        # Only try to pull if we don't already have it
        try:
            models = client.list()
            if "models" in models:
                # Check if model exists (by tagless name)
                model_base = model_name.split(':')[0]
                for model in models["models"]:
                    if "name" in model and model_base in model["name"]:
                        logger.info(f"Model '{model_name}' already exists")
                        model_exists = True
                        break
        except Exception:
            # If we can't check, assume it doesn't exist
            pass
        
        # Pull the model if it doesn't exist
        if not model_exists:
            try:
                client.pull(model_name)
                logger.info(f"Successfully pulled model '{model_name}'")
            except Exception as pull_error:
                logger.error(f"Error pulling model '{model_name}': {pull_error}")
                return 1
        
        logger.info(f"Ollama model '{model_name}' is ready to use")
        return 0
        
    except Exception as e:
        logger.error(f"Error setting up Ollama model: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 