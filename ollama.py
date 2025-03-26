#!/usr/bin/env python3
"""
Ollama Manager Utility

This script provides utilities to manage Ollama instances for local VLM inference:
- Check if Ollama is installed
- Start the Ollama service
- Pull models from the Ollama library
- Query models with text and images
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import requests
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ollama_manager.log")
    ]
)
logger = logging.getLogger(__name__)

# Global constants
DEFAULT_PORT = 11434
VISION_COMPATIBLE_MODELS = ["llava", "bakllava", "llava-next"]

# Default system prompts
DEFAULT_SYSTEM_PROMPT = """You are a vision-enabled AI controlling a Duck Robot. 
You can see through the robot's camera and respond to natural language commands.
You can control the robot's movement (forward, backward, left, right), 
rotation (turn left, right), and camera orientation (look up, down, left, right).
When given a command, respond with a brief explanation of what you're seeing
and what action you'll take."""

class OllamaManager:
    """Utility class to manage Ollama instances for local VLM inference."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        timeout: int = 60,
        auto_start: bool = True,
        system_prompt: Optional[str] = None
    ):
        """Initialize the Ollama manager.
        
        Args:
            host: Host to connect to
            port: Port to connect to
            timeout: Timeout for Ollama startup in seconds
            auto_start: Whether to automatically start Ollama if it's installed but not running
            system_prompt: Default system prompt to use with models
        """
        self.system = platform.system().lower()
        self.host = host
        self.port = port
        self.timeout = timeout
        self.auto_start = auto_start
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/api"
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        
        # Track if we started the process
        self.process = None
        self.started_by_us = False
        
        logger.info(f"Initialized Ollama manager for {self.system} with URL {self.base_url}")
        
        # Auto-check and start if requested
        if auto_start:
            if not self.is_running() and self.is_installed():
                logger.info("Ollama installed but not running. Starting...")
                self.start()
    
    def is_installed(self) -> bool:
        """Check if Ollama is installed.
        
        Returns:
            True if Ollama is installed
        """
        # Try checking if it's in PATH
        try:
            which_cmd = "where" if self.system == "windows" else "which"
            result = subprocess.run(
                [which_cmd, "ollama"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                install_path = result.stdout.strip()
                logger.info(f"Ollama found in PATH at {install_path}")
                return True
            
            # Check common install locations
            common_paths = {
                "linux": ["/usr/local/bin/ollama", "/usr/bin/ollama"],
                "darwin": ["/usr/local/bin/ollama", "/opt/homebrew/bin/ollama"],
                "windows": [
                    os.path.expanduser("~/ollama/ollama.exe"),
                    "C:\\Program Files\\Ollama\\ollama.exe"
                ]
            }
            
            for path in common_paths.get(self.system, []):
                if os.path.exists(path):
                    logger.info(f"Ollama executable found at {path}")
                    return True
            
            logger.info("Ollama not installed")
            return False
        except Exception as e:
            logger.debug(f"Error checking for Ollama: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if Ollama is running.
        
        Returns:
            True if Ollama is running
        """
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=2)
            if response.status_code == 200:
                logger.info("Ollama service is running")
                return True
            else:
                logger.info(f"Ollama service returned status code {response.status_code}")
                return False
        except Exception as e:
            logger.info(f"Ollama service is not running: {e}")
            return False
    
    def start(self) -> bool:
        """Start the Ollama service.
        
        Returns:
            True if the service was started successfully
        """
        if self.is_running():
            logger.info("Ollama service is already running")
            return True
        
        if not self.is_installed():
            logger.error("Ollama is not installed. Please install Ollama first.")
            return False
        
        try:
            # Start Ollama as a background process
            logger.info("Starting Ollama service...")
            
            if self.system == "windows":
                # Windows needs special handling for background processes
                # Use creationflags to create a new process group
                from subprocess import CREATE_NEW_PROCESS_GROUP, DETACHED_PROCESS
                self.process = subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                # Unix systems
                self.process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Flag that we started the process
            self.started_by_us = True
            
            # Wait for the service to become available
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                if self.is_running():
                    logger.info("Ollama service started successfully")
                    return True
                time.sleep(1)
            
            logger.error(f"Timeout waiting for Ollama service to start after {self.timeout} seconds")
            return False
            
        except Exception as e:
            logger.error(f"Error starting Ollama service: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the Ollama service if we started it.
        
        Returns:
            True if the service was stopped successfully
        """
        if not self.started_by_us or self.process is None:
            logger.info("Not stopping Ollama service as it wasn't started by us")
            return True
        
        try:
            # Terminate the process
            logger.info("Stopping Ollama service...")
            
            if self.system == "windows":
                # Windows needs taskkill for reliable termination
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)])
            else:
                # Unix systems
                self.process.terminate()
                self.process.wait(timeout=10)
            
            logger.info("Ollama service stopped")
            self.process = None
            self.started_by_us = False
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Ollama service: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models.
        
        Returns:
            List of model info dictionaries
        """
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            response.raise_for_status()
            result = response.json()
            models = result.get("models", [])
            
            # Log models in user-friendly format
            if models:
                logger.info(f"Available models: {', '.join([m['name'] for m in models])}")
            else:
                logger.info("No models found")
            
            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if the model exists
        """
        try:
            models = self.list_models()
            return any(model["name"] == model_name for model in models)
        except Exception as e:
            logger.error(f"Error checking if model exists: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get info for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model info dictionary if found, None otherwise
        """
        try:
            models = self.list_models()
            for model in models:
                if model["name"] == model_name:
                    return model
            
            logger.warning(f"Model {model_name} not found")
            return None
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from the Ollama library.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if the model was pulled successfully
        """
        try:
            logger.info(f"Pulling model {model_name}...")
            
            # Check if model is already available
            if self.model_exists(model_name):
                logger.info(f"Model {model_name} is already available")
                return True
            
            # Pull the model using Ollama CLI
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Model {model_name} pulled successfully")
                return True
            else:
                logger.error(f"Error pulling model {model_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def query_model(self, 
                    model_name: str, 
                    prompt: str, 
                    images: Optional[List[str]] = None, 
                    system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Query a model with a prompt and optional images.
        
        Args:
            model_name: Name of the model to query
            prompt: Text prompt to send to the model
            images: List of base64-encoded images to include
            system_prompt: Optional system prompt to provide context
            
        Returns:
            Response from the model
        """
        try:
            # Check if model is available, pull if not
            if not self.model_exists(model_name):
                logger.info(f"Model {model_name} not found, pulling...")
                if not self.pull_model(model_name):
                    return {"error": f"Failed to pull model {model_name}"}
            
            # Prepare request
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # Use provided system prompt or default
            if system_prompt or self.system_prompt:
                payload["system"] = system_prompt or self.system_prompt
                
            if images:
                # Check if model supports vision
                if not any(vm in model_name for vm in VISION_COMPATIBLE_MODELS):
                    logger.warning(f"Model {model_name} may not support vision. Compatible models: {VISION_COMPATIBLE_MODELS}")
                
                payload["images"] = images
            
            logger.debug(f"Sending query to model {model_name}")
            
            # Make request
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Received response from model {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error querying model {model_name}: {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            if hasattr(self, 'started_by_us') and self.started_by_us:
                self.stop()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def start_ollama_server(model_name: str = "llava", host: str = "localhost", port: int = DEFAULT_PORT, system_prompt: Optional[str] = None) -> OllamaManager:
    """Convenience function to start an Ollama server with the given model.
    
    Args:
        model_name: Name of the model to use
        host: Host to connect to
        port: Port to connect to
        system_prompt: Optional system prompt to use with the model
        
    Returns:
        OllamaManager instance
    """
    manager = OllamaManager(
        host=host, 
        port=port, 
        auto_start=True,
        system_prompt=system_prompt
    )
    
    # Pull the model if it's not already available
    if not manager.model_exists(model_name):
        manager.pull_model(model_name)
    
    return manager


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Ollama Manager Utility")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start Ollama service")
    start_parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    start_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop Ollama service")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    
    # Pull model command
    pull_parser = subparsers.add_parser("pull", help="Pull a model from the Ollama library")
    pull_parser.add_argument("model", type=str, help="Name of the model to pull")
    
    # Query model command
    query_parser = subparsers.add_parser("query", help="Query a model with a prompt")
    query_parser.add_argument("model", type=str, help="Name of the model to query")
    query_parser.add_argument("prompt", type=str, help="Text prompt to send to the model")
    query_parser.add_argument("--image", type=str, help="Path to an image to include")
    query_parser.add_argument("--system", type=str, help="System prompt to provide context")
    
    # Global arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create manager
    manager = OllamaManager(
        host=args.host if hasattr(args, 'host') else "localhost",
        port=args.port if hasattr(args, 'port') else DEFAULT_PORT,
        auto_start=False
    )
    
    # Process commands
    if args.command == "start":
        if manager.start():
            print("Ollama service started")
            return 0
        else:
            print("Failed to start Ollama service")
            return 1
    
    elif args.command == "stop":
        # Create a new manager with the current host/port
        manager.started_by_us = True  # Trick it into stopping
        if manager.stop():
            print("Ollama service stopped")
            return 0
        else:
            print("Failed to stop Ollama service")
            return 1
    
    elif args.command == "list":
        models = manager.list_models()
        if models:
            print("Available models:")
            for model in models:
                size_mb = model.get("size", 0) / (1024 * 1024)
                print(f"- {model['name']}: {size_mb:.1f} MB")
            return 0
        else:
            print("No models found or failed to list models")
            return 1
    
    elif args.command == "pull":
        if manager.pull_model(args.model):
            print(f"Model {args.model} pulled successfully")
            return 0
        else:
            print(f"Failed to pull model {args.model}")
            return 1
    
    elif args.command == "query":
        # Prepare image if provided
        images = None
        if args.image:
            try:
                with open(args.image, "rb") as f:
                    img_data = f.read()
                import base64
                images = [base64.b64encode(img_data).decode()]
            except Exception as e:
                print(f"Error reading image: {e}")
                return 1
        
        # Query the model
        result = manager.query_model(
            model_name=args.model,
            prompt=args.prompt,
            images=images,
            system_prompt=args.system
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
        else:
            print(f"Response: {result.get('response', '')}")
            return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    # Add missing import for CLI functionality
    import base64
    sys.exit(main()) 