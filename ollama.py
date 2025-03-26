#!/usr/bin/env python3
"""
Ollama Manager Utility

This script provides utilities to manage Ollama instances for local VLM inference:
- Check if Ollama is installed
- Install Ollama if necessary
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
OLLAMA_DOWNLOAD_URLS = {
    "linux": "https://ollama.com/download/ollama-linux-amd64",
    "darwin": "https://ollama.com/download/ollama-darwin-amd64",
    "windows": "https://ollama.com/download/ollama-windows-amd64.zip"
}

DEFAULT_INSTALL_PATHS = {
    "linux": "/usr/local/bin/ollama",
    "darwin": "/usr/local/bin/ollama",
    "windows": os.path.expanduser("~/ollama/ollama.exe")
}

DEFAULT_PORT = 11434
VISION_COMPATIBLE_MODELS = ["llava", "bakllava", "llava-next"]

class OllamaManager:
    """Utility class to manage Ollama instances for local VLM inference."""
    
    def __init__(
        self,
        install_path: Optional[str] = None,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        timeout: int = 120,
        auto_launch: bool = True
    ):
        """Initialize the Ollama manager.
        
        Args:
            install_path: Path to the Ollama executable
            host: Host to connect to
            port: Port to connect to
            timeout: Timeout for Ollama startup in seconds
            auto_launch: Whether to automatically launch Ollama if not running
        """
        self.system = platform.system().lower()
        
        if self.system not in OLLAMA_DOWNLOAD_URLS:
            raise ValueError(f"Unsupported system: {self.system}")
        
        self.install_path = install_path or DEFAULT_INSTALL_PATHS.get(self.system)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.auto_launch = auto_launch
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/api"
        
        # Track if we started the process
        self.process = None
        self.started_by_us = False
        
        logger.info(f"Initialized Ollama manager for {self.system} with URL {self.base_url}")
        
        # Auto-check and launch if requested
        if auto_launch:
            if not self.is_running():
                if not self.is_installed():
                    logger.info("Ollama not installed. Attempting to install...")
                    self.install()
                logger.info("Starting Ollama service...")
                self.start()
    
    def is_installed(self) -> bool:
        """Check if Ollama is installed.
        
        Returns:
            True if Ollama is installed
        """
        if os.path.exists(self.install_path):
            logger.info(f"Ollama executable found at {self.install_path}")
            return True
        
        # Try checking if it's in PATH
        try:
            result = subprocess.run(
                ["which", "ollama"] if self.system != "windows" else ["where", "ollama"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.install_path = result.stdout.strip()
                logger.info(f"Ollama found in PATH at {self.install_path}")
                return True
        except Exception as e:
            logger.debug(f"Error checking for Ollama in PATH: {e}")
        
        logger.info("Ollama not installed")
        return False
    
    def install(self) -> bool:
        """Install Ollama.
        
        Returns:
            True if installation was successful
        """
        try:
            if self.system == "linux":
                logger.info("Installing Ollama on Linux...")
                # Create install directory if it doesn't exist
                install_dir = os.path.dirname(self.install_path)
                os.makedirs(install_dir, exist_ok=True)
                
                # Download and install
                download_url = OLLAMA_DOWNLOAD_URLS["linux"]
                subprocess.run(
                    ["curl", "-L", download_url, "-o", self.install_path],
                    check=True
                )
                subprocess.run(["chmod", "+x", self.install_path], check=True)
                logger.info(f"Ollama installed to {self.install_path}")
                return True
                
            elif self.system == "darwin":
                logger.info("Installing Ollama on macOS...")
                # Try with brew first
                try:
                    result = subprocess.run(
                        ["brew", "install", "ollama"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    logger.info("Ollama installed with Homebrew")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to install with Homebrew: {e}, falling back to manual install")
                
                # Manual install
                install_dir = os.path.dirname(self.install_path)
                os.makedirs(install_dir, exist_ok=True)
                
                download_url = OLLAMA_DOWNLOAD_URLS["darwin"]
                subprocess.run(
                    ["curl", "-L", download_url, "-o", self.install_path],
                    check=True
                )
                subprocess.run(["chmod", "+x", self.install_path], check=True)
                logger.info(f"Ollama installed to {self.install_path}")
                return True
                
            elif self.system == "windows":
                logger.info("Installing Ollama on Windows...")
                # Create install directory
                install_dir = os.path.dirname(self.install_path)
                os.makedirs(install_dir, exist_ok=True)
                
                # Download zip
                download_url = OLLAMA_DOWNLOAD_URLS["windows"]
                zip_path = os.path.join(install_dir, "ollama.zip")
                subprocess.run(
                    ["curl", "-L", download_url, "-o", zip_path],
                    check=True
                )
                
                # Extract zip
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(install_dir)
                
                logger.info(f"Ollama installed to {install_dir}")
                return True
            
            else:
                logger.error(f"Unsupported system for installation: {self.system}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing Ollama: {e}")
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
            logger.error("Ollama is not installed, can't start service")
            return False
        
        try:
            # Start Ollama as a background process
            logger.info(f"Starting Ollama service with {self.install_path}...")
            
            if self.system == "windows":
                # Windows needs special handling for background processes
                # Use creationflags to create a new process group
                from subprocess import CREATE_NEW_PROCESS_GROUP, DETACHED_PROCESS
                self.process = subprocess.Popen(
                    [self.install_path, "serve"],
                    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                # Unix systems
                self.process = subprocess.Popen(
                    [self.install_path, "serve"],
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
            logger.info(f"Available models: {', '.join([m['name'] for m in models])}")
            
            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
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
            if self.get_model_info(model_name):
                logger.info(f"Model {model_name} is already available")
                return True
            
            # Pull the model using Ollama CLI
            result = subprocess.run(
                [self.install_path, "pull", model_name],
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
            if not self.get_model_info(model_name):
                logger.info(f"Model {model_name} not found, pulling...")
                if not self.pull_model(model_name):
                    return {"error": f"Failed to pull model {model_name}"}
            
            # Prepare request
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
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


def start_ollama_server(model_name: str = "llava", host: str = "localhost", port: int = DEFAULT_PORT) -> OllamaManager:
    """Convenience function to start an Ollama server with the given model.
    
    Args:
        model_name: Name of the model to use
        host: Host to connect to
        port: Port to connect to
        
    Returns:
        OllamaManager instance
    """
    manager = OllamaManager(host=host, port=port, auto_launch=True)
    
    # Pull the model if it's not already available
    if not manager.get_model_info(model_name):
        manager.pull_model(model_name)
    
    return manager


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Ollama Manager Utility")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install Ollama")
    
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
    parser.add_argument("--path", type=str, help="Path to the Ollama executable")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create manager
    manager = OllamaManager(install_path=args.path, auto_launch=False)
    
    # Process commands
    if args.command == "install":
        if manager.install():
            print("Ollama installed successfully")
            return 0
        else:
            print("Failed to install Ollama")
            return 1
    
    elif args.command == "start":
        if manager.start():
            print("Ollama service started")
            return 0
        else:
            print("Failed to start Ollama service")
            return 1
    
    elif args.command == "stop":
        # Create a new manager with the current host/port
        manager = OllamaManager(install_path=args.path, auto_launch=False)
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