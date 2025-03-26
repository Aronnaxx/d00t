#!/usr/bin/env python3
"""
Start Robot Script

A single entry point to launch the Duck Robot with all features enabled:
- MuJoCo simulation
- Ollama VLM integration
- Voice commands and TTS
- Autonomous exploration

This script handles the complete setup and configuration of all components.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Import our configuration module
try:
    from config import config, get_system_prompt, is_model_available
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    DEFAULT_SYSTEM_PROMPT = """You are a vision-enabled AI controlling a Duck Robot. 
You can see through the robot's camera and respond to natural language commands.
You can control the robot's movement (forward, backward, left, right), 
rotation (turn left, right), and camera orientation (look up, down, left, right).
When given a command, respond with a brief explanation of what you're seeing
and what action you'll take."""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("duck_robot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("start_robot")

def ensure_onnx_model(model_path="model.onnx"):
    """Ensure the ONNX model exists at the specified path.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        str: The absolute path to the ONNX model
    """
    model_path = Path(model_path).absolute()
    
    if not model_path.exists():
        logger.warning(f"ONNX model not found at {model_path}")
        logger.info("Please specify the correct model path with --onnx-model")
        # Could add automatic download here in the future
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    
    logger.info(f"Using ONNX model at {model_path}")
    return str(model_path)

def ensure_ollama():
    """Ensure Ollama is running (but not install it)."""
    try:
        # Import the Ollama manager
        from ollama import OllamaManager
        
        logger.info("Checking Ollama status")
        manager = OllamaManager(auto_start=True)
        
        if not manager.is_installed():
            logger.error("Ollama is not installed. Please install Ollama first - see README.md for instructions.")
            return False
        
        if not manager.is_running():
            logger.info("Starting Ollama service")
            if not manager.start():
                logger.error("Failed to start Ollama service")
                return False
        
        # Check available models
        models = manager.list_models()
        if models:
            logger.info(f"Available Ollama models: {', '.join([m['name'] for m in models])}")
        else:
            logger.warning("No Ollama models found. You may need to pull a model like 'ollama pull llava'")
        
        return True
    except ImportError:
        logger.error("Ollama manager module not found. Please ensure it's installed.")
        return False
    except Exception as e:
        logger.error(f"Error setting up Ollama: {e}")
        return False

def check_voice_deps():
    """Check if voice dependencies are available.
    
    Returns:
        bool: True if voice dependencies are available, False otherwise
    """
    try:
        # Try to import required packages
        import sounddevice
        # If we get here, the import was successful
        logger.info("Voice dependencies are available")
        return True
    except (ImportError, OSError) as e:
        logger.warning(f"Voice dependencies not available: {e}")
        logger.warning("Voice and TTS features will be disabled")
        logger.info("To enable voice features, install the required dependencies:")
        logger.info("  sudo apt-get install portaudio19-dev")
        logger.info("  pip install sounddevice")
        return False

def start_robot(args):
    """Start the Duck Robot with the specified configuration."""
    try:
        # Ensure ONNX model exists
        onnx_path = ensure_onnx_model(args.onnx_model)
        
        # Check if voice dependencies are available
        voice_available = check_voice_deps()
        
        # Disable voice features if dependencies aren't available
        if args.voice and not voice_available:
            logger.warning("Voice command recognition requested but dependencies missing - disabling voice features")
            args.voice = False
        
        if args.tts and not voice_available:
            logger.warning("Text-to-speech requested but dependencies missing - disabling TTS features")
            args.tts = False
        
        # Check Ollama if using it
        if args.vlm_type == "ollama":
            if not ensure_ollama():
                logger.error("Failed to setup Ollama. Exiting.")
                return False
            
            if args.ollama_model:
                # Check if the model exists and pull if needed
                from ollama import OllamaManager
                manager = OllamaManager()
                
                if not manager.model_exists(args.ollama_model):
                    logger.info(f"Ollama model {args.ollama_model} not found. You need to pull it first.")
                    logger.info(f"Run: ollama pull {args.ollama_model}")
                    return False
        
        # Determine system prompt
        if CONFIG_AVAILABLE:
            # Get appropriate system prompt based on mode
            mode = "default"
            if args.autonomous:
                mode = "autonomous"
            elif args.interactive:
                mode = "interactive"
            
            system_prompt = get_system_prompt(mode, args.system_prompt)
        else:
            system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
        
        # Construct the command to launch the robot
        cmd = [sys.executable, "-m", "vlm_integration"]
        
        # Add arguments
        cmd.extend(["--vlm-type", args.vlm_type])
        cmd.extend(["--onnx-model", onnx_path])
        
        if args.vlm_type == "ollama":
            cmd.extend(["--ollama-model", args.ollama_model])
            cmd.extend(["--ollama-url", args.ollama_url])
            cmd.extend(["--system-prompt", system_prompt])
            
            if args.no_auto_launch:
                cmd.append("--no-auto-launch")
        else:
            if args.vlm_api:
                cmd.extend(["--vlm-api", args.vlm_api])
            if args.vlm_key:
                cmd.extend(["--vlm-key", args.vlm_key])
        
        # Mode selection
        if args.autonomous:
            cmd.append("--autonomous")
        
        if args.interactive:
            cmd.append("--interactive")
        
        # Voice options
        if args.voice:
            cmd.append("--voice")
        
        if args.tts:
            cmd.append("--tts")
            cmd.extend(["--tts-model", args.tts_model])
        
        if args.ollama_stt:
            cmd.append("--ollama-stt")
        
        if args.ollama_tts:
            cmd.append("--ollama-tts")
        
        # Other options
        if args.debug:
            cmd.append("--debug")
        
        # Print the command for reference
        cmd_str = " ".join(cmd)
        logger.info(f"Launching Duck Robot with command: {cmd_str}")
        
        # Execute the command
        process = subprocess.run(cmd)
        return process.returncode == 0
    
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Error starting Duck Robot: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Start Duck Robot with all features")
    
    # Use config defaults if available
    default_ollama_model = config.get("ollama_model", "llava") if CONFIG_AVAILABLE else "llava"
    default_tts_model = config.get("tts_model", "gemma3:4b") if CONFIG_AVAILABLE else "gemma3:4b"
    default_use_voice = config.get("use_voice", False) if CONFIG_AVAILABLE else False
    default_use_tts = config.get("use_tts", False) if CONFIG_AVAILABLE else False
    default_use_ollama_stt = config.get("use_ollama_stt", False) if CONFIG_AVAILABLE else False
    default_use_ollama_tts = config.get("use_ollama_tts", True) if CONFIG_AVAILABLE else True
    default_autonomous = config.get("autonomous_mode", False) if CONFIG_AVAILABLE else False
    default_interactive = config.get("interactive_mode", False) if CONFIG_AVAILABLE else False
    default_onnx_model = config.get("onnx_model_path", "model.onnx") if CONFIG_AVAILABLE else "model.onnx"
    default_duck_api = config.get("duck_api_url", "http://localhost:5000") if CONFIG_AVAILABLE else "http://localhost:5000"
    
    # Basic configuration
    parser.add_argument("--onnx-model", type=str, default=default_onnx_model, 
                        help="Path to the ONNX model file")
    parser.add_argument("--duck-api", type=str, default=default_duck_api, 
                        help="URL for the Duck Robot API")
    
    # VLM configuration
    parser.add_argument("--vlm-type", type=str, choices=["external", "ollama"], default="ollama",
                        help="Type of VLM to use")
    
    # Ollama options
    parser.add_argument("--ollama-model", type=str, default=default_ollama_model,
                        help="Model name for Ollama (e.g., llava, bakllava, gemma3:4b)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="URL for the Ollama API")
    parser.add_argument("--no-auto-launch", action="store_true",
                        help="Don't automatically launch Ollama")
    parser.add_argument("--system-prompt", type=str,
                        help="System prompt for the VLM to provide context")
    
    # External VLM options
    parser.add_argument("--vlm-api", type=str, 
                        help="URL for the external VLM API")
    parser.add_argument("--vlm-key", type=str,
                        help="API key for the external VLM API")
    
    # Mode options
    mode_group = parser.add_argument_group("Mode Options")
    mode_group.add_argument("--autonomous", action="store_true", default=default_autonomous,
                           help="Run in autonomous exploration mode")
    mode_group.add_argument("--interactive", action="store_true", default=default_interactive,
                           help="Run in interactive mode")
    
    # Voice options
    voice_group = parser.add_argument_group("Voice Options")
    voice_group.add_argument("--voice", action="store_true", default=default_use_voice,
                            help="Enable voice command recognition")
    voice_group.add_argument("--tts", action="store_true", default=default_use_tts,
                            help="Enable text-to-speech")
    voice_group.add_argument("--tts-model", type=str, default=default_tts_model,
                            help="Model to use for TTS")
    voice_group.add_argument("--ollama-stt", action="store_true", default=default_use_ollama_stt,
                            help="Use Ollama for speech recognition")
    voice_group.add_argument("--ollama-tts", action="store_true", default=default_use_ollama_tts,
                            help="Use Ollama for text-to-speech")
    
    # Other options
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Set the log level for all handlers
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.vlm_type == "external" and not args.vlm_api and not args.interactive and not args.autonomous:
        logger.error("External VLM type requires --vlm-api parameter unless in interactive or autonomous mode")
        return 1
    
    # If no mode is specified, default to autonomous with voice if those flags are set
    if not args.autonomous and not args.interactive:
        if args.voice:
            logger.info("No mode specified but voice enabled. Defaulting to autonomous mode.")
            args.autonomous = True
        else:
            logger.info("No mode specified. Defaulting to autonomous mode.")
            args.autonomous = True
    
    # Start the robot
    success = start_robot(args)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 