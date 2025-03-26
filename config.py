#!/usr/bin/env python3
"""
Duck Robot Configuration

This file contains the configuration settings for the Duck Robot system,
including model selections, system prompts, and other parameters.
"""

import logging
import os
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("duck_robot")

# Ollama VLM Models
# These are the vision-capable models that can be used with Ollama
VLM_MODELS = {
    "llava": {
        "description": "Basic vision-language model",
        "capabilities": ["vision", "text"],
        "vision_quality": "good",
        "size": "medium",
    },
    "bakllava": {
        "description": "Enhanced vision capabilities based on mistral",
        "capabilities": ["vision", "text"],
        "vision_quality": "better",
        "size": "large",
    },
    "llava-next": {
        "description": "Latest version with improved performance",
        "capabilities": ["vision", "text"],
        "vision_quality": "best",
        "size": "large",
    },
    "gemma3:4b": {
        "description": "Smaller model suitable for TTS and STT",
        "capabilities": ["vision", "text", "tts", "stt"],
        "vision_quality": "decent",
        "size": "small",
    },
    "gemma2:9b": {
        "description": "Mid-sized model with good vision capabilities",
        "capabilities": ["vision", "text"],
        "vision_quality": "good",
        "size": "medium",
    },
}

# Text-to-Speech Models
TTS_MODELS = {
    "gemma3:4b": {
        "description": "Lightweight model for TTS",
        "quality": "decent",
        "size": "small",
    },
    "mistral:7b": {
        "description": "Mid-sized model for better TTS",
        "quality": "good",
        "size": "medium",
    },
}

# Speech-to-Text Models
STT_MODELS = {
    "whisper:tiny": {
        "description": "Very small whisper model",
        "accuracy": "basic",
        "size": "tiny",
    },
    "whisper:base": {
        "description": "Base whisper model",
        "accuracy": "decent",
        "size": "small",
    },
    "whisper:small": {
        "description": "Small whisper model, good balance",
        "accuracy": "good",
        "size": "medium",
    },
    "whisper:medium": {
        "description": "Medium whisper model, better accuracy",
        "accuracy": "very good",
        "size": "large",
    },
    "whisper:large": {
        "description": "Large whisper model, best accuracy",
        "accuracy": "excellent",
        "size": "very large",
    },
    "gemma3:4b": {
        "description": "Ollama-based speech recognition",
        "accuracy": "decent",
        "size": "small",
    },
}

# Default configurations
DEFAULT_CONFIG = {
    # Ollama settings
    "ollama_url": "http://localhost:11434",
    "ollama_model": "gemma3:4b",  # Setting gemma3 as default as requested
    
    # TTS settings
    "tts_model": "gemma3:4b",
    "use_tts": False,
    "use_ollama_tts": True,
    
    # STT settings
    "use_voice": False,
    "use_ollama_stt": False,
    "whisper_model": "base",
    
    # Mode settings
    "autonomous_mode": False,
    "interactive_mode": False,
    
    # Robot API settings
    "duck_api_url": "http://localhost:5000",
    
    # ONNX model
    "onnx_model_path": "model.onnx",
    
    # Exploration settings
    "exploration_interval": 3.0,  # seconds between autonomous decisions
    
    # Debug settings
    "debug": False,
}

# System prompts for different modes
SYSTEM_PROMPTS = {
    "default": """You are a vision-enabled AI controlling a Duck Robot. 
You can see through the robot's camera and respond to natural language commands.
You can control the robot's movement (forward, backward, left, right), 
rotation (turn left, right), and camera orientation (look up, down, left, right).
When given a command, respond with a brief explanation of what you're seeing
and what action you'll take.""",

    "autonomous": """You are an autonomous vision-enabled Duck Robot exploring your environment.
Your goal is to navigate and explore interesting areas while avoiding obstacles.
You should describe what you see and make decisions about where to move next.
You can control your movement (forward, backward, left, right, turn) and
camera orientation (look up, down, left, right).""",

    "interactive": """You are a vision-enabled AI assistant controlling a Duck Robot.
You're helping a human user by responding to their commands.
You can see through the robot's camera and interpret the environment.
You can control the robot's movement and camera orientation based on user instructions.
Be concise and helpful in your responses.""",
}

def load_user_config() -> Dict[str, Any]:
    """Load user configuration from environment variables or files."""
    config = DEFAULT_CONFIG.copy()
    
    # Environment variable overrides
    env_vars = {
        "DUCK_OLLAMA_MODEL": "ollama_model",
        "DUCK_TTS_MODEL": "tts_model",
        "DUCK_USE_TTS": "use_tts",
        "DUCK_USE_VOICE": "use_voice",
        "DUCK_AUTONOMOUS": "autonomous_mode",
        "DUCK_INTERACTIVE": "interactive_mode",
        "DUCK_API_URL": "duck_api_url",
        "DUCK_ONNX_MODEL": "onnx_model_path",
        "DUCK_DEBUG": "debug",
    }
    
    for env_var, config_key in env_vars.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            # Convert string to appropriate type
            if value.lower() in ("true", "yes", "1"):
                config[config_key] = True
            elif value.lower() in ("false", "no", "0"):
                config[config_key] = False
            elif value.isdigit():
                config[config_key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                config[config_key] = float(value)
            else:
                config[config_key] = value
    
    # User config file override
    user_config_path = os.path.expanduser("~/.duck_robot/config.py")
    if os.path.exists(user_config_path):
        try:
            # Use a restricted exec to load the user config
            user_config = {}
            with open(user_config_path, "r") as f:
                exec(f.read(), {}, user_config)
            
            # Update the config with user settings
            for key, value in user_config.items():
                if key in config:
                    config[key] = value
            
            logger.info(f"Loaded user configuration from {user_config_path}")
        except Exception as e:
            logger.error(f"Error loading user configuration: {e}")
    
    return config

def get_system_prompt(mode: str = "default", custom_prompt: Optional[str] = None) -> str:
    """Get the appropriate system prompt based on mode."""
    if custom_prompt:
        return custom_prompt
    
    if mode in SYSTEM_PROMPTS:
        return SYSTEM_PROMPTS[mode]
    
    return SYSTEM_PROMPTS["default"]

def is_model_available(model_name: str, model_type: str = "vlm") -> bool:
    """Check if a model is available for the specified type."""
    if model_type == "vlm":
        return model_name in VLM_MODELS
    elif model_type == "tts":
        return model_name in TTS_MODELS
    elif model_type == "stt":
        return model_name in STT_MODELS
    
    return False

def get_model_info(model_name: str, model_type: str = "vlm") -> Dict[str, Any]:
    """Get information about a specific model."""
    if model_type == "vlm" and model_name in VLM_MODELS:
        return VLM_MODELS[model_name]
    elif model_type == "tts" and model_name in TTS_MODELS:
        return TTS_MODELS[model_name]
    elif model_type == "stt" and model_name in STT_MODELS:
        return STT_MODELS[model_name]
    
    return {"description": "Unknown model", "capabilities": [], "size": "unknown"}

# Load configuration at import time
config = load_user_config()

# Export the config as a module variable
__all__ = ["config", "get_system_prompt", "is_model_available", "get_model_info", 
           "VLM_MODELS", "TTS_MODELS", "STT_MODELS", "SYSTEM_PROMPTS"] 