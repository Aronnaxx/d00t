#!/usr/bin/env python3
"""
Duck Robot User Configuration Example

This file demonstrates how to customize the Duck Robot configuration.
You can copy this file to ~/.duck_robot/config.py to apply your settings.

Alternatively, you can use environment variables to override settings:
export DUCK_OLLAMA_MODEL=gemma3:4b
export DUCK_USE_TTS=true
export DUCK_USE_VOICE=true
"""

# Ollama settings
ollama_model = "gemma3:4b"  # Use Gemma 3 4B model for vision

# TTS settings
tts_model = "gemma3:4b"  # Use Gemma 3 for text-to-speech
use_tts = True  # Enable text-to-speech
use_ollama_tts = True  # Use Ollama for TTS

# STT settings
use_voice = True  # Enable voice commands
use_ollama_stt = True  # Use Ollama for speech recognition

# Mode settings
autonomous_mode = True  # Enable autonomous mode by default

# Custom system prompt
custom_system_prompt = """You are a smart Duck Robot controlled by the Gemma 3 model.
You can see through your camera and understand voice commands.
You're exploring the environment autonomously but can also respond to user commands.
Be concise in your responses and describe what you see in brief terms."""

# Example for adding the custom prompt to the configuration
system_prompt = custom_system_prompt 