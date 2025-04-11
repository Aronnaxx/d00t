#!/usr/bin/env python
"""
Duck VLA (Vision-Language-Action) Control System - Main Entry Point

This script serves as the entry point for the Duck VLA system, orchestrating
the interaction between vision, language understanding, and action modules.
"""

import argparse
import logging
import sys
import time
import os
from pathlib import Path

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("duck_vla")

# Import all necessary modules
try:
    from duck_vla.core_ai.decision_loop import DecisionLoop
    from duck_vla.actions.mujoco_connector import MujocoConnector

    logger.info("Successfully imported required modules")
except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Duck VLA - Vision-Language-Action Control System")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--simulate", action="store_true", help="Run in simulation mode using OpenDuckPlayground"
    )
    parser.add_argument("--no-audio", action="store_true", help="Disable audio input/output")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera input")
    parser.add_argument(
        "--no-cli", action="store_true", help="Disable CLI for direct command input"
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        help="Vision model to use (default: from env var DUCK_VISION_MODEL or 'gemma3')",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        help="Path to ONNX model for simulation (default: from env var DUCK_ONNX_MODEL or system default)",
    )

    # Add LLM provider options
    llm_group = parser.add_argument_group("LLM Provider Options")
    llm_group.add_argument(
        "--llm-provider",
        type=str,
        choices=["ollama", "openai", "anthropic"],
        default="ollama",
        help="LLM provider to use for natural language processing (default: ollama)",
    )
    llm_group.add_argument(
        "--llm-model", type=str, help="Specific model to use with the selected LLM provider"
    )
    llm_group.add_argument(
        "--system-prompt", type=str, help="Custom system prompt to use for the LLM"
    )

    # Add API authentication options
    api_group = parser.add_argument_group("API Authentication")
    api_group.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key (if not set in OPENAI_API_KEY environment variable)",
    )
    api_group.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key (if not set in ANTHROPIC_API_KEY environment variable)",
    )
    api_group.add_argument(
        "--ollama-host",
        type=str,
        help="Ollama host URL (if not set in OLLAMA_HOST environment variable, defaults to http://localhost:11434)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the Duck VLA system."""
    args = parse_arguments()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Set API keys from command line if provided
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
        logger.debug("Set OpenAI API key from command line")

    if args.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api_key
        logger.debug("Set Anthropic API key from command line")

    if args.ollama_host:
        os.environ["OLLAMA_HOST"] = args.ollama_host
        logger.debug(f"Set Ollama host to {args.ollama_host}")

    # Get vision and ONNX model settings
    vision_model = args.vision_model
    onnx_model_path = args.onnx_model

    # If provided on command line, override environment variables
    if vision_model:
        os.environ["DUCK_VISION_MODEL"] = vision_model
    if onnx_model_path:
        os.environ["DUCK_ONNX_MODEL"] = onnx_model_path

    # Get LLM model from environment variable if not provided
    llm_model = args.llm_model or os.environ.get("DUCK_LLM_MODEL")
    if not llm_model and args.llm_provider == "ollama":
        # Default to gemma3:latest for Ollama
        llm_model = "gemma3:latest"

    logger.info("Starting Duck VLA system...")
    logger.info(f"Running in {'simulation' if args.simulate else 'real'} mode")
    logger.info(f"Audio input/output {'disabled' if args.no_audio else 'enabled'}")
    logger.info(f"Camera input {'disabled' if args.no_camera else 'enabled'}")
    logger.info(f"CLI controller {'disabled' if args.no_cli else 'enabled'}")
    logger.info(f"LLM provider: {args.llm_provider}")
    if llm_model:
        logger.info(f"LLM model: {llm_model}")
    logger.info(f"Vision model: {vision_model or os.environ.get('DUCK_VISION_MODEL', 'gemma3')}")
    if onnx_model_path or os.environ.get("DUCK_ONNX_MODEL"):
        logger.info(f"ONNX model: {onnx_model_path or os.environ.get('DUCK_ONNX_MODEL')}")

    try:
        # Create and run the main decision loop
        decision_loop = DecisionLoop(
            simulate=args.simulate,
            audio_enabled=not args.no_audio,
            camera_enabled=not args.no_camera,
            cli_enabled=not args.no_cli,
            vision_model=vision_model,
            onnx_model_path=onnx_model_path,
            llm_provider=args.llm_provider,
            llm_model=llm_model,
            system_prompt=args.system_prompt,
        )

        logger.info("Decision loop initialized, starting main loop")
        decision_loop.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1

    logger.info("Duck VLA system shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
