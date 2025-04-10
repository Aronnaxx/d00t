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
from pathlib import Path

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("duck_vla")

# Import all necessary modules
try:
    from duck_vla.brain.decision_loop import DecisionLoop
    logger.info("Successfully imported decision loop module")
except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Duck VLA - Vision-Language-Action Control System"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    parser.add_argument(
        "--simulate", 
        action="store_true", 
        help="Run in simulation mode using OpenDuckPlayground"
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
    """Main entry point for the Duck VLA system."""
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info("Starting Duck VLA system...")
    logger.info(f"Running in {'simulation' if args.simulate else 'real'} mode")
    logger.info(f"Audio input/output {'disabled' if args.no_audio else 'enabled'}")
    logger.info(f"Camera input {'disabled' if args.no_camera else 'enabled'}")
    
    try:
        # Create and run the main decision loop
        decision_loop = DecisionLoop(
            simulate=args.simulate,
            audio_enabled=not args.no_audio,
            camera_enabled=not args.no_camera
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
