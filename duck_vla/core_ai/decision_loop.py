"""
Brain module: Decision Loop - Main controller for the Duck VLA system

This module contains the core decision-making loop that integrates
vision, language, and action components to create autonomous behavior.
"""

import logging
import time
import os
import traceback
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class DecisionLoop:
    """
    Main control loop for Duck VLA system.
    
    This class coordinates the interaction between:
    - Camera input (vision)
    - Speech-to-text and intent parsing (language)
    - Movement and emote actions (action)
    - CLI for direct command input (optional)
    """
    
    def __init__(
        self,
        simulate: bool = False,
        audio_enabled: bool = True,
        camera_enabled: bool = True,
        cli_enabled: bool = True,
        local_model: bool = True,
        vision_model: Optional[str] = None,
        onnx_model_path: Optional[str] = None,
    ):
        """
        Initialize the decision loop.
        
        Args:
            simulate: Whether to run in simulation mode using OpenDuckPlayground
            audio_enabled: Whether to enable audio input/output
            camera_enabled: Whether to enable camera input
            cli_enabled: Whether to enable CLI for direct command input
            vision_model: Vision model name to use (default from env or 'moondream')
            onnx_model_path: Path to ONNX model for simulation (default from env or None)
        """
        self.simulate = simulate
        self.audio_enabled = audio_enabled
        self.camera_enabled = camera_enabled
        self.cli_enabled = cli_enabled
        
        # if local_model is True, we will use the local model
        # if local_model is False, we will use the external model

        self.local_model = local_model
        # Get vision model from environment or parameter
        self.vision_model = vision_model or os.environ.get("DUCK_VISION_MODEL", "moondream")
        
        # Get ONNX model path from environment or parameter
        self.onnx_model_path = onnx_model_path or os.environ.get("DUCK_ONNX_MODEL", None)
        
        self.running = False
        self.current_state = {"status": "initializing"}
        
        logger.debug("Setting up component imports...")
        logger.info(f"Using vision model: {self.vision_model}")
        if self.onnx_model_path:
            logger.info(f"Using ONNX model: {self.onnx_model_path}")
        
        # Initialize components with graceful fallbacks
        self.camera = None
        self.vision = None
        self.stt = None
        self.intent_parser = None
        self.audio_system = None
        self.cli = None
        self.motion = None
        self.emotes = None
        
        # Ensure vision system is always initialized
        self._init_vision_components(force_initialize=True)

        # Initialize audio components if audio is enabled    
        if audio_enabled:
            self._init_audio_components()
        
        # Initialize CLI if enabled
        if cli_enabled:
            self._init_cli_controller()
            
        # Initialize action system
        self._init_action_system()
            
        logger.info("Decision loop initialization complete")
    
    def _init_vision_components(self, force_initialize=False):
        """Initialize vision and camera components with graceful error handling."""
        try:
            from duck_vla.camera.arducam_capture import ArduCamCapture
            # TODO -- import either the ollama or external api model interface from our central_model

            logger.debug("Initializing camera and vision modules")

            # Try to initialize camera but continue if hardware not available
            if self.camera_enabled and not force_initialize:
                try:
                    self.camera = ArduCamCapture()
                    logger.info("Camera initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize camera hardware: {e}")
                    logger.info("Continuing without camera hardware, but vision processing will still be available")
                    self.camera = None

            # Initialize vision model regardless of camera hardware status
            try:
                self.vision = MoondreamVision(
                    backend="ollama",
                    model_id=self.vision_model
                )
                logger.info(f"Vision system initialized with model: {self.vision_model}")
            except Exception as e:
                logger.error(f"Failed to initialize vision system: {e}")
                logger.debug(traceback.format_exc())
                self.vision = None

        except ImportError as e:
            logger.warning(f"Failed to import vision modules: {e}")
            self.camera = None
            self.vision = None
        except Exception as e:
            logger.error(f"Unexpected error initializing vision: {e}")
            logger.debug(traceback.format_exc())
            self.camera = None
            self.vision = None
    
    def _init_audio_components(self):
        """Initialize audio components with graceful error handling."""
        try:
            # First try importing the modules
            try:
                from duck_vla.sounds.stt import SpeechToText
                from duck_vla.brain.intent_parser import IntentParser
                from duck_vla.sounds.audio import AudioSystem
                
                logger.debug("Successfully imported audio modules")
            except ImportError as e:
                logger.warning(f"Failed to import audio modules: {e}")
                return
            
            # Then try initializing each component separately
            try:
                self.intent_parser = IntentParser()
                logger.info("Intent parser initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize intent parser: {e}")
                self.intent_parser = None
            
            try:
                self.stt = SpeechToText()
                logger.info("Speech-to-text initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize speech-to-text: {e}") 
                logger.warning("Continuing without audio input capability")
                self.stt = None
            
            try:
                self.audio_system = AudioSystem()
                logger.info("Audio system initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize audio system: {e}")
                logger.warning("Continuing without audio output capability")
                self.audio_system = None
                
        except Exception as e:
            logger.error(f"Unexpected error initializing audio components: {e}")
            logger.debug(traceback.format_exc())
            self.stt = None
            self.intent_parser = None
            self.audio_system = None
    
    def _init_cli_controller(self):
        """Initialize CLI controller with error handling."""
        try:
            from duck_vla.utils.cli_controller import CLIController
            
            logger.debug("Initializing CLI controller")
            self.cli = CLIController()
            logger.info("CLI controller initialized successfully")
        except ImportError as e:
            logger.warning(f"Failed to import CLI controller: {e}")
            self.cli = None
        except Exception as e:
            logger.error(f"Unexpected error initializing CLI controller: {e}")
            logger.debug(traceback.format_exc())
            self.cli = None
    
    def _init_action_system(self):
        """Initialize action system components with error handling."""
        try:
            if self.simulate:
                logger.info("Using simulation-based movement")
                # Import simulation-specific modules here
                try:
                    from duck_vla.action.motion_controller import SimulatedMotionController
                    
                    # Create with ONNX model path if specified
                    self.motion = SimulatedMotionController(onnx_model_path=self.onnx_model_path)
                    logger.info("Simulated motion controller initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize simulated motion controller: {e}")
                    logger.debug(traceback.format_exc())
                    self.motion = None
            else:
                logger.info("Using real hardware movement")
                try:
                    from duck_vla.action.motion_controller import MotionController
                    self.motion = MotionController()
                    logger.info("Hardware motion controller initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize hardware motion controller: {e}")
                    logger.debug(traceback.format_exc())
                    self.motion = None
            
            try:
                from duck_vla.action.emotes import EmoteController
                self.emotes = EmoteController(audio_enabled=self.audio_enabled)
                logger.info("Emote controller initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize emote controller: {e}")
                self.emotes = None
            
        except ImportError as e:
            logger.error(f"Failed to import action modules: {e}")
            self.motion = None
            self.emotes = None
            
        # Verify we have at least some action capability
        if self.motion is None:
            logger.warning("Motion controller not available - limited functionality")
        
    def run(self) -> None:
        """Run the main decision loop."""
        logger.info("Starting decision loop")
        self.running = True
        
        # Start CLI if enabled
        if self.cli_enabled and self.cli:
            logger.info("Starting CLI controller")
            self.cli.start()
        
        try:
            while self.running:
                self._process_cycle()
                time.sleep(0.1)  # Short sleep to prevent CPU overuse
                
        except KeyboardInterrupt:
            logger.info("Decision loop interrupted")
            self.running = False
        except Exception as e:
            logger.exception(f"Unexpected error in decision loop: {e}")
            self.running = False
        finally:
            self._cleanup()
    
    def _process_cycle(self) -> None:
        """Process a single cycle of the decision loop."""
        # Update state with timing info for debugging
        cycle_start_time = time.time()
        self.current_state["cycle_timestamp"] = cycle_start_time
        
        # 1. Check for CLI input if enabled
        command = None
        if self.cli_enabled and self.cli:
            cli_command = self.cli.get_command(timeout=0.01)
            if cli_command:
                logger.info(f"Received CLI command: {cli_command}")
                command = cli_command
        
        # 2. Check for audio input if enabled and no CLI command
        if command is None and self.audio_enabled and self.stt:
            logger.debug("Listening for commands")
            audio_input = self.stt.listen()
            if audio_input:
                logger.info(f"Heard: {audio_input}")
                # Parse the intent from speech
                if self.intent_parser:
                    intent_data = self.intent_parser.parse(audio_input)
                    if intent_data:
                        logger.debug(f"Parsed intent: {intent_data}")
                        command = intent_data
        
        # 3. Capture image if camera is enabled and hardware available
        frame = None
        vision_result = None
        if self.camera_enabled and self.camera:
            # logger.debug("Capturing image frame")
            try:
                frame = self.camera.capture()
                # Pass to vision system for processing if available
                if frame is not None and self.vision:
                    logger.debug("Processing image with vision model")
                    vision_result = self.vision.process_image(frame)
                    logger.debug(f"Vision result: {vision_result}")
            except Exception as e:
                logger.error(f"Error during camera capture or vision processing: {e}")
        elif self.camera_enabled and self.vision:
            # We have vision processing but no camera hardware
            logger.debug("Vision system available but no camera feed")
        
        # 4. Decide on action based on commands and vision
        action = self._decide_action(command, vision_result)
        
        # 5. Execute action if available
        if action:
            logger.debug(f"Executing action: {action}")
            self._execute_action(action)
        
        # Log cycle duration for performance monitoring
        cycle_duration = time.time() - cycle_start_time
        # logger.debug(f"Cycle completed in {cycle_duration:.3f} seconds")
        
    def _decide_action(
        self, 
        command: Optional[Dict[str, Any]], 
        vision_result: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Decide what action to take based on command and vision input.
        
        Args:
            command: Command from CLI or speech input
            vision_result: Result from vision processing
            
        Returns:
            Action to execute or None
        """
        # If we have a direct command, prioritize it
        if command:
            return command
        
        # Otherwise, use vision result to determine action
        if vision_result and self.vision:
            # Analyze vision result and return appropriate action
            # This is where more sophisticated decision-making would happen
            
            # Simple example: if we see a person, wave at them
            if "person" in str(vision_result).lower():
                return {
                    "action_type": "emote",
                    "params": {"emotion": "wave"}
                }
                
            # More decision logic would go here
        
        # No action decided
        return None
    
    def _execute_action(self, action):
        """
        Execute a given action using the appropriate controller.
        
        Args:
            action: Action dictionary with type and parameters
        """
        if not action:
            logger.warning("Received empty action, ignoring")
            return False
        
        try:
            action_type = action.get("action_type", "")
            params = action.get("params", {})
            
            logger.debug(f"Executing action: {action}")
            
            if action_type == "move":
                # Movement action
                direction = params.get("direction", "")
                speed = params.get("speed", 0.5)
                duration = params.get("duration", None)
                
                if not direction:
                    logger.warning("Missing direction parameter for move action")
                    return False
                
                logger.info(f"Moving {direction} at speed {speed}")
                return self.motion.move(direction, speed, duration)
                
            elif action_type == "turn":
                # Turn action
                direction = params.get("direction", "left")
                rate = params.get("rate", 0.5) 
                angle = params.get("angle", None)
                
                logger.info(f"Turning {direction} at rate {rate}" + 
                           (f" by {angle} degrees" if angle else " continuously"))
                return self.motion.turn(direction, rate, angle)
                
            elif action_type == "look_at":
                # Head movement action
                target = params.get("target", None)
                yaw = params.get("yaw", 0.0)
                pitch = params.get("pitch", 0.0)
                roll = params.get("roll", 0.0)
                
                if target:
                    logger.info(f"Looking at {target}")
                else:
                    logger.info(f"Setting head position: yaw={yaw}, pitch={pitch}, roll={roll}")
                    
                return self.motion.look_at(target, yaw, pitch, roll)
                
            elif action_type == "emote":
                # Emote action
                emotion = params.get("emotion", "neutral")
                intensity = params.get("intensity", 0.5)
                
                logger.info(f"Expressing emotion: {emotion} with intensity {intensity}")
                return self.emotes.express(emotion, intensity)
                
            elif action_type == "stop":
                # Stop all movement
                logger.info("Stopping all movement")
                return self.motion.stop()
                
            else:
                logger.warning(f"Unknown or unsupported action type: {action_type}")
                return False
                
        except Exception as e:
            logger.exception(f"Error executing action: {e}")
            return False
    
    def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        logger.info("Cleaning up resources...")
        
        # Stop CLI if it was started
        if self.cli:
            logger.debug("Stopping CLI controller")
            self.cli.stop()
        
        # Close camera if opened
        if self.camera:
            logger.debug("Closing camera")
            try:
                self.camera.close()
            except Exception as e:
                logger.error(f"Error closing camera: {e}")
        
        # Clean up audio resources
        if self.audio_system:
            logger.debug("Cleaning up audio system")
            try:
                self.audio_system.close()
            except Exception as e:
                logger.error(f"Error closing audio system: {e}")
        
        # Clean up motion controller
        if self.motion:
            logger.debug("Cleaning up motion controller")
            try:
                self.motion.close()
            except Exception as e:
                logger.error(f"Error closing motion controller: {e}")
        
        logger.info("Cleanup complete")
