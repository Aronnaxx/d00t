"""
Brain module: Decision Loop - Main controller for the Duck VLA system

This module contains the core decision-making loop that integrates
vision, language, and action components to create autonomous behavior.
"""

import logging
import time
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
    ):
        """
        Initialize the decision loop.
        
        Args:
            simulate: Whether to run in simulation mode using OpenDuckPlayground
            audio_enabled: Whether to enable audio input/output
            camera_enabled: Whether to enable camera input
            cli_enabled: Whether to enable CLI for direct command input
        """
        self.simulate = simulate
        self.audio_enabled = audio_enabled
        self.camera_enabled = camera_enabled
        self.cli_enabled = cli_enabled
        
        self.running = False
        self.current_state = {"status": "initializing"}
        
        logger.debug("Setting up component imports...")
        
        # Import components conditionally based on configuration
        if camera_enabled:
            try:
                from duck_vla.camera.arducam_capture import ArduCamCapture
                from duck_vla.vision.moondream_wrapper import MoondreamVision
                
                logger.debug("Initializing camera and vision modules")
                self.camera = ArduCamCapture()
                # Use Ollama backend with moondream model
                self.vision = MoondreamVision(
                    backend="ollama",
                    model_id="moondream"
                )
            except ImportError as e:
                logger.warning(f"Failed to initialize camera/vision modules: {e}")
                self.camera = None
                self.vision = None
        else:
            logger.info("Camera input disabled, skipping camera and vision initialization")
            self.camera = None
            self.vision = None
            
        if audio_enabled:
            try:
                from duck_vla.language.stt import SpeechToText
                from duck_vla.language.intent_parser import IntentParser
                from duck_vla.utils.audio import AudioSystem
                
                logger.debug("Initializing audio and language modules")
                self.stt = SpeechToText()
                self.intent_parser = IntentParser()
                self.audio_system = AudioSystem()
            except ImportError as e:
                logger.warning(f"Failed to initialize audio/language modules: {e}")
                self.stt = None
                self.intent_parser = None
                self.audio_system = None
        else:
            logger.info("Audio disabled, skipping audio and language initialization")
            self.stt = None
            self.intent_parser = None
            self.audio_system = None
        
        # Initialize CLI if enabled
        if cli_enabled:
            try:
                from duck_vla.cli_controller import CLIController
                
                logger.debug("Initializing CLI controller")
                self.cli = CLIController()
            except ImportError as e:
                logger.warning(f"Failed to initialize CLI controller: {e}")
                self.cli = None
        else:
            logger.info("CLI disabled, skipping CLI controller initialization")
            self.cli = None
            
        # Initialize action system
        try:
            if self.simulate:
                logger.info("Using simulation-based movement")
                # Import simulation-specific modules here
                from duck_vla.action.motion_controller import SimulatedMotionController as MotionController
            else:
                logger.info("Using real hardware movement")
                from duck_vla.action.motion_controller import MotionController
                
            from duck_vla.action.emotes import EmoteController
            
            self.motion = MotionController()
            self.emotes = EmoteController(audio_enabled=audio_enabled)
            
        except ImportError as e:
            logger.critical(f"Failed to initialize action modules: {e}")
            raise RuntimeError("Cannot continue without action modules")
            
        logger.info("Decision loop initialization complete")
        
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
        
        # 3. Capture image if camera is enabled
        frame = None
        vision_result = None
        if self.camera_enabled and self.camera:
            logger.debug("Capturing image frame")
            frame = self.camera.capture_frame()
            if frame is not None and self.vision:
                # Process the frame with vision system
                logger.debug("Processing frame with vision system")
                vision_result = self.vision.process_frame(frame)
                logger.debug(f"Vision result: {vision_result}")
        
        # 4. Make decision based on inputs
        action = self._decide_action(command, vision_result)
        
        # 5. Execute action
        if action:
            logger.info(f"Executing action: {action}")
            self._execute_action(action)
        
        # Log timing for performance monitoring
        cycle_duration = time.time() - cycle_start_time
        logger.debug(f"Decision cycle completed in {cycle_duration:.4f}s")
    
    def _decide_action(
        self, 
        command: Optional[Dict[str, Any]], 
        vision_result: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Determine what action to take based on current inputs.
        
        Args:
            command: Parsed command intent from speech input or CLI
            vision_result: Results from vision processing
            
        Returns:
            Action dict or None if no action needed
        """
        # Simple priority-based decision making
        # 1. Explicit commands take precedence
        if command:
            logger.debug("Deciding action based on explicit command")
            return {
                "type": command.get("action_type", "unknown"),
                "params": command.get("params", {}),
                "source": "command"
            }
        
        # 2. React to vision if available
        if vision_result:
            logger.debug("Deciding action based on vision input")
            # Example: If we see a person, look at them
            if vision_result.get("person_detected"):
                return {
                    "type": "look_at",
                    "params": {"target": "person"},
                    "source": "vision"
                }
        
        # Default: No action needed
        return None
    
    def _execute_action(self, action: Dict[str, Any]) -> None:
        """
        Execute the specified action.
        
        Args:
            action: Action dict with type and parameters
        """
        action_type = action.get("type", "")
        params = action.get("params", {})
        
        try:
            if action_type == "move":
                self.motion.move(**params)
            elif action_type == "turn":
                self.motion.turn(**params)
            elif action_type == "look_at":
                self.motion.look_at(**params)
            elif action_type == "emote":
                self.emotes.play(params.get("emote", "neutral"))
            elif action_type == "stop":
                self.motion.stop()
            elif action_type == "get_status":
                status = self.motion.get_status() if self.motion else {"error": "Motion controller not available"}
                logger.info(f"Duck status: {status}")
            else:
                logger.warning(f"Unknown action type: {action_type}")
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
    
    def _cleanup(self) -> None:
        """Clean up resources before shutting down."""
        logger.info("Cleaning up decision loop resources")
        
        # Clean up CLI if initialized
        if hasattr(self, "cli") and self.cli:
            logger.debug("Cleaning up CLI controller")
            self.cli.stop()
        
        # Clean up camera if initialized
        if hasattr(self, "camera") and self.camera:
            logger.debug("Cleaning up camera")
            self.camera.release()
        
        # Clean up audio if initialized
        if hasattr(self, "audio_system") and self.audio_system:
            logger.debug("Cleaning up audio system")
            self.audio_system.close()
        
        # Clean up motion controller if initialized
        if hasattr(self, "motion") and self.motion:
            logger.debug("Cleaning up motion controller")
            self.motion.stop()
