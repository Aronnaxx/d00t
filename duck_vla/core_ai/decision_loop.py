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
        llm_provider: str = "ollama",
        llm_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the decision loop.
        
        Args:
            simulate: Whether to run in simulation mode using OpenDuckPlayground
            audio_enabled: Whether to enable audio input/output
            camera_enabled: Whether to enable camera input
            cli_enabled: Whether to enable CLI for direct command input
            local_model: Whether to use a local model (Ollama) or remote API
            vision_model: Vision model name to use (default from env or 'gemma3')
            onnx_model_path: Path to ONNX model for simulation (default from env or None)
            llm_provider: LLM provider to use ('ollama', 'openai', 'anthropic')
            llm_model: Specific model to use with the LLM provider
            system_prompt: Custom system prompt to use for the LLM
        """
        self.simulate = simulate
        self.audio_enabled = audio_enabled
        self.camera_enabled = camera_enabled
        self.cli_enabled = cli_enabled
        
        # if local_model is True, we will use the local model
        # if local_model is False, we will use the external model
        self.local_model = local_model
        
        # LLM provider settings
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        
        # Get vision model from environment or parameter
        self.vision_model = vision_model or os.environ.get("DUCK_VISION_MODEL", "gemma3")
        
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
        self.central_model = None
        
        # Initialize LLM and AI components
        self._init_ai_components()
        
        # Initialize vision only if camera is enabled
        if camera_enabled:
            self._init_vision_components()

        # Initialize audio components if audio is enabled    
        if audio_enabled:
            self._init_audio_components()
        
        # Initialize CLI if enabled
        if cli_enabled:
            self._init_cli_controller()
            
        # Initialize action system
        self._init_action_system()
            
        logger.info("Decision loop initialization complete")
    
    def _init_ai_components(self):
        """Initialize AI components including LLM provider"""
        try:
            from duck_vla.core_ai.central_model import CentralModel
            
            logger.debug(f"Initializing central AI model with provider: {self.llm_provider}")
            
            # Initialize the central model with the specified provider and settings
            self.central_model = CentralModel(
                provider_type=self.llm_provider,
                model_name=self.llm_model,
                system_prompt=self.system_prompt,
                debug_mode=logger.level == logging.DEBUG
            )
            
            logger.info(f"Central AI model initialized with {self.llm_provider} provider")
            
            # Also initialize the intent parser
            from duck_vla.core_ai.intent_parser import IntentParser
            self.intent_parser = IntentParser()
            logger.info("Intent parser initialized")
            
        except ImportError as e:
            logger.error(f"Failed to import AI modules: {e}")
            logger.debug(traceback.format_exc())
            self.central_model = None
        except Exception as e:
            logger.error(f"Unexpected error initializing AI components: {e}")
            logger.debug(traceback.format_exc())
            self.central_model = None
    
    def _init_vision_components(self):
        """Initialize vision and camera components with graceful error handling."""
        try:
            from duck_vla.camera.arducam_capture import ArduCamCapture
            
            logger.debug("Initializing camera module")

            # Try to initialize camera but continue if hardware not available
            try:
                self.camera = ArduCamCapture()
                logger.info("Camera initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize camera hardware: {e}")
                logger.info("Continuing without camera hardware, but vision processing will still be available")
                self.camera = None

            # Vision processing will use our central model if available
            if self.central_model is not None:
                logger.info("Vision processing will use the central AI model")
                # In a real implementation, we'd configure the central model for vision tasks
                # self.vision = self.central_model
            else:
                logger.warning("Central AI model not available, vision processing will be limited")

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
                from duck_vla.sounds.audio import AudioSystem
                
                logger.debug("Successfully imported audio modules")
            except ImportError as e:
                logger.warning(f"Failed to import audio modules: {e}")
                return
            
            # Intent parser is now initialized in _init_ai_components
            
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
            self.audio_system = None
    
    def _init_cli_controller(self):
        """Initialize CLI controller with error handling."""
        try:
            from duck_vla.utils.cli_controller import CLIController
            
            logger.debug("Initializing CLI controller")
            self.cli = CLIController(debug=logger.level == logging.DEBUG)
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
            from duck_vla.actions.movement import Movement
            
            logger.info(f"Initializing {'simulated' if self.simulate else 'real'} movement")
            
            # Create the movement controller with the appropriate mode
            self.motion = Movement(
                simulated=self.simulate,
                debug_logging=logger.level == logging.DEBUG
            )
            logger.info(f"{'Simulated' if self.simulate else 'Real'} movement controller initialized")
            
            # Initialize emote system
            try:
                from duck_vla.actions.emotes import EmoteController
                self.emotes = EmoteController(audio_enabled=self.audio_enabled)
                logger.info("Emote controller initialized successfully")
            except ImportError as e:
                logger.warning(f"Failed to import emote controller: {e}")
                self.emotes = None
            except Exception as e:
                logger.warning(f"Failed to initialize emote controller: {e}")
                self.emotes = None
            
        except ImportError as e:
            logger.error(f"Failed to import action modules: {e}")
            logger.debug(traceback.format_exc())
            self.motion = None
            self.emotes = None
        except Exception as e:
            logger.error(f"Failed to initialize movement controller: {e}")
            logger.debug(traceback.format_exc())
            self.motion = None
            
        # Verify we have at least some action capability
        if self.motion is None:
            logger.warning("Motion controller not available - limited functionality")
        
    def run(self) -> None:
        """Run the main decision loop."""
        logger.info("Starting decision loop")
        self.running = True
        
        # Start CLI if enabled and connect it to the central model and movement controller
        if self.cli_enabled and self.cli:
            logger.info("Starting CLI controller")
            
            # Connect the CLI to the central model for natural language processing
            if self.central_model:
                self.cli.set_central_model(self.central_model)
                logger.debug("Connected CLI to central model for natural language processing")
            
            # Connect the CLI to the movement controller for direct commands
            if self.motion:
                self.cli.set_movement_controller(self.motion)
                logger.debug("Connected CLI to movement controller for direct commands")
                
            self.cli.start()
        
        try:
            # Main loop
            while self.running:
                self._process_cycle()
                time.sleep(0.1)  # Small delay to prevent CPU spinning
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            logger.debug(traceback.format_exc())
        finally:
            self._cleanup()
            
        logger.info("Decision loop stopped")
    
    def _process_cycle(self) -> None:
        """Process a single cycle of the decision loop."""
        # Get command from CLI or speech input
        command = self._get_command()
        
        # Get camera frame and perform vision processing if available
        vision_result = self._get_vision()
        
        # Make decision based on command and vision input
        action = self._decide_action(command, vision_result)
        
        # Execute action if any
        if action:
            self._execute_action(action)
            
        # Update current state
        self.current_state = {
            "status": "running",
            "last_command": command,
            "last_vision": vision_result,
            "last_action": action
        }
    
    def _get_command(self) -> Optional[Dict[str, Any]]:
        """Get command from available input sources."""
        # Check CLI queue first
        if self.cli_enabled and self.cli:
            cli_command = self.cli.get_command(timeout=0.01)
            if cli_command:
                logger.debug(f"Got command from CLI: {cli_command}")
                return cli_command
                
        # If STT is enabled, check for spoken commands
        if self.audio_enabled and self.stt:
            try:
                spoken_text = self.stt.get_latest_text()
                if spoken_text:
                    logger.debug(f"Got spoken text: {spoken_text}")
                    
                    # Parse the intent if possible
                    if self.intent_parser:
                        intent = self.intent_parser.parse(spoken_text)
                        if intent:
                            logger.debug(f"Parsed intent: {intent}")
                            return intent
                    
                    # If no intent was parsed but we have a central model,
                    # we can still try to process as natural language
                    if self.central_model:
                        return {
                            "intent_type": "natural_language",
                            "text": spoken_text,
                            "confidence": 0.8,
                            "original_text": spoken_text
                        }
            except Exception as e:
                logger.error(f"Error getting speech input: {e}")
                
        return None
    
    def _get_vision(self) -> Optional[Dict[str, Any]]:
        """Get and process camera input if available."""
        # Implementation would depend on camera and vision modules
        # For now, just return None
        return None
    
    def _decide_action(
        self, 
        command: Optional[Dict[str, Any]], 
        vision_result: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Decide what action to take based on command and vision input.
        
        Args:
            command: Parsed command input, if any
            vision_result: Vision processing result, if any
            
        Returns:
            Action to execute, or None if no action
        """
        # If we have a central model, use it to decide the action
        if self.central_model and command:
            try:
                # Process the command using the central model
                action_code = self.central_model.process_command(
                    command=command.get("text", ""),
                    context={"vision": vision_result} if vision_result else {}
                )
                
                # Parse the resulting action code
                # In a real implementation, this would execute the Python code
                # or transform it into action commands
                
                return {"type": "ai_generated", "code": action_code}
                
            except Exception as e:
                logger.error(f"Error processing command with central model: {e}")
                logger.debug(traceback.format_exc())
                return None
        
        # If we have a command but no central model, use rule-based approach
        elif command:
            # Simple rule-based action selection
            # In a real implementation, this would use the intent parser
            return {"type": "rule_based", "command": command}
            
        return None
    
    def _execute_action(self, action):
        """
        Execute the given action.
        
        Args:
            action: Action dictionary to execute
        """
        if not action:
            return
            
        try:
            action_type = action.get("type")
            
            if action_type == "ai_generated" and "code" in action:
                # Execute AI-generated Python code
                # SECURITY NOTE: In a production system, you would want to
                # carefully validate and sandbox this code execution
                logger.debug(f"Executing AI-generated code: {action['code']}")
                
                # In a real implementation, we'd execute the code safely
                # For now, just log it
                logger.info(f"Would execute AI code: {action['code'][:100]}...")
                
            elif action_type == "rule_based" and "command" in action:
                # Execute rule-based command
                command = action["command"]
                intent_type = command.get("intent_type")
                
                if not self.motion:
                    logger.warning("Motion controller not available, can't execute movement command")
                    return
                    
                # Handle different intent types
                if intent_type.startswith("move_"):
                    direction = intent_type.replace("move_", "")
                    if direction == "forward":
                        self.motion.move_forward()
                    elif direction == "backward":
                        self.motion.move_backward()
                        
                elif intent_type.startswith("turn_"):
                    direction = intent_type.replace("turn_", "")
                    if direction == "left":
                        self.motion.turn_left()
                    elif direction == "right":
                        self.motion.turn_right()
                        
                elif intent_type == "stop":
                    self.motion.stop()
                    
                elif intent_type.startswith("look_"):
                    target = intent_type.replace("look_", "")
                    if target == "up":
                        self.motion.look_up()
                    elif target == "down":
                        self.motion.look_down()
                    elif target == "left":
                        self.motion.look_left()
                    elif target == "right":
                        self.motion.look_right()
                
                logger.debug(f"Executed rule-based command: {intent_type}")
                
            else:
                logger.warning(f"Unknown action type: {action_type}")
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            logger.debug(traceback.format_exc())
    
    def _cleanup(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Cleaning up resources...")
        
        # Stop CLI controller if running
        if self.cli:
            try:
                self.cli.stop()
                logger.debug("CLI controller stopped")
            except Exception as e:
                logger.warning(f"Error stopping CLI controller: {e}")
        
        # Clean up motion controller if available
        if self.motion:
            try:
                self.motion.cleanup()
                logger.debug("Motion controller cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up motion controller: {e}")
        
        # Clean up audio system if available
        if self.audio_system:
            try:
                self.audio_system.cleanup()
                logger.debug("Audio system cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up audio system: {e}")
        
        logger.info("Cleanup complete")
