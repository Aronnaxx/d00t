import argparse
import base64
import json
import logging
import os
import time
import threading
import queue
from io import BytesIO
from typing import Dict, List, Optional, Any, Union

import numpy as np
import requests
from PIL import Image

# Import our client
from client import DuckRobotClient

# Import Ollama manager
try:
    from ollama import OllamaManager, start_ollama_server
    OLLAMA_MANAGER_AVAILABLE = True
except ImportError:
    OLLAMA_MANAGER_AVAILABLE = False

# Try to import voice integration
try:
    from voice_integration import VoiceCommandProcessor
    VOICE_INTEGRATION_AVAILABLE = True
except ImportError:
    VOICE_INTEGRATION_AVAILABLE = False
    logger.warning("Voice integration module not found. Install voice_integration.py for voice command capabilities.")

# Set up logging with detailed format for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vlm_integration.log")
    ]
)
logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API for vision models."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llava"):
        """Initialize the Ollama client.
        
        Args:
            base_url: URL for the Ollama API
            model_name: Name of the model to use (e.g., llava, bakllava)
        """
        self.base_url = base_url
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"
        self.ollama_manager = None
        
        # If ollama.py is available, use it to manage Ollama
        if OLLAMA_MANAGER_AVAILABLE:
            try:
                host, port = self._parse_url(base_url)
                self.ollama_manager = start_ollama_server(
                    model_name=model_name,
                    host=host,
                    port=port
                )
                logger.info(f"Using Ollama manager with model {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to start Ollama using manager: {e}, falling back to direct API")
        
        # Fall back to direct API
        # Test the connection
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                if model_name not in model_names:
                    logger.warning(f"Model '{model_name}' not found in available models: {model_names}")
                else:
                    logger.info(f"Successfully connected to Ollama with model: {model_name}")
            else:
                logger.warning(f"Failed to get Ollama models. Status code: {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {e}")
    
    def _parse_url(self, url: str) -> tuple:
        """Parse host and port from URL."""
        if "://" in url:
            url = url.split("://")[1]
        
        if ":" in url:
            host, port = url.split(":")
            return host, int(port)
        
        return url, 11434
    
    def query_with_image(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Query the Ollama model with an image and text prompt.
        
        Args:
            image: PIL Image to process
            prompt: Text prompt to send with the image
            
        Returns:
            Response from the model
        """
        # If using Ollama manager, use it to query the model
        if self.ollama_manager:
            try:
                # Convert image to base64
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                response = self.ollama_manager.query_model(
                    model_name=self.model_name,
                    prompt=prompt,
                    images=[img_str]
                )
                
                return {
                    "text": response.get("response", ""),
                    "model": self.model_name
                }
            except Exception as e:
                logger.error(f"Error querying Ollama via manager: {e}")
                return {"error": str(e)}
        
        # Fall back to direct API
        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Create the prompt with image
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [img_str],
                "stream": False
            }
            
            # Make the request
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Format response for compatibility with other VLMs
            return {
                "text": result.get("response", ""),
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """Clean up Ollama manager."""
        if self.ollama_manager and hasattr(self.ollama_manager, 'stop'):
            logger.info("Stopping Ollama manager")
            self.ollama_manager.stop()

class AutonomousExplorer:
    """Class to handle autonomous exploration behavior."""
    
    def __init__(self, controller):
        """Initialize the autonomous explorer.
        
        Args:
            controller: VLMDuckController instance
        """
        self.controller = controller
        self.running = False
        self.explore_thread = None
        self.command_queue = queue.Queue()
        self.last_decision_time = 0
        self.decision_interval = 3.0  # Make a new decision every 3 seconds
        self.exploring = False
        self.current_instruction = None
        
        # Exploration prompts
        self.exploration_prompts = [
            "Describe what you see in the environment. What is interesting? What should I explore?",
            "You are a robot in an exploration mission. What do you observe in this scene? What should you do next?",
            "As an autonomous robot, decide what to do next based on this camera view. Describe what you see and choose a direction to move.",
            "You are exploring this environment. What do you notice that's worth investigating? Where should you go?",
            "Make a decision about where to go next based on this camera view. Explain your reasoning briefly."
        ]
    
    def start(self):
        """Start autonomous exploration."""
        if self.running:
            return
        
        self.running = True
        self.exploring = True
        self.explore_thread = threading.Thread(target=self._exploration_loop)
        self.explore_thread.daemon = True
        self.explore_thread.start()
        
        logger.info("Autonomous exploration started")
    
    def stop(self):
        """Stop autonomous exploration."""
        self.running = False
        if self.explore_thread and self.explore_thread.is_alive():
            self.explore_thread.join(timeout=2.0)
        
        # Send stop command to robot
        self.controller.duck_client.send_direct_command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        logger.info("Autonomous exploration stopped")
    
    def pause(self):
        """Pause autonomous exploration."""
        self.exploring = False
        # Send stop command to robot
        self.controller.duck_client.send_direct_command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        logger.info("Autonomous exploration paused")
    
    def resume(self):
        """Resume autonomous exploration."""
        self.exploring = True
        logger.info("Autonomous exploration resumed")
    
    def queue_command(self, command_text):
        """Queue a command to be executed.
        
        Args:
            command_text: Natural language command text
        """
        self.command_queue.put(command_text)
        logger.info(f"Command queued: {command_text}")
    
    def _exploration_loop(self):
        """Main loop for autonomous exploration."""
        try:
            while self.running:
                # Process any queued user commands first
                try:
                    command_text = self.command_queue.get_nowait()
                    # Process user command
                    logger.info(f"Processing user command: {command_text}")
                    self._process_user_command(command_text)
                    # Skip the autonomous decision this cycle
                    time.sleep(1.0)
                    continue
                except queue.Empty:
                    pass
                
                # If exploration is paused, just wait
                if not self.exploring:
                    time.sleep(0.5)
                    continue
                
                # Check if it's time to make a new decision
                current_time = time.time()
                if current_time - self.last_decision_time < self.decision_interval:
                    time.sleep(0.1)
                    continue
                
                # Make autonomous decision
                self._make_autonomous_decision()
                self.last_decision_time = time.time()
                
                # Sleep a bit to avoid tight loop
                time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error in exploration loop: {e}")
            self.running = False
    
    def _process_user_command(self, command_text):
        """Process a user command.
        
        Args:
            command_text: Natural language command text
        """
        # Check for special commands first
        lower_cmd = command_text.lower()
        
        if "stop" in lower_cmd or "halt" in lower_cmd:
            # Stop exploration
            self.pause()
            self.controller.duck_client.send_direct_command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.current_instruction = "Stopped as requested."
            return
        
        if "start" in lower_cmd or "continue" in lower_cmd or "resume" in lower_cmd or "explore" in lower_cmd:
            # Resume exploration
            self.resume()
            self.current_instruction = "Resuming exploration."
            return
        
        # Otherwise, treat it as a normal command
        # Get current frame
        frame = self.controller.get_current_frame()
        if frame is None:
            logger.warning("Failed to get frame for command processing")
            return
        
        # Formulate a prompt that includes the command
        prompt = f"You are controlling a robot with a camera. The user has given you this command: '{command_text}'. Based on what you see in the camera view, execute this command. Describe what you're seeing and your planned action."
        
        # Query VLM
        vlm_response = self.controller.query_vlm(frame, prompt)
        logger.info(f"VLM response for user command: {json.dumps(vlm_response, indent=2)}")
        
        # Process response to command
        command = self.controller.process_vlm_response_to_command(vlm_response)
        if command:
            # Send command to duck robot
            result = self.controller.duck_client.send_direct_command(command)
            logger.info(f"Sent command {command} for user instruction, result: {result}")
            self.controller.command_history.append(command)
            
            # Store current instruction
            if "text" in vlm_response:
                self.current_instruction = vlm_response["text"]
        else:
            logger.warning(f"Failed to extract a valid command from VLM response for user instruction: {command_text}")
    
    def _make_autonomous_decision(self):
        """Make an autonomous decision about what to do next."""
        # Get current frame
        frame = self.controller.get_current_frame()
        if frame is None:
            logger.warning("Failed to get frame for autonomous decision")
            return
        
        # Save the frame for debugging
        frame_path = f"explorer_frame_{int(time.time())}.png"
        frame.save(frame_path)
        
        # Select a random exploration prompt
        prompt = np.random.choice(self.exploration_prompts)
        
        # Add context from recent commands
        context = "\n".join([f"Previous action: {cmd}" for cmd in self.controller.command_history[-3:]])
        full_prompt = f"{prompt}\n\nContext:\n{context}" if self.controller.command_history else prompt
        
        # Query VLM
        vlm_response = self.controller.query_vlm(frame, full_prompt)
        logger.info(f"Autonomous exploration VLM response: {json.dumps(vlm_response, indent=2)}")
        
        # Process response to command
        command = self.controller.process_vlm_response_to_command(vlm_response)
        if command:
            # Send command to duck robot
            result = self.controller.duck_client.send_direct_command(command)
            logger.info(f"Autonomous action: {command}, result: {result}")
            self.controller.command_history.append(command)
            
            # Store current instruction
            if "text" in vlm_response:
                self.current_instruction = vlm_response["text"]
        else:
            # If we couldn't extract a command, try a random movement
            logger.warning("Failed to extract a valid command for autonomous exploration, using random movement")
            # Random gentle movement - forward, left or right turn
            rand_cmd = np.random.choice([
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Forward
                [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],  # Turn left
                [0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0]  # Turn right
            ])
            self.controller.duck_client.send_direct_command(rand_cmd)
            self.controller.command_history.append(rand_cmd)
            self.current_instruction = "Exploring randomly."

class VLMDuckController:
    """Class to integrate a VLM with the Duck Robot."""
    
    def __init__(
        self, 
        duck_api_url: str = "http://localhost:5000",
        vlm_type: str = "external",
        vlm_api_url: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llava",
        tts_model: str = "gemma3:4b",
        auto_launch_ollama: bool = True,
        use_voice: bool = False,
        use_tts: bool = False,
        use_ollama_stt: bool = False,
        use_ollama_tts: bool = True,
        autonomous_mode: bool = False
    ):
        """Initialize the VLM Duck Controller.
        
        Args:
            duck_api_url: URL for the Duck Robot API
            vlm_type: Type of VLM to use ('external', 'ollama')
            vlm_api_url: URL for the external VLM API
            vlm_api_key: API key for the external VLM API
            ollama_url: URL for the Ollama API
            ollama_model: Model name for Ollama
            tts_model: Model name for text-to-speech
            auto_launch_ollama: Whether to automatically launch Ollama if using it
            use_voice: Whether to use voice commands
            use_tts: Whether to use text-to-speech
            use_ollama_stt: Whether to use Ollama for speech recognition
            use_ollama_tts: Whether to use Ollama for text-to-speech
            autonomous_mode: Whether to start in autonomous mode
        """
        self.duck_client = DuckRobotClient(api_url=duck_api_url)
        self.vlm_type = vlm_type
        self.vlm_api_url = vlm_api_url
        self.vlm_api_key = vlm_api_key
        self.auto_launch_ollama = auto_launch_ollama
        self.use_voice = use_voice
        self.use_tts = use_tts
        self.use_ollama_stt = use_ollama_stt
        self.use_ollama_tts = use_ollama_tts
        self.tts_model = tts_model
        self.autonomous_mode = autonomous_mode
        
        # Initialize Ollama client if using Ollama
        self.ollama_client = None
        if vlm_type == "ollama":
            self.ollama_client = OllamaClient(
                base_url=ollama_url,
                model_name=ollama_model
            )
        
        # Initialize voice command processor if enabled
        self.voice_processor = None
        if use_voice and VOICE_INTEGRATION_AVAILABLE:
            logger.info("Initializing voice command processor")
            self.voice_processor = VoiceCommandProcessor(
                command_callback=self._process_voice_command,
                use_tts=use_tts,
                use_microphone=True,
                use_ollama_stt=use_ollama_stt,
                use_ollama_tts=use_ollama_tts
            )
        elif use_voice:
            logger.warning("Voice integration requested but not available. Install voice_integration.py")
        
        # Initialize autonomous explorer if enabled
        self.explorer = None
        if autonomous_mode:
            logger.info("Initializing autonomous explorer")
            self.explorer = AutonomousExplorer(self)
        
        # Initialize state
        self.last_frame = None
        self.last_frame_time = 0
        self.command_history = []
        self.running = False
        self.command_queue = queue.Queue()
        self.process_thread = None
    
    def _process_voice_command(self, command_text):
        """Process a voice command.
        
        Args:
            command_text: Command text from speech recognition
        """
        if self.explorer and self.explorer.running:
            # Queue the command for the explorer to process
            self.explorer.queue_command(command_text)
        else:
            # Process the command directly
            self.command_queue.put(command_text)
    
    def initialize_simulation(self, onnx_model_path: str) -> bool:
        """Initialize the Duck Robot simulation.
        
        Args:
            onnx_model_path: Path to the ONNX model
            
        Returns:
            True if initialization was successful
        """
        try:
            result = self.duck_client.initialize(onnx_model_path=onnx_model_path)
            success = result.get("success", False)
            
            if success:
                logger.info("Simulation initialized successfully")
                # Start the process thread
                self.running = True
                self.process_thread = threading.Thread(target=self._process_loop)
                self.process_thread.daemon = True
                self.process_thread.start()
                
                # Start voice processing if enabled
                if self.voice_processor:
                    self.voice_processor.start()
                
                # Start autonomous exploration if enabled
                if self.explorer and self.autonomous_mode:
                    self.explorer.start()
            else:
                logger.error(f"Failed to initialize simulation: {result.get('error', 'Unknown error')}")
            
            return success
        except Exception as e:
            logger.error(f"Error initializing simulation: {e}")
            return False
    
    def _process_loop(self):
        """Main processing loop for commands."""
        try:
            while self.running:
                try:
                    # Process any queued commands
                    command_text = self.command_queue.get(timeout=0.1)
                    logger.info(f"Processing command: {command_text}")
                    
                    # Get current frame
                    frame = self.get_current_frame()
                    if frame:
                        # Create a prompt with the command
                        prompt = f"You are controlling a duck robot. The user has issued this command: '{command_text}'. Based on what you see, execute this command. Respond with the appropriate action."
                        
                        # Query VLM
                        vlm_response = self.query_vlm(frame, prompt)
                        logger.info(f"VLM response for command '{command_text}': {json.dumps(vlm_response, indent=2)}")
                        
                        # Process response to get robot command
                        robot_command = self.process_vlm_response_to_command(vlm_response)
                        if robot_command:
                            # Send command to robot
                            result = self.duck_client.send_direct_command(robot_command)
                            logger.info(f"Sent command {robot_command} for user command '{command_text}', result: {result}")
                            self.command_history.append(robot_command)
                            
                            # Speak response if TTS is enabled
                            if self.voice_processor and self.use_tts:
                                if "text" in vlm_response:
                                    self.voice_processor.speak(f"I'll {vlm_response['text']}")
                        else:
                            logger.warning(f"Failed to extract a valid command from VLM response for: {command_text}")
                except queue.Empty:
                    # No commands to process
                    pass
                except Exception as e:
                    logger.error(f"Error processing command: {e}")
                
                # Sleep a bit to avoid tight loop
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in process loop: {e}")
            self.running = False
    
    def get_current_frame(self) -> Optional[Image.Image]:
        """Get the current frame from the Duck Robot.
        
        Returns:
            PIL Image if successful, None otherwise
        """
        try:
            # Don't request frames too quickly
            current_time = time.time()
            if current_time - self.last_frame_time < 0.1:  # Max 10 FPS
                if self.last_frame:
                    return self.last_frame
            
            # Get fresh frame
            response = self.duck_client.get_frame()
            
            if not response.get("success", False):
                logger.error(f"Failed to get frame: {response.get('error')}")
                return None
            
            # Decode base64 image
            img_data = base64.b64decode(response["image"])
            img = Image.open(BytesIO(img_data))
            
            # Cache and return
            self.last_frame = img
            self.last_frame_time = current_time
            return img
        
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
    
    def query_vlm(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Query the configured VLM with the given image and prompt.
        
        Args:
            image: PIL Image to send to the VLM
            prompt: Text prompt to send to the VLM
            
        Returns:
            Response from the VLM
        """
        # Use Ollama if configured
        if self.vlm_type == "ollama" and self.ollama_client:
            return self.ollama_client.query_with_image(image, prompt)
        
        # Use external VLM API
        elif self.vlm_type == "external" and self.vlm_api_url:
            return self.query_external_vlm(image, prompt)
        
        else:
            logger.error(f"No valid VLM configuration for type: {self.vlm_type}")
            return {"error": "No valid VLM configuration"}
    
    def query_external_vlm(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Query an external VLM API with the given image and prompt.
        
        Args:
            image: PIL Image to send to the VLM
            prompt: Text prompt to send to the VLM
            
        Returns:
            Response from the VLM API
        """
        if not self.vlm_api_url:
            logger.error("No VLM API URL provided")
            return {"error": "No VLM API URL provided"}
        
        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Prepare request
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.vlm_api_key:
                headers["Authorization"] = f"Bearer {self.vlm_api_key}"
            
            payload = {
                "image": img_str,
                "prompt": prompt
            }
            
            # Make request
            response = requests.post(
                self.vlm_api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Error querying external VLM: {e}")
            return {"error": str(e)}
    
    def process_vlm_response_to_command(self, vlm_response: Dict[str, Any]) -> Optional[List[float]]:
        """Process a VLM response into a Duck Robot command.
        
        Args:
            vlm_response: Response from the VLM
            
        Returns:
            Command array if successful, None otherwise
        """
        try:
            # This implementation will depend on the specific VLM API you're using
            # First check for error
            if "error" in vlm_response:
                logger.error(f"Error in VLM response: {vlm_response['error']}")
                return None
                
            # Check if it directly returns a command array
            if "command" in vlm_response:
                return vlm_response["command"]
            
            # Or maybe it returns a text description that we need to parse
            if "text" in vlm_response:
                text = vlm_response["text"].lower()
                logger.debug(f"Processing VLM text response: {text}")
                
                # Simple keyword-based parsing
                if "move forward" in text or "go forward" in text:
                    return [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif "move backward" in text or "go backward" in text or "go back" in text:
                    return [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif "turn left" in text:
                    return [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
                elif "turn right" in text:
                    return [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]
                elif "move left" in text or "go left" in text or "strafe left" in text:
                    return [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif "move right" in text or "go right" in text or "strafe right" in text:
                    return [0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif "look up" in text:
                    return [0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0]
                elif "look down" in text:
                    return [0.0, 0.0, 0.0, -0.3, 0.5, 0.0, 0.0]
                elif "look left" in text:
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0]
                elif "look right" in text:
                    return [0.0, 0.0, 0.0, 0.0, 0.0, -0.8, 0.0]
                elif "stop" in text or "halt" in text or "stay" in text:
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            logger.warning(f"Could not extract command from VLM response: {vlm_response}")
            return None
        
        except Exception as e:
            logger.error(f"Error processing VLM response: {e}")
            return None
    
    def run_loop(self, 
                prompt_template: str = "What should the robot do next? Describe any obstacles or interesting features you see.",
                max_iterations: int = 100,
                delay: float = 1.0) -> None:
        """Run a continuous loop of getting frames, querying the VLM, and sending commands.
        
        Args:
            prompt_template: Template for the prompt to send to the VLM
            max_iterations: Maximum number of iterations to run
            delay: Delay between iterations in seconds
        """
        logger.info("Starting VLM control loop")
        
        for i in range(max_iterations):
            try:
                logger.info(f"Iteration {i+1}/{max_iterations}")
                
                # Get current frame
                frame = self.get_current_frame()
                if frame is None:
                    logger.warning("Failed to get frame, skipping iteration")
                    time.sleep(delay)
                    continue
                
                # Save the frame for debugging
                frame_path = f"frame_{i:04d}.png"
                frame.save(frame_path)
                logger.debug(f"Saved frame to {frame_path}")
                
                # Generate prompt with history context
                context = "\n".join([f"Previous action: {cmd}" for cmd in self.command_history[-3:]])
                full_prompt = f"{prompt_template}\n\nContext:\n{context}" if self.command_history else prompt_template
                
                # Query VLM
                vlm_response = self.query_vlm(frame, full_prompt)
                logger.info(f"VLM response: {json.dumps(vlm_response, indent=2)}")
                
                # Process response to command
                command = self.process_vlm_response_to_command(vlm_response)
                if command:
                    # Send command to duck robot
                    result = self.duck_client.send_direct_command(command)
                    logger.info(f"Sent command {command}, result: {result}")
                    self.command_history.append(command)
                else:
                    logger.warning("Failed to extract a valid command, sending stop command")
                    self.duck_client.send_direct_command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
                # Delay before next iteration
                time.sleep(delay)
            
            except KeyboardInterrupt:
                logger.info("Loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in VLM control loop: {e}")
                time.sleep(delay)
        
        logger.info("VLM control loop completed")

    def run_nl_command_mode(self) -> None:
        """Run in interactive natural language command mode."""
        logger.info("Starting interactive NL command mode. Enter commands or 'quit' to exit.")
        
        try:
            while True:
                # Get user input
                nl_command = input("> ")
                
                if nl_command.lower() in ("quit", "exit", "q"):
                    break
                
                # Add to command queue instead of processing directly
                self.command_queue.put(nl_command)
                
                # Get and display the current frame
                frame = self.get_current_frame()
                if frame:
                    frame_path = "current_frame.png"
                    frame.save(frame_path)
                    print(f"Frame saved to {frame_path}")
        
        except KeyboardInterrupt:
            print("\nExiting interactive mode")
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        # Stop voice processing if active
        if self.voice_processor:
            self.voice_processor.stop()
        
        # Stop autonomous exploration if active
        if self.explorer and self.explorer.running:
            self.explorer.stop()
        
        # Stop processing thread
        self.running = False
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        
        # Shutdown the duck robot
        try:
            self.duck_client.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down duck robot: {e}")

def main() -> None:
    """Main function to run the VLM Duck Controller."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="VLM Duck Robot Controller")
    parser.add_argument("--duck-api", type=str, default="http://localhost:5000", help="URL for the Duck Robot API")
    
    # VLM type selection
    parser.add_argument("--vlm-type", type=str, choices=["external", "ollama"], default="ollama", 
                        help="Type of VLM to use (external API or local Ollama)")
    
    # External VLM API options
    parser.add_argument("--vlm-api", type=str, help="URL for the external VLM API")
    parser.add_argument("--vlm-key", type=str, help="API key for the external VLM API")
    
    # Ollama options
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL for the Ollama API")
    parser.add_argument("--ollama-model", type=str, default="llava", help="Model name for Ollama (e.g., llava, bakllava)")
    parser.add_argument("--no-auto-launch", action="store_true", help="Don't automatically launch Ollama")
    
    # Simulation options
    parser.add_argument("--onnx-model", type=str, default="model.onnx", help="Path to the ONNX model file")
    
    # Mode options
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--autonomous", action="store_true", help="Run in autonomous exploration mode")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for VLM loop")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between iterations in seconds")
    
    # Voice integration options
    parser.add_argument("--voice", action="store_true", help="Enable voice command recognition")
    parser.add_argument("--tts", action="store_true", help="Enable text-to-speech")
    parser.add_argument("--tts-model", type=str, default="gemma3:4b", help="Model to use for TTS")
    parser.add_argument("--ollama-stt", action="store_true", help="Use Ollama for speech recognition")
    parser.add_argument("--ollama-tts", action="store_true", help="Use Ollama for text-to-speech")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Initialize controller based on VLM type
    if args.vlm_type == "ollama":
        logger.info(f"Using Ollama model {args.ollama_model} at {args.ollama_url}")
        controller = VLMDuckController(
            duck_api_url=args.duck_api,
            vlm_type="ollama",
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            tts_model=args.tts_model,
            auto_launch_ollama=not args.no_auto_launch,
            use_voice=args.voice,
            use_tts=args.tts,
            use_ollama_stt=args.ollama_stt,
            use_ollama_tts=args.ollama_tts,
            autonomous_mode=args.autonomous
        )
    else:
        # External VLM API
        if not args.vlm_api and not args.interactive and not args.autonomous:
            logger.error("External VLM type requires --vlm-api parameter")
            return
        
        controller = VLMDuckController(
            duck_api_url=args.duck_api,
            vlm_type="external",
            vlm_api_url=args.vlm_api,
            vlm_api_key=args.vlm_key,
            tts_model=args.tts_model,
            use_voice=args.voice,
            use_tts=args.tts,
            use_ollama_stt=args.ollama_stt,
            use_ollama_tts=args.ollama_tts,
            autonomous_mode=args.autonomous
        )
    
    # Initialize simulation
    logger.info(f"Initializing simulation with ONNX model: {args.onnx_model}")
    if not controller.initialize_simulation(args.onnx_model):
        logger.error("Failed to initialize simulation, exiting")
        return
    
    # Run in selected mode
    try:
        if args.interactive:
            controller.run_nl_command_mode()
        elif args.autonomous:
            # In autonomous mode, the explorer is already running
            # Keep the main thread alive
            logger.info("Running in autonomous mode. Press Ctrl+C to stop")
            print("Autonomous exploration active. You can speak commands or type them below.")
            print("Example commands: 'explore that area', 'move forward', 'turn left', 'stop'")
            
            if args.voice:
                print("Voice commands enabled. Speak clearly into your microphone.")
            
            # Allow text input even in autonomous mode
            while True:
                try:
                    nl_command = input("> ")
                    if nl_command.lower() in ("quit", "exit", "q"):
                        break
                    # Add to command queue or explorer queue
                    if controller.explorer and controller.explorer.running:
                        controller.explorer.queue_command(nl_command)
                    else:
                        controller.command_queue.put(nl_command)
                except (KeyboardInterrupt, EOFError):
                    break
        elif args.vlm_type == "ollama" or args.vlm_api:
            controller.run_loop(max_iterations=args.iterations, delay=args.delay)
        else:
            logger.error("Either --interactive, --autonomous mode, or a valid VLM configuration must be specified")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        controller.cleanup()


# Example usage of the VLM integration
if __name__ == "__main__":
    main() 