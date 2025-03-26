import argparse
import base64
import json
import logging
import os
import time
from io import BytesIO
from typing import Dict, List, Optional, Any, Union

import requests
from PIL import Image

# Import our client
from client import DuckRobotClient

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

class VLMDuckController:
    """Class to integrate a VLM with the Duck Robot."""
    
    def __init__(
        self, 
        duck_api_url: str = "http://localhost:5000",
        vlm_api_url: Optional[str] = None,
        vlm_api_key: Optional[str] = None
    ):
        """Initialize the VLM Duck Controller.
        
        Args:
            duck_api_url: URL for the Duck Robot API
            vlm_api_url: URL for the VLM API (if using an external VLM)
            vlm_api_key: API key for the VLM API (if required)
        """
        self.duck_client = DuckRobotClient(api_url=duck_api_url)
        self.vlm_api_url = vlm_api_url
        self.vlm_api_key = vlm_api_key
        
        # Initialize state
        self.last_frame = None
        self.last_frame_time = 0
        self.command_history = []
    
    def initialize_simulation(self, onnx_model_path: str) -> bool:
        """Initialize the Duck Robot simulation.
        
        Args:
            onnx_model_path: Path to the ONNX model
            
        Returns:
            True if initialization was successful
        """
        try:
            result = self.duck_client.initialize(onnx_model_path=onnx_model_path)
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Error initializing simulation: {e}")
            return False
    
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
            logger.error(f"Error querying VLM: {e}")
            return {"error": str(e)}
    
    def process_vlm_response_to_command(self, vlm_response: Dict[str, Any]) -> Optional[List[float]]:
        """Process a VLM response into a Duck Robot command.
        
        This is a simple example that could be expanded based on the VLM's output format.
        
        Args:
            vlm_response: Response from the VLM API
            
        Returns:
            Command array if successful, None otherwise
        """
        try:
            # This implementation will depend on the specific VLM API you're using
            # Here's a simple example that assumes the VLM returns a "command" key with an array
            if "command" in vlm_response:
                return vlm_response["command"]
            
            # Or maybe it returns a text description that we need to parse
            if "text" in vlm_response:
                text = vlm_response["text"].lower()
                
                # Simple keyword-based parsing
                if "move forward" in text:
                    return [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif "move backward" in text:
                    return [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif "turn left" in text:
                    return [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
                elif "turn right" in text:
                    return [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]
                elif "look up" in text:
                    return [0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0]
                elif "look down" in text:
                    return [0.0, 0.0, 0.0, -0.3, 0.5, 0.0, 0.0]
                elif "stop" in text:
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
                
                # Query VLM if API URL is provided
                if self.vlm_api_url:
                    # Generate prompt with history context
                    context = "\n".join([f"Previous action: {cmd}" for cmd in self.command_history[-3:]])
                    full_prompt = f"{prompt_template}\n\nContext:\n{context}" if self.command_history else prompt_template
                    
                    vlm_response = self.query_vlm(frame, full_prompt)
                    logger.info(f"VLM response: {json.dumps(vlm_response, indent=2)}")
                    
                    # Process response to command
                    command = self.process_vlm_response_to_command(vlm_response)
                    if command:
                        # Send command to duck robot
                        result = self.duck_client.send_direct_command(command)
                        logger.info(f"Sent command {command}, result: {result}")
                        self.command_history.append(command)
                
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
                
                # Send NL command
                result = self.duck_client.send_nl_command(nl_command)
                print(json.dumps(result, indent=2))
                
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


def vlm_test_script() -> None:
    """Sample script showing how a VLM/LLM system could control the Duck Robot."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="VLM Duck Robot Controller")
    parser.add_argument("--duck-api", type=str, default="http://localhost:5000", help="URL for the Duck Robot API")
    parser.add_argument("--vlm-api", type=str, help="URL for the VLM API")
    parser.add_argument("--vlm-key", type=str, help="API key for the VLM API")
    parser.add_argument("--onnx-model", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for VLM loop")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between iterations in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Initialize controller
    controller = VLMDuckController(
        duck_api_url=args.duck_api,
        vlm_api_url=args.vlm_api,
        vlm_api_key=args.vlm_key
    )
    
    # Initialize simulation
    if not controller.initialize_simulation(args.onnx_model):
        logger.error("Failed to initialize simulation, exiting")
        return
    
    # Run in selected mode
    try:
        if args.interactive:
            controller.run_nl_command_mode()
        elif args.vlm_api:
            controller.run_loop(max_iterations=args.iterations, delay=args.delay)
        else:
            logger.error("Either --interactive or --vlm-api must be specified")
    finally:
        # Shutdown the simulation when done
        controller.duck_client.shutdown()


# Example usage of the VLM integration
if __name__ == "__main__":
    vlm_test_script() 