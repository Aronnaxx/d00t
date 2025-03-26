import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union

import requests
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DuckRobotClient:
    """Client for interacting with the Duck Robot API."""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        """Initialize the Duck Robot client.
        
        Args:
            api_url: Base URL for the Duck Robot API.
        """
        self.api_url = api_url
        self.session = requests.Session()
        
        # Test the connection
        try:
            self.get_status()
            logger.info(f"Successfully connected to Duck Robot API at {api_url}")
        except requests.RequestException as e:
            logger.warning(f"Failed to connect to Duck Robot API: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the MuJoCo simulation.
        
        Returns:
            Dict containing status information.
        """
        response = self.session.get(f"{self.api_url}/status")
        response.raise_for_status()
        return response.json()
    
    def initialize(
        self, 
        onnx_model_path: str,
        model_path: Optional[str] = None,
        reference_data: Optional[str] = None,
        standing: bool = False
    ) -> Dict[str, Any]:
        """Initialize the MuJoCo simulation.
        
        Args:
            onnx_model_path: Path to the ONNX model file.
            model_path: Optional path to the XML model file.
            reference_data: Optional path to the reference data file.
            standing: Whether to start in standing mode.
            
        Returns:
            Dict containing result information.
        """
        data = {
            "onnx_model_path": onnx_model_path,
            "standing": standing
        }
        
        if model_path:
            data["model_path"] = model_path
            
        if reference_data:
            data["reference_data"] = reference_data
        
        response = self.session.post(f"{self.api_url}/initialize", json=data)
        response.raise_for_status()
        return response.json()
    
    def shutdown(self) -> Dict[str, Any]:
        """Shutdown the MuJoCo simulation.
        
        Returns:
            Dict containing result information.
        """
        response = self.session.post(f"{self.api_url}/shutdown")
        response.raise_for_status()
        return response.json()
    
    def send_direct_command(self, command: List[float]) -> Dict[str, Any]:
        """Send a direct command to the Duck Robot.
        
        Args:
            command: List of 7 floats: [vx, vy, omega, neck_pitch, head_pitch, head_yaw, head_roll]
            
        Returns:
            Dict containing result information.
        """
        data = {
            "command_type": "direct",
            "command": command
        }
        
        response = self.session.post(f"{self.api_url}/command", json=data)
        response.raise_for_status()
        return response.json()
    
    def send_nl_command(self, nl_command: str) -> Dict[str, Any]:
        """Send a natural language command to the Duck Robot.
        
        Args:
            nl_command: Natural language command string.
            
        Returns:
            Dict containing result information.
        """
        data = {
            "command_type": "natural_language",
            "nl_command": nl_command
        }
        
        response = self.session.post(f"{self.api_url}/command", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_frame(self) -> Dict[str, Any]:
        """Get the current camera frame from the simulation.
        
        Returns:
            Dict containing frame information, including base64-encoded image.
        """
        response = self.session.get(f"{self.api_url}/frame")
        response.raise_for_status()
        return response.json()
    
    def get_api_help(self) -> Dict[str, Any]:
        """Get API documentation.
        
        Returns:
            Dict containing API documentation.
        """
        response = self.session.get(f"{self.api_url}/help")
        response.raise_for_status()
        return response.json()
    
    def save_frame_to_file(self, output_path: str) -> bool:
        """Get the current frame and save it to a file.
        
        Args:
            output_path: Path to save the image to.
            
        Returns:
            True if successful, False otherwise.
        """
        import base64
        
        try:
            frame_data = self.get_frame()
            
            if not frame_data.get("success", False):
                logger.error(f"Failed to get frame: {frame_data.get('error', 'Unknown error')}")
                return False
            
            # Decode the base64 image
            img_data = base64.b64decode(frame_data["image"])
            
            # Write to file
            with open(output_path, "wb") as f:
                f.write(img_data)
            
            logger.info(f"Saved frame to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving frame to file: {e}")
            return False


def main():
    """Command-line interface for the Duck Robot client."""
    parser = argparse.ArgumentParser(description="Duck Robot Client")
    parser.add_argument("--api-url", type=str, default="http://localhost:5000", help="Base URL for the Duck Robot API")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get the status of the MuJoCo simulation")
    
    # Initialize command
    init_parser = subparsers.add_parser("initialize", help="Initialize the MuJoCo simulation")
    init_parser.add_argument("--onnx-model-path", type=str, required=True, help="Path to the ONNX model file")
    init_parser.add_argument("--model-path", type=str, help="Path to the XML model file")
    init_parser.add_argument("--reference-data", type=str, help="Path to the reference data file")
    init_parser.add_argument("--standing", action="store_true", help="Start in standing mode")
    
    # Shutdown command
    shutdown_parser = subparsers.add_parser("shutdown", help="Shutdown the MuJoCo simulation")
    
    # Direct command
    direct_parser = subparsers.add_parser("direct", help="Send a direct command to the Duck Robot")
    direct_parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity")
    direct_parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity")
    direct_parser.add_argument("--omega", type=float, default=0.0, help="Angular velocity")
    direct_parser.add_argument("--neck-pitch", type=float, default=0.0, help="Neck pitch")
    direct_parser.add_argument("--head-pitch", type=float, default=0.0, help="Head pitch")
    direct_parser.add_argument("--head-yaw", type=float, default=0.0, help="Head yaw")
    direct_parser.add_argument("--head-roll", type=float, default=0.0, help="Head roll")
    
    # Natural language command
    nl_parser = subparsers.add_parser("nl", help="Send a natural language command to the Duck Robot")
    nl_parser.add_argument("command", type=str, help="Natural language command")
    
    # Frame command
    frame_parser = subparsers.add_parser("frame", help="Get the current camera frame from the simulation")
    frame_parser.add_argument("--output", type=str, help="Path to save the frame to")
    
    # Help command
    help_parser = subparsers.add_parser("help", help="Get API documentation")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    client = DuckRobotClient(api_url=args.api_url)
    
    if args.command == "status":
        status = client.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == "initialize":
        result = client.initialize(
            onnx_model_path=args.onnx_model_path,
            model_path=args.model_path,
            reference_data=args.reference_data,
            standing=args.standing
        )
        print(json.dumps(result, indent=2))
    
    elif args.command == "shutdown":
        result = client.shutdown()
        print(json.dumps(result, indent=2))
    
    elif args.command == "direct":
        command = [
            args.vx,
            args.vy,
            args.omega,
            args.neck_pitch,
            args.head_pitch,
            args.head_yaw,
            args.head_roll
        ]
        result = client.send_direct_command(command)
        print(json.dumps(result, indent=2))
    
    elif args.command == "nl":
        result = client.send_nl_command(args.command)
        print(json.dumps(result, indent=2))
    
    elif args.command == "frame":
        if args.output:
            success = client.save_frame_to_file(args.output)
            if success:
                print(f"Frame saved to {args.output}")
            else:
                print("Failed to save frame")
        else:
            result = client.get_frame()
            print(f"Frame received, image data length: {len(result.get('image', ''))}")
    
    elif args.command == "help":
        help_info = client.get_api_help()
        print(json.dumps(help_info, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 