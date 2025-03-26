import argparse
import logging
import os
import threading
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import flask
from flask import Flask, request, jsonify
import cv2
import base64
from flask_cors import CORS

# Import the mujoco inference machinery
from playground.open_duck_mini_v2.mujoco_infer import MjInfer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Global variables
mj_infer: Optional[MjInfer] = None
mj_infer_thread: Optional[threading.Thread] = None
running = False
latest_image = None
command_queue = []
current_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Default command vector
image_lock = threading.Lock()

def capture_frame(width: int = 640, height: int = 480) -> Optional[str]:
    """Capture the current frame from the MuJoCo viewer and encode as base64."""
    global latest_image, mj_infer
    
    if mj_infer is None:
        return None
    
    try:
        with image_lock:
            if latest_image is None:
                return None
            
            # Convert to RGB format for display
            img_rgb = cv2.cvtColor(latest_image, cv2.COLOR_BGR2RGB)
            
            # Resize the image if needed
            if img_rgb.shape[0] != height or img_rgb.shape[1] != width:
                img_rgb = cv2.resize(img_rgb, (width, height))
            
            # Encode the image as PNG and convert to base64
            _, buffer = cv2.imencode('.png', img_rgb)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            return img_str
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        logger.error(traceback.format_exc())
        return None

def update_commands(new_commands: List[float]) -> None:
    """Update the global command queue."""
    global command_queue
    command_queue.append(new_commands)
    
    # Keep only the last 5 commands
    if len(command_queue) > 5:
        command_queue.pop(0)

def mj_thread_function() -> None:
    """Thread function to run the MuJoCo simulation."""
    global mj_infer, running, latest_image, command_queue, current_command
    
    logger.info("Starting MuJoCo thread")
    
    try:
        while running:
            # Update the commands if there are any in the queue
            if command_queue:
                current_command = command_queue.pop(0)
                mj_infer.commands = current_command
                logger.debug(f"Updated command to: {current_command}")
            
            # Try to grab the latest frame
            img = mj_infer.data.render("rgb", 640, 480, camera="RearQuad")
            with image_lock:
                latest_image = img
            
            # Step the simulation
            mj_infer.step()
            
            # Sleep to maintain a reasonable frame rate
            time.sleep(0.01)
    except Exception as e:
        running = False
        logger.error(f"Error in MuJoCo thread: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("MuJoCo thread stopped")

def initialize_mujoco(
    model_path: str, 
    reference_data: str, 
    onnx_model_path: str, 
    standing: bool = False
) -> bool:
    """Initialize the MuJoCo simulation."""
    global mj_infer, mj_infer_thread, running
    
    try:
        mj_infer = MjInfer(
            model_path=model_path,
            reference_data=reference_data,
            onnx_model_path=onnx_model_path,
            standing=standing
        )
        
        # Start the MuJoCo thread
        running = True
        mj_infer_thread = threading.Thread(target=mj_thread_function)
        mj_infer_thread.daemon = True
        mj_infer_thread.start()
        
        logger.info("MuJoCo initialization successful")
        return True
    except Exception as e:
        logger.error(f"Error initializing MuJoCo: {e}")
        logger.error(traceback.format_exc())
        return False

def shutdown_mujoco() -> None:
    """Shutdown the MuJoCo simulation."""
    global mj_infer, mj_infer_thread, running
    
    running = False
    
    if mj_infer_thread is not None:
        mj_infer_thread.join(timeout=2.0)
        mj_infer_thread = None
    
    mj_infer = None
    logger.info("MuJoCo shutdown complete")

# API routes
@app.route('/status', methods=['GET'])
def get_status() -> flask.Response:
    """Get the status of the MuJoCo simulation."""
    global mj_infer, running
    
    status = {
        "running": running,
        "initialized": mj_infer is not None,
        "current_command": current_command
    }
    
    return jsonify(status)

@app.route('/initialize', methods=['POST'])
def api_initialize_mujoco() -> flask.Response:
    """Initialize the MuJoCo simulation."""
    global mj_infer
    
    if mj_infer is not None:
        return jsonify({"success": False, "error": "MuJoCo already initialized"})
    
    data = request.json
    model_path = data.get('model_path', 'playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml')
    reference_data = data.get('reference_data', 'playground/open_duck_mini_v2/data/polynomial_coefficients.pkl')
    onnx_model_path = data.get('onnx_model_path', 'model.onnx')
    standing = data.get('standing', False)
    
    if not os.path.exists(onnx_model_path):
        return jsonify({"success": False, "error": f"ONNX model file not found: {onnx_model_path}"})
    
    success = initialize_mujoco(model_path, reference_data, onnx_model_path, standing)
    
    return jsonify({"success": success})

@app.route('/shutdown', methods=['POST'])
def api_shutdown_mujoco() -> flask.Response:
    """Shutdown the MuJoCo simulation."""
    global mj_infer
    
    if mj_infer is None:
        return jsonify({"success": False, "error": "MuJoCo not initialized"})
    
    shutdown_mujoco()
    
    return jsonify({"success": True})

@app.route('/command', methods=['POST'])
def api_send_command() -> flask.Response:
    """Send a command to the MuJoCo simulation."""
    global mj_infer
    
    if mj_infer is None:
        return jsonify({"success": False, "error": "MuJoCo not initialized"})
    
    data = request.json
    command_type = data.get('command_type', 'direct')
    
    if command_type == 'direct':
        # Format expected: [vx, vy, omega, neck_pitch, head_pitch, head_yaw, head_roll]
        command = data.get('command', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Validate command
        if not isinstance(command, list) or len(command) != 7:
            return jsonify({
                "success": False, 
                "error": "Command must be a list of 7 floats: [vx, vy, omega, neck_pitch, head_pitch, head_yaw, head_roll]"
            })
        
        update_commands(command)
        return jsonify({"success": True})
    
    elif command_type == 'natural_language':
        # Process natural language commands
        nl_command = data.get('nl_command', '')
        
        # Simple mapping of commands to actions
        if 'forward' in nl_command.lower():
            cmd = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Move forward
        elif 'backward' in nl_command.lower():
            cmd = [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Move backward
        elif 'left' in nl_command.lower():
            if 'turn' in nl_command.lower():
                cmd = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]  # Turn left
            else:
                cmd = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]  # Strafe left
        elif 'right' in nl_command.lower():
            if 'turn' in nl_command.lower():
                cmd = [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]  # Turn right
            else:
                cmd = [0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0]  # Strafe right
        elif 'look up' in nl_command.lower():
            cmd = [0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0]  # Look up
        elif 'look down' in nl_command.lower():
            cmd = [0.0, 0.0, 0.0, -0.3, 0.5, 0.0, 0.0]  # Look down
        elif 'look left' in nl_command.lower():
            cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0]  # Look left
        elif 'look right' in nl_command.lower():
            cmd = [0.0, 0.0, 0.0, 0.0, 0.0, -0.8, 0.0]  # Look right
        elif 'stop' in nl_command.lower():
            cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Stop
        else:
            return jsonify({
                "success": False,
                "error": f"Could not parse natural language command: {nl_command}"
            })
        
        update_commands(cmd)
        return jsonify({
            "success": True,
            "parsed_command": cmd,
            "nl_command": nl_command
        })
    
    else:
        return jsonify({
            "success": False,
            "error": f"Unknown command type: {command_type}"
        })

@app.route('/frame', methods=['GET'])
def api_get_frame() -> flask.Response:
    """Get the current frame from the MuJoCo viewer."""
    global mj_infer
    
    if mj_infer is None:
        return jsonify({"success": False, "error": "MuJoCo not initialized"})
    
    # Get the current frame
    img_str = capture_frame()
    
    if img_str is None:
        return jsonify({"success": False, "error": "Failed to capture frame"})
    
    return jsonify({
        "success": True,
        "image": img_str,
        "timestamp": time.time()
    })

@app.route('/help', methods=['GET'])
def api_help() -> flask.Response:
    """Get API documentation."""
    docs = {
        "endpoints": [
            {
                "path": "/status",
                "method": "GET",
                "description": "Get the status of the MuJoCo simulation",
                "parameters": None
            },
            {
                "path": "/initialize",
                "method": "POST",
                "description": "Initialize the MuJoCo simulation",
                "parameters": {
                    "model_path": "Path to the XML model file (optional)",
                    "reference_data": "Path to the reference data file (optional)",
                    "onnx_model_path": "Path to the ONNX model file (required)",
                    "standing": "Whether to start in standing mode (optional, default: false)"
                }
            },
            {
                "path": "/shutdown",
                "method": "POST",
                "description": "Shutdown the MuJoCo simulation",
                "parameters": None
            },
            {
                "path": "/command",
                "method": "POST",
                "description": "Send a command to the duck robot",
                "parameters": {
                    "command_type": "Type of command: 'direct' or 'natural_language'",
                    "command": "For 'direct' type: Array of 7 floats [vx, vy, omega, neck_pitch, head_pitch, head_yaw, head_roll]",
                    "nl_command": "For 'natural_language' type: A string describing the desired action"
                }
            },
            {
                "path": "/frame",
                "method": "GET",
                "description": "Get the current camera frame from the simulation",
                "parameters": None
            }
        ]
    }
    
    return jsonify(docs)

def main():
    parser = argparse.ArgumentParser(description="Duck Robot API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Duck Robot API Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
