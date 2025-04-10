"""
Action module: Joystick interface for duck movement

This module provides an interface to control the duck in simulation by
translating commands into joystick-like inputs for MuJoCo.
"""

import logging
import time
import threading
import queue
import os
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class JoystickInterface:
    """
    Interface to the MuJoCo duck simulation.
    
    Translates high-level movement commands into joystick-like inputs
    that the simulation can understand.
    """
    
    def __init__(self, simulate: bool = True):
        """
        Initialize the joystick interface.
        
        Args:
            simulate: Whether to use simulation mode (always True for this class)
        """
        self.simulate = simulate
        if not simulate:
            logger.warning("Non-simulation mode requested, but this is a simulation-only interface")
            self.simulate = True
            
        logger.info("Initializing duck joystick interface")
        
        # Command queue for the MuJoCo thread
        self.command_queue = queue.Queue()
        
        # MuJoCo state
        self.mujoco_thread = None
        self.running = False
        
        # Movement parameters from MuJoCo simulation ranges
        # These match the ranges in the MuJoCo code
        self.COMMANDS_RANGE_X = [-0.15, 0.15]  # Forward/backward
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]    # Left/right
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # Turn
        
        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]
        
        # Current commands state
        self.current_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Configure debug mode from environment variable
        self.debug = os.environ.get("DUCK_DEBUG", "0") == "1"
        
        logger.debug("Joystick interface initialized")
    
    def start_mujoco_thread(self, onnx_model_path: str = None):
        """
        Start the MuJoCo thread if it's not already running.
        
        Args:
            onnx_model_path: Path to the ONNX model for the simulation
        """
        if self.running:
            logger.warning("MuJoCo thread already running")
            return False
        
        # Try to find a valid ONNX model
        if not onnx_model_path:
            logger.debug("No ONNX model specified, searching for provided models...")
            
            # List of potential model locations to try, in order of priority
            potential_paths = []
            
            # First check if env var is set (highest priority)
            if "DUCK_ONNX_MODEL" in os.environ:
                env_path = os.environ["DUCK_ONNX_MODEL"]
                logger.debug(f"Found DUCK_ONNX_MODEL environment variable: {env_path}")
                potential_paths.append(env_path)
            
            # Check duck_vla/onnx directory (second priority)
            duck_vla_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            onnx_dir = os.path.join(duck_vla_dir, "onnx")
            
            if os.path.exists(onnx_dir) and os.path.isdir(onnx_dir):
                logger.debug(f"Checking for ONNX models in {onnx_dir}")
                # List all .onnx files in this directory
                try:
                    for filename in os.listdir(onnx_dir):
                        if filename.endswith(".onnx"):
                            model_path = os.path.join(onnx_dir, filename)
                            potential_paths.append(model_path)
                            logger.debug(f"Found potential ONNX model: {model_path}")
                except Exception as e:
                    logger.error(f"Error listing files in {onnx_dir}: {e}")
            
            # Check workspace root directory
            workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            
            # Look for specific model names in workspace directory
            model_names = ["BEST_WALK_ONNX_2.onnx", "policy_1.onnx", "duck_model.onnx"]
            for name in model_names:
                potential_paths.append(os.path.join(workspace_dir, name))
            
            # Check for onnx files directly in workspace
            try:
                for filename in os.listdir(workspace_dir):
                    if filename.endswith(".onnx"):
                        model_path = os.path.join(workspace_dir, filename)
                        if model_path not in potential_paths:  # avoid duplicates
                            potential_paths.append(model_path)
                            logger.debug(f"Found potential ONNX model in workspace: {model_path}")
            except Exception as e:
                logger.error(f"Error listing files in workspace: {e}")
            
            # Check playground data directory as last resort
            playground_dir = os.path.join(workspace_dir, "submodules/open_duck_playground")
            playground_data_dir = os.path.join(playground_dir, "playground/open_duck_mini_v2/data")
            
            if os.path.exists(playground_data_dir) and os.path.isdir(playground_data_dir):
                potential_paths.append(os.path.join(playground_data_dir, "BEST_WALK_ONNX_2.onnx"))
                potential_paths.append(os.path.join(playground_data_dir, "policy_1.onnx"))
            
            # Try each path until we find one
            for path in potential_paths:
                if os.path.exists(path):
                    logger.info(f"Found ONNX model at: {path}")
                    onnx_model_path = path
                    break
                else:
                    logger.debug(f"ONNX model not found at: {path}")
        
        # Check if we found a model
        if not onnx_model_path or not os.path.exists(onnx_model_path):
            logger.error("Could not find a valid ONNX model. Please specify one with --onnx-model")
            logger.error("ONNX models should be placed in the duck_vla/onnx directory")
            return False
        
        logger.info(f"Starting MuJoCo thread with model: {onnx_model_path}")
        
        # Create and start the thread
        self.mujoco_thread = threading.Thread(
            target=self._run_mujoco_simulation,
            args=(onnx_model_path,),
            daemon=True
        )
        self.running = True
        self.mujoco_thread.start()
        return True
    
    def _run_mujoco_simulation(self, onnx_model_path: str):
        """
        Run the MuJoCo simulation in a separate thread.
        
        Args:
            onnx_model_path: Path to the ONNX model for the simulation
        """
        try:
            import sys
            import mujoco
            import numpy as np
            from playground.open_duck_mini_v2.mujoco_infer import MjInfer
            
            logger.info("Initializing MuJoCo simulation")
            
            # Set up default paths
            reference_data = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
            model_path = "playground/open_duck_mini_v2/xmls/open_duck_mini_v2_flat.xml"
            
            # Create the simulation object
            sim = MjInfer(model_path, reference_data, onnx_model_path, standing=False)
            
            # Main control loop
            while self.running:
                # Check for commands
                try:
                    # Non-blocking queue get
                    cmd = self.command_queue.get(block=False)
                    if cmd:
                        logger.debug(f"Processing command: {cmd}")
                        cmd_type = cmd.get("type")
                        
                        if cmd_type == "walking":
                            # Walking command: [lin_vel_x, lin_vel_y, ang_vel, ...]
                            sim.commands[0] = cmd.get("x", 0.0)
                            sim.commands[1] = cmd.get("y", 0.0)
                            sim.commands[2] = cmd.get("theta", 0.0)
                        
                        elif cmd_type == "head":
                            # Head command: [..., neck_pitch, head_pitch, head_yaw, head_roll]
                            sim.commands[3] = cmd.get("neck_pitch", 0.0)
                            sim.commands[4] = cmd.get("head_pitch", 0.0)
                            sim.commands[5] = cmd.get("head_yaw", 0.0)
                            sim.commands[6] = cmd.get("head_roll", 0.0)
                        
                        elif cmd_type == "reset":
                            # Reset the simulation
                            sim.reset()
                            
                        # Update current commands
                        self.current_commands = sim.commands.copy()
                            
                except queue.Empty:
                    # No commands, continue simulation
                    pass
                
                # Sleep to avoid burning CPU
                time.sleep(0.01)
                
        except ImportError as e:
            logger.error(f"Failed to import MuJoCo dependencies: {e}")
            logger.error("Make sure MuJoCo and Open Duck Playground are installed")
        except Exception as e:
            logger.exception(f"Error in MuJoCo thread: {e}")
        finally:
            logger.info("MuJoCo thread exiting")
            self.running = False
    
    def set_walking_params(self, x_vel: float, y_vel: float, yaw_vel: float) -> bool:
        """
        Set walking parameters.
        
        Args:
            x_vel: Forward velocity (-1.0 to 1.0, scaled to simulation limits)
            y_vel: Lateral velocity (-1.0 to 1.0, scaled to simulation limits)
            yaw_vel: Turning velocity (-1.0 to 1.0, scaled to simulation limits)
            
        Returns:
            Success flag
        """
        # Log command
        logger.debug(f"Setting walking params: x={x_vel}, y={y_vel}, yaw={yaw_vel}")
        
        # Clamp inputs to (-1, 1) range and scale to simulation limits
        x_scaled = self._scale_to_range(x_vel, self.COMMANDS_RANGE_X)
        y_scaled = self._scale_to_range(y_vel, self.COMMANDS_RANGE_Y)
        yaw_scaled = self._scale_to_range(yaw_vel, self.COMMANDS_RANGE_THETA)
        
        # Queue command for the MuJoCo thread
        self.command_queue.put({
            "type": "walking",
            "x": x_scaled,
            "y": y_scaled,
            "theta": yaw_scaled
        })
        
        # Simulation-only debug mode - log commands directly
        if self.debug:
            logger.info(f"Walking command: x={x_scaled}, y={y_scaled}, theta={yaw_scaled}")
            
        return True
    
    def set_head_position(self, yaw: float, pitch: float, roll: float) -> bool:
        """
        Set head position.
        
        Args:
            yaw: Head yaw angle in degrees (scaled to simulation limits)
            pitch: Head pitch angle in degrees (scaled to simulation limits)
            roll: Head roll angle in degrees (scaled to simulation limits)
            
        Returns:
            Success flag
        """
        # Log command
        logger.debug(f"Setting head position: yaw={yaw}, pitch={pitch}, roll={roll}")
        
        # Scale values to simulation ranges (convert from degrees to radians)
        neck_pitch = self._scale_to_range(pitch / 45.0, self.NECK_PITCH_RANGE)
        head_pitch = self._scale_to_range(pitch / 45.0, self.HEAD_PITCH_RANGE)
        head_yaw = self._scale_to_range(yaw / 45.0, self.HEAD_YAW_RANGE) 
        head_roll = self._scale_to_range(roll / 45.0, self.HEAD_ROLL_RANGE)
        
        # Queue command for the MuJoCo thread
        self.command_queue.put({
            "type": "head",
            "neck_pitch": neck_pitch,
            "head_pitch": head_pitch,
            "head_yaw": head_yaw,
            "head_roll": head_roll
        })
        
        # Simulation-only debug mode - log commands directly
        if self.debug:
            logger.info(f"Head command: neck_pitch={neck_pitch}, head_pitch={head_pitch}, head_yaw={head_yaw}, head_roll={head_roll}")
            
        return True
    
    def enable(self) -> bool:
        """
        Enable the joystick interface.
        
        Returns:
            Success flag
        """
        # Start the MuJoCo thread if not already running
        if not self.running:
            return self.start_mujoco_thread()
        return True
    
    def disable(self) -> bool:
        """
        Disable the joystick interface.
        
        Returns:
            Success flag
        """
        # Stop the MuJoCo thread
        if self.running:
            logger.info("Disabling joystick interface")
            self.running = False
            if self.mujoco_thread:
                self.mujoco_thread.join(timeout=1.0)
        return True
    
    def _scale_to_range(self, value: float, range_limits: List[float]) -> float:
        """
        Scale a value from (-1, 1) to the specified range.
        
        Args:
            value: Input value in (-1, 1) range
            range_limits: Target range as [min, max]
            
        Returns:
            Scaled value
        """
        # Clamp input to [-1, 1]
        clamped = max(-1.0, min(1.0, value))
        
        # Map [-1, 1] to [min, max]
        if clamped < 0:
            return -clamped * range_limits[0]  # Scale negative values to the negative portion of the range
        else:
            return clamped * range_limits[1]   # Scale positive values to the positive portion of the range 