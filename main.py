#!/usr/bin/env python3
# main.py - Main entry point for robot simulation
"""
This script loads an OpenDuckMini robot into a warehouse environment and controls it using a trained ONNX model.
It also allows for keyboard/mouse control.

Usage:
    uv run main.py --robot_usd=assets/robot.usd --model_file=path/to/model.onnx
"""

import os
import time
import logging
import argparse
import numpy as np
import torch

# Set up logging with file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("robot_sim.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="OpenDuckMini in warehouse with ONNX model")
# Add app launcher arguments
AppLauncher.add_app_launcher_args(parser)
# Add custom arguments
parser.add_argument("--robot_usd", type=str, default="assets/robot.usd", 
                   help="Path to robot USD file")
parser.add_argument("--model_file", type=str, default="model.onnx",
                   help="Path to ONNX model file")
parser.add_argument("--duration", type=float, default=300.0,
                   help="Duration to run the demo in seconds")
# Parse arguments
args = parser.parse_args()

# Launch the simulator app
logger.info("Launching Isaac Sim...")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest of the application follows."""
# Import modules after simulation is started
import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from perception.model_inference import load_onnx_model, preprocess_image, run_inference, create_dummy_onnx_model
from simulation.environment import load_warehouse_stage, initialize_world
from simulation.robot_loader import load_robot
from control.controller import get_robot_articulation, command_differential_drive

class RobotDemo:
    """Main class for OpenDuckMini demo with ONNX model and keyboard control."""
    
    def __init__(self, robot_usd_path="assets/robot.usd", model_path="model.onnx"):
        """Initialize the demo.
        
        Args:
            robot_usd_path: Path to the robot USD file
            model_path: Path to the ONNX model file
        """
        logger.info("Initializing OpenDuckMini demo")
        
        # Load the warehouse stage
        logger.info("Loading warehouse environment")
        load_warehouse_stage()
        
        # Initialize the simulation world
        logger.info("Initializing simulation world")
        self.env = initialize_world()
        
        # Create dummy ONNX model if needed
        if not os.path.exists(model_path):
            logger.warning(f"ONNX model not found at {model_path}, creating a dummy model")
            create_dummy_onnx_model(model_path)
        
        # Load the ONNX model
        logger.info(f"Loading ONNX model from {model_path}")
        self.model_session, self.input_name, self.input_shape = load_onnx_model(model_path)
        if self.model_session is None:
            logger.error("Failed to load ONNX model")
        
        # Load the robot
        logger.info(f"Loading robot from {robot_usd_path}")
        if not os.path.exists(robot_usd_path):
            logger.warning(f"Robot USD file not found at {robot_usd_path}, check if create_test_robot.py exists")
            try:
                from create_test_robot import main as create_robot
                create_robot()
            except ImportError:
                logger.error("Could not import create_test_robot.py, robot loading may fail")
        
        # Load the robot into the simulation
        self.robot_loaded = load_robot(
            robot_usd_path,
            prim_path="/World/Robot", 
            translation=(0, 0, 0.1),  # Slight elevation to avoid ground penetration
            orientation=(0, 0, 0, 1)   # Default orientation
        )
        
        if not self.robot_loaded:
            logger.error("Failed to load robot, demo may not work correctly")
        
        # Wheel parameters for the differential drive controller
        self.wheel_params = {
            "left_joint_name": "left_wheel_joint",
            "right_joint_name": "right_wheel_joint",
            "axle_length": 0.5,     # Distance between wheels in meters
            "wheel_radius": 0.1     # Wheel radius in meters
        }
        
        # Get robot articulation for control
        logger.info("Setting up robot control")
        self.dc_interface, self.robot_art = get_robot_articulation("/World/Robot")
        
        if self.dc_interface is None or self.robot_art is None:
            logger.error("Failed to get robot articulation, control will not work")
        
        # Initialize camera if needed for perception
        try:
            from perception.camera import initialize_camera, get_camera_frame
            
            self.camera = initialize_camera(
                prim_path="/World/Robot/CameraSensor",
                resolution=(640, 480),
                position=[0.2, 0, 0.2],  # Forward and up from robot center
                rotation=[0, -20, 0]     # Tilt down slightly
            )
            
            if self.camera is not None:
                logger.info("Camera initialized successfully")
                self.get_camera_frame = get_camera_frame
            else:
                logger.warning("Failed to initialize camera")
                self.get_camera_frame = None
        except Exception as e:
            logger.error(f"Error setting up camera: {e}")
            self.camera = None
            self.get_camera_frame = None
        
        # Set up keyboard/mouse control
        self.setup_keyboard_control()
        
        # Initialize control parameters
        self.use_model_control = True  # Toggle between model and keyboard control
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        # Step the simulation a few times to ensure everything is initialized
        logger.info("Initializing simulation...")
        for _ in range(10):
            self.env.step(None)
    
    def setup_keyboard_control(self):
        """Set up keyboard control for the robot."""
        logger.info("Setting up keyboard control")
        
        # Set up keyboard interface
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        
        # Define key mappings 
        self.key_to_control = {
            "UP": (0.5, 0.0),      # Forward
            "DOWN": (-0.3, 0.0),    # Backward
            "LEFT": (0.0, 0.5),     # Turn left
            "RIGHT": (0.0, -0.5),   # Turn right
            "w": (0.5, 0.0),        # Forward (WASD alternative)
            "s": (-0.3, 0.0),       # Backward (WASD alternative)
            "a": (0.0, 0.5),        # Turn left (WASD alternative)
            "d": (0.0, -0.5),       # Turn right (WASD alternative)
            "SPACE": (0.0, 0.0)     # Stop
        }
        
        # Tracking currently pressed keys
        self.active_keys = set()
        
        logger.info("Keyboard controls:")
        logger.info("  Arrow keys / WASD: Move robot")
        logger.info("  Space: Stop robot")
        logger.info("  M: Toggle between model control and keyboard control")
        logger.info("  ESC: Exit demo")
    
    def _on_keyboard_event(self, event):
        """Handle keyboard events."""
        # Key press event
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Add key to active keys set
            self.active_keys.add(event.input.name)
            
            # Handle special keys
            if event.input.name == "ESCAPE":
                logger.info("ESC pressed, exiting demo")
                simulation_app.close()
            elif event.input.name == "m":
                # Toggle between model and keyboard control
                self.use_model_control = not self.use_model_control
                logger.info(f"Switched to {'model' if self.use_model_control else 'keyboard'} control")
        
        # Key release event
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # Remove key from active keys
            if event.input.name in self.active_keys:
                self.active_keys.remove(event.input.name)
    
    def update_keyboard_control(self):
        """Update robot control based on currently pressed keys."""
        # Default to zero velocity
        linear_vel = 0.0
        angular_vel = 0.0
        
        # Process all active keys (allowing for combinations)
        for key in self.active_keys:
            if key in self.key_to_control:
                lin, ang = self.key_to_control[key]
                linear_vel += lin
                angular_vel += ang
        
        # Update control values
        self.linear_velocity = linear_vel
        self.angular_velocity = angular_vel
    
    def run_demo(self, duration=60.0):
        """Run the demo for the specified duration.
        
        Args:
            duration: Time in seconds to run the demo
        """
        logger.info(f"Starting demo for {duration} seconds")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while simulation_app.is_running() and (time.time() - start_time) < duration:
                # Get simulation time
                sim_time = time.time() - start_time
                
                # Update control from keyboard if using keyboard control
                if not self.use_model_control:
                    self.update_keyboard_control()
                
                # If using model control and camera is available, use ONNX model for control
                elif self.use_model_control and self.camera is not None and self.get_camera_frame is not None and self.model_session is not None:
                    # Capture camera frame
                    frame = self.get_camera_frame(self.camera)
                    if frame is not None:
                        # Run ONNX inference
                        processed_frame = preprocess_image(frame)
                        v_lin, v_ang = run_inference(self.model_session, self.input_name, processed_frame)
                        self.linear_velocity = v_lin
                        self.angular_velocity = v_ang
                
                # Print command occasionally
                if frame_count % 30 == 0:
                    control_mode = "Model" if self.use_model_control else "Keyboard"
                    logger.info(f"[{sim_time:.2f}s] {control_mode} control: v_lin={self.linear_velocity:.2f} m/s, v_ang={self.angular_velocity:.2f} rad/s")
                
                # Command the robot if control is available
                if self.dc_interface is not None and self.robot_art is not None:
                    command_differential_drive(
                        self.dc_interface,
                        self.robot_art,
                        self.linear_velocity,
                        self.angular_velocity,
                        self.wheel_params
                    )
                
                # Step the simulation
                self.env.step(None)
                frame_count += 1
                
                # Small sleep to avoid maxing CPU
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Error in simulation loop: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info("Demo completed")


def main():
    """Main function."""
    logger.info("Starting main function")
    
    # Create and run the demo
    demo = RobotDemo(args.robot_usd, args.model_file)
    demo.run_demo(duration=args.duration)
    
    logger.info("Main function completed")
    simulation_app.close()


if __name__ == "__main__":
    main()
