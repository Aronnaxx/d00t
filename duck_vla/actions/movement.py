# This is a motion controller for the duck_vla project

# If we are running in simulation mode, then we will pass arrow keys as incoming commands
# if we are running in real mode, then we will pass the actual commands as an xbox controller joystick

# For simulator mode we should reference submodules/open_duck_playground/playground/open_duck_mini_v2/mujoco_infer.py for how
# the keybinds are used and for what (e.g. arrow keys for forward, backward, left, right)

# For the real version we should reference submodules/open_duck_mini_runtime/scripts/v2_rl_walk_mujoco.py
# and pass the commands to the runtime instead of using the joystick / xbox controller

import logging
import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class Movement:
    """
    Controls movement for the Duck robot.
    
    This class provides an abstraction over both the simulated and real robot,
    allowing unified control through the same interface.
    """
    
    # Command ranges from the reference implementation
    COMMAND_RANGES = {
        "x": [-0.15, 0.15],      # Linear velocity X (forward/backward)
        "y": [-0.2, 0.2],        # Linear velocity Y (left/right)
        "yaw": [-1.0, 1.0],      # Angular velocity (rotation)
        "neck_pitch": [-0.34, 1.1],
        "head_pitch": [-0.78, 0.78],
        "head_yaw": [-1.5, 1.5],
        "head_roll": [-0.5, 0.5],
    }
    
    def __init__(self, simulated: bool = True, debug_logging: bool = False):
        """
        Initialize the movement controller.
        
        Args:
            simulated: Whether to use simulation or real hardware
            debug_logging: Enable debug logging
        """
        self.simulated = simulated
        self.debug = debug_logging
        
        if debug_logging:
            logger.setLevel(logging.DEBUG)
        
        # Current command values (zeroed by default)
        self.current_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Track if controller is initialized
        self.initialized = False
        self._init_controller()
        
        logger.info(f"Movement controller initialized in {'simulation' if simulated else 'real'} mode")
    
    def _init_controller(self):
        """Initialize either the simulated or real controller"""
        if self.simulated:
            self._init_simulation()
        else:
            self._init_real_hardware()
    
    def _init_simulation(self):
        """Initialize the simulated controller (mujoco-based)"""
        try:
            # We're not actually initializing Mujoco here, as that would be done
            # by the simulation environment. Instead, we're just setting up the
            # interface to communicate commands to it.
            logger.info("Initialized simulated movement controller")
            self.initialized = True
            
            # When a key is pressed, we'll record it here
            self.last_key = None
            
        except Exception as e:
            logger.error(f"Failed to initialize simulated controller: {e}")
            self.initialized = False
    
    def _init_real_hardware(self):
        """Initialize the real hardware controller"""
        try:
            # Check if we have the required hardware libraries
            # If they're not available, we can still operate but without real control
            try:
                # Import necessary modules for the real robot
                # These imports would normally come from the robot's runtime
                logger.debug("Checking for hardware control libraries")
                
                # This would be the actual import in a real implementation
                # from mini_bdx_runtime.rustypot_position_hwi import HWI
                
                # For now, we're just simulating the availability
                real_hardware_available = os.path.exists("/dev/ttyACM0")  # Common port for Arduino/microcontroller
                
                if not real_hardware_available:
                    logger.warning("No hardware detected at expected port, running in mock mode")
                
                self.initialized = True
                logger.info("Initialized real hardware movement controller")
                
            except ImportError as e:
                logger.warning(f"Hardware control libraries not available: {e}")
                logger.warning("Running in mock hardware mode")
                self.initialized = True
                
        except Exception as e:
            logger.error(f"Failed to initialize real hardware controller: {e}")
            self.initialized = False
    
    def set_movement_command(self, 
                          linear_x: float = 0.0, 
                          linear_y: float = 0.0, 
                          angular_z: float = 0.0) -> bool:
        """
        Set movement command values.
        
        Args:
            linear_x: Forward/backward velocity (-1.0 to 1.0)
            linear_y: Left/right velocity (-1.0 to 1.0)
            angular_z: Angular velocity/rotation (-1.0 to 1.0)
            
        Returns:
            True if command was set successfully
        """
        if not self.initialized:
            logger.warning("Movement controller not initialized")
            return False
        
        # Normalize values to the expected ranges
        x_value = max(self.COMMAND_RANGES["x"][0], min(self.COMMAND_RANGES["x"][1], linear_x))
        y_value = max(self.COMMAND_RANGES["y"][0], min(self.COMMAND_RANGES["y"][1], linear_y))
        yaw_value = max(self.COMMAND_RANGES["yaw"][0], min(self.COMMAND_RANGES["yaw"][1], angular_z))
        
        # Update command values
        self.current_commands[0] = x_value
        self.current_commands[1] = y_value
        self.current_commands[2] = yaw_value
        
        if self.debug:
            logger.debug(f"Set movement command: x={x_value}, y={y_value}, yaw={yaw_value}")
        
        return True
    
    def set_head_position(self,
                       neck_pitch: float = 0.0,
                       head_pitch: float = 0.0,
                       head_yaw: float = 0.0,
                       head_roll: float = 0.0) -> bool:
        """
        Set head position values.
        
        Args:
            neck_pitch: Neck pitch angle
            head_pitch: Head pitch angle
            head_yaw: Head yaw angle
            head_roll: Head roll angle
            
        Returns:
            True if command was set successfully
        """
        if not self.initialized:
            logger.warning("Movement controller not initialized")
            return False
        
        # Normalize values to the expected ranges
        neck_pitch_value = max(self.COMMAND_RANGES["neck_pitch"][0], 
                              min(self.COMMAND_RANGES["neck_pitch"][1], neck_pitch))
        head_pitch_value = max(self.COMMAND_RANGES["head_pitch"][0], 
                              min(self.COMMAND_RANGES["head_pitch"][1], head_pitch))
        head_yaw_value = max(self.COMMAND_RANGES["head_yaw"][0], 
                            min(self.COMMAND_RANGES["head_yaw"][1], head_yaw))
        head_roll_value = max(self.COMMAND_RANGES["head_roll"][0], 
                             min(self.COMMAND_RANGES["head_roll"][1], head_roll))
        
        # Update command values
        self.current_commands[3] = neck_pitch_value
        self.current_commands[4] = head_pitch_value
        self.current_commands[5] = head_yaw_value
        self.current_commands[6] = head_roll_value
        
        if self.debug:
            logger.debug(f"Set head position: neck_pitch={neck_pitch_value}, head_pitch={head_pitch_value}, "
                        f"head_yaw={head_yaw_value}, head_roll={head_roll_value}")
        
        return True
    
    def move_forward(self, speed: float = 0.5, duration: float = 1.0) -> bool:
        """
        Move forward at a specified speed for a specified duration.
        
        Args:
            speed: Movement speed (0.0 to 1.0)
            duration: Duration in seconds
            
        Returns:
            True if movement started successfully
        """
        # Scale speed to the robot's forward velocity range
        scaled_speed = speed * self.COMMAND_RANGES["x"][1]
        
        # Set the movement command
        success = self.set_movement_command(linear_x=scaled_speed)
        
        # If requested and command was set successfully, wait for the specified duration
        if success and duration > 0:
            time.sleep(duration)
            # Stop movement after duration
            self.set_movement_command(linear_x=0.0)
        
        return success
    
    def move_backward(self, speed: float = 0.5, duration: float = 1.0) -> bool:
        """
        Move backward at a specified speed for a specified duration.
        
        Args:
            speed: Movement speed (0.0 to 1.0)
            duration: Duration in seconds
            
        Returns:
            True if movement started successfully
        """
        # Scale speed to the robot's backward velocity range (negative for backward)
        scaled_speed = -speed * abs(self.COMMAND_RANGES["x"][0])
        
        # Set the movement command
        success = self.set_movement_command(linear_x=scaled_speed)
        
        # If requested and command was set successfully, wait for the specified duration
        if success and duration > 0:
            time.sleep(duration)
            # Stop movement after duration
            self.set_movement_command(linear_x=0.0)
        
        return success
    
    def turn_left(self, speed: float = 0.5, duration: float = 1.0) -> bool:
        """
        Turn left at a specified speed for a specified duration.
        
        Args:
            speed: Turn speed (0.0 to 1.0)
            duration: Duration in seconds
            
        Returns:
            True if turn started successfully
        """
        # Scale speed to the robot's angular velocity range
        scaled_speed = speed * self.COMMAND_RANGES["yaw"][1]
        
        # Set the movement command
        success = self.set_movement_command(angular_z=scaled_speed)
        
        # If requested and command was set successfully, wait for the specified duration
        if success and duration > 0:
            time.sleep(duration)
            # Stop movement after duration
            self.set_movement_command(angular_z=0.0)
        
        return success
    
    def turn_right(self, speed: float = 0.5, duration: float = 1.0) -> bool:
        """
        Turn right at a specified speed for a specified duration.
        
        Args:
            speed: Turn speed (0.0 to 1.0)
            duration: Duration in seconds
            
        Returns:
            True if turn started successfully
        """
        # Scale speed to the robot's angular velocity range (negative for right)
        scaled_speed = -speed * abs(self.COMMAND_RANGES["yaw"][1])
        
        # Set the movement command
        success = self.set_movement_command(angular_z=scaled_speed)
        
        # If requested and command was set successfully, wait for the specified duration
        if success and duration > 0:
            time.sleep(duration)
            # Stop movement after duration
            self.set_movement_command(angular_z=0.0)
        
        return success
    
    def stop(self) -> bool:
        """
        Stop all movement.
        
        Returns:
            True if stop command was sent successfully
        """
        # Set all movement commands to zero
        success = self.set_movement_command(0.0, 0.0, 0.0)
        
        return success
    
    def look_up(self, amount: float = 0.5) -> bool:
        """
        Look upward.
        
        Args:
            amount: How much to look up (0.0 to 1.0)
            
        Returns:
            True if head position was set successfully
        """
        # Scale amount to the robot's head pitch range
        scaled_amount = amount * self.COMMAND_RANGES["head_pitch"][1]
        
        # Set the head position
        return self.set_head_position(head_pitch=scaled_amount)
    
    def look_down(self, amount: float = 0.5) -> bool:
        """
        Look downward.
        
        Args:
            amount: How much to look down (0.0 to 1.0)
            
        Returns:
            True if head position was set successfully
        """
        # Scale amount to the robot's head pitch range (negative for down)
        scaled_amount = -amount * abs(self.COMMAND_RANGES["head_pitch"][0])
        
        # Set the head position
        return self.set_head_position(head_pitch=scaled_amount)
    
    def look_left(self, amount: float = 0.5) -> bool:
        """
        Look to the left.
        
        Args:
            amount: How much to look left (0.0 to 1.0)
            
        Returns:
            True if head position was set successfully
        """
        # Scale amount to the robot's head yaw range
        scaled_amount = amount * self.COMMAND_RANGES["head_yaw"][1]
        
        # Set the head position
        return self.set_head_position(head_yaw=scaled_amount)
    
    def look_right(self, amount: float = 0.5) -> bool:
        """
        Look to the right.
        
        Args:
            amount: How much to look right (0.0 to 1.0)
            
        Returns:
            True if head position was set successfully
        """
        # Scale amount to the robot's head yaw range (negative for right)
        scaled_amount = -amount * abs(self.COMMAND_RANGES["head_yaw"][0])
        
        # Set the head position
        return self.set_head_position(head_yaw=scaled_amount)
    
    def reset_head(self) -> bool:
        """
        Reset head position to default.
        
        Returns:
            True if head position was reset successfully
        """
        # Set all head position values to zero
        return self.set_head_position(0.0, 0.0, 0.0, 0.0)
    
    def get_current_command(self) -> List[float]:
        """
        Get the current command values.
        
        Returns:
            List of current command values [x, y, yaw, neck_pitch, head_pitch, head_yaw, head_roll]
        """
        return self.current_commands
    
    def simulate_key_press(self, key_code: int) -> None:
        """
        Simulate a key press for the simulated environment.
        
        This is used to simulate keyboard input for the Mujoco simulator,
        using the same key codes as in mujoco_infer.py.
        
        Args:
            key_code: Key code to simulate
        """
        if not self.simulated:
            logger.warning("Key simulation only available in simulation mode")
            return
        
        self.last_key = key_code
        
        # Map key codes to commands as in mujoco_infer.py
        if key_code == 265:  # arrow up
            self.move_forward()
        elif key_code == 264:  # arrow down
            self.move_backward()
        elif key_code == 263:  # arrow left
            self.turn_left()
        elif key_code == 262:  # arrow right
            self.turn_right()
        elif key_code == 72:  # h - toggle head control mode
            # This would toggle head control mode in the simulation
            pass
        else:
            logger.debug(f"Unmapped key code: {key_code}")
    
    def cleanup(self) -> None:
        """Clean up resources when shutting down"""
        # Stop all movement
        self.stop()
        # Reset head position
        self.reset_head()
        logger.info("Movement controller cleaned up")