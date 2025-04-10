"""
Action module: Motion controller for duck movement

This module provides an interface to control duck movements by translating
high-level commands into joystick-like movement controls.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class MotionController:
    """
    Controls duck movement by interfacing with the joystick control system.
    
    This wraps the existing joystick interface for the duck droid and
    provides higher-level movement commands.
    """
    
    def __init__(self, simulate: bool = False):
        """
        Initialize the motion controller.
        
        Args:
            simulate: Whether to use simulation mode
        """
        self.simulate = simulate
        logger.info(f"Initializing motion controller (simulate={simulate})")
        
        # Movement state tracking
        self.current_speed = 0.0
        self.current_turn_rate = 0.0
        self.current_strafe_rate = 0.0
        self.current_head_position = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        
        # Movement limits
        self.max_speed = 1.0
        self.max_turn_rate = 1.0
        self.max_strafe_rate = 1.0
        self.head_limits = {
            "yaw": (-45.0, 45.0),
            "pitch": (-30.0, 30.0),
            "roll": (-20.0, 20.0)
        }
        
        # Performance tracking
        self.last_command_time = 0
        self.command_count = 0
        
        # Import the appropriate joystick interface
        self._initialize_interface()
    
    def _initialize_interface(self) -> None:
        """Initialize the appropriate joystick interface."""
        try:
            # Import our JoystickInterface wrapper
            from duck_vla.action.joystick_interface import JoystickInterface
            
            # Create joystick interface with simulation flag
            self.joystick = JoystickInterface(simulate=self.simulate)
            logger.info(f"Initialized joystick interface (simulate={self.simulate})")
            
        except Exception as e:
            logger.exception(f"Error initializing joystick interface: {e}")
            # Create a mock interface as fallback
            logger.warning("Using mock joystick interface as fallback")
            self.joystick = self._create_mock_joystick()
    
    def _create_mock_joystick(self) -> object:
        """Create a mock joystick interface for testing."""
        logger.debug("Creating mock joystick interface")
        
        class MockJoystick:
            """Mock joystick interface that logs commands."""
            
            def __init__(self):
                self.logger = logging.getLogger("MockJoystick")
                self.logger.info("Mock joystick created")
            
            def set_walking_params(self, x_vel, y_vel, yaw_vel):
                self.logger.info(f"Walking: x_vel={x_vel:.2f}, y_vel={y_vel:.2f}, yaw_vel={yaw_vel:.2f}")
                return True
            
            def set_head_position(self, yaw, pitch, roll):
                self.logger.info(f"Head: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")
                return True
            
            def enable(self):
                self.logger.info("Joystick enabled")
                return True
            
            def disable(self):
                self.logger.info("Joystick disabled")
                return True
                
        return MockJoystick()
    
    def move(
        self, 
        direction: str = "forward", 
        speed: float = 0.5,
        duration: Optional[float] = None
    ) -> bool:
        """
        Move the duck in a specified direction.
        
        Args:
            direction: Movement direction ('forward', 'backward', 'left', 'right')
            speed: Movement speed as fraction of maximum (0.0-1.0)
            duration: Movement duration in seconds or None for continuous
            
        Returns:
            Success flag
        """
        # Validate and normalize inputs
        speed = max(0.0, min(speed, 1.0))
        direction = direction.lower()
        
        logger.info(f"Moving {direction} at speed {speed:.2f}" + 
                   (f" for {duration:.2f}s" if duration else " continuously"))
        
        try:
            # Convert direction to joystick parameters
            x_vel, y_vel, yaw_vel = 0.0, 0.0, 0.0
            
            if direction == "forward":
                x_vel = speed
            elif direction == "backward":
                x_vel = -speed
            elif direction == "left":
                y_vel = speed
            elif direction == "right":
                y_vel = -speed
            else:
                logger.warning(f"Unknown direction: {direction}")
                return False
            
            # Update state
            self.current_speed = speed if x_vel != 0 else 0.0
            self.current_strafe_rate = abs(y_vel) if y_vel != 0 else 0.0
            self.last_command_time = time.time()
            self.command_count += 1
            
            # Execute movement through joystick interface
            success = self.joystick.set_walking_params(x_vel, y_vel, yaw_vel)
            
            # If duration specified, schedule stop
            if duration is not None and success:
                # In a more complex implementation, you might use a timer thread
                # For this example, we'll simulate by sleeping and then stopping
                if not self.simulate:  # Only sleep in real mode
                    time.sleep(duration)
                    self.stop()
            
            return success
            
        except Exception as e:
            logger.exception(f"Error during movement: {e}")
            return False
    
    def turn(self, direction: str = "left", rate: float = 0.5, angle: Optional[float] = None) -> bool:
        """
        Turn the duck.
        
        Args:
            direction: Turn direction ('left', 'right', 'around')
            rate: Turn rate as fraction of maximum (0.0-1.0)
            angle: Turn angle in degrees or None for continuous
            
        Returns:
            Success flag
        """
        # Validate and normalize inputs
        rate = max(0.0, min(rate, 1.0))
        direction = direction.lower()
        
        angle_str = f" by {angle:.1f}°" if angle is not None else " continuously"
        logger.info(f"Turning {direction} at rate {rate:.2f}{angle_str}")
        
        try:
            # Convert direction to joystick parameters
            x_vel, y_vel, yaw_vel = 0.0, 0.0, 0.0
            
            if direction == "left":
                yaw_vel = rate
            elif direction == "right":
                yaw_vel = -rate
            elif direction == "around":
                # For "around", we turn left at max rate
                yaw_vel = 1.0
                # And if no angle specified, use 180 degrees
                if angle is None:
                    angle = 180.0
            else:
                logger.warning(f"Unknown turn direction: {direction}")
                return False
            
            # Update state
            self.current_turn_rate = abs(yaw_vel) if yaw_vel != 0 else 0.0
            self.last_command_time = time.time()
            self.command_count += 1
            
            # Execute turn through joystick interface
            success = self.joystick.set_walking_params(x_vel, y_vel, yaw_vel)
            
            # If angle specified, calculate duration and schedule stop
            if angle is not None and success:
                # Very rough approximation:
                # At max rate (1.0), duck turns at about 90 degrees/second
                max_turn_rate_deg_per_sec = 90.0
                turn_time = angle / (max_turn_rate_deg_per_sec * abs(yaw_vel))
                
                logger.debug(f"Turn time for {angle:.1f}° at rate {abs(yaw_vel):.2f}: {turn_time:.2f}s")
                
                if not self.simulate:  # Only sleep in real mode
                    time.sleep(turn_time)
                    self.stop()
            
            return success
            
        except Exception as e:
            logger.exception(f"Error during turn: {e}")
            return False
    
    def look_at(self, target: str = None, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> bool:
        """
        Point the duck's head in a specific direction.
        
        Args:
            target: Named target or None for direct angle control
            yaw: Head yaw angle in degrees (-45 to 45)
            pitch: Head pitch angle in degrees (-30 to 30)
            roll: Head roll angle in degrees (-20 to 20)
            
        Returns:
            Success flag
        """
        logger.info(f"Looking at {target if target else 'position'} (yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°)")
        
        try:
            # If target is specified, convert to angles
            if target:
                # Very basic target-to-angle mapping
                if target == "person":
                    # Look straight ahead and slightly up
                    yaw, pitch, roll = 0.0, 10.0, 0.0
                elif target == "up":
                    yaw, pitch, roll = 0.0, 30.0, 0.0
                elif target == "down":
                    yaw, pitch, roll = 0.0, -30.0, 0.0
                elif target == "left":
                    yaw, pitch, roll = -45.0, 0.0, 0.0
                elif target == "right":
                    yaw, pitch, roll = 45.0, 0.0, 0.0
                else:
                    logger.warning(f"Unknown look target: {target}")
                    return False
            
            # Clamp angles to limits
            yaw = max(self.head_limits["yaw"][0], min(yaw, self.head_limits["yaw"][1]))
            pitch = max(self.head_limits["pitch"][0], min(pitch, self.head_limits["pitch"][1]))
            roll = max(self.head_limits["roll"][0], min(roll, self.head_limits["roll"][1]))
            
            # Update state
            self.current_head_position = {"yaw": yaw, "pitch": pitch, "roll": roll}
            self.last_command_time = time.time()
            self.command_count += 1
            
            # Execute head movement through joystick interface
            success = self.joystick.set_head_position(yaw, pitch, roll)
            return success
            
        except Exception as e:
            logger.exception(f"Error during look_at: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop all duck movement.
        
        Returns:
            Success flag
        """
        logger.info("Stopping all movement")
        
        try:
            # Update state
            self.current_speed = 0.0
            self.current_turn_rate = 0.0
            self.current_strafe_rate = 0.0
            self.last_command_time = time.time()
            self.command_count += 1
            
            # Stop movement through joystick interface
            success = self.joystick.set_walking_params(0.0, 0.0, 0.0)
            return success
            
        except Exception as e:
            logger.exception(f"Error during stop: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current motion controller status.
        
        Returns:
            Dict with status information
        """
        return {
            "current_speed": self.current_speed,
            "current_turn_rate": self.current_turn_rate,
            "current_strafe_rate": self.current_strafe_rate,
            "head_position": self.current_head_position,
            "last_command_time": self.last_command_time,
            "command_count": self.command_count,
        }
    

class SimulatedMotionController(MotionController):
    """
    Simulated motion controller for the duck.
    
    This extends the base MotionController to provide simulation-specific functionality.
    """
    
    def __init__(self):
        """Initialize the simulated motion controller."""
        super().__init__(simulate=True)
        logger.info("Using SimulatedMotionController")
    
    def move(
        self, 
        direction: str = "forward", 
        speed: float = 0.5,
        duration: Optional[float] = None
    ) -> bool:
        """
        Move the duck in a simulated environment.
        
        Args:
            direction: Movement direction ('forward', 'backward', 'left', 'right')
            speed: Movement speed as fraction of maximum (0.0-1.0)
            duration: Movement duration in seconds or None for continuous
            
        Returns:
            Success flag
        """
        # Add debug logging for simulation
        logger.debug(f"SIMULATION: Moving {direction} at speed {speed:.2f}" + 
                    (f" for {duration:.2f}s" if duration else " continuously"))
        
        # Call the parent implementation
        return super().move(direction, speed, duration)
    
    def turn(self, direction: str = "left", rate: float = 0.5, angle: Optional[float] = None) -> bool:
        """
        Turn the duck in a simulated environment.
        
        Args:
            direction: Turn direction ('left', 'right', 'around')
            rate: Turn rate as fraction of maximum (0.0-1.0)
            angle: Turn angle in degrees or None for continuous
            
        Returns:
            Success flag
        """
        # Add debug logging for simulation
        angle_str = f" by {angle:.1f}°" if angle is not None else " continuously"
        logger.debug(f"SIMULATION: Turning {direction} at rate {rate:.2f}{angle_str}")
        
        # Call the parent implementation
        return super().turn(direction, rate, angle)
    
    def look_at(self, target: str = None, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> bool:
        """
        Point the duck's head in a simulated environment.
        
        Args:
            target: Named target or None for direct angle control
            yaw: Head yaw angle in degrees (-45 to 45)
            pitch: Head pitch angle in degrees (-30 to 30)
            roll: Head roll angle in degrees (-20 to 20)
            
        Returns:
            Success flag
        """
        # Add debug logging for simulation
        if target:
            logger.debug(f"SIMULATION: Looking at target: {target}")
        else:
            logger.debug(f"SIMULATION: Setting head position: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")
        
        # Call the parent implementation
        return super().look_at(target, yaw, pitch, roll)
