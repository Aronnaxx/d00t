"""
Action module: Joystick interface for duck movement

This module provides a wrapper interface for the Open Duck Playground joystick
control system. It handles the differences between simulation and hardware interfaces.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class JoystickInterface:
    """
    Interface to the duck's joystick control system.
    
    This class wraps the Open Duck Playground's Joystick class for simulation,
    and provides a consistent interface that can be swapped with hardware control.
    """
    
    def __init__(self, simulate: bool = False, task: str = "flat_terrain"):
        """
        Initialize the joystick interface.
        
        Args:
            simulate: Whether to use simulation mode
            task: The simulation task type to use (e.g., "flat_terrain", "rough_terrain")
        """
        self.simulate = simulate
        self.task = task
        logger.info(f"Initializing joystick interface (simulate={simulate}, task={task})")
        
        # Initialize the appropriate joystick implementation
        self._initialize_implementation()
    
    def _initialize_implementation(self) -> None:
        """Initialize the appropriate joystick implementation."""
        try:
            if self.simulate:
                # Add the Open Duck Playground to the Python path
                playground_path = Path(__file__).parent.parent.parent / "submodules" / "open_duck_playground"
                if str(playground_path) not in sys.path:
                    sys.path.insert(0, str(playground_path))
                    logger.debug(f"Added {playground_path} to Python path")
                
                # Import simulation interface
                logger.debug("Importing simulation joystick interface")
                from playground.open_duck_mini_v2.joystick import Joystick
                
                # Create instance with the specified task
                self._joystick = Joystick(task=self.task)
                logger.info(f"Initialized simulation joystick with task: {self.task}")
                
                # Set up simulation-specific methods
                self._setup_simulation_methods()
            else:
                # Import real hardware interface or use mock
                logger.debug("Using mock joystick interface for hardware")
                self._setup_mock_methods()
        except Exception as e:
            logger.exception(f"Error initializing joystick implementation: {e}")
            logger.warning("Using mock joystick interface as fallback")
            self._setup_mock_methods()
    
    def _setup_simulation_methods(self) -> None:
        """Set up methods for simulation interface."""
        # Currently, we need to implement translation between our API and the simulation's API
        # This may be expanded in the future as needed
        pass
    
    def _setup_mock_methods(self) -> None:
        """Set up methods for mock interface."""
        self._joystick = None
        logger.info("Mock joystick interface set up")
    
    def set_walking_params(self, x_vel: float, y_vel: float, yaw_vel: float) -> bool:
        """
        Set walking parameters for movement.
        
        Args:
            x_vel: Forward/backward velocity (-1.0 to 1.0)
            y_vel: Left/right velocity (-1.0 to 1.0)
            yaw_vel: Rotational velocity (-1.0 to 1.0)
            
        Returns:
            Success flag
        """
        logger.debug(f"Setting walking params: x_vel={x_vel:.2f}, y_vel={y_vel:.2f}, yaw_vel={yaw_vel:.2f}")
        
        try:
            if self.simulate and self._joystick is not None:
                # For simulation, we'll need to add implementation specific to the duck playground
                # This is a placeholder
                logger.debug("Using simulation joystick for walking params")
                # TODO: Implement translation between our API and simulation API
                return True
            else:
                # Mock implementation
                logger.info(f"MOCK: Walking: x_vel={x_vel:.2f}, y_vel={y_vel:.2f}, yaw_vel={yaw_vel:.2f}")
                return True
        except Exception as e:
            logger.exception(f"Error setting walking params: {e}")
            return False
    
    def set_head_position(self, yaw: float, pitch: float, roll: float) -> bool:
        """
        Set the head position of the duck.
        
        Args:
            yaw: Head yaw angle in degrees
            pitch: Head pitch angle in degrees
            roll: Head roll angle in degrees
            
        Returns:
            Success flag
        """
        logger.debug(f"Setting head position: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")
        
        try:
            if self.simulate and self._joystick is not None:
                # For simulation, we'll need to add implementation specific to the duck playground
                # This is a placeholder
                logger.debug("Using simulation joystick for head position")
                # TODO: Implement translation between our API and simulation API
                return True
            else:
                # Mock implementation
                logger.info(f"MOCK: Head: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")
                return True
        except Exception as e:
            logger.exception(f"Error setting head position: {e}")
            return False
    
    def enable(self) -> bool:
        """
        Enable the joystick interface.
        
        Returns:
            Success flag
        """
        logger.debug("Enabling joystick interface")
        
        try:
            if self.simulate and self._joystick is not None:
                # Simulation enable logic
                logger.debug("Enabling simulation joystick")
                # No specific enable method in the simulation, so just return success
                return True
            else:
                # Mock implementation
                logger.info("MOCK: Joystick enabled")
                return True
        except Exception as e:
            logger.exception(f"Error enabling joystick: {e}")
            return False
    
    def disable(self) -> bool:
        """
        Disable the joystick interface.
        
        Returns:
            Success flag
        """
        logger.debug("Disabling joystick interface")
        
        try:
            if self.simulate and self._joystick is not None:
                # Simulation disable logic
                logger.debug("Disabling simulation joystick")
                # No specific disable method in the simulation, so just return success
                return True
            else:
                # Mock implementation
                logger.info("MOCK: Joystick disabled")
                return True
        except Exception as e:
            logger.exception(f"Error disabling joystick: {e}")
            return False 