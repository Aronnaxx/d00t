"""
Simulated Camera Module for Duck VLA System

This module provides a simulated camera for testing without hardware.
It uses test images to simulate a real camera.
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np

from duck_vla.camera.test_images import TestImageProvider

logger = logging.getLogger(__name__)

class SimulatedCamera:
    """
    Simulated camera for testing without hardware.
    
    This class implements the same interface as ArduCamCapture but uses
    test images instead of a real camera.
    """
    
    def __init__(self, 
                 test_dir: Optional[str] = None,
                 frame_rate: int = 30,
                 resolution: Tuple[int, int] = (640, 480)):
        """
        Initialize the simulated camera.
        
        Args:
            test_dir: Directory containing test images or None to use default
            frame_rate: Simulated frame rate in frames per second
            resolution: Simulated camera resolution (width, height)
        """
        self.frame_provider = TestImageProvider(test_dir)
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.last_frame_time = 0
        self.frame_count = 0
        self.is_running = False
        
        # Camera properties
        self.properties = {
            "resolution": self.resolution,
            "frame_rate": self.frame_rate,
            "camera_type": "simulated",
        }
        
        logger.info(f"Initialized simulated camera: {self.properties}")
    
    def start(self) -> bool:
        """
        Start the simulated camera.
        
        Returns:
            Success flag
        """
        logger.info("Starting simulated camera")
        self.is_running = True
        self.last_frame_time = time.time()
        return True
    
    def stop(self) -> bool:
        """
        Stop the simulated camera.
        
        Returns:
            Success flag
        """
        logger.info("Stopping simulated camera")
        self.is_running = False
        return True
    
    def release(self) -> None:
        """Release camera resources."""
        logger.info("Releasing simulated camera resources")
        self.stop()
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the simulated camera.
        
        Returns:
            Image frame as numpy array (BGR format) or None if error
        """
        if not self.is_running:
            logger.warning("Cannot capture frame: camera not started")
            return None
        
        # Calculate time since last frame to simulate frame rate
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        frame_interval = 1.0 / self.frame_rate
        
        # If not enough time has passed since the last frame, return None
        if elapsed < frame_interval:
            # This is a bit unrealistic but simulates frame rate
            time.sleep(frame_interval - elapsed)
        
        # Get the next test image
        frame = self.frame_provider.get_next_image()
        if frame is None:
            logger.warning("Failed to get test image")
            return None
        
        # Update state
        self.last_frame_time = time.time()
        self.frame_count += 1
        
        # Log every 30 frames
        if self.frame_count % 30 == 0:
            logger.debug(f"Captured {self.frame_count} frames from simulated camera")
        
        return frame
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get camera properties.
        
        Returns:
            Dictionary of camera properties
        """
        # Add dynamic properties
        self.properties["frame_count"] = self.frame_count
        self.properties["uptime"] = time.time() - self.last_frame_time
        
        return self.properties

# Test function if run directly
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and test the simulated camera
    camera = SimulatedCamera()
    
    try:
        logger.info("Starting camera test")
        camera.start()
        
        # Capture and display a few frames
        for i in range(5):
            frame = camera.capture_frame()
            if frame is not None:
                logger.info(f"Frame {i}: shape={frame.shape}, dtype={frame.dtype}")
            else:
                logger.warning(f"Frame {i}: No frame captured")
            
            # Sleep to simulate processing time
            time.sleep(0.1)
        
        # Get camera properties
        props = camera.get_properties()
        logger.info(f"Camera properties: {props}")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted")
    except Exception as e:
        logger.exception(f"Test error: {e}")
    finally:
        # Clean up
        camera.release()
        logger.info("Test completed") 