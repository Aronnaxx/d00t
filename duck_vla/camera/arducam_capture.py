"""
Camera module: Arducam capture functionality

This module handles capturing frames from the Arducam camera.
"""

import logging
import time
from typing import Optional, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ArduCamCapture:
    """
    Handles capturing frames from Arducam on Radxa Zero 3W.
    """

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Initialize the camera capture system.

        Args:
            camera_id: Camera device ID
            width: Capture width in pixels
            height: Capture height in pixels
        """
        logger.debug(f"Initializing ArduCam capture (id={camera_id}, {width}x{height})")
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None

        # Status tracking
        self.last_capture_time = 0
        self.frame_count = 0
        self.errors = 0

        # Try to initialize the camera
        self._initialize_camera()

    def _initialize_camera(self) -> bool:
        """
        Initialize the camera hardware connection.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Opening camera {self.camera_id}")
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Validate settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            logger.debug(f"Camera initialized with resolution: {actual_width}x{actual_height}")

            # Warm up the camera with a few captures
            for _ in range(5):
                self.cap.read()
                time.sleep(0.1)

            return True

        except Exception as e:
            logger.exception(f"Error initializing camera: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.

        Returns:
            numpy.ndarray: Camera frame if successful, None otherwise
        """
        if not self.cap or not self.cap.isOpened():
            logger.warning("Camera not initialized, attempting to reinitialize")
            if not self._initialize_camera():
                return None

        try:
            # Track timing for performance monitoring
            start_time = time.time()

            # Capture frame
            ret, frame = self.cap.read()

            # Update stats
            self.frame_count += 1
            self.last_capture_time = time.time()
            capture_duration = self.last_capture_time - start_time

            if not ret or frame is None:
                logger.error("Failed to capture frame")
                self.errors += 1
                return None

            logger.debug(f"Frame captured: {frame.shape}, took {capture_duration:.4f}s")

            return frame

        except Exception as e:
            logger.exception(f"Error capturing frame: {e}")
            self.errors += 1
            return None

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current camera status.

        Returns:
            Dict with camera status information
        """
        return {
            "initialized": self.cap is not None and self.cap.isOpened(),
            "frame_count": self.frame_count,
            "error_count": self.errors,
            "last_capture_time": self.last_capture_time,
            "resolution": f"{self.width}x{self.height}",
        }

    def release(self) -> None:
        """Release camera resources."""
        logger.debug("Releasing camera resources")
        if self.cap:
            self.cap.release()
            self.cap = None
