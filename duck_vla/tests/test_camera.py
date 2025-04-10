import unittest
import logging
import sys
import os
import numpy as np

# Add project root to sys.path to allow importing duck_vla modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from duck_vla.camera.arducam_capture import ArducamCapture
    CAMERA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import ArducamCapture, camera tests will be skipped: {e}")
    CAMERA_AVAILABLE = False
except Exception as e:
    # Catch other potential errors during import (e.g., missing libraries like picamera2)
    logging.warning(f"An unexpected error occurred importing ArducamCapture: {e}")
    CAMERA_AVAILABLE = False


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@unittest.skipIf(not CAMERA_AVAILABLE, "ArducamCapture or dependencies not available")
class TestCamera(unittest.TestCase):
    """Tests for the Arducam camera capture functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up the camera instance once for all tests."""
        logger.info("Setting up TestCamera class...")
        try:
            cls.camera = ArducamCapture()
            logger.info("ArducamCapture initialized successfully for testing.")
        except Exception as e:
            logger.exception("Failed to initialize ArducamCapture in setUpClass")
            cls.camera = None # Ensure camera is None if initialization fails
            raise unittest.SkipTest(f"Skipping camera tests due to initialization failure: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up camera resources after all tests."""
        logger.info("Tearing down TestCamera class...")
        if hasattr(cls, 'camera') and cls.camera is not None:
            try:
                cls.camera.close()
                logger.info("Camera closed successfully.")
            except Exception as e:
                logger.error(f"Error closing camera in tearDownClass: {e}")

    def test_01_camera_initialization(self):
        """Test if the camera object was initialized."""
        logger.debug("Running test_01_camera_initialization...")
        self.assertIsNotNone(self.camera, "Camera object should not be None after setup")
        logger.info("Camera initialization test passed.")

    def test_02_capture_frame(self):
        """Test capturing a single frame."""
        logger.debug("Running test_02_capture_frame...")
        self.assertIsNotNone(self.camera, "Camera object is None, skipping capture test")
        try:
            frame = self.camera.capture_frame()
            self.assertIsNotNone(frame, "Captured frame should not be None")
            self.assertIsInstance(frame, np.ndarray, "Captured frame should be a NumPy array")
            self.assertGreater(frame.size, 0, "Captured frame should not be empty")
            self.assertEqual(len(frame.shape), 3, "Frame should have 3 dimensions (H, W, C)")
            logger.info(f"Frame captured successfully, shape: {frame.shape}")
        except Exception as e:
            logger.exception("Error capturing frame")
            self.fail(f"capture_frame raised an exception: {e}")

    def test_03_get_metadata(self):
        """Test getting camera metadata."""
        logger.debug("Running test_03_get_metadata...")
        self.assertIsNotNone(self.camera, "Camera object is None, skipping metadata test")
        try:
            metadata = self.camera.get_metadata()
            self.assertIsNotNone(metadata, "Metadata should not be None")
            self.assertIsInstance(metadata, dict, "Metadata should be a dictionary")
            logger.info(f"Camera metadata retrieved: {metadata}")
            # Add more specific checks if needed, e.g., presence of certain keys
            self.assertIn('SensorResolution', metadata, "'SensorResolution' key missing in metadata")
        except Exception as e:
            logger.exception("Error getting metadata")
            self.fail(f"get_metadata raised an exception: {e}")

if __name__ == '__main__':
    logger.info("Starting camera tests...")
    unittest.main()
    logger.info("Camera tests finished.") 