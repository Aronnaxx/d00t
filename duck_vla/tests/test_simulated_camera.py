"""
Test script for the simulated camera module
"""

import logging
import unittest
import sys
import os
import tempfile
from pathlib import Path
import time
import numpy as np

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("test_simulated_camera")

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestSimulatedCamera(unittest.TestCase):
    """Test suite for the simulated camera module."""
    
    def setUp(self):
        """Set up test fixtures."""
        logger.info("Setting up test fixtures")
        
        # Create a temporary directory for test images
        self.test_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary test directory: {self.test_dir}")
    
    def tearDown(self):
        """Tear down test fixtures."""
        logger.info("Tearing down test fixtures")
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.test_dir)
        logger.info(f"Removed temporary test directory: {self.test_dir}")
    
    def test_test_image_provider(self):
        """Test the TestImageProvider class."""
        from duck_vla.camera.test_images import TestImageProvider
        
        # Create a test image provider with the temporary directory
        provider = TestImageProvider(self.test_dir)
        
        # Check that the provider created a sample image
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "sample_test_image.png")))
        
        # Get an image
        image = provider.get_next_image()
        self.assertIsNotNone(image)
        self.assertIsInstance(image, np.ndarray)
        
        # Check image dimensions (should be a color image)
        self.assertEqual(len(image.shape), 3)
        self.assertGreaterEqual(image.shape[0], 1)  # Height
        self.assertGreaterEqual(image.shape[1], 1)  # Width
        self.assertEqual(image.shape[2], 3)  # Color channels
        
        # Test random image
        random_image = provider.get_random_image()
        self.assertIsNotNone(random_image)
        self.assertIsInstance(random_image, np.ndarray)
    
    def test_simulated_camera(self):
        """Test the SimulatedCamera class."""
        from duck_vla.camera.simulated_camera import SimulatedCamera
        
        # Create a simulated camera with the temporary directory
        camera = SimulatedCamera(test_dir=self.test_dir)
        
        # Check initial state
        self.assertFalse(camera.is_running)
        self.assertEqual(camera.frame_count, 0)
        
        # Start the camera
        result = camera.start()
        self.assertTrue(result)
        self.assertTrue(camera.is_running)
        
        # Capture a frame
        frame = camera.capture_frame()
        self.assertIsNotNone(frame)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(camera.frame_count, 1)
        
        # Test frame dimensions
        self.assertEqual(len(frame.shape), 3)
        
        # Capture multiple frames quickly
        frames = []
        for _ in range(5):
            frame = camera.capture_frame()
            frames.append(frame)
        
        # Should have 5 more frames
        self.assertEqual(camera.frame_count, 6)
        
        # All frames should be valid
        for frame in frames:
            self.assertIsNotNone(frame)
            self.assertIsInstance(frame, np.ndarray)
        
        # Test properties
        props = camera.get_properties()
        self.assertEqual(props["frame_count"], 6)
        self.assertEqual(props["camera_type"], "simulated")
        self.assertIn("uptime", props)
        
        # Stop the camera
        result = camera.stop()
        self.assertTrue(result)
        self.assertFalse(camera.is_running)
        
        # Trying to capture a frame now should return None
        frame = camera.capture_frame()
        self.assertIsNone(frame)
        
        # Release resources
        camera.release()
    
    def test_camera_performance(self):
        """Test camera performance in terms of frame rate."""
        from duck_vla.camera.simulated_camera import SimulatedCamera
        
        # Create a camera with a specified frame rate
        target_fps = 10
        camera = SimulatedCamera(test_dir=self.test_dir, frame_rate=target_fps)
        
        # Start the camera
        camera.start()
        
        # Capture frames for a few seconds and measure rate
        frames = []
        start_time = time.time()
        duration = 2.0  # seconds
        
        while time.time() - start_time < duration:
            frame = camera.capture_frame()
            if frame is not None:
                frames.append(frame)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Calculate actual frame rate
        actual_fps = len(frames) / actual_duration
        
        # The actual frame rate should be close to the target
        self.assertGreaterEqual(actual_fps, target_fps * 0.9)  # Allow 10% error margin
        self.assertLessEqual(actual_fps, target_fps * 1.1)     # Allow 10% error margin
        
        # Stop and release
        camera.stop()
        camera.release()

if __name__ == "__main__":
    unittest.main() 