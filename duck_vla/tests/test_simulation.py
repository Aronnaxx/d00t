"""
Test script for verifying simulation mode functionality
"""

import logging
import time
import unittest
import sys
from pathlib import Path

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("test_simulation")

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestSimulation(unittest.TestCase):
    """Test suite for simulation mode functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        logger.info("Setting up test fixtures")
    
    def tearDown(self):
        """Tear down test fixtures."""
        logger.info("Tearing down test fixtures")
    
    def test_simulated_motion_controller(self):
        """Test that the SimulatedMotionController can be instantiated and used."""
        try:
            # Import the controller
            from duck_vla.action.motion_controller import SimulatedMotionController
            
            # Create an instance
            controller = SimulatedMotionController()
            logger.info("Successfully created SimulatedMotionController")
            
            # Test basic movements
            self.assertTrue(controller.move(direction="forward", speed=0.5, duration=0.1))
            self.assertTrue(controller.turn(direction="left", rate=0.5, angle=45))
            self.assertTrue(controller.stop())
            
            # Test head movement
            self.assertTrue(controller.look_at(yaw=10.0, pitch=5.0))
            
            logger.info("All SimulatedMotionController tests passed")
        except Exception as e:
            logger.exception(f"Error in simulated motion controller test: {e}")
            self.fail(f"Exception in simulated motion controller test: {e}")
    
    def test_decision_loop_simulation(self):
        """Test that the DecisionLoop can be instantiated in simulation mode."""
        try:
            # Import the decision loop
            from duck_vla.brain.decision_loop import DecisionLoop
            
            # Create an instance in simulation mode
            loop = DecisionLoop(
                simulate=True,
                audio_enabled=False,
                camera_enabled=False
            )
            logger.info("Successfully created DecisionLoop in simulation mode")
            
            # Verify simulation flag is set
            self.assertTrue(loop.simulate)
            
            # Run a few cycles
            for _ in range(3):
                loop._process_cycle()
                time.sleep(0.1)
            
            logger.info("All DecisionLoop simulation tests passed")
        except Exception as e:
            logger.exception(f"Error in decision loop simulation test: {e}")
            self.fail(f"Exception in decision loop simulation test: {e}")

if __name__ == "__main__":
    unittest.main() 