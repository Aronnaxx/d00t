# simulation/robot_loader.py
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_robot(usd_path, prim_path="/World/CustomRobot", translation=(0, 0, 0), orientation=(0, 0, 0, 1)):
    """Load a robot USD into the simulation stage.
    
    Args:
        usd_path (str): Path to the robot USD file
        prim_path (str): Target path in the stage hierarchy
        translation (tuple): Initial position (x, y, z)
        orientation (tuple): Initial orientation as quaternion (qx, qy, qz, qw)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import omni modules 
        from omni.isaac.core.utils.stage import add_reference_to_stage
        import omni.isaac.core.utils.prims as prim_utils
        
        # Check if file exists when using local path
        if not usd_path.startswith("omniverse://") and not os.path.exists(usd_path):
            logger.error(f"Robot USD file not found at {usd_path}")
            return False
            
        # Add the robot USD as a reference to the stage
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        
        # Set the initial transform
        prim_utils.set_transform(prim_path, translation=translation, orientation=orientation)
        
        logger.info(f"Robot loaded at {prim_path}:")
        logger.info(f"  - USD source: {usd_path}")
        logger.info(f"  - Position: {translation}")
        logger.info(f"  - Orientation: {orientation}")
        return True
    except Exception as e:
        logger.error(f"Failed to load robot: {str(e)}")
        return False

if __name__ == "__main__":
    # For testing this module independently
    import argparse
    from isaaclab.app import AppLauncher
    import omni.usd
    import time
    
    # Create parser and launch app
    parser = argparse.ArgumentParser(description="Robot Loader Test")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])  # Empty list to avoid using command line args
    
    # Initialize simulation for standalone testing
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app
    
    stage = omni.usd.get_context().get_stage()
    
    # Load robot
    success = load_robot("assets/robot.usd", translation=(1, 0, 0))
    
    if success:
        logger.info("Robot loaded successfully for testing")
    else:
        logger.error("Failed to load robot in test mode")
    
    # Keep the app open briefly to see the results
    time.sleep(5)
    sim_app.close()
