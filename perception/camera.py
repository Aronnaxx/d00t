# perception/camera.py
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_camera(prim_path="/World/CustomRobot/CameraSensor", 
                      resolution=(640, 480), 
                      position=np.array([0.0, 0.0, 1.0]),
                      rotation=np.array([0.0, 0.0, 0.0]),
                      focal_length=24.0):
    """Initialize a camera sensor on the robot.
    
    Args:
        prim_path (str): Path where the camera will be created in the stage
        resolution (tuple): Image resolution as (width, height)
        position (np.array): Position of camera relative to parent (usually robot)
        rotation (np.array): Rotation in euler angles (degrees)
        focal_length (float): Focal length in mm
        
    Returns:
        Camera: Camera object if successful, None otherwise
    """
    try:
        # Import Isaac sensor modules after simulation is initialized
        from omni.isaac.sensor import Camera
        import omni.kit.commands
        
        # Check if camera already exists at that path and remove it to avoid conflicts
        # This is safe to do as we're initializing a new one
        try:
            omni.kit.commands.execute("DeletePrims", paths=[prim_path])
            logger.info(f"Removed existing camera at {prim_path}")
        except:
            pass
            
        # Create the camera
        camera = Camera(
            prim_path=prim_path,
            name="robot_camera",
            resolution=resolution,
            position=position,
            rotation=rotation,
            focal_length=focal_length
        )
        
        # Initialize the camera (required to start rendering)
        camera.initialize()
        
        logger.info(f"Camera initialized at {prim_path} with resolution {resolution}")
        logger.info(f"  - Position: {position}")
        logger.info(f"  - Rotation: {rotation}")
        logger.info(f"  - Focal length: {focal_length}mm")
        
        return camera
    except Exception as e:
        logger.error(f"Error initializing camera: {str(e)}")
        return None

def get_camera_frame(camera):
    """Get the latest RGB frame from the camera.
    
    Args:
        camera: Camera object from initialize_camera()
        
    Returns:
        np.ndarray: RGB image as numpy array of shape (H, W, 3)
                   or None if camera is invalid
    """
    if camera is None:
        logger.error("Invalid camera object")
        return None
        
    try:
        # Get RGBA image from camera
        rgba = camera.get_rgba()
        
        if rgba is None or rgba.size == 0:
            logger.warning("Camera returned empty frame")
            return None
            
        # Convert to RGB by removing alpha channel
        rgb = rgba[..., :3]
        return rgb
    except Exception as e:
        logger.error(f"Error getting camera frame: {str(e)}")
        return None

def get_camera_depth(camera):
    """Get the latest depth map from the camera.
    
    Args:
        camera: Camera object from initialize_camera()
        
    Returns:
        np.ndarray: Depth image as numpy array of shape (H, W)
                   or None if camera is invalid
    """
    if camera is None:
        logger.error("Invalid camera object")
        return None
        
    try:
        # Get depth from camera
        depth = camera.get_linear_depth()
        return depth
    except Exception as e:
        logger.error(f"Error getting depth: {str(e)}")
        return None

if __name__ == "__main__":
    # For testing the camera independently
    import argparse
    from isaaclab.app import AppLauncher
    import time
    
    # Create parser and launch app
    parser = argparse.ArgumentParser(description="Camera Test")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])  # Empty list to avoid using command line args
    
    # Initialize simulation for testing
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app
    
    # Now we can import the Isaac-specific modules
    import matplotlib.pyplot as plt
    import omni.isaac.core.utils.stage as stage_utils
    import omni.isaac.core.utils.xformable as xform_utils
    
    # Add a simple default stage
    stage_utils.create_new_stage()
    
    # Initialize the camera
    cam = initialize_camera(position=np.array([0.0, 0.0, 1.0]))
    
    # Step the simulation to ensure the camera has a frame
    for _ in range(10):
        sim_app.update()
    
    # Get the frame and show it
    frame = get_camera_frame(cam)
    if frame is not None:
        logger.info(f"Captured frame of shape {frame.shape}")
        
        # Display the image (this works in a script but may not work in the notebook)
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(frame)
            plt.title("Camera View")
            plt.savefig("camera_test.png")
            logger.info("Test image saved to camera_test.png")
        except Exception as e:
            logger.error(f"Could not save image: {e}")
    else:
        logger.error("Failed to capture camera frame")
        
    # Clean up
    sim_app.close()
