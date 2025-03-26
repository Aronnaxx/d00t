# control/controller.py
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_robot_articulation(robot_path="/World/CustomRobot"):
    """Get a handle to the robot's articulation for control.
    
    Args:
        robot_path (str): Path to the robot prim in the stage
        
    Returns:
        tuple: (dc_interface, robot_articulation) if successful, (None, None) otherwise
    """
    try:
        # Import modules after simulation is initialized
        import omni.isaac.dynamic_control as dc
        
        dc_interface = dc._dynamic_control.acquire_dynamic_control_interface()
        robot_art = dc_interface.get_articulation(robot_path)
        
        if robot_art is None:
            logger.error(f"No articulation found at {robot_path}")
            logger.error("Make sure the robot is loaded and is an articulation.")
            return None, None
            
        logger.info(f"Found robot articulation at {robot_path}")
        return dc_interface, robot_art
    except Exception as e:
        logger.error(f"Error getting robot articulation: {str(e)}")
        return None, None

def command_differential_drive(dc_interface, robot_art, v_lin, v_ang, wheel_params):
    """Control a differential drive robot by setting wheel velocities.
    
    Args:
        dc_interface: Dynamic control interface
        robot_art: Robot articulation handle
        v_lin (float): Linear velocity command (m/s)
        v_ang (float): Angular velocity command (rad/s)
        wheel_params (dict): Dictionary with keys:
            - left_joint_name: Name of the left wheel joint
            - right_joint_name: Name of the right wheel joint
            - axle_length: Distance between wheels (m)
            - wheel_radius: Radius of the wheels (m)
            
    Returns:
        bool: True if command sent successfully, False otherwise
    """
    if dc_interface is None or robot_art is None:
        logger.error("Invalid dynamic control interface or articulation")
        return False
        
    try:
        # Find the wheel joints
        left_joint = dc_interface.find_articulation_dof(robot_art, wheel_params["left_joint_name"])
        right_joint = dc_interface.find_articulation_dof(robot_art, wheel_params["right_joint_name"])
        
        if left_joint is None or right_joint is None:
            logger.error(f"Could not find wheel joints. Left joint: {wheel_params['left_joint_name']}, Right joint: {wheel_params['right_joint_name']}")
            return False
        
        # Calculate differential drive kinematics
        L = wheel_params["axle_length"]   # Distance between wheels
        R = wheel_params["wheel_radius"]  # Wheel radius
        
        # Convert linear and angular velocity to wheel speeds (rad/s)
        v_left = (v_lin - (v_ang * L) / 2.0) / R
        v_right = (v_lin + (v_ang * L) / 2.0) / R
        
        # Apply velocity commands to wheels
        dc_interface.set_dof_velocity_target(left_joint, float(v_left))
        dc_interface.set_dof_velocity_target(right_joint, float(v_right))
        
        return True
    except Exception as e:
        logger.error(f"Error commanding differential drive: {str(e)}")
        return False

if __name__ == "__main__":
    # For testing the controller independently
    import argparse
    from isaaclab.app import AppLauncher
    import time
    
    # Create parser and launch app
    parser = argparse.ArgumentParser(description="Controller Test")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])  # Empty list to avoid using command line args
    
    # Launch sim
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app
    
    # Example wheel parameters
    wheel_params = {
        "left_joint_name": "left_wheel_joint",
        "right_joint_name": "right_wheel_joint",
        "axle_length": 0.5,
        "wheel_radius": 0.1
    }
    
    # Test the controller (simulation would need to be running)
    dc_interface, robot_art = get_robot_articulation()
    if dc_interface and robot_art:
        logger.info("Testing forward motion...")
        command_differential_drive(dc_interface, robot_art, 1.0, 0.0, wheel_params)
        time.sleep(2)
        
        logger.info("Testing turning motion...")
        command_differential_drive(dc_interface, robot_art, 0.5, 0.5, wheel_params)
        time.sleep(2)
        
        logger.info("Testing stop...")
        command_differential_drive(dc_interface, robot_art, 0.0, 0.0, wheel_params)
    else:
        logger.error("Test skipped due to missing robot articulation")
    
    # Close simulation
    sim_app.close()
