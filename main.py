# main.py
import os
import argparse
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import isaaclab
from isaaclab.app import AppLauncher

# Import our modules
from simulation.environment import launch_simulation, load_warehouse_stage, initialize_world
from simulation.robot_loader import load_robot
from control.controller import get_robot_articulation, command_differential_drive
from perception.camera import initialize_camera, get_camera_frame
from perception.model_inference import load_onnx_model, preprocess_image, run_inference, create_dummy_onnx_model
from robot_logging.telemetry import log_robot_state, export_logs, initialize_video_writer, log_camera_frame, reset_logs

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Isaac Lab Robot Integration Demo")
    
    # Add AppLauncher arguments first (headless, livestream, enable_cameras, device, etc.)
    AppLauncher.add_app_launcher_args(parser)
    
    # Then add our custom arguments
    parser.add_argument("--robot-usd", type=str, default="assets/robot.usd", help="Path to robot USD file")
    parser.add_argument("--model-file", type=str, default="model.onnx", help="Path to ONNX model file")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to store logs")
    
    return parser.parse_args()

def main():
    """Main function to run the simulation with robot control via ONNX model."""
    # Parse arguments
    args = parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    reset_logs()  # Clear any previous logs
    
    # Check if model exists, create dummy one if needed
    if not os.path.exists(args.model_file):
        logger.info(f"ONNX model not found at {args.model_file}, creating a dummy model...")
        create_dummy_onnx_model(args.model_file)
    
    # Launch simulation and load environment
    logger.info("Launching Isaac Sim...")
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app
    
    logger.info("Loading warehouse environment...")
    load_warehouse_stage()
    
    logger.info("Initializing simulation world...")
    world = initialize_world()
    
    # Load the robot model into the environment
    logger.info(f"Loading robot from {args.robot_usd}...")
    success = load_robot(args.robot_usd, translation=(1, 0, 0))
    if not success:
        logger.error("Failed to load robot. Exiting.")
        sim_app.close()
        return
    
    # Initialize robot control
    logger.info("Initializing robot control...")
    dc_interface, robot_art = get_robot_articulation()
    if dc_interface is None or robot_art is None:
        logger.error("Failed to get robot articulation. Exiting.")
        sim_app.close()
        return
    
    wheel_params = {
        "left_joint_name": "left_wheel_joint",
        "right_joint_name": "right_wheel_joint",
        "axle_length": 0.5,
        "wheel_radius": 0.1
    }
    
    # Initialize camera
    logger.info("Setting up camera sensor...")
    camera = initialize_camera()
    if camera is None:
        logger.error("Failed to initialize camera. Exiting.")
        sim_app.close()
        return
    
    # Load ONNX model
    logger.info(f"Loading ONNX model from {args.model_file}...")
    session, input_name, input_shape = load_onnx_model(args.model_file)
    if session is None:
        logger.error("Failed to load ONNX model. Exiting.")
        sim_app.close()
        return
    
    # Initialize video writer for camera logging
    logger.info("Setting up video recording...")
    video_writer = initialize_video_writer()
    if video_writer is None:
        logger.warning("Failed to initialize video writer. Continuing without video recording.")
    
    # Step the simulation a few times to ensure everything is initialized
    logger.info("Initializing simulation...")
    for _ in range(10):
        world.step(render=True)
    
    # Main simulation loop
    logger.info("Starting main simulation loop...")
    start_time = time.time()
    frame_count = 0
    try:
        while sim_app.is_running():
            # Get simulation time
            sim_time = time.time() - start_time
            
            # Capture camera frame
            frame = get_camera_frame(camera)
            if frame is None:
                logger.warning("Failed to get camera frame")
                world.step(render=True)
                continue
            
            # Run ONNX inference
            processed_frame = preprocess_image(frame)
            v_lin, v_ang = run_inference(session, input_name, processed_frame)
            
            # Print command occasionally
            if frame_count % 30 == 0:
                logger.info(f"[{sim_time:.2f}s] Command: v_lin={v_lin:.2f} m/s, v_ang={v_ang:.2f} rad/s")
            
            # Command the robot based on inference results
            command_differential_drive(dc_interface, robot_art, v_lin, v_ang, wheel_params)
            
            # Get robot state for logging
            # In a real implementation, you would get these from the simulator
            # For now, we're using placeholder values that match the robot's commanded movement
            # You would replace these with actual values from Isaac Sim
            position = [sim_time * v_lin, 0.0, 0.0]  # Simple integration of velocity
            orientation = [1.0, 0.0, 0.0, 0.0]       # Identity quaternion
            linear_vel = [v_lin, 0.0, 0.0]           # Forward velocity only
            angular_vel = [0.0, 0.0, v_ang]          # Yaw rate only
            
            # Log telemetry data
            log_robot_state(sim_time, position, orientation, linear_vel, angular_vel, [v_lin, v_ang])
            
            # Log camera frame to video
            if video_writer is not None:
                log_camera_frame(video_writer, frame)
            
            # Step the simulation
            world.step(render=True)
            frame_count += 1
            
            # Optional: slow down the simulation to real-time
            # time.sleep(0.01)  # Uncomment to limit to ~100fps

    except KeyboardInterrupt:
        logger.info("\nSimulation interrupted by user")
    except Exception as e:
        logger.error(f"Error in simulation loop: {str(e)}")
    finally:
        # Export logs and cleanup
        logger.info("Exporting telemetry logs...")
        export_logs()
        
        if video_writer is not None:
            logger.info("Finalizing video recording...")
            video_writer.release()
        
        logger.info("Closing simulation...")
        sim_app.close()
        logger.info("Done!")

if __name__ == "__main__":
    main()
