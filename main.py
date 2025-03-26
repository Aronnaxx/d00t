# main.py
from simulation.environment import launch_simulation, load_warehouse_stage, initialize_world
from simulation.robot_loader import load_robot
from control.controller import get_robot_articulation, command_differential_drive
from perception.camera import initialize_camera, get_camera_frame
from perception.model_inference import load_onnx_model, preprocess_image, run_inference
from robot_logging.telemetry import log_robot_state, export_logs, initialize_video_writer, log_camera_frame
import time

def main():
    # Launch simulation and load environment
    sim_app = launch_simulation(headless=False)
    load_warehouse_stage()
    world = initialize_world()

    # Load the robot model into the environment
    load_robot("assets/robot.usd", translation=(1, 0, 0))

    # Initialize robot control
    dc_interface, robot_art = get_robot_articulation()
    wheel_params = {
        "left_joint_name": "left_wheel_joint",
        "right_joint_name": "right_wheel_joint",
        "axle_length": 0.5,
        "wheel_radius": 0.1
    }

    # Initialize camera and ONNX model
    camera = initialize_camera()
    session, input_name = load_onnx_model()
    video_writer = initialize_video_writer()

    # Main simulation loop
    start_time = time.time()
    frame_count = 0
    try:
        while sim_app.is_running():
            # Get simulation time
            sim_time = time.time() - start_time

            # Capture camera frame and run ONNX inference
            frame = get_camera_frame(camera)
            processed_frame = preprocess_image(frame)
            v_lin, v_ang = run_inference(session, input_name, processed_frame)

            # Command the robot based on inference results
            command_differential_drive(dc_interface, robot_art, v_lin, v_ang, wheel_params)

            # Log telemetry data (example: log state here; implement functions to get pose, velocities, etc.)
            # For illustration, using dummy values:
            position = [1.0, 0.0, 0.0]
            orientation = [1.0, 0.0, 0.0, 0.0]
            linear_vel = [v_lin, 0.0, 0.0]
            angular_vel = [0.0, 0.0, v_ang]
            log_robot_state(sim_time, position, orientation, linear_vel, angular_vel, [v_lin, v_ang])

            # Log camera frame to video
            log_camera_frame(video_writer, frame)

            # Step the simulation (implementation depends on Isaac Lab’s API, e.g., world.step(render=True))
            world.step(render=True)
            frame_count += 1

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # Export logs and cleanup
        export_logs()
        video_writer.release()
        sim_app.close()

if __name__ == "__main__":
    main()
