# robot_logging/telemetry.py
import json
import time
import cv2
import os
import numpy as np
from datetime import datetime

# Create a global list for logging entries
telemetry_log = []

def log_robot_state(sim_time, position, orientation, linear_vel, angular_vel, command):
    """Log the robot's state at a specific time.
    
    Args:
        sim_time (float): Simulation time in seconds
        position (list/array): [x, y, z] position
        orientation (list/array): [qw, qx, qy, qz] quaternion orientation
        linear_vel (list/array): [vx, vy, vz] linear velocity
        angular_vel (list/array): [wx, wy, wz] angular velocity
        command (list/array): [v_lin, v_ang] command velocities
    """
    try:
        entry = {
            "time": float(sim_time),
            "position": [float(x) for x in position],
            "orientation": [float(x) for x in orientation],
            "linear_vel": [float(x) for x in linear_vel],
            "angular_vel": [float(x) for x in angular_vel],
            "command": [float(x) for x in command]
        }
        telemetry_log.append(entry)
    except Exception as e:
        print(f"Error logging robot state: {str(e)}")

def ensure_directory_exists(filepath):
    """Ensure the directory for the given filepath exists.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        bool: True if directory exists or was created, False otherwise
    """
    try:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        return True
    except Exception as e:
        print(f"Error creating directory: {str(e)}")
        return False

def export_logs(filename=None):
    """Export the telemetry logs to a JSON file.
    
    Args:
        filename (str, optional): Path to export the logs.
            If None, a timestamped filename will be used.
            
    Returns:
        str: Path to the saved file, or None if failed
    """
    if not telemetry_log:
        print("No telemetry data to export")
        return None
        
    # Generate timestamped filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/telemetry_{timestamp}.json"
    
    try:
        # Ensure logs directory exists
        if ensure_directory_exists(filename):
            with open(filename, "w") as f:
                json.dump(telemetry_log, f, indent=4)
            print(f"Telemetry exported: {len(telemetry_log)} records to {filename}")
            return filename
        else:
            return None
    except Exception as e:
        print(f"Error exporting logs: {str(e)}")
        return None

def initialize_video_writer(filename=None, resolution=(640, 480), fps=30):
    """Initialize an OpenCV VideoWriter for logging camera frames.
    
    Args:
        filename (str, optional): Path to save the video.
            If None, a timestamped filename will be used.
        resolution (tuple): Video resolution (width, height)
        fps (int): Frames per second
        
    Returns:
        cv2.VideoWriter: Video writer object or None if initialization failed
    """
    # Generate timestamped filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/video_{timestamp}.avi"
    
    try:
        # Ensure logs directory exists
        if ensure_directory_exists(filename):
            # Create video writer with MJPG codec
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(filename, fourcc, fps, resolution)
            
            if not writer.isOpened():
                print(f"Failed to open video writer for {filename}")
                return None
                
            print(f"Video recording initialized to {filename} ({resolution[0]}x{resolution[1]} @ {fps}fps)")
            return writer
        else:
            return None
    except Exception as e:
        print(f"Error initializing video writer: {str(e)}")
        return None

def log_camera_frame(video_writer, frame):
    """Log a camera frame to the video writer.
    
    Args:
        video_writer (cv2.VideoWriter): Video writer object
        frame (np.ndarray): RGB image as numpy array
        
    Returns:
        bool: True if frame was written successfully, False otherwise
    """
    if video_writer is None or frame is None:
        return False
        
    try:
        # Convert from RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)
        return True
    except Exception as e:
        print(f"Error logging camera frame: {str(e)}")
        return False

def reset_logs():
    """Clear the telemetry log."""
    global telemetry_log
    telemetry_log = []
    print("Telemetry log reset")

if __name__ == "__main__":
    # For testing the telemetry module independently
    import random
    
    # Clear any previous logs
    reset_logs()
    
    # Create a test video writer
    video_writer = initialize_video_writer(resolution=(320, 240))
    
    # Generate some random telemetry data and frames
    for i in range(10):
        # Log state
        position = [random.uniform(-10, 10) for _ in range(3)]
        orientation = [random.uniform(-1, 1) for _ in range(4)]
        linear_vel = [random.uniform(-2, 2) for _ in range(3)]
        angular_vel = [random.uniform(-1, 1) for _ in range(3)]
        command = [random.uniform(-1, 1), random.uniform(-0.5, 0.5)]
        
        log_robot_state(i*0.1, position, orientation, linear_vel, angular_vel, command)
        
        # Log frame
        if video_writer is not None:
            # Create a test frame with a moving ball
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            center = (int(160 + 100*np.sin(i*0.5)), int(120 + 80*np.cos(i*0.5)))
            cv2.circle(frame, center, 20, (0, 0, 255), -1)
            log_camera_frame(video_writer, frame)
    
    # Export logs
    export_logs()
    
    # Close video writer
    if video_writer is not None:
        video_writer.release()
        print("Video writer released")
