# robot_logging/telemetry.py
import json
import time
import cv2

# Create a global list for logging entries
telemetry_log = []

def log_robot_state(sim_time, position, orientation, linear_vel, angular_vel, command):
    entry = {
        "time": sim_time,
        "position": position,
        "orientation": orientation,
        "linear_vel": linear_vel,
        "angular_vel": angular_vel,
        "command": command
    }
    telemetry_log.append(entry)

def export_logs(filename="logs/telemetry.json"):
    with open(filename, "w") as f:
        json.dump(telemetry_log, f, indent=4)
    print(f"Telemetry exported to {filename}.")

def initialize_video_writer(filename="logs/run.avi", resolution=(640,480), fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(filename, fourcc, fps, resolution)
    return writer

def log_camera_frame(video_writer, frame):
    # Convert from RGB to BGR for OpenCV
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(bgr_frame)
