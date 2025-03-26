# perception/camera.py
import numpy as np
from omni.isaac.sensor import Camera

def initialize_camera(prim_path="/World/CustomRobot/CameraSensor", resolution=(640, 480), position=np.array([0.0, 0.0, 1.0])):
    camera = Camera(
        prim_path=prim_path,
        name="robot_camera",
        resolution=resolution,
        position=position
    )
    camera.initialize()
    print("Camera sensor initialized.")
    return camera

def get_camera_frame(camera):
    rgba = camera.get_rgba()  # Returns a NumPy array (H, W, 4)
    rgb = rgba[..., :3]       # Remove alpha channel
    return rgb

if __name__ == "__main__":
    cam = initialize_camera()
    frame = get_camera_frame(cam)
    print("Captured a frame of shape:", frame.shape)
