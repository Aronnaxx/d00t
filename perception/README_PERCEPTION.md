# Perception Module

This module integrates the camera sensor and ONNX-based model inference to enable autonomous behavior via vision.

## Components

### camera.py
- **Purpose:** Attach a camera sensor to the robot, initialize it, and stream images.
- **Key Functions:**
  - **Initialize Camera:**  
    Set resolution, position, and parameters.
  - **Capture Frames:**  
    Retrieve image data (as NumPy arrays) for real-time processing.
