# Logging Module

This module provides telemetry logging to record the robot's trajectory, sensor readings, control commands, and camera frames for offline analysis and model retraining.

## Components

### telemetry.py
- **Purpose:** Log various aspects of the simulation such as robot state, IMU data, control commands, and optionally camera images or video.
- **Key Functions:**
  - **Log State Data:**  
    Record timestamp, position, orientation, linear and angular velocities, and issued control commands.
  - **Log IMU Data:**  
    (Optional) Capture and record simulated IMU readings.
  - **Log Camera Data:**  
    Save individual frames or compile a video stream.
  - **Export Logs:**  
    Write log entries to a JSON or CSV file at the end of the simulation.