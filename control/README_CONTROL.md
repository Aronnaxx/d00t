# Control Module

This module provides functions for external Python control of your custom robot using Isaac Lab's dynamic control interface.

## Components

### controller.py
- **Purpose:** Retrieve the robot's dynamic control handle and command its joints or wheels.
- **Key Functions:**
  - **Access Articulation:**  
    Get a handle on the robot's articulation using Isaac Sim’s dynamic control.
  - **Set Commands:**  
    Send velocity/position commands to joints (e.g., for a differential drive robot).
  - **Control Loop Integration:**  
    Designed to be called every simulation step to update robot control.
