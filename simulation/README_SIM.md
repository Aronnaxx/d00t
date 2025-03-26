# Simulation Module

This module is responsible for launching NVIDIA Isaac Lab, loading the Warehouse environment, and inserting your custom robot into the scene.

## Components

### environment.py
- **Purpose:** Launch Isaac Lab, load the Warehouse environment USD, and optionally set up the simulation world.
- **Key Functions:**
  - **Initialization:**  
    Set up the Isaac Sim application with your desired configuration (GUI vs. headless).
  - **Load Environment:**  
    Open the Warehouse stage using the USD API.
  - **World Setup:**  
    Optionally, instantiate a `World` object to manage simulation stepping.
