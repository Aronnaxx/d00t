# Isaac Lab Robot Integration

This project demonstrates how to integrate a custom robot into NVIDIA Isaac Lab, complete with external control, camera streaming, ONNX model inference, and telemetry logging.

## Project Structure

```
project/
├── assets/               # Contains robot USD files
├── logs/                 # Directory for telemetry and video logs
├── main.py               # Main entry point for the simulation
├── control/              # Robot control module
│   └── controller.py     # Differential drive controller
├── perception/           # Vision and ONNX model module
│   ├── camera.py         # Camera sensor setup and access
│   └── model_inference.py # ONNX model loading and inference
├── robot_logging/        # Data logging module
│   └── telemetry.py      # Utilities for logging robot data
├── simulation/           # Simulation environment module
│   ├── environment.py    # Isaac Sim initialization and stage loading
│   └── robot_loader.py   # Robot loading and placement
├── create_test_robot.py  # Utility to create a test robot USD
├── run.sh                # Convenience script to run the simulation
├── setup.sh              # Setup script to initialize the project
└── pyproject.toml        # Project configuration for UV
```

## Prerequisites

1. NVIDIA Isaac Sim: Will be installed via UV package manager
2. Python 3.10 (required by Isaac Lab)
3. [UV Package Manager](https://github.com/astral-sh/uv): The modern Python package manager

## Getting Started

### 1. Install UV Package Manager

If you don't have UV already installed, you can install it with:

```bash
curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/astral-sh/uv/main/install.sh | sh
```

### 2. Setup the Project

Run the setup script to initialize the project, install dependencies, and create a test robot:

```bash
./setup.sh
```

This will:
- Install all required dependencies including Isaac Lab and Isaac Sim
- Create necessary directories
- Generate a test robot USD file

### 3. Running the Simulation

Once setup is complete, you can run the simulation with:

```bash
# Using the convenience script
./run.sh

# Or directly with UV
uv run main
```

Optional command-line arguments:
- `--headless`: Run in headless mode (no GUI)
- `--robot PATH`: Specify a custom robot USD file path
- `--model PATH`: Specify a custom ONNX model file path
- `--log-dir DIR`: Specify a directory for logs

Example with custom settings:
```bash
uv run main -- --robot assets/my_custom_robot.usd --model models/nav_policy.onnx
```

### 4. Running Individual Components

You can run individual components of the project with UV for testing:

```bash
# Create a test robot USD
uv run create-robot

# Test the environment
uv run run-env

# Test the camera
uv run run-camera

# Test the model inference
uv run run-model

# Test the controller
uv run run-controller

# Test the telemetry logging
uv run run-telemetry
```

## Components

### Simulation Environment
The `simulation` module handles launching Isaac Sim, loading the warehouse environment, and inserting the robot.

### Robot Control
The `control` module provides access to the robot's articulation and implements differential drive control.

### Perception
The `perception` module sets up the camera sensor and handles ONNX model inference:
- The camera is attached to the robot and provides RGB frames
- The ONNX model takes these frames and outputs velocity commands

### Telemetry Logging
The `robot_logging` module records the robot's state and camera frames for later analysis:
- Robot telemetry is saved as a JSON file
- Camera frames are recorded as a video

## Customization

### Using Your Own Robot
1. Prepare your robot as a USD file (convert from URDF if needed using Isaac Lab tools)
2. Update the joint names in `main.py` to match your robot's configuration
3. Run with `--robot` pointing to your USD file

### Using Your Own ONNX Model
1. Prepare an ONNX model that takes camera images and outputs velocity commands
2. You may need to update `model_inference.py` to match your model's input/output formats
3. Run with `--model` pointing to your ONNX file

## Troubleshooting

- **UV Installation Issues**: If you have problems with UV, check the [UV documentation](https://github.com/astral-sh/uv)
- **Nucleus Path Issues**: If you encounter errors loading the warehouse stage, check if the Nucleus path is correct for your installation
- **Joint Names**: Ensure the joint names in the wheel parameters match your robot's joint names
- **Camera Issues**: If camera frames are empty, try stepping the simulation a few more frames before using them

## License

This project is governed by the terms of the BSD-3 License, in alignment with the Isaac Lab framework license.