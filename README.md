# Open Duck VLM Integration

This project provides an API and client libraries to allow Vision Language Models (VLMs) to control the Open Duck robot in MuJoCo simulation. The system enables external AI systems to view the robot's environment through camera frames and send movement commands.

## System Architecture

```mermaid
graph TD
    subgraph "MuJoCo Simulation"
        M[MuJoCo Thread] -- renders --> CF[Camera Frames]
        ONNX[ONNX Model] -- controls --> Robot[Duck Robot]
        M -- steps simulation --> Robot
    end
    
    subgraph "API Server"
        Flask[Flask API Server] -- starts/manages --> M
        Flask -- gets frames from --> CF
        Flask -- sends commands to --> Robot
        Flask -- exposes endpoints --> API[REST API Endpoints]
    end
    
    subgraph "Client Libraries"
        PythonClient[Python Client]
        CLITool[Command Line Tool]
        VLMIntegration[VLM Integration]
    end
    
    API -- consumed by --> PythonClient
    PythonClient -- used by --> CLITool
    PythonClient -- used by --> VLMIntegration
    
    subgraph "External VLM"
        VLMService[VLM API Service]
    end
    
    VLMIntegration -- sends frames to --> VLMService
    VLMService -- returns commands --> VLMIntegration
```

## Control Flow

```mermaid
sequenceDiagram
    participant VLM as VLM/LLM System
    participant Client as Duck Client
    participant API as API Server
    participant MuJoCo as MuJoCo Simulation
    
    VLM->>Client: Initialize simulation
    Client->>API: POST /initialize
    API->>MuJoCo: Start simulation thread
    API-->>Client: Success response
    Client-->>VLM: Simulation ready
    
    loop Control Loop
        VLM->>Client: Request camera frame
        Client->>API: GET /frame
        API->>MuJoCo: Get current frame
        MuJoCo-->>API: Return frame data
        API-->>Client: Frame (base64 encoded)
        Client-->>VLM: Processed frame
        
        Note over VLM: Process image and decide action
        
        VLM->>Client: Send command
        Client->>API: POST /command
        API->>MuJoCo: Update command queue
        MuJoCo->>MuJoCo: Execute command
        API-->>Client: Command received
        Client-->>VLM: Command executed
    end
    
    VLM->>Client: Shutdown simulation
    Client->>API: POST /shutdown
    API->>MuJoCo: Stop simulation thread
    API-->>Client: Success response
    Client-->>VLM: Simulation stopped
```

## Natural Language Command Processing

```mermaid
graph LR
    NLCommand[Natural Language Command] --> Parser[Command Parser]
    
    Parser --> Forward[Move Forward]
    Parser --> Backward[Move Backward]
    Parser --> TurnLeft[Turn Left]
    Parser --> TurnRight[Turn Right]
    Parser --> LookUp[Look Up]
    Parser --> LookDown[Look Down]
    
    Forward --> CommandVector["[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"]
    Backward --> CommandVector2["[-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"]
    TurnLeft --> CommandVector3["[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]"]
    TurnRight --> CommandVector4["[0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]"]
    LookUp --> CommandVector5["[0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0]"]
    LookDown --> CommandVector6["[0.0, 0.0, 0.0, -0.3, 0.5, 0.0, 0.0]"]
    
    CommandVector --> RobotControl[Robot Control System]
    CommandVector2 --> RobotControl
    CommandVector3 --> RobotControl
    CommandVector4 --> RobotControl
    CommandVector5 --> RobotControl
    CommandVector6 --> RobotControl
```

## Files Overview

- `main.py`: Flask API server that interfaces with MuJoCo and exposes control endpoints
- `client.py`: Python client library for interacting with the API
- `vlm_integration.py`: Example implementation showing how a VLM can control the duck robot

## Installation

### Prerequisites

- Python 3.8+
- MuJoCo
- Open Duck Playground repository

### Setup

1. Install dependencies:

```bash
pip install -e .
# or individually
pip install flask flask-cors pillow requests opencv-python numpy mujoco etils
```

2. Ensure you have trained an ONNX model for the Open Duck robot (or use a pre-trained one)

## Usage

### Starting the API Server

```bash
python main.py --host 0.0.0.0 --port 5000
```

Additional options:
- `--debug`: Enable debug mode

### Using the Client Library

The `client.py` module provides a Python client for interacting with the duck robot:

```python
from client import DuckRobotClient

# Initialize client
client = DuckRobotClient(api_url="http://localhost:5000")

# Initialize the MuJoCo simulation
client.initialize(onnx_model_path="path/to/duck_policy.onnx")

# Send a command
# Format: [vx, vy, omega, neck_pitch, head_pitch, head_yaw, head_roll]
client.send_direct_command([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Move forward

# Get a camera frame
frame_data = client.get_frame()
with open("frame.png", "wb") as f:
    f.write(base64.b64decode(frame_data["image"]))

# Send a natural language command
client.send_nl_command("move forward and turn left")

# Shutdown when done
client.shutdown()
```

### Command-line Client

You can also use the client from the command line:

```bash
# Get status
python client.py status

# Initialize simulation
python client.py initialize --onnx-model-path path/to/duck_policy.onnx

# Send direct command (move forward)
python client.py direct --vx 0.1

# Send natural language command
python client.py nl "move forward and look left"

# Get a frame and save it
python client.py frame --output frame.png

# Shutdown
python client.py shutdown
```

### VLM Integration

The `vlm_integration.py` file demonstrates how to integrate a VLM with the Duck Robot:

```bash
# Run in interactive mode (natural language commands)
python vlm_integration.py --onnx-model path/to/duck_policy.onnx --interactive

# Use an external VLM API
python vlm_integration.py --onnx-model path/to/duck_policy.onnx --vlm-api https://your-vlm-api.com --vlm-key your_api_key
```

## API Endpoints

The main API server provides these endpoints:

- `GET /status`: Get the status of the MuJoCo simulation
- `POST /initialize`: Initialize the MuJoCo simulation
- `POST /shutdown`: Shutdown the MuJoCo simulation
- `POST /command`: Send a command to the Duck Robot
- `GET /frame`: Get the current camera frame
- `GET /help`: Get API documentation

## Command Format

Direct commands to the duck robot use a 7-element array:

```
[vx, vy, omega, neck_pitch, head_pitch, head_yaw, head_roll]
```

Where:
- `vx`: Forward velocity (-0.15 to 0.2)
- `vy`: Lateral velocity (-0.2 to 0.2)
- `omega`: Angular velocity/turning (-1.0 to 1.0)
- `neck_pitch`: Neck pitch angle (-0.34 to 1.1)
- `head_pitch`: Head pitch angle (-0.78 to 0.78)
- `head_yaw`: Head yaw angle (-1.5 to 1.5)
- `head_roll`: Head roll angle (-0.5 to 0.5)

## VLM Integration

To integrate your own VLM:

1. Implement an API endpoint that accepts:
   - An image (base64 encoded)
   - A text prompt

2. Update the `process_vlm_response_to_command` method in `VLMDuckController` to parse your VLM's response format

3. Use the `vlm_integration.py` script with your VLM API endpoint

## Debugging

- All components include detailed logging
- The VLM integration saves frames to disk for debugging
- Use the `--debug` flag for more verbose output

## Example VLM Workflow

```mermaid
flowchart TD
    Start([Start]) --> InitServer[Start API Server]
    InitServer --> InitClient[Initialize Client]
    InitClient --> InitSim[Initialize Simulation with ONNX model]
    
    InitSim --> Loop{Control Loop}
    
    Loop --> GetFrame[Get Camera Frame]
    GetFrame --> ProcessFrame[Send Frame to VLM]
    ProcessFrame --> ParseResponse[Parse VLM Response]
    ParseResponse --> SendCommand[Send Command to Robot]
    SendCommand --> Loop
    
    Loop -- End loop --> Shutdown[Shutdown Simulation]
    Shutdown --> End([End])
```

## Notes

- The MuJoCo simulation runs in a separate thread for improved performance
- Command queueing ensures smooth motion even with delayed VLM responses
- Natural language commands are mapped to direct commands using simple keyword matching
- Fixed initialization issue by setting default ONNX model path to "model.onnx"