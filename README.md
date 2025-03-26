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
    
    subgraph "VLM Options"
        OllamaVLM[Local Ollama]
        ExternalVLM[External VLM API]
    end
    
    VLMIntegration -- uses --> OllamaVLM
    VLMIntegration -- uses --> ExternalVLM
    OllamaVLM -- analyzes --> CF
    ExternalVLM -- analyzes --> CF
    OllamaVLM -- issues commands to --> Robot
    ExternalVLM -- issues commands to --> Robot
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
    Parser --> LookLeft[Look Left]
    Parser --> LookRight[Look Right]
    Parser --> StrafeLeft[Strafe Left]
    Parser --> StrafeRight[Strafe Right]
    Parser --> Stop[Stop]
    
    Forward --> CommandVector["[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"]
    Backward --> CommandVector2["[-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"]
    TurnLeft --> CommandVector3["[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]"]
    TurnRight --> CommandVector4["[0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]"]
    StrafeLeft --> CommandVector5["[0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]"]
    StrafeRight --> CommandVector6["[0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0]"]
    LookUp --> CommandVector7["[0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0]"]
    LookDown --> CommandVector8["[0.0, 0.0, 0.0, -0.3, 0.5, 0.0, 0.0]"]
    LookLeft --> CommandVector9["[0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0]"]
    LookRight --> CommandVector10["[0.0, 0.0, 0.0, 0.0, 0.0, -0.8, 0.0]"]
    Stop --> CommandVector11["[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"]
    
    CommandVector --> RobotControl[Robot Control System]
    CommandVector2 --> RobotControl
    CommandVector3 --> RobotControl
    CommandVector4 --> RobotControl
    CommandVector5 --> RobotControl
    CommandVector6 --> RobotControl
    CommandVector7 --> RobotControl
    CommandVector8 --> RobotControl
    CommandVector9 --> RobotControl
    CommandVector10 --> RobotControl
    CommandVector11 --> RobotControl
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
- For local VLM: [Ollama](https://github.com/ollama/ollama) with llava or similar multimodal model

### Setup

1. Install dependencies:

```bash
pip install -e .
# or individually
pip install flask flask-cors pillow requests opencv-python numpy mujoco etils
```

2. Ensure you have trained an ONNX model for the Open Duck robot (or use a pre-trained one)

3. If using Ollama locally, install it:

```bash
# Linux
curl https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Windows
# Download installer from https://ollama.ai/download
```

4. Pull a multimodal model in Ollama:

```bash
ollama pull llava
```

## Usage

### Starting the API Server

```bash
# Using Python directly
python main.py --host 0.0.0.0 --port 5000

# Using UV
uv run main
```

Additional options:
- `--debug`: Enable debug mode

### Running with Local Ollama VLM (Recommended for Robot Deployment)

```bash
# Using Python directly
python vlm_integration.py --vlm-type=ollama --ollama-model=llava --onnx-model=path/to/model.onnx

# Using UV
uv run ollama-run
```

### Running with External VLM API

```bash
# Using Python directly
python vlm_integration.py --vlm-type=external --vlm-api=https://your-api-endpoint --onnx-model=path/to/model.onnx

# Using UV
uv run external-run https://your-api-endpoint
```

### Interactive Mode

```bash
# Using Python directly
python vlm_integration.py --interactive --onnx-model=path/to/model.onnx

# Using UV
uv run interactive
```

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

## VLM Integration Options

### Local Ollama (Recommended for Deployment)

For deployment on the robot itself, we recommend using Ollama for local inference:

```mermaid
graph TD
    subgraph "Robot"
        API[API Server]
        MuJoCo[MuJoCo Simulation]
        Ollama[Ollama VLM]
        Camera[Camera Feed]
        
        Camera --> API
        API --> MuJoCo
        API --> Ollama
        Ollama --> API
    end
    
    RemoteClient[Remote Client] --> API
```

Benefits:
- No internet connection required
- Lower latency
- Privacy (no data leaves the robot)
- Can run offline

Configuration options:
- `--ollama-url`: URL where Ollama is running (default: http://localhost:11434)
- `--ollama-model`: Model to use (default: llava)

### External VLM API

For development or to use more powerful models:

```mermaid
graph TD
    subgraph "Robot"
        API[API Server]
        MuJoCo[MuJoCo Simulation]
        Camera[Camera Feed]
    end
    
    subgraph "Cloud"
        VLM[External VLM]
    end
    
    Camera --> API
    API --> MuJoCo
    API --> VLM
    VLM --> API
```

Configuration options:
- `--vlm-api`: URL of the external VLM API
- `--vlm-key`: API key for authentication (if required)

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
    
    InitSim --> VLMChoice{Choose VLM Type}
    VLMChoice -->|Local| InitOllama[Initialize Ollama]
    VLMChoice -->|External| ConfigAPI[Configure External API]
    
    InitOllama --> Loop
    ConfigAPI --> Loop
    
    Loop{Control Loop} --> GetFrame[Get Camera Frame]
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
- For deployment on a robot, the Ollama integration provides local inference capabilities