# Open Duck VLM Integration

This project integrates Vision Language Models (VLMs) with the Open Duck Robot, enabling natural language control of the duck robot in the MuJoCo simulation environment.

## Project Overview

The integration allows users to:
1. Control the duck robot using natural language commands
2. Use vision-language models to interpret the robot's surroundings
3. Send appropriate commands based on visual input and language processing

## Architecture

```mermaid
graph TD
    subgraph "VLM Integration"
        A[VLM Duck Controller] --> B[Duck Robot Client]
        A --> C1[External VLM API]
        A --> C2[Local Ollama VLM]
        C2 --> D[Ollama Manager]
    end
    subgraph "Duck Robot API"
        B --> E[API Server]
        E --> F[MuJoCo Inference]
        F --> G[ONNX Model]
    end
    H[User] --> A
    
    style C1 stroke:#4285F4,stroke-width:2px
    style C2 stroke:#34A853,stroke-width:2px
    style D stroke:#34A853,stroke-width:2px
```

## Control Flow

```mermaid
sequenceDiagram
    participant User
    participant VLM as VLM Duck Controller
    participant Ollama as Ollama Manager
    participant API as Duck Robot API
    participant MuJoCo as MuJoCo Simulation

    User->>VLM: Launch with selected VLM type
    
    alt Local Ollama VLM
        VLM->>Ollama: Check if installed
        Ollama->>Ollama: Install if needed
        VLM->>Ollama: Start server
        Ollama->>Ollama: Pull model if needed
    else External VLM
        VLM->>VLM: Configure external API client
    end
    
    VLM->>API: Initialize simulation
    API->>MuJoCo: Setup simulation environment
    
    loop Control Loop
        VLM->>API: Get current frame
        API->>MuJoCo: Capture frame
        MuJoCo-->>API: Return frame
        API-->>VLM: Return frame
        
        alt Local Ollama VLM
            VLM->>Ollama: Query with image & prompt
            Ollama-->>VLM: Return response
        else External VLM
            VLM->>External: Query with image & prompt
            External-->>VLM: Return response
        end
        
        VLM->>VLM: Process response to command
        VLM->>API: Send command
        API->>MuJoCo: Execute command
    end
    
    User->>VLM: Stop simulation
    VLM->>API: Shutdown
    VLM->>Ollama: Stop server if started by us
```

## Natural Language Command Processing

```mermaid
flowchart LR
    A[Visual Input] --> B[VLM Processing]
    C[Text Prompt] --> B
    B --> D{Command Type}
    D -->|Movement| E[Forward/Backward/Left/Right]
    D -->|Rotation| F[Turn Left/Right]
    D -->|Camera| G[Look Up/Down/Left/Right]
    D -->|Stop| H[Halt]
    
    E --> I[Command Array Generation]
    F --> I
    G --> I
    H --> I
    
    I --> J[Send to Duck Robot API]
```

## Installation

### Prerequisites

- Python 3.8 or higher
- MuJoCo environment
- ONNX model for Duck Robot control

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/open_duck_vlm.git
   cd open_duck_vlm
   ```

2. Install dependencies using UV:
   ```bash
   uv venv
   uv pip install -e .
   ```

3. Set up a VLM:
   - Option 1: **Local Ollama VLM** (recommended for robot deployment)
     ```bash
     # Install and set up Ollama automatically
     uv run ollama-manager install
     
     # Or manually from https://ollama.com
     ```
     
   - Option 2: External VLM API (cloud-based solution)
     - Obtain API access to a vision-language model service

## Usage

### Running with Ollama (Local VLM)

```bash
# Start with default settings
uv run ollama-run

# With specific model
uv run ollama-run --ollama-model=llava-next

# Check available Ollama models
uv run ollama-manager list

# Pull a specific model
uv run ollama-manager pull llava
```

### Running with External VLM API

```bash
uv run external-run --vlm-api=https://your-vlm-api.com/vision --vlm-key=your-api-key
```

### Interactive Mode

Run in interactive mode to send direct commands:

```bash
uv run interactive
```

## VLM Options

### Local Ollama VLM

The Ollama Manager utility (`ollama.py`) provides a convenient way to:

- Automatically install Ollama if not present
- Manage Ollama service lifecycle
- Pull and query vision-capable models
- Handle image-based prompts

Supported vision models:
- `llava` - Basic vision-language capabilities
- `bakllava` - Enhanced vision capabilities
- `llava-next` - Latest version with improved performance

### External VLM API

For higher-quality results or when local resources are constrained, you can connect to:
- OpenAI GPT-4 Vision
- Anthropic Claude 3 Vision
- Other compatible VLM APIs

## API Endpoints

The Duck Robot API provides the following endpoints:

- `GET /status`: Check the status of the simulation
- `POST /initialize`: Initialize the simulation with an ONNX model
- `POST /command`: Send a command to the duck robot
- `POST /nl_command`: Send a natural language command
- `GET /frame`: Get the current frame from the simulation
- `POST /shutdown`: Shutdown the simulation

## Project Structure

```
open_duck_vlm/
├── playground/                # Duck robot simulation code
│   └── open_duck_mini_v2/     # Duck robot model
│       ├── mujoco_infer.py    # MuJoCo inference
│       └── common/            # Common utilities
├── main.py                    # API server
├── client.py                  # Client for the API server
├── vlm_integration.py         # VLM integration
├── ollama.py                  # Ollama manager utility
└── README.md                  # This file
```

## Development

For development, install the additional development dependencies:

```bash
uv pip install -e ".[dev]"
```

## Deployment Recommendations

For deploying on the actual robot:

1. Use the Ollama VLM option for local inference to reduce latency
2. Pre-download the required models before deployment
3. Ensure the robot has sufficient GPU resources for VLM inference
4. Configure automatic startup of the Ollama service

## License

[MIT License](LICENSE)