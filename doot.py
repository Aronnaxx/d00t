#!/usr/bin/env python
"""
Duck VLA Unified Runner (DOOT)

This script consolidates all Duck VLA simulation functionality into a single command.
It focuses on MuJoCo inference simulation with options for various configurations.

Usage:
    uv run doot.py [options]
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
import json
import time
import glob
import traceback

# Configure detailed logging for easier debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("duck_vla_runner")


def parse_arguments():
    """Parse command line arguments with all available options."""
    parser = argparse.ArgumentParser(
        description="Duck VLA Unified Runner (DOOT) - Simplified simulation system"
    )

    # Main mode options
    mode_group = parser.add_argument_group("Simulation Mode")
    mode_group.add_argument(
        "--cli-mode", action="store_true", help="Run with CLI control instead of ONNX model"
    )
    mode_group.add_argument(
        "--playground-only",
        action="store_true",
        help="Run the Open Duck Playground directly without Duck VLA",
    )

    # Input/output options
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument("--no-audio", action="store_true", help="Disable audio input/output")
    io_group.add_argument("--no-camera", action="store_true", help="Disable camera input")

    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--vision-model", type=str, default="gemma3", help="Vision model to use (default: gemma3)"
    )
    model_group.add_argument(
        "--onnx-model",
        type=str,
        help="Path to specific ONNX model file (will auto-detect if not specified)",
    )
    model_group.add_argument(
        "--verbose-vision",
        action="store_true",
        help="Enable verbose output for vision model interactions",
    )

    # Environment setup
    setup_group = parser.add_argument_group("Environment Setup")
    setup_group.add_argument(
        "--setup", action="store_true", help="Run setup to ensure environment is ready"
    )
    setup_group.add_argument(
        "--test-imports", action="store_true", help="Test if playground imports work correctly"
    )

    # Other options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def find_onnx_model():
    """
    Find an ONNX model in the expected locations.
    Returns path to the first valid ONNX model found.
    """
    logger.debug("Searching for ONNX model...")

    # Get workspace directory
    workspace_dir = Path(__file__).parent.absolute()

    # Check duck_vla/onnx directory first (preferred location)
    onnx_dir = workspace_dir / "duck_vla" / "onnx"
    if onnx_dir.exists() and onnx_dir.is_dir():
        onnx_files = list(onnx_dir.glob("*.onnx"))
        if onnx_files:
            model_path = str(onnx_files[0])
            logger.info(f"Found ONNX model in duck_vla/onnx: {onnx_files[0].name}")
            return model_path

    # Check root directory
    root_onnx_files = list(workspace_dir.glob("*.onnx"))
    if root_onnx_files:
        model_path = str(root_onnx_files[0])
        logger.info(f"Found ONNX model in root directory: {root_onnx_files[0].name}")
        return model_path

    # Check for specific named models in various locations
    model_names = ["BEST_WALK_ONNX_2.onnx", "policy_1.onnx", "duck_model.onnx"]
    for name in model_names:
        for search_dir in [workspace_dir, workspace_dir / "models", workspace_dir / "duck_vla"]:
            path = search_dir / name
            if path.exists():
                logger.info(f"Found ONNX model: {path}")
                return str(path)

    logger.warning("No ONNX model found in any of the expected locations")
    return None


def setup_ollama_model(model_name="gemma3", debug=False):
    """Set up the specified Ollama model."""
    logger.info(f"Setting up Ollama model: {model_name}")

    # Map vision model names to Ollama model names if needed
    model_mapping = {
        "gemma3": "gemma3:latest",
        "gemma": "gemma:latest",
        "llama3": "llama3:latest",
        "llama": "llama3:latest",
    }

    # Get the actual Ollama model name to use
    ollama_model = model_mapping.get(model_name, model_name)
    logger.debug(f"Mapped model name '{model_name}' to Ollama model '{ollama_model}'")

    try:
        import ollama

        client = ollama.Client()

        # Check if model exists
        models = client.list()
        model_exists = False

        if isinstance(models, dict) and "models" in models:
            for model in models["models"]:
                if isinstance(model, dict) and "name" in model and ollama_model in model["name"]:
                    logger.info(f"Ollama model '{ollama_model}' already exists")
                    model_exists = True
                    break

        # Pull model if needed
        if not model_exists:
            logger.info(f"Pulling Ollama model '{ollama_model}'...")
            client.pull(ollama_model)
            logger.info(f"Successfully pulled model '{ollama_model}'")

        # Set environment variable so other components know which model to use
        os.environ["DUCK_LLM_MODEL"] = ollama_model
        logger.debug(f"Set environment variable DUCK_LLM_MODEL={ollama_model}")

        return True

    except ImportError:
        logger.error("Ollama Python package not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
            logger.info("Ollama package installed. Please run again.")
        except Exception as e:
            logger.error(f"Failed to install Ollama package: {e}")
        return False

    except Exception as e:
        logger.error(f"Error setting up Ollama model: {e}")
        logger.error("Ensure Ollama server is running with 'ollama serve'")
        return False


def check_environment():
    """Check if the environment is set up properly."""
    logger.info("Checking environment...")

    # Check if playground submodule exists
    workspace_dir = Path(__file__).parent.absolute()
    playground_path = workspace_dir / "submodules" / "open_duck_playground"

    if not playground_path.exists():
        logger.error(f"Open Duck Playground not found at {playground_path}")
        logger.error("Please run ./setup_duck_vla.sh to set up the environment")
        return False

    logger.info("Environment check passed")
    return True


def test_playground_imports():
    """Test if playground imports work correctly."""
    logger.info("Testing playground imports...")

    workspace_dir = Path(__file__).parent.absolute()
    playground_path = workspace_dir / "submodules" / "open_duck_playground"

    if not playground_path.exists():
        logger.error(f"Playground directory not found at {playground_path}")
        return False

    # Add playground to path temporarily for testing
    sys.path.insert(0, str(playground_path))

    try:
        import playground
        import playground.open_duck_mini_v2
        import playground.open_duck_mini_v2.joystick

        logger.info("Successfully imported playground modules")
        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    finally:
        # Remove from path
        if str(playground_path) in sys.path:
            sys.path.remove(str(playground_path))


def run_mujoco_simulation(onnx_model_path, debug=False):
    """Run the MuJoCo simulation with the specified ONNX model."""
    logger.info("Running MuJoCo simulation...")

    workspace_dir = Path(__file__).parent.absolute()
    playground_path = workspace_dir / "submodules" / "open_duck_playground"

    if not playground_path.exists():
        logger.error(f"Playground directory not found at {playground_path}")
        return 1

    # Change to playground directory
    os.chdir(playground_path)
    logger.debug(f"Changed directory to: {playground_path}")

    # Build command
    cmd = ["uv", "run", "playground/open_duck_mini_v2/mujoco_infer.py", "-o", onnx_model_path]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        process = subprocess.run(cmd, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, terminating...")
        return 0


def run_cli_mode(
    vision_model="gemma3", no_camera=False, no_audio=False, debug=False, verbose_vision=False
):
    """Run Duck VLA with CLI control."""
    logger.info("Running Duck VLA with CLI control...")

    workspace_dir = Path(__file__).parent.absolute()
    playground_path = workspace_dir / "submodules" / "open_duck_playground"

    if not playground_path.exists():
        logger.error(f"Playground directory not found at {playground_path}")
        return 1

    # Find an ONNX model first
    onnx_model_path = find_onnx_model()
    if not onnx_model_path:
        logger.error("No ONNX model found. Please place one in duck_vla/onnx directory")
        return 1

    logger.info(f"Using ONNX model: {onnx_model_path}")

    # Check if playground is already running
    logger.info("Checking for existing playground simulation...")
    # This is just a simple check - if needed, you can add more sophisticated detection
    playground_running = False
    try:
        # A very simple check for now - this could be improved
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            cmd = proc.info.get("cmdline", [])
            if cmd and any("mujoco_infer.py" in arg for arg in cmd if arg):
                logger.info(f"Found existing playground process: {proc.info['pid']}")
                playground_running = True
                break
    except ImportError:
        logger.warning("psutil not installed, can't check for existing processes")
        # Assume playground is running since the user mentioned it
        playground_running = True

    # Start MuJoCo visualization in background if not already running
    mujoco_process = None
    if not playground_running:
        logger.info("Starting MuJoCo visualization in background...")
        try:
            # Save current directory to restore it later
            original_dir = os.getcwd()

            # Change to playground directory
            os.chdir(playground_path)

            # Build command for MuJoCo visualization
            mujoco_cmd = [
                "uv",
                "run",
                "playground/open_duck_mini_v2/mujoco_infer.py",
                "-o",
                onnx_model_path,
            ]

            logger.info(f"Running MuJoCo command: {' '.join(mujoco_cmd)}")

            # Start MuJoCo process in background
            mujoco_process = subprocess.Popen(
                mujoco_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Give MuJoCo time to start
            logger.info("Waiting for MuJoCo to initialize...")
            time.sleep(2)

            # Change back to original directory
            os.chdir(original_dir)

            # Check if MuJoCo process is still running
            if mujoco_process.poll() is not None:
                returncode = mujoco_process.poll()
                stdout, stderr = mujoco_process.communicate()
                logger.error(f"MuJoCo process failed with code {returncode}")
                logger.error(f"Stdout: {stdout.decode('utf-8')}")
                logger.error(f"Stderr: {stderr.decode('utf-8')}")
                return 1

            logger.info("MuJoCo visualization started successfully")
        except Exception as e:
            logger.error(f"Failed to start MuJoCo visualization: {e}")
            return 1
    else:
        logger.info("Using existing playground simulation - no need to start a new one")

    # Set up environment for CLI control
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{playground_path}:{env.get('PYTHONPATH', '')}"
    env["DUCK_VISION_MODEL"] = vision_model
    env["DUCK_ONNX_MODEL"] = os.path.abspath(onnx_model_path)
    
    # Always set DUCK_CONNECT_EXISTING to 1 - this is crucial to prevent two MuJoCo instances
    # The first one is started by doot.py, the second one should connect to the first
    env["DUCK_CONNECT_EXISTING"] = "1"
    logger.debug("Set DUCK_CONNECT_EXISTING=1 to ensure VLA connects to existing MuJoCo")

    # Configure verbose vision output if requested
    if verbose_vision:
        # Set environment variable for detailed vision output
        env["DUCK_VERBOSE_VISION"] = "1"
        # Ensure vision module logs are at debug level
        env["DUCK_LOG_LEVEL"] = "DEBUG"
        logger.info("Verbose vision output enabled")

    # Build command for CLI control
    cmd = ["uv", "run", "-m", "duck_vla.run_duck", "--simulate"]

    if no_camera:
        cmd.append("--no-camera")
    if no_audio:
        cmd.append("--no-audio")
    if debug:
        cmd.append("--debug")

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        process = subprocess.run(cmd, env=env, check=True)
        return_code = process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return_code = e.returncode
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, terminating...")
        return_code = 0
    finally:
        # Terminate MuJoCo process when CLI is closed if we started it
        if mujoco_process and mujoco_process.poll() is None:
            logger.info("Terminating MuJoCo visualization process...")
            try:
                mujoco_process.terminate()
                mujoco_process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error terminating MuJoCo process: {e}")

    return return_code


def run_playground_directly(debug=False):
    """Run the Open Duck Playground directly."""
    logger.info("Running Open Duck Playground directly...")

    workspace_dir = Path(__file__).parent.absolute()
    playground_path = workspace_dir / "submodules" / "open_duck_playground"

    if not playground_path.exists():
        logger.error(f"Playground directory not found at {playground_path}")
        return 1

    # Change directory to playground
    os.chdir(playground_path)

    # Build command
    cmd = ["uv", "run", "playground/open_duck_mini_v2/runner.py"]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        process = subprocess.run(cmd, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, terminating...")
        return 0


def setup_environment(debug=False):
    """Set up the Duck VLA environment."""
    logger.info("Setting up Duck VLA environment...")

    workspace_dir = Path(__file__).parent.absolute()

    # Create onnx directory if it doesn't exist
    onnx_dir = workspace_dir / "duck_vla" / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # Create README if it doesn't exist
    readme_path = onnx_dir / "README.md"
    if not readme_path.exists():
        with open(readme_path, "w") as f:
            f.write(
                """# ONNX Models Directory

Place your ONNX model files (.onnx) in this directory. 
The Duck VLA system will use the first .onnx file it finds in this directory.

You can download pre-trained models from the Open Duck GitHub repository.
"""
            )

    # Check if submodules directory exists
    submodules_dir = workspace_dir / "submodules"
    if not submodules_dir.exists():
        submodules_dir.mkdir(parents=True, exist_ok=True)

    # Check for open_duck_playground
    playground_path = submodules_dir / "open_duck_playground"
    if not playground_path.exists():
        logger.info("Cloning open_duck_playground repository...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/open-duck/open-duck-playground.git",
                    str(playground_path),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            logger.error(
                "Please manually download from https://github.com/open-duck/open-duck-playground"
            )
            return False

    # Install required packages
    logger.info("Installing required packages...")
    try:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "-U",
                "ollama",
                "transformers",
                "torch",
                "numpy",
                "pillow",
                "opencv-python",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install required packages: {e}")
        return False

    # Install open_duck_playground
    logger.info("Installing open_duck_playground...")
    try:
        subprocess.run(["uv", "pip", "install", "-e", "."], cwd=playground_path, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install open_duck_playground: {e}")
        return False

    logger.info("Environment setup completed successfully")
    return True


def main():
    """Main entry point for Duck VLA unified runner."""
    args = parse_arguments()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
        # Log all arguments
        logger.debug(f"Command line arguments: {json.dumps(vars(args), indent=2)}")

    # Run setup if requested
    if args.setup:
        if not setup_environment(debug=args.debug):
            return 1

    # Test imports if requested
    if args.test_imports:
        if not test_playground_imports():
            logger.error("Playground imports test failed")
            return 1
        logger.info("Playground imports test passed")
        if not args.playground_only and not args.cli_mode:
            # If only testing imports, exit now
            return 0

    # Check environment
    if not check_environment():
        logger.error("Environment check failed. Run with --setup to set up the environment.")
        return 1

    # Set up Ollama model if needed
    if not args.playground_only:
        if not setup_ollama_model(args.vision_model, args.debug):
            logger.error(f"Failed to set up Ollama model: {args.vision_model}")
            return 1

    # Determine which mode to run
    if args.playground_only:
        # Run Open Duck Playground directly
        return run_playground_directly(debug=args.debug)

    elif args.cli_mode:
        # Run Duck VLA with CLI control
        return run_cli_mode(
            vision_model=args.vision_model,
            no_camera=args.no_camera,
            no_audio=args.no_audio,
            debug=args.debug,
            verbose_vision=args.verbose_vision,
        )

    else:
        # Default: Run MuJoCo simulation with ONNX model
        onnx_model_path = args.onnx_model
        if not onnx_model_path:
            onnx_model_path = find_onnx_model()
            if not onnx_model_path:
                logger.error("No ONNX model found. Please provide one with --onnx-model")
                logger.error("or place a model in duck_vla/onnx directory")
                return 1

        # Run MuJoCo simulation with ONNX model
        return run_mujoco_simulation(onnx_model_path, debug=args.debug)


if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    elapsed_time = time.time() - start_time
    logger.info(f"Execution completed in {elapsed_time:.2f} seconds with exit code {exit_code}")
    sys.exit(exit_code)
