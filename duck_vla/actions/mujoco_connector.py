"""
Mujoco Connector - Bridge between Movement class and MjInfer Mujoco simulation

This module connects the duck_vla Movement controller with the Mujoco simulation
from the Open Duck Playground.
"""

import logging
import threading
import time
import os
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class MujocoConnector:
    """
    Connects the Movement controller to the Mujoco simulation.

    This class handles starting the Mujoco simulation in a separate thread
    and forwarding key commands from the Movement class to the simulation.
    """

    def __init__(
        self,
        onnx_model_path: Optional[str] = None,
        debug: bool = False,
        connect_to_existing: bool = False,
    ):
        """
        Initialize the Mujoco connector.

        Args:
            onnx_model_path: Path to ONNX model to use for simulation
            debug: Enable debug logging
            connect_to_existing: If True, don't start a new simulation, connect to existing one
        """
        self.debug = debug
        self.connect_to_existing = connect_to_existing

        if debug:
            logger.setLevel(logging.DEBUG)

        # Check if we should connect to an existing simulation
        if self.connect_to_existing:
            logger.info("Configured to connect to existing MuJoCo simulation")

        # Set default ONNX model path if none is provided
        self.onnx_model_path = onnx_model_path
        if not self.onnx_model_path:
            # Try to find the ONNX model in common locations
            potential_paths = [
                # Check current directory
                os.path.join(os.getcwd(), "BEST_WALK_ONNX_2.onnx"),
                # Check in duck_vla/onnx directory
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "onnx", "BEST_WALK_ONNX_2.onnx"
                ),
                # Check in root directory
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "BEST_WALK_ONNX_2.onnx",
                ),
            ]

            for path in potential_paths:
                if os.path.exists(path):
                    self.onnx_model_path = path
                    logger.info(f"Found ONNX model at: {path}")
                    break

            if not self.onnx_model_path:
                logger.error("No ONNX model path provided and couldn't find one automatically")
                raise FileNotFoundError(
                    "ONNX model not found. Please specify path or place it in a standard location"
                )

        # Import MjInfer class from open_duck_playground
        try:
            from playground.open_duck_mini_v2.mujoco_infer import MjInfer

            self.MjInfer = MjInfer
            logger.info("Successfully imported MjInfer from open_duck_playground")
        except ImportError as e:
            logger.error(f"Failed to import MjInfer: {e}")
            logger.error("Make sure open_duck_playground is in your PYTHONPATH")
            raise ImportError("Failed to import MjInfer from open_duck_playground") from e

        # Initialize simulation properties
        self.simulation = None
        self.simulation_thread = None
        self.running = False
        self.movement_controller = None

        logger.info("MujocoConnector initialized successfully")

    def start_simulation(self, standing: bool = False) -> bool:
        """
        Start the Mujoco simulation in a separate thread.

        Args:
            standing: Start in standing mode (vs. walking mode)

        Returns:
            True if simulation was started successfully
        """
        if self.running:
            logger.warning("Simulation is already running")
            return False

        # If we're configured to connect to an existing simulation, skip starting a new one
        if self.connect_to_existing:
            logger.info("Using existing MuJoCo simulation instead of starting new one")
            self.running = True

            # Set up a mock simulation object with the expected interface
            # This will allow our code to work without a direct simulation connection
            # The actual key events will be handled through the movement controller's callback
            from types import SimpleNamespace

            self.simulation = SimpleNamespace(
                commands={},
                head_control_mode=False,
                key_callback=lambda keycode: logger.debug(
                    f"Mock key callback with code: {keycode}"
                ),
            )

            return True

        # Find the absolute paths to the model and reference data
        # Look for the files in the open_duck_playground submodule
        submodule_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "submodules",
            "open_duck_playground",
        )

        reference_data = os.path.join(
            submodule_path, "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
        )
        model_path = os.path.join(
            submodule_path, "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        )

        logger.debug(f"Using model path: {model_path}")
        logger.debug(f"Using reference data path: {reference_data}")

        # Verify that the files exist
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False

        if not os.path.exists(reference_data):
            logger.error(f"Reference data file not found: {reference_data}")
            return False

        try:
            # Create the MjInfer instance
            self.simulation = self.MjInfer(
                model_path=model_path,
                reference_data=reference_data,
                onnx_model_path=self.onnx_model_path,
                standing=standing,
            )

            # Override the key_callback method to forward commands from our Movement class
            original_key_callback = self.simulation.key_callback

            def wrapped_key_callback(keycode):
                """Wrapper around the original key_callback"""
                logger.debug(f"Key callback called with keycode: {keycode}")

                # Call the original callback
                original_key_callback(keycode)

                # Update the movement controller if it exists
                if self.movement_controller is not None:
                    # Update the movement controller's head control mode
                    if keycode == 72:  # 'h' key
                        self.movement_controller.head_control_mode = (
                            self.simulation.head_control_mode
                        )
                        logger.debug(
                            f"Updated movement controller head control mode: {self.movement_controller.head_control_mode}"
                        )

                    # Update current commands in the movement controller
                    self.movement_controller.current_commands = self.simulation.commands.copy()
                    logger.debug(
                        f"Updated movement controller commands: {self.movement_controller.current_commands}"
                    )

            # Replace the key_callback with our wrapped version
            self.simulation.key_callback = wrapped_key_callback

            # Start the simulation in a separate thread
            self.simulation_thread = threading.Thread(target=self._run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()

            self.running = True
            logger.info("Mujoco simulation started successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return False

    def _run_simulation(self):
        """Run the simulation (called in a separate thread)"""
        try:
            logger.info("Mujoco simulation thread started")
            self.simulation.run()
        except Exception as e:
            logger.error(f"Error in simulation thread: {e}")
        finally:
            logger.info("Mujoco simulation thread ended")
            self.running = False

    def connect_movement_controller(self, movement_controller) -> bool:
        """
        Connect a Movement controller to the Mujoco simulation.

        This allows the Movement controller to send commands to the simulation
        and receive updates from the simulation.

        Args:
            movement_controller: Movement controller to connect

        Returns:
            True if connection was successful
        """
        if not self.running:
            logger.error("Simulation not running, can't connect movement controller")
            return False

        if not movement_controller:
            logger.error("Movement controller is None, can't connect")
            return False

        # Store the movement controller reference
        self.movement_controller = movement_controller

        # Register our key_callback with the movement controller
        # This creates a bidirectional connection - the movement controller can send keys to us,
        # and we forward them to the simulation
        if self.connect_to_existing:
            # When connecting to an existing simulation, the movement controller's callback
            # is what does the actual work - it sends keyboard events to the window
            logger.info("Registering movement controller for connecting to existing simulation")
            # We don't need to register our callback, as the existing window is managed separately
        else:
            # When running our own simulation, we need to register our callback
            logger.info("Registering simulation callback with movement controller")
            movement_controller.register_sim_callback(self.simulation.key_callback)

        logger.info("Movement controller connected to Mujoco simulation")
        return True

    def is_running(self) -> bool:
        """
        Check if the simulation is running.

        Returns:
            True if the simulation is running
        """
        return (
            self.running
            and self.simulation_thread is not None
            and self.simulation_thread.is_alive()
        )

    def stop_simulation(self) -> bool:
        """
        Stop the Mujoco simulation.

        Returns:
            True if simulation was stopped successfully
        """
        if not self.running:
            logger.warning("Simulation is not running")
            return False

        # There's no clean way to stop the Mujoco simulation from outside,
        # as it runs in a while True loop with a KeyboardInterrupt exception handler
        # So we'll just set the running flag to False and let the thread terminate on its own
        self.running = False

        # Wait for the thread to terminate (with timeout)
        if self.simulation_thread is not None:
            self.simulation_thread.join(timeout=5)

            if self.simulation_thread.is_alive():
                logger.warning("Simulation thread did not terminate gracefully")
                return False

        logger.info("Mujoco simulation stopped successfully")
        return True
