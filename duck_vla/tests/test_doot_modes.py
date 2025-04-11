#!/usr/bin/env python
"""
Test for execution modes functionality in doot.py

This tests the run_mujoco_simulation, run_cli_mode, and run_playground_directly
functions to ensure they correctly execute the respective modes.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add parent directory to Python path to import doot module
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import doot


class TestDootModes(unittest.TestCase):
    """Test class for execution modes functionality in doot.py"""

    @patch("doot.Path.exists")
    @patch("doot.os.chdir")
    @patch("doot.subprocess.run")
    def test_run_mujoco_simulation(self, mock_run, mock_chdir, mock_exists):
        """Test that run_mujoco_simulation executes the correct command"""
        # Mock the existence of the playground directory
        mock_exists.return_value = True

        # Mock subprocess run to return a CompletedProcess with returncode 0
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # Call function
        onnx_model_path = "/path/to/model.onnx"
        result = doot.run_mujoco_simulation(onnx_model_path, debug=True)

        # Check result
        self.assertEqual(result, 0)

        # Verify the correct directory was changed to
        mock_chdir.assert_called_once()
        chdir_args = mock_chdir.call_args[0][0]
        self.assertTrue("submodules/open_duck_playground" in str(chdir_args))

        # Verify the correct command was run
        mock_run.assert_called_once()
        run_args = mock_run.call_args[0][0]
        self.assertTrue("mujoco_infer.py" in run_args[2])
        self.assertEqual(run_args[3], "-o")
        self.assertEqual(run_args[4], "/path/to/model.onnx")
        self.assertEqual(run_args[5], "--verbose")  # Debug flag should add --verbose

    @patch("doot.Path.exists")
    @patch("doot.os.chdir")
    @patch("doot.subprocess.run")
    def test_run_mujoco_simulation_error(self, mock_run, mock_chdir, mock_exists):
        """Test that run_mujoco_simulation handles errors correctly"""
        # Mock the existence of the playground directory
        mock_exists.return_value = True

        # Mock subprocess run to raise CalledProcessError
        mock_run.side_effect = doot.subprocess.CalledProcessError(1, "command")

        # Call function
        onnx_model_path = "/path/to/model.onnx"
        result = doot.run_mujoco_simulation(onnx_model_path)

        # Check result
        self.assertEqual(result, 1)

    @patch("doot.Path.exists")
    @patch("doot.subprocess.run")
    def test_run_cli_mode(self, mock_run, mock_exists):
        """Test that run_cli_mode executes the correct command"""
        # Mock the existence of the playground directory
        mock_exists.return_value = True

        # Mock subprocess run to return a CompletedProcess with returncode 0
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # Call function with all options
        result = doot.run_cli_mode(
            vision_model="custom_model", no_camera=True, no_audio=True, debug=True
        )

        # Check result
        self.assertEqual(result, 0)

        # Verify the correct command was run with all options
        mock_run.assert_called_once()
        run_args = mock_run.call_args[0][0]
        run_env = mock_run.call_args[1]["env"]

        self.assertEqual(run_args[0:4], ["uv", "run", "-m", "duck_vla.run_duck"])
        self.assertTrue("--simulate" in run_args)
        self.assertTrue("--no-camera" in run_args)
        self.assertTrue("--no-audio" in run_args)
        self.assertTrue("--debug" in run_args)
        self.assertEqual(run_env["DUCK_VISION_MODEL"], "custom_model")

    @patch("doot.Path.exists")
    @patch("doot.os.chdir")
    @patch("doot.subprocess.run")
    def test_run_playground_directly(self, mock_run, mock_chdir, mock_exists):
        """Test that run_playground_directly executes the correct command"""
        # Mock the existence of the playground directory
        mock_exists.return_value = True

        # Mock subprocess run to return a CompletedProcess with returncode 0
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # Call function
        result = doot.run_playground_directly(debug=True)

        # Check result
        self.assertEqual(result, 0)

        # Verify the correct directory was changed to
        mock_chdir.assert_called_once()
        chdir_args = mock_chdir.call_args[0][0]
        self.assertTrue("submodules/open_duck_playground" in str(chdir_args))

        # Verify the correct command was run
        mock_run.assert_called_once()
        run_args = mock_run.call_args[0][0]
        self.assertEqual(run_args[0:2], ["uv", "run"])
        self.assertTrue("runner.py" in run_args[2])

    @patch("doot.Path.exists")
    def test_run_playground_directly_no_playground(self, mock_exists):
        """Test that run_playground_directly handles missing playground"""
        # Mock the non-existence of the playground directory
        mock_exists.return_value = False

        # Call function
        result = doot.run_playground_directly()

        # Check result
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
