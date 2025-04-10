#!/usr/bin/env python
"""
Test Duck Playground Imports

This script tests if the Open Duck Playground modules can be imported
correctly when the PYTHONPATH is set appropriately.
"""

import sys
import os
from pathlib import Path

# Add the Open Duck Playground to the Python path
playground_path = Path(__file__).parent / "submodules" / "open_duck_playground"
sys.path.insert(0, str(playground_path))

print(f"Testing Duck Playground imports from: {playground_path}")
print(f"Python path: {sys.path}")

try:
    print("\nTrying import: playground")
    import playground
    print(f"Success! playground.__file__: {playground.__file__}")
    
    print("\nTrying import: playground.open_duck_mini_v2")
    import playground.open_duck_mini_v2
    print(f"Success! playground.open_duck_mini_v2.__file__: {playground.open_duck_mini_v2.__file__}")
    
    print("\nTrying import: playground.open_duck_mini_v2.joystick")
    import playground.open_duck_mini_v2.joystick
    print(f"Success! playground.open_duck_mini_v2.joystick.__file__: {playground.open_duck_mini_v2.joystick.__file__}")
    
    print("\nTrying to access JoystickInterface class")
    if hasattr(playground.open_duck_mini_v2.joystick, 'JoystickInterface'):
        print("JoystickInterface class found!")
    else:
        print("JoystickInterface class not found.")
        # Try to list available attributes
        print(f"Available attributes: {dir(playground.open_duck_mini_v2.joystick)}")
        
    # Check if the Joystick class exists (which is different from JoystickInterface)
    if hasattr(playground.open_duck_mini_v2.joystick, 'Joystick'):
        print("\nJoystick class found!")
        joystick_class = playground.open_duck_mini_v2.joystick.Joystick
        print(f"Joystick class: {joystick_class}")
    
    print("\nAll imports successful!")
    
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
    
print("\nTest completed successfully.") 