"""
CLI Controller for Duck VLA system

This module provides a command-line interface to control the Duck VLA system
through typed commands instead of voice.
"""

import logging
import time
import threading
import queue
import cmd
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class DuckCLI(cmd.Cmd):
    """
    Command-line interface for controlling the Duck VLA system.
    
    This allows direct text input of commands instead of using voice commands.
    """
    
    prompt = "duck> "
    intro = "Duck VLA CLI Controller. Type 'help' for available commands."
    
    def __init__(self, command_queue: queue.Queue):
        """
        Initialize the CLI controller.
        
        Args:
            command_queue: Queue to put commands into for processing by the main loop
        """
        super().__init__()
        self.command_queue = command_queue
        self.running = False
        logger.debug("CLI controller initialized")
    
    def do_move(self, arg):
        """
        Move the duck in a direction.
        
        Usage: 
          move forward [speed] [duration]
          move backward [speed] [duration]
          move left [speed] [duration]
          move right [speed] [duration]
          
        Examples:
          move forward        (Move forward at default speed continuously)
          move forward 0.7    (Move forward at 70% speed continuously)
          move forward 0.5 2  (Move forward at 50% speed for 2 seconds)
        """
        args = arg.split()
        if not args:
            print("Error: Direction required. Use 'forward', 'backward', 'left', or 'right'.")
            return
        
        direction = args[0].lower()
        if direction not in ["forward", "backward", "left", "right"]:
            print(f"Error: Unknown direction '{direction}'. Use 'forward', 'backward', 'left', or 'right'.")
            return
        
        speed = 0.5  # Default speed
        duration = None  # Default duration (continuous)
        
        # Parse speed if provided
        if len(args) > 1:
            try:
                speed = float(args[1])
                speed = max(0.0, min(1.0, speed))  # Clamp to 0.0-1.0
            except ValueError:
                print(f"Error: Invalid speed '{args[1]}'. Use a number between 0.0 and 1.0.")
                return
        
        # Parse duration if provided
        if len(args) > 2:
            try:
                duration = float(args[2])
                if duration <= 0:
                    print("Error: Duration must be positive.")
                    return
            except ValueError:
                print(f"Error: Invalid duration '{args[2]}'. Use a positive number.")
                return
        
        # Create and queue command
        command = {
            "action_type": "move",
            "params": {
                "direction": direction,
                "speed": speed,
                "duration": duration
            }
        }
        
        self.command_queue.put(command)
        duration_str = f" for {duration}s" if duration else " continuously"
        print(f"Moving {direction} at {speed*100:.0f}% speed{duration_str}")
    
    def do_turn(self, arg):
        """
        Turn the duck.
        
        Usage:
          turn left [rate] [angle]
          turn right [rate] [angle]
          turn around [rate]
          
        Examples:
          turn left           (Turn left at default rate continuously)
          turn right 0.7      (Turn right at 70% rate continuously)
          turn left 0.5 90    (Turn left at 50% rate for 90 degrees)
          turn around         (Turn around 180 degrees)
        """
        args = arg.split()
        if not args:
            print("Error: Direction required. Use 'left', 'right', or 'around'.")
            return
        
        direction = args[0].lower()
        if direction not in ["left", "right", "around"]:
            print(f"Error: Unknown direction '{direction}'. Use 'left', 'right', or 'around'.")
            return
        
        rate = 0.5  # Default rate
        angle = None  # Default angle (continuous)
        
        if direction == "around":
            angle = 180.0  # Default angle for "around"
        
        # Parse rate if provided
        if len(args) > 1:
            try:
                rate = float(args[1])
                rate = max(0.0, min(1.0, rate))  # Clamp to 0.0-1.0
            except ValueError:
                print(f"Error: Invalid rate '{args[1]}'. Use a number between 0.0 and 1.0.")
                return
        
        # Parse angle if provided
        if len(args) > 2 and direction != "around":  # "around" already has a fixed angle
            try:
                angle = float(args[2])
                if angle <= 0:
                    print("Error: Angle must be positive.")
                    return
            except ValueError:
                print(f"Error: Invalid angle '{args[2]}'. Use a positive number.")
                return
        
        # Create and queue command
        command = {
            "action_type": "turn",
            "params": {
                "direction": direction,
                "rate": rate,
                "angle": angle
            }
        }
        
        self.command_queue.put(command)
        angle_str = f" by {angle}°" if angle else " continuously"
        print(f"Turning {direction} at {rate*100:.0f}% rate{angle_str}")
    
    def do_look(self, arg):
        """
        Point the duck's head.
        
        Usage:
          look at <target>       (Look at named target: person, up, down, left, right)
          look yaw,pitch,roll    (Set specific head angles in degrees)
          
        Examples:
          look at person         (Look at a person)
          look at up             (Look up)
          look 30,10,0           (Look 30° right, 10° up, no roll)
        """
        if not arg:
            print("Error: Target or angles required. Use 'look at <target>' or 'look yaw,pitch,roll'.")
            return
        
        parts = arg.split()
        
        # Handle "look at <target>"
        if parts[0].lower() == "at" and len(parts) > 1:
            target = parts[1].lower()
            valid_targets = ["person", "up", "down", "left", "right"]
            if target not in valid_targets:
                print(f"Error: Unknown target '{target}'. Valid targets: {', '.join(valid_targets)}")
                return
            
            command = {
                "action_type": "look_at",
                "params": {
                    "target": target
                }
            }
            
            self.command_queue.put(command)
            print(f"Looking at {target}")
            return
        
        # Handle direct angle specification
        try:
            angles = [float(a.strip()) for a in arg.split(",")]
            if len(angles) != 3:
                print("Error: When specifying angles directly, provide all three: yaw,pitch,roll")
                return
            
            yaw, pitch, roll = angles
            
            # Validate angle ranges
            if not -45 <= yaw <= 45:
                print("Warning: Yaw should be between -45 and 45 degrees")
            if not -30 <= pitch <= 30:
                print("Warning: Pitch should be between -30 and 30 degrees")
            if not -20 <= roll <= 20:
                print("Warning: Roll should be between -20 and 20 degrees")
            
            command = {
                "action_type": "look_at",
                "params": {
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll
                }
            }
            
            self.command_queue.put(command)
            print(f"Setting head position to yaw={yaw}°, pitch={pitch}°, roll={roll}°")
            
        except ValueError:
            print("Error: Invalid angle format. Use comma-separated numbers: yaw,pitch,roll")
    
    def do_emote(self, arg):
        """
        Play an emote or sound.
        
        Usage:
          emote <name>
          
        Examples:
          emote happy
          emote confused
          emote hello
        """
        if not arg:
            print("Error: Emote name required.")
            return
        
        emote_name = arg.strip().lower()
        command = {
            "action_type": "emote",
            "params": {
                "emote": emote_name
            }
        }
        
        self.command_queue.put(command)
        print(f"Playing emote: {emote_name}")
    
    def do_stop(self, arg):
        """Stop all duck movement."""
        command = {
            "action_type": "stop",
            "params": {}
        }
        
        self.command_queue.put(command)
        print("Stopping all movement")
    
    def do_status(self, arg):
        """Get the current duck status."""
        command = {
            "action_type": "get_status",
            "params": {}
        }
        
        self.command_queue.put(command)
        print("Requesting status...")
    
    def do_exit(self, arg):
        """Exit the CLI controller."""
        print("Exiting CLI controller...")
        self.running = False
        return True
    
    def do_quit(self, arg):
        """Exit the CLI controller."""
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Exit on Ctrl+D."""
        print()  # Print newline before exiting
        return self.do_exit(arg)

class CLIController:
    """
    Controller for managing the CLI interface in a separate thread.
    """
    
    def __init__(self):
        """Initialize the CLI controller."""
        self.command_queue = queue.Queue()
        self.cli = DuckCLI(self.command_queue)
        self.cli_thread = None
        self.running = False
        logger.info("CLI controller initialized")
    
    def start(self):
        """Start the CLI controller in a separate thread."""
        if self.running:
            logger.warning("CLI controller already running")
            return
        
        self.running = True
        self.cli.running = True
        self.cli_thread = threading.Thread(target=self._run_cli, daemon=True)
        self.cli_thread.start()
        logger.info("CLI controller started")
    
    def _run_cli(self):
        """Run the CLI loop."""
        try:
            self.cli.cmdloop()
        except Exception as e:
            logger.exception(f"Error in CLI thread: {e}")
        finally:
            logger.info("CLI thread exiting")
            self.running = False
    
    def get_command(self, timeout=0.1):
        """
        Get a command from the queue if available.
        
        Args:
            timeout: How long to wait for a command (seconds)
            
        Returns:
            Command dict or None if no command available
        """
        try:
            return self.command_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the CLI controller."""
        logger.info("Stopping CLI controller")
        self.running = False
        self.cli.running = False
        
        # If running in same thread (for testing), this would exit immediately
        if not self.cli_thread or not self.cli_thread.is_alive():
            return
        
        # Give the thread a chance to exit gracefully
        self.cli_thread.join(1.0)
        if self.cli_thread.is_alive():
            logger.warning("CLI thread did not exit gracefully") 