"""
CLI Controller for Duck VLA system.

This module provides a command-line interface for controlling
the Duck VLA system through text commands.
"""

import cmd
import logging
import threading
import time
import queue
import re
from typing import Optional, Dict, Any, Callable, List, Tuple

logger = logging.getLogger(__name__)

class CLIController(cmd.Cmd):
    """
    Command-line interface for controlling the Duck VLA system.
    
    This class extends Python's cmd module to provide a command-line
    interface for sending commands to the Duck VLA system.
    """
    
    INTRO = "Duck VLA CLI Controller. Type 'help' for available commands."
    PROMPT = "duck> "
    
    def __init__(self, debug: bool = False):
        """
        Initialize the CLI controller.
        
        Args:
            debug: Whether to enable debug logging
        """
        super().__init__()
        
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
            
        self.intro = self.INTRO
        self.prompt = self.PROMPT
        
        # Queue for commands from CLI to decision loop
        self.command_queue = queue.Queue()
        
        # Thread for running the CLI
        self.cli_thread = None
        self.running = False
        
        # Central model for processing natural language
        self.central_model = None
        
        # Movement controller for direct commands
        self.movement = None
        
        logger.debug("CLI controller initialized")
        logger.info("CLI controller initialized")
        
    def start(self):
        """Start the CLI controller in a separate thread."""
        if self.cli_thread is not None and self.cli_thread.is_alive():
            logger.warning("CLI controller already running")
            return
            
        self.running = True
        self.cli_thread = threading.Thread(target=self._run_cli)
        self.cli_thread.daemon = True
        self.cli_thread.start()
        logger.info("CLI controller started")
        
    def stop(self):
        """Stop the CLI controller."""
        self.running = False
        if self.cli_thread and self.cli_thread.is_alive():
            # The thread will terminate on next prompt
            logger.debug("Waiting for CLI thread to terminate")
            self.cli_thread.join(timeout=1.0)
            
        logger.info("CLI controller stopped")
        
    def _run_cli(self):
        """Run the CLI loop in a separate thread."""
        while self.running:
            try:
                self.cmdloop()
                break
            except KeyboardInterrupt:
                print("\nKeyboard interrupt. Type 'exit' or 'quit' to exit.")
            except Exception as e:
                logger.error(f"Error in CLI loop: {e}")
                time.sleep(1)  # Prevent fast-spinning on error
                
    def set_central_model(self, central_model):
        """Set the central model for processing natural language"""
        self.central_model = central_model
        
    def set_movement_controller(self, movement):
        """Set the movement controller for direct commands"""
        self.movement = movement
    
    def get_command(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get the next command from the queue.
        
        Args:
            timeout: Maximum time to wait for a command
            
        Returns:
            Command dictionary or None if queue is empty
        """
        try:
            return self.command_queue.get(block=timeout is not None, timeout=timeout)
        except queue.Empty:
            return None
        
    def default(self, line: str) -> bool:
        """
        Handle unknown commands as natural language input.
        
        Args:
            line: Command line entered by the user
            
        Returns:
            True to continue, False to stop
        """
        # Check if central model is available for natural language processing
        if self.central_model:
            try:
                # Process the line as natural language
                response = self.central_model.process_natural_language(line)
                print(response)
                return True
            except Exception as e:
                logger.error(f"Error processing natural language: {e}")
                print(f"Error: {e}")
                print("Make sure Ollama is running with 'ollama serve' and a model is pulled.")
                return True
        else:
            print(f"Unknown syntax: {line}")
            print("Natural language processing not available. Try basic commands like 'move forward'.")
            return True
        
    def emptyline(self) -> bool:
        """Do nothing on empty line."""
        return True
        
    def do_exit(self, arg: str) -> bool:
        """Exit the CLI controller."""
        return self._do_quit(arg)
        
    def do_quit(self, arg: str) -> bool:
        """Exit the CLI controller."""
        return self._do_quit(arg)
        
    def _do_quit(self, arg: str) -> bool:
        """Shared implementation for exit and quit commands."""
        print("Exiting CLI controller.")
        self.running = False
        return True
        
    def do_move(self, arg: str) -> bool:
        """
        Move the duck in a specified direction.
        
        Usage: move <direction> [speed] [duration]
        
        Examples:
            move forward
            move backward 0.5
            move forward 0.7 2.0
        """
        args = arg.lower().split()
        if not args:
            print("Error: Direction required.")
            return True
            
        direction = args[0]
        speed = float(args[1]) if len(args) > 1 else 0.5
        duration = float(args[2]) if len(args) > 2 else 1.0
        
        if self.movement:
            if direction == "forward":
                self.movement.move_forward(speed, duration)
            elif direction == "backward":
                self.movement.move_backward(speed, duration)
            else:
                print(f"Unknown direction: {direction}")
        else:
            # Queue command for the decision loop
            command = {
                "intent_type": f"move_{direction}",
                "action_type": "move",
                "params": {"direction": direction, "speed": speed, "duration": duration},
                "confidence": 1.0,
                "original_text": arg
            }
            self.command_queue.put(command)
            print(f"Queued movement command: {direction}, speed={speed}, duration={duration}")
            
        return True
        
    def do_turn(self, arg: str) -> bool:
        """
        Turn the duck in a specified direction.
        
        Usage: turn <direction> [speed] [duration]
        
        Examples:
            turn left
            turn right 0.5
            turn left 0.7 2.0
        """
        args = arg.lower().split()
        if not args:
            print("Error: Direction required.")
            return True
            
        direction = args[0]
        speed = float(args[1]) if len(args) > 1 else 0.5
        duration = float(args[2]) if len(args) > 2 else 1.0
        
        if self.movement:
            if direction == "left":
                self.movement.turn_left(speed, duration)
            elif direction == "right":
                self.movement.turn_right(speed, duration)
            else:
                print(f"Unknown direction: {direction}")
        else:
            # Queue command for the decision loop
            command = {
                "intent_type": f"turn_{direction}",
                "action_type": "turn",
                "params": {"direction": direction, "speed": speed, "duration": duration},
                "confidence": 1.0,
                "original_text": arg
            }
            self.command_queue.put(command)
            print(f"Queued turn command: {direction}, speed={speed}, duration={duration}")
            
        return True
        
    def do_look(self, arg: str) -> bool:
        """
        Control the duck's head position.
        
        Usage: 
            look <direction>
            look at <target>
            
        Examples:
            look up
            look down
            look left
            look right
            look at person
        """
        if not arg:
            print("Error: Direction or target required.")
            return True
            
        # Check for "look at <target>" pattern
        at_match = re.match(r"at\s+(.+)", arg.lower())
        if at_match:
            target = at_match.group(1)
            
            # Queue command for the decision loop
            command = {
                "intent_type": "look_at",
                "action_type": "look_at",
                "params": {"target": target},
                "confidence": 1.0,
                "original_text": arg
            }
            self.command_queue.put(command)
            print(f"Queued look at command: {target}")
            return True
            
        # Handle direction
        direction = arg.lower()
        
        if self.movement:
            if direction == "up":
                self.movement.look_up()
            elif direction == "down":
                self.movement.look_down()
            elif direction == "left":
                self.movement.look_left()
            elif direction == "right":
                self.movement.look_right()
            else:
                print(f"Unknown direction: {direction}")
        else:
            # Queue command for the decision loop
            command = {
                "intent_type": f"look_{direction}",
                "action_type": "look",
                "params": {"direction": direction},
                "confidence": 1.0,
                "original_text": arg
            }
            self.command_queue.put(command)
            print(f"Queued look command: {direction}")
            
        return True
        
    def do_stop(self, arg: str) -> bool:
        """
        Stop all movement.
        
        Usage: stop
        """
        if self.movement:
            self.movement.stop()
        else:
            # Queue command for the decision loop
            command = {
                "intent_type": "stop",
                "action_type": "stop",
                "params": {},
                "confidence": 1.0,
                "original_text": "stop"
            }
            self.command_queue.put(command)
            print("Queued stop command")
            
        return True
        
    def do_status(self, arg: str) -> bool:
        """
        Get current status of the duck.
        
        Usage: status
        """
        # This would query the decision loop for current status
        print("Status information not available yet")
        return True
        
    def do_emote(self, arg: str) -> bool:
        """
        Express an emotion or play a sound.
        
        Usage: emote <name> [intensity]
        
        Examples:
            emote happy
            emote sad 0.8
        """
        args = arg.lower().split()
        if not args:
            print("Error: Emote name required.")
            return True
            
        emote_name = args[0]
        intensity = float(args[1]) if len(args) > 1 else 0.5
        
        # Queue command for the decision loop
        command = {
            "intent_type": f"emote_{emote_name}",
            "action_type": "emote",
            "params": {"emote": emote_name, "intensity": intensity},
            "confidence": 1.0,
            "original_text": arg
        }
        self.command_queue.put(command)
        print(f"Queued emote command: {emote_name}, intensity={intensity}")
        
        return True
        
    def do_chat(self, arg: str) -> bool:
        """
        Directly chat with the Duck VLA using the LLM.
        
        Usage: chat <message>
        
        Examples:
            chat What's the weather like?
            chat Tell me a joke.
        """
        if not arg:
            print("Error: Message required.")
            return True
            
        if self.central_model:
            try:
                # Process the message through the LLM
                response = self.central_model.process_natural_language(arg)
                print(response)
            except Exception as e:
                logger.error(f"Error processing chat: {e}")
                print(f"Error: {e}")
        else:
            print("Natural language processing not available.")
            print("Make sure Ollama is running with 'ollama serve' and a model is pulled.")
            
        return True 