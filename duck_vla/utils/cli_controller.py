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
import sys
import traceback
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
        
        # Configure line ending handling
        self.use_rawinput = True  # Use Python's raw_input which handles line endings
        self.stdin = sys.stdin     # Ensure we're using the right input stream
        
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
                # Use a custom cmdloop to ensure proper line ending handling
                self.cmdloop()
                break
            except KeyboardInterrupt:
                print("\nKeyboard interrupt. Type 'exit' or 'quit' to exit.")
            except Exception as e:
                logger.error(f"Error in CLI loop: {e}")
                logger.error(traceback.format_exc())  # Log full traceback for debugging
                time.sleep(1)  # Prevent fast-spinning on error
    
    def precmd(self, line):
        """Process command line before execution and strip any unexpected characters."""
        # Strip carriage returns and other whitespace
        return line.strip()
        
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
        # Strip any carriage returns or unexpected characters
        line = line.strip()
        
        if not line:
            return True
            
        # Check if central model is available for natural language processing
        if self.central_model:
            try:
                logger.info(f"Processing command: '{line}'")
                print(f"Processing: {line}")
                
                # Set a timeout for the entire operation
                max_total_time = 40  # Total seconds to wait for complete operation
                operation_start = time.time()
                
                # Process the line as natural language with stream=False to avoid hanging
                response = self.central_model.process_command(line, stream=False)
                
                # If we've taken too long already, bail out
                if time.time() - operation_start > max_total_time:
                    logger.warning(f"Command processing timed out after {max_total_time} seconds")
                    print("\n[Command processing timed out]")
                    return True
                
                # Handle generator responses (streaming)
                if hasattr(response, '__iter__') and hasattr(response, '__next__'):
                    logger.debug("Got streaming response, consuming generator")
                    collected_response = ""
                    try:
                        # Add a timeout mechanism to prevent infinite loops
                        max_wait_time = 30  # seconds
                        start_time = time.time()
                        
                        for chunk in response:
                            collected_response += chunk
                            # For interactive experience, print chunks as they arrive
                            print(chunk, end="", flush=True)
                            
                            # Check if we've been waiting too long
                            if time.time() - start_time > max_wait_time:
                                logger.warning(f"Response streaming timed out after {max_wait_time} seconds")
                                print("\n[Response timed out]")
                                break
                                
                            # Also check overall operation time
                            if time.time() - operation_start > max_total_time:
                                logger.warning(f"Total operation timed out after {max_total_time} seconds")
                                print("\n[Operation timed out]")
                                break
                        
                        print()  # Add newline at the end
                    except Exception as e:
                        logger.error(f"Error consuming response stream: {e}")
                        print(f"\nError in response stream: {e}")
                    
                    # Execute the code if it's valid Python
                    if collected_response and self.movement:
                        try:
                            logger.debug(f"Executing response as code: {collected_response}")
                            exec(collected_response)
                        except Exception as e:
                            logger.error(f"Error executing code: {e}")
                            print(f"Error executing response as code: {e}")
                else:
                    # Regular string response
                    print(response)
                    
                    # Execute the code if it's valid Python
                    if response and self.movement:
                        try:
                            logger.debug(f"Executing response as code: {response}")
                            exec(response)
                        except Exception as e:
                            logger.error(f"Error executing code: {e}")
                            print(f"Error executing response as code: {e}")
                
                logger.info("Command processing completed")
                return True
            except Exception as e:
                logger.error(f"Error processing natural language: {e}")
                logger.error(traceback.format_exc())  # Log full traceback
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
        
    def do_command(self, arg: str) -> bool:
        """
        Process a command using the action prompt template and execute the resulting code.
        
        Usage: command <instruction>
        
        Examples:
            command walk forward for 2 seconds
            command turn around in a circle
        """
        if not arg:
            print("Error: Command required.")
            return True
            
        if not self.central_model:
            print("Error: Central model not available.")
            return True
            
        try:
            # Process the command and get Python code
            logger.debug(f"Processing command: {arg}")
            response = self.central_model.process_command(arg, stream=True)
            
            # Handle streaming response
            collected_code = ""
            print("Response:")
            
            # Add a timeout mechanism to prevent infinite loops
            max_wait_time = 30  # seconds
            start_time = time.time()
            
            for chunk in response:
                collected_code += chunk
                print(chunk, end="", flush=True)
                
                # Check if we've been waiting too long
                if time.time() - start_time > max_wait_time:
                    logger.warning(f"Response streaming timed out after {max_wait_time} seconds")
                    print("\n[Response timed out]")
                    break
                    
            print("\n")
            
            # Execute the generated code if movement controller is available
            if self.movement and collected_code:
                try:
                    logger.debug(f"Executing code: {collected_code}")
                    print("Executing...")
                    exec(collected_code)
                    print("Done.")
                except Exception as e:
                    logger.error(f"Error executing code: {e}")
                    print(f"Error executing code: {e}")
            elif not self.movement:
                print("Warning: Movement controller not available. Code not executed.")
                
            return True
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            print(f"Error: {e}")
            return True
        
    def do_show_model(self, arg: str) -> bool:
        """
        Show information about the current model.
        
        Usage: show_model
        """
        if not self.central_model:
            print("Error: Central model not available.")
            return True
            
        print(f"Provider: {self.central_model.provider_type}")
        print(f"Model: {self.central_model.model_name}")
        print(f"Provider available: {self.central_model.llm_provider is not None}")
        
        if self.central_model.llm_provider:
            models = self.central_model.llm_provider.get_models()
            print(f"Available models: {', '.join(models) if models else 'None'}")
            
        return True
        
    def do_check_ollama(self, arg: str) -> bool:
        """
        Check if Ollama is running and available.
        
        Usage: check_ollama
        """
        try:
            import ollama
            import subprocess
            
            client = ollama.Client()
            print("Checking Ollama connection...")
            
            # First try API method
            try:
                response = client.list()
                print("Ollama API is running!")
                
                # Check models via API
                if isinstance(response, dict) and 'models' in response:
                    models = response['models']
                    if models:
                        print(f"Found {len(models)} models via API:")
                        for model in models:
                            if isinstance(model, dict) and 'name' in model:
                                print(f"  - {model['name']}")
                    else:
                        print("No models found via API.")
                else:
                    print("No models available via API.")
                    
                # Now try CLI method as a backup
                print("\nChecking via CLI command (more reliable):")
                try:
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                    if result.returncode == 0:
                        print("Models found via CLI:")
                        output_lines = result.stdout.strip().split('\n')
                        # Skip header line
                        for line in output_lines[1:]:
                            # Split by whitespace and get first column (model name)
                            if line.strip():
                                parts = line.split()
                                if parts:
                                    model_name = parts[0]
                                    print(f"  - {model_name}")
                        
                        # Tell the user which to pull if gemma3:latest isn't found
                        if not any("gemma3:latest" in line for line in output_lines):
                            print("\ngemma3:latest model not found. Pull with: ollama pull gemma3:latest")
                    else:
                        print(f"Error running 'ollama list': {result.stderr}")
                except Exception as e:
                    print(f"Error checking models via CLI: {e}")
                    
            except Exception as e:
                print(f"Error connecting to Ollama API: {e}")
                print("Make sure Ollama server is running with: ollama serve")
                
                # Try CLI as fallback
                try:
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                    if result.returncode == 0:
                        print("\nModels found via CLI:")
                        print(result.stdout)
                    else:
                        print(f"Error running 'ollama list': {result.stderr}")
                except Exception as cli_error:
                    print(f"Error checking models via CLI: {cli_error}")
                
        except ImportError:
            print("Ollama package not installed.")
            print("Install with: pip install ollama")
            
        return True
        
    def do_pull_model(self, arg: str) -> bool:
        """
        Pull an Ollama model.
        
        Usage: pull_model [model_name]
        
        If model_name is not provided, it will pull the current model.
        
        Examples:
            pull_model
            pull_model gemma3:latest
            pull_model llama3:latest
        """
        model_name = arg.strip() if arg.strip() else None
        
        if not self.central_model:
            print("Error: Central model not available.")
            return True
            
        if not self.central_model.llm_provider:
            print("Error: LLM provider not available.")
            return True
            
        # Use current model if not specified
        if not model_name:
            model_name = self.central_model.model_name
            
        print(f"Pulling model: {model_name}...")
        
        try:
            success = self.central_model.llm_provider.pull_model() if not model_name else False
            
            # If we're using a different model than the current one, create a temporary provider
            if model_name and model_name != self.central_model.model_name:
                from duck_vla.core_ai.llm_provider import LLMProviderFactory
                temp_provider = LLMProviderFactory.create_provider(
                    provider_type=self.central_model.provider_type,
                    model_name=model_name
                )
                success = temp_provider.pull_model()
                
            if success:
                print(f"Successfully pulled model: {model_name}")
                # Update the current model if requested
                if model_name != self.central_model.model_name:
                    switch = input(f"Switch to model {model_name}? (y/n): ").lower()
                    if switch.startswith('y'):
                        if self.central_model.switch_model(model_name):
                            print(f"Switched to model: {model_name}")
                        else:
                            print("Failed to switch model.")
            else:
                print(f"Failed to pull model: {model_name}")
                
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            print(f"Error: {e}")
            
        return True
        
    def do_test_llm(self, arg: str) -> bool:
        """
        Test the LLM provider with a simple message to check responsiveness.
        
        Usage: test_llm
        """
        if not self.central_model:
            print("Error: Central model not available")
            return True
            
        test_message = "Please respond with a single word: hello"
        print(f"Sending test message to LLM: '{test_message}'")
        
        try:
            # Force non-streaming for this test
            start_time = time.time()
            response = self.central_model.process_natural_language(test_message)
            elapsed = time.time() - start_time
            
            print(f"Response received in {elapsed:.2f} seconds:")
            print(f"{response}")
            
            if elapsed > 10.0:
                print("\nWarning: LLM is responding slowly (>10 seconds)")
            elif elapsed < 1.0:
                print("\nLLM responded very quickly - check if it's actually processing requests")
            else:
                print("\nLLM response time looks normal")
                
            return True
            
        except Exception as e:
            logger.error(f"Error testing LLM: {e}")
            print(f"Error: {e}")
            print("Make sure Ollama is running with 'ollama serve' and a model is pulled")
            return True 