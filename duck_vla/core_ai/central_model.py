import time
import logging
import sounddevice as sd
import soundfile as sf
import tempfile
import os
from typing import Optional, Dict, Any, Union, Generator

# Import our LLM provider
from duck_vla.core_ai.llm_provider import LLMProviderFactory, LLMProvider

logger = logging.getLogger(__name__)


class CentralModel:
    """
    Central AI model for the Duck VLA system.

    This class manages interactions with the LLM, handling:
    - System prompts and context
    - Processing commands and generating responses
    - Streaming output to actions
    """

    DEFAULT_SYSTEM_PROMPT = """You are a 2-foot-tall wheeled bipedal (two-legged) robot with a playful personality. Your name is Duck VLA. You can see your surroundings through your camera and respond to voice commands."""

    DEFAULT_ACTION_PROMPT = """I will give you a user request, and you will write python code to execute the command.

Available commands:
- say(text) - Make the robot speak the text
- move_forward(speed=0.5, duration=1.0) - Move forward at a given speed (0.0-1.0) for a duration in seconds
- move_backward(speed=0.5, duration=1.0) - Move backward at a given speed (0.0-1.0) for a duration in seconds
- turn_left(speed=0.5, duration=1.0) - Turn left at a given speed (0.0-1.0) for a duration in seconds
- turn_right(speed=0.5, duration=1.0) - Turn right at a given speed (0.0-1.0) for a duration in seconds
- stop() - Stop all movement
- look_up() - Look upward
- look_down() - Look downward
- look_left() - Look to the left
- look_right() - Look to the right
- reset_head() - Reset head position to default

Specify a sequence of commands by concatenating commands with newlines.
If a request is impossible to perform, use the say function to explain why.
Otherwise, make your best effort to perform the request.

Respond in python code ONLY. Don't use any loops, if statements, or indentation in your response.
"""

    def __init__(
        self,
        provider_type: str = "ollama",
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        action_prompt: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        Initialize the central model.

        Args:
            provider_type: LLM provider type ("ollama")
            model_name: Specific model name to use with the provider
            system_prompt: Custom system prompt to use
            action_prompt: Custom action prompt template
            debug_mode: Whether to enable debug logging
        """
        self.debug_mode = debug_mode
        if debug_mode:
            logger.setLevel(logging.DEBUG)

        # Set default prompts
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.action_prompt = action_prompt or self.DEFAULT_ACTION_PROMPT

        # Initialize LLM provider
        self.provider_type = provider_type

        # Use a default model name if none provided
        if model_name is None:
            if provider_type.lower() == "ollama":
                self.model_name = "gemma3:4b"
            else:
                self.model_name = "gemma3:4b"  # Default for all providers
        else:
            self.model_name = model_name

        logger.info(f"Initializing {provider_type} provider with model: {self.model_name}")

        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize the LLM provider using the factory"""
        try:
            # Create the provider using the factory
            self.llm_provider = LLMProviderFactory.create_provider(
                provider_type=self.provider_type, model_name=self.model_name
            )

            # Check if provider is available
            if not self.llm_provider.is_available():
                logger.warning(f"{self.provider_type} provider not available")

                # Try to pull the model
                try:
                    logger.info(f"Attempting to pull model {self.model_name}...")
                    self.llm_provider.pull_model()
                except Exception as e:
                    logger.error(f"Failed to pull model: {e}")
                    if self.provider_type == "ollama":
                        logger.warning("Make sure Ollama is running with: ollama serve")
            else:
                logger.info(f"Using {self.provider_type} provider with model {self.model_name}")

                # Check if model is in the list of available models
                models = self.llm_provider.get_models()
                if not models:
                    logger.warning("No models found via API. Checking if model exists via CLI...")
                    # Try directly checking if the model exists using CLI
                    try:
                        import subprocess

                        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                        if result.returncode == 0:
                            output = result.stdout
                            if self.model_name in output:
                                logger.info(f"Model {self.model_name} found via CLI command")
                                return
                    except Exception as cli_error:
                        logger.error(f"Error checking model via CLI: {cli_error}")

                # If we reach here, we need to check explicitly
                if self.model_name not in models:
                    logger.warning(
                        f"Model {self.model_name} not found in available models: {', '.join(models)}"
                    )
                    logger.info(f"Attempting to pull model {self.model_name}...")
                    # Use CLI method for more reliable pulling
                    try:
                        import subprocess

                        logger.info(f"Pulling model via CLI: ollama pull {self.model_name}")
                        pull_process = subprocess.run(
                            ["ollama", "pull", self.model_name], capture_output=True, text=True
                        )
                        if pull_process.returncode == 0:
                            logger.info(f"Successfully pulled model {self.model_name}")
                        else:
                            logger.error(f"Failed to pull model via CLI: {pull_process.stderr}")
                            # Fall back to API method
                            self.llm_provider.pull_model()
                    except Exception as cli_error:
                        logger.error(f"Error pulling model via CLI: {cli_error}")
                        # Fall back to API method
                        self.llm_provider.pull_model()

        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            self.llm_provider = None

    def generate_response(
        self,
        user_input: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response to user input.

        Args:
            user_input: User input text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated response as string or generator if streaming
        """
        if not self.llm_provider:
            logger.error("LLM provider not available")
            return "I'm sorry, I'm having trouble connecting to my brain. Make sure the LLM provider is running."

        logger.debug(f"Generating response for input: {user_input}")

        # Generate response using LLM provider
        return self.llm_provider.generate(
            prompt=user_input,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

    def process_command(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Process a command and generate actionable code.

        Args:
            command: User command text
            context: Additional context for the command
            temperature: Sampling temperature
            stream: Whether to stream the response

        Returns:
            Generated Python code as string or generator if streaming
        """
        if not self.llm_provider:
            logger.error("LLM provider not available")
            return "say('I\\'m sorry, I\\'m having trouble connecting to my brain. Make sure the LLM provider is running.')"

        logger.debug(f"Processing command: {command}")

        # Construct the prompt with action template
        context_str = ""
        if context:
            context_str = "\nContext:\n" + "\n".join([f"{k}: {v}" for k, v in context.items()])

        prompt = f"{self.action_prompt}{context_str}\n\nRequest: {command}\nCode:"

        # Generate Python code using LLM provider
        return self.llm_provider.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=500,
            stream=stream,
        )

    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt"""
        self.system_prompt = new_prompt
        logger.debug(f"Updated system prompt: {new_prompt}")

    def update_action_prompt(self, new_prompt: str) -> None:
        """Update the action prompt template"""
        self.action_prompt = new_prompt
        logger.debug(f"Updated action prompt: {new_prompt}")

    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model within the same provider.

        Args:
            model_name: The model name to switch to

        Returns:
            True if switch was successful, False otherwise
        """
        try:
            logger.info(f"Switching to model: {model_name}")

            # Create a new provider with the new model
            new_provider = LLMProviderFactory.create_provider(
                provider_type=self.provider_type, model_name=model_name
            )

            if new_provider.is_available():
                # Check if model is available
                models = new_provider.get_models()
                if model_name not in models:
                    logger.info(f"Model {model_name} not found, pulling it now...")
                    if not new_provider.pull_model():
                        logger.warning(f"Failed to pull model {model_name}")
                        return False

                self.llm_provider = new_provider
                self.model_name = model_name
                logger.info(f"Successfully switched to model {model_name}")
                return True
            else:
                logger.warning(f"Provider not available")
                return False

        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False

    def switch_provider(self, provider_type: str, model_name: Optional[str] = None) -> bool:
        """
        Switch to a different provider and/or model.

        Args:
            provider_type: The provider type to switch to
            model_name: Optional model name to use, or None for default

        Returns:
            True if switch was successful, False otherwise
        """
        try:
            logger.info(f"Switching to provider: {provider_type}")

            # Create a new provider with the specified type
            new_provider = LLMProviderFactory.create_provider(
                provider_type=provider_type, model_name=model_name or self.model_name
            )

            if new_provider.is_available():
                self.llm_provider = new_provider
                self.provider_type = provider_type

                if model_name:
                    self.model_name = model_name

                logger.info(f"Successfully switched to provider {provider_type}")
                return True
            else:
                logger.warning(f"Provider {provider_type} not available")
                return False

        except Exception as e:
            logger.error(f"Failed to switch provider: {e}")
            return False

    def process_natural_language(
        self, text: str, context: Optional[Dict[str, Any]] = None, temperature: float = 0.7
    ) -> str:
        """
        Process natural language input and generate a response.

        This is different from process_command as it returns a conversational
        response rather than executable code.

        Args:
            text: User text input
            context: Additional context for the interaction
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        if not self.llm_provider:
            logger.error("LLM provider not available")
            return "I'm sorry, I'm having trouble connecting to my brain. Make sure the LLM provider is running."

        logger.debug(f"Processing natural language: {text}")

        # Construct a prompt for conversation
        context_str = ""
        if context:
            context_str = "\nContext:\n" + "\n".join([f"{k}: {v}" for k, v in context.items()])

        # Simple conversational prompt
        prompt = f"User: {text}{context_str}\nRespond as the Duck VLA robot:\n"

        try:
            # Generate response using LLM provider
            response = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=temperature,
                max_tokens=500,
                stream=False,
            )
            return response
        except Exception as e:
            logger.error(f"Error processing natural language: {e}")
            return "I'm sorry, I'm having trouble processing that right now. Make sure the LLM provider is running."
