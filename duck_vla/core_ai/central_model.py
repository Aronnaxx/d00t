import time
import logging
import sounddevice as sd
import soundfile as sf
import tempfile
import os
from typing import Optional, Dict, Any, Union, Generator

# Import our LLM providers
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
            provider_type: LLM provider type ('ollama', 'openai', 'anthropic')
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
        logger.info(f"Initializing LLM provider: {provider_type}")
        self.provider_type = provider_type
        self.model_name = model_name
        
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the LLM provider"""
        try:
            self.llm = LLMProviderFactory.create_provider(
                self.provider_type, 
                self.model_name
            )
            
            # Check if provider is available
            if not self.llm.is_available():
                logger.warning(f"{self.provider_type} provider not available, falling back to default")
                self.llm = LLMProviderFactory.get_default_provider()
                self.provider_type = "default fallback"
                
            logger.info(f"Using {self.provider_type} provider")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            logger.warning("Falling back to default provider")
            self.llm = LLMProviderFactory.get_default_provider()
            self.provider_type = "default fallback"
    
    def generate_response(
        self, 
        user_input: str, 
        temperature: float = 0.7, 
        max_tokens: int = 500,
        stream: bool = False
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
        logger.debug(f"Generating response for input: {user_input}")
        
        # Generate response using LLM provider
        return self.llm.generate(
            prompt=user_input,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
    
    def process_command(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        stream: bool = True
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
        logger.debug(f"Processing command: {command}")
        
        # Construct the prompt with action template
        context_str = ""
        if context:
            context_str = "\nContext:\n" + "\n".join([f"{k}: {v}" for k, v in context.items()])
        
        prompt = f"{self.action_prompt}{context_str}\n\nRequest: {command}\nCode:"
        
        # Generate Python code using LLM provider
        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=500,
            stream=stream
        )
        
    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt"""
        self.system_prompt = new_prompt
        logger.debug(f"Updated system prompt: {new_prompt}")
        
    def update_action_prompt(self, new_prompt: str) -> None:
        """Update the action prompt template"""
        self.action_prompt = new_prompt
        logger.debug(f"Updated action prompt: {new_prompt}")
        
    def switch_provider(self, provider_type: str, model_name: Optional[str] = None) -> bool:
        """
        Switch to a different LLM provider.
        
        Args:
            provider_type: The provider type to switch to
            model_name: Optional model name for the new provider
            
        Returns:
            True if switch was successful, False otherwise
        """
        try:
            logger.info(f"Switching provider to {provider_type}")
            new_provider = LLMProviderFactory.create_provider(provider_type, model_name)
            
            if new_provider.is_available():
                self.llm = new_provider
                self.provider_type = provider_type
                self.model_name = model_name
                logger.info(f"Successfully switched to {provider_type}")
                return True
            else:
                logger.warning(f"Provider {provider_type} not available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to switch provider: {e}")
            return False
