import logging
import os
import json
import subprocess
from typing import Dict, List, Optional, Union, Generator, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass
    
    @abstractmethod
    def get_models(self) -> List[str]:
        """Get list of available models"""
        pass
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                temperature: float = 0.7, 
                max_tokens: int = 1000,
                stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate text using the LLM"""
        pass
    
    @abstractmethod
    def pull_model(self) -> bool:
        """Pull or download the model if needed"""
        pass

class OllamaProvider(LLMProvider):
    """Ollama-based LLM provider for local inference using the official Python client"""
    
    def __init__(self, model_name: str = "gemma:latest"):
        """Initialize Ollama provider with a model name"""
        self.model_name = model_name
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        logger.info(f"Initializing Ollama provider with model: {model_name}")
        
        # Import Ollama client library
        try:
            import ollama
            # Set OLLAMA_HOST environment variable if provided
            if self.host != "http://localhost:11434":
                os.environ["OLLAMA_HOST"] = self.host
            self.client = ollama
            logger.debug(f"Ollama client imported, host: {self.host}")
        except ImportError:
            logger.error("Ollama package not installed. Install with: pip install ollama")
            logger.error("Alternatively, run: python -m pip install ollama")
            self.client = None
        
    def is_available(self) -> bool:
        """Check if Ollama is available by listing models"""
        if not self.client:
            return False
            
        try:
            # Try to list models as a simple health check
            self.client.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            logger.warning("Make sure Ollama is running with 'ollama serve'")
            return False
    
    def get_models(self) -> List[str]:
        """Get list of available models"""
        if not self.client:
            return []
            
        try:
            models = self.client.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            logger.warning(f"Failed to get Ollama models: {e}")
            return []
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                temperature: float = 0.7, 
                max_tokens: int = 1000,
                stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate text using Ollama API"""
        if not self.client:
            logger.error("Ollama client not available")
            return "" if not stream else (yield "")
        
        # Prepare request options
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        
        if system_prompt:
            options["system"] = system_prompt
            
        try:
            if stream:
                return self._stream_response(prompt, options)
            else:
                response = self.client.generate(model=self.model_name, prompt=prompt, options=options)
                return response.get("response", "")
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {e}")
            return "" if not stream else (yield "")
    
    def _stream_response(self, prompt: str, options: Dict[str, Any]) -> Generator[str, None, None]:
        """Stream response from Ollama API"""
        try:
            # Enable streaming in options
            options["stream"] = True
            
            # Use the stream method
            for chunk in self.client.generate(model=self.model_name, prompt=prompt, options=options):
                if "response" in chunk:
                    yield chunk["response"]
        except Exception as e:
            logger.error(f"Error streaming response from Ollama: {e}")
            yield ""
    
    def pull_model(self) -> bool:
        """Pull the model if it doesn't exist"""
        if not self.client:
            return False
            
        try:
            # Check if model exists
            models = self.get_models()
            if self.model_name not in models:
                logger.info(f"Model {self.model_name} not found, pulling it now...")
                self.client.pull(self.model_name)
                logger.info(f"Successfully pulled model {self.model_name}")
                return True
            return True
        except Exception as e:
            logger.error(f"Error pulling model {self.model_name}: {e}")
            return False

class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(provider_type: str, model_name: Optional[str] = None) -> LLMProvider:
        """Create an LLM provider based on type"""
        provider_type = provider_type.lower()
        
        if provider_type == "ollama":
            return OllamaProvider(model_name or "gemma:latest")
        else:
            logger.warning(f"Unknown provider type: {provider_type}, using Ollama as fallback")
            return OllamaProvider(model_name or "gemma:latest")
    
    @staticmethod
    def get_default_provider() -> LLMProvider:
        """Get the default provider based on availability"""
        # First try Ollama
        ollama = OllamaProvider()
        if ollama.is_available():
            return ollama
        
        # Default to Ollama even if not available (will handle error)
        logger.warning("No available LLM provider found, defaulting to Ollama")
        return ollama 