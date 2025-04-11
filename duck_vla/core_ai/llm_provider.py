import logging
import os
import json
import subprocess
import time
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
    
    def __init__(self, model_name: str = "gemma3:latest"):
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
            
        models = []
        
        # First try using the API
        try:
            response = self.client.list()
            if isinstance(response, dict) and 'models' in response:
                # Check that each model has a name field
                for model in response['models']:
                    if isinstance(model, dict) and 'name' in model:
                        models.append(model['name'])
            
            # If we found models via API, return them
            if models:
                return models
                
            # If API returned no models, try subprocess as fallback
            logger.debug("API returned no models, trying CLI fallback")
        except Exception as e:
            logger.warning(f"Failed to get Ollama models via API: {e}")
            logger.debug("Trying CLI fallback")
        
        # Fallback: Use subprocess to call 'ollama list' directly
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                # Skip header line
                for line in output_lines[1:]:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            models.append(parts[0])  # First column is model name
                
                if models:
                    logger.debug(f"Found {len(models)} models via CLI")
                    return models
        except Exception as e:
            logger.warning(f"Failed to get Ollama models via CLI: {e}")
        
        logger.warning("No Ollama models found via API or CLI")
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
            return "Error: Ollama client not available" if not stream else (yield "Error: Ollama client not available")
        
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
                # Set a timeout for non-streaming requests
                options["timeout"] = 30.0  # 30 seconds timeout
                
                logger.debug(f"Sending generate request to Ollama with prompt: {prompt[:50]}...")
                start_time = time.time()
                
                try:
                    response = self.client.generate(model=self.model_name, prompt=prompt, options=options)
                    elapsed = time.time() - start_time
                    logger.debug(f"Received response from Ollama in {elapsed:.2f} seconds")
                    return response.get("response", "")
                except Exception as e:
                    logger.error(f"Ollama request failed: {e}")
                    # Try a fallback with subprocess
                    logger.info("Attempting fallback with ollama CLI...")
                    result = self._generate_with_subprocess(prompt, system_prompt, temperature, max_tokens)
                    if result:
                        return result
                    return f"Error generating response: {e}"
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {e}")
            return f"Error: {str(e)}" if not stream else (yield f"Error: {str(e)}")
    
    def _generate_with_subprocess(self, prompt, system_prompt, temperature, max_tokens):
        """Fallback method to generate text using subprocess with the ollama CLI command"""
        try:
            # Build the CLI command
            cmd = ["ollama", "run", self.model_name, prompt]
            
            logger.debug(f"Running fallback command: {' '.join(cmd)}")
            
            # Run with timeout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                logger.debug(f"Received response from ollama CLI: {response[:100]}...")
                return response
            else:
                logger.error(f"Ollama CLI command failed with code {result.returncode}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Ollama CLI command timed out")
            return "Error: Request timed out"
        except Exception as e:
            logger.error(f"Error using ollama CLI: {e}")
            return None
    
    def _stream_response(self, prompt: str, options: Dict[str, Any]) -> Generator[str, None, None]:
        """Stream response from Ollama API"""
        try:
            # Enable streaming in options
            options["stream"] = True
            
            # Set a timeout for the generator
            options["timeout"] = 15.0  # seconds - adjust based on expected response times
            
            # Track time to prevent infinitely waiting
            start_time = time.time()
            max_wait_time = 30.0  # Maximum total time to wait
            
            logger.debug(f"Starting Ollama streaming request with prompt: {prompt[:50]}...")
            
            try:
                # Use the stream method
                stream = self.client.generate(model=self.model_name, prompt=prompt, options=options)
                
                # Safety counter to prevent infinite streaming
                chunks_received = 0
                max_chunks = 1000  # Set a reasonable limit
                last_data_time = time.time()  # Track when we last received data
                data_timeout = 5.0  # Seconds to wait with no data before terminating
                
                for chunk in stream:
                    # Process the chunk
                    current_time = time.time()
                    if "response" in chunk:
                        content = chunk["response"]
                        chunks_received += 1
                        last_data_time = current_time
                        yield content
                    
                    # Check for timeout or too many chunks
                    elapsed = current_time - start_time
                    data_silence = current_time - last_data_time
                    
                    if elapsed > max_wait_time:
                        logger.warning(f"Streaming response terminated: timeout after {elapsed:.2f} seconds")
                        yield "\n[Response timed out]"
                        break
                        
                    if chunks_received > max_chunks:
                        logger.warning(f"Streaming response terminated: too many chunks ({chunks_received})")
                        yield "\n[Response too long]"
                        break
                        
                    if data_silence > data_timeout:
                        logger.warning(f"Streaming response terminated: no data for {data_silence:.2f} seconds")
                        yield "\n[Response stalled]"
                        break
                        
                logger.debug(f"Stream completed: received {chunks_received} chunks in {time.time() - start_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error in streaming response from Ollama API: {e}")
                yield f"\nError in stream: {str(e)}"
                
                # Try the subprocess fallback
                logger.info("Attempting fallback with ollama CLI...")
                result = self._generate_with_subprocess(prompt, options.get("system"), options.get("temperature", 0.7), options.get("num_predict", 1000))
                if result:
                    yield f"\nFallback response: {result}"
                
        except Exception as e:
            logger.error(f"Error in streaming setup: {e}")
            yield f"\nError: {str(e)}"
    
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
            return OllamaProvider(model_name or "gemma3:latest")
        else:
            logger.warning(f"Unknown provider type: {provider_type}, using Ollama as fallback")
            return OllamaProvider(model_name or "gemma3:latest")
    
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