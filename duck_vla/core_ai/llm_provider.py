import logging
import os
import json
import subprocess
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Generator, Any

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Base abstract class for LLM providers"""
    
    @abstractmethod
    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, 
                 max_tokens: int = 1000,
                 stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate text from a prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available"""
        pass

class OllamaProvider(LLMProvider):
    """Ollama-based LLM provider for local inference"""
    
    def __init__(self, model_name: str = "mistral:latest"):
        """Initialize Ollama provider with a model name"""
        self.model_name = model_name
        self.base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        logger.info(f"Initialized Ollama provider with model: {model_name}")
        
    def is_available(self) -> bool:
        """Check if Ollama is available by pinging the API"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, 
                 max_tokens: int = 1000, 
                 stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate text using Ollama API"""
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        # Set API endpoint
        api_url = f"{self.base_url}/api/generate"
        
        if stream:
            return self._stream_response(api_url, payload)
        else:
            try:
                response = requests.post(api_url, json=payload)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                logger.error(f"Error generating text with Ollama: {e}")
                return ""
    
    def _stream_response(self, api_url: str, payload: Dict[str, Any]) -> Generator[str, None, None]:
        """Stream response from Ollama API"""
        try:
            with requests.post(api_url, json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        if "response" in json_response:
                            yield json_response["response"]
        except Exception as e:
            logger.error(f"Error streaming response from Ollama: {e}")
            yield ""

class OpenAIProvider(LLMProvider):
    """OpenAI API-based LLM provider"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize OpenAI provider with model name"""
        self.model_name = model_name
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            logger.info(f"Initialized OpenAI provider with model: {model_name}")
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if the OpenAI API is available"""
        return self.client is not None and os.environ.get("OPENAI_API_KEY") is not None
    
    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, 
                 max_tokens: int = 1000,
                 stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate text using OpenAI API"""
        if not self.is_available():
            logger.error("OpenAI client not available")
            return "" if not stream else (yield "")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            if stream:
                return self._stream_response(messages, temperature, max_tokens)
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return "" if not stream else (yield "")
    
    def _stream_response(self, messages: List[Dict], temperature: float, max_tokens: int) -> Generator[str, None, None]:
        """Stream response from OpenAI API"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error streaming response from OpenAI: {e}")
            yield ""

class AnthropicProvider(LLMProvider):
    """Anthropic API-based LLM provider"""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229"):
        """Initialize Anthropic provider with model name"""
        self.model_name = model_name
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            logger.info(f"Initialized Anthropic provider with model: {model_name}")
        except ImportError:
            logger.error("Anthropic package not installed. Install with: pip install anthropic")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if the Anthropic API is available"""
        return self.client is not None and os.environ.get("ANTHROPIC_API_KEY") is not None
    
    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, 
                 max_tokens: int = 1000,
                 stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate text using Anthropic API"""
        if not self.is_available():
            logger.error("Anthropic client not available")
            return "" if not stream else (yield "")
        
        try:
            system = system_prompt if system_prompt else ""
            if stream:
                return self._stream_response(prompt, system, temperature, max_tokens)
            else:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {e}")
            return "" if not stream else (yield "")
    
    def _stream_response(self, prompt: str, system: str, temperature: float, max_tokens: int) -> Generator[str, None, None]:
        """Stream response from Anthropic API"""
        try:
            stream = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.type == "content_block_delta" and chunk.delta and chunk.delta.text:
                    yield chunk.delta.text
        except Exception as e:
            logger.error(f"Error streaming response from Anthropic: {e}")
            yield ""

class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(provider_type: str, model_name: Optional[str] = None) -> LLMProvider:
        """Create an LLM provider based on type"""
        provider_type = provider_type.lower()
        
        if provider_type == "ollama":
            return OllamaProvider(model_name or "mistral:latest")
        elif provider_type == "openai":
            return OpenAIProvider(model_name or "gpt-3.5-turbo")
        elif provider_type == "anthropic":
            return AnthropicProvider(model_name or "claude-3-sonnet-20240229")
        else:
            logger.warning(f"Unknown provider type: {provider_type}, using Ollama as fallback")
            return OllamaProvider(model_name or "mistral:latest")
    
    @staticmethod
    def get_default_provider() -> LLMProvider:
        """Get the default provider based on availability"""
        # First try Ollama
        ollama = OllamaProvider()
        if ollama.is_available():
            return ollama
        
        # Then try OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            openai = OpenAIProvider()
            if openai.is_available():
                return openai
        
        # Then try Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            anthropic = AnthropicProvider()
            if anthropic.is_available():
                return anthropic
        
        # Default to Ollama even if not available (will handle error)
        logger.warning("No available LLM provider found, defaulting to Ollama")
        return ollama 