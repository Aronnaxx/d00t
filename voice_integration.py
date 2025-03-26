#!/usr/bin/env python3
"""
Voice Integration Module for Open Duck Robot

This module provides speech recognition and text-to-speech capabilities for the Duck Robot,
allowing it to accept voice commands and respond audibly. It includes:
- Microphone audio capture and processing
- Speech-to-text using Whisper or Ollama LLM
- Text-to-speech using Ollama LLM
- Voice command processing pipeline

The module can operate standalone for testing or be integrated with vlm_integration.py
for full robot control via voice commands.
"""

import argparse
import base64
import io
import json
import logging
import os
import queue
import tempfile
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import whisper
from PIL import Image

try:
    from ollama import OllamaManager, start_ollama_server
except ImportError:
    OllamaManager = None
    start_ollama_server = None
    logging.warning("Ollama manager not available. Voice features may be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("voice_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("voice_integration")

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
BUFFER_DURATION = 0.1  # seconds
ENERGY_THRESHOLD = 0.03  # Threshold for detecting speech
SILENCE_DURATION = 1.0  # seconds of silence to end recording
MAX_RECORDING_DURATION = 10.0  # seconds, max duration of a single command

# Model names
WHISPER_MODEL_NAME = "base"  # Options: tiny, base, small, medium, large
OLLAMA_STT_MODEL_NAME = "gemma3:4b"
OLLAMA_TTS_MODEL_NAME = "gemma3:4b"

class AudioBuffer:
    """
    Manages audio data and provides voice activity detection
    """
    def __init__(self, 
                 sample_rate: int = SAMPLE_RATE, 
                 channels: int = CHANNELS,
                 energy_threshold: float = ENERGY_THRESHOLD,
                 silence_duration: float = SILENCE_DURATION):
        self.sample_rate = sample_rate
        self.channels = channels
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.buffer: List[np.ndarray] = []
        self.is_speaking = False
        self.last_speech_time = 0
        
    def add_audio(self, audio_data: np.ndarray) -> None:
        """Add audio data to the buffer"""
        self.buffer.append(audio_data.copy())
        energy = np.mean(np.abs(audio_data))
        
        # Voice activity detection
        if energy > self.energy_threshold:
            self.is_speaking = True
            self.last_speech_time = time.time()
        elif self.is_speaking and (time.time() - self.last_speech_time) > self.silence_duration:
            self.is_speaking = False
    
    def get_audio(self) -> np.ndarray:
        """Get all audio data from the buffer"""
        if not self.buffer:
            return np.array([])
        return np.concatenate(self.buffer)
    
    def clear(self) -> None:
        """Clear the audio buffer"""
        self.buffer = []
        self.is_speaking = False
        
    def has_speech_ended(self) -> bool:
        """Check if speech has ended based on silence duration"""
        return self.is_speaking and (time.time() - self.last_speech_time) > self.silence_duration

class SpeechRecognizer:
    """
    Handles speech-to-text conversion using either Whisper or Ollama
    """
    def __init__(self, 
                 use_ollama: bool = False,
                 whisper_model: str = WHISPER_MODEL_NAME,
                 ollama_model: str = OLLAMA_STT_MODEL_NAME,
                 ollama_url: str = "http://localhost:11434",
                 sample_rate: int = SAMPLE_RATE):
        self.use_ollama = use_ollama and OllamaManager is not None
        self.sample_rate = sample_rate
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        
        # Initialize appropriate model
        if self.use_ollama:
            self.ollama_manager = OllamaManager()
            logger.info(f"Using Ollama for speech recognition with model {ollama_model}")
            if not self.ollama_manager.is_running():
                logger.info("Starting Ollama server...")
                self.ollama_manager.start()
                time.sleep(2)  # Wait for server to initialize
            
            if not self.ollama_manager.model_exists(ollama_model):
                logger.info(f"Pulling Ollama model {ollama_model}...")
                self.ollama_manager.pull_model(ollama_model)
        else:
            logger.info(f"Loading Whisper model: {whisper_model}")
            self.whisper_model = whisper.load_model(whisper_model)
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Transcribed text
        """
        if len(audio_data) == 0:
            return ""
        
        # Ensure audio is mono
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if self.use_ollama:
            return self._transcribe_with_ollama(audio_data)
        else:
            return self._transcribe_with_whisper(audio_data)
    
    def _transcribe_with_whisper(self, audio_data: np.ndarray) -> str:
        """Transcribe using the Whisper model"""
        try:
            # Create a temporary file to store audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                sf.write(tmp_file.name, audio_data, self.sample_rate)
                
                # Transcribe using whisper
                result = self.whisper_model.transcribe(tmp_file.name)
                text = result["text"].strip()
                
                logger.debug(f"Whisper transcription: {text}")
                return text
        except Exception as e:
            logger.error(f"Error transcribing with Whisper: {e}")
            return ""
    
    def _transcribe_with_ollama(self, audio_data: np.ndarray) -> str:
        """Transcribe using the Ollama LLM"""
        try:
            # Save audio to WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                sf.write(tmp_file.name, audio_data, self.sample_rate)
                
                # Read file and encode as base64
                with open(tmp_file.name, "rb") as f:
                    audio_bytes = f.read()
                
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                
                # Create prompt for transcription
                prompt = "Please transcribe the following audio accurately. Return only the transcribed text."
                
                # Send request to Ollama
                response = self.ollama_manager.query_model(
                    self.ollama_model,
                    prompt=prompt,
                    images=[audio_base64]
                )
                
                logger.debug(f"Ollama transcription raw response: {response}")
                
                # Extract the transcription from the response
                text = response.strip()
                return text
        except Exception as e:
            logger.error(f"Error transcribing with Ollama: {e}")
            return ""
    
    def close(self):
        """Clean up resources"""
        if self.use_ollama and hasattr(self, 'ollama_manager'):
            # Don't stop Ollama here, as it might be used by other components
            pass

class TextToSpeech:
    """
    Handles text-to-speech conversion using Ollama
    """
    def __init__(self, 
                 model_name: str = OLLAMA_TTS_MODEL_NAME,
                 ollama_url: str = "http://localhost:11434",
                 sample_rate: int = SAMPLE_RATE):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.sample_rate = sample_rate
        
        if OllamaManager is not None:
            self.ollama_manager = OllamaManager()
            logger.info(f"Using Ollama for text-to-speech with model {model_name}")
            if not self.ollama_manager.is_running():
                logger.info("Starting Ollama server...")
                self.ollama_manager.start()
                time.sleep(2)  # Wait for server to initialize
            
            if not self.ollama_manager.model_exists(model_name):
                logger.info(f"Pulling Ollama model {model_name}...")
                self.ollama_manager.pull_model(model_name)
        else:
            logger.warning("Ollama manager not available. TTS features will not work.")
            self.ollama_manager = None
    
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as numpy array or None if conversion failed
        """
        if not text or self.ollama_manager is None:
            return None
        
        try:
            # Create prompt for TTS
            prompt = f"Generate speech audio for the following text: '{text}'. Return only the audio data."
            
            # Send request to Ollama
            response = self.ollama_manager.query_model(
                self.model_name,
                prompt=prompt
            )
            
            logger.debug(f"Ollama TTS raw response: {response}")
            
            # This is a simplified implementation. In reality, we would need to:
            # 1. Parse the response to extract any audio data or instructions
            # 2. Convert text to audio using a TTS library or service
            # 3. Return the audio data
            
            # For now, we'll simulate TTS by creating a beep sound
            logger.info(f"TTS would say: '{text}'")
            
            # Generate a simple beep sound as placeholder
            duration = min(3.0, 0.1 * len(text.split()))  # Duration based on text length
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            beep = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            
            return beep
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None
    
    def speak(self, text: str) -> None:
        """
        Speak the given text through the default audio output
        
        Args:
            text: Text to speak
        """
        audio_data = self.synthesize(text)
        if audio_data is not None:
            try:
                sd.play(audio_data, self.sample_rate)
                sd.wait()  # Wait until audio has finished playing
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'ollama_manager') and self.ollama_manager is not None:
            # Don't stop Ollama here, as it might be used by other components
            pass

class MicrophoneListener:
    """
    Listens to the microphone and processes audio for speech recognition
    """
    def __init__(self, 
                 callback: Callable[[str], None],
                 sample_rate: int = SAMPLE_RATE,
                 channels: int = CHANNELS,
                 buffer_duration: float = BUFFER_DURATION,
                 energy_threshold: float = ENERGY_THRESHOLD,
                 silence_duration: float = SILENCE_DURATION,
                 max_recording_duration: float = MAX_RECORDING_DURATION,
                 use_ollama_stt: bool = False):
        self.callback = callback
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_duration = buffer_duration
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.max_recording_duration = max_recording_duration
        
        self.audio_buffer = AudioBuffer(
            sample_rate=sample_rate,
            channels=channels,
            energy_threshold=energy_threshold,
            silence_duration=silence_duration
        )
        
        self.recognizer = SpeechRecognizer(
            use_ollama=use_ollama_stt,
            sample_rate=sample_rate
        )
        
        self.is_listening = False
        self.listen_thread = None
        self.processing_thread = None
        self.audio_queue = queue.Queue()
        
        # Flags for speech detection state
        self.recording_active = False
        self.recording_start_time = 0
        
        logger.info("Microphone listener initialized")
    
    def start(self) -> None:
        """Start listening to the microphone"""
        if self.is_listening:
            logger.warning("Microphone listener is already active")
            return
        
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.processing_thread = threading.Thread(target=self._processing_loop)
        
        self.listen_thread.daemon = True
        self.processing_thread.daemon = True
        
        self.listen_thread.start()
        self.processing_thread.start()
        
        logger.info("Microphone listener started")
    
    def stop(self) -> None:
        """Stop listening to the microphone"""
        self.is_listening = False
        
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2.0)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.audio_queue.put(None)  # Signal to stop processing
            self.processing_thread.join(timeout=2.0)
        
        self.recognizer.close()
        logger.info("Microphone listener stopped")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio input"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Add audio data to queue
        self.audio_queue.put(indata.copy())
    
    def _listen_loop(self) -> None:
        """Main loop for listening to microphone"""
        try:
            block_size = int(self.sample_rate * self.buffer_duration)
            
            with sd.InputStream(callback=self._audio_callback,
                              samplerate=self.sample_rate,
                              channels=self.channels,
                              blocksize=block_size):
                logger.info("Microphone stream started")
                
                # Keep the stream open until stop() is called
                while self.is_listening:
                    time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error in microphone listener: {e}")
            self.is_listening = False
    
    def _processing_loop(self) -> None:
        """Process audio data from the queue"""
        while self.is_listening or not self.audio_queue.empty():
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=0.5)
                
                if audio_data is None:
                    break  # Stop signal
                
                # Process audio chunk
                self._process_audio_chunk(audio_data)
                
                # Mark task as done
                self.audio_queue.task_done()
            
            except queue.Empty:
                # Queue is empty, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
    
    def _process_audio_chunk(self, audio_data: np.ndarray) -> None:
        """Process a chunk of audio data"""
        energy = np.mean(np.abs(audio_data))
        
        # Check if this is speech
        if not self.recording_active and energy > self.energy_threshold:
            # Start of speech detected
            logger.debug("Speech detected, starting recording")
            self.recording_active = True
            self.recording_start_time = time.time()
            self.audio_buffer.clear()
        
        # If we're recording, add data to buffer
        if self.recording_active:
            self.audio_buffer.add_audio(audio_data)
            
            # Check if recording should end
            current_time = time.time()
            recording_duration = current_time - self.recording_start_time
            
            if (energy < self.energy_threshold and 
                recording_duration > self.silence_duration):
                # End of speech detected (silence)
                self._finalize_recording()
            elif recording_duration >= self.max_recording_duration:
                # Maximum recording duration reached
                logger.debug("Max recording duration reached")
                self._finalize_recording()
    
    def _finalize_recording(self) -> None:
        """Process the completed recording"""
        logger.debug("End of speech detected, processing")
        
        # Get all audio data
        audio_data = self.audio_buffer.get_audio()
        
        if len(audio_data) > 0:
            # Transcribe audio
            text = self.recognizer.transcribe(audio_data)
            
            if text:
                logger.info(f"Recognized: '{text}'")
                # Call the callback with transcribed text
                if self.callback:
                    self.callback(text)
            else:
                logger.debug("No text recognized")
        
        # Reset state
        self.recording_active = False
        self.audio_buffer.clear()

class VoiceCommandProcessor:
    """
    Manages voice command processing, including speech recognition and TTS responses
    """
    def __init__(self, 
                 command_callback: Callable[[str], None],
                 use_microphone: bool = True,
                 use_tts: bool = True,
                 use_ollama_stt: bool = False,
                 use_ollama_tts: bool = True):
        self.command_callback = command_callback
        self.use_microphone = use_microphone
        self.use_tts = use_tts
        
        # Initialize TTS if enabled
        self.tts = None
        if use_tts:
            logger.info("Initializing text-to-speech")
            self.tts = TextToSpeech()
        
        # Initialize microphone listener if enabled
        self.mic_listener = None
        if use_microphone:
            logger.info("Initializing microphone listener")
            self.mic_listener = MicrophoneListener(
                callback=self._on_speech_recognized,
                use_ollama_stt=use_ollama_stt
            )
        
        logger.info("Voice command processor initialized")
    
    def start(self) -> None:
        """Start processing voice commands"""
        if self.mic_listener:
            self.mic_listener.start()
            self.speak("Voice command processing started. I'm listening.")
    
    def stop(self) -> None:
        """Stop processing voice commands"""
        if self.mic_listener:
            self.speak("Voice command processing stopped.")
            self.mic_listener.stop()
        
        if self.tts:
            self.tts.close()
    
    def speak(self, text: str) -> None:
        """
        Speak the given text using TTS
        
        Args:
            text: Text to speak
        """
        logger.info(f"Robot says: '{text}'")
        
        if self.use_tts and self.tts:
            self.tts.speak(text)
    
    def process_command(self, command_text: str) -> None:
        """
        Process a command from text input
        
        Args:
            command_text: Command text to process
        """
        if not command_text:
            return
        
        logger.info(f"Processing command: '{command_text}'")
        
        # Call the command callback
        if self.command_callback:
            self.command_callback(command_text)
    
    def _on_speech_recognized(self, text: str) -> None:
        """
        Callback for speech recognition
        
        Args:
            text: Recognized speech text
        """
        self.process_command(text)

def main():
    """Run the voice integration module for testing"""
    parser = argparse.ArgumentParser(description="Voice integration for Duck Robot")
    parser.add_argument("--no-mic", action="store_true", help="Disable microphone input")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    parser.add_argument("--use-ollama-stt", action="store_true", help="Use Ollama for speech-to-text")
    parser.add_argument("--use-ollama-tts", action="store_true", help="Use Ollama for text-to-speech")
    args = parser.parse_args()
    
    def command_handler(command):
        """Test command handler"""
        print(f"Command received: '{command}'")
        
        # Echo the command back as speech
        if not args.no_tts and processor:
            processor.speak(f"I heard: {command}")
    
    # Initialize voice command processor
    processor = VoiceCommandProcessor(
        command_callback=command_handler,
        use_microphone=not args.no_mic,
        use_tts=not args.no_tts,
        use_ollama_stt=args.use_ollama_stt,
        use_ollama_tts=args.use_ollama_tts
    )
    
    try:
        # Start processing
        processor.start()
        
        print("Voice command testing mode. Type 'exit' to quit.")
        print("You can also speak into your microphone.")
        
        # Allow text input as an alternative to voice
        while True:
            text_input = input("> ")
            if text_input.lower() in ["exit", "quit", "q"]:
                break
            
            processor.process_command(text_input)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        processor.stop()
        print("Voice integration test completed.")

if __name__ == "__main__":
    main() 