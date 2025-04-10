"""
Language module: Speech-to-text functionality

This module handles speech recognition using Vosk for offline processing.
"""

import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel

# Configure Vosk logging (0 for most verbose, higher numbers for less)
SetLogLevel(0)

logger = logging.getLogger(__name__)

class SpeechToText:
    """
    Speech recognition using Vosk for offline processing.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        device_index: Optional[int] = None,
        buffer_duration: float = 2.0,
    ):
        """
        Initialize the speech recognition system.
        
        Args:
            model_path: Path to Vosk model directory
            sample_rate: Audio sample rate in Hz
            device_index: Audio device index or None for default
            buffer_duration: Duration of audio buffer in seconds
        """
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.buffer_duration = buffer_duration
        self.buffer_size = int(self.sample_rate * buffer_duration)
        
        # Default model path if not specified
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "../models/vosk")
        self.model_path = model_path
        
        logger.info(f"Initializing speech-to-text (model={model_path}, rate={sample_rate}Hz)")
        
        # Audio streams and processing
        self.stream = None
        self.audio_queue = queue.Queue()
        self.recording = False
        self.recognizer = None
        
        # Results
        self.latest_text = ""
        self.confidence = 0.0
        
        # Load model
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the Vosk model."""
        try:
            # Create model directory if needed
            Path(self.model_path).mkdir(parents=True, exist_ok=True)
            
            # Check if model exists, if not, log message about downloading
            model_complete_path = os.path.join(self.model_path, "model")
            if not os.path.exists(model_complete_path) or not os.path.isdir(model_complete_path):
                logger.warning(f"Vosk model not found at {model_complete_path}")
                logger.warning("Please download a model from https://alphacephei.com/vosk/models")
                logger.warning("and extract it to the models directory.")
                logger.warning("For English, the small model is recommended: vosk-model-small-en-us-0.15.zip")
                raise FileNotFoundError(f"Vosk model not found at {model_complete_path}")
            
            # Load the model
            logger.debug(f"Loading Vosk model from {model_complete_path}")
            self.model = Model(model_complete_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            
            # Enable words with timestamps
            self.recognizer.SetWords(True)
            
            logger.info("Speech recognition model loaded successfully")
            
        except Exception as e:
            logger.exception(f"Error loading speech recognition model: {e}")
            raise RuntimeError(f"Failed to initialize speech recognition: {e}")
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream processing."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Put audio data in queue
        self.audio_queue.put(bytes(indata))
    
    def start_listening(self) -> None:
        """Start listening for audio input."""
        if self.recording:
            logger.warning("Already recording")
            return
            
        try:
            logger.debug("Starting audio stream")
            self.recording = True
            self.stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                device=self.device_index,
                dtype="int16",
                channels=1,
                callback=self._audio_callback
            )
            self.stream.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            logger.info("Audio recording started")
            
        except Exception as e:
            logger.exception(f"Error starting audio recording: {e}")
            self.recording = False
            if self.stream:
                self.stream.close()
                self.stream = None
    
    def stop_listening(self) -> None:
        """Stop listening for audio input."""
        if not self.recording:
            return
            
        logger.debug("Stopping audio stream")
        self.recording = False
        
        if self.stream:
            self.stream.close()
            self.stream = None
            
        # Process any remaining audio in the queue
        while not self.audio_queue.empty():
            data = self.audio_queue.get()
            if self.recognizer and data:
                self.recognizer.AcceptWaveform(data)
        
        logger.info("Audio recording stopped")
    
    def _process_audio(self) -> None:
        """Process audio data from the queue."""
        logger.debug("Audio processing thread started")
        
        while self.recording:
            try:
                # Get audio data from queue with timeout
                data = self.audio_queue.get(timeout=1.0)
                
                if self.recognizer and data:
                    # Process audio data
                    if self.recognizer.AcceptWaveform(data):
                        # Get recognition result
                        result_json = self.recognizer.Result()
                        result = json.loads(result_json)
                        
                        if "text" in result and result["text"].strip():
                            self.latest_text = result["text"]
                            self.confidence = result.get("confidence", 0.0)
                            
                            logger.info(f"Recognized: '{self.latest_text}' (conf: {self.confidence:.2f})")
                
            except queue.Empty:
                # Queue timeout, just continue
                pass
            except Exception as e:
                logger.exception(f"Error processing audio: {e}")
        
        logger.debug("Audio processing thread stopped")
    
    def listen(self) -> Optional[str]:
        """
        Listen for a single utterance.
        
        Returns:
            Recognized text or None if no utterance detected
        """
        # Make sure we're recording
        if not self.recording:
            self.start_listening()
            
        try:
            # Reset previous result
            previous_text = self.latest_text
            self.latest_text = ""
            
            # Wait for a short period to collect audio
            # In a real application, you'd want smarter voice activity detection
            import time
            time.sleep(3.0)  # Wait for 3 seconds of audio
            
            # Check if we got new text
            if self.latest_text and self.latest_text != previous_text:
                return self.latest_text
            return None
            
        except Exception as e:
            logger.exception(f"Error during listening: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the speech recognition system.
        
        Returns:
            Dict with status information
        """
        return {
            "recording": self.recording,
            "latest_text": self.latest_text,
            "confidence": self.confidence,
            "sample_rate": self.sample_rate,
            "model_path": self.model_path,
        }
    
    def close(self) -> None:
        """Release resources."""
        logger.debug("Closing speech recognition resources")
        self.stop_listening()
