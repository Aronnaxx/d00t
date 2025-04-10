"""
Utils module: Audio handling utilities

This module provides audio recording and playback functionality.
"""

import logging
import time
import threading
import queue
from typing import Dict, Any, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

class AudioSystem:
    """
    Handles audio recording and playback for the duck.
    
    This class manages microphone input and speaker output.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        channels: int = 1,
    ):
        """
        Initialize the audio system.
        
        Args:
            sample_rate: Audio sample rate in Hz
            input_device_index: Audio input device index or None for default
            output_device_index: Audio output device index or None for default
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.channels = channels
        
        logger.info(f"Initializing audio system (rate={sample_rate}Hz, channels={channels})")
        
        # Playback state
        self.playing = False
        self.play_thread = None
        
        # Recording state
        self.recording = False
        self.record_thread = None
        self.audio_queue = queue.Queue()
        self.recorded_frames = []
        
        # Log available audio devices
        self._log_device_info()
    
    def _log_device_info(self) -> None:
        """Log information about available audio devices."""
        try:
            devices = sd.query_devices()
            logger.debug(f"Found {len(devices)} audio devices")
            
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            output_devices = [d for d in devices if d['max_output_channels'] > 0]
            
            logger.debug(f"Available input devices: {len(input_devices)}")
            for i, device in enumerate(input_devices):
                logger.debug(f"  {i}: {device['name']} (channels: {device['max_input_channels']})")
                
            logger.debug(f"Available output devices: {len(output_devices)}")
            for i, device in enumerate(output_devices):
                logger.debug(f"  {i}: {device['name']} (channels: {device['max_output_channels']})")
                
            # Log selected devices
            try:
                default_input = sd.query_devices(kind='input')
                logger.debug(f"Default input device: {default_input['name']}")
            except:
                logger.warning("Could not determine default input device")
                
            try:
                default_output = sd.query_devices(kind='output')
                logger.debug(f"Default output device: {default_output['name']}")
            except:
                logger.warning("Could not determine default output device")
            
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
    
    def play_sound(self, filename: str) -> bool:
        """
        Play a sound file.
        
        Args:
            filename: Path to sound file
            
        Returns:
            Success flag
        """
        if self.playing:
            logger.warning("Already playing sound, ignoring request")
            return False
            
        logger.debug(f"Playing sound file: {filename}")
        
        try:
            # Load audio file
            data, file_sample_rate = sf.read(filename, always_2d=True)
            
            # Convert sample rate if needed
            if file_sample_rate != self.sample_rate:
                logger.debug(f"Converting sample rate from {file_sample_rate} to {self.sample_rate}")
                # In a real implementation, you'd resample here
                # For simplicity, we'll just use the file's sample rate
                play_sample_rate = file_sample_rate
            else:
                play_sample_rate = self.sample_rate
            
            # Convert to mono if needed
            if data.shape[1] > self.channels:
                logger.debug(f"Converting from {data.shape[1]} channels to {self.channels}")
                data = data[:, :self.channels]
            
            # Play audio in a separate thread
            self.playing = True
            self.play_thread = threading.Thread(
                target=self._play_audio_thread,
                args=(data, play_sample_rate)
            )
            self.play_thread.daemon = True
            self.play_thread.start()
            
            return True
            
        except Exception as e:
            logger.exception(f"Error playing sound file: {e}")
            self.playing = False
            return False
    
    def _play_audio_thread(self, data: np.ndarray, sample_rate: int) -> None:
        """Audio playback thread function."""
        try:
            logger.debug(f"Starting audio playback ({len(data)} samples)")
            sd.play(
                data,
                sample_rate,
                device=self.output_device_index,
                blocking=True
            )
            logger.debug("Audio playback complete")
        except Exception as e:
            logger.error(f"Error in audio playback: {e}")
        finally:
            self.playing = False
    
    def start_recording(self, duration: Optional[float] = None) -> bool:
        """
        Start recording audio.
        
        Args:
            duration: Recording duration in seconds or None for continuous
            
        Returns:
            Success flag
        """
        if self.recording:
            logger.warning("Already recording audio")
            return False
            
        logger.info(f"Starting audio recording" + 
                   (f" for {duration:.1f}s" if duration else " continuously"))
        
        try:
            # Reset state
            self.recording = True
            self.recorded_frames = []
            
            # Start recording thread
            self.record_thread = threading.Thread(
                target=self._record_audio_thread,
                args=(duration,)
            )
            self.record_thread.daemon = True
            self.record_thread.start()
            
            return True
            
        except Exception as e:
            logger.exception(f"Error starting audio recording: {e}")
            self.recording = False
            return False
    
    def _record_audio_thread(self, duration: Optional[float]) -> None:
        """Audio recording thread function."""
        try:
            def callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio recording status: {status}")
                self.recorded_frames.append(indata.copy())
            
            # Calculate stream duration
            if duration:
                stream_duration = duration
            else:
                stream_duration = None  # Continuous
            
            # Start recording stream
            logger.debug(f"Starting audio recording stream (duration={stream_duration}s)")
            with sd.InputStream(
                samplerate=self.sample_rate,
                device=self.input_device_index,
                channels=self.channels,
                callback=callback
            ):
                if duration:
                    # Wait for specified duration
                    time.sleep(duration)
                else:
                    # Run until stop_recording is called
                    while self.recording:
                        time.sleep(0.1)
            
            logger.debug("Audio recording complete")
            
        except Exception as e:
            logger.error(f"Error in audio recording: {e}")
        finally:
            self.recording = False
    
    def stop_recording(self) -> np.ndarray:
        """
        Stop recording and return the recorded audio.
        
        Returns:
            Numpy array with recorded audio data
        """
        if not self.recording:
            logger.warning("Not recording audio")
            return np.array([])
            
        logger.info("Stopping audio recording")
        
        # Signal recording thread to stop
        self.recording = False
        
        # Wait for thread to finish
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=1.0)
        
        # Combine recorded frames
        if self.recorded_frames:
            result = np.concatenate(self.recorded_frames, axis=0)
            logger.debug(f"Recorded {len(result)} samples")
            return result
        else:
            logger.warning("No audio data recorded")
            return np.array([])
    
    def save_recording(self, filename: str) -> bool:
        """
        Save the last recording to a file.
        
        Args:
            filename: Output file path
            
        Returns:
            Success flag
        """
        if not self.recorded_frames:
            logger.warning("No audio data to save")
            return False
            
        logger.info(f"Saving audio recording to {filename}")
        
        try:
            # Combine recorded frames
            data = np.concatenate(self.recorded_frames, axis=0)
            
            # Write to file
            sf.write(
                filename,
                data,
                self.sample_rate,
                subtype='PCM_16'
            )
            
            logger.debug(f"Saved {len(data)} samples to {filename}")
            return True
            
        except Exception as e:
            logger.exception(f"Error saving audio recording: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current audio system status.
        
        Returns:
            Dict with status information
        """
        return {
            "recording": self.recording,
            "playing": self.playing,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "recorded_frames": len(self.recorded_frames),
        }
    
    def close(self) -> None:
        """Release audio resources."""
        logger.debug("Closing audio system")
        
        # Stop recording if active
        if self.recording:
            self.stop_recording()
            
        # Wait for playback to finish
        if self.playing and self.play_thread:
            self.play_thread.join(timeout=1.0)
