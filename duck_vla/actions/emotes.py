"""
Action module: Emote controller for duck sounds

This module handles playing emotive sounds and beeps for the duck.
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class EmoteController:
    """
    Controls duck emotes by playing appropriate sound effects.
    
    This module manages the duck's expressive beeps and sound effects.
    """
    
    def __init__(self, audio_enabled: bool = True, sounds_dir: Optional[str] = None, audio_system: Optional[Any] = None):
        """
        Initialize the emote controller.
        
        Args:
            audio_enabled: Whether to enable audio feedback
            sounds_dir: Directory containing sound files for emotes
            audio_system: Existing AudioSystem instance to reuse (optional)
        """
        from pathlib import Path
        
        self.audio_enabled = audio_enabled
        
        # Set up sounds directory
        if sounds_dir is None:
            # Default to sounds directory under duck_vla/sounds
            script_dir = Path(__file__).parent
            self.sounds_dir = str(script_dir.parent / "sounds")
        else:
            self.sounds_dir = sounds_dir
            
        logger.info(f"Initializing emote controller (audio_enabled={audio_enabled})")
        
        # Configure sound map - category to filename pattern mapping
        self.sound_map = {
            "happy": ["happy*.wav"],
            "sad": ["sad*.wav"],
            "curious": ["curious*.wav", "interest*.wav"],
            "afraid": ["afraid*.wav", "scared*.wav"],
            "hello": ["hello*.wav", "greeting*.wav"],
            "goodbye": ["goodbye*.wav", "bye*.wav"],
            "neutral": ["neutral*.wav"],
            "error": ["error*.wav", "warning*.wav"],
            "success": ["success*.wav", "win*.wav"],
            "processing": ["processing*.wav", "thinking*.wav"],
            "warning": ["warning*.wav", "alert*.wav"],
        }
        
        # Import audio utilities if enabled
        if audio_enabled:
            if audio_system is not None:
                # Reuse existing audio system
                self.audio_system = audio_system
                logger.debug("Reusing existing audio system for emotes")
            else:
                try:
                    from duck_vla.sounds.audio import AudioSystem
                    self.audio_system = AudioSystem()
                    logger.debug("Audio system initialized for emotes")
                except ImportError as e:
                    logger.warning(f"Failed to initialize audio system for emotes: {e}")
                    self.audio_system = None
                    self.audio_enabled = False
        else:
            logger.info("Audio disabled for emotes")
            self.audio_system = None
            
        # Check for sound files and log warnings if missing
        self._check_sound_files()
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        Path(self.sounds_dir).mkdir(parents=True, exist_ok=True)
    
    def _check_sound_files(self) -> None:
        """Check if required sound files exist and log warnings for missing files."""
        # Track missing sound categories to warn only once per category
        missing_categories = set()
        
        # Check each emote category
        for emote, filenames in self.sound_map.items():
            has_any_file = False
            
            for filename in filenames:
                file_path = os.path.join(self.sounds_dir, filename)
                if os.path.exists(file_path):
                    has_any_file = True
                    break
            
            if not has_any_file:
                missing_categories.add(emote)
                logger.warning(f"No sound files found for emote '{emote}'")
        
        # If any categories are missing, provide information on how to add sound files
        if missing_categories:
            logger.warning(f"Missing sound files for {len(missing_categories)} emote categories")
            logger.info(f"Place .wav files in {self.sounds_dir} following the naming convention in the sound_map")
            
            # Create placeholder file with instructions if directory is empty
            if not os.listdir(self.sounds_dir):
                readme_path = os.path.join(self.sounds_dir, "README.txt")
                try:
                    with open(readme_path, 'w') as f:
                        f.write("Duck VLA Sound Files\n\n")
                        f.write("Place .wav files in this directory with the following naming convention:\n\n")
                        for emote in self.sound_map:
                            f.write(f"{emote}_N.wav - where N is a number (1, 2, etc.)\n")
                        f.write("\nExample: happy_1.wav, happy_2.wav, sad_1.wav\n")
                    logger.info(f"Created README at {readme_path}")
                except Exception as e:
                    logger.error(f"Error creating README: {e}")
    
    def play(self, emote: str) -> bool:
        """
        Play an emote sound.
        
        Args:
            emote: Emote type to play
            
        Returns:
            Success flag
        """
        if not self.audio_enabled or self.audio_system is None:
            logger.debug(f"Audio disabled, would play '{emote}' emote")
            return False
            
        logger.info(f"Playing '{emote}' emote")
        
        try:
            # Find matching sound files
            filenames = self.sound_map.get(emote, self.sound_map.get("neutral", []))
            
            if not filenames:
                logger.warning(f"No sound files defined for emote '{emote}'")
                return False
            
            # Filter to only existing files
            existing_files = []
            for filename in filenames:
                file_path = os.path.join(self.sounds_dir, filename)
                if os.path.exists(file_path):
                    existing_files.append(file_path)
            
            if not existing_files:
                logger.warning(f"No sound files found for emote '{emote}'")
                return False
            
            # Choose a random sound file from available options
            chosen_file = random.choice(existing_files)
            
            # Play the sound
            success = self.audio_system.play_sound(chosen_file)
            return success
            
        except Exception as e:
            logger.exception(f"Error playing emote '{emote}': {e}")
            return False
    
    def get_available_emotes(self) -> List[str]:
        """
        Get a list of available emote types.
        
        Returns:
            List of emote type strings
        """
        return list(self.sound_map.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current emote controller status.
        
        Returns:
            Dict with status information
        """
        # Count available sound files
        available_sounds = {}
        for emote, filenames in self.sound_map.items():
            count = 0
            for filename in filenames:
                if os.path.exists(os.path.join(self.sounds_dir, filename)):
                    count += 1
            available_sounds[emote] = count
        
        return {
            "audio_enabled": self.audio_enabled,
            "sounds_dir": self.sounds_dir,
            "available_sounds": available_sounds,
        }
