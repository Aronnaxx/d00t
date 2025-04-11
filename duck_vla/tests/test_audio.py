import unittest
import logging
import sys
import os
import time
import wave
import numpy as np

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from duck_vla.sounds.audio import AudioPlayer, AudioRecorder, find_beep_file

    AUDIO_UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import audio utils, audio tests will be skipped: {e}")
    AUDIO_UTILS_AVAILABLE = False
except Exception as e:
    logging.warning(f"An unexpected error occurred importing audio utils: {e}")
    AUDIO_UTILS_AVAILABLE = False

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define path for test audio files
TEST_SOUNDS_DIR = os.path.join(project_root, "duck_vla", "tests", "test_sounds")
TEST_RECORDING_FILE = os.path.join(TEST_SOUNDS_DIR, "test_recording.wav")
TEST_BEEP_FILE = os.path.join(TEST_SOUNDS_DIR, "test_beep.wav")


# Helper function to create a dummy WAV file
def create_dummy_wav(filename, duration=0.1, samplerate=16000, channels=1, sampwidth=2):
    logger.debug(f"Creating dummy WAV file: {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(samplerate)
        # Create a simple sine wave (or just silence)
        # n_frames = int(duration * samplerate)
        # sine_wave = (np.sin(2 * np.pi * 440 * np.arange(n_frames) / samplerate) * 32767).astype(np.int16)
        # wf.writeframes(sine_wave.tobytes())
        # For simplicity, write silence
        n_frames = int(duration * samplerate)
        silence = np.zeros(n_frames * channels, dtype=np.int16)
        wf.writeframes(silence.tobytes())
    logger.debug(f"Dummy WAV file created: {filename}")


# Create necessary files/dirs for tests
if AUDIO_UTILS_AVAILABLE:
    create_dummy_wav(TEST_BEEP_FILE)
    # Clean up old recording if it exists
    if os.path.exists(TEST_RECORDING_FILE):
        os.remove(TEST_RECORDING_FILE)


@unittest.skipIf(not AUDIO_UTILS_AVAILABLE, "Audio utils or dependencies not available")
class TestAudio(unittest.TestCase):
    """Tests for audio playback and recording functionality."""

    @classmethod
    def setUpClass(cls):
        logger.info("Setting up TestAudio class...")
        cls.player = None
        cls.recorder = None
        try:
            cls.player = AudioPlayer()
            cls.recorder = AudioRecorder(output_filename=TEST_RECORDING_FILE)
            logger.info("AudioPlayer and AudioRecorder initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize AudioPlayer/Recorder in setUpClass")
            if cls.player:
                cls.player.close()
            if cls.recorder:
                cls.recorder.close()
            raise unittest.SkipTest(f"Skipping audio tests due to initialization failure: {e}")

    @classmethod
    def tearDownClass(cls):
        logger.info("Tearing down TestAudio class...")
        if hasattr(cls, "player") and cls.player:
            cls.player.close()
        if hasattr(cls, "recorder") and cls.recorder:
            cls.recorder.close()
        # Clean up test files
        logger.debug("Cleaning up test audio files...")
        if os.path.exists(TEST_RECORDING_FILE):
            try:
                os.remove(TEST_RECORDING_FILE)
                logger.debug(f"Removed {TEST_RECORDING_FILE}")
            except OSError as e:
                logger.error(f"Error removing test recording file: {e}")
        if os.path.exists(TEST_BEEP_FILE):
            try:
                os.remove(TEST_BEEP_FILE)
                logger.debug(f"Removed {TEST_BEEP_FILE}")
            except OSError as e:
                logger.error(f"Error removing test beep file: {e}")
        if os.path.exists(TEST_SOUNDS_DIR) and not os.listdir(TEST_SOUNDS_DIR):
            try:
                os.rmdir(TEST_SOUNDS_DIR)
                logger.debug(f"Removed empty directory: {TEST_SOUNDS_DIR}")
            except OSError as e:
                logger.error(f"Error removing test sounds directory: {e}")

    def test_01_initialization(self):
        """Test if player and recorder objects were initialized."""
        logger.debug("Running test_01_initialization...")
        self.assertIsNotNone(self.player, "AudioPlayer object should not be None")
        self.assertIsNotNone(self.recorder, "AudioRecorder object should not be None")
        logger.info("AudioPlayer and AudioRecorder initialization test passed.")

    def test_02_find_beep_file(self):
        """Test finding a beep sound file."""
        logger.debug("Running test_02_find_beep_file...")
        # We created a dummy beep in the test_sounds dir
        found_path = find_beep_file("test_beep", sounds_dir=TEST_SOUNDS_DIR)
        self.assertEqual(
            found_path, TEST_BEEP_FILE, f"Should find the dummy beep file at {TEST_BEEP_FILE}"
        )
        logger.info(f"Found beep file: {found_path}")

        # Test finding a non-existent file
        non_existent_path = find_beep_file("non_existent_sound", sounds_dir=TEST_SOUNDS_DIR)
        self.assertIsNone(non_existent_path, "Should return None for a non-existent sound")
        logger.info("Non-existent beep file test passed.")

    def test_03_play_sound(self):
        """Test playing a sound file (uses the dummy beep)."""
        logger.debug("Running test_03_play_sound...")
        self.assertTrue(os.path.exists(TEST_BEEP_FILE), f"Test beep file missing: {TEST_BEEP_FILE}")
        try:
            logger.info(f"Attempting to play {TEST_BEEP_FILE}...")
            self.player.play_sound(TEST_BEEP_FILE)
            # We can't easily verify audio output, so success is just not crashing.
            # A short sleep might be needed if playback is async and quick,
            # although the current implementation seems synchronous.
            # time.sleep(0.2) # Add if needed
            logger.info("Playback command executed without errors.")
        except Exception as e:
            logger.exception("Error during playback")
            self.fail(f"play_sound raised an exception: {e}")

    # @unittest.skip("Skipping recording test temporarily - requires audio input device")
    def test_04_record_audio(self):
        """Test recording a short audio clip."""
        logger.debug("Running test_04_record_audio...")
        record_duration = 0.5  # seconds - keep it short

        if os.path.exists(TEST_RECORDING_FILE):
            logger.warning(f"Removing existing test recording file: {TEST_RECORDING_FILE}")
            os.remove(TEST_RECORDING_FILE)

        try:
            logger.info(f"Starting {record_duration}s recording to {TEST_RECORDING_FILE}...")
            self.recorder.start_recording()
            time.sleep(record_duration)
            self.recorder.stop_recording()
            logger.info("Recording stopped.")

            # Check if the file was created
            self.assertTrue(
                os.path.exists(TEST_RECORDING_FILE),
                f"Recording file was not created at {TEST_RECORDING_FILE}",
            )
            logger.info(f"Recording file created: {TEST_RECORDING_FILE}")

            # Optional: Check file size or basic WAV properties
            file_size = os.path.getsize(TEST_RECORDING_FILE)
            self.assertGreater(
                file_size, 44, "WAV file should be larger than header size"
            )  # 44 bytes is typical WAV header
            logger.info(f"Recording file size: {file_size} bytes")

            # Verify WAV header (basic checks)
            with wave.open(TEST_RECORDING_FILE, "rb") as wf:
                self.assertEqual(
                    wf.getnchannels(), self.recorder.channels, "Channel count mismatch"
                )
                self.assertEqual(wf.getframerate(), self.recorder.rate, "Sample rate mismatch")
                self.assertEqual(
                    wf.getsampwidth(), self.recorder.sample_format_width, "Sample width mismatch"
                )
                n_frames = wf.getnframes()
                expected_frames = int(record_duration * self.recorder.rate)
                # Allow some tolerance for timing variations
                self.assertAlmostEqual(
                    n_frames / self.recorder.rate,
                    record_duration,
                    delta=0.1,
                    msg="Recorded duration mismatch",
                )
                logger.info(
                    f"WAV properties verified: {wf.getnchannels()}ch, {wf.getframerate()}Hz, {wf.getsampwidth()}bytes/sample, {n_frames} frames"
                )

        except Exception as e:
            # Catch specific audio device errors if possible
            # if isinstance(e, SystemError) and "No Default Output Device Available" in str(e):
            #     logger.warning(f"Skipping recording test: No default audio device found. {e}")
            #     raise unittest.SkipTest("Audio device not available for recording test.")
            # elif isinstance(e, IOError) and "Invalid input device" in str(e):
            #      logger.warning(f"Skipping recording test: Invalid input device. {e}")
            #      raise unittest.SkipTest("Invalid input device for recording test.")
            # else:
            logger.exception("Error during recording")
            self.fail(f"Audio recording raised an exception: {e}")


if __name__ == "__main__":
    logger.info("Starting audio tests...")
    unittest.main()
    logger.info("Audio tests finished.")
