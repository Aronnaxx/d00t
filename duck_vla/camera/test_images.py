"""
Test Images for Camera Simulation

This module provides test images for simulation testing.
"""

import logging
import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Define test image directory relative to this file
TEST_IMAGES_DIR = Path(__file__).parent / "test_images"


class TestImageProvider:
    """Provides test images for simulation testing."""

    def __init__(self, test_dir: Optional[str] = None):
        """
        Initialize the test image provider.

        Args:
            test_dir: Directory containing test images or None to use default
        """
        self.test_dir = Path(test_dir) if test_dir else TEST_IMAGES_DIR
        self.current_index = 0
        self.image_files: List[Path] = []

        # Create test directory if it doesn't exist
        if not self.test_dir.exists():
            logger.info(f"Creating test image directory: {self.test_dir}")
            os.makedirs(self.test_dir, exist_ok=True)

            # Create a sample test image if none exist
            self._create_sample_test_image()

        # Load image file list
        self._load_image_files()

        logger.info(f"Initialized test image provider with {len(self.image_files)} images")

    def _load_image_files(self) -> None:
        """Load the list of image files from the test directory."""
        # Find all image files in the test directory
        self.image_files = list(self.test_dir.glob("*.png"))
        self.image_files.extend(self.test_dir.glob("*.jpg"))
        self.image_files.extend(self.test_dir.glob("*.jpeg"))

        if not self.image_files:
            logger.warning(f"No test images found in {self.test_dir}")
            self._create_sample_test_image()
            self._load_image_files()

    def _create_sample_test_image(self) -> None:
        """Create a sample test image if none exist."""
        try:
            import cv2
            import numpy as np

            logger.info("Creating sample test image")

            # Create a simple color gradient image
            height, width = 480, 640
            image = np.zeros((height, width, 3), dtype=np.uint8)

            # Create a simple color gradient
            for y in range(height):
                for x in range(width):
                    # Create RGB channels with gradients
                    b = int(255 * (x / width))
                    g = int(255 * (y / height))
                    r = int(255 * ((x + y) / (width + height)))
                    image[y, x] = [b, g, r]

            # Add some shapes to the image
            # Circle in the center
            cv2.circle(image, (width // 2, height // 2), 50, (0, 0, 255), -1)
            # Rectangle in the corner
            cv2.rectangle(image, (50, 50), (150, 150), (0, 255, 0), -1)
            # Text
            cv2.putText(
                image,
                "Test Image",
                (width // 2 - 100, height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Save the image
            output_path = self.test_dir / "sample_test_image.png"
            cv2.imwrite(str(output_path), image)
            logger.info(f"Created sample test image at {output_path}")

        except ImportError:
            logger.warning("OpenCV not available, creating simple test image")
            # Create a simple black and white test pattern
            height, width = 480, 640
            image = np.zeros((height, width, 3), dtype=np.uint8)

            # Create a checkerboard pattern
            square_size = 40
            for y in range(0, height, square_size):
                for x in range(0, width, square_size):
                    if (x // square_size + y // square_size) % 2 == 0:
                        image[y : y + square_size, x : x + square_size] = 255

            # Save the image using PIL
            try:
                from PIL import Image

                output_path = self.test_dir / "sample_test_image.png"
                Image.fromarray(image).save(str(output_path))
                logger.info(f"Created simple test image at {output_path}")
            except ImportError:
                logger.error("Neither OpenCV nor PIL available to create test image")

    def get_next_image(self) -> Optional[np.ndarray]:
        """
        Get the next test image in sequence.

        Returns:
            Image as a numpy array (BGR format) or None if no images
        """
        if not self.image_files:
            logger.warning("No test images available")
            return None

        # Get the next image file
        image_file = self.image_files[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.image_files)

        try:
            # Try loading with OpenCV first
            try:
                import cv2

                image = cv2.imread(str(image_file))
                logger.debug(f"Loaded test image: {image_file}")
                return image
            except ImportError:
                # Fall back to PIL if OpenCV is not available
                from PIL import Image
                import numpy as np

                pil_image = Image.open(image_file)
                # Convert to numpy array (RGB format)
                image = np.array(pil_image)
                # Convert RGB to BGR for consistency with OpenCV
                if image.shape[-1] == 3:  # Color image
                    image = image[:, :, ::-1].copy()

                logger.debug(f"Loaded test image with PIL: {image_file}")
                return image

        except Exception as e:
            logger.error(f"Error loading test image {image_file}: {e}")
            return None

    def get_random_image(self) -> Optional[np.ndarray]:
        """
        Get a random test image.

        Returns:
            Image as a numpy array (BGR format) or None if no images
        """
        if not self.image_files:
            logger.warning("No test images available")
            return None

        # Get a random image file
        import random

        image_file = random.choice(self.image_files)

        try:
            # Try loading with OpenCV first
            try:
                import cv2

                image = cv2.imread(str(image_file))
                logger.debug(f"Loaded random test image: {image_file}")
                return image
            except ImportError:
                # Fall back to PIL if OpenCV is not available
                from PIL import Image
                import numpy as np

                pil_image = Image.open(image_file)
                # Convert to numpy array (RGB format)
                image = np.array(pil_image)
                # Convert RGB to BGR for consistency with OpenCV
                if image.shape[-1] == 3:  # Color image
                    image = image[:, :, ::-1].copy()

                logger.debug(f"Loaded random test image with PIL: {image_file}")
                return image

        except Exception as e:
            logger.error(f"Error loading random test image {image_file}: {e}")
            return None


# Simple test if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    provider = TestImageProvider()
    image = provider.get_next_image()
    if image is not None:
        print(f"Loaded test image: shape={image.shape}, dtype={image.dtype}")
    else:
        print("No test image available")
