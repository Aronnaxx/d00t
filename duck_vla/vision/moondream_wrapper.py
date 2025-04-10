"""
Vision module: Moondream integration for image understanding

This module wraps the Moondream vision-language model to provide
image captioning and visual question answering capabilities.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np
try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MoondreamVision:
    """
    Wrapper for vision-language models for image understanding.
    Supports HuggingFace Moondream and Ollama models.

    This class provides image captioning and visual question answering
    capabilities using the specified model backend.
    """

    def __init__(
        self,
        backend: str = "huggingface", # or "ollama"
        model_id: str = "huggingface/Moondream1", # HuggingFace ID or Ollama model name
        ollama_host: Optional[str] = None, # e.g., "http://localhost:11434"
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_half_precision: bool = True,
    ):
        """
        Initialize the vision system.

        Args:
            backend: Model backend to use ('huggingface' or 'ollama').
            model_id: HuggingFace model ID or Ollama model name (e.g., 'moondream', 'llava').
                      For Ollama, ensure the model is pulled (`ollama pull <model_name>`).
            ollama_host: Host URL for the Ollama API. Defaults to Ollama client default.
            model_dir: Custom directory to load/save HuggingFace models. Ignored for Ollama.
            device: Device to run HuggingFace model on ('cpu', 'cuda', or None for auto). Ignored for Ollama.
            use_half_precision: Whether to use FP16 for HuggingFace model. Ignored for Ollama.
        """
        logger.info(f"Initializing vision system with backend: {backend}, model: {model_id}")
        self.backend = backend.lower()
        self.model_id = model_id
        self.model_dir = model_dir
        self.device = device
        self.use_half_precision = use_half_precision
        self.ollama_host = ollama_host

        self.processor = None
        self.model = None
        self.ollama_client = None

        if self.backend == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                logger.error("HuggingFace Transformers or PyTorch not installed. Please install them to use the 'huggingface' backend.")
                raise ImportError("HuggingFace Transformers or PyTorch not installed.")
            self._initialize_huggingface_model()
        elif self.backend == "ollama":
            if not OLLAMA_AVAILABLE:
                logger.error("Ollama client not installed. Please install it (`pip install ollama`) to use the 'ollama' backend.")
                raise ImportError("Ollama client not installed.")
            self._initialize_ollama_client()
        else:
            logger.error(f"Unsupported backend: {self.backend}. Choose 'huggingface' or 'ollama'.")
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _initialize_huggingface_model(self) -> None:
        """Initialize the HuggingFace Moondream model and processor."""
        if not HUGGINGFACE_AVAILABLE: return # Should have been caught in __init__

        self.model_dir = self.model_dir or os.path.join(os.path.dirname(__file__), "../models/moondream")

        # Determine device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.use_half_precision = self.use_half_precision and self.device == "cuda"

        logger.info(f"Initializing HuggingFace Moondream (device={self.device}, fp16={self.use_half_precision})")

        try:
            # Create model directory if needed
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)

            logger.debug(f"Loading HuggingFace processor from {self.model_id}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.model_dir
            )

            logger.debug(f"Loading HuggingFace model from {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.model_dir,
                torch_dtype=torch.float16 if self.use_half_precision else torch.float32
            )

            # Move model to device
            self.model.to(self.device)

            logger.info(f"HuggingFace Moondream model loaded successfully to {self.device}")

        except Exception as e:
            logger.exception(f"Failed to initialize HuggingFace Moondream model: {e}")
            raise RuntimeError(f"Failed to initialize HuggingFace Moondream: {e}")

    def _initialize_ollama_client(self) -> None:
        """Initialize the Ollama client."""
        if not OLLAMA_AVAILABLE: return # Should have been caught in __init__

        logger.info(f"Initializing Ollama client for model '{self.model_id}'" + (f" at host {self.ollama_host}" if self.ollama_host else ""))
        try:
            self.ollama_client = ollama.Client(host=self.ollama_host)
            
            # Check connection - just make a basic API call
            logger.debug("Testing Ollama connection...")
            self.ollama_client.list()
            logger.debug("Ollama connection successful")
            
            # No need to check if model exists - just use it directly
            # Ollama will handle model loading when we make API calls
            logger.info(f"Ollama client initialized successfully for model '{self.model_id}'")

        except Exception as e:
            logger.exception(f"Failed to initialize Ollama client: {e}")
            raise RuntimeError(f"Failed to initialize Ollama: {e}")

    def _pull_ollama_model(self, model_name: str):
        """Attempts to pull the specified Ollama model."""
        if not self.ollama_client:
            logger.error("Ollama client not initialized, cannot pull model.")
            return

        logger.info(f"Pulling Ollama model '{model_name}'. This may take a while...")
        current_digest = ""
        try:
            for progress in self.ollama_client.pull(model_name, stream=True):
                digest = progress.get("digest", "")
                if digest != current_digest and current_digest != "":
                    # Log completion of a layer or file
                     logger.info(f"Pulling '{model_name}': {current_digest} complete.")
                current_digest = digest

                if total := progress.get("total"):
                     logger.debug(f"Pulling '{model_name}': {progress.get('completed', 0) / total * 100:.2f}%")

                if status := progress.get("status"):
                     if "pulling" in status.lower():
                        logger.debug(f"Pulling '{model_name}': {status}")
                     else:
                        logger.info(f"Pulling '{model_name}': {status}")


            logger.info(f"Successfully pulled Ollama model '{model_name}'.")
        except Exception as e:
            logger.error(f"Error pulling Ollama model '{model_name}': {e}")
            raise # Re-raise the exception


    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a camera frame and extract visual information using the configured backend.

        Args:
            frame: Image as numpy array (HxWxC BGR format expected by OpenCV)

        Returns:
            Dict with visual processing results (caption, person_detected, scene analysis).
        """
        logger.debug(f"Processing frame using {self.backend} backend.")
        if self.backend == "huggingface":
            if self.model is None or self.processor is None:
                logger.error("HuggingFace model not initialized")
                return {"error": "HuggingFace model not initialized"}
            return self._process_frame_huggingface(frame)
        elif self.backend == "ollama":
            if self.ollama_client is None:
                logger.error("Ollama client not initialized")
                return {"error": "Ollama client not initialized"}
            return self._process_frame_ollama(frame)
        else:
            # This case should ideally not be reached due to __init__ checks
            logger.error(f"Invalid backend '{self.backend}' during frame processing.")
            return {"error": f"Invalid backend '{self.backend}'"}

    def _process_frame_huggingface(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame using HuggingFace Moondream."""
        try:
            logger.debug("Processing frame with HuggingFace Moondream")
            # Moondream expects RGB PIL Image, convert from BGR numpy array
            from PIL import Image
            image_rgb = Image.fromarray(frame[:, :, ::-1]) # Convert BGR to RGB

            # Get basic caption
            caption = self._generate_caption_huggingface(image_rgb)

            # Check if we see a person
            contains_person = self._check_for_person_huggingface(image_rgb, caption)

            # Get basic scene understanding
            scene_info = self._analyze_scene_huggingface(image_rgb)

            results = {
                "caption": caption,
                "person_detected": contains_person,
                "scene": scene_info,
                "backend": "huggingface"
            }

            logger.debug(f"HuggingFace frame processing complete: {results}")
            return results

        except Exception as e:
            logger.exception(f"Error processing frame with HuggingFace: {e}")
            return {"error": str(e), "backend": "huggingface"}

    def _process_frame_ollama(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame using Ollama."""
        try:
            logger.debug(f"Processing frame with Ollama model '{self.model_id}'")
            # Ollama Python library expects image bytes. Convert BGR numpy array to PNG bytes.
            import cv2
            _, img_encoded = cv2.imencode('.png', frame)
            image_bytes = img_encoded.tobytes()
            
            # Get basic caption
            caption = self._generate_caption_ollama(image_bytes)
            
            # Check if we see a person
            contains_person = self._check_for_person_ollama(image_bytes, caption)
            
            # Get basic scene understanding
            scene_info = self._analyze_scene_ollama(image_bytes)
            
            results = {
                "caption": caption,
                "person_detected": contains_person,
                "scene": scene_info,
                "backend": "ollama"
            }
            
            logger.debug(f"Ollama frame processing complete: {results}")
            return results
            
        except Exception as e:
            logger.exception(f"Error processing frame with Ollama: {e}")
            # Return minimal error response but don't crash
            return {
                "error": str(e),
                "caption": "Error processing image",
                "person_detected": False,
                "scene": {"error": str(e)},
                "backend": "ollama"
            }


    def _generate_caption_huggingface(self, image: 'Image.Image') -> str:
        """Generate a caption for the image using HuggingFace Moondream."""
        if not HUGGINGFACE_AVAILABLE or self.model is None or self.processor is None:
            logger.error("HuggingFace components not available for caption generation.")
            return "Error: HuggingFace components unavailable"
        try:
            logger.debug("Generating image caption (HuggingFace)")
            prompt = "<image>\nDescribe this image in detail."
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)

            caption = self.processor.decode(output[0], skip_special_tokens=True)
            caption = caption.split(prompt.split('\n')[1])[-1].strip() # More robust split

            logger.debug(f"HuggingFace caption generated: {caption}")
            return caption

        except Exception as e:
            logger.exception(f"Error generating HuggingFace caption: {e}")
            return "Error generating HuggingFace caption"

    def _generate_caption_ollama(self, image_bytes: bytes) -> str:
        """Generate a caption for the image using Ollama."""
        if not OLLAMA_AVAILABLE or self.ollama_client is None:
             logger.error("Ollama client not available for caption generation.")
             return "Error: Ollama client unavailable"
        try:
            logger.debug("Generating image caption (Ollama)")
            prompt = "Describe this image in detail."
            response = self.ollama_client.generate(
                model=self.model_id,
                prompt=prompt,
                images=[image_bytes],
                stream=False # Get full response at once
            )
            caption = response.get('response', 'Error: No response from Ollama').strip()
            logger.debug(f"Ollama caption generated: {caption}")
            return caption
        except Exception as e:
             logger.exception(f"Error generating Ollama caption: {e}")
             return "Error generating Ollama caption"


    def _check_for_person_huggingface(self, image: 'Image.Image', caption: Optional[str] = None) -> bool:
        """Check if a person is present in the image using HuggingFace Moondream."""
        if not HUGGINGFACE_AVAILABLE or self.model is None or self.processor is None:
            logger.error("HuggingFace components not available for person check.")
            return False

        # Quick check in caption if available
        if caption and "error" not in caption.lower():
            person_keywords = ["person", "people", "man", "woman", "child", "boy", "girl", "human"]
            if any(keyword in caption.lower() for keyword in person_keywords):
                logger.debug("Person likely detected in HuggingFace caption")
                return True

        try:
            logger.debug("Checking for person with HuggingFace VQA")
            prompt = "<image>\nIs there a person in this image? Answer yes or no."
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)

            answer = self.processor.decode(output[0], skip_special_tokens=True)
            answer = answer.split(prompt.split('\n')[1])[-1].strip().lower() # More robust split

            contains_person = "yes" in answer
            logger.debug(f"HuggingFace person detection VQA result: '{answer}' -> {contains_person}")
            return contains_person

        except Exception as e:
            logger.exception(f"Error checking for person with HuggingFace: {e}")
            return False

    def _check_for_person_ollama(self, image_bytes: bytes, caption: Optional[str] = None) -> bool:
        """Check if a person is present in the image using Ollama."""
        if not OLLAMA_AVAILABLE or self.ollama_client is None:
             logger.error("Ollama client not available for person check.")
             return False

        # Quick check in caption if available
        if caption and "error" not in caption.lower():
            person_keywords = ["person", "people", "man", "woman", "child", "boy", "girl", "human"]
            if any(keyword in caption.lower() for keyword in person_keywords):
                logger.debug("Person likely detected in Ollama caption")
                return True

        try:
            logger.debug("Checking for person with Ollama VQA")
            prompt = "Is there a person in this image? Answer strictly with 'Yes.' or 'No.'."
            response = self.ollama_client.generate(
                model=self.model_id,
                prompt=prompt,
                images=[image_bytes],
                stream=False,
                options={"temperature": 0.0} # Make it deterministic
            )
            answer = response.get('response', 'Error').strip().lower()
            # Handle potential variations like "Yes." vs "yes"
            contains_person = answer.startswith("yes")
            logger.debug(f"Ollama person detection VQA result: '{answer}' -> {contains_person}")
            return contains_person
        except Exception as e:
             logger.exception(f"Error checking for person with Ollama: {e}")
             return False


    def _analyze_scene_huggingface(self, image: 'Image.Image') -> Dict[str, Any]:
        """Analyze the scene using HuggingFace Moondream VQA."""
        if not HUGGINGFACE_AVAILABLE or self.model is None or self.processor is None:
            logger.error("HuggingFace components not available for scene analysis.")
            return {"error": "HuggingFace components unavailable"}

        scene_info = {}
        try:
            questions = [
                "What is the main object or focus in this image?",
                "Is this scene primarily indoors or outdoors?",
                "What colors are most prominent in this image?",
            ]

            for question in questions:
                logger.debug(f"Asking HuggingFace VQA question: {question}")
                prompt = f"<image>\n{question}"
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)

                answer = self.processor.decode(output[0], skip_special_tokens=True)
                answer = answer.split(question)[-1].strip()

                # Use a simplified key
                key = question.lower().replace("?", "").replace("what is", "").replace("what", "").replace("is this", "").strip()
                key = '_'.join(key.split()[:3]) # e.g., 'main_object_or', 'scene_primarily_indoors', 'colors_are_most'
                scene_info[key] = answer
                logger.debug(f"Scene analysis (HF) - {key}: {answer}")


            logger.debug(f"HuggingFace scene analysis complete: {scene_info}")
            return scene_info

        except Exception as e:
            logger.exception(f"Error analyzing scene with HuggingFace: {e}")
            scene_info["error"] = str(e)
            return scene_info

    def _analyze_scene_ollama(self, image_bytes: bytes) -> Dict[str, Any]:
         """Analyze the scene using Ollama VQA."""
         if not OLLAMA_AVAILABLE or self.ollama_client is None:
             logger.error("Ollama client not available for scene analysis.")
             return {"error": "Ollama client unavailable"}

         scene_info = {}
         logger.warning("Detailed scene analysis with Ollama is currently basic/not fully implemented.")
         # Placeholder implementation - can be expanded similarly to HuggingFace version
         # Making multiple calls to Ollama for different questions can be slow.
         # Consider a single prompt asking for multiple details if performance is an issue.
         try:
             # Example: Ask one combined question
             prompt = "Describe the main subject, whether it's indoors or outdoors, and the prominent colors."
             logger.debug(f"Asking Ollama VQA question: {prompt}")
             response = self.ollama_client.generate(
                 model=self.model_id,
                 prompt=prompt,
                 images=[image_bytes],
                 stream=False
             )
             answer = response.get('response', 'Error').strip()
             scene_info['ollama_description'] = answer
             logger.debug(f"Ollama scene analysis response: {answer}")

         except Exception as e:
             logger.exception(f"Error analyzing scene with Ollama: {e}")
             scene_info["error"] = str(e)

         return scene_info

    def ask_question(self, image: Union[np.ndarray, bytes, 'Image.Image'], question: str) -> str:
        """
        Ask a specific question about the image using the configured backend.

        Args:
            image: Image input. Can be:
                   - Numpy array (HxWxC BGR)
                   - Image bytes (e.g., PNG) - preferred for Ollama
                   - PIL Image object - preferred for HuggingFace
            question: The question to ask about the image.

        Returns:
            The answer from the VLM.
        """
        logger.debug(f"Asking question using {self.backend}: '{question}'")

        if self.backend == "huggingface":
             if not HUGGINGFACE_AVAILABLE or self.model is None or self.processor is None:
                 logger.error("HuggingFace components not available for VQA.")
                 return "Error: HuggingFace components unavailable"

             # Ensure image is PIL Image
             pil_image: 'Image.Image'
             if isinstance(image, np.ndarray):
                 from PIL import Image
                 pil_image = Image.fromarray(image[:, :, ::-1]) # BGR to RGB
             elif not hasattr(image, 'save'): # Check if it looks like a PIL Image
                 logger.error("Invalid image type for HuggingFace backend. Expected NumPy array or PIL Image.")
                 return "Error: Invalid image type for HuggingFace"
             else:
                 pil_image = image

             try:
                 prompt = f"<image>\n{question}"
                 inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)
                 with torch.no_grad():
                     output = self.model.generate(**inputs, max_new_tokens=50, do_sample=False) # Adjust max_tokens as needed
                 answer = self.processor.decode(output[0], skip_special_tokens=True)
                 answer = answer.split(question)[-1].strip()
                 logger.debug(f"HuggingFace VQA answer: {answer}")
                 return answer
             except Exception as e:
                 logger.exception(f"Error asking question with HuggingFace: {e}")
                 return f"Error asking HuggingFace: {e}"

        elif self.backend == "ollama":
             if not OLLAMA_AVAILABLE or self.ollama_client is None:
                 logger.error("Ollama client not available for VQA.")
                 return "Error: Ollama client unavailable"

             # Ensure image is bytes
             image_bytes: bytes
             if isinstance(image, np.ndarray):
                 import cv2
                 _, img_encoded = cv2.imencode('.png', image)
                 image_bytes = img_encoded.tobytes()
             elif isinstance(image, bytes):
                 image_bytes = image
             elif hasattr(image, 'save'): # Handle PIL Image
                 import io
                 byte_arr = io.BytesIO()
                 image.save(byte_arr, format='PNG')
                 image_bytes = byte_arr.getvalue()
             else:
                  logger.error("Invalid image type for Ollama backend. Expected NumPy array, bytes, or PIL Image.")
                  return "Error: Invalid image type for Ollama"

             try:
                 response = self.ollama_client.generate(
                     model=self.model_id,
                     prompt=question,
                     images=[image_bytes],
                     stream=False
                 )
                 answer = response.get('response', 'Error: No response from Ollama').strip()
                 logger.debug(f"Ollama VQA answer: {answer}")
                 return answer
             except Exception as e:
                 logger.exception(f"Error asking question with Ollama: {e}")
                 return f"Error asking Ollama: {e}"

        else:
            logger.error(f"Invalid backend '{self.backend}' during question asking.")
            return f"Error: Invalid backend '{self.backend}'"

# Example Usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Starting Moondream wrapper test...")

    # --- Test HuggingFace Backend ---
    if HUGGINGFACE_AVAILABLE:
        logger.info("\n--- Testing HuggingFace Backend ---")
        try:
            # Create a dummy black image
            dummy_image_hf = np.zeros((100, 100, 3), dtype=np.uint8)
            from PIL import Image
            dummy_pil_image = Image.fromarray(dummy_image_hf)

            vision_hf = MoondreamVision(backend="huggingface") # Uses default moondream1

            # Test processing
            logger.info("Testing frame processing (HuggingFace)...")
            results_hf = vision_hf.process_frame(dummy_image_hf)
            logger.info(f"HuggingFace process_frame results: {results_hf}")

            # Test VQA
            logger.info("Testing VQA (HuggingFace)...")
            question_hf = "What color is this image?"
            answer_hf = vision_hf.ask_question(dummy_pil_image, question_hf)
            logger.info(f"HuggingFace VQA - Q: '{question_hf}' A: '{answer_hf}'")

        except Exception as e:
            logger.exception(f"Error testing HuggingFace backend: {e}")
    else:
        logger.warning("Skipping HuggingFace tests: transformers or torch not installed.")


    # --- Test Ollama Backend ---
    if OLLAMA_AVAILABLE:
        logger.info("\n--- Testing Ollama Backend ---")
        try:
            # Create dummy image bytes
            import cv2
            dummy_image_ollama = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.putText(dummy_image_ollama, 'Test', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, img_encoded = cv2.imencode('.png', dummy_image_ollama)
            dummy_image_bytes = img_encoded.tobytes()

            # Test with default moondream (requires `ollama pull moondream`)
            logger.info("Initializing Ollama with 'moondream' model...")
            # Ensure Ollama server is running: `ollama serve`
            # Ensure model is pulled: `ollama pull moondream`
            vision_ollama = MoondreamVision(backend="ollama", model_id="moondream")

            # Test processing
            logger.info("Testing frame processing (Ollama)...")
            results_ollama = vision_ollama.process_frame(dummy_image_ollama)
            logger.info(f"Ollama process_frame results: {results_ollama}")

            # Test VQA
            logger.info("Testing VQA (Ollama)...")
            question_ollama = "What text is written in the image?"
            answer_ollama = vision_ollama.ask_question(dummy_image_bytes, question_ollama)
            logger.info(f"Ollama VQA - Q: '{question_ollama}' A: '{answer_ollama}'")

            # Example with another Ollama VLM like Llava (requires `ollama pull llava`)
            # logger.info("\nInitializing Ollama with 'llava' model...")
            # vision_llava = MoondreamVision(backend="ollama", model_id="llava")
            # results_llava = vision_llava.process_frame(dummy_image_ollama)
            # logger.info(f"Ollama (Llava) process_frame results: {results_llava}")
            # answer_llava = vision_llava.ask_question(dummy_image_bytes, question_ollama)
            # logger.info(f"Ollama (Llava) VQA - Q: '{question_ollama}' A: '{answer_llava}'")


        except Exception as e:
            logger.exception(f"Error testing Ollama backend: {e}. Ensure Ollama server is running and model is pulled.")
            print("\nTroubleshooting Ollama:")
            print("1. Is the Ollama server running? Try `ollama serve` in your terminal.")
            print("2. Have you pulled the model? Try `ollama pull moondream` (or the model you specified).")
            print("3. Is the Ollama API accessible (default: http://localhost:11434)? Check network/firewall.")

    else:
        logger.warning("Skipping Ollama tests: ollama client not installed.")

    logger.info("\nMoondream wrapper test finished.")
