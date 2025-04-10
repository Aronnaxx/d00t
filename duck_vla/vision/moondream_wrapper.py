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
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class MoondreamVision:
    """
    Wrapper for Moondream vision-language model for image understanding.
    
    This class provides image captioning and visual question answering
    capabilities using the Moondream model.
    """
    
    def __init__(
        self,
        model_id: str = "huggingface/Moondream1",
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_half_precision: bool = True,
    ):
        """
        Initialize the Moondream vision system.
        
        Args:
            model_id: HuggingFace model ID or local path
            model_dir: Custom directory to load/save model
            device: Device to run model on ('cpu', 'cuda', or None for auto)
            use_half_precision: Whether to use FP16 for faster inference
        """
        self.model_id = model_id
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), "../models/moondream")
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.use_half_precision = use_half_precision and self.device == "cuda"
        
        logger.info(f"Initializing Moondream vision system (device={self.device}, fp16={self.use_half_precision})")
        
        self.processor = None
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize the Moondream model and processor."""
        try:
            # Create model directory if needed
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Loading Moondream processor from {self.model_id}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.model_dir
            )
            
            logger.debug(f"Loading Moondream model from {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.model_dir,
                torch_dtype=torch.float16 if self.use_half_precision else torch.float32
            )
            
            # Move model to device
            self.model.to(self.device)
            
            logger.info(f"Moondream model loaded successfully to {self.device}")
            
        except Exception as e:
            logger.exception(f"Failed to initialize Moondream model: {e}")
            raise RuntimeError(f"Failed to initialize Moondream: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a camera frame and extract visual information.
        
        Args:
            frame: Image as numpy array (HxWxC)
            
        Returns:
            Dict with visual processing results
        """
        if self.model is None or self.processor is None:
            logger.error("Model not initialized")
            return {"error": "Model not initialized"}
            
        try:
            logger.debug("Processing frame with Moondream")
            
            # Get basic caption
            caption = self._generate_caption(frame)
            
            # Check if we see a person
            contains_person = self._check_for_person(frame, caption)
            
            # Get basic scene understanding
            scene_info = self._analyze_scene(frame)
            
            results = {
                "caption": caption,
                "person_detected": contains_person,
                "scene": scene_info
            }
            
            logger.debug(f"Frame processing complete: {results}")
            return results
            
        except Exception as e:
            logger.exception(f"Error processing frame: {e}")
            return {"error": str(e)}
    
    def _generate_caption(self, image: np.ndarray) -> str:
        """Generate a caption for the image."""
        try:
            logger.debug("Generating image caption")
            
            # Process image with Moondream processor
            inputs = self.processor(
                text="<image>\nDescribe this image in detail.", 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            # Decode the output
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            # Remove the prompt
            caption = caption.split("Describe this image in detail.")[-1].strip()
            
            logger.debug(f"Caption generated: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Error generating caption"
    
    def _check_for_person(self, image: np.ndarray, caption: Optional[str] = None) -> bool:
        """
        Check if a person is present in the image.
        
        Uses either caption keywords or directly asks VLM.
        """
        # Quick check in caption if available
        if caption:
            person_keywords = ["person", "people", "man", "woman", "child", "boy", "girl", "human"]
            if any(keyword in caption.lower() for keyword in person_keywords):
                logger.debug("Person detected in caption")
                return True
        
        try:
            # Directly ask the model
            logger.debug("Checking for person with VQA")
            
            inputs = self.processor(
                text="<image>\nIs there a person in this image? Answer yes or no.", 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            
            # Decode the output
            answer = self.processor.decode(output[0], skip_special_tokens=True)
            # Remove the prompt
            answer = answer.split("Is there a person in this image? Answer yes or no.")[-1].strip().lower()
            
            contains_person = "yes" in answer
            logger.debug(f"Person detection VQA result: {answer} -> {contains_person}")
            
            return contains_person
            
        except Exception as e:
            logger.error(f"Error checking for person: {e}")
            return False
    
    def _analyze_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the scene in more detail with specific questions.
        
        Returns a dict with scene information.
        """
        scene_info = {}
        
        try:
            # Ask about scene properties
            questions = [
                "What is the main object or focus in this image?",
                "Is this indoors or outdoors?",
                "What colors are prominent in this image?",
            ]
            
            for question in questions:
                logger.debug(f"Asking VQA question: {question}")
                inputs = self.processor(
                    text=f"<image>\n{question}", 
                    images=image, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate answer
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False
                    )
                
                # Decode the output
                answer = self.processor.decode(output[0], skip_special_tokens=True)
                # Remove the prompt
                answer = answer.split(question)[-1].strip()
                
                # Store result with simplified key
                key = question.replace("?", "").replace("What is", "").replace("What", "").replace("Is this", "").strip()
                key = key.split(" ")[0].lower()
                scene_info[key] = answer
                
            logger.debug(f"Scene analysis complete: {scene_info}")
            
        except Exception as e:
            logger.error(f"Error analyzing scene: {e}")
            scene_info["error"] = str(e)
            
        return scene_info
    
    def ask_question(self, image: np.ndarray, question: str) -> str:
        """
        Ask a custom question about the image.
        
        Args:
            image: Image as numpy array
            question: Question text
            
        Returns:
            Answer text
        """
        if not question.strip():
            return "No question provided"
            
        try:
            logger.debug(f"Processing VQA: {question}")
            
            # Process input
            inputs = self.processor(
                text=f"<image>\n{question}", 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            # Decode the output
            answer = self.processor.decode(output[0], skip_special_tokens=True)
            # Remove the question
            answer = answer.split(question)[-1].strip()
            
            logger.debug(f"VQA answer: {answer}")
            return answer
            
        except Exception as e:
            logger.exception(f"Error in visual QA: {e}")
            return f"Error processing question: {str(e)}"
