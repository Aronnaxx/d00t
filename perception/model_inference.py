# perception/model_inference.py
import onnxruntime as ort
import cv2
import numpy as np
import os

def load_onnx_model(model_path="model.onnx"):
    """Load an ONNX model for inference.
    
    Args:
        model_path (str): Path to the ONNX model file
        
    Returns:
        tuple: (onnx_session, input_name, input_shape) or (None, None, None) if failed
    """
    try:
        # Verify model file exists
        if not os.path.exists(model_path):
            print(f"Error: ONNX model file not found at {model_path}")
            return None, None, None
            
        # Create ONNX Runtime inference session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, session_options)
        
        # Get model input details
        model_inputs = session.get_inputs()
        input_name = model_inputs[0].name
        input_shape = model_inputs[0].shape
        
        # Get model output details for logging
        model_outputs = session.get_outputs()
        output_name = model_outputs[0].name
        output_shape = model_outputs[0].shape
        
        print(f"Loaded ONNX model from {model_path}")
        print(f"  - Input: '{input_name}' with shape {input_shape}")
        print(f"  - Output: '{output_name}' with shape {output_shape}")
        
        return session, input_name, input_shape
    except Exception as e:
        print(f"Error loading ONNX model: {str(e)}")
        return None, None, None

def preprocess_image(image, target_size=(224, 224), normalize=True, transpose=True):
    """Preprocess an image for model inference.
    
    Args:
        image (np.ndarray): RGB image as numpy array
        target_size (tuple): Target size as (width, height)
        normalize (bool): Whether to normalize pixel values to [0,1]
        transpose (bool): Whether to transpose from HWC to CHW format
        
    Returns:
        np.ndarray: Preprocessed image ready for the model
                   or None if preprocessing failed
    """
    if image is None:
        print("Error: Image is None")
        return None
        
    try:
        # Resize image to target size
        resized = cv2.resize(image, target_size)
        
        # Normalize to [0,1] if requested
        if normalize:
            processed = resized.astype('float32') / 255.0
        else:
            processed = resized.astype('float32')
            
        # Rearrange from HWC to CHW (Height, Width, Channels) -> (Channels, Height, Width)
        if transpose:
            processed = np.transpose(processed, (2, 0, 1))
            
        # Add batch dimension -> (1, C, H, W) or (1, H, W, C) depending on transpose
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def run_inference(session, input_name, processed_image):
    """Run inference with the ONNX model.
    
    Args:
        session: ONNX Runtime session
        input_name (str): Name of the input tensor
        processed_image (np.ndarray): Preprocessed image tensor
        
    Returns:
        tuple: (v_lin, v_ang) velocities or (0.0, 0.0) if inference failed
    """
    if session is None or processed_image is None:
        print("Error: Invalid session or input")
        return 0.0, 0.0
        
    try:
        # Run inference
        outputs = session.run(None, {input_name: processed_image})
        
        # Extract velocities from model output
        # Note: Adjust this according to your model's specific output format
        # This assumes outputs[0] is a tensor with shape [1, 2] representing [v_lin, v_ang]
        if outputs[0].shape[1] >= 2:
            v_lin, v_ang = outputs[0][0][0], outputs[0][0][1]
            return float(v_lin), float(v_ang)
        else:
            print(f"Warning: Unexpected output shape: {outputs[0].shape}")
            return 0.0, 0.0
    except Exception as e:
        print(f"Error running inference: {str(e)}")
        return 0.0, 0.0

def create_dummy_onnx_model(output_path="model.onnx"):
    """Create a dummy ONNX model for testing purposes.
    This function creates a simple model that takes a tensor of shape [1,3,224,224]
    and outputs a tensor of shape [1,2] with constant values.
    
    Args:
        output_path (str): Path where to save the model
        
    Returns:
        bool: True if model was created successfully, False otherwise
    """
    try:
        import onnx
        from onnx import helper
        from onnx import TensorProto
        
        # Define model inputs and outputs
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])
        
        # Create a simple model with a ReduceMean operation
        # This will average the input and produce a tensor with linear and angular velocities
        node_def = helper.make_node(
            'ReduceMean',
            inputs=['input'],
            outputs=['reduced'],
            axes=[2, 3],  # Average across height and width dimensions
            keepdims=0
        )
        
        # Reshape to ensure [1, 2] output shape
        reshape_node = helper.make_node(
            'Reshape',
            inputs=['reduced'],
            outputs=['output'],
            shape=[1, 2]
        )
        
        # Create the graph and model
        graph_def = helper.make_graph(
            [node_def, reshape_node],
            'dummy-model',
            [X],
            [Y]
        )
        
        model_def = helper.make_model(graph_def, producer_name='isaac-lab-demo')
        
        # Save the model
        onnx.save(model_def, output_path)
        print(f"Created dummy ONNX model at {output_path}")
        return True
    except Exception as e:
        print(f"Error creating dummy ONNX model: {str(e)}")
        return False

if __name__ == "__main__":
    # For testing this module independently
    
    # Create a dummy model if needed
    if not os.path.exists("model.onnx"):
        create_dummy_onnx_model()
    
    # Load the model
    session, input_name, input_shape = load_onnx_model()
    
    if session is not None:
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Preprocess the image
        processed = preprocess_image(dummy_image)
        
        # Run inference
        v_lin, v_ang = run_inference(session, input_name, processed)
        
        print(f"Inference output: v_lin={v_lin}, v_ang={v_ang}")
    else:
        print("Test failed: Could not load model")
