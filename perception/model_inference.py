# perception/model_inference.py
import onnxruntime as ort
import cv2
import numpy as np

def load_onnx_model(model_path="model.onnx"):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    print(f"Loaded ONNX model from {model_path}.")
    return session, input_name

def preprocess_image(image, target_size=(224, 224)):
    # Resize image to target size
    resized = cv2.resize(image, target_size)
    # Normalize image to [0,1] and convert to float32
    norm = resized.astype('float32') / 255.0
    # Rearrange to (C, H, W) and add batch dimension -> (1, C, H, W)
    processed = np.transpose(norm, (2, 0, 1))[None, ...]
    return processed

def run_inference(session, input_name, processed_image):
    outputs = session.run(None, {input_name: processed_image})
    # For example, assume outputs are [linear_velocity, angular_velocity]
    v_lin, v_ang = outputs[0][0]  # Adjust based on model output format
    return v_lin, v_ang

if __name__ == "__main__":
    session, input_name = load_onnx_model()
    # Dummy image for testing
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed = preprocess_image(dummy_image)
    v_lin, v_ang = run_inference(session, input_name, processed)
    print(f"Inference output: v_lin={v_lin}, v_ang={v_ang}")
