# ONNX Models Directory

Place your ONNX model files (.onnx) in this directory. The Duck VLA system will use the first .onnx file it finds in this directory.

## Using ONNX Models

1. Place any ONNX model file (with `.onnx` extension) in this directory
2. The `run_duck_sim.py` script will automatically find and use it
3. Only the first ONNX file found will be used

## Obtaining ONNX Models

You can download pre-trained models from the Open Duck GitHub repository:
https://github.com/open-duck/open-duck-playground/tree/main/playground/open_duck_mini_v2/onnx

Common models include:
- `BEST_WALK_ONNX_2.onnx` - Standard walking model
- `policy_1.onnx` - Alternative policy model

## Debugging

If you have issues with your ONNX model:

1. Verify it has the `.onnx` extension
2. Run with debug logging: `uv run run_duck_sim.py --debug`
3. Check the log output for any errors related to the ONNX model loading 