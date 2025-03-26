#!/usr/bin/env python3
"""
Create a simple dummy ONNX model for testing.
This model takes an input of shape (1, 3, 224, 224) and outputs two values.
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.helper import make_tensor_value_info

def create_dummy_onnx_model(output_path="model.onnx"):
    """Create a dummy ONNX model for testing."""
    
    # Define the network inputs and outputs
    X = make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
    Y = make_tensor_value_info('output', TensorProto.FLOAT, [2])
    
    # Define a node that computes the sum of the input
    node_add = helper.make_node(
        'ReduceSum',
        inputs=['input'],
        outputs=['sum'],
        keepdims=0,
        name='node_1'
    )
    
    # Define a node that computes constant values
    node_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['constant'],
        value=helper.make_tensor(
            name='const_tensor',
            data_type=TensorProto.FLOAT,
            dims=[2],
            vals=[0.5, 0.1]  # Forward speed and angular speed
        ),
        name='node_2'
    )
    
    # Define a node to multiply the sum by constants
    node_mul = helper.make_node(
        'Mul',
        inputs=['sum', 'constant'],
        outputs=['output'],
        name='node_3'
    )
    
    # Create the graph
    graph = helper.make_graph(
        [node_add, node_constant, node_mul],
        'test-model',
        [X],
        [Y]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='dummy-model')
    model.opset_import[0].version = 12
    
    # Check the model
    onnx.checker.check_model(model)
    
    # Save the model
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else '.', exist_ok=True)
    onnx.save(model, output_path)
    
    print(f"Dummy ONNX model created at {output_path}")
    return output_path

if __name__ == "__main__":
    create_dummy_onnx_model() 