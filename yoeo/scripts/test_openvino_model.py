#!/usr/bin/env python3
"""
Test script untuk memverifikasi OpenVINO model dapat di-load dan di-run
"""
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openvino.runtime import Core

print("=" * 60)
print("Testing OpenVINO YOEO Model")
print("=" * 60)

# Load model
model_path = "weights/yoeo.xml"
print(f"\n1. Loading model from: {model_path}")

core = Core()
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

print("   ✓ Model loaded successfully!")

# Get input/output info
input_layer = compiled_model.input(0)
output_det = compiled_model.output(0)
output_seg = compiled_model.output(1)

print(f"\n2. Model Information:")
print(f"   Input shape: {input_layer.shape}")
print(f"   Input name: {input_layer.any_name}")
print(f"   Detection output shape: {output_det.shape}")
print(f"   Segmentation output shape: {output_seg.shape}")

# Test inference dengan dummy input
print(f"\n3. Testing inference with dummy input...")
batch_size, channels, height, width = input_layer.shape
dummy_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)

print(f"   Running inference on {height}x{width} image...")
results = compiled_model([dummy_input])

detections = results[output_det]
segmentations = results[output_seg]

print(f"   ✓ Inference successful!")
print(f"   Detection output shape: {detections.shape}")
print(f"   Segmentation output shape: {segmentations.shape}")

print(f"\n{'=' * 60}")
print("✓ All tests passed!")
print(f"{'=' * 60}")
print("\nModel is ready for real-time inference!")
print("\nTo run webcam inference:")
print("  python yoeo/scripts/openvino_realtime_inference.py \\")
print("      --model weights/yoeo.xml \\")
print("      --source 0")
print("\nOr use the quick start script:")
print("  ./run_openvino_inference.sh")
