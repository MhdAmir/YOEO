#!/usr/bin/env python3
"""
Script untuk convert model YOEO dari PyTorch (.pth) ke OpenVINO IR format
Langkah: PyTorch (.pth) -> ONNX (.onnx) -> OpenVINO IR (.xml, .bin)
"""
import argparse
import os
import sys

import onnx
import torch

# Import dari YOEO
import yoeo.models


def convert_pytorch_to_onnx(model_cfg: str, weights_pth: str, output_path: str, 
                           image_size: int = 416, batch_size: int = 1) -> str:
    """
    Convert PyTorch model ke ONNX format
    
    Args:
        model_cfg: Path ke file konfigurasi model (.cfg)
        weights_pth: Path ke file weights PyTorch (.pth)
        output_path: Path untuk menyimpan file ONNX
        image_size: Ukuran input image (default: 416)
        batch_size: Batch size untuk inference (default: 1)
        
    Returns:
        Path ke file ONNX yang telah dibuat
    """
    print("=" * 60)
    print("Step 1: Converting PyTorch model to ONNX...")
    print("=" * 60)
    
    # Load PyTorch model
    print(f"Loading PyTorch model from: {weights_pth}")
    pytorch_model = yoeo.models.load_model(model_cfg, weights_pth)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    pytorch_model.to(device)
    pytorch_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_path,
        verbose=False,
        export_params=True,
        input_names=["InputLayer"],
        output_names=["Detections", "Segmentations"],
        opset_version=11,
        do_constant_folding=True
    )
    
    # Validate ONNX model
    print("Validating ONNX model...")
    onnx_model = onnx.load(output_path)
    try:
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid!")
    except onnx.checker.ValidationError as e:
        print(f"✗ ONNX model is invalid: {e}")
        sys.exit(1)
    
    print(f"✓ ONNX model saved to: {output_path}")
    return output_path


def convert_onnx_to_openvino(onnx_path: str, output_dir: str) -> tuple:
    """
    Convert ONNX model ke OpenVINO IR format
    
    Args:
        onnx_path: Path ke file ONNX
        output_dir: Directory untuk menyimpan file IR (.xml dan .bin)
        
    Returns:
        Tuple (xml_path, bin_path)
    """
    print("\n" + "=" * 60)
    print("Step 2: Converting ONNX model to OpenVINO IR...")
    print("=" * 60)
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct Model Optimizer command
    mo_command = f"""mo
        --input_model "{onnx_path}"
        --output_dir "{output_dir}"
        --input InputLayer
        --output Detections,Segmentations
        --framework onnx
        --static_shape
        --batch 1
    """
    mo_command = " ".join(mo_command.split())
    
    print(f"Running Model Optimizer command:")
    print(mo_command)
    print()
    
    # Run Model Optimizer
    result = os.system(mo_command)
    
    print("=" * 60)
    if result == 0:
        print("✓ Model conversion to OpenVINO IR was successful!")
        
        # Get output file paths
        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        xml_path = os.path.join(output_dir, f"{model_name}.xml")
        bin_path = os.path.join(output_dir, f"{model_name}.bin")
        
        print(f"  XML file: {xml_path}")
        print(f"  BIN file: {bin_path}")
        
        return xml_path, bin_path
    else:
        print("✗ Model conversion to OpenVINO IR failed!")
        print("\nNote: Make sure OpenVINO Model Optimizer (mo) is installed.")
        print("Install with: pip install openvino-dev")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOEO PyTorch model to OpenVINO IR format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python convert_yoeo_to_openvino.py \\
      --config config/yoeo.cfg \\
      --weights weights/yoeo.pth \\
      --output weights/

This will create:
  - weights/yoeo.onnx
  - weights/yoeo.xml
  - weights/yoeo.bin
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model configuration file (.cfg)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to PyTorch weights file (.pth)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='weights/',
        help='Output directory for converted models (default: weights/)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=416,
        help='Input image size (default: 416)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)
    
    # Create ONNX output path
    weights_basename = os.path.splitext(os.path.basename(args.weights))[0]
    onnx_path = os.path.join(args.output, f"{weights_basename}.onnx")
    
    # Step 1: PyTorch -> ONNX
    onnx_path = convert_pytorch_to_onnx(
        model_cfg=args.config,
        weights_pth=args.weights,
        output_path=onnx_path,
        image_size=args.img_size
    )
    
    # Step 2: ONNX -> OpenVINO IR
    xml_path, bin_path = convert_onnx_to_openvino(
        onnx_path=onnx_path,
        output_dir=args.output
    )
    
    print("\n" + "=" * 60)
    print("✓ Conversion completed successfully!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  1. ONNX model: {onnx_path}")
    print(f"  2. OpenVINO XML: {xml_path}")
    print(f"  3. OpenVINO BIN: {bin_path}")
    print(f"\nYou can now use the OpenVINO IR model for inference!")
    print(f"Run: python yoeo/scripts/openvino_realtime_inference.py --model {xml_path}")


if __name__ == "__main__":
    main()
