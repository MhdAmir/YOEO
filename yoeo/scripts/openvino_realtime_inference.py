#!/usr/bin/env python3
"""
Real-time video inference menggunakan OpenVINO IR model YOEO
Menampilkan FPS counter dan hasil deteksi/segmentasi
"""
import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
from openvino.runtime import Core

# Import dari YOEO utils
from yoeo.utils.class_config import ClassConfig
from yoeo.utils.utils import non_max_suppression, rescale_boxes, rescale_segmentation
from yoeo.utils.dataclasses import GroupConfig, ClassNames


class OpenVINOYOEO:
    """Wrapper untuk YOEO model dalam format OpenVINO IR"""
    
    def __init__(self, model_xml: str, device: str = "CPU"):
        """
        Initialize OpenVINO model
        
        Args:
            model_xml: Path ke OpenVINO IR XML file
            device: Device untuk inference (CPU, GPU, atau AUTO)
        """
        print(f"Loading OpenVINO model from: {model_xml}")
        
        # Initialize OpenVINO runtime
        self.core = Core()
        
        # Load model
        self.model = self.core.read_model(model=model_xml)
        self.compiled_model = self.core.compile_model(model=self.model, device_name=device)
        
        # Get input and output information
        self.input_layer = self.compiled_model.input(0)
        self.output_detections = self.compiled_model.output(0)
        self.output_segmentations = self.compiled_model.output(1)
        
        # Get input shape
        self.input_shape = self.input_layer.shape
        self.batch_size = self.input_shape[0]
        self.channels = self.input_shape[1]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        print(f"✓ Model loaded successfully!")
        print(f"  Device: {device}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Input size: {self.input_height}x{self.input_width}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image untuk inference
        
        Args:
            image: Input image (BGR format dari OpenCV)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize dengan padding untuk maintain aspect ratio
        ih, iw = image_rgb.shape[:2]
        h, w = self.input_height, self.input_width
        scale = min(w/iw, h/ih)
        nw, nh = int(iw*scale), int(ih*scale)
        
        image_resized = cv2.resize(image_rgb, (nw, nh))
        
        # Create padded image
        image_padded = np.full((h, w, 3), 128, dtype=np.uint8)
        dw, dh = (w - nw) // 2, (h - nh) // 2
        image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
        
        # Normalize ke [0, 1] dan convert ke CHW format
        image_normalized = image_padded.astype(np.float32) / 255.0
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        
        # Add batch dimension
        image_batch = np.expand_dims(image_chw, axis=0)
        
        return image_batch
    
    def infer(self, image: np.ndarray) -> tuple:
        """
        Run inference pada image
        
        Args:
            image: Preprocessed image tensor
            
        Returns:
            Tuple (detections, segmentations)
        """
        # Run inference
        results = self.compiled_model([image])
        
        detections = results[self.output_detections]
        segmentations = results[self.output_segmentations]
        
        return detections, segmentations


class FPSCounter:
    """Counter untuk menghitung FPS"""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter
        
        Args:
            window_size: Ukuran window untuk averaging FPS
        """
        self.window_size = window_size
        self.timestamps = []
    
    def update(self):
        """Update timestamp"""
        current_time = time.time()
        self.timestamps.append(current_time)
        
        # Keep only recent timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
    
    def get_fps(self) -> float:
        """
        Get current FPS
        
        Returns:
            FPS value
        """
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff == 0:
            return 0.0
        
        fps = (len(self.timestamps) - 1) / time_diff
        return fps


def draw_detections(image: np.ndarray, detections: np.ndarray, 
                    class_names: list, conf_threshold: float = 0.5) -> np.ndarray:
    """
    Draw bounding boxes dan labels pada image
    
    Args:
        image: Input image
        detections: Array deteksi [x1, y1, x2, y2, conf, class_id]
        class_names: List nama kelas
        conf_threshold: Confidence threshold
        
    Returns:
        Image dengan deteksi yang sudah digambar
    """
    if detections is None or len(detections) == 0:
        return image
    
    # Generate random colors untuk setiap class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
    
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        
        if conf < conf_threshold:
            continue
        
        # Convert ke integer
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        
        # Get color
        color = tuple(map(int, colors[cls]))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_names[cls]}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, label_size[1] + 10)
        
        cv2.rectangle(image, (x1, label_y - label_size[1] - 10), 
                     (x1 + label_size[0], label_y), color, -1)
        cv2.putText(image, label, (x1, label_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def draw_segmentation(image: np.ndarray, segmentation: np.ndarray, 
                      class_names: list, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay segmentation mask pada image dengan legend per class
    
    Args:
        image: Input image
        segmentation: Segmentation mask [H, W] or [B, H, W]
        class_names: List nama kelas
        alpha: Transparansi overlay
        
    Returns:
        Image dengan segmentation overlay dan legend per class
    """
    if segmentation is None or segmentation.size == 0:
        return image
    
    # Remove batch dimension if present
    while len(segmentation.shape) > 2:
        segmentation = segmentation[0]
    
    # Ensure we have 2D array
    if len(segmentation.shape) != 2:
        print(f"Warning: Unexpected segmentation shape: {segmentation.shape}")
        return image
    
    # Generate random colors untuk setiap class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
    
    # Create colored segmentation mask
    h, w = segmentation.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Track which classes are present
    unique_classes = np.unique(segmentation)
    class_pixels = {}  # Count pixels per class
    
    for class_id in unique_classes:
        if class_id < len(class_names):  # Valid class
            mask = segmentation == class_id
            pixel_count = np.sum(mask)
            class_pixels[int(class_id)] = pixel_count
            colored_mask[mask] = colors[class_id]
    
    # Resize mask ke ukuran image
    if colored_mask.shape[:2] != image.shape[:2]:
        colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]))
        # Resize segmentation juga untuk drawing borders
        segmentation_resized = cv2.resize(segmentation.astype(np.uint8), 
                                         (image.shape[1], image.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
    else:
        segmentation_resized = segmentation.astype(np.uint8)
    
    # Blend dengan image original
    output = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    # Draw contours untuk setiap class (outline)
    for class_id in class_pixels.keys():
        if class_id == 0:  # Skip background
            continue
        
        # Create binary mask untuk class ini
        class_mask = (segmentation_resized == class_id).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours dengan warna class
        color = tuple(map(int, colors[class_id]))
        cv2.drawContours(output, contours, -1, color, 2)
    
    # Draw legend dengan statistik per class
    legend_x = 10
    legend_y = image.shape[0] - 20  # Start dari bawah
    legend_height = 30
    legend_width = 250
    
    # Sort classes by pixel count (descending)
    sorted_classes = sorted(class_pixels.items(), key=lambda x: x[1], reverse=True)
    
    # Draw semi-transparent background untuk legend
    num_classes = len(sorted_classes)
    if num_classes > 0:
        overlay = output.copy()
        legend_bg_height = (num_classes * legend_height) + 20
        cv2.rectangle(overlay, 
                     (legend_x - 5, legend_y - legend_bg_height - 5),
                     (legend_x + legend_width + 5, legend_y + 15),
                     (0, 0, 0), -1)
        output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
    
    # Draw legend entries
    total_pixels = image.shape[0] * image.shape[1]
    
    for idx, (class_id, pixel_count) in enumerate(sorted_classes):
        if class_id == 0:  # Skip background
            continue
        
        y_pos = legend_y - (idx * legend_height)
        
        # Draw color box
        color = tuple(map(int, colors[class_id]))
        cv2.rectangle(output, 
                     (legend_x, y_pos - 20),
                     (legend_x + 30, y_pos - 5),
                     color, -1)
        cv2.rectangle(output, 
                     (legend_x, y_pos - 20),
                     (legend_x + 30, y_pos - 5),
                     (255, 255, 255), 1)
        
        # Calculate percentage
        percentage = (pixel_count / total_pixels) * 100
        
        # Draw class name dan percentage
        text = f"{class_names[class_id]}: {percentage:.1f}%"
        cv2.putText(output, text,
                   (legend_x + 40, y_pos - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1, cv2.LINE_AA)
    
    return output


def create_segmentation_grid(segmentation: np.ndarray, class_names: list, 
                            original_shape: tuple) -> np.ndarray:
    """
    Create grid view dengan mask per class secara terpisah
    
    Args:
        segmentation: Segmentation mask [H, W]
        class_names: List nama kelas
        original_shape: Shape dari original image (H, W)
        
    Returns:
        Grid image dengan mask per class
    """
    if segmentation is None or segmentation.size == 0:
        return None
    
    # Remove batch dimension if present
    while len(segmentation.shape) > 2:
        segmentation = segmentation[0]
    
    # Resize segmentation ke ukuran original
    if segmentation.shape[:2] != original_shape[:2]:
        segmentation = cv2.resize(segmentation.astype(np.uint8), 
                                 (original_shape[1], original_shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
    
    # Get unique classes
    unique_classes = [c for c in np.unique(segmentation) if c > 0]  # Skip background
    
    if len(unique_classes) == 0:
        return None
    
    # Calculate grid dimensions
    num_classes = len(unique_classes)
    cols = min(3, num_classes)  # Max 3 columns
    rows = (num_classes + cols - 1) // cols
    
    # Create individual class masks
    h, w = segmentation.shape
    cell_height = min(h // 2, 300)  # Limit cell size
    cell_width = min(w // 2, 400)
    
    # Create grid canvas
    grid_height = rows * cell_height + (rows + 1) * 10  # Add spacing
    grid_width = cols * cell_width + (cols + 1) * 10
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Generate colors
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
    
    # Fill grid dengan mask per class
    for idx, class_id in enumerate(unique_classes):
        row = idx // cols
        col = idx % cols
        
        # Calculate position
        y_start = row * (cell_height + 10) + 10
        x_start = col * (cell_width + 10) + 10
        
        # Create binary mask untuk class ini
        class_mask = (segmentation == class_id).astype(np.uint8)
        
        # Resize ke cell size
        class_mask_resized = cv2.resize(class_mask, (cell_width, cell_height))
        
        # Create colored version
        colored_mask = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        color = colors[class_id]
        colored_mask[class_mask_resized > 0] = color
        
        # Place in grid
        grid[y_start:y_start+cell_height, x_start:x_start+cell_width] = colored_mask
        
        # Draw border
        cv2.rectangle(grid,
                     (x_start, y_start),
                     (x_start + cell_width, y_start + cell_height),
                     (255, 255, 255), 2)
        
        # Draw class label
        label = class_names[class_id]
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_x = x_start + (cell_width - label_size[0]) // 2
        label_y = y_start - 5
        
        # Draw label background
        cv2.rectangle(grid,
                     (label_x - 5, label_y - label_size[1] - 5),
                     (label_x + label_size[0] + 5, label_y + 5),
                     tuple(map(int, color)), -1)
        
        # Draw label text
        cv2.putText(grid, label,
                   (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 2, cv2.LINE_AA)
        
        # Calculate and draw pixel count
        pixel_count = np.sum(class_mask)
        percentage = (pixel_count / (h * w)) * 100
        stats_text = f"{percentage:.1f}%"
        
        stats_y = y_start + cell_height + 20
        cv2.putText(grid, stats_text,
                   (x_start + 5, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1, cv2.LINE_AA)
    
    return grid


def process_video(model: OpenVINOYOEO, video_source, class_config: ClassConfig,
                 conf_thres: float = 0.5, nms_thres: float = 0.5,
                 show_seg: bool = True, save_output: str = None, show_grid: bool = False):
    """
    Process video stream dengan model OpenVINO
    
    Args:
        model: OpenVINO YOEO model
        video_source: Video source (0 untuk webcam, atau path ke video file)
        class_config: Class configuration
        conf_thres: Confidence threshold
        nms_thres: NMS threshold
        show_seg: Show segmentation overlay
        save_output: Path untuk save output video (optional)
        show_grid: Show grid view dengan mask per class
    """
    # Open video capture
    if isinstance(video_source, int) or video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))
        print(f"Opening webcam {video_source}...")
    else:
        cap = cv2.VideoCapture(video_source)
        print(f"Opening video file: {video_source}")
    
    if not cap.isOpened():
        print(f"Error: Cannot open video source {video_source}")
        sys.exit(1)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video resolution: {frame_width}x{frame_height}")
    print(f"Video FPS: {fps_video}")
    
    # Setup video writer jika save_output diset
    video_writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_output, fourcc, fps_video, 
                                      (frame_width, frame_height))
        print(f"Saving output to: {save_output}")
    
    # Initialize FPS counter
    fps_counter = FPSCounter()
    
    # Get class names
    class_names = class_config.get_ungrouped_det_class_names()
    group_config = class_config.get_group_config()
    
    print("\nStarting inference...")
    print("Press 'q' to quit, 's' to toggle segmentation, 'g' to toggle grid view")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame")
                break
            
            # Preprocess image
            input_tensor = model.preprocess_image(frame)
            
            # Run inference
            t_start = time.time()
            detections, segmentations = model.infer(input_tensor)
            inference_time = (time.time() - t_start) * 1000  # ms
            
            # Post-process detections
            import torch
            detections_tensor = torch.from_numpy(detections)
            detections_nms = non_max_suppression(
                prediction=detections_tensor,
                conf_thres=conf_thres,
                iou_thres=nms_thres,
                group_config=group_config
            )
            
            # Rescale boxes ke ukuran original image
            if len(detections_nms[0]) > 0:
                detections_rescaled = rescale_boxes(
                    detections_nms[0], 
                    model.input_height, 
                    (frame_height, frame_width)
                )
                detections_np = detections_rescaled.numpy()
            else:
                detections_np = np.array([])
            
            # Rescale segmentation
            segmentation_rescaled = rescale_segmentation(
                torch.from_numpy(segmentations),
                (frame_height, frame_width)
            )
            # Remove batch dimension and convert to numpy
            segmentation_np = segmentation_rescaled.squeeze(0).cpu().numpy()
            
            # Draw segmentation jika enabled
            if show_seg and segmentation_np is not None:
                frame = draw_segmentation(frame, segmentation_np, class_names)
            
            # Draw detections
            frame = draw_detections(frame, detections_np, class_names, conf_thres)
            
            # Update FPS counter
            fps_counter.update()
            current_fps = fps_counter.get_fps()
            
            # Draw FPS dan inference time
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw detection count
            det_count = len(detections_np) if len(detections_np) > 0 else 0
            cv2.putText(frame, f"Detections: {det_count}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show main frame
            cv2.imshow('YOEO OpenVINO Real-time Inference', frame)
            
            # Show grid view jika enabled
            if show_grid and segmentation_np is not None:
                grid_view = create_segmentation_grid(
                    segmentation_np, 
                    class_names,
                    (frame_height, frame_width)
                )
                if grid_view is not None:
                    cv2.imshow('Segmentation Per Class', grid_view)
            
            # Save frame jika video writer ada
            if video_writer:
                video_writer.write(frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                show_seg = not show_seg
                print(f"Segmentation overlay: {'ON' if show_seg else 'OFF'}")
            elif key == ord('g'):
                show_grid = not show_grid
                if not show_grid:
                    cv2.destroyWindow('Segmentation Per Class')
                print(f"Grid view: {'ON' if show_grid else 'OFF'}")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nAverage FPS: {fps_counter.get_fps():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time video inference dengan YOEO OpenVINO model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Webcam inference
  python yoeo/scripts/openvino_realtime_inference.py \\
      --model weights/yoeo.xml \\
      --source 0

  # Video file inference dengan grid view
  python yoeo/scripts/openvino_realtime_inference.py \\
      --model weights/yoeo.xml \\
      --source video.mp4 \\
      --save output.mp4 \\
      --show-grid

  # GPU inference
  python yoeo/scripts/openvino_realtime_inference.py \\
      --model weights/yoeo.xml \\
      --source 0 \\
      --device GPU

Controls:
  q - Quit
  s - Toggle segmentation overlay
  g - Toggle grid view (per-class masks)
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to OpenVINO IR model (.xml file)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source (0 for webcam, or path to video file)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU', 'AUTO'],
        help='OpenVINO device to run inference on'
    )
    
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.5,
        help='Confidence threshold for detections'
    )
    
    parser.add_argument(
        '--nms-thres',
        type=float,
        default=0.4,
        help='NMS threshold for detections'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save output video (optional)'
    )
    
    parser.add_argument(
        '--show-grid',
        action='store_true',
        help='Show per-class segmentation grid view on startup'
    )
    
    parser.add_argument(
        '--classes',
        type=str,
        default='data/yoeo_names.yaml',
        help='Path to class names file (default: data/yoeo_names.yaml)'
    )
    
    parser.add_argument(
        '--class-config',
        type=str,
        default='class_config/default.yaml',
        help='Path to class config file (default: class_config/default.yaml)'
    )
    
    parser.add_argument(
        '--no-seg',
        action='store_true',
        help='Disable segmentation overlay'
    )
    
    args = parser.parse_args()
    
    # Validate model file
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Load class configuration
    print(f"Loading class configuration from: {args.classes}")
    class_names = ClassNames.load_from(args.classes)
    class_config = ClassConfig.load_from(args.class_config, class_names)
    
    # Initialize OpenVINO model
    model = OpenVINOYOEO(args.model, device=args.device)
    
    # Process video
    process_video(
        model=model,
        video_source=args.source,
        class_config=class_config,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        show_seg=not args.no_seg,
        save_output=args.save,
        show_grid=args.show_grid
    )


if __name__ == "__main__":
    main()
