# YOEO OpenVINO Real-time Inference

Tutorial lengkap untuk convert model YOEO dari PyTorch ke OpenVINO dan menjalankan inference real-time dengan FPS counter.

## Prerequisites

Install dependencies yang diperlukan:

```bash
# Activate virtualenv
source botenv/bin/activate

# Install OpenVINO development tools
pip install openvino-dev openvino

# Install onnx untuk validasi
pip install onnx

# Install OpenCV untuk video processing
pip install opencv-python
```

## Step 1: Convert Model PyTorch ke OpenVINO

Convert model `yoeo.pth` ke format OpenVINO IR:

```bash
python yoeo/scripts/convert_yoeo_to_openvino.py \
    --config config/yoeo.cfg \
    --weights weights/yoeo.pth \
    --output weights/
```

Script ini akan:
1. Load model PyTorch dari `weights/yoeo.pth`
2. Convert ke ONNX format (`weights/yoeo.onnx`)
3. Convert ONNX ke OpenVINO IR (`weights/yoeo.xml` dan `weights/yoeo.bin`)

**Output files:**
- `weights/yoeo.onnx` - Model dalam format ONNX
- `weights/yoeo.xml` - OpenVINO IR model definition
- `weights/yoeo.bin` - OpenVINO IR model weights

## Step 2: Real-time Inference

### Webcam Inference

Jalankan inference menggunakan webcam dengan FPS display:

```bash
python yoeo/scripts/openvino_realtime_inference.py \
    --model weights/yoeo.xml \
    --source 0
```

### Video File Inference

Inference pada video file:

```bash
python yoeo/scripts/openvino_realtime_inference.py \
    --model weights/yoeo.xml \
    --source path/to/video.mp4
```

### Inference dengan GPU

Gunakan GPU untuk inference lebih cepat (jika tersedia):

```bash
python yoeo/scripts/openvino_realtime_inference.py \
    --model weights/yoeo.xml \
    --source 0 \
    --device GPU
```

### Save Output Video

Save hasil inference ke video file:

```bash
python yoeo/scripts/openvino_realtime_inference.py \
    --model weights/yoeo.xml \
    --source video.mp4 \
    --save output.mp4
```

## Command Options

### Conversion Script (`convert_yoeo_to_openvino.py`)

```
--config     Path ke model configuration file (.cfg)
--weights    Path ke PyTorch weights file (.pth)
--output     Output directory untuk model yang diconvert (default: weights/)
--img-size   Input image size (default: 416)
```

### Inference Script (`openvino_realtime_inference.py`)

```
--model       Path ke OpenVINO IR model (.xml file) [REQUIRED]
--source      Video source (0 untuk webcam, atau path ke video file) [default: 0]
--device      Device untuk inference: CPU, GPU, atau AUTO [default: CPU]
--classes     Path ke class names file [default: data/yoeo_names.yaml]
--conf-thres  Confidence threshold [default: 0.5]
--nms-thres   NMS threshold [default: 0.5]
--no-seg      Disable segmentation overlay
--save        Save output video ke file (optional)
```

## Keyboard Controls

Saat menjalankan inference:
- **`q`** - Quit/keluar dari aplikasi
- **`s`** - Toggle segmentation overlay on/off

## Display Information

Real-time display menampilkan:
- **FPS** - Frames per second (rata-rata)
- **Inference time** - Waktu inference per frame dalam milliseconds
- **Detections** - Jumlah objek yang terdeteksi
- **Bounding boxes** - Kotak pembatas objek dengan label dan confidence
- **Segmentation mask** - Overlay segmentasi (jika enabled)

## Performance Tips

1. **Use GPU**: Jika punya GPU Intel/NVIDIA yang support OpenVINO, gunakan `--device GPU`
2. **Lower resolution**: Gunakan `--img-size 320` saat convert untuk inference lebih cepat (trade-off: akurasi menurun)
3. **Adjust thresholds**: Naikkan `--conf-thres` untuk mengurangi false positives
4. **Disable segmentation**: Gunakan `--no-seg` untuk fokus pada deteksi saja (lebih cepat)

## Troubleshooting

### Error: "mo: command not found"

Install OpenVINO Model Optimizer:
```bash
pip install openvino-dev
```

### Error: "Cannot open video source"

- Untuk webcam: pastikan webcam terdeteksi (`ls /dev/video*`)
- Untuk video file: pastikan path file benar dan format supported

### Low FPS

- Gunakan `--device GPU` jika tersedia
- Turunkan resolution saat convert (`--img-size 320`)
- Gunakan `--no-seg` untuk disable segmentation

### Model conversion failed

- Pastikan PyTorch model ter-load dengan benar
- Check CUDA/GPU availability jika menggunakan GPU
- Pastikan semua dependencies ter-install

## Example Complete Workflow

```bash
# 1. Activate environment
source botenv/bin/activate

# 2. Convert model
python yoeo/scripts/convert_yoeo_to_openvino.py \
    --config config/yoeo.cfg \
    --weights weights/yoeo.pth \
    --output weights/

# 3. Run webcam inference
python yoeo/scripts/openvino_realtime_inference.py \
    --model weights/yoeo.xml \
    --source 0

# 4. Run video file inference dengan save output
python yoeo/scripts/openvino_realtime_inference.py \
    --model weights/yoeo.xml \
    --source input_video.mp4 \
    --save output_video.mp4 \
    --conf-thres 0.6
```

## Architecture

```
PyTorch Model (.pth)
    ↓ (convert_yoeo_to_openvino.py)
ONNX Model (.onnx)
    ↓ (Model Optimizer)
OpenVINO IR (.xml + .bin)
    ↓ (openvino_realtime_inference.py)
Real-time Inference + FPS Display
```

## Notes

- Default input size: 416x416 pixels
- Supported devices: CPU, GPU (Intel), AUTO
- Output format: RGB dengan bounding boxes dan segmentation overlay
- FPS counter menggunakan sliding window averaging (30 frames)
