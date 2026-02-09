# YOEO OpenVINO - Quick Start Guide

## ✓ Conversion Berhasil!

Model YOEO telah berhasil dikonversi dari PyTorch ke OpenVINO IR format.

### File yang Dihasilkan:

```
weights/
├── yoeo.pth     (27 MB) - Model PyTorch original
├── yoeo.onnx    (27 MB) - Model dalam format ONNX
├── yoeo.xml     (165 KB) - OpenVINO IR model definition
└── yoeo.bin     (14 MB) - OpenVINO IR model weights (compressed to FP16)
```

**Note:** OpenVINO model (~14 MB) lebih kecil dari PyTorch model (~27 MB) karena menggunakan FP16 precision.

## 🚀 Cara Menjalankan Inference

### Opsi 1: Quick Start Script (Recommended)

Cara termudah untuk menjalankan inference:

```bash
./run_openvino_inference.sh
```

Script ini akan:
1. Check dependencies
2. Load OpenVINO model
3. Tanya pilihan: webcam, video file, atau webcam dengan recording
4. Jalankan inference real-time dengan FPS display

### Opsi 2: Manual Command

#### Webcam Inference

```bash
python yoeo/scripts/openvino_realtime_inference.py \
    --model weights/yoeo.xml \
    --source 0
```

#### Video File Inference

```bash
python yoeo/scripts/openvino_realtime_inference.py \
    --model weights/yoeo.xml \
    --source path/to/video.mp4
```

#### Webcam dengan Save Output

```bash
python yoeo/scripts/openvino_realtime_inference.py \
    --model weights/yoeo.xml \
    --source 0 \
    --save output.mp4
```

#### GPU Inference (Jika Tersedia)

```bash
python yoeo/scripts/openvino_realtime_inference.py \
    --model weights/yoeo.xml \
    --source 0 \
    --device GPU
```

## ⌨️ Keyboard Controls

Saat menjalankan inference:
- **`q`** - Quit/keluar
- **`s`** - Toggle segmentation overlay on/off

## 📊 Display Information

Real-time display menampilkan:
- **FPS** - Frames per second (rata-rata 30 frame terakhir)
- **Inference time** - Waktu inference per frame (milliseconds)
- **Detections** - Jumlah objek yang terdeteksi
- **Bounding boxes** - Kotak dengan label dan confidence score
- **Segmentation mask** - Overlay warna untuk setiap class (toggle dengan 's')

## 🎛️ Parameter Options

```bash
--model         Path ke OpenVINO IR model (.xml) [REQUIRED]
--source        Video source (0 = webcam, atau path ke file)
--device        CPU, GPU, atau AUTO (default: CPU)
--conf-thres    Confidence threshold (default: 0.5)
--nms-thres     NMS threshold (default: 0.5)
--no-seg        Disable segmentation overlay
--save          Save output video ke file
```

## 🔧 Troubleshooting

### Webcam tidak terdeteksi

```bash
# Check available video devices
ls /dev/video*

# Try different camera index
python yoeo/scripts/openvino_realtime_inference.py --model weights/yoeo.xml --source 1
```

### FPS rendah

1. **Gunakan GPU** (jika tersedia):
   ```bash
   --device GPU
   ```

2. **Disable segmentation**:
   ```bash
   --no-seg
   ```

3. **Convert dengan resolusi lebih kecil** (trade-off: akurasi menurun):
   ```bash
   python yoeo/scripts/convert_yoeo_to_openvino.py \
       --config config/yoeo.cfg \
       --weights weights/yoeo.pth \
       --output weights/ \
       --img-size 320
   ```

### Error: Module not found

```bash
# Activate virtualenv
source botenv/bin/activate

# Install dependencies
pip install openvino openvino-dev onnx opencv-python torch torchvision
```

## 📈 Performance Expectations

### CPU (Intel i5/i7):
- FPS: 5-15 fps (tergantung CPU generation)
- Inference time: ~60-200ms per frame

### GPU (Intel Integrated):
- FPS: 10-25 fps
- Inference time: ~40-100ms per frame

### GPU (Dedicated Intel/NVIDIA dengan OpenVINO support):
- FPS: 20-60+ fps
- Inference time: ~15-50ms per frame

## 📝 Example Workflow

```bash
# 1. Activate environment
source botenv/bin/activate

# 2. Test model
python yoeo/scripts/test_openvino_model.py

# 3. Run webcam inference
./run_openvino_inference.sh
# Pilih opsi 1 untuk webcam

# 4. Process video file
python yoeo/scripts/openvino_realtime_inference.py \
    --model weights/yoeo.xml \
    --source input.mp4 \
    --save output.mp4 \
    --conf-thres 0.6
```

## 📚 Files Created

```
yoeo/scripts/
├── convert_yoeo_to_openvino.py      # Convert PyTorch -> OpenVINO
├── openvino_realtime_inference.py   # Real-time inference script
└── test_openvino_model.py           # Test model loading

run_openvino_inference.sh             # Quick start script
OPENVINO_INFERENCE.md                 # Detailed documentation
QUICK_START.md                        # This file
```

## 🎯 Next Steps

1. **Test dengan webcam:**
   ```bash
   ./run_openvino_inference.sh
   ```

2. **Adjust threshold untuk akurasi lebih baik:**
   - Naikkan `--conf-thres` (e.g., 0.6) untuk mengurangi false positives
   - Turunkan untuk deteksi lebih sensitif

3. **Optimize untuk kecepatan:**
   - Use GPU jika tersedia
   - Disable segmentation dengan press 's' atau `--no-seg`
   - Convert model dengan `--img-size 320` untuk inference lebih cepat

## ✅ Status

- ✓ PyTorch model loaded
- ✓ ONNX conversion successful
- ✓ OpenVINO IR conversion successful
- ✓ Model validation passed
- ✓ Test inference successful
- ✓ Ready for real-time inference!

**Model siap digunakan untuk real-time object detection dan segmentation!** 🎉
