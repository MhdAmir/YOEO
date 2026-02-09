#!/bin/bash
# Quick start script untuk YOEO OpenVINO conversion dan inference

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}YOEO OpenVINO Setup & Inference${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check if botenv exists
if [ ! -d "botenv" ]; then
    echo -e "${RED}Error: Virtual environment 'botenv' not found!${NC}"
    echo "Please create it first with: python3 -m venv botenv"
    exit 1
fi

# Activate virtualenv
echo -e "${YELLOW}Activating virtual environment...${NC}"
source botenv/bin/activate

# Check if dependencies are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import openvino" 2>/dev/null || {
    echo -e "${YELLOW}Installing required dependencies...${NC}"
    pip install openvino openvino-dev onnx opencv-python
}

# Check if model files exist
if [ ! -f "weights/yoeo.xml" ]; then
    echo ""
    echo -e "${YELLOW}OpenVINO model not found. Starting conversion...${NC}"
    echo ""
    
    # Check if PyTorch weights exist
    if [ ! -f "weights/yoeo.pth" ]; then
        echo -e "${RED}Error: PyTorch weights not found at weights/yoeo.pth${NC}"
        echo "Please download weights first!"
        exit 1
    fi
    
    # Run conversion
    python yoeo/scripts/convert_yoeo_to_openvino.py \
        --config config/yoeo.cfg \
        --weights weights/yoeo.pth \
        --output weights/
    
    echo ""
    echo -e "${GREEN}✓ Conversion completed!${NC}"
    echo ""
else
    echo -e "${GREEN}✓ OpenVINO model found at weights/yoeo.xml${NC}"
fi

# Ask user for inference option
echo ""
echo -e "${YELLOW}Choose inference option:${NC}"
echo "  1) Webcam (default)"
echo "  2) Video file"
echo "  3) Webcam dengan save output"
echo "  4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1|"")
        echo ""
        echo -e "${GREEN}Starting webcam inference...${NC}"
        echo -e "${YELLOW}Press 'q' to quit, 's' to toggle segmentation${NC}"
        echo ""
        python yoeo/scripts/openvino_realtime_inference.py \
            --model weights/yoeo.xml \
            --source 2
        ;;
    2)
        echo ""
        read -p "Enter video file path: " video_path
        if [ ! -f "$video_path" ]; then
            echo -e "${RED}Error: Video file not found!${NC}"
            exit 1
        fi
        echo ""
        echo -e "${GREEN}Starting video inference...${NC}"
        echo ""
        python yoeo/scripts/openvino_realtime_inference.py \
            --model weights/yoeo.xml \
            --source "$video_path"
        ;;
    3)
        echo ""
        read -p "Enter output video filename [output.mp4]: " output_file
        output_file=${output_file:-output.mp4}
        echo ""
        echo -e "${GREEN}Starting webcam inference with recording...${NC}"
        echo -e "${YELLOW}Output will be saved to: $output_file${NC}"
        echo -e "${YELLOW}Press 'q' to quit, 's' to toggle segmentation${NC}"
        echo ""
        python yoeo/scripts/openvino_realtime_inference.py \
            --model weights/yoeo.xml \
            --source 0 \
            --save "$output_file"
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice!${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
