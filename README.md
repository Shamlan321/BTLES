# Bottle Liquid Level Inspection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Professional automated liquid level quality control system using computer vision and deep learning**

A high-performance PyQt5-based GUI application for real-time bottle liquid level inspection using YOLOv8 object detection with ByteTrack tracking. Designed for production line quality control, laboratory testing, and automated visual inspection.

![Python](https://img.shields.io/badge/PyQt5-5.15+-green)
![YOLO](https://img.shields.io/badge/YOLOv8-8.0+-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-orange)

---

## 🌟 Features

### Core Capabilities
- **Real-time Processing**: Live video analysis at production line speeds (≥8-10 bottles/sec)
- **AI-Powered Detection**: YOLOv8 + ByteTrack for robust bottle tracking and identification
- **Precision Measurement**: Pixel-perfect liquid level measurement with ±1px fine adjustment
- **Automated Classification**: Pass/Fail decision based on configurable tolerance thresholds
- **Multi-Source Input**: Support for video files (MP4, AVI, MOV, MKV) and camera feeds (USB, RTSP, IP cameras)

### User Interface
- **Intuitive Controls**: Keyboard shortcuts and mouse controls for precise calibration
- **Live Visualization**: Real-time overlay of detection zones, bounding boxes, and measurements
- **Statistical Dashboard**: Live counters for total bottles, pass/reject rates, and defect percentages
- **Configuration Management**: Save/load presets for different products and production lines
- **Data Export**: CSV export for quality assurance records and traceability

### Advanced Features
- **Virtual Line Calibration**: Interactive positioning with keyboard arrow keys (1px fine / 10px coarse)
- **Adaptive Detection**: Configurable confidence thresholds and gradient sensitivity
- **Multi-Bottle Tracking**: Persistent tracking IDs across frames for accurate counting
- **Tolerance Bands**: Visual feedback for acceptable liquid level ranges
- **Performance Monitoring**: FPS counter and processing time metrics

---

## 📋 System Requirements

### Hardware
- **CPU**: Intel i5 or equivalent (i7/i9 recommended for real-time processing)
- **GPU**: NVIDIA CUDA-compatible GPU (optional but recommended for faster inference)
- **RAM**: 8GB minimum, 16GB recommended
- **Camera**: USB 3.0 webcam or industrial camera (for live inspection)

### Software
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS (Intel/Apple Silicon)
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (if using GPU acceleration)

---

## 🚀 Installation

### Step 1: Clone or Download

```bash
git clone https://github.com/yourusername/bottle-liquid-inspection.git
cd bottle-liquid-inspection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download YOLO Model

Download the pre-trained YOLOv8 model (`best.pt`) and place it in the project directory, or use your custom-trained model.

**Option A: Use Provided Model**
If you have a trained model, place it at: `./best.pt`

**Option B: Train Your Own**
See [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for custom model training instructions.

---

## 💻 Quick Start

### Launch the Application

```bash
python bottle_liquid_inspector.py
```

### Basic Workflow

1. **Load Video/Camera**
   - Press `Ctrl+O` to load a video file
   - Or press `Ctrl+C` to use a camera feed
   
2. **Calibrate Virtual Line**
   - Pause the video when a bottle with correct liquid level is visible
   - Use **Arrow Keys** (↑↓) for fine adjustment (1px)
   - Use **Shift+Arrow Keys** for coarse adjustment (10px)
   - Position the green target line at the desired liquid level

3. **Set Tolerance**
   - Adjust the tolerance value (± pixels) in the control panel
   - Red dashed lines show the acceptable range

4. **Start Inspection**
   - Click **Play** button or press **Space** to start
   - Monitor real-time statistics in the status bar
   - Bottles are automatically classified as PASS/REJECT

5. **Export Results**
   - Press `Ctrl+E` to export results to CSV
   - Data includes: Track ID, timestamp, liquid level, status, deviation

---

## 🎮 User Guide

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Load video file |
| `Ctrl+C` | Use camera feed |
| `Ctrl+S` | Save configuration |
| `Ctrl+L` | Load configuration |
| `Ctrl+E` | Export results (CSV) |
| `Ctrl+Q` | Quit application |
| `Space` | Play/Pause |
| `↑` | Move target line up (1px) |
| `↓` | Move target line down (1px) |
| `Shift+↑` | Move target line up (10px) |
| `Shift+↓` | Move target line down (10px) |

### UI Components

#### **Video Preview Panel (Left)**
- Real-time video display with overlay graphics
- Zoom slider for detailed inspection
- Green line: Target liquid level
- Red dashed lines: Tolerance bounds
- Cyan vertical line: Detection zone boundary

#### **Control Panel (Right)**
- **Virtual Line Calibration**: Set target Y position
- **Fine/Coarse Adjustment**: Button controls for manual positioning
- **Detection Settings**: Confidence, Gaussian kernel, gradient threshold
- **Model Selection**: Browse and load custom YOLO models

#### **Status Bar (Bottom)**
- FPS: Frames per second processing rate
- Bottles: Total count of processed bottles
- Pass: Number of bottles within tolerance
- Reject: Number of bottles below target level

### Calibration Process

The accuracy of liquid level detection depends on proper calibration:

1. **Load Reference Video**: Use a video with clear bottles at correct fill levels
2. **Pause on Clear Frame**: Stop when a bottle is centered and liquid level is visible
3. **Position Target Line**: 
   - Use arrow keys for pixel-perfect positioning
   - Align with the top of the liquid meniscus
   - Typical precision: ±1-2 pixels
4. **Set Tolerance**: Define acceptable deviation (e.g., ±5px)
5. **Save Configuration**: Store settings for future use

### Interpreting Results

**PASS** (Green box):
- Liquid level is at or above target line (within tolerance)
- Bottle meets quality standards

**REJECT_LOW** (Red box):
- Liquid level is below the tolerance range
- Bottle is under-filled and should be rejected

**NOT_DETECTED** (Orange box):
- Liquid level could not be reliably detected
- May require manual inspection or parameter tuning

---

## ⚙️ Configuration

### Configuration File Structure

Configurations are saved as JSON files with the following structure:

```json
{
  "yolo_model": "best.pt",
  "target_line_y": 730,
  "tolerance": 5,
  "detection_line_x_percent": 0.5,
  "conf_threshold": 0.3,
  "iou_threshold": 0.45,
  "gaussian_kernel": 5,
  "min_gradient_threshold": 10,
  "display_scale": 0.7,
  "show_debug": false,
  "show_tracking_ids": true
}
```

### Key Parameters

| Parameter | Description | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `target_line_y` | Target liquid level Y coordinate (pixels) | 0-4096 | 730 |
| `tolerance` | Acceptable deviation from target (pixels) | 1-50 | 5 |
| `conf_threshold` | YOLO detection confidence threshold | 0.1-1.0 | 0.3 |
| `gaussian_kernel` | Blur kernel size for edge detection | 1-15 (odd) | 5 |
| `min_gradient_threshold` | Minimum gradient for liquid detection | 1-50 | 10 |

### Production Line Setup

For multiple products or production lines:

1. Create separate config files (e.g., `config_500ml.json`, `config_1l.json`)
2. Load appropriate config before starting inspection
3. Save product-specific calibrations for quick changeovers

---

## 🔧 Advanced Usage

### Custom Model Training

For best results, train a custom YOLOv8 model on your specific bottles:

1. **Dataset Preparation**: Collect 100-500 labeled images of your bottles
2. **Annotation**: Use tools like Roboflow, CVAT, or LabelImg
3. **Training**: Follow [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)
4. **Validation**: Test on unseen images to verify accuracy

### Performance Optimization

**Maximize FPS:**
- Use GPU acceleration (CUDA)
- Reduce input resolution (maintain aspect ratio)
- Lower `conf_threshold` (trade-off: more false positives)
- Process every Nth frame instead of all frames

**Improve Detection Accuracy:**
- Increase `conf_threshold` (trade-off: may miss some bottles)
- Tune `gaussian_kernel` for your lighting conditions
- Adjust `min_gradient_threshold` based on liquid contrast
- Ensure consistent backlighting in production environment

### Integration with PLC/SCADA

The system can integrate with industrial automation:

```python
# Example: Modbus TCP communication
from pymodbus.client import ModbusTcpClient

client = ModbusTcpClient('192.168.1.100')

# Write reject signal when bottle fails
if result['status'] == 'REJECT_LOW':
    client.write_coil(0, True, unit=1)
```

See [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) for PLC communication examples.

---

## 📊 Data Export Format

Results are exported as CSV with the following columns:

```csv
Track ID,Timestamp,Liquid Level Y,Status,Deviation,Bounding Box
1,2024-03-31T10:15:23.456,728,PASS,-2,"(120, 50, 280, 450)"
2,2024-03-31T10:15:24.123,715,REJECT_LOW,-15,"(320, 50, 480, 450)"
```

**Fields:**
- **Track ID**: Unique identifier for each bottle
- **Timestamp**: ISO 8601 format datetime
- **Liquid Level Y**: Detected liquid level in pixels (from image top)
- **Status**: PASS / REJECT_LOW / NOT_DETECTED
- **Deviation**: Difference from target line (positive = below target)
- **Bounding Box**: YOLO detection box coordinates (x1, y1, x2, y2)

---

## 🐛 Troubleshooting

### Common Issues

**Issue: Low FPS (<5)**
- **Solution**: Enable GPU, reduce resolution, or process every 2nd frame

**Issue: Bottles not detected**
- **Solution**: Lower `conf_threshold`, ensure good lighting, check model compatibility

**Issue: Inconsistent liquid level readings**
- **Solution**: Increase `min_gradient_threshold`, improve backlighting, reduce glare

**Issue: Many false rejections**
- **Solution**: Increase `tolerance`, verify target line calibration, check for bottle wobble

**Issue: Application won't start**
- **Solution**: Verify Python 3.8+, install dependencies, check PyQt5 installation

### Performance Tips

1. **Lighting**: Use diffuse backlighting to enhance liquid/air interface contrast
2. **Camera Position**: Mount camera perpendicular to bottle movement
3. **Focus**: Ensure bottles are in sharp focus at detection zone
4. **Stability**: Minimize vibration and bottle wobble
5. **Background**: Use uniform high-contrast background

---

## 📚 Documentation

- **[User Manual](docs/USER_MANUAL.md)**: Detailed usage instructions
- **[Training Guide](docs/TRAINING_GUIDE.md)**: Custom YOLO model training
- **[API Reference](docs/API_REFERENCE.md)**: Developer documentation
- **[Integration Guide](docs/INTEGRATION_GUIDE.md)**: PLC/SCADA integration
- **[Installation](docs/INSTALLATION.md)**: Platform-specific setup

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for:

- Reporting bugs and feature requests
- Submitting pull requests
- Code style and testing requirements
- Community code of conduct

### Development Setup

```bash
# Fork and clone repository
git clone https://github.com/yourusername/bottle-liquid-inspection.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use this software in academic research, please cite:

```bibtex
@software{bottle_liquid_inspection_2024,
  title = {Bottle Liquid Level Inspection System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/bottle-liquid-inspection}
}
```

---

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for excellent object detection framework
- **ByteTrack**: Multi-object tracking algorithm
- **OpenCV**: Computer vision library
- **PyQt5**: Cross-platform GUI framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/bottle-liquid-inspection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/bottle-liquid-inspection/discussions)
- **Email**: your.email@example.com

---

## 🗺️ Roadmap

- [ ] Multi-camera support
- [ ] Cloud data logging integration
- [ ] Machine learning-based anomaly detection
- [ ] Web-based dashboard for remote monitoring
- [ ] Docker containerization for easy deployment
- [ ] Support for multiple bottle types simultaneously

---

**Made with ❤️ for quality control and automation**

*Last updated: March 31, 2024*
