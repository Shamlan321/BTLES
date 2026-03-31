# Installation Guide

Detailed installation instructions for different operating systems and configurations.

## Table of Contents

- [Windows Installation](#windows)
- [Linux Installation](#linux)
- [macOS Installation](#macos)
- [GPU Acceleration Setup](#gpu-acceleration)
- [Docker Deployment](#docker)
- [Troubleshooting](#troubleshooting)

---

## Windows

### Prerequisites

1. **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation
   - ✅ Verify: Open PowerShell and run `python --version`

2. **Git** (optional): Download from [git-scm.com](https://git-scm.com/)

### Step-by-Step Installation

#### 1. Clone or Download Repository

```powershell
# Using Git
git clone https://github.com/yourusername/bottle-liquid-inspection.git
cd bottle-liquid-inspection

# OR download ZIP and extract
```

#### 2. Create Virtual Environment

```powershell
# Navigate to project directory
cd C:\path\to\bottle-liquid-inspection

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate
```

You should see `(venv)` prefix in your PowerShell prompt.

#### 3. Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

This installs:
- ultralytics (YOLOv8)
- opencv-python
- numpy
- PyQt5

#### 4. Verify Installation

```powershell
python -c "import cv2; import ultralytics; import PyQt5; print('✓ All dependencies installed')"
```

#### 5. Download YOLO Model

Place your trained `best.pt` model file in the project root directory.

#### 6. Run Application

```powershell
python bottle_liquid_inspector.py
```

---

## Linux

### Ubuntu/Debian

#### Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install -y python3 python3-pip python3-venv git

# Install system dependencies for OpenCV
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libqt5widgets5
```

#### Installation Steps

```bash
# Clone repository
git clone https://github.com/yourusername/bottle-liquid-inspection.git
cd bottle-liquid-inspection

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2; import ultralytics; import PyQt5; print('Success!')"

# Run application
python bottle_liquid_inspector.py
```

### Fedora/RHEL

```bash
# Install dependencies
sudo dnf install -y python3 python3-pip python3-virtualenv git mesa-libGL

# Create virtual environment and install (same as Ubuntu)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## macOS

### Intel and Apple Silicon (M1/M2)

#### Prerequisites

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11 git
```

#### Installation Steps

```bash
# Clone repository
git clone https://github.com/yourusername/bottle-liquid-inspection.git
cd bottle-liquid-inspection

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
# For M1/M2 Macs, install torch separately first
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# Then install main requirements
pip install -r requirements.txt

# Run application
python bottle_liquid_inspector.py
```

**Note for M1/M2**: PyQt5 may have issues with native ARM builds. If you encounter problems:

```bash
# Try alternative Qt backend
pip install pyside6
```

Then modify `bottle_liquid_inspector.py` imports:
```python
from PySide6.QtWidgets import ...  # Instead of PyQt5
```

---

## GPU Acceleration

### NVIDIA CUDA Setup (Windows/Linux)

#### Prerequisites

- NVIDIA GPU with compute capability ≥ 3.5
- CUDA Toolkit 11.7 or 11.8
- cuDNN libraries

#### Step 1: Install CUDA Toolkit

**Windows:**
1. Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Run installer, follow default options
3. Verify: `nvcc --version`

**Linux (Ubuntu):**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-8
```

#### Step 2: Install PyTorch with CUDA

```bash
# Uninstall CPU-only torch if installed
pip uninstall torch torchvision

# Install GPU-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Step 3: Verify GPU Access

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

Expected output:
```
CUDA available: True
GPU count: 1
```

#### Step 4: Configure Application

The application will automatically use GPU if available. No configuration changes needed.

**Monitor GPU Usage:**
```bash
# Windows
nvidia-smi

# Linux
watch -n 1 nvidia-smi
```

---

## Docker

### Prerequisites

- Docker Desktop (Windows/macOS) or Docker CE (Linux)
- NVIDIA Container Toolkit (for GPU support)

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libqt5widgets5 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose X11 for GUI (Linux only)
ENV DISPLAY=:0

# Default command
CMD ["python", "bottle_liquid_inspector.py"]
```

### Build and Run

```bash
# Build Docker image
docker build -t bottle-inspector:latest .

# Run (basic)
docker run -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    bottle-inspector:latest

# Run with GPU support
docker run -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    bottle-inspector:latest
```

**Note**: Docker GUI requires X11 forwarding setup. For production, consider using a headless mode or VNC.

---

## Troubleshooting

### Common Installation Issues

#### Issue: `pip install` fails with permission errors

**Solution (Windows):**
```powershell
# Run PowerShell as Administrator
.\venv\Scripts\Activate
pip install -r requirements.txt
```

**Solution (Linux/Mac):**
```bash
# Never use sudo with pip in virtualenv
# Make sure virtualenv is activated
source venv/bin/activate
pip install -r requirements.txt
```

#### Issue: PyQt5 import error on Linux

```bash
# Ubuntu/Debian
sudo apt install -y python3-pyqt5

# Or reinstall via pip
pip uninstall PyQt5
pip install PyQt5==5.15.9
```

#### Issue: OpenCV cannot find Qt backend

```bash
# Reinstall opencv with Qt support
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

#### Issue: CUDA not detected

1. Verify CUDA installation:
   ```bash
   nvcc --version
   ```

2. Check PyTorch CUDA compatibility:
   ```python
   python -c "import torch; print(torch.version.cuda)"
   ```

3. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

#### Issue: Application window doesn't open (macOS)

```bash
# Grant permissions
xattr -cr /path/to/python.app

# Or try running without virtualenv
python3 bottle_liquid_inspector.py
```

#### Issue: Very slow performance on CPU

- Ensure you're not running in debug mode
- Reduce camera resolution if possible
- Use smaller YOLO model (yolov8n instead of yolov8s)
- Consider adding GPU acceleration

### Performance Benchmarks

Expected FPS on different hardware (1920x1080 video, imgsz=640):

| Hardware | FPS | Notes |
|----------|-----|-------|
| RTX 3090 | 45-60 | GPU accelerated |
| RTX 3060 | 30-45 | GPU accelerated |
| GTX 1080 | 20-30 | GPU accelerated |
| i7-12700K (CPU only) | 8-12 | AVX2 optimized |
| Raspberry Pi 4 | 1-2 | Not recommended |

### Getting Help

If you encounter issues not covered here:

1. **Check GitHub Issues**: https://github.com/yourusername/bottle-liquid-inspection/issues
2. **System Information**: Include OS, Python version, GPU, and error messages
3. **Logs**: Share output from the terminal/console

---

## Next Steps

After successful installation:

1. ✅ Read the [README.md](../README.md) for usage instructions
2. ✅ Review [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for custom model training
3. ✅ Check [USER_MANUAL.md](USER_MANUAL.md) for detailed operation guide

**Happy inspecting!** 🎉
