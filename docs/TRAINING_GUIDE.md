# Training Guide: Custom YOLOv8 Model for Bottle Detection

This guide walks you through training a custom YOLOv8 model for your specific bottle types and production environment.

## Table of Contents

1. [Dataset Preparation](#dataset-preparation)
2. [Annotation](#annotation)
3. [Training Setup](#training-setup)
4. [Training Process](#training-process)
5. [Model Evaluation](#model-evaluation)
6. [Export and Deployment](#export-and-deployment)

---

## Dataset Preparation

### Step 1: Collect Images

**Quantity**: 100-500 images minimum (more is better)

**Image Sources:**
- Capture from your production line camera at various times of day
- Include different lighting conditions
- Capture bottles at different positions on the conveyor
- Include partially occluded bottles
- Add images with reflections, glare, or shadows

**Best Practices:**
- Use the actual camera that will be used in production
- Maintain consistent resolution (e.g., 1920x1080)
- Include empty bottles and over-filled bottles as edge cases
- Capture bottles at different speeds if applicable

### Step 2: Organize Dataset Structure

Create the following directory structure:

```
bottle_dataset/
├── images/
│   ├── train/          # 80% of images
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/            # 20% of images
│       ├── img_081.jpg
│       ├── img_082.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img_001.txt
    │   ├── img_002.txt
    │   └── ...
    └── val/
        ├── img_081.txt
        ├── img_082.txt
        └── ...
```

---

## Annotation

### Option A: Roboflow (Recommended for Beginners)

1. **Upload Images**: Go to [roboflow.com](https://roboflow.com) and create a project
2. **Label Images**: Use their online annotation tool
3. **Export Format**: Select "YOLOv8" format
4. **Download**: Get the preprocessed dataset with train/val splits

### Option B: CVAT (Advanced Features)

1. Install CVAT: `docker run -p 8080:8080 openvino/cvat`
2. Create project and upload images
3. Draw bounding boxes around bottles
4. Export in YOLO format

### Option C: LabelImg (Offline Desktop Tool)

```bash
pip install labelimg
labelimg
```

1. Open image directory
2. Set save format to "YOLO"
3. Draw rectangles around bottles
4. Save annotations

### Annotation Classes

For basic liquid level inspection, use a single class:

```
0: bottle
```

If you want to detect multiple bottle types or states:

```
0: bottle_500ml
1: bottle_1l
2: bottle_1.5l
3: cap_missing
```

### Annotation Quality Tips

- **Tight Boxes**: Draw boxes tightly around the entire bottle including cap
- **Consistency**: Be consistent with box tightness across all images
- **Occlusions**: Annotate partially visible bottles (mark only visible portion)
- **Edge Cases**: Include and properly annotate under-filled and over-filled bottles

---

## Training Setup

### Hardware Requirements

**Minimum:**
- GPU: GTX 1060 6GB or better
- RAM: 8GB
- Storage: 50GB free space

**Recommended:**
- GPU: RTX 3060 12GB or better
- RAM: 16GB
- Storage: 100GB SSD

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install ultralytics
pip install ultralytics

# Install PyTorch with CUDA (for GPU acceleration)
# Visit https://pytorch.org/get-started/locally/ for latest command
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Dataset Configuration File

Create `dataset.yaml`:

```yaml
path: ./bottle_dataset
train: images/train
val: images/val

nc: 1  # number of classes
names: ['bottle']  # class names

# Augmentation parameters
augment: true
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 2.0
translate: 0.1
scale: 0.5
shear: 2.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
```

---

## Training Process

### Basic Training Script

Create `train_model.py`:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8s.pt')  # Start with small model

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU device ID (or 'cpu')
    workers=8,
    optimizer='auto',
    patience=50,  # Early stopping patience
    exist_ok=True,
    project='runs/detect',
    name='bottle_detection_v1'
)

print(f"Training complete! Model saved to: {results.save_dir}")
```

### Run Training

```bash
python train_model.py
```

### Training Parameters Explained

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `epochs` | Number of training iterations | 100-300 |
| `imgsz` | Input image size | 640 (or match your camera aspect ratio) |
| `batch` | Batch size | 16 (adjust based on GPU memory) |
| `device` | GPU device | `0` for first GPU, `cpu` for CPU-only |
| `patience` | Early stopping epochs | 50 (stop if no improvement) |
| `lr0` | Initial learning rate | 0.01 (default works well) |

### Monitoring Training

Ultralytics automatically creates:
- `runs/detect/bottle_detection_v1/weights/best.pt` - Best model
- `runs/detect/bottle_detection_v1/results.csv` - Training metrics
- `runs/detect/bottle_detection_v1/results.png` - Training curves

View training progress:

```bash
# Open TensorBoard
tensorboard --logdir runs/detect
```

---

## Model Evaluation

### Test the Model

Create `evaluate_model.py`:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/bottle_detection_v1/weights/best.pt')

# Run inference on validation set
metrics = model.val(data='dataset.yaml')

print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

### Key Metrics

- **mAP50**: Mean Average Precision at IoU=0.50 (target: >0.90)
- **mAP50-95**: mAP averaged over IoU thresholds 0.50-0.95 (target: >0.70)
- **Precision**: What fraction of detected bottles are actual bottles (target: >0.95)
- **Recall**: What fraction of actual bottles are detected (target: >0.95)

### Visual Inspection

```python
import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/bottle_detection_v1/weights/best.pt')

# Test on sample images
image_path = 'bottle_dataset/images/val/img_081.jpg'
results = model(image_path)

# Show results
results[0].show()  # Display with bounding boxes
```

### Common Issues and Solutions

**Issue: Low Precision (<0.80)**
- Increase confidence threshold during inference
- Add more diverse training data
- Reduce false positives by adding negative samples (images without bottles)

**Issue: Low Recall (<0.80)**
- Lower confidence threshold
- Ensure all bottles in training set are properly annotated
- Add more training images with small/occluded bottles

**Issue: Overfitting (high train accuracy, low val accuracy)**
- Reduce model complexity (use yolov8n instead of yolov8l)
- Increase augmentation
- Add more training data
- Use early stopping

---

## Export and Deployment

### Export to ONNX (Optional)

For deployment on edge devices:

```python
from ultralytics import YOLO

model = YOLO('runs/detect/bottle_detection_v1/weights/best.pt')

# Export to ONNX
model.export(format='onnx', opset=12)

# Export to TensorRT (NVIDIA Jetson)
model.export(format='engine', dynamic=True)
```

### Deploy to Production

1. **Copy Model**: Move `best.pt` to your application directory
2. **Update Config**: Set `yolo_model` path in `config.json`
3. **Test**: Run inference on production camera feed
4. **Monitor**: Track detection accuracy and adjust parameters as needed

```bash
# Copy model
cp runs/detect/bottle_detection_v1/weights/best.pt ./best.pt

# Update config
# Edit config.json: "yolo_model": "best.pt"

# Run application
python bottle_liquid_inspector.py
```

### Performance Optimization

**For Real-time Processing:**

1. **Use Smaller Model**: YOLOv8n or YOLOv8s (faster but slightly less accurate)
2. **Reduce Input Size**: `imgsz=416` instead of 640 (trades accuracy for speed)
3. **Batch Processing**: Process multiple frames together if latency allows
4. **GPU Acceleration**: Ensure CUDA is properly configured

```python
# Optimized inference settings
results = model.track(
    frame,
    conf=0.3,  # Lower confidence for more detections
    iou=0.45,  # Standard NMS threshold
    tracker="bytetrack.yaml",
    persist=True,
    verbose=False,
    imgsz=416  # Smaller input for faster processing
)
```

---

## Advanced: Multi-Class Detection

### Training for Multiple Bottle Types

1. **Annotate with Multiple Classes**:
   ```
   0: bottle_small
   1: bottle_medium
   2: bottle_large
   ```

2. **Update Dataset YAML**:
   ```yaml
   nc: 3
   names: ['bottle_small', 'bottle_medium', 'bottle_large']
   ```

3. **Train**: Same process, model will learn all classes

4. **Configure Application**: Set different target lines per bottle type

---

## Troubleshooting

### Training Fails to Start

**Error: CUDA out of memory**
- Reduce `batch` size (try 8 or 4)
- Use smaller model (yolov8n)
- Reduce `imgsz`

**Error: No labels found**
- Check file paths in `dataset.yaml`
- Verify label files are in YOLO format (class x_center y_center width height)
- Ensure coordinates are normalized (0-1 range)

### Poor Detection Quality

**Model misses bottles:**
- Add more training images with similar scenarios
- Increase image diversity (lighting, angles, positions)
- Check annotation quality (tight boxes)

**Many false positives:**
- Add negative samples (images without bottles)
- Increase confidence threshold in application
- Fine-tune with harder examples

---

## Next Steps

After successful training:

1. **Document Your Model**: Record training data size, parameters, and performance
2. **Version Control**: Tag model versions (v1.0, v1.1, etc.)
3. **Continuous Improvement**: Collect edge cases and retrain periodically
4. **Share**: Consider publishing your dataset/model for community benefit

---

## Resources

- **Ultralytics YOLOv8 Docs**: https://docs.ultralytics.com/
- **Roboflow Universe**: https://universe.roboflow.com/ (pretrained models)
- **Papers With Code**: https://paperswithcode.com/task/object-detection (latest research)
- **YOLOv8 GitHub**: https://github.com/ultralytics/ultralytics

---

**Need Help?** Join the Ultralytics Discord community or post questions on Stack Overflow with tag `yolov8`.
