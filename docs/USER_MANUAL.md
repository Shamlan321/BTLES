# User Manual: Bottle Liquid Level Inspection System

## Table of Contents

1. [Getting Started](#getting-started)
2. [User Interface Overview](#user-interface-overview)
3. [Basic Operations](#basic-operations)
4. [Calibration Guide](#calibration-guide)
5. [Advanced Settings](#advanced-settings)
6. [Data Management](#data-management)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First Launch

1. **Activate Virtual Environment** (if using):
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Launch Application**:
   ```bash
   python bottle_liquid_inspector.py
   ```

3. **Initial Screen**: You should see the main window with:
   - Black video preview area (left)
   - Control panel with settings (right)
   - Status bar at bottom showing "Ready"

### Quick Test Run

Before production use, test with the provided demo video:

1. Press `Ctrl+O` and select `coke_wan.mp4`
2. Click **Play** button
3. Observe bottle detection and classification
4. Check statistics in status bar

---

## User Interface Overview

### Main Window Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Menu Bar  File  Edit  View  Help                           │
├─────────────────────────────────────────────────────────────┤
│  Toolbar  [Load Video] [Play] [Stop]  FPS:45  Bottles:12    │
├──────────────────────────┬──────────────────────────────────┤
│                          │  Virtual Line Calibration        │
│  VIDEO PREVIEW           │  Target Line Y: [730]            │
│  (Live/Recorded)         │  Fine Adjust: [↑1px] [↓1px]      │
│                          │  Coarse: [↑↑10px] [↓↓10px]       │
│                          │  Tolerance: [±5px]               │
│                          ├──────────────────────────────────┤
│                          │  Detection Settings              │
│                          │  Confidence: [0.30]              │
│                          │  Gaussian Kernel: [5]            │
│                          │  Min Gradient: [10]              │
│                          ├──────────────────────────────────┤
│                          │  YOLO Model                      │
│                          │  [best.pt] [Browse...]           │
│                          ├──────────────────────────────────┤
│                          │  [Reset Statistics]              │
└──────────────────────────┴──────────────────────────────────┘
│  Status Bar: Bottles: 12 | ✓ Pass: 10 | ✗ Reject: 2        │
└─────────────────────────────────────────────────────────────┘
```

### Component Functions

#### **Video Preview Panel**
- Displays live camera feed or recorded video
- Shows real-time overlay graphics:
  - **Green horizontal line**: Target liquid level
  - **Red dashed lines**: Tolerance bounds (± tolerance)
  - **Cyan vertical line**: Detection zone boundary (50% frame width)
  - **Colored rectangles**: Detected bottles with track IDs
  - **Liquid level indicator**: Horizontal line inside each bottle

#### **Control Panel Sections**

1. **Virtual Line Calibration**
   - Set precise Y-coordinate for target liquid level
   - Fine adjustment (1 pixel) for precision
   - Coarse adjustment (10 pixels) for quick changes

2. **Detection Settings**
   - Confidence threshold for YOLO detections
   - Gaussian blur kernel size for edge detection
   - Minimum gradient threshold for liquid surface detection

3. **YOLO Model**
   - Path to trained PyTorch model (.pt file)
   - Browse button to select custom models

#### **Status Bar**
Real-time statistics display:
- Total bottles processed
- Number of passing bottles (green)
- Number of rejected bottles (red)

---

## Basic Operations

### Loading Video Source

#### Option 1: Load Video File

1. Press `Ctrl+O` or go to **File → Load Video**
2. Navigate to video file (supported: MP4, AVI, MOV, MKV)
3. Select file and click **Open**
4. Video path appears in toolbar
5. Processing starts automatically

#### Option 2: Use Camera Feed

1. Press `Ctrl+C` or go to **File → Use Camera**
2. Enter camera index:
   - `0` = Default webcam
   - `1` = Secondary camera
   - `2` = Third camera, etc.
3. Click **OK**
4. Live feed appears in preview window

### Playback Controls

#### Play/Pause
- **Button**: Click ▶ Play / ⏸ Pause button
- **Keyboard**: Press `Space`
- **Effect**: 
  - When paused: Video freezes, processing stops
  - When playing: Video runs at normal speed

#### Stop
- **Button**: Click ■ Stop button
- **Effect**: 
  - Stops video processing
  - Clears tracking history
  - Resets statistics
  - Returns to initial state

### Zoom Control

Use the zoom slider below video preview:
- **Slide Left**: Zoom out (50-99%)
- **Slide Right**: Zoom in (101-200%)
- **Center**: 100% (native resolution)

**Tip**: Zoom is useful for:
- Checking fine details during calibration
- Verifying liquid level detection accuracy
- Inspecting small bottles

---

## Calibration Guide

Proper calibration is critical for accurate liquid level detection.

### Step-by-Step Calibration

#### Step 1: Prepare Reference Sample

**Ideal Reference:**
- Bottle filled to EXACTLY the desired level
- Clear, well-lit image
- Bottle positioned upright (not tilted)
- Minimal glare or reflections

**For Video Files:**
1. Load video
2. Play until a good reference bottle appears
3. Press `Space` to pause

**For Live Camera:**
1. Place reference bottle in view
2. Ensure good lighting
3. Position bottle centrally in frame

#### Step 2: Initial Line Position

**Method A: Direct Input**
1. Look at control panel → Virtual Line Calibration
2. Enter approximate Y value in "Target Line Y" spinbox
3. Press Enter

**Method B: Button Adjustment**
1. Start with current value (e.g., 730)
2. Use coarse buttons (↑↑10px / ↓↓10px) to get close
3. Watch green line move on video preview

#### Step 3: Fine-Tune Position

**Critical Step - Use Arrow Keys:**

1. **Ensure video is paused** on clear bottle image
2. Press `↑` arrow key repeatedly to move line UP (1px per press)
3. Press `↓` arrow key to move line DOWN (1px per press)
4. Align the green line with the **top of the liquid meniscus**

**Meniscus Alignment:**
- For clear liquids: Align with the curved surface top
- For dark liquids (like Coke): Align with visible liquid/air boundary
- Be consistent across all calibrations

**Visual Feedback:**
```
     ───────────────  ← Green target line (should align here)
    │                │
    │    ████████    │  ← Liquid surface (meniscus)
    │    ████████    │
    │    ████████    │
    │    ████████    │
```

#### Step 4: Set Tolerance

1. Determine acceptable deviation (typical: ±3-10 pixels)
2. In control panel, adjust "Tolerance (±px)" spinbox
3. Red dashed lines appear showing tolerance range

**Tolerance Guidelines:**
- **High precision** (±2-3px): Critical fill applications
- **Standard** (±5px): Most beverage bottles
- **Lenient** (±10px): Large containers, non-critical fills

**Visual Check:**
```
     ───────────────  ← Upper tolerance bound (target + tolerance)
     ═══════════════  ← Green target line
     ───────────────  ← Lower tolerance bound (target - tolerance)
```

#### Step 5: Save Configuration

1. Press `Ctrl+S` or **File → Save Configuration**
2. Choose filename (e.g., `config_500ml_coke.json`)
3. Click **Save**

**Best Practice**: Create separate configs for:
- Different bottle sizes (500ml, 1L, 1.5L)
- Different products (Coke, Sprite, Water)
- Different production lines

### Calibration Verification

After calibration, verify accuracy:

1. **Resume playback** or introduce new bottles
2. **Observe classifications**:
   - Correctly filled bottles → **PASS** (green box)
   - Under-filled bottles → **REJECT_LOW** (red box)
3. **Check liquid level values**:
   - Should be close to target Y value (within tolerance)
   - Consistent readings across multiple bottles

**If Results Are Inconsistent:**
- Re-check meniscus alignment
- Verify lighting conditions
- Ensure bottles are upright
- Consider adjusting min_gradient_threshold

---

## Advanced Settings

### Detection Parameters

Access these in the "Detection Settings" panel.

#### Confidence Threshold

**What it does**: Minimum confidence score for YOLO to consider a detection valid.

**Range**: 0.1 - 1.0  
**Default**: 0.30

**Adjustment Guide**:
- **Lower (0.20-0.25)**: Detect more bottles, risk false positives
- **Higher (0.40-0.50)**: Fewer false positives, may miss some bottles

**When to Adjust**:
- If bottles are missed: Lower to 0.25
- If false detections occur: Raise to 0.40

#### Gaussian Kernel Size

**What it does**: Blur intensity applied before edge detection to reduce noise.

**Range**: 1-15 (must be odd number)  
**Default**: 5

**Adjustment Guide**:
- **Smaller (3)**: Preserves fine details, sensitive to noise
- **Larger (7-9)**: Smoother edges, may lose fine liquid surface details

**When to Adjust**:
- Noisy images (poor lighting): Increase to 7
- Sharp, clear images: Keep at 5 or reduce to 3

#### Minimum Gradient Threshold

**What it does**: Minimum intensity change required to detect liquid surface.

**Range**: 1-50  
**Default**: 10

**Adjustment Guide**:
- **Lower (5-8)**: Detect subtle liquid surfaces, may trigger on noise
- **Higher (15-20)**: Only strong edges detected, may miss low-contrast liquids

**When to Adjust**:
- Low-contrast liquids (water in clear bottle): Lower to 8
- High-contrast liquids (dark soda): Keep at 10-12
- Many false detections: Increase to 15

### Keyboard Shortcuts Reference

| Shortcut | Action | Context |
|----------|--------|---------|
| `Ctrl+O` | Load video file | Anytime |
| `Ctrl+C` | Use camera feed | Anytime |
| `Ctrl+S` | Save configuration | Anytime |
| `Ctrl+L` | Load configuration | Anytime |
| `Ctrl+E` | Export results (CSV) | After processing |
| `Ctrl+Q` | Quit application | Anytime |
| `Space` | Play/Pause | During video playback |
| `↑` | Move target line up 1px | When video paused |
| `↓` | Move target line down 1px | When video paused |
| `Shift+↑` | Move target line up 10px | When video paused |
| `Shift+↓` | Move target line down 10px | When video paused |

---

## Data Management

### Exporting Results

#### CSV Export

1. **Process bottles** (video must be running or completed)
2. Press `Ctrl+E` or **File → Export Results (CSV)**
3. Choose save location and filename
4. Click **Save**

**CSV File Structure**:
```csv
Track ID,Timestamp,Liquid Level Y,Status,Deviation,Bounding Box
1,2024-03-31T10:15:23.456,728,PASS,-2,"(120, 50, 280, 450)"
2,2024-03-31T10:15:24.123,715,REJECT_LOW,-15,"(320, 50, 480, 450)"
3,2024-03-31T10:15:25.789,NOT_DETECTED,NOT_DETECTED,N/A,"(520, 50, 680, 450)"
```

**Column Meanings**:
- **Track ID**: Unique bottle identifier
- **Timestamp**: ISO 8601 datetime when bottle crossed detection line
- **Liquid Level Y**: Detected liquid level in pixels (from image top)
- **Status**: PASS / REJECT_LOW / NOT_DETECTED
- **Deviation**: Pixels difference from target (negative = above target)
- **Bounding Box**: YOLO detection coordinates (x1, y1, x2, y2)

#### Using Exported Data

**Excel/Google Sheets Analysis**:
```excel
=COUNTIF(D:D, "PASS") / COUNTA(A:A)  // Pass rate percentage
=AVERAGE(E:E)  // Average deviation
```

**Quality Control Report**:
- Total bottles inspected
- Pass rate percentage
- Reject rate percentage
- Average fill level deviation
- Trend analysis over time

### Configuration Management

#### Save Configuration Presets

Create presets for different products:

1. **Calibrate** for Product A (e.g., 500ml Coke)
2. **Save As**: `config_coke_500ml.json`
3. **Change product** to Product B (e.g., 1L Sprite)
4. **Re-calibrate** target line and tolerance
5. **Save As**: `config_sprite_1l.json`

#### Load Configuration

1. Press `Ctrl+L` or **File → Load Configuration**
2. Navigate to config file (e.g., `config_coke_500ml.json`)
3. Click **Open**
4. All settings update automatically

**Pro Tip**: Store configs in a `configs/` subdirectory for organization:
```
bottle-liquid-inspection/
├── configs/
│   ├── coke_500ml.json
│   ├── sprite_1l.json
│   └── water_1.5l.json
├── bottle_liquid_inspector.py
└── best.pt
```

### Resetting Statistics

To clear counters without restarting:

1. Click **Reset Statistics** button in control panel
2. Confirms reset
3. All counters return to zero
4. Tracking history cleared

**Use Cases**:
- Starting new production batch
- Testing different configurations
- Clearing erroneous data

---

## Troubleshooting

### Common Operational Issues

#### Issue: No Bottles Detected

**Symptoms**: Video plays but no bounding boxes appear

**Possible Causes & Solutions**:

1. **Model not loaded**:
   - Check if `best.pt` exists in project directory
   - Verify model path in control panel
   - Click **Browse...** and re-select model

2. **Confidence too high**:
   - Lower confidence threshold to 0.20
   - See if detections appear
   - Gradually increase back up

3. **Wrong video source**:
   - Ensure video file is valid
   - Try different video file
   - For camera: verify correct camera index

#### Issue: Many False Detections

**Symptoms**: Boxes appear on non-bottle objects

**Solutions**:
- Increase confidence threshold to 0.40-0.50
- Retrain model with more diverse negative samples
- Check lighting conditions (reduce glare/shadows)

#### Issue: Inconsistent Liquid Levels

**Symptoms**: Same bottle shows different levels frame-to-frame

**Solutions**:
- Pause video when measuring (don't trust moving bottles)
- Increase min_gradient_threshold to 15-20
- Ensure consistent backlighting
- Check for bottle wobble or vibration

#### Issue: All Bottles Rejected

**Symptoms**: Even correctly-filled bottles marked as REJECT_LOW

**Solutions**:
- Re-calibrate target line (likely misaligned)
- Increase tolerance value
- Verify target_line_y matches actual liquid level

#### Issue: Application Runs Slowly

**Symptoms**: FPS < 10, laggy interface

**Solutions**:
- Enable GPU acceleration (see INSTALLATION.md)
- Reduce camera/video resolution
- Use smaller YOLO model (yolov8n instead of yolov8s)
- Close other applications

#### Issue: Cannot Export Results

**Symptoms**: Export button greyed out or error on export

**Solutions**:
- Ensure at least one bottle has been processed
- Check file permissions in save directory
- Try different save location
- Restart application

### Performance Optimization

#### Maximize Detection Speed

1. **Hardware**:
   - Use dedicated GPU (NVIDIA RTX series recommended)
   - Ensure adequate cooling (thermal throttling reduces performance)

2. **Software**:
   - Close unnecessary background applications
   - Use wired Ethernet for IP cameras (not WiFi)
   - Set power plan to "High Performance"

3. **Application Settings**:
   - Reduce display zoom to 100% or less
   - Disable debug visualizations
   - Process every Nth frame (requires code modification)

#### Improve Detection Accuracy

1. **Optimal Lighting**:
   - Diffuse backlighting (light behind bottles)
   - Avoid direct reflections on bottle surface
   - Consistent lighting throughout production run

2. **Camera Positioning**:
   - Perpendicular to bottle movement
   - Fixed, stable mount (no vibration)
   - Proper focus (manual focus preferred)

3. **Calibration Quality**:
   - Use high-quality reference sample
   - Calibrate at operating temperature
   - Re-calibrate if lighting changes

### Getting Help

If problems persist:

1. **Check Logs**: Review terminal/console output for error messages
2. **GitHub Issues**: Search existing issues or create new one
3. **Documentation**: Review TRAINING_GUIDE.md and INSTALLATION.md
4. **Community**: Post on Ultralytics forums or Stack Overflow

---

## Appendix A: Production Checklist

**Daily Startup**:
- [ ] Load appropriate configuration for product
- [ ] Verify camera is clean and focused
- [ ] Test with known-good reference bottle
- [ ] Confirm pass/reject decisions match expectations

**During Production**:
- [ ] Monitor reject rate (alert if >5%)
- [ ] Periodically verify with reference samples
- [ ] Log any unusual patterns or errors

**Shutdown**:
- [ ] Export daily results to CSV
- [ ] Save final statistics
- [ ] Clean camera lens if needed
- [ ] Document any issues or maintenance needs

**Weekly Maintenance**:
- [ ] Back up configuration files
- [ ] Review trend data for drift
- [ ] Clean optical components
- [ ] Verify lighting consistency

---

**End of User Manual**

For technical details, see:
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Custom model training
- [INSTALLATION.md](INSTALLATION.md) - Setup and installation
- [API_REFERENCE.md](API_REFERENCE.md) - Developer documentation
