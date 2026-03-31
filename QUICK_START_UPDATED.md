# Quick Start Guide - Updated UI with Dark Theme

## 🚀 Launch & Basic Operation

### Step 1: Start Application
```bash
python bottle_liquid_inspector.py
```

**What You'll See:**
- Modern dark themed window (1600x1000)
- Large black video preview area (left)
- Styled control panel with gray backgrounds (right)
- Blue and cyan accent colors throughout

---

### Step 2: Load Your Video
**Method A - Keyboard Shortcut:**
Press `Ctrl+O`

**Method B - Menu:**
Click **File** → **Load Video**

**Browse to your video file** (MP4, AVI, MOV, or MKV)

---

### Step 3: Open Calibration Mode ⭐ NEW!

Look for the **green button** below video preview:

```
┌─────────────────────────────┐
│   Video Preview             │
│                             │
│  [Large Black Display]      │
│                             │
│  Zoom: [====●====] 100%     │
│                             │
│  ⚙ Open Calibration Mode   │  ← CLICK THIS!
└─────────────────────────────┘
```

**Calibration dialog opens in a new window**

---

### Step 4: Calibrate with Real-time Feedback

#### In Calibration Dialog:

1. **Find Clear Bottle Frame**
   - Click **▶ Play/Pause** button
   - Or press **Space** key
   - Pause when bottle is clearly visible

2. **Adjust Target Line**
   
   **Option A: Button Controls**
   ```
   Fine Adjust (1px):
   [↑ Up 1px]  [↓ Down 1px]
   
   Coarse Adjust (10px):
   [↑↑ Up 10px]  [↓↓ Down 10px]
   ```
   
   **Option B: Keyboard**
   - Press `↑` or `↓` for fine adjustment
   - Watch green line move on video in real-time!

3. **See Live Updates**
   - Green target line moves as you adjust
   - Red dashed lines show tolerance bounds
   - Current Y value updates instantly
   
   ```
   Current Target Line Y: 730
   ```

4. **Save & Close**
   - Click **💾 Save & Close** when aligned
   - Confirmation message appears
   - Dialog closes automatically

---

### Step 5: Run Inspection

Back in main window:

1. **Click Play** button (or press Space)
2. **Watch bottles get classified**:
   - ✅ **Green box** = PASS
   - ❌ **Red box** = REJECT_LOW
   - ⚠️ **Orange box** = NOT_DETECTED

3. **Monitor Statistics** (bottom status bar):
   ```
   FPS: 45 | Bottles: 15 | ✓ Pass: 12 | ✗ Reject: 3
   ```

---

### Step 6: Export Results

**Quick Export:**
Press `Ctrl+E`

**Menu Export:**
File → Export Results (CSV)

**Choose save location** → Results saved with timestamps

---

## 🎨 UI Layout Reference

### Main Window Structure

```
┌──────────────────────────────────────────────────────────────┐
│  Menu Bar: File  Edit  Help                                  │
├──────────────────────────────────────────────────────────────┤
│  Toolbar: [Load Video] [Play] [Stop]  Stats...               │
├───────────────────────┬──────────────────────────────────────┤
│                       │  📊 Quick Statistics                 │
│  VIDEO PREVIEW        │  Total: 15 | Pass: 12 | Reject: 3    │
│  (960x720 min)        ├──────────────────────────────────────┤
│                       │  🎯 Virtual Line Calibration         │
│  [Black display area] │  Target Line Y: [730]                │
│  Shows:               │  Fine: [↑ 1px] [↓ 1px]               │
│  - Live video         │  Coarse: [↑↑ 10px] [↓↓ 10px]         │
│  - Green target line  │  Tolerance: [±5px]                   │
│  - Red tolerance bands│  ─────────────────────────────────   │
│  - Bottle boxes       │  🔍 Detection Settings               │
│  - Track IDs          │  Confidence: [0.30]                  │
│  - Liquid levels      │  Gaussian Kernel: [5]                │
│                       │  Min Gradient: [10]                  │
│  [Zoom slider]        │  ─────────────────────────────────   │
│  [⚙ Calibration Btn]  │  🤖 YOLO Model                       │
│                       │  [best.pt] [Browse...]               │
│                       │  ─────────────────────────────────   │
│                       │  [🔄 Reset Stats] [📊 Export CSV]    │
└───────────────────────┴──────────────────────────────────────┘
│  Status Bar: FPS | Bottles | Pass | Reject                   │
└──────────────────────────────────────────────────────────────┘
```

### Calibration Dialog

```
┌─────────────────────────────────────────────────┐
│ 🎯 Calibration Mode - Real-time Adjustment     │
├─────────────────────────────────────────────────┤
│ 📋 Instructions:                                │
│ • Find clear bottle frame                      │
│ • Adjust line in real-time                     │
│ • Use arrow keys for precision                 │
│ • Save when done                               │
├─────────────────────────────────────────────────┤
│ [▶ Play/Pause (Space)]  [■ Stop]               │
├─────────────────────────────────────────────────┤
│ Current Target Line Y: 730                     │
├─────────────────────────────────────────────────┤
│ Fine Adjust (1px):                             │
│ [↑ Up 1px]  [↓ Down 1px]                       │
├─────────────────────────────────────────────────┤
│ Coarse Adjust (10px):                          │
│ [↑↑ Up 10px]  [↓↓ Down 10px]                   │
├─────────────────────────────────────────────────┤
│ ⌨️ Space=Play/Pause | ↑↓=Fine | Shift+↑↓=Coarse│
├─────────────────────────────────────────────────┤
│            [💾 Save & Close]  [Cancel]          │
└─────────────────────────────────────────────────┘
```

---

## ⌨️ Complete Shortcut Reference

### Main Window
| Key | Action |
|-----|--------|
| `Ctrl+O` | Load video file |
| `Ctrl+C` | Use camera feed |
| `Ctrl+S` | Save configuration |
| `Ctrl+L` | Load configuration |
| `Ctrl+E` | Export results CSV |
| `Ctrl+Q` | Quit application |
| `Space` | Toggle play/pause |
| `↑` | Move target line up 1px |
| `↓` | Move target line down 1px |
| `Shift+↑` | Move target line up 10px |
| `Shift+↓` | Move target line down 10px |

### Calibration Dialog
| Key | Action |
|-----|--------|
| `Space` | Play/Pause video |
| `↑` | Fine adjust up 1px |
| `↓` | Fine adjust down 1px |

---

## 💡 Pro Tips

### For Best Calibration Results

1. **Use Good Lighting**
   - Ensure bottle is well-lit
   - Minimal glare or reflections
   - Clear liquid/air interface visible

2. **Pause on Best Frame**
   - Bottle centered and upright
   - Liquid level clearly visible
   - No motion blur

3. **Start with Coarse, Then Fine**
   - Use 10px adjustments first to get close
   - Switch to 1px for precise alignment
   - Align with top of liquid meniscus

4. **Verify While Moving**
   - Unpause briefly to see line on moving bottles
   - Re-pause if needed for final tweaks
   - Line should intersect liquid surface

### Workflow Optimization

**Fast Setup (< 2 minutes):**
1. Load video → Play
2. Pause on clear bottle
3. Open calibration mode
4. Quick coarse adjustments (10px)
5. Fine-tune with arrow keys (1px)
6. Save & Close
7. Resume playback
8. Monitor results

**Product Changeover:**
1. Load saved config: `Ctrl+L` → select product config
2. Verify calibration with reference bottle
3. Minor adjustments if needed
4. Resume production

---

## 🎯 Common Tasks

### Task: Create New Product Preset

1. Load video with new product bottle
2. Calibrate target line
3. Set appropriate tolerance
4. Adjust detection settings if needed
5. **Save Configuration**: `Ctrl+S` → `config_productname.json`
6. Document settings for future reference

### Task: Quick Quality Check

1. Load current production video
2. Observe pass/reject ratio in status bar
3. If reject rate > expected:
   - Pause and check recent rejects
   - Verify calibration is still accurate
   - Adjust if lighting changed

### Task: Export Daily Report

1. After production run complete
2. Press `Ctrl+E`
3. Filename: `results_YYYYMMDD.csv`
4. Open in Excel for analysis
5. Calculate pass rate, trends, etc.

---

## 🔧 Troubleshooting Quick Fixes

### Issue: Can't See Target Line
**Solution**: Look for thin green horizontal line across video. If still not visible, increase tolerance to see red dashed bounds.

### Issue: Calibration Not Responding
**Solution**: Ensure video is paused when making fine adjustments. Use calibration mode for best responsiveness.

### Issue: Dark Theme Too Dark
**Solution**: Increase room lighting or reduce screen brightness. Theme optimized for typical industrial lighting.

### Issue: Buttons Too Small
**Solution**: Updated buttons are already 30% larger. If needed, reduce screen resolution scaling in OS settings.

---

## ✅ First-Time User Checklist

- [ ] Application launches successfully
- [ ] Dark theme displays correctly
- [ ] Video loads and plays
- [ ] Calibration mode opens when button clicked
- [ ] Arrow keys adjust target line
- [ ] Green line visible on video
- [ ] Statistics update when bottles detected
- [ ] CSV export works
- [ ] Configuration saves and reloads

---

## 🎉 You're Ready!

Your updated system features:
- ✨ Modern dark theme for reduced eye strain
- ✨ 50% larger video preview
- ✨ Real-time calibration with live feedback
- ✨ Intuitive controls and clear visual hierarchy
- ✨ Professional appearance for production use

**Happy inspecting!** 🚀

---

**Need More Help?**
- See [USER_MANUAL.md](docs/USER_MANUAL.md) for detailed operations
- See [UI_UPDATES_SUMMARY.md](UI_UPDATES_SUMMARY.md) for complete changelog
- See [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for custom model training
