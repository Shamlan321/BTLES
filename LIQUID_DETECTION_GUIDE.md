# Understanding Liquid Level Detection

## Coordinate System

In image coordinates:
- **Y=0** is at the **TOP** of the image
- **Y increases DOWNWARD**

Example for frame height = 720px:
```
Y=0   ← Top of image
Y=100
Y=200
Y=300 ← Liquid level here
Y=400
Y=720 ← Bottom of image
```

## Your Configuration

```python
TARGET_LINE_Y = 253  # This is the target liquid level
TOLERANCE = 5        # ±5 pixels acceptable
```

## Decision Logic

```python
deviation = detected_level - TARGET_LINE_Y

if deviation <= TOLERANCE:
    PASS  # Liquid is good
else:
    REJECT_LOW  # Liquid is too low
```

## Examples

### Case 1: Liquid ABOVE target (Good - should PASS)
```
Detected Y = 100
Target Y = 253
Deviation = 100 - 253 = -153

-153 <= 5 → TRUE → **PASS** ✓
```

### Case 2: Liquid AT target (Good - should PASS)
```
Detected Y = 253
Target Y = 253
Deviation = 253 - 253 = 0

0 <= 5 → TRUE → **PASS** ✓
```

### Case 3: Liquid SLIGHTLY below target (Acceptable - should PASS)
```
Detected Y = 256
Target Y = 253
Deviation = 256 - 253 = +3

+3 <= 5 → TRUE → **PASS** ✓
```

### Case 4: Liquid TOO FAR below target (Bad - should REJECT)
```
Detected Y = 300
Target Y = 253
Deviation = 300 - 253 = +47

+47 <= 5 → FALSE → **REJECT_LOW** ✓
```

## Your Console Output Analysis

From your run:
```
Bottle #7: PASS (Level: 86, Target: 253, Dev: -167)
Bottle #28: PASS (Level: 87, Target: 253, Dev: -166)
```

**Analysis:**
- Detected liquid at Y=86, Y=87 (very high in frame)
- Target is at Y=253 (lower in frame)
- Deviation is negative (-167, -166)
- **This means liquid is WELL ABOVE target line**
- **Status: PASS is CORRECT!** ✓

## Problem Diagnosis

If you're seeing bottles with **low liquid** being marked as **PASS**, here's what might be happening:

### Issue 1: Wrong Detection
The liquid detector might be finding the **TOP of the bottle** or **cap** instead of the **liquid surface**.

**Solution:** The detector scans top-to-bottom looking for where pixels become "filled". This should find the liquid surface, but might find the bottle top if that's more visible.

### Issue 2: Target Line is Too Low
Your `TARGET_LINE_Y = 253` might be positioned too low in the frame, so even low-liquid bottles appear above it.

**Solution:** Adjust `TARGET_LINE_Y` to match where the liquid SHOULD be in properly filled bottles.

### Issue 3: Inverted Binary Image
The adaptive threshold might be inverting what should be white/black.

**Check:** Look at the "Preprocessed (Method 2)" window:
- Liquid should appear WHITE
- Empty space should appear BLACK

If inverted, the detector will find wrong surface.

## How to Fix

### Step 1: Verify Target Line Position
Look at the video with green line overlay:
- The green line should be at the **correct liquid level** for good bottles
- If it's too high or too low, adjust `TARGET_LINE_Y`

### Step 2: Check Binary Preprocessing
Look at the "Preprocessed (Method 2)" window:
- Liquid-filled areas should be **WHITE**
- Empty areas should be **BLACK**
- If inverted, we need to invert the threshold

### Step 3: Verify Detection
Run with debug output (already added) to see:
```
Bottle #X: Detected liquid level at Y=XXX (Target=253)
```

Compare detected Y values for:
- Full bottles (should be low Y, like 80-120)
- Low bottles (should be high Y, like 300-400)

### Step 4: Adjust Logic if Needed

If the issue is that low bottles have **higher Y values** but are still passing:

**Current:**
```python
if deviation <= TOLERANCE:  # Passes for deviation from -∞ to +5
    PASS
```

**Should be (if you want stricter):**
```python
if -TOLERANCE <= deviation <= TOLERANCE:  # Passes only within ±tolerance
    PASS
```

## Testing Steps

1. **Run the updated script:**
   ```bash
   python bottle_yolo_inspector_merged.py
   ```

2. **Watch console output** for debug messages like:
   ```
   Bottle #1: Detected liquid level at Y=XXX (Target=253)
   ```

3. **Check if detected levels match reality:**
   - Full bottles: Low Y values (< TARGET)
   - Low bottles: High Y values (> TARGET + TOLERANCE)

4. **Verify in video:**
   - Pause and check if horizontal line is at correct liquid level
   - Compare to green target line

5. **Adjust TARGET_LINE_Y if needed**

Let me know what values you see in the debug output!
