"""
Quick Configuration File for Bottle Inspector
Edit these values and restart the inspector to apply changes
"""

# === VIDEO SOURCE ===
# Options:
#   - File path: 'bottle_test.mp4'
#   - Webcam: 0
#   - IP Camera: 'rtsp://username:password@ip:port/stream'
VIDEO_SOURCE = 'bottle_test.mp4'

# === TARGET LINE CONFIGURATION ===
# From your calibration (line_find.py)
TARGET_Y_PIXEL = 190  # The Y-coordinate where liquid should be
TARGET_Y_RELATIVE = 0.2639  # Relative position (0.0 = top, 1.0 = bottom)
TOLERANCE = 15  # How many pixels above/below is acceptable (+/-)

# === DETECTION ZONE ===
# The horizontal zone where bottles are inspected (left side of frame)
# Keep this SMALL so only 1-2 bottles are inspected at a time
DETECTION_ZONE_X_START = 0  # Left edge
DETECTION_ZONE_X_END = 200  # Right edge of detection zone (pixels from left)

# === BOTTLE DETECTION PARAMETERS ===
# Adjust these if bottles aren't being detected correctly
MIN_CONTOUR_AREA = 2000  # Minimum size to be considered a bottle
MAX_CONTOUR_AREA = 50000  # Maximum size (to filter out background)

# Bottle shape requirements (Height/Width ratio)
MIN_ASPECT_RATIO = 1.5  # Bottles are taller than wide
MAX_ASPECT_RATIO = 5.0  # But not extremely thin

# === TRACKING PARAMETERS ===
# How the system tracks bottles across frames
TRACKING_DISTANCE = 100  # Max pixels a bottle can move between frames
MIN_FRAMES_TO_TRACK = 3  # How many frames before making PASS/REJECT decision

# === ADVANCED: LIQUID DETECTION ===
# Threshold for detecting liquid (darker pixels)
# Lower value = requires darker pixels to be considered liquid
# Range: 0-255 (0=black, 255=white)
LIQUID_BRIGHTNESS_THRESHOLD = 150

# === DISPLAY OPTIONS ===
SHOW_DETECTION_ZONE = True  # Draw the detection zone rectangle
SHOW_TOLERANCE_LINES = True  # Draw tolerance bounds
SHOW_LIQUID_LEVEL = True  # Draw detected liquid level on each bottle
SHOW_STATISTICS = True  # Show statistics overlay
