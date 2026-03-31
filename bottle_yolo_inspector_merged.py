#!/usr/bin/env python3
"""
bottle_yolo_inspector_merged.py
YOLOv8 + ByteTrack with DETECTION MERGING for split bottles
Method 2 (Adaptive Threshold) for liquid level detection

Fixes issue where metal rod splits bottles into multiple detections
"""
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
VIDEO_PATH = 'coke.mp4'
YOLO_MODEL_PATH = 'best.pt'
TARGET_LINE_Y = 246  # Calibrated Y position for liquid level
TOLERANCE = 5  # Pixels tolerance
RELATIVE_Y = 0.3375  # Relative position (for reference)

# YOLO parameters
CONF_THRESHOLD = 0.5  # Confidence threshold for detections
IOU_THRESHOLD = 0.45  # IoU threshold for NMS
TRACKER_TYPE = "bytetrack.yaml"  # ByteTrack tracker

# Detection Merging Parameters (NEW)
ENABLE_MERGING = True  # Enable/disable detection merging
MERGE_HORIZONTAL_DISTANCE = 50  # Max horizontal distance (pixels) to merge detections
MERGE_VERTICAL_GAP_RATIO = 0.3  # Max vertical gap as ratio of total height
MIN_BBOX_HEIGHT = 50  # Minimum bounding box height to consider

# Tracking parameters
MIN_FRAMES_TO_CONFIRM = 3  # Minimum frames to track before making decision
DETECTION_ZONE_X = 0.7  # Start tracking when bottle reaches 70% of frame width
DECISION_ZONE_X = 0.85  # Make decision when bottle reaches 85% of frame width

# Visualization
SHOW_DEBUG = True  # Show preprocessing steps
SHOW_CONFIDENCE = True  # Show YOLO confidence scores
SHOW_MERGING_DEBUG = False  # Show merging visualization

# =============================================================================
# DETECTION MERGING UTILITIES
# =============================================================================
def should_merge_detections(bbox1: Tuple[float, float, float, float],
                            bbox2: Tuple[float, float, float, float]) -> bool:
    """
    Determine if two bounding boxes should be merged.
    Returns True if boxes are vertically aligned and close together.
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate centers and dimensions
    cx1 = (x1_1 + x2_1) / 2
    cx2 = (x1_2 + x2_2) / 2
    h1 = y2_1 - y1_1
    h2 = y2_2 - y1_2
    
    # Check horizontal distance (should be close in X)
    horizontal_dist = abs(cx1 - cx2)
    if horizontal_dist > MERGE_HORIZONTAL_DISTANCE:
        return False
    
    # Check vertical alignment (should have some gap or overlap in Y)
    # Gap between boxes
    if y1_2 > y2_1:  # bbox2 is below bbox1
        gap = y1_2 - y2_1
    elif y1_1 > y2_2:  # bbox1 is below bbox2
        gap = y1_1 - y2_2
    else:  # overlapping
        gap = 0
    
    # Total height if merged
    total_height = max(y2_1, y2_2) - min(y1_1, y1_2)
    
    # Should merge if gap is small relative to total height
    gap_ratio = gap / total_height if total_height > 0 else 1.0
    
    return gap_ratio < MERGE_VERTICAL_GAP_RATIO

def merge_bboxes(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """
    Merge multiple bounding boxes into one encompassing box.
    """
    if not bboxes:
        return (0, 0, 0, 0)
    
    x1_min = min(bbox[0] for bbox in bboxes)
    y1_min = min(bbox[1] for bbox in bboxes)
    x2_max = max(bbox[2] for bbox in bboxes)
    y2_max = max(bbox[3] for bbox in bboxes)
    
    return (x1_min, y1_min, x2_max, y2_max)

def merge_split_detections(track_ids: np.ndarray,
                           bboxes: np.ndarray,
                           confidences: np.ndarray) -> Tuple[List[int], List, List[float]]:
    """
    Merge detections that likely belong to the same bottle (split by obstruction).
    
    Returns:
        merged_track_ids: List of track IDs (using lowest ID from merged group)
        merged_bboxes: List of merged bounding boxes
        merged_confidences: List of average confidences
    """
    if not ENABLE_MERGING:
        return track_ids.tolist(), bboxes.tolist(), confidences.tolist()
    
    n = len(track_ids)
    if n == 0:
        return [], [], []
    
    # Create groups of detections to merge
    merged_groups = []  # List of lists of indices
    used_indices = set()
    
    for i in range(n):
        if i in used_indices:
            continue
        
        # Start a new group with this detection
        current_group = [i]
        used_indices.add(i)
        
        # Try to find other detections that should merge with this one
        for j in range(i + 1, n):
            if j in used_indices:
                continue
            
            # Check if any box in current group should merge with bbox j
            should_merge = False
            for group_idx in current_group:
                if should_merge_detections(bboxes[group_idx], bboxes[j]):
                    should_merge = True
                    break
            
            if should_merge:
                current_group.append(j)
                used_indices.add(j)
        
        merged_groups.append(current_group)
    
    # Create merged detections
    merged_track_ids = []
    merged_bboxes = []
    merged_confidences = []
    
    for group in merged_groups:
        # Use the lowest track ID from the group
        group_track_ids = [track_ids[i] for i in group]
        merged_id = min(group_track_ids)
        
        # Merge bounding boxes
        group_bboxes = [bboxes[i] for i in group]
        merged_bbox = merge_bboxes(group_bboxes)
        
        # Average confidences
        group_confidences = [confidences[i] for i in group]
        avg_confidence = np.mean(group_confidences)
        
        # Only keep if merged bbox has reasonable height
        _, y1, _, y2 = merged_bbox
        if (y2 - y1) >= MIN_BBOX_HEIGHT:
            merged_track_ids.append(merged_id)
            merged_bboxes.append(merged_bbox)
            merged_confidences.append(avg_confidence)
    
    return merged_track_ids, merged_bboxes, merged_confidences

# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class TrackedBottle:
    """Represents a tracked bottle with YOLO track ID"""
    track_id: int
    positions: deque  # History of bounding boxes (x1, y1, x2, y2)
    liquid_levels: deque  # History of detected liquid levels
    confidences: deque  # History of YOLO confidence scores
    status: str = "TRACKING"  # TRACKING, PASS, REJECT_HIGH, REJECT_LOW
    decision_made: bool = False
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], 
                 confidence: float, frame_num: int):
        self.track_id = track_id
        self.positions = deque(maxlen=30)
        self.liquid_levels = deque(maxlen=30)
        self.confidences = deque(maxlen=30)
        self.positions.append(bbox)
        self.confidences.append(confidence)
        self.status = "TRACKING"
        self.decision_made = False
        self.first_seen_frame = frame_num
        self.last_seen_frame = frame_num
    
    def update(self, bbox: Tuple[float, float, float, float], confidence: float, frame_num: int):
        """Update bottle position and confidence"""
        self.positions.append(bbox)
        self.confidences.append(confidence)
        self.last_seen_frame = frame_num
    
    def add_liquid_level(self, level_y: int):
        """Add detected liquid level"""
        self.liquid_levels.append(level_y)
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of latest bounding box"""
        if not self.positions:
            return (0, 0)
        x1, y1, x2, y2 = self.positions[-1]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)
    
    def get_latest_bbox(self) -> Tuple[int, int, int, int]:
        """Get latest bounding box as (x, y, w, h)"""
        if not self.positions:
            return (0, 0, 0, 0)
        x1, y1, x2, y2 = self.positions[-1]
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    
    def get_latest_bbox_xyxy(self) -> Tuple[int, int, int, int]:
        """Get latest bounding box as (x1, y1, x2, y2)"""
        if not self.positions:
            return (0, 0, 0, 0)
        x1, y1, x2, y2 = self.positions[-1]
        return (int(x1), int(y1), int(x2), int(y2))
    
    def frames_tracked(self) -> int:
        """Number of frames this bottle has been tracked"""
        return len(self.positions)
    
    def get_avg_confidence(self) -> float:
        """Get average YOLO confidence"""
        if not self.confidences:
            return 0.0
        return float(np.mean(list(self.confidences)))

# =============================================================================
# LIQUID LEVEL DETECTION - GRADIENT PROFILE ANALYSIS
# =============================================================================
class GradientProfileDetector:
    """
    Advanced Liquid Level Detector using Gradient Profiling and Cap Detection.
    Algorithm Spec:
    1. Grayscale + CLAHE (Contrast Boost)
    2. Gaussian Blur (Noise Reduction)
    3. Cap Detection (Top 30%)
    4. Safe Zone Definition (Below Cap, Above Rails)
    5. 1D Intensity Profile -> Smooth -> Gradient
    6. Find strongest Dark->Bright transition
    """
    def __init__(self):
        self.name = "Gradient Profile Analysis"
        
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Not used directly in this pipeline as we process per-ROI, 
        # but kept for interface compatibility
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def detect_liquid_level(self, frame: np.ndarray, 
                           bottle_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[int], str]:
        """
        Detect liquid level using gradient profile analysis.
        Returns: (liquid_y, status_hint)
        """
        x, y, w, h = bottle_bbox
        
        # Safety check
        if w <= 0 or h <= 0:
            return None, "ERROR"
            
        # Extract ROI
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None, "ERROR"

        # 1. Grayscale conversion
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # 2. CLAHE (Adaptive Histogram Equalization)
        # Evens out lighting and boosts local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # 3. Gaussian Blur (Mild)
        # Reduces high-frequency noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # 4. Cap Detection (Top Region)
        # Look for strongest horizontal edge in top 30%
        roi_h, roi_w = blurred.shape
        cap_search_limit = int(roi_h * 0.30)
        
        cap_y = 0 # Default if not found
        
        if cap_search_limit > 5:
            top_region = blurred[:cap_search_limit, :]
            # Sobel Y for horizontal edges
            sobel_y = cv2.Sobel(top_region, cv2.CV_64F, 0, 1, ksize=3)
            # Project to 1D
            top_profile = np.mean(np.abs(sobel_y), axis=1)
            # Find max gradient (strongest edge)
            if len(top_profile) > 0:
                cap_y = np.argmax(top_profile)
        
        # 5. Define Safe Liquid Zone
        # Start below cap, end above bottom rails (e.g. 65%)
        safe_start = cap_y + int(roi_h * 0.10) # 10% margin below cap
        safe_end = int(roi_h * 0.65) # Avoid bottom rails
        
        if safe_start >= safe_end:
            # Fallback if cap detection was too low or bottle is weird
            safe_start = int(roi_h * 0.20)
            safe_end = int(roi_h * 0.70)

        # Extract Safe Zone
        safe_zone = blurred[safe_start:safe_end, :]
        
        if safe_zone.size == 0:
            return None, "ERROR"

        # 6. Row Intensity Profile
        # Average intensity across columns for each row
        # Black liquid = Low Intensity, Air/Foam = High Intensity
        intensity_profile = np.mean(safe_zone, axis=1)

        # 7. Smoothing the 1D signal
        # 1D Gaussian smoothing
        sigma = 2
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0: kernel_size += 1
        kernel = cv2.getGaussianKernel(kernel_size, sigma).flatten()
        smoothed_profile = np.convolve(intensity_profile, kernel, mode='same')

        # 8. Gradient Analysis
        # We want Dark -> Bright transition (Positive Gradient)
        gradient = np.gradient(smoothed_profile)
        
        # Find strongest positive gradient
        # (Liquid is dark, Air is bright -> Intensity increases)
        max_grad_idx = np.argmax(gradient)
        max_grad_val = gradient[max_grad_idx]
        
        # Threshold for weak gradients (no clear liquid level)
        if max_grad_val < 2.0: # Heuristic threshold
             # Check if it's all dark (Full) or all bright (Empty)
             avg_val = np.mean(smoothed_profile)
             if avg_val < 50: # Arbitrary dark threshold
                 return None, "NOT_FOUND_HIGH" # Too full (liquid covers zone)
             else:
                 return None, "NOT_FOUND_LOW" # Too empty (no liquid in zone)

        # Calculate absolute Y coordinate
        liquid_level_y = y + safe_start + max_grad_idx
        
        return liquid_level_y, "FOUND"

# =============================================================================
# YOLO BOTTLE TRACKER WITH MERGING
# =============================================================================
class YOLOBottleTracker:
    """YOLO-based bottle tracker with split detection merging"""
    
    def __init__(self, model_path: str, liquid_detector: GradientProfileDetector):
        # Load YOLO model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.liquid_detector = liquid_detector
        
        # Tracked bottles dictionary: track_id -> TrackedBottle
        self.bottles: Dict[int, TrackedBottle] = {}
        self.frame_count = 0
        
        # Statistics
        self.stats = {
            'total_bottles': 0,
            'passed': 0,
            'rejected_high': 0,  # NEW: Overfilled bottles
            'rejected_low': 0,
            'currently_tracking': 0,
            'detections_merged': 0
        }
        
        print(f"✓ YOLO model loaded successfully")
        print(f"✓ Tracker: ByteTrack")
        print(f"✓ Liquid detector: {liquid_detector.name}")
        print(f"✓ Detection merging: {'ENABLED' if ENABLE_MERGING else 'DISABLED'}")
    
    def update(self, frame: np.ndarray, binary_frame: np.ndarray):
        """Update tracking with new frame"""
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Run YOLO detection with tracking
        results = self.model.track(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            tracker=TRACKER_TYPE,
            persist=True,
            verbose=False
        )
        
        # Process detections
        current_track_ids = set()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # Check if tracking IDs are available
            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
                bboxes = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                
                # MERGE SPLIT DETECTIONS (NEW)
                original_count = len(track_ids)
                merged_ids, merged_bboxes, merged_confs = merge_split_detections(
                    track_ids, bboxes, confidences
                )
                
                # Track if merging occurred
                if len(merged_ids) < original_count:
                    self.stats['detections_merged'] += (original_count - len(merged_ids))
                
                for track_id, bbox, conf in zip(merged_ids, merged_bboxes, merged_confs):
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2
                    
                    # Only track bottles in detection zone
                    if cx < width * DETECTION_ZONE_X:
                        continue
                    
                    current_track_ids.add(track_id)
                    
                    # Update or create tracked bottle
                    if track_id in self.bottles:
                        self.bottles[track_id].update(bbox, conf, self.frame_count)
                    else:
                        # New bottle detected
                        self.bottles[track_id] = TrackedBottle(
                            track_id, bbox, conf, self.frame_count
                        )
                    
                    # Detect liquid level
                    bottle = self.bottles[track_id]
                    if not bottle.decision_made:
                        x, y, w, h = bottle.get_latest_bbox()
                        # Pass the FULL frame to the new detector, not binary
                        liquid_level, status_hint = self.liquid_detector.detect_liquid_level(
                            frame, (x, y, w, h)
                        )
                        
                        if liquid_level is not None:
                            bottle.add_liquid_level(liquid_level)
                            # Debug: Show liquid level detection
                            if len(bottle.liquid_levels) <= 5:
                                print(f"  Bottle #{track_id}: Detected liquid level at Y={liquid_level} (Target={TARGET_LINE_Y})")
                        else:
                            # Handle cases where liquid is outside the ROI
                            if status_hint == "NOT_FOUND_LOW":
                                bottle.add_liquid_level(TARGET_LINE_Y + 100) # Way below target
                                if len(bottle.liquid_levels) <= 5:
                                    print(f"  Bottle #{track_id}: Liquid too low (not in ROI)")
                                    
                            elif status_hint == "NOT_FOUND_HIGH":
                                bottle.add_liquid_level(TARGET_LINE_Y - 100) # Way above target
                                if len(bottle.liquid_levels) <= 5:
                                    print(f"  Bottle #{track_id}: Liquid too high (not in ROI)")
                        
                        # Make decision if bottle has reached decision zone
                        if cx > width * DECISION_ZONE_X and bottle.frames_tracked() >= MIN_FRAMES_TO_CONFIRM:
                            self._make_decision(bottle)
        
        # Update currently tracking count
        self.stats['currently_tracking'] = len([b for b in self.bottles.values() 
                                                if not b.decision_made and 
                                                b.track_id in current_track_ids])
    
    def _make_decision(self, bottle: TrackedBottle):
        """Make PASS/REJECT decision for a bottle"""
        if len(bottle.liquid_levels) == 0:
            # No liquid level detected, skip decision
            return
        
        # Use median liquid level for robust decision
        median_level = int(np.median(list(bottle.liquid_levels)))
        deviation = median_level - TARGET_LINE_Y
        
        if -TOLERANCE <= deviation <= TOLERANCE:
            bottle.status = "PASS"
            self.stats['passed'] += 1
        elif deviation < -TOLERANCE:
            bottle.status = "REJECT_HIGH"
            self.stats['rejected_high'] += 1
        else:
            bottle.status = "REJECT_LOW"
            self.stats['rejected_low'] += 1
        
        bottle.decision_made = True
        self.stats['total_bottles'] += 1
        
        avg_conf = bottle.get_avg_confidence()
        print(f"Bottle #{bottle.track_id}: {bottle.status} "
              f"(Level: {median_level}, Target: {TARGET_LINE_Y}, Dev: {deviation:+d}, "
              f"Conf: {avg_conf:.2f})")
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking visualization"""
        vis_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw target line (GREEN)
        cv2.line(vis_frame, (0, TARGET_LINE_Y), (width, TARGET_LINE_Y), (0, 255, 0), 2)
        cv2.putText(vis_frame, "TARGET", (width - 100, TARGET_LINE_Y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw tolerance bounds (RED dashed)
        for y in [TARGET_LINE_Y - TOLERANCE, TARGET_LINE_Y + TOLERANCE]:
            for i in range(0, width, 20):
                cv2.line(vis_frame, (i, y), (min(i+10, width), y), (0, 0, 255), 1)
        
        # Draw detection zone (CYAN)
        detection_x = int(width * DETECTION_ZONE_X)
        cv2.line(vis_frame, (detection_x, 0), (detection_x, height), (255, 255, 0), 2)
        cv2.putText(vis_frame, "DETECT", (detection_x + 5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw decision zone (MAGENTA)
        decision_x = int(width * DECISION_ZONE_X)
        cv2.line(vis_frame, (decision_x, 0), (decision_x, height), (255, 0, 255), 2)
        cv2.putText(vis_frame, "DECIDE", (decision_x + 5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Draw tracked bottles
        for bottle in self.bottles.values():
            x1, y1, x2, y2 = bottle.get_latest_bbox_xyxy()
            center = bottle.get_center()
            
            # Color based on status
            if bottle.status == "PASS":
                color = (0, 255, 0)  # Green
                thickness = 3
            elif bottle.status == "REJECT_HIGH":
                color = (255, 0, 255)  # Magenta - overfilled
                thickness = 3
            elif bottle.status == "REJECT_LOW":
                color = (0, 0, 255)  # Red - underfilled
                thickness = 3
            else:
                color = (255, 255, 255)  # White (tracking)
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw liquid level if detected
            if len(bottle.liquid_levels) > 0:
                liquid_y = int(np.median(list(bottle.liquid_levels)))
                cv2.line(vis_frame, (x1, liquid_y), (x2, liquid_y), color, 2)
                
                # Draw deviation indicator
                deviation = liquid_y - TARGET_LINE_Y
                dev_text = f"{deviation:+d}px"
                cv2.putText(vis_frame, dev_text, (x2 + 5, liquid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw ID, status, and confidence
            avg_conf = bottle.get_avg_confidence()
            if SHOW_CONFIDENCE:
                label = f"ID:{bottle.track_id} {bottle.status} ({avg_conf:.2f})"
            else:
                label = f"ID:{bottle.track_id} {bottle.status}"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_frame, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
            cv2.putText(vis_frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw statistics panel
        self._draw_stats_panel(vis_frame)
        
        return vis_frame
    
    def _draw_stats_panel(self, frame: np.ndarray):
        """Draw statistics panel on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent background
        overlay = frame.copy()
        panel_height = 145  # Increased for merge stats
        cv2.rectangle(overlay, (10, 10), (450, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw statistics
        y_offset = 35
        line_height = 25
        
        stats_lines = [
            f"Method: {self.liquid_detector.name} + YOLO + Merging",
            f"Total: {self.stats['total_bottles']} | Pass: {self.stats['passed']}",
            f"Reject High: {self.stats['rejected_high']} | Reject Low: {self.stats['rejected_low']}",
            f"Tracking: {self.stats['currently_tracking']} | Merged: {self.stats['detections_merged']}",
            f"Target: {TARGET_LINE_Y}px ±{TOLERANCE}px"
        ]
        
        for i, line in enumerate(stats_lines):
            cv2.putText(frame, line, (15, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    print("="*70)
    print("YOLO + BYTETRACK + DETECTION MERGING")
    print("="*70)
    print(f"YOLO Model: {YOLO_MODEL_PATH}")
    print(f"Tracker: ByteTrack")
    print(f"Liquid Detection: Gradient Profile Analysis")
    print(f"Detection Merging: {'ENABLED' if ENABLE_MERGING else 'DISABLED'}")
    print(f"Target Line Y: {TARGET_LINE_Y} pixels (Relative: {RELATIVE_Y:.4f})")
    print(f"Tolerance: ±{TOLERANCE} pixels")
    print(f"Detection Zone: {DETECTION_ZONE_X*100:.0f}% of frame width")
    print(f"Decision Zone: {DECISION_ZONE_X*100:.0f}% of frame width")
    print("="*70)
    print("Controls: [SPACE] Pause/Resume | [Q] Quit | [R] Reset Stats")
    print("="*70)
    
    # Initialize components
    liquid_detector = GradientProfileDetector()
    
    try:
        tracker = YOLOBottleTracker(YOLO_MODEL_PATH, liquid_detector)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure 'best.pt' is in the current directory")
        return
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return
    
    # Get video FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 24  # Default fallback
    frame_delay = 1.0 / video_fps
    
    print(f"Video FPS: {video_fps:.2f} | Frame delay: {frame_delay*1000:.2f}ms")
    print("="*70)
    
    paused = False
    
    # For FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0
    
    while True:
        frame_start_time = time.time()
        
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                # End of video - pause on last frame
                print("\nEnd of video reached")
                break
            
            # Preprocess for liquid level detection
            # Note: GradientProfileDetector does per-ROI preprocessing, 
            # so we just pass the frame or a dummy binary frame if needed.
            # But YOLOBottleTracker.update expects a binary frame for the old detector.
            # We updated YOLOBottleTracker to pass the FULL frame to detect_liquid_level.
            # But we still need to pass something as the second arg to update().
            # Let's pass a dummy or the grayscale frame.
            binary_frame = liquid_detector.preprocess(frame)
            
            # Update tracker
            tracker.update(frame, binary_frame)
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                fps_display = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
        
        # Visualize
        vis_frame = tracker.visualize(frame)
        
        # Add FPS counter
        processing_time = (time.time() - frame_start_time) * 1000
        cv2.putText(vis_frame, f"FPS: {fps_display:.1f} | Process: {processing_time:.1f}ms",
                   (vis_frame.shape[1]-300, vis_frame.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show frames
        cv2.imshow('YOLO Bottle Inspector (Merged)', vis_frame)
        
        if SHOW_DEBUG:
            # Show the preprocessed frame (grayscale)
            cv2.imshow('Preprocessed (Gradient Analysis)', binary_frame)
        
        # Calculate wait time to maintain video FPS
        elapsed = time.time() - frame_start_time
        wait_time = max(1, int((frame_delay - elapsed) * 1000))
        
        # Handle key presses
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord(' '):  # Pause/Resume
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Reset statistics
            tracker.stats = {
                'total_bottles': 0,
                'passed': 0,
                'rejected_high': 0,
                'rejected_low': 0,
                'currently_tracking': 0,
                'detections_merged': 0
            }
            print("Statistics reset!")
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Total Bottles Processed: {tracker.stats['total_bottles']}")
    print(f"Passed: {tracker.stats['passed']} "
          f"({tracker.stats['passed']/max(tracker.stats['total_bottles'],1)*100:.1f}%)")
    print(f"Rejected (High): {tracker.stats['rejected_high']} "
          f"({tracker.stats['rejected_high']/max(tracker.stats['total_bottles'],1)*100:.1f}%)")
    print(f"Rejected (Low): {tracker.stats['rejected_low']} "
          f"({tracker.stats['rejected_low']/max(tracker.stats['total_bottles'],1)*100:.1f}%)")
    print(f"Split Detections Merged: {tracker.stats['detections_merged']}")
    print("="*70)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
