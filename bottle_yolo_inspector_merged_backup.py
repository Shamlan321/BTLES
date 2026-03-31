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
    status: str = "TRACKING"  # TRACKING, PASS, REJECT_LOW
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
# LIQUID LEVEL DETECTION - METHOD 2 (ADAPTIVE THRESHOLD)
# =============================================================================
class AdaptiveThresholdDetector:
    """
    METHOD 2: Adaptive Thresholding
    Best for: Varying lighting conditions across the frame
    """
    
    def __init__(self):
        self.name = "Adaptive Threshold"
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for liquid level detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
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
                        liquid_level = self.liquid_detector.detect_liquid_level(
                            binary_frame, (x, y, w, h)
                        )
                        
                        if liquid_level is not None:
                            bottle.add_liquid_level(liquid_level)
                            # Debug: Show liquid level detection
                            if len(bottle.liquid_levels) <= 5:  # Only show first few detections
                                print(f"  Bottle #{track_id}: Detected liquid level at Y={liquid_level} (Target={TARGET_LINE_Y})")
                        
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
        
        # Decision logic:
        # PASS if liquid is at or above target line (within tolerance below)
        # REJECT if liquid is below target line by more than tolerance
        if deviation <= TOLERANCE:
            bottle.status = "PASS"
            self.stats['passed'] += 1
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
            elif bottle.status == "REJECT_LOW":
                color = (0, 0, 255)  # Red
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
            f"Total: {self.stats['total_bottles']} | Pass: {self.stats['passed']} | Reject: {self.stats['rejected_low']}",
            f"Tracking: {self.stats['currently_tracking']} bottles",
            f"Merged: {self.stats['detections_merged']} split detections",  # NEW
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
    print(f"Liquid Detection: Adaptive Threshold (Method 2)")
    print(f"Detection Merging: {'ENABLED' if ENABLE_MERGING else 'DISABLED'}")
    print(f"Target Line Y: {TARGET_LINE_Y} pixels (Relative: {RELATIVE_Y:.4f})")
    print(f"Tolerance: ±{TOLERANCE} pixels")
    print(f"Detection Zone: {DETECTION_ZONE_X*100:.0f}% of frame width")
    print(f"Decision Zone: {DECISION_ZONE_X*100:.0f}% of frame width")
    print("="*70)
    print("Controls: [SPACE] Pause/Resume | [Q] Quit | [R] Reset Stats")
    print("="*70)
    
    # Initialize components
    liquid_detector = AdaptiveThresholdDetector()
    
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
            cv2.imshow('Preprocessed (Method 2)', binary_frame)
        
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
    print(f"Rejected (Low): {tracker.stats['rejected_low']} "
          f"({tracker.stats['rejected_low']/max(tracker.stats['total_bottles'],1)*100:.1f}%)")
    print(f"Split Detections Merged: {tracker.stats['detections_merged']}")
    print("="*70)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
