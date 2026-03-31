import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
VIDEO_PATH = 'coke.mp4'
TARGET_LINE_Y = 253  # Calibrated Y position
TOLERANCE = 5  # Pixels tolerance
RELATIVE_Y = 0.3514  # Relative position (for reference)

# Detection Methods - Switch between these
DETECTION_METHOD = "METHOD_2"  # Options: METHOD_1, METHOD_2, METHOD_3

# Tracking parameters
MIN_BOTTLE_AREA = 2000  # Minimum area to consider as bottle
MAX_BOTTLE_AREA = 50000  # Maximum area to consider as bottle
DETECTION_ZONE_X = 0.7  # Detect bottles when they reach 70% of frame width
MAX_TRACKING_DISTANCE = 100  # Max pixels to associate same bottle between frames
MIN_FRAMES_TO_CONFIRM = 3  # Minimum frames to track before making decision

# Visualization
SHOW_DEBUG = True  # Show preprocessing steps
SHOW_CONTOURS = True  # Show detected contours

# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class Bottle:
    id: int
    positions: deque  # History of (x, y, width, height)
    liquid_levels: deque  # History of detected liquid levels
    status: str = "TRACKING"  # TRACKING, PASS, REJECT_HIGH, REJECT_LOW
    decision_made: bool = False
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    
    def __init__(self, bottle_id: int, x: int, y: int, w: int, h: int, frame_num: int):
        self.id = bottle_id
        self.positions = deque(maxlen=30)
        self.liquid_levels = deque(maxlen=30)
        self.positions.append((x, y, w, h))
        self.status = "TRACKING"
        self.decision_made = False
        self.first_seen_frame = frame_num
        self.last_seen_frame = frame_num
    
    def update_position(self, x: int, y: int, w: int, h: int, frame_num: int):
        self.positions.append((x, y, w, h))
        self.last_seen_frame = frame_num
    
    def add_liquid_level(self, level_y: int):
        self.liquid_levels.append(level_y)
    
    def get_center(self) -> Tuple[int, int]:
        if not self.positions:
            return (0, 0)
        x, y, w, h = self.positions[-1]
        return (x + w // 2, y + h // 2)
    
    def get_latest_position(self) -> Tuple[int, int, int, int]:
        if not self.positions:
            return (0, 0, 0, 0)
        return self.positions[-1]
    
    def frames_tracked(self) -> int:
        return len(self.positions)

# =============================================================================
# DETECTION METHODS
# =============================================================================

class DetectionMethod:
    """Base class for detection methods"""
    
    def __init__(self, name: str):
        self.name = name
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def detect_liquid_level(self, frame: np.ndarray, bottle_roi: Tuple[int, int, int, int]) -> Optional[int]:
        raise NotImplementedError


class Method1_OtsuGaussian(DetectionMethod):
    """
    METHOD 1: Otsu Thresholding with Gaussian Blur
    Best for: Clear bottles with good contrast
    """
    
    def __init__(self):
        super().__init__("Otsu + Gaussian (Recommended)")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def detect_liquid_level(self, binary_frame: np.ndarray, bottle_roi: Tuple[int, int, int, int]) -> Optional[int]:
        x, y, w, h = bottle_roi
        
        # Extract ROI
        roi = binary_frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        # Find the liquid surface by scanning from top to bottom
        # Looking for the transition from empty (white/black) to liquid (black/white)
        threshold_ratio = 0.4  # At least 40% of row should be filled
        
        for i in range(roi.shape[0]):
            row = roi[i, :]
            white_pixels = np.sum(row == 255)
            fill_ratio = white_pixels / row.shape[0]
            
            # Found liquid surface
            if fill_ratio > threshold_ratio:
                return y + i
        
        return None


class Method2_AdaptiveThreshold(DetectionMethod):
    """
    METHOD 2: Adaptive Thresholding
    Best for: Varying lighting conditions across the frame
    """
    
    def __init__(self):
        super().__init__("Adaptive Threshold")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        return adaptive
    
    def detect_liquid_level(self, binary_frame: np.ndarray, bottle_roi: Tuple[int, int, int, int]) -> Optional[int]:
        x, y, w, h = bottle_roi
        roi = binary_frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        threshold_ratio = 0.4
        
        for i in range(roi.shape[0]):
            row = roi[i, :]
            white_pixels = np.sum(row == 255)
            fill_ratio = white_pixels / row.shape[0]
            
            if fill_ratio > threshold_ratio:
                return y + i
        
        return None


class Method3_EdgeBased(DetectionMethod):
    """
    METHOD 3: Edge Detection with Morphological Operations
    Best for: Clear liquid boundaries, challenging lighting
    """
    
    def __init__(self):
        super().__init__("Edge Detection + Morphology")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological closing to fill gaps
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def detect_liquid_level(self, binary_frame: np.ndarray, bottle_roi: Tuple[int, int, int, int]) -> Optional[int]:
        x, y, w, h = bottle_roi
        roi = binary_frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        # Find edges
        edges = cv2.Canny(roi, 50, 150)
        
        # Find horizontal lines (liquid surface)
        # Sum along horizontal axis
        horizontal_projection = np.sum(edges, axis=1)
        
        # Find the strongest horizontal line in upper half of bottle
        upper_half = horizontal_projection[:h//2]
        if len(upper_half) == 0:
            return None
        
        max_idx = np.argmax(upper_half)
        
        if horizontal_projection[max_idx] > w * 0.3:  # Strong horizontal line detected
            return y + max_idx
        
        # Fallback: threshold method
        threshold_ratio = 0.4
        for i in range(roi.shape[0]):
            row = roi[i, :]
            white_pixels = np.sum(row == 255)
            fill_ratio = white_pixels / row.shape[0]
            
            if fill_ratio > threshold_ratio:
                return y + i
        
        return None


# =============================================================================
# BOTTLE TRACKER
# =============================================================================

class BottleTracker:
    def __init__(self, detection_method: DetectionMethod):
        self.bottles: List[Bottle] = []
        self.next_id = 1
        self.detection_method = detection_method
        self.frame_count = 0
        
        # Statistics
        self.stats = {
            'total_bottles': 0,
            'passed': 0,
            'rejected_low': 0
        }
    
    def find_bottles(self, binary_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find bottle contours in the frame"""
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bottles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if MIN_BOTTLE_AREA < area < MAX_BOTTLE_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (bottles are taller than wide)
                aspect_ratio = h / w if w > 0 else 0
                if 1.5 < aspect_ratio < 6:  # Reasonable bottle proportions
                    bottles.append((x, y, w, h))
        
        return bottles
    
    def match_bottle(self, x: int, y: int, w: int, h: int) -> Optional[Bottle]:
        """Match detected bottle to existing tracked bottle"""
        center_x = x + w // 2
        center_y = y + h // 2
        
        best_match = None
        min_distance = MAX_TRACKING_DISTANCE
        
        for bottle in self.bottles:
            if bottle.decision_made:
                continue
            
            bottle_center = bottle.get_center()
            distance = np.sqrt((center_x - bottle_center[0])**2 + (center_y - bottle_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                best_match = bottle
        
        return best_match
    
    def update(self, frame: np.ndarray, binary_frame: np.ndarray):
        """Update tracking for current frame"""
        self.frame_count += 1
        detected_bottles = self.find_bottles(binary_frame)
        
        # Match or create bottles
        matched_ids = set()
        
        for x, y, w, h in detected_bottles:
            center_x = x + w // 2
            
            # Only track bottles in detection zone
            if center_x < frame.shape[1] * DETECTION_ZONE_X:
                continue
            
            matched_bottle = self.match_bottle(x, y, w, h)
            
            if matched_bottle:
                matched_bottle.update_position(x, y, w, h, self.frame_count)
                matched_ids.add(matched_bottle.id)
            else:
                # Create new bottle
                new_bottle = Bottle(self.next_id, x, y, w, h, self.frame_count)
                self.bottles.append(new_bottle)
                matched_ids.add(self.next_id)
                self.next_id += 1
        
        # Detect liquid levels and make decisions
        for bottle in self.bottles:
            if bottle.decision_made:
                continue
            
            if bottle.id not in matched_ids:
                # Bottle not detected this frame
                if self.frame_count - bottle.last_seen_frame > 10:
                    # Lost tracking - make decision if we have enough data
                    if bottle.frames_tracked() >= MIN_FRAMES_TO_CONFIRM:
                        self._make_decision(bottle)
                continue
            
            # Detect liquid level
            x, y, w, h = bottle.get_latest_position()
            liquid_level = self.detection_method.detect_liquid_level(binary_frame, (x, y, w, h))
            
            if liquid_level is not None:
                bottle.add_liquid_level(liquid_level)
            
            # Make decision if bottle has passed detection zone and we have enough data
            center_x = x + w // 2
            if center_x > frame.shape[1] * 0.85 and bottle.frames_tracked() >= MIN_FRAMES_TO_CONFIRM:
                self._make_decision(bottle)
    
    def _make_decision(self, bottle: Bottle):
        """Make PASS/REJECT decision for a bottle"""
        if len(bottle.liquid_levels) == 0:
            return
        
        # Use median liquid level for robust decision
        median_level = int(np.median(list(bottle.liquid_levels)))
        deviation = median_level - TARGET_LINE_Y
        
        # PASS if liquid is at or above target line (within tolerance below)
        # REJECT only if liquid is below the target line by more than tolerance
        if deviation <= TOLERANCE:
            bottle.status = "PASS"
            self.stats['passed'] += 1
        else:
            bottle.status = "REJECT_LOW"
            self.stats['rejected_low'] += 1
        
        bottle.decision_made = True
        self.stats['total_bottles'] += 1
        
        print(f"Bottle #{bottle.id}: {bottle.status} (Level: {median_level}, Target: {TARGET_LINE_Y}, Dev: {deviation:+d})")
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking visualization"""
        vis_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw target line (GREEN)
        cv2.line(vis_frame, (0, TARGET_LINE_Y), (width, TARGET_LINE_Y), (0, 255, 0), 2)
        
        # Draw tolerance bounds (RED dashed)
        for y in [TARGET_LINE_Y - TOLERANCE, TARGET_LINE_Y + TOLERANCE]:
            for i in range(0, width, 20):
                cv2.line(vis_frame, (i, y), (min(i+10, width), y), (0, 0, 255), 1)
        
        # Draw detection zone (CYAN)
        detection_x = int(width * DETECTION_ZONE_X)
        cv2.line(vis_frame, (detection_x, 0), (detection_x, height), (255, 255, 0), 1)
        
        # Draw tracked bottles
        for bottle in self.bottles:
            x, y, w, h = bottle.get_latest_position()
            center = bottle.get_center()
            
            # Color based on status
            if bottle.status == "PASS":
                color = (0, 255, 0)  # Green
            elif bottle.status == "REJECT_LOW":
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 255)  # White (tracking)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw liquid level if detected
            if len(bottle.liquid_levels) > 0:
                liquid_y = int(np.median(list(bottle.liquid_levels)))
                cv2.line(vis_frame, (x, liquid_y), (x+w, liquid_y), color, 2)
            
            # Draw ID and status
            label = f"#{bottle.id} {bottle.status}"
            cv2.putText(vis_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw statistics
        y_offset = 30
        cv2.putText(vis_frame, f"Method: {self.detection_method.name}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(vis_frame, f"Total: {self.stats['total_bottles']} | Pass: {self.stats['passed']} | Reject: {self.stats['rejected_low']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    # Select detection method
    if DETECTION_METHOD == "METHOD_1":
        method = Method1_OtsuGaussian()
    elif DETECTION_METHOD == "METHOD_2":
        method = Method2_AdaptiveThreshold()
    elif DETECTION_METHOD == "METHOD_3":
        method = Method3_EdgeBased()
    else:
        print(f"Unknown method: {DETECTION_METHOD}, using METHOD_1")
        method = Method1_OtsuGaussian()
    
    print("="*70)
    print("BOTTLE LIQUID LEVEL DETECTION SYSTEM")
    print("="*70)
    print(f"Detection Method: {method.name}")
    print(f"Target Line Y: {TARGET_LINE_Y} pixels (Relative: {RELATIVE_Y:.4f})")
    print(f"Tolerance: ±{TOLERANCE} pixels")
    print(f"Detection Zone: {DETECTION_ZONE_X*100:.0f}% of frame width")
    print("="*70)
    print("Controls: [SPACE] Pause/Resume | [Q] Quit | [R] Reset Stats | [1/2/3] Switch Method")
    print("="*70)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return
    
    # Get video FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 24  # Default fallback
    frame_delay = 1.0 / video_fps  # Time between frames in seconds
    
    print(f"Video FPS: {video_fps:.2f} | Frame delay: {frame_delay*1000:.2f}ms")
    print("="*70)
    
    tracker = BottleTracker(method)
    paused = False
    
    # For FPS calculation and timing
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0
    last_frame_time = time.time()
    
    while True:
        # Calculate time to maintain video FPS
        frame_start_time = time.time()
        
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Preprocess frame
            binary_frame = method.preprocess(frame)
            
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
        
        # Add FPS counter and processing time info
        processing_time = (time.time() - frame_start_time) * 1000  # in ms
        cv2.putText(vis_frame, f"FPS: {fps_display:.1f} | Process: {processing_time:.1f}ms", 
                   (vis_frame.shape[1]-300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show frames
        cv2.imshow('Bottle Detection', vis_frame)
        
        if SHOW_DEBUG:
            cv2.imshow('Preprocessed', binary_frame)
        
        # Calculate wait time to maintain video FPS
        elapsed = time.time() - frame_start_time
        wait_time = max(1, int((frame_delay - elapsed) * 1000))  # Convert to milliseconds
        
        # Handle key presses
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord(' '):  # Pause/Resume
            paused = not paused
        elif key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Reset statistics
            tracker.stats = {'total_bottles': 0, 'passed': 0, 'rejected_low': 0}
            print("Statistics reset!")
        elif key == ord('1'):  # Switch to Method 1
            method = Method1_OtsuGaussian()
            tracker.detection_method = method
            print(f"Switched to: {method.name}")
        elif key == ord('2'):  # Switch to Method 2
            method = Method2_AdaptiveThreshold()
            tracker.detection_method = method
            print(f"Switched to: {method.name}")
        elif key == ord('3'):  # Switch to Method 3
            method = Method3_EdgeBased()
            tracker.detection_method = method
            print(f"Switched to: {method.name}")
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Total Bottles Processed: {tracker.stats['total_bottles']}")
    print(f"Passed: {tracker.stats['passed']} ({tracker.stats['passed']/max(tracker.stats['total_bottles'],1)*100:.1f}%)")
    print(f"Rejected (Low): {tracker.stats['rejected_low']} ({tracker.stats['rejected_low']/max(tracker.stats['total_bottles'],1)*100:.1f}%)")
    print("="*70)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()