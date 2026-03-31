import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
VIDEO_PATH = 'sprite.mp4'
YOLO_MODEL_PATH = 'sprite_yolov8s.pt'  # Your trained YOLOv8s model

# Detection line configuration (vertical line in middle of frame)
DETECTION_LINE_POSITION = 0.5  # 50% of frame width (middle)

# Liquid level detection parameters
TARGET_LINE_Y = 730  # Your calibrated target line (from calibration)
TOLERANCE = 5  # Pixels tolerance for pass/reject
MIN_GRADIENT_THRESHOLD = 10  # Minimum gradient to consider valid transition

# Gaussian blur parameters (from your analysis)
GAUSSIAN_KERNEL_SIZE = 5  # Best performing kernel size from your tests

# Tracking parameters
CONF_THRESHOLD = 0.3  # YOLO confidence threshold (lowered for better detection)
IOU_THRESHOLD = 0.45  # IoU threshold for ByteTrack

# Display settings
DISPLAY_SCALE = 0.7  # Scale factor for display (0.7 = 70% of original size)
# Set to 1.0 for full size, or adjust based on your screen (e.g., 0.5 for 50%)

# Visualization
SHOW_DEBUG = True  # Show cropped ROI and processed image
SHOW_TRACKING_IDS = True  # Show ByteTrack IDs on bottles

# =============================================================================
# LIQUID LEVEL DETECTION (From your variations_creator.py)
# =============================================================================

def detect_liquid_level(gray_img, min_gradient_threshold=10):
    """
    Detect liquid level in grayscale image of liquid/air interface
    
    Args:
        gray_img: Grayscale image showing liquid/air interface
        min_gradient_threshold: Minimum gradient value to be considered valid
    
    Returns:
        liquid_level: Y position of detected liquid level (or None if not found)
        gradient_info: Dictionary with gradient analysis details
    """
    if gray_img.size == 0 or gray_img.shape[0] < 2:
        return None, {}
    
    # Calculate vertical intensity profile (average across width)
    intensity_profile = np.mean(gray_img, axis=1)
    
    # Find the steepest gradient (liquid/air transition)
    gradients = np.diff(intensity_profile)
    abs_gradients = np.abs(gradients)
    
    if len(abs_gradients) == 0:
        return None, {}
    
    # Find strongest gradient
    max_gradient_idx = np.argmax(abs_gradients)
    max_gradient_value = abs_gradients[max_gradient_idx]
    
    # Verify we have a significant transition (not just noise)
    if max_gradient_value < min_gradient_threshold:
        # Fallback: Look for consistent dark region (black liquid)
        dark_threshold = np.mean(intensity_profile) * 0.7
        liquid_region = np.where(intensity_profile < dark_threshold)[0]
        
        if len(liquid_region) > 0:
            # Liquid region found - use top of liquid region
            liquid_level = liquid_region[0]
        else:
            # No clear liquid region - return None
            return None, {
                'intensity_profile': intensity_profile,
                'gradients': gradients,
                'max_gradient_idx': max_gradient_idx,
                'max_gradient_value': max_gradient_value
            }
    else:
        # Convert to position within image
        liquid_level = max_gradient_idx + 1
    
    return liquid_level, {
        'intensity_profile': intensity_profile,
        'gradients': gradients,
        'max_gradient_idx': max_gradient_idx,
        'max_gradient_value': max_gradient_value
    }


def process_roi_with_gaussian(roi_img):
    """
    Process ROI (Region of Interest) with Gaussian blur and detect liquid level
    
    Args:
        roi_img: BGR image of the liquid region (cropped from YOLO bbox)
    
    Returns:
        liquid_level_relative: Liquid level position relative to ROI top (pixels)
        processed_img: Processed grayscale image for visualization
        gradient_info: Gradient analysis information
    """
    if roi_img.size == 0:
        return None, None, {}
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur (your best performing method)
    blurred = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)
    
    # Detect liquid level
    liquid_level_relative, gradient_info = detect_liquid_level(
        blurred, 
        min_gradient_threshold=MIN_GRADIENT_THRESHOLD
    )
    
    return liquid_level_relative, blurred, gradient_info


# =============================================================================
# BOTTLE TRACKER WITH BYTETRACK
# =============================================================================

class BottleTracker:
    def __init__(self, model_path, detection_line_x, target_line_y, tolerance):
        """
        Initialize bottle tracker with YOLO and ByteTrack
        
        Args:
            model_path: Path to trained YOLO model (best.pt)
            detection_line_x: X position of detection line (pixel coordinate)
            target_line_y: Target liquid level Y position (pixel coordinate)
            tolerance: Tolerance for pass/reject decision (pixels)
        """
        # Load YOLO model with ByteTrack tracker
        self.model = YOLO(model_path)
        
        self.detection_line_x = detection_line_x
        self.target_line_y = target_line_y
        self.tolerance = tolerance
        
        # Tracking state
        self.processed_bottles = set()  # Set of track IDs that have been processed
        self.bottle_results = {}  # Store results for each bottle ID
        
        # Statistics
        self.stats = {
            'total_bottles': 0,
            'passed': 0,
            'rejected_low': 0,
            'not_detected': 0
        }
    
    def has_crossed_line(self, bbox, track_id):
        """
        Check if bottle has crossed the detection line and hasn't been processed yet
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: ByteTrack ID of the bottle
        
        Returns:
            True if bottle just crossed the line, False otherwise
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        
        # Check if center has crossed the line and hasn't been processed
        if center_x >= self.detection_line_x and track_id not in self.processed_bottles:
            return True
        return False
    
    def process_bottle(self, frame, bbox, track_id):
        """
        Process a bottle that has crossed the detection line
        
        Args:
            frame: Full video frame
            bbox: Bounding box [x1, y1, x2, y2] from YOLO
            track_id: ByteTrack ID
        
        Returns:
            result_dict: Dictionary with processing results
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop the ROI from YOLO bounding box
        roi = frame[y1:y2, x1:x2]
        
        # Process ROI with Gaussian blur method
        liquid_level_relative, processed_img, gradient_info = process_roi_with_gaussian(roi)
        
        # Calculate absolute liquid level position in frame
        liquid_level_absolute = None
        if liquid_level_relative is not None:
            liquid_level_absolute = y1 + liquid_level_relative
        
        # Make pass/reject decision
        status = self._classify_bottle(liquid_level_absolute)
        
        # Store results
        result = {
            'track_id': track_id,
            'bbox': (x1, y1, x2, y2),
            'roi': roi,
            'processed_img': processed_img,
            'liquid_level_relative': liquid_level_relative,
            'liquid_level_absolute': liquid_level_absolute,
            'status': status,
            'gradient_info': gradient_info
        }
        
        self.bottle_results[track_id] = result
        self.processed_bottles.add(track_id)
        
        # Update statistics
        self.stats['total_bottles'] += 1
        if status == 'PASS':
            self.stats['passed'] += 1
        elif status == 'REJECT_LOW':
            self.stats['rejected_low'] += 1
        else:
            self.stats['not_detected'] += 1
        
        # Print result
        level_str = f"Y={liquid_level_absolute}" if liquid_level_absolute else "NOT DETECTED"
        deviation = liquid_level_absolute - self.target_line_y if liquid_level_absolute else None
        dev_str = f", Dev={deviation:+d}px" if deviation is not None else ""
        
        print(f"Bottle #{track_id}: {status} (Level: {level_str}{dev_str})")
        
        return result
    
    def _classify_bottle(self, liquid_level_absolute):
        """
        Classify bottle as PASS or REJECT based on liquid level
        
        Args:
            liquid_level_absolute: Absolute Y position of liquid level in frame
        
        Returns:
            status: 'PASS', 'REJECT_LOW', or 'NOT_DETECTED'
        """
        if liquid_level_absolute is None:
            return "NOT_DETECTED"
        
        deviation = liquid_level_absolute - self.target_line_y
        
        # PASS if liquid is at or above target line (within tolerance below)
        if deviation <= self.tolerance:
            return "PASS"
        else:
            return "REJECT_LOW"
    
    def detect_and_track(self, frame):
        """
        Run YOLO detection with ByteTrack tracking
        
        Args:
            frame: Input video frame
        
        Returns:
            results: YOLO tracking results
            newly_processed: List of newly processed bottle results
        """
        # Run YOLO with ByteTrack tracker
        # Important: Set imgsz to match or be close to training resolution
        results = self.model.track(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            tracker="bytetrack.yaml",  # Use ByteTrack
            persist=True,  # Persist tracks across frames
            verbose=False,
            imgsz=640  # Standard YOLO input size - prevents distortion
        )
        
        newly_processed = []
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                track_ids = result.boxes.id.cpu().numpy().astype(int)  # Track IDs
                confidences = result.boxes.conf.cpu().numpy()  # Confidences
                
                # Check each tracked bottle
                for bbox, track_id, conf in zip(boxes, track_ids, confidences):
                    # Check if bottle has crossed detection line
                    if self.has_crossed_line(bbox, track_id):
                        # Process this bottle
                        result_dict = self.process_bottle(frame, bbox, track_id)
                        newly_processed.append(result_dict)
        
        return results, newly_processed
    
    def visualize(self, frame, results):
        """
        Visualize tracking, detection line, and liquid levels
        
        Args:
            frame: Input video frame
            results: YOLO tracking results
        
        Returns:
            vis_frame: Annotated frame
        """
        vis_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw target line (GREEN - horizontal)
        cv2.line(vis_frame, (0, self.target_line_y), (width, self.target_line_y), 
                (0, 255, 0), 2)
        cv2.putText(vis_frame, "TARGET LINE", (10, self.target_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw tolerance bounds (RED dashed - horizontal)
        for y in [self.target_line_y - self.tolerance, self.target_line_y + self.tolerance]:
            for i in range(0, width, 20):
                cv2.line(vis_frame, (i, y), (min(i+10, width), y), (0, 0, 255), 1)
        
        # Draw detection line (CYAN - vertical in middle)
        cv2.line(vis_frame, (self.detection_line_x, 0), (self.detection_line_x, height),
                (255, 255, 0), 2)
        cv2.putText(vis_frame, "DETECTION LINE", (self.detection_line_x + 5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw YOLO detections and tracking
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                for bbox, track_id, conf in zip(boxes, track_ids, confidences):
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Determine color based on processing status
                    if track_id in self.bottle_results:
                        result_data = self.bottle_results[track_id]
                        status = result_data['status']
                        
                        if status == 'PASS':
                            color = (0, 255, 0)  # Green
                        elif status == 'REJECT_LOW':
                            color = (0, 0, 255)  # Red
                        else:  # NOT_DETECTED
                            color = (0, 165, 255)  # Orange
                        
                        # Draw liquid level line if detected
                        liquid_level = result_data['liquid_level_absolute']
                        if liquid_level is not None:
                            cv2.line(vis_frame, (x1, int(liquid_level)), (x2, int(liquid_level)),
                                   color, 2)
                    else:
                        # Not yet processed
                        color = (255, 255, 255)  # White
                    
                    # Draw bounding box
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw track ID and confidence
                    if SHOW_TRACKING_IDS:
                        label = f"ID:{track_id}"
                        if track_id in self.bottle_results:
                            label += f" {self.bottle_results[track_id]['status']}"
                        
                        cv2.putText(vis_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw statistics
        stats_y = height - 80
        cv2.rectangle(vis_frame, (5, stats_y - 25), (350, height - 5), (0, 0, 0), -1)
        cv2.putText(vis_frame, f"Total: {self.stats['total_bottles']} | Pass: {self.stats['passed']} | Reject: {self.stats['rejected_low']}",
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Not Detected: {self.stats['not_detected']}",
                   (10, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    print("="*70)
    print("YOLO + BYTETRACK LIQUID LEVEL DETECTION SYSTEM")
    print("="*70)
    print(f"YOLO Model: {YOLO_MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Detection Line: {DETECTION_LINE_POSITION*100:.0f}% of frame width")
    print(f"Target Liquid Level: Y={TARGET_LINE_Y} pixels")
    print(f"Tolerance: ±{TOLERANCE} pixels")
    print(f"Gaussian Kernel: {GAUSSIAN_KERNEL_SIZE}x{GAUSSIAN_KERNEL_SIZE}")
    print(f"Min Gradient Threshold: {MIN_GRADIENT_THRESHOLD}")
    print("="*70)
    print("Controls: [SPACE] Pause/Resume | [Q] Quit | [R] Reset Stats | [D] Toggle Debug")
    print("="*70)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frame_delay = 1.0 / video_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate detection line position
    detection_line_x = int(width * DETECTION_LINE_POSITION)
    
    print(f"Video: {width}x{height} @ {video_fps:.2f} FPS")
    print(f"Detection line at X={detection_line_x} pixels")
    print("="*70)
    
    # Initialize tracker
    tracker = BottleTracker(
        model_path=YOLO_MODEL_PATH,
        detection_line_x=detection_line_x,
        target_line_y=TARGET_LINE_Y,
        tolerance=TOLERANCE
    )
    
    paused = False
    show_debug = SHOW_DEBUG
    fps_display = 0
    fps_counter = 0
    fps_start_time = time.time()
    
    # Create main window with normal flag (resizable, maintains aspect ratio)
    cv2.namedWindow('Bottle Liquid Level Detection', cv2.WINDOW_NORMAL)
    
    # Calculate appropriate window size (fit to screen while maintaining aspect ratio)
    screen_width = 1920  # Adjust if you have different screen resolution
    screen_height = 1080
    scale = min(screen_width / width, screen_height / height, 1.0)  # Don't upscale
    window_width = int(width * scale)
    window_height = int(height * scale)
    cv2.resizeWindow('Bottle Liquid Level Detection', window_width, window_height)
    
    print(f"Display window: {window_width}x{window_height} (scale: {scale:.2f}x)")
    print("="*70)
    
    # Debug window setup
    if show_debug:
        cv2.namedWindow('Debug - ROI & Processed', cv2.WINDOW_NORMAL)
    
    while True:
        frame_start_time = time.time()
        
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker.processed_bottles.clear()  # Reset tracking on loop
                continue
            
            # Detect and track bottles
            results, newly_processed = tracker.detect_and_track(frame)
            
            # Show debug view for newly processed bottles
            if show_debug and len(newly_processed) > 0:
                # Show the last processed bottle's ROI and processed image
                last_result = newly_processed[-1]
                
                if last_result['roi'] is not None and last_result['processed_img'] is not None:
                    roi = last_result['roi']
                    processed = last_result['processed_img']
                    
                    # Create debug visualization
                    debug_frame = np.zeros((max(roi.shape[0], processed.shape[0]), 
                                           roi.shape[1] + processed.shape[1] + 20, 3), 
                                          dtype=np.uint8)
                    
                    # Add ROI (original)
                    debug_frame[:roi.shape[0], :roi.shape[1]] = roi
                    
                    # Add processed image
                    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    
                    # Draw liquid level line on processed image
                    if last_result['liquid_level_relative'] is not None:
                        liquid_y = int(last_result['liquid_level_relative'])
                        cv2.line(processed_bgr, (0, liquid_y), (processed_bgr.shape[1], liquid_y),
                               (0, 0, 255), 2)
                    
                    debug_frame[:processed_bgr.shape[0], 
                               roi.shape[1]+20:roi.shape[1]+20+processed_bgr.shape[1]] = processed_bgr
                    
                    # Add labels
                    cv2.putText(debug_frame, "Original ROI", (5, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(debug_frame, "Gaussian Blur + Detection", 
                               (roi.shape[1]+25, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(debug_frame, f"Bottle #{last_result['track_id']}: {last_result['status']}",
                               (5, debug_frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow('Debug - ROI & Processed', debug_frame)
            
            # Update FPS counter
            fps_counter += 1
            if fps_counter >= 30:
                fps_display = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
        
        # Visualize main frame
        vis_frame = tracker.visualize(frame, results)
        
        # Add FPS and processing time
        processing_time = (time.time() - frame_start_time) * 1000
        cv2.putText(vis_frame, f"FPS: {fps_display:.1f} | Process: {processing_time:.1f}ms",
                   (width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Bottle Liquid Level Detection', vis_frame)
        
        # Maintain video FPS
        elapsed = time.time() - frame_start_time
        wait_time = max(1, int((frame_delay - elapsed) * 1000))
        
        # Handle key presses
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord(' '):  # Pause/Resume
            paused = not paused
        elif key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Reset stats
            tracker.stats = {'total_bottles': 0, 'passed': 0, 'rejected_low': 0, 'not_detected': 0}
            tracker.processed_bottles.clear()
            tracker.bottle_results.clear()
            print("Statistics and tracking reset!")
        elif key == ord('d'):  # Toggle debug window
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow('Debug - ROI & Processed')
            else:
                cv2.namedWindow('Debug - ROI & Processed', cv2.WINDOW_NORMAL)
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Total Bottles Processed: {tracker.stats['total_bottles']}")
    print(f"Passed: {tracker.stats['passed']} ({tracker.stats['passed']/max(tracker.stats['total_bottles'],1)*100:.1f}%)")
    print(f"Rejected (Low): {tracker.stats['rejected_low']} ({tracker.stats['rejected_low']/max(tracker.stats['total_bottles'],1)*100:.1f}%)")
    print(f"Not Detected: {tracker.stats['not_detected']} ({tracker.stats['not_detected']/max(tracker.stats['total_bottles'],1)*100:.1f}%)")
    print("="*70)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()