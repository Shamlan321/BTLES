import cv2
import numpy as np
from collections import deque
from datetime import datetime

# --- CONFIGURATION ---
VIDEO_SOURCE = 'bottle_test.mp4' 
TARGET_Y_PIXEL = 190
TARGET_Y_RELATIVE = 0.2639
TOLERANCE = 0  # Strict tolerance as requested

# Detection Zone
DETECTION_ZONE_X_START = 0
DETECTION_ZONE_X_END = 250
DECISION_LINE_X = 180

# Tracking
TRACKING_DISTANCE = 100
MIN_FRAMES_TO_TRACK = 3

# Visualization Colors
COLOR_PASS = (0, 255, 0)      # Green
COLOR_REJECT = (0, 0, 255)    # Red
COLOR_TRACKING = (255, 255, 0)# Cyan
COLOR_LINE = (0, 255, 0)
COLOR_TOLERANCE = (0, 165, 255)
COLOR_ZONE = (255, 200, 100)

def nothing(x):
    pass

class Bottle:
    def __init__(self, bottle_id, rect, frame_num):
        self.id = bottle_id
        self.rect = rect # (x, y, w, h)
        self.first_frame = frame_num
        self.last_frame = frame_num
        self.liquid_levels = []
        self.status = "TRACKING"
        self.finalized = False
        self.centroid = self._get_centroid(rect)
        
    def _get_centroid(self, rect):
        x, y, w, h = rect
        return (int(x + w/2), int(y + h/2))
    
    def update(self, rect, frame_num):
        self.rect = rect
        self.last_frame = frame_num
        self.centroid = self._get_centroid(rect)
    
    def add_liquid_level(self, level_y):
        if level_y is not None:
            self.liquid_levels.append(level_y)
    
    def evaluate(self, target_y, tolerance):
        if self.finalized: return self.status
        if len(self.liquid_levels) < MIN_FRAMES_TO_TRACK: return "TRACKING"
        
        median_level = np.median(self.liquid_levels)
        # Strict check
        if abs(median_level - target_y) <= tolerance:
            self.status = "PASS"
        else:
            self.status = "REJECTED"
        return self.status

class BottleInspector:
    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            print(f"Warning: Could not open {video_source}, trying webcam 0...")
            self.cap = cv2.VideoCapture(0)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30
        self.frame_delay = int(1000 / self.fps)
        
        ret, frame = self.cap.read()
        if not ret: raise ValueError("Cannot read video")
        self.height, self.width = frame.shape[:2]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.target_y = TARGET_Y_PIXEL
        self.tolerance = TOLERANCE
        
        self.bottles = {}
        self.next_bottle_id = 1
        self.frame_count = 0
        
        self.total_pass = 0
        self.total_rejected = 0
        self.total_processed = 0
        
        self.fps_history = deque(maxlen=30)
        self.last_time = cv2.getTickCount()

        # HSV Calibration Window
        cv2.namedWindow("Calibration")
        # Default values for Olive Oil (Yellowish/Gold)
        cv2.createTrackbar("H Min", "Calibration", 10, 179, nothing)
        cv2.createTrackbar("H Max", "Calibration", 40, 179, nothing)
        cv2.createTrackbar("S Min", "Calibration", 100, 255, nothing)
        cv2.createTrackbar("S Max", "Calibration", 255, 255, nothing)
        cv2.createTrackbar("V Min", "Calibration", 50, 255, nothing)
        cv2.createTrackbar("V Max", "Calibration", 255, 255, nothing)
        cv2.createTrackbar("Show Mask", "Calibration", 0, 1, nothing)

    def get_hsv_values(self):
        h_min = cv2.getTrackbarPos("H Min", "Calibration")
        h_max = cv2.getTrackbarPos("H Max", "Calibration")
        s_min = cv2.getTrackbarPos("S Min", "Calibration")
        s_max = cv2.getTrackbarPos("S Max", "Calibration")
        v_min = cv2.getTrackbarPos("V Min", "Calibration")
        v_max = cv2.getTrackbarPos("V Max", "Calibration")
        return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])

    def merge_vertical_blobs(self, blobs):
        """
        Merge blobs that are vertically aligned (split by iron rod).
        Input: List of (x, y, w, h) tuples
        Output: List of merged (x, y, w, h) tuples
        """
        if not blobs: return []
        
        # Sort by X coordinate
        blobs.sort(key=lambda b: b[0])
        
        merged = []
        used_indices = set()
        
        for i in range(len(blobs)):
            if i in used_indices: continue
            
            x1, y1, w1, h1 = blobs[i]
            cx1 = x1 + w1/2
            
            # Look for a vertical match
            best_match = -1
            
            for j in range(i + 1, len(blobs)):
                if j in used_indices: continue
                
                x2, y2, w2, h2 = blobs[j]
                cx2 = x2 + w2/2
                
                # Check if they are vertically aligned (similar X center)
                if abs(cx1 - cx2) < 30: # 30px tolerance for vertical alignment
                    best_match = j
                    break
            
            if best_match != -1:
                # Merge the two blobs
                x2, y2, w2, h2 = blobs[best_match]
                
                # New bounding box covering both
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1+w1, x2+w2) - new_x
                new_h = max(y1+h1, y2+h2) - new_y
                
                merged.append((new_x, new_y, new_w, new_h))
                used_indices.add(i)
                used_indices.add(best_match)
            else:
                # No match found, keep original
                merged.append((x1, y1, w1, h1))
                used_indices.add(i)
                
        return merged

    def find_liquid_blobs(self, frame):
        """Detect liquid blobs and merge split ones"""
        zone = frame[:, DETECTION_ZONE_X_START:DETECTION_ZONE_X_END]
        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        lower, upper = self.get_hsv_values()
        mask = cv2.inRange(hsv, lower, upper)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        if cv2.getTrackbarPos("Show Mask", "Calibration") == 1:
            cv2.imshow("Mask", mask)
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500: # Lowered threshold to detect split parts
                x, y, w, h = cv2.boundingRect(cnt)
                # Adjust to global coordinates
                x += DETECTION_ZONE_X_START
                raw_blobs.append((x, y, w, h))
        
        # Merge split blobs (Iron Rod fix)
        merged_blobs = self.merge_vertical_blobs(raw_blobs)
        
        return merged_blobs

    def process_frame(self, frame):
        self.frame_count += 1
        display_frame = frame.copy()
        
        # Draw Static Lines
        cv2.rectangle(display_frame, (DETECTION_ZONE_X_START, 0), (DETECTION_ZONE_X_END, self.height), COLOR_ZONE, 1)
        cv2.line(display_frame, (DECISION_LINE_X, 0), (DECISION_LINE_X, self.height), (255, 255, 0), 1)
        
        cv2.line(display_frame, (0, self.target_y), (self.width, self.target_y), COLOR_LINE, 2)
        # Tolerance lines (might be on top of target if tolerance is 0)
        if self.tolerance > 0:
            cv2.line(display_frame, (0, self.target_y - self.tolerance), (self.width, self.target_y - self.tolerance), COLOR_TOLERANCE, 1)
            cv2.line(display_frame, (0, self.target_y + self.tolerance), (self.width, self.target_y + self.tolerance), COLOR_TOLERANCE, 1)
        
        # Detect Liquid Blobs (Merged)
        liquid_rects = self.find_liquid_blobs(frame)
        current_frame_ids = []
        
        for rect in liquid_rects:
            x, y, w, h = rect
            
            # The Liquid Level is the TOP of the blob (y)
            liquid_level = y
            
            # Centroid
            cx = int(x + w/2)
            cy = int(y + h/2)
            
            # Match to existing bottle
            matched_id = None
            for bid, bottle in self.bottles.items():
                if bottle.centroid:
                    dist = np.sqrt((cx - bottle.centroid[0])**2 + (cy - bottle.centroid[1])**2)
                    if dist < TRACKING_DISTANCE:
                        matched_id = bid
                        break
            
            if matched_id is None:
                matched_id = self.next_bottle_id
                self.next_bottle_id += 1
                self.bottles[matched_id] = Bottle(matched_id, rect, self.frame_count)
            else:
                self.bottles[matched_id].update(rect, self.frame_count)
                
            current_frame_ids.append(matched_id)
            bottle = self.bottles[matched_id]
            
            bottle.add_liquid_level(liquid_level)
            status = bottle.evaluate(self.target_y, self.tolerance)
            
            # Decision Logic
            if cx > DECISION_LINE_X and not bottle.finalized and status != "TRACKING":
                bottle.finalized = True
                if status == "PASS": self.total_pass += 1
                elif status == "REJECTED": self.total_rejected += 1
                self.total_processed += 1
            
            # VISUALIZATION
            color = COLOR_TRACKING
            if status == "PASS": color = COLOR_PASS
            elif status == "REJECTED": color = COLOR_REJECT
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw Level Line
            cv2.line(display_frame, (x, liquid_level), (x+w, liquid_level), (255, 0, 255), 2)
            
            # Label
            cv2.putText(display_frame, f"{status}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Cleanup
        to_remove = [bid for bid, b in self.bottles.items() if bid not in current_frame_ids and self.frame_count - b.last_frame > 5]
        for bid in to_remove: del self.bottles[bid]
            
        self._draw_stats(display_frame)
        return display_frame

    def _draw_stats(self, frame):
        curr_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_time - self.last_time)
        self.fps_history.append(fps)
        self.last_time = curr_time
        
        cv2.rectangle(frame, (10, 10), (250, 130), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {np.mean(self.fps_history):.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"PASS: {self.total_pass}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PASS, 2)
        cv2.putText(frame, f"REJECT: {self.total_rejected}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_REJECT, 2)
        cv2.putText(frame, f"TOTAL: {self.total_processed}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    def run(self):
        print("Controls: SPACE=Pause, Q=Quit")
        print("Use 'Calibration' window to adjust color detection!")
        paused = False
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                processed = self.process_frame(frame)
                cv2.imshow("Bottle Inspector", processed)
            
            key = cv2.waitKey(self.frame_delay) & 0xFF
            if key == ord('q') or key == 27: break
            if key == 32: paused = not paused
            
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        BottleInspector(VIDEO_SOURCE).run()
    except Exception as e:
        print(f"Error: {e}")
