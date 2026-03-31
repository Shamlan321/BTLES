import cv2
import numpy as np
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Liquid Level Calibration Tool')
    parser.add_argument('--video', type=str, required=True, help='Path to video file or camera index')
    parser.add_argument('--output', type=str, default='liquid_level_calibration.txt', help='Output file for calibration')
    args = parser.parse_args()
    
    # Open video source
    try:
        video_source = int(args.video)
        cap = cv2.VideoCapture(video_source)
    except ValueError:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.video}")
        return
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate display size (max 1280 width, maintain aspect ratio)
    max_display_width = 1280
    if frame_width > max_display_width:
        scale_factor = max_display_width / frame_width
        display_width = max_display_width
        display_height = int(frame_height * scale_factor)
    else:
        display_width = frame_width
        display_height = frame_height
        scale_factor = 1.0
    
    # Initial line position (middle of the frame, scaled to display size)
    line_y = display_height // 2
    line_color = (0, 255, 0)  # Green
    calibration_complete = False
    window_resized = False
    
    print("="*50)
    print("LIQUID LEVEL CALIBRATION TOOL")
    print("="*50)
    print("\nINSTRUCTIONS:")
    print("1. Play the video and pause when a bottle with CORRECT liquid level is visible")
    print("2. Click on the video at the desired liquid level position")
    print("3. Fine-tune with UP/DOWN arrow keys if needed")
    print("4. Press 'S' to save the calibration")
    print("5. Press 'Q' to quit without saving")
    print(f"\nOriginal frame size: {frame_width}x{frame_height}")
    print(f"Display window size: {display_width}x{display_height}")
    print(f"Initial reference line: Y={line_y} (relative: {line_y/display_height:.4f})")
    print("="*50)

    def mouse_callback(event, x, y, flags, param):
        """Mouse callback to set reference line position"""
        nonlocal line_y
        if event == cv2.EVENT_LBUTTONDOWN:
            line_y = y
            print(f"\rSelected position: Y={line_y} (relative: {line_y/display_height:.4f})    ", end="")

    # Create resizable window and set mouse callback
    cv2.namedWindow("Liquid Level Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Liquid Level Calibration", display_width, display_height)
    cv2.setMouseCallback("Liquid Level Calibration", mouse_callback)
    
    # Control variables
    paused = False
    frame = None
    
    while True:
        if not paused or frame is None:
            ret, frame = cap.read()
            if not ret:
                # Restart video if at end
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        
        # Resize frame to current window size if window was resized
        current_window_size = cv2.getWindowImageRect("Liquid Level Calibration")
        window_w, window_h = current_window_size[2], current_window_size[3]
        
        if window_w > 0 and window_h > 0:
            display_frame = cv2.resize(frame, (window_w, window_h))
            # Adjust line_y proportionally if window size changed
            if not window_resized:
                window_resized = True
            # Calculate scale for current window
            y_scale = window_h / display_height
        else:
            display_frame = frame
            y_scale = 1.0
        
        # Draw reference line (scaled to current window size)
        frame_with_line = display_frame.copy()
        current_line_y = int(line_y * y_scale)
        cv2.line(frame_with_line, (0, current_line_y), (window_w, current_line_y), line_color, 2)
        
        # Display position info
        cv2.putText(frame_with_line, f"Y={line_y} (rel: {line_y/display_height:.4f})", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame_with_line, "CLICK: Set line | ARROWS: Fine-tune | S: Save | Q: Quit", 
                   (10, window_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Liquid Level Calibration", frame_with_line)
        
        key = cv2.waitKey(30 if not paused else 0)
        
        # Handle key presses
        if key == -1:  # No key pressed
            continue
            
        key_char = key & 0xFF
        
        if key_char == ord('q'):
            print("\nExiting without saving calibration.")
            break
        elif key_char == ord('s'):
            print(f"\n\nCalibration saved! Liquid level reference Y={line_y}")
            print(f"Relative position: {line_y/frame_height:.4f}")
            
            with open(args.output, 'w') as f:
                f.write(f"# Liquid Level Calibration\n")
                f.write(f"reference_y={line_y}\n")
                f.write(f"display_height={display_height}\n")
                f.write(f"frame_height={frame_height}\n")
                f.write(f"frame_width={frame_width}\n")
                f.write(f"relative_position={line_y/display_height:.4f}\n")
            
            print(f"Calibration saved to {args.output}")
            print("You can now update your main program with these values.")
            calibration_complete = True
            break
        elif key_char == ord(' '):
            paused = not paused
            print(f"\nVideo {'paused' if paused else 'resumed'}")
        elif key == 65362 or key == 2490368:  # Up arrow
            line_y = max(0, line_y - 1)
            print(f"\rFine-tuned position: Y={line_y} (relative: {line_y/display_height:.4f})    ", end="")
        elif key == 65364 or key == 2621440:  # Down arrow
            line_y = min(display_height - 1, line_y + 1)
            print(f"\rFine-tuned position: Y={line_y} (relative: {line_y/display_height:.4f})    ", end="")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if not calibration_complete:
        print("\nCalibration not saved. Run the tool again when ready.")

if __name__ == "__main__":
    main()