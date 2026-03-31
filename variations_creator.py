import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg

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
    # Calculate vertical intensity profile (average across width)
    intensity_profile = np.mean(gray_img, axis=1)
    
    # Find the steepest gradient (liquid/air transition)
    gradients = np.diff(intensity_profile)
    abs_gradients = np.abs(gradients)
    
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

def create_intensity_profile_visualization(gray_img, liquid_level=None, title="Intensity Profile"):
    """Create a visualization of the vertical intensity profile with detected level"""
    # Calculate vertical intensity profile
    intensity_profile = np.mean(gray_img, axis=1)
    
    # Create a figure for the profile
    fig = plt.figure(figsize=(6, 4))
    plt.plot(intensity_profile, 'g-', linewidth=1.5)
    plt.title(title)
    plt.xlabel('Y Position')
    plt.ylabel('Intensity')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Mark detected liquid level if available
    if liquid_level is not None:
        plt.axvline(x=liquid_level, color='r', linestyle='--', 
                   label=f'Liquid Level (Y={liquid_level})')
        plt.legend()
    
    # Convert matplotlib figure to OpenCV image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(int(height), int(width), 4)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    return img

def create_gradient_visualization(gray_img, liquid_level=None, title="Gradient Analysis"):
    """Create a visualization of the vertical gradient with detected level"""
    # Calculate vertical intensity profile
    intensity_profile = np.mean(gray_img, axis=1)
    
    # Calculate gradients
    gradients = np.diff(intensity_profile)
    abs_gradients = np.abs(gradients)
    
    # Create a figure for the gradient
    fig = plt.figure(figsize=(6, 4))
    
    # Plot intensity
    plt.subplot(2, 1, 1)
    plt.plot(intensity_profile, 'g-', linewidth=1.5)
    plt.title('Intensity Profile')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot gradients
    plt.subplot(2, 1, 2)
    plt.plot(abs_gradients, 'r-', linewidth=1.5)
    plt.title('Gradient Magnitude')
    plt.xlabel('Y Position')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Mark strongest gradient
    if liquid_level is not None and liquid_level < len(abs_gradients):
        plt.axvline(x=liquid_level, color='b', linestyle='--', 
                   label=f'Detected Level (Y={liquid_level})')
        plt.legend()
    
    # Convert matplotlib figure to OpenCV image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(int(height), int(width), 4)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    return img

def process_image_for_detection(img, method_name):
    """
    Process an image for liquid level detection based on method name
    
    Args:
        img: Input image (BGR)
        method_name: Name of the processing method
    
    Returns:
        processed_img: Processed image for display
        detection_img: Image with liquid level detection
        liquid_level: Detected liquid level (or None)
        method_title: Updated method title with detection info
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply specific processing based on method
    if "Gaussian" in method_name:
        ksize = int(method_name.split("k=")[-1].split(")")[0])
        processed = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    elif "CLAHE" in method_name:
        if "clip=" in method_name:
            clip = float(method_name.split("clip=")[-1].split(")")[0])
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
    elif "Binary" in method_name:
        thresh = int(method_name.split("thresh=")[-1].split(")")[0])
        _, processed = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    elif "Adaptive" in method_name:
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        processed = adaptive
    elif "Canny" in method_name:
        params = method_name.split("low=")[-1].split(")")[0]
        low, high = map(int, params.split(", high="))
        processed = cv2.Canny(gray, low, high)
    elif "Sobel" in method_name:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        processed = np.uint8(255 * sobel / np.max(sobel))
    elif "HSV" in method_name:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        processed = v
    elif "LAB" in method_name:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        processed = l
    elif "Gradient" in method_name:
        # Special case - this is a visualization, not processing
        return img, img, None, method_name
    elif "Intensity" in method_name:
        # Special case - this is a visualization, not processing
        return img, img, None, method_name
    else:
        processed = gray
    
    # Detect liquid level
    liquid_level, gradient_info = detect_liquid_level(processed)
    
    # Create display image
    if len(processed.shape) == 2:
        display_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        display_img = processed.copy()
    
    # Draw liquid level line if detected
    if liquid_level is not None:
        cv2.line(display_img, (0, liquid_level), (display_img.shape[1], liquid_level), 
                (0, 0, 255), 2)
        cv2.circle(display_img, (display_img.shape[1]//2, liquid_level), 4, (0, 0, 255), -1)
    
    # Create title with detection info
    method_title = method_name
    if liquid_level is not None:
        method_title += f" (Y={liquid_level})"
    
    return processed, display_img, liquid_level, method_title

def create_analysis_grid(image_path):
    """Create a comprehensive analysis grid of the image with liquid level detection"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Create various processed versions with detection
    results = []
    
    # 1. Original image
    _, orig_with_line, orig_level, orig_title = process_image_for_detection(img, "Original")
    results.append((orig_title, orig_with_line))
    
    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray_with_line, gray_level, gray_title = process_image_for_detection(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Grayscale")
    results.append((gray_title, gray_with_line))
    
    # 3. Gaussian blur (different kernel sizes)
    for ksize in [3, 5, 7, 9]:
        _, blurred_with_line, level, title = process_image_for_detection(
            img, f"Gaussian Blur (k={ksize})")
        results.append((title, blurred_with_line))
    
    # 4. CLAHE (different clip limits)
    for clip in [2.0, 3.0, 4.0]:
        _, clahe_with_line, level, title = process_image_for_detection(
            img, f"CLAHE (clip={clip})")
        results.append((title, clahe_with_line))
    
    # 5. Binary thresholding (different thresholds)
    for thresh in [50, 100, 150]:
        _, binary_with_line, level, title = process_image_for_detection(
            img, f"Binary (thresh={thresh})")
        results.append((title, binary_with_line))
    
    # 6. Adaptive thresholding
    _, adaptive_with_line, level, title = process_image_for_detection(
        img, "Adaptive Threshold")
    results.append((title, adaptive_with_line))
    
    # 7. Canny edge detection (different thresholds)
    for low, high in [(50, 150), (100, 200), (30, 90)]:
        _, edges_with_line, level, title = process_image_for_detection(
            img, f"Canny (low={low}, high={high})")
        results.append((title, edges_with_line))
    
    # 8. Sobel edge detection
    _, sobel_with_line, level, title = process_image_for_detection(
        img, "Sobel Edge Detection")
    results.append((title, sobel_with_line))
    
    # 9. HSV color space (show value channel)
    _, hsv_with_line, level, title = process_image_for_detection(
        img, "HSV - Value Channel")
    results.append((title, hsv_with_line))
    
    # 10. LAB color space (show L channel)
    _, lab_with_line, level, title = process_image_for_detection(
        img, "LAB - L Channel")
    results.append((title, lab_with_line))
    
    # 11. Intensity profile visualization
    intensity_profile = create_intensity_profile_visualization(gray)
    results.append(("Intensity Profile", intensity_profile))
    
    # 12. Gradient visualization
    gradient_vis = create_gradient_visualization(gray)
    results.append(("Gradient Analysis", gradient_vis))
    
    # 13. CLAHE + Binary threshold combo
    _, clahe_binary_with_line, level, title = process_image_for_detection(
        img, "CLAHE + Binary Threshold")
    results.append((title, clahe_binary_with_line))
    
    # 14. CLAHE + Adaptive threshold combo
    _, clahe_adaptive_with_line, level, title = process_image_for_detection(
        img, "CLAHE + Adaptive Threshold")
    results.append((title, clahe_adaptive_with_line))
    
    # Calculate grid dimensions (4 columns)
    cols = 4
    rows = (len(results) + cols - 1) // cols
    
    # Create a blank canvas for the grid
    cell_height = 200
    cell_width = 300
    grid = np.ones((rows * cell_height, cols * cell_width, 3), dtype=np.uint8) * 240
    
    # Fill the grid with results
    for i, (title, img) in enumerate(results):
        r = i // cols
        c = i % cols
        
        # Resize image to fit cell
        if img.shape[0] > cell_height or img.shape[1] > cell_width:
            scale = min(cell_height / img.shape[0], cell_width / img.shape[1])
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            resized = cv2.resize(img, new_size)
        else:
            resized = img.copy()
        
        # Place in grid
        y_offset = r * cell_height + (cell_height - resized.shape[0]) // 2
        x_offset = c * cell_width + (cell_width - resized.shape[1]) // 2
        
        # Add title
        cv2.putText(grid, title, (c * cell_width + 5, y_offset - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        
        # Place image
        if len(resized.shape) == 2:  # Grayscale
            grid[y_offset:y_offset+resized.shape[0], 
                 x_offset:x_offset+resized.shape[1], 
                 :] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        else:  # Color
            grid[y_offset:y_offset+resized.shape[0], 
                 x_offset:x_offset+resized.shape[1], 
                 :] = resized
    
    return grid

def main():
    parser = argparse.ArgumentParser(description='Liquid Level Detection Image Analysis Tool')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='liquid_analysis.jpg', help='Output file path')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    print("="*70)
    print("LIQUID LEVEL DETECTION IMAGE ANALYSIS TOOL")
    print("="*70)
    print("\nINSTRUCTIONS:")
    print("1. This tool analyzes your image with multiple computer vision techniques")
    print("2. Each image shows the detected liquid level (red line)")
    print("3. Look for methods that clearly detect the liquid/air interface")
    print("4. For black liquid in transparent bottles, focus on:")
    print("   - CLAHE enhanced images")
    print("   - Gradient analysis")
    print("   - ROI-focused processing")
    print("="*70)
    
    # Create analysis grid
    print(f"Analyzing image: {args.image}")
    grid = create_analysis_grid(args.image)
    
    if grid is None:
        return
    
    # Display the grid
    cv2.imshow("Liquid Level Detection Analysis", grid)
    
    # Wait for key press
    print("\nPress any key on the image window to continue...")
    print("Press 's' to save the analysis grid")
    print("Press 'q' to quit without saving")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(args.output, grid)
            print(f"\nSaved analysis grid to: {args.output}")
            break
    
    cv2.destroyAllWindows()
    
    # Show specific recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR TRANSPARENT BOTTLES WITH BLACK LIQUID:")
    print("="*70)
    print("1. BEST OVERALL APPROACH: CLAHE + Gradient Analysis")
    print("   - Use CLAHE (clip=3.0) to enhance contrast")
    print("   - Analyze gradient to find steepest transition")
    print("   - The red line shows detected liquid level")
    print("\n2. AVOID:")
    print("   - Simple binary thresholding (too sensitive to lighting)")
    print("   - Full-image processing (bottle labels cause false detections)")
    print("\n3. FOR YOUR SETUP:")
    print("   - If virtual line is at Y=249, analyze ±60px around it")
    print("   - Use gradient threshold > 15 to filter out noise")
    print("   - Focus on methods that show a clear red line at the interface")
    print("="*70)

if __name__ == "__main__":
    main()