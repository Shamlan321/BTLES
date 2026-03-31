import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os

class DetectionVariation:
    """Holder for different parameter sets to test."""
    def __init__(self, name: str, params: Dict[str, Any], description: str = ""):
        self.name = name
        self.params = params
        self.description = description

def detect_bottle_regions_enhanced(
    gray_image: np.ndarray,
    min_area: int = 3000,
    min_aspect_ratio: float = 2.0,
    max_aspect_ratio: float = 7.0
) -> List[Tuple[int, int, int, int]]:
    """
    Enhanced bottle detection that works for transparent bottles with dark liquid.
    Uses multiple strategies to ensure robust detection.
    """
    bottles = []
    
    # Strategy 1: Adaptive Thresholding (good for separating bottles from background)
    thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / float(w)
            if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                bottles.append((x, y, w, h))
    
    # Strategy 2: If no bottles found, try edge detection (for very clean backgrounds)
    if len(bottles) == 0:
        edges = cv2.Canny(gray_image, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / float(w)
                if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                    bottles.append((x, y, w, h))
    
    # Remove overlapping detections
    bottles = non_max_suppression(bottles, overlap_thresh=0.5)
    
    return sorted(bottles, key=lambda b: b[0])  # Sort left to right

def non_max_suppression(
    boxes: List[Tuple[int, int, int, int]],
    overlap_thresh: float = 0.5
) -> List[Tuple[int, int, int, int]]:
    """Remove overlapping bounding boxes."""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy array
    boxes_np = np.array([[x, y, x+w, y+h] for x, y, w, h in boxes], dtype=float)
    
    # Compute area
    area = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
    
    # Sort by area (largest first)
    idxs = np.argsort(area)[::-1]
    
    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        
        # Compute IoU with remaining boxes
        xx1 = np.maximum(boxes_np[i, 0], boxes_np[idxs[1:], 0])
        yy1 = np.maximum(boxes_np[i, 1], boxes_np[idxs[1:], 1])
        xx2 = np.minimum(boxes_np[i, 2], boxes_np[idxs[1:], 2])
        yy2 = np.minimum(boxes_np[i, 3], boxes_np[idxs[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[1:]]
        
        idxs = idxs[np.where(overlap <= overlap_thresh)[0] + 1]
    
    return [boxes[i] for i in pick]

def detect_liquid_level_variation(
    bottle_roi: np.ndarray,
    params: Dict[str, Any]
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect liquid level with a specific parameter set.
    Optimized for high-contrast black liquid scenario.
    """
    # Extract parameters
    blur_ksize = params.get('blur_ksize', (3, 3))
    blur_sigma = params.get('blur_sigma', 0)
    canny_low = params.get('canny_low', 30)
    canny_high = params.get('canny_high', 90)
    hough_threshold = params.get('hough_threshold', 40)
    min_line_length_ratio = params.get('min_line_length_ratio', 0.5)
    max_line_gap = params.get('max_line_gap', 15)
    lower_region_ratio = params.get('lower_region_ratio', 0.25)
    
    # Preprocessing
    blurred = cv2.GaussianBlur(bottle_roi, blur_ksize, blur_sigma)
    
    # Edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=int(bottle_roi.shape[1] * min_line_length_ratio),
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return None
    
    # Find the most likely liquid surface line
    roi_height = bottle_roi.shape[0]
    valid_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # More flexible horizontal check
        is_horizontal = abs(y2 - y1) < 20
        
        # Must be below the middle of the bottle (liquid is in lower half)
        min_y = min(y1, y2)
        is_in_lower_region = min_y > roi_height * lower_region_ratio
        
        # Line must be long enough
        line_length = abs(x2 - x1)
        min_length = int(bottle_roi.shape[1] * min_line_length_ratio)
        
        if is_horizontal and is_in_lower_region and line_length >= min_length:
            # Score: lower position is better, longer line is better
            score = (min_y * 2) + (line_length / bottle_roi.shape[1] * 50)
            valid_lines.append((score, (x1, y1, x2, y2)))
    
    if not valid_lines:
        return None
    
    return max(valid_lines, key=lambda x: x[0])[1]

def create_labeled_result(
    frame: np.ndarray,
    bottles: List[Tuple[int, int, int, int]],
    variation: DetectionVariation,
    output_dir: str
) -> Tuple[np.ndarray, int]:
    """
    Create a result image with clear labeling of parameters and results.
    Returns image and detection count.
    """
    result = frame.copy()
    params = variation.params
    
    # Add variation name and key parameters at top
    header_text = f"{variation.name}: Canny=({params['canny_low']}-{params['canny_high']})"
    cv2.putText(result, header_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add parameter details
    param_text = f"Blur={params['blur_ksize']}, Hough_th={params['hough_threshold']}"
    cv2.putText(result, param_text, (10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    total_detected = 0
    
    # Process each bottle
    for idx, (x, y, w, h) in enumerate(bottles, 1):
        # Extract ROI from grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bottle_gray_roi = gray_frame[y:y+h, x:x+w]
        
        # Detect liquid level
        liquid_line = detect_liquid_level_variation(bottle_gray_roi, params)
        
        if liquid_line is not None:
            # Draw prominent green line with label
            x1, y1, x2, y2 = liquid_line
            start_point = (x + x1, y + y1)
            end_point = (x + x2, y + y2)
            
            # Thicker line for visibility
            cv2.line(result, start_point, end_point, (0, 255, 0), 4)
            
            # Add "LEVEL" label near the line
            mid_x = (start_point[0] + end_point[0]) // 2
            label_y = start_point[1] - 5
            cv2.putText(result, "LEVEL", (mid_x - 20, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            total_detected += 1
            status = "DETECTED"
            color = (0, 255, 0)
        else:
            status = "NOT FOUND"
            color = (0, 0, 255)
        
        # Draw bottle bounding box
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Add bottle status
        cv2.putText(result, f"B{idx}: {status}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add summary at bottom
    summary_text = f"SUMMARY: {total_detected}/{len(bottles)} bottles"
    cv2.putText(result, summary_text, (10, result.shape[0] - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Save individual result
    output_path = os.path.join(output_dir, f"{variation.name}.jpg")
    cv2.imwrite(output_path, result)
    print(f"  Saved: {variation.name} - {summary_text}")
    
    return result, total_detected

def process_with_variations(
    image_path: str,
    variations: List[DetectionVariation],
    output_dir: str = "liquid_detection_results"
) -> None:
    """
    Main function to process all variations and generate comparison outputs.
    """
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect bottles once using enhanced detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bottles = detect_bottle_regions_enhanced(gray, min_area=3000)
    
    if not bottles:
        print("No bottles detected! Check image quality and bottle visibility.")
        return
    
    print(f"Detected {len(bottles)} bottles. Testing {len(variations)} parameter variations...\n")
    
    variation_results = []
    performance_data = []
    
    # Process each variation
    for variation in variations:
        print(f"Processing: {variation.name}")
        print(f"  Description: {variation.description}")
        
        result, detect_count = create_labeled_result(frame, bottles, variation, output_dir)
        variation_results.append(result)
        
        performance_data.append({
            'name': variation.name,
            'detected': detect_count,
            'total': len(bottles),
            'rate': f"{detect_count/len(bottles)*100:.1f}%"
        })
        
        print(f"  Results: {detect_count}/{len(bottles)} detected\n")
    
    # Create comparison grid
    if len(variation_results) > 1:
        create_comparison_grid(variation_results, variations, output_dir)
    
    # Print performance summary table
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Variation':<20} {'Detected':<10} {'Total':<8} {'Rate':<8}")
    print("-"*60)
    for data in performance_data:
        print(f"{data['name']:<20} {data['detected']:<10} {data['total']:<8} {data['rate']:<8}")
    print("="*60)
    
    # Recommend best variation
    best = max(performance_data, key=lambda x: x['detected'])
    print(f"\nBEST PERFORMANCE: {best['name']} ({best['rate']} detection rate)")
    
    print(f"\nAll results saved to: {output_dir}/")

def create_comparison_grid(
    results: List[np.ndarray],
    variations: List[DetectionVariation],
    output_dir: str
) -> None:
    """
    Create a side-by-side comparison grid of all variations.
    """
    n_variations = len(results)
    cols = 2 if n_variations <= 4 else 3
    rows = (n_variations + cols - 1) // cols
    
    # Resize images to common size for grid
    target_h = min(r.shape[0] for r in results)
    target_w = min(r.shape[1] for r in results)
    target_size = (target_w, target_h)
    
    # Resize all results to target size
    resized_results = []
    for result in results:
        resized = cv2.resize(result, target_size)
        resized_results.append(resized)
    
    # Create grid canvas
    grid_h = target_h * rows
    grid_w = target_w * cols
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Place each result in grid
    for idx, result in enumerate(resized_results):
        row = idx // cols
        col = idx % cols
        
        y_start = row * target_h
        x_start = col * target_w
        
        grid[y_start:y_start+target_h, x_start:x_start+target_w] = result
        
        # Add index label on grid for easy reference
        label = f"{idx+1}. {variations[idx].name}"
        cv2.putText(grid, label, (x_start+5, y_start+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Save grid
    grid_path = os.path.join(output_dir, "comparison_grid.jpg")
    cv2.imwrite(grid_path, grid)
    print(f"\nComparison grid saved to: {grid_path}")

def main():
    """Run variations for black liquid detection."""
    INPUT_IMAGE = "op.png"  # Change this to your image path
    
    # Define comprehensive parameter variations for black liquid scenario
    variations = [
        DetectionVariation(
            "Baseline", 
            {
                'blur_ksize': (3, 3), 'blur_sigma': 0,
                'canny_low': 30, 'canny_high': 90,
                'hough_threshold': 40, 'min_line_length_ratio': 0.5,
                'max_line_gap': 15, 'lower_region_ratio': 0.25
            },
            "Default balanced parameters"
        ),
        
        DetectionVariation(
            "Low_Canny_Sensitive", 
            {
                'blur_ksize': (3, 3), 'blur_sigma': 0,
                'canny_low': 15, 'canny_high': 45,
                'hough_threshold': 25, 'min_line_length_ratio': 0.4,
                'max_line_gap': 20, 'lower_region_ratio': 0.25
            },
            "More sensitive edge detection for faint lines"
        ),
        
        DetectionVariation(
            "High_Canny_Strict", 
            {
                'blur_ksize': (5, 5), 'blur_sigma': 0,
                'canny_low': 60, 'canny_high': 120,
                'hough_threshold': 50, 'min_line_length_ratio': 0.6,
                'max_line_gap': 10, 'lower_region_ratio': 0.25
            },
            "Stricter edge detection to reduce false positives"
        ),
        
        DetectionVariation(
            "Heavy_Blur_Noise_Reduction", 
            {
                'blur_ksize': (7, 7), 'blur_sigma': 0,
                'canny_low': 30, 'canny_high': 90,
                'hough_threshold': 35, 'min_line_length_ratio': 0.5,
                'max_line_gap': 15, 'lower_region_ratio': 0.25
            },
            "More blur to handle noisy images"
        ),
        
        DetectionVariation(
            "No_Blur_Sharp", 
            {
                'blur_ksize': (1, 1), 'blur_sigma': 0,
                'canny_low': 40, 'canny_high': 100,
                'hough_threshold': 45, 'min_line_length_ratio': 0.5,
                'max_line_gap': 15, 'lower_region_ratio': 0.25
            },
            "Minimal blur for sharp edges"
        ),
        
        DetectionVariation(
            "Very_Sensitive", 
            {
                'blur_ksize': (3, 3), 'blur_sigma': 0,
                'canny_low': 10, 'canny_high': 30,
                'hough_threshold': 15, 'min_line_length_ratio': 0.3,
                'max_line_gap': 25, 'lower_region_ratio': 0.25
            },
            "Highly sensitive - may detect more but with more false positives"
        ),
        
        DetectionVariation(
            "Conservative", 
            {
                'blur_ksize': (5, 5), 'blur_sigma': 1,
                'canny_low': 50, 'canny_high': 100,
                'hough_threshold': 60, 'min_line_length_ratio': 0.7,
                'max_line_gap': 8, 'lower_region_ratio': 0.3
            },
            "Conservative - only strong, clear lines"
        ),
    ]
    
    # Process all variations
    process_with_variations(
        image_path=INPUT_IMAGE,
        variations=variations,
        output_dir="black_liquid_variations"
    )

if __name__ == "__main__":
    main()