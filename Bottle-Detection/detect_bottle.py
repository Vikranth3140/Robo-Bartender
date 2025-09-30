import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from typing import List, Dict, Tuple, Optional, Any

class BottleDetector:
    def __init__(self, model_path=None):
        """
        Initialize the bottle detector with YOLOv11n model
        """
        # Load YOLOv11n model (will download automatically if not present)
        if model_path is None:
            self.model = YOLO('yolo11n.pt')  # YOLOv11 nano model
        else:
            self.model = YOLO(model_path)
        
        # Define color ranges in HSV - expanded for better detection
        self.color_ranges = {
            'red': [
                (0, 50, 50), (10, 255, 255),      # Lower red range
                (170, 50, 50), (180, 255, 255)    # Upper red range
            ],
            'green': [(30, 40, 40), (90, 255, 255)],  # Expanded green range for lime/bright green
            'blue': [(90, 50, 50), (130, 255, 255)]
        }
    
    def debug_all_detections(self, image_path: str, confidence_threshold: float = 0.25) -> None:
        """
        Debug function to show all detections in the image
        """
        print(f"\n=== DEBUG: All detections in {image_path} ===")
        
        # Run inference with lower threshold
        results = self.model(image_path, conf=confidence_threshold)
        
        # COCO class names for reference
        coco_classes = {
            39: "bottle", 47: "cup", 46: "wine glass", 44: "bowl", 
            41: "wine bottle", 40: "vase", 67: "dining table"
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"Found {len(boxes)} objects:")
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = coco_classes.get(class_id, f"class_{class_id}")
                    
                    print(f"  {i+1}. Class {class_id} ({class_name}): {confidence:.3f}")
            else:
                print("No objects detected")
        print("=" * 50)

    def detect_bottles(self, image_path: str, confidence_threshold: float = 0.3) -> Tuple[List[Dict], Any, Any]:
        """
        Detect bottles in the image using YOLOv11n
        """
        # Run inference
        results = self.model(image_path, conf=confidence_threshold)
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        original_image = image.copy()
        
        bottle_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if detected object is a bottle or similar container
                    # Class 39: bottle, Class 47: cup, Class 40: vase
                    bottle_classes = [39, 47, 40]  # bottle, cup, and vase classes
                    if class_id in bottle_classes and confidence >= confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Extract bottle region
                        bottle_roi = image[y1:y2, x1:x2]
                        
                        # Determine bottle color
                        color = self.determine_bottle_color(bottle_roi)
                        
                        bottle_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'color': color,
                            'roi': bottle_roi
                        })
                        
                        # Draw bounding box and label
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Bottle ({color}): {confidence:.2f}"
                        cv2.putText(image, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return bottle_detections, image, original_image
    
    def analyze_bottle_hsv(self, bottle_roi: np.ndarray) -> None:
        """
        Analyze and print HSV statistics of bottle region for debugging
        """
        if bottle_roi.size == 0:
            print("  Empty bottle region")
            return
        
        hsv = cv2.cvtColor(bottle_roi, cv2.COLOR_BGR2HSV)
        
        # Get mean HSV values using proper typing
        mean_h = float(np.mean(hsv[:, :, 0].astype(np.float32)))
        mean_s = float(np.mean(hsv[:, :, 1].astype(np.float32)))
        mean_v = float(np.mean(hsv[:, :, 2].astype(np.float32)))
        
        print(f"  Mean HSV: H={mean_h:.1f}, S={mean_s:.1f}, V={mean_v:.1f}")
        
        # Show HSV ranges for reference
        print(f"  Current ranges - Green: H(30-90), S(40-255), V(40-255)")
    
    def determine_bottle_color(self, bottle_roi: np.ndarray) -> str:
        """
        Determine the dominant color of the bottle (red, green, or blue)
        """
        if bottle_roi.size == 0:
            return "unknown"
        
        # Debug: Analyze HSV values
        self.analyze_bottle_hsv(bottle_roi)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(bottle_roi, cv2.COLOR_BGR2HSV)
        
        color_scores = {}
        
        # Check each color
        for color_name, ranges in self.color_ranges.items():
            if color_name == 'red':
                # Red has two ranges due to hue wraparound
                lower1 = np.array(ranges[0], dtype=np.uint8)
                upper1 = np.array(ranges[1], dtype=np.uint8)
                lower2 = np.array(ranges[2], dtype=np.uint8)
                upper2 = np.array(ranges[3], dtype=np.uint8)
                
                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower = np.array(ranges[0], dtype=np.uint8)
                upper = np.array(ranges[1], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
            
            # Count pixels in color range
            color_pixels = cv2.countNonZero(mask)
            total_pixels = hsv.shape[0] * hsv.shape[1]
            
            # Calculate percentage
            if total_pixels > 0:
                color_scores[color_name] = color_pixels / total_pixels
            else:
                color_scores[color_name] = 0
        
        # Debug: Print color scores
        print(f"  Color analysis: Red={color_scores['red']:.3f}, Green={color_scores['green']:.3f}, Blue={color_scores['blue']:.3f}")
        
        # Find dominant color with lower threshold for better detection
        max_score = max(color_scores.values())
        if max_score > 0.05:  # Lowered threshold from 0.1 to 0.05 (5% of pixels)
            dominant_color = max(color_scores.keys(), key=lambda k: color_scores[k])
            print(f"  Detected color: {dominant_color} (score: {max_score:.3f})")
            return dominant_color
        else:
            print(f"  No dominant color found (max score: {max_score:.3f})")
            return "other"
    
    def process_image(self, image_path: str, output_path: Optional[str] = None, show_result: bool = True) -> Optional[List[Dict]]:
        """
        Process an image and detect bottles with their colors
        """
        print(f"Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found!")
            return None
        
        try:
            # Debug: Show all detections first
            self.debug_all_detections(image_path, confidence_threshold=0.25)
            
            # Detect bottles
            detections, result_image, original_image = self.detect_bottles(image_path)
            
            # Print results
            print(f"\nFound {len(detections)} bottle(s):")
            for i, detection in enumerate(detections, 1):
                bbox = detection['bbox']
                color = detection['color']
                confidence = detection['confidence']
                print(f"  Bottle {i}: {color} color, confidence: {confidence:.2f}, bbox: {bbox}")
            
            # Save result if output path provided
            if output_path and result_image is not None:
                cv2.imwrite(output_path, result_image)
                print(f"\nResult saved to: {output_path}")
            
            # Show result
            if show_result and result_image is not None and original_image is not None:
                # Resize images for display if they're too large
                height, width = result_image.shape[:2]
                if width > 1200:
                    scale = 1200 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    result_image = cv2.resize(result_image, (new_width, new_height))
                    original_image = cv2.resize(original_image, (new_width, new_height))
                
                # Display results
                cv2.imshow('Original Image', original_image)
                cv2.imshow('Bottle Detection Result', result_image)
                print("\nPress any key to close the windows...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return detections
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

def main():
    """
    Main function to run bottle detection
    """
    # Initialize detector
    detector = BottleDetector()
    
    # Example usage - you can modify this path
    image_path = input("Enter the path to your image file: ").strip('"')
    
    if not image_path:
        print("No image path provided. Exiting...")
        return
    
    # Process the image
    output_path = os.path.splitext(image_path)[0] + "_detected.jpg"
    detections = detector.process_image(image_path, output_path, show_result=True)
    
    if detections:
        print(f"\nDetection complete! Found {len(detections)} bottle(s).")
        print("Colors detected:", [d['color'] for d in detections])
    else:
        print("\nNo bottles detected in the image.")

if __name__ == "__main__":
    main()
