# Bottle Detection and Color Classification

This project uses YOLOv11n to detect bottles in images and classify their colors as red, green, or blue.

## Features

- **Bottle Detection**: Uses YOLOv11n (nano) model for fast and accurate bottle detection
- **Color Classification**: Analyzes detected bottles to determine if they are red, green, or blue
- **Visual Output**: Displays detection results with bounding boxes and color labels
- **Batch Processing**: Can process multiple images programmatically

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Interactive Mode
Run the main script for interactive bottle detection:
```bash
python detect_bottle.py
```

### Option 2: Programmatic Usage
```python
from detect_bottle import BottleDetector

# Initialize detector
detector = BottleDetector()

# Process an image
detections = detector.process_image("path/to/your/image.jpg", 
                                   output_path="result.jpg", 
                                   show_result=True)

# Print results
for detection in detections:
    print(f"Found {detection['color']} bottle with confidence {detection['confidence']:.2f}")
```

## How It Works

1. **Object Detection**: YOLOv11n detects objects in the image and filters for bottles (COCO class 39)
2. **Color Analysis**: For each detected bottle:
   - Extracts the bottle region from the image
   - Converts to HSV color space for better color detection
   - Analyzes pixel colors in predefined ranges for red, green, and blue
   - Determines the dominant color based on pixel percentage

## Color Ranges (HSV)

- **Red**: Hue 0-10 and 170-180 (due to wraparound)
- **Green**: Hue 40-80
- **Blue**: Hue 100-130

## Output

The system provides:
- Console output with detection results
- Annotated image with bounding boxes and labels
- Saved result image (optional)
- Detection data including coordinates, confidence, and color

## Notes

- The first run will download the YOLOv11n model (~6MB)
- Works best with clear, well-lit images of bottles
- Minimum 10% of pixels must match a color range for classification
- Bottles not matching red/green/blue are classified as "other"