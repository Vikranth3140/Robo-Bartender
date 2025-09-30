"""
Debug script to test detection of sports bottles and water bottles
"""
from detect_bottle import BottleDetector
import os

def debug_bottle_detection():
    """Debug bottle detection for sports/water bottles"""
    print("🔍 Debug: Sports Bottle Detection Test")
    print("=" * 60)
    
    # Initialize detector
    detector = BottleDetector()
    
    # Get image path
    image_path = input("Enter path to your red bottle image: ").strip().strip('"')
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n📊 Testing different confidence thresholds...")
    
    # Test with different confidence levels
    for conf in [0.1, 0.25, 0.4, 0.5]:
        print(f"\n--- Confidence threshold: {conf} ---")
        try:
            detections, _, _ = detector.detect_bottles(image_path, confidence_threshold=conf)
            if detections:
                print(f"✅ Found {len(detections)} bottle(s)")
                for i, det in enumerate(detections, 1):
                    print(f"   {i}. Color: {det['color']}, Conf: {det['confidence']:.3f}")
            else:
                print("❌ No bottles detected")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎯 Testing with expanded object classes...")
    
    # Test the updated detection
    output_path = os.path.splitext(image_path)[0] + "_debug_detection.jpg"
    detections = detector.process_image(image_path, output_path, show_result=False)
    
    if detections:
        print(f"\n🎉 SUCCESS! Detected {len(detections)} bottle(s)")
        for detection in detections:
            print(f"   Color: {detection['color']}")
            print(f"   Confidence: {detection['confidence']:.3f}")
            print(f"   Bounding box: {detection['bbox']}")
    else:
        print(f"\n💡 Suggestions:")
        print("1. Try a different angle or lighting")
        print("2. Ensure the bottle is clearly visible")
        print("3. Check if the bottle shape is too different from training data")

if __name__ == "__main__":
    debug_bottle_detection()