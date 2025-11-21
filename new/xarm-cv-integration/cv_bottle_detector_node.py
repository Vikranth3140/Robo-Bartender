# cv_bottle_detector_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import threading
from ultralytics import YOLO
import numpy as np


class BottleDetector:
    def __init__(self, model_path=None):
        self.model = YOLO(model_path or "yolo11n.pt")

        # HSV RANGES
        self.color_ranges = {
            'red': [
                (0, 50, 50), (10, 255, 255),
                (170, 50, 50), (180, 255, 255)
            ],
            'green': [(30, 40, 40), (90, 255, 255)],
            'blue': [(80, 40, 40), (140, 255, 255)]
        }

    def determine_bottle_color(self, bottle_roi):
        if bottle_roi.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(bottle_roi, cv2.COLOR_BGR2HSV)
        color_scores = {}

        for color_name, ranges in self.color_ranges.items():
            if color_name == 'red':
                lower1 = np.array(ranges[0], dtype=np.uint8)
                upper1 = np.array(ranges[1], dtype=np.uint8)
                lower2 = np.array(ranges[2], dtype=np.uint8)
                upper2 = np.array(ranges[3], dtype=np.uint8)
                mask = cv2.bitwise_or(
                    cv2.inRange(hsv, lower1, upper1),
                    cv2.inRange(hsv, lower2, upper2)
                )
            else:
                lower = np.array(ranges[0], dtype=np.uint8)
                upper = np.array(ranges[1], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)

            color_pixels = cv2.countNonZero(mask)
            total_pixels = hsv.shape[0] * hsv.shape[1]
            score = color_pixels / total_pixels if total_pixels > 0 else 0
            color_scores[color_name] = score

        # Decision
        max_score = max(color_scores.values())
        if max_score > 0.02:
            return max(color_scores, key=color_scores.get)
        elif color_scores['blue'] > 0.01:
            return 'blue'

        return 'other'

    def process_frame(self, frame, confidence_threshold=0.1):
        results = self.model(frame, conf=confidence_threshold)
        detections = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    bottle_classes = [39, 41, 40, 45, 75]
                    if class_id in bottle_classes and confidence >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        roi = frame[y1:y2, x1:x2]
                        color = self.determine_bottle_color(roi)

                        detections.append(color)

                        # Draw
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Bottle ({color})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return detections, frame


class BottleDetectionNode(Node):
    def __init__(self):
        super().__init__('cv_bottle_detector_node')

        # Publisher to the xArm pipeline
        self.publisher = self.create_publisher(String, '/bottle_detection_result', 10)

        self.detector = BottleDetector()
        self.cap = cv2.VideoCapture(0)

        self.get_logger().info("Multi-bottle detection node running...")

        threading.Thread(target=self.capture_loop, daemon=True).start()

    def capture_loop(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                continue

            detected_colors, annotated_frame = self.detector.process_frame(frame)

            # Keep only red, green, blue
            filtered = [c for c in detected_colors if c in ['red', 'green', 'blue']]
            filtered = list(set(filtered))  # remove duplicates

            msg = String()

            if filtered:
                msg.data = "detected: " + " ".join(filtered)
            else:
                msg.data = "detected: none"

            self.publisher.publish(msg)
            self.get_logger().info(f"Published: {msg.data}")

            cv2.imshow("Bottle Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = BottleDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
