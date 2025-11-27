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
                mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                                      cv2.inRange(hsv, lower2, upper2))
            else:
                lower = np.array(ranges[0], dtype=np.uint8)
                upper = np.array(ranges[1], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)

            color_pixels = cv2.countNonZero(mask)
            total_pixels = hsv.shape[0] * hsv.shape[1]
            color_scores[color_name] = color_pixels / total_pixels if total_pixels > 0 else 0

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

                    # Bottle-like classes from YOLO
                    bottle_classes = [39, 41, 40, 45, 75]

                    if class_id in bottle_classes and confidence >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bottle_roi = frame[y1:y2, x1:x2]
                        color = self.determine_bottle_color(bottle_roi)

                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'color': color,
                            'confidence': confidence
                        })

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Bottle ({color})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return detections, frame


class BottleDetectionNode(Node):
    def __init__(self):
        super().__init__('cv_bottle_detector_node')

        # Subscriber: what the user wants
        self.subscription = self.create_subscription(
            String,
            '/requested_bottle_color',
            self.listener_callback,
            10
        )

        # Old robot pipeline
        self.result_pub = self.create_publisher(String, '/bottle_detection_result', 10)

        # Conversational feedback
        self.feedback_pub = self.create_publisher(String, '/tts_feedback', 10)

        # NEW: Publish every detected bottle color
        self.color_pub = self.create_publisher(String, '/cv_detected_color', 10)

        self.detector = BottleDetector()
        self.color_requested = None

        self.cap = cv2.VideoCapture(0)
        self.get_logger().info("Bottle Detection Node is running…")

        threading.Thread(target=self.capture_loop, daemon=True).start()

    def listener_callback(self, msg):
        self.color_requested = msg.data.strip().lower()
        self.get_logger().info(f"User requested bottle color: {self.color_requested}")

    def capture_loop(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                continue

            detections, annotated_frame = self.detector.process_frame(frame)

            # Publish ALL detected bottle colors continuously
            if detections:
                detected_colors = set(d['color'] for d in detections)
                for c in detected_colors:
                    msg = String()
                    msg.data = c
                    self.color_pub.publish(msg)

            # Handle user request → respond ONCE
            if self.color_requested:
                found = any(d['color'] == self.color_requested for d in detections)

                result_msg = String()
                feedback_msg = String()

                if found:
                    result_msg.data = f"{self.color_requested} bottle detected!"
                    feedback_msg.data = "Found your bottle, pouring now."
                else:
                    result_msg.data = f"No {self.color_requested} bottle found."
                    feedback_msg.data = "I cannot see that bottle right now."

                # Publish detection result
                self.result_pub.publish(result_msg)
                self.get_logger().info(result_msg.data)

                # Push conversational feedback
                self.feedback_pub.publish(feedback_msg)
                self.get_logger().info(f"TTS feedback sent: {feedback_msg.data}")

                # Reset request so it only answers once
                self.color_requested = None

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
