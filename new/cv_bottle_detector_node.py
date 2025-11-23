import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import threading
from ultralytics import YOLO
import numpy as np
import time

class BottleDetector:
    def __init__(self, model_path=None):
        self.model = YOLO(model_path or "yolo11n.pt")
        self.color_ranges = {
            'red': [(0,50,50),(10,255,255),(170,50,50),(180,255,255)],
            'green': [(30,40,40),(90,255,255)],
            'blue': [(80,40,40),(140,255,255)],
        }

    def determine_bottle_color(self, roi):
        if roi.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        scores = {}
        for name, ranges in self.color_ranges.items():
            if name == 'red':
                m1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                m2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
                mask = cv2.bitwise_or(m1, m2)
            else:
                mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
            score = cv2.countNonZero(mask) / (hsv.shape[0] * hsv.shape[1])
            scores[name] = score
        if max(scores.values()) > 0.02:
            return max(scores, key=scores.get)
        return "other"

    def process_frame(self, frame, conf=0.1):
        results = self.model(frame, conf=conf, verbose=False)
        detections = []
        for res in results:
            for box in res.boxes:
                cid = int(box.cls[0])
                if cid in [39,41,40,45,75]:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    roi = frame[y1:y2, x1:x2]
                    detections.append(self.determine_bottle_color(roi))
        return detections


class BottleDetectionNode(Node):
    def __init__(self):
        super().__init__('cv_bottle_detector_node')

        self.publisher = self.create_publisher(String, '/bottle_detection_result', 10)
        self.detector = BottleDetector()
        self.cap = cv2.VideoCapture(0)

        self.requested_colors = []
        self.active_request = False

        self.create_subscription(String, '/requested_bottle_color', self.requested_colors_callback, 10)
        self.create_subscription(String, '/order_status', self.order_done_callback, 10)

        threading.Thread(target=self.capture_loop, daemon=True).start()

    def requested_colors_callback(self, msg: String):
        self.requested_colors = msg.data.split()
        if self.requested_colors:
            self.active_request = True
            self.get_logger().info("CV ACTIVE: New request received.")
        else:
            self.active_request = False
            self.get_logger().info("CV PAUSED: No active request.")

    def order_done_callback(self, msg: String):
        if msg.data == "done":
            self.get_logger().info("Order finished — CV PAUSED.")
            self.active_request = False
            self.requested_colors = []

    def capture_loop(self):
        while rclpy.ok():
            # ---- paused ----
            if not self.active_request:
                time.sleep(0.1)
                continue

            # ---- active ----
            ret, frame = self.cap.read()
            if not ret:
                continue

            detected = self.detector.process_frame(frame)
            detected_set = set(detected)
            req_set = set(self.requested_colors)

            if req_set and req_set.issubset(detected_set):
                msg = String()
                msg.data = "detected: " + " ".join(sorted(req_set))
                self.publisher.publish(msg)
                self.get_logger().info(f"Published: {msg.data}")

        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = BottleDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
