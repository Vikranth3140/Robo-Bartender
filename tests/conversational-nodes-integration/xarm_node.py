#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from xarm.wrapper import XArmAPI
import time
import threading
import traceback


class XArmNode(Node):
    def __init__(self):
        super().__init__('xarm_node')

        # ----- parameters -----
        self.declare_parameter('ip', '192.168.1.152')
        ip = self.get_parameter('ip').get_parameter_value().string_value
        self.get_logger().info(f'Connecting to xArm at {ip} ...')

        # ----- connect arm -----
        self.arm = XArmAPI(ip, baud_checkset=False)
        self._init_arm()

        # ----- workspace positions -----
        self.BOTTLES = {
            "green": {"pos": (418.9, 30.8, 153),  "rpy": (90.7, -68.2, 88.2)},
            "red":   {"pos": (380, -140.2, 153),  "rpy": (90.7, -68.2, 88.2)},
            "blue":  {"pos": (225.5, -140.2, 153), "rpy": (90.7, -68.2, 88.2)},
        }

        self.CUP_POS = (240.4, 19.1, 153)
        self.CUP_RPY = (89, -40.2, 91.4)

        self.Z_APPROACH = 220
        self.Z_PICK = 150
        self.Z_POUR = 180

        # ----- control state -----
        self.requested_colors = []
        self.cv_detected_colors = []
        self.processing = False   # no cooldown anymore

        # ----- publishers -----
        self.status_pub = self.create_publisher(String, '/order_status', 10)

        # ----- subscribers -----
        self.create_subscription(String, '/requested_bottle_color', self.requested_colors_callback, 10)
        self.create_subscription(String, '/bottle_detection_result', self.cv_detection_callback, 10)

        self.get_logger().info("xArm Node ready. Waiting for requested + detection messages...")

    # ------------------------------
    # ARM init
    # ------------------------------
    def _init_arm(self):
        try:
            self.arm.clean_warn()
            self.arm.clean_error()
            self.arm.motion_enable(True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(1)
            self.get_logger().info('xArm initialized successfully.')
        except Exception as e:
            self.get_logger().error(f"Failed to initialize xArm: {e}")
            self.get_logger().error(traceback.format_exc())

    # ------------------------------
    # Subscribers
    # ------------------------------
    def requested_colors_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        colors = [c.lower() for c in text.split() if c.lower() in self.BOTTLES]
        self.requested_colors = colors

        self.get_logger().info(f"[requested] {self.requested_colors}")
        self.try_execute()

    def cv_detection_callback(self, msg: String):
        text = msg.data.strip().lower().replace("detected:", "").strip()
        if not text:
            return

        detected = [c for c in text.split() if c in self.BOTTLES]
        self.cv_detected_colors = list(set(detected))

        self.get_logger().info(f"[cv] detected: {self.cv_detected_colors}")
        self.try_execute()

    # ------------------------------
    # Execution gating
    # ------------------------------
    def try_execute(self):
        if self.processing:
            return

        if not self.requested_colors or not self.cv_detected_colors:
            return

        req = set(self.requested_colors)
        det = set(self.cv_detected_colors)

        # must match EXACTLY
        if req.issubset(det):
            self.processing = True
            worker = threading.Thread(
                target=self._execute_pipeline_thread,
                args=(list(self.requested_colors),),
                daemon=True
            )
            worker.start()

    # ------------------------------
    # Pipeline
    # ------------------------------
    def _execute_pipeline_thread(self, ordered_colors):
        try:
            for color in ordered_colors:
                self._pick_pour_return(color)

            # return arm home
            self.arm.reset(wait=True)

            msg = String()
            msg.data = "done"
            self.status_pub.publish(msg)
            self.get_logger().info("Published: order complete")

        except Exception as e:
            self.get_logger().error(f"Pipeline error: {e}")

        finally:
            self.processing = False  # allow next request immediately

    # ------------------------------
    # Movement helper
    # ------------------------------
    def _move(self, x, y, z, rx, ry, rz, speed=50):
        self.arm.set_position(
            x=x, y=y, z=z,
            roll=rx, pitch=ry, yaw=rz,
            speed=speed, wait=True
        )

    # ------------------------------
    # Pick + Pour + Return
    # ------------------------------
    def _pick_pour_return(self, color):
        data = self.BOTTLES[color]
        bx, by, bz = data["pos"]
        brx, bry, brz = data["rpy"]

        TILT_RY = bry - 25
        POUR_RY = TILT_RY - 40

        # pick
        self._move(bx, by, self.Z_APPROACH, brx, bry, brz)
        self._move(bx, by, self.Z_PICK, brx, bry, brz)
        self._move(bx, by, self.Z_PICK, brx, TILT_RY, brz)
        self._move(bx, by, bz, brx, TILT_RY, brz)
        self._move(bx, by, self.Z_APPROACH, brx, TILT_RY, brz)

        # pour
        cx, cy, cz = self.CUP_POS
        self._move(cx, cy, self.Z_APPROACH, brx, TILT_RY, brz)
        self._move(cx, cy, self.Z_POUR, brx, TILT_RY, brz)
        self._move(cx, cy, self.Z_POUR, brx, POUR_RY, brz)
        self._move(cx, cy, self.Z_POUR, brx, TILT_RY, brz)
        self._move(cx, cy, self.Z_APPROACH, brx, TILT_RY, brz)

        # return bottle
        self._move(bx, by, self.Z_APPROACH, brx, TILT_RY, brz)
        self._move(bx, by, self.Z_PICK, brx, TILT_RY, brz)
        self._move(bx, by, bz, brx, TILT_RY, brz)
        self._move(bx, by, bz, brx, bry, brz)
        self._move(bx, by, self.Z_APPROACH, brx, bry, brz)


def main(args=None):
    rclpy.init(args=args)
    node = XArmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
