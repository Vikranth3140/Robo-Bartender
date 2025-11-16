#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from xarm.wrapper import XArmAPI
from xarm import version
import time
import traceback


class XArmNode(Node):
    def __init__(self):
        super().__init__('xarm_node')

        # --- Connect to the arm ---
        self.declare_parameter('ip', '192.168.1.152')
        ip = self.get_parameter('ip').get_parameter_value().string_value

        self.get_logger().info(f'Connecting to xArm at {ip} ...')
        self.arm = XArmAPI(ip, baud_checkset=False)
        self._init_arm()

        # --- Create subscriber ---
        self.subscription = self.create_subscription(
            String,
            '/bottle_detection_result',
            self.detection_callback,
            10
        )
        self.get_logger().info("xArm Node ready. Waiting for detection messages...")

    def _init_arm(self):
        """Initialize the xArm safely."""
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

    def detection_callback(self, msg: String):
        """Respond to bottle detection messages."""
        text = msg.data.lower()
        self.get_logger().info(f"Received detection message: {text}")

        # Check for green bottle detection
        if "green bottle detected" in text:
            self.get_logger().info("Green bottle detected — moving arm gently forward (test movement).")
            self.move_forward(20)  # Move 5 mm forward (safe small motion)
        else:
            self.get_logger().info("No green bottle detected, staying put.")

    def move_forward(self, distance_mm: float):
        """Move arm forward by a small distance in mm along the X-axis."""
        try:
            code, position = self.arm.get_position(is_radian=False)
            if code != 0:
                self.get_logger().error(f"Failed to get current position, code={code}")
                return

            x, y, z, roll, pitch, yaw = position
            new_x = x + distance_mm
            new_pose = [new_x, y, z, roll, pitch, yaw]

            self.get_logger().info(f"Executing safe test movement to: {new_pose}")
            code = self.arm.set_position(
                *new_pose,
                speed=20,     # very slow, safe speed
                mvacc=500,    # low acceleration
                radius=0.0,
                wait=True
            )

            if code == 0:
                self.get_logger().info("Safe forward test movement complete.")
            else:
                self.get_logger().error(f"Movement failed with code {code}")

        except Exception as e:
            self.get_logger().error(
                f"Exception in move_forward: {e}\n{traceback.format_exc()}"
            )


    def destroy_node(self):
        """Ensure the arm disconnects safely when node shuts down."""
        self.get_logger().info("Disconnecting xArm...")
        self.arm.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = XArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    print(f'[INFO] Using xArm-Python-SDK v{version.__version__}')
    main()
