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

        # Arm connection
        self.declare_parameter('ip', '192.168.1.152')
        ip = self.get_parameter('ip').get_parameter_value().string_value
        self.get_logger().info(f'Connecting to xArm at {ip} ...')
        self.arm = XArmAPI(ip, baud_checkset=False)
        self._init_arm()

        # Hard-coded bottle positions
        self.bottle_positions = {
            "red":    [200, -100, 150, 180, 0, 0],
            "green":  [220,    0, 150, 180, 0, 0],
            "blue":   [200,  100, 150, 180, 0, 0]
        }
        self.cup_position  = [250, 0, 150, 180, 0, 0]
        self.home_position = [180, 0, 200, 180, 0, 0]

        # Detection subscriber
        self.subscription = self.create_subscription(
            String,
            '/bottle_detection_result',
            self.detection_callback,
            10
        )

        # Cooldown timestamp
        self.last_action_time = 0
        self.cooldown_seconds = 60  # 1 minute

        self.get_logger().info("xArm Node ready. Waiting for detection messages...")

    # ---------------------------------------------------------
    # ARM INITIALIZATION
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # CALLBACK: RECEIVE DETECTED BOTTLES
    # ---------------------------------------------------------


    def detection_callback(self, msg: String):
        text = msg.data.lower()
        current_time = time.time()

        # If cooldown is active, ignore any message
        if (current_time - self.last_action_time) < self.cooldown_seconds:
            self.get_logger().info("Cooldown active. Ignoring detection message.")
            return

        # Extract bottles
        bottles = self._extract_bottles(text)

        # If nothing detected, do not update cooldown
        if not bottles:
            self.get_logger().info("No valid bottles detected. Doing nothing.")
            return

        # Set cooldown immediately for any valid detection
        self.last_action_time = current_time

        self.get_logger().info(f"Received detection message: {text}")
        self.get_logger().info(f"Bottles detected: {bottles}")

        # Process bottles
        for bottle in bottles:
            self.handle_bottle(bottle)


    # _extract_bottles and handle_bottle remain the same as your previous code

    def _extract_bottles(self, text: str):
        """Extract bottle names: red, green, blue."""
        bottles = []
        for color in self.bottle_positions.keys():
            if color in text:
                bottles.append(color)
        return bottles

    # ---------------------------------------------------------
    # MAIN ROUTINE: bottle → cup → home
    # ---------------------------------------------------------
    def handle_bottle(self, bottle: str):
        self.get_logger().info(f"Handling bottle: {bottle}")

        bottle_pose = self.bottle_positions[bottle]
        self.move_to_pose(bottle_pose, label=f"{bottle} bottle")
        self.move_to_pose(self.cup_position, label="cup position")
        self.move_to_pose(self.home_position, label="home")

    # ---------------------------------------------------------
    # MOVEMENT WRAPPER
    # ---------------------------------------------------------
    def move_to_pose(self, pose, label=""):
        try:
            self.get_logger().info(f"Moving to {label}: {pose}")
            code = self.arm.set_position(
                *pose,
                speed=40,
                mvacc=500,
                radius=0.0,
                wait=True
            )
            if code != 0:
                self.get_logger().error(f"Movement to {label} failed, code={code}")
            else:
                self.get_logger().info(f"Reached {label}")

        except Exception as e:
            self.get_logger().error(
                f"Exception in move_to_pose ({label}): {e}\n{traceback.format_exc()}"
            )

    # ---------------------------------------------------------
    # SHUTDOWN
    # ---------------------------------------------------------
    def destroy_node(self):
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
