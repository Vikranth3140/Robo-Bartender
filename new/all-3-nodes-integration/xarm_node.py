#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from xarm.wrapper import XArmAPI
from xarm import version
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

        # ----- workspace positions (update if needed) -----
        self.BOTTLES = {
            "green": {"pos": (418.9, 30.8, 153),  "rpy": (90.7, -68.2, 88.2)},
            "red":   {"pos": (380, -140.2, 153),  "rpy": (90.7, -68.2, 88.2)},
            "blue":  {"pos": (225.5, -140.2, 153), "rpy": (90.7, -68.2, 88.2)},
        }
        self.CUP_POS = (240.4, 19.1, 153)
        self.CUP_RPY = (89, -40.2, 91.4)

        # heights
        self.Z_APPROACH = 220
        self.Z_PICK = 150
        self.Z_POUR = 180

        # ----- control state -----
        self.requested_colors = []     # list, order-preserving from conversational node
        self.cv_detected_colors = []   # list from CV (no guaranteed order)
        self.processing = False
        self.cooldown_active = False
        self.cooldown_seconds = 60

        # ----- subscribers -----
        # conversational requested colors
        self.create_subscription(
            String,
            '/requested_bottle_color',
            self.requested_colors_callback,
            10
        )

        # CV detection results
        self.create_subscription(
            String,
            '/bottle_detection_result',
            self.cv_detection_callback,
            10
        )

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
        # conversational node publishes like "green blue"
        if not text:
            self.get_logger().info("Requested colors empty.")
            return

        # parse and store list preserving order
        colors = text.split()
        colors = [c.lower() for c in colors if c.lower() in self.BOTTLES.keys()]
        self.requested_colors = colors
        self.get_logger().info(f"[requested] {self.requested_colors}")

        # Try to execute if conditions met
        self.try_execute()

    def cv_detection_callback(self, msg: String):
        text = msg.data.strip().lower()
        # cv publishes like "detected: green blue" or "green"
        # clean and parse
        text = text.replace("detected:", "").strip()
        if not text:
            self.get_logger().debug("CV detection empty.")
            return

        detected = text.split()
        detected = [c for c in detected if c in self.BOTTLES.keys()]
        # store unique set (we only need membership)
        # but keep as list for logging
        self.cv_detected_colors = list(set(detected))
        self.get_logger().info(f"[cv] detected: {self.cv_detected_colors}")

        # Try to execute if conditions met
        self.try_execute()

    # ------------------------------
    # Execution gating
    # ------------------------------
    def try_execute(self):
        """Check gating: requested set must be subset of detected set,
        not currently processing, and not in cooldown."""
        if self.processing:
            self.get_logger().info("Already processing a request — ignoring new trigger.")
            return

        if self.cooldown_active:
            self.get_logger().info("Cooldown active — ignoring new trigger.")
            return

        if not self.requested_colors:
            self.get_logger().info("No requested colors set — waiting.")
            return

        if not self.cv_detected_colors:
            self.get_logger().info("No CV detections — waiting.")
            return

        # require all requested colors to be present in CV detection
        requested_set = set(self.requested_colors)
        detected_set = set(self.cv_detected_colors)

        if requested_set.issubset(detected_set):
            # proceed: run the pipeline in a new thread to avoid blocking callbacks
            self.get_logger().info(f"Match found — executing pipeline for: {self.requested_colors}")
            self.processing = True
            worker = threading.Thread(target=self._execute_pipeline_thread, args=(list(self.requested_colors),), daemon=True)
            worker.start()
        else:
            self.get_logger().info(f"Requested colors {self.requested_colors} not fully detected by CV {self.cv_detected_colors}. Waiting.")

    # ------------------------------
    # Pipeline runner (thread)
    # ------------------------------
    def _execute_pipeline_thread(self, ordered_colors):
        """Runs full pick->pour->return sequence for each color in ordered_colors."""
        try:
            for color in ordered_colors:
                if color not in self.BOTTLES:
                    self.get_logger().warn(f"Unknown color requested: {color} — skipping")
                    continue
                self.get_logger().info(f"Starting routine for: {color}")
                self._pick_pour_return(color)
                self.get_logger().info(f"Finished routine for: {color}")

            # after finishing all requested colors, go home (reset)
            self.get_logger().info("All requested colors processed. Resetting arm to home.")
            try:
                self.arm.reset(wait=True)
            except Exception as e:
                self.get_logger().error(f"Error while resetting arm: {e}")

        except Exception as e:
            self.get_logger().error(f"Exception during pipeline: {e}\n{traceback.format_exc()}")
        finally:
            # start cooldown
            self.processing = False
            self._start_cooldown()

    # ------------------------------
    # Cooldown
    # ------------------------------
    def _start_cooldown(self):
        self.cooldown_active = True
        self.get_logger().info(f"Starting cooldown for {self.cooldown_seconds} seconds.")
        t = threading.Timer(self.cooldown_seconds, self._end_cooldown)
        t.daemon = True
        t.start()

    def _end_cooldown(self):
        self.cooldown_active = False
        # clear requested and detected lists (so next conversational message is fresh)
        self.requested_colors = []
        self.cv_detected_colors = []
        self.get_logger().info("Cooldown ended. Ready for new requests.")

    # ------------------------------
    # Motion helper
    # ------------------------------
    def _move(self, x, y, z, rx, ry, rz, speed=50):
        try:
            code = self.arm.set_position(
                x=x, y=y, z=z,
                roll=rx, pitch=ry, yaw=rz,
                speed=speed, wait=True
            )
            if code != 0:
                self.get_logger().error(f"Move returned non-zero code: {code} for pos {(x,y,z,rx,ry,rz)}")
            else:
                self.get_logger().debug(f"Moved to {(x,y,z,rx,ry,rz)}")
        except Exception as e:
            self.get_logger().error(f"Exception while moving: {e}")

    # ------------------------------
    # Pick / pour / return sequence for a single color
    # ------------------------------
    def _pick_pour_return(self, color):
        """Single bottle pick, pour, place (uses no gripper calls)."""
        data = self.BOTTLES[color]
        bx, by, bz = data["pos"]
        brx, bry, brz = data["rpy"]

        # 1) move above bottle
        self._move(bx, by, self.Z_APPROACH, brx, bry, brz, speed=80)

        # 2) pre-pick
        self._move(bx, by, self.Z_PICK, brx, bry, brz, speed=40)

        # 3) tilt for gripping (maintain orientation afterward)
        TILT_RY = bry - 25
        self._move(bx, by, self.Z_PICK, brx, TILT_RY, brz, speed=25)
        time.sleep(0.5)

        # 4) move down to bottle
        self._move(bx, by, bz, brx, TILT_RY, brz, speed=20)
        time.sleep(0.5)

        # (pick assumed, no gripper)
        # 5) lift up with tilt maintained
        self._move(bx, by, self.Z_APPROACH, brx, TILT_RY, brz, speed=60)

        # 6) move above cup (keep tilt to keep bottle stable)
        cx, cy, cz = self.CUP_POS
        self._move(cx, cy, self.Z_APPROACH, brx, TILT_RY, brz, speed=80)

        # 7) move down to pour level
        self._move(cx, cy, self.Z_POUR, brx, TILT_RY, brz, speed=40)

        # 8) pour by tilting pitch (RY)
        POUR_RY = TILT_RY - 40
        self._move(cx, cy, self.Z_POUR, brx, POUR_RY, brz, speed=25)
        time.sleep(1.5)

        # 9) Reset tilt steady (back to stable tilt)
        self._move(cx, cy, self.Z_POUR, brx, TILT_RY, brz, speed=25)

        # 10) lift up from cup
        self._move(cx, cy, self.Z_APPROACH, brx, TILT_RY, brz, speed=60)

        # 11) return bottle to original location (still tilted)
        self._move(bx, by, self.Z_APPROACH, brx, TILT_RY, brz, speed=80)
        self._move(bx, by, self.Z_PICK,     brx, TILT_RY, brz, speed=40)
        self._move(bx, by, bz,               brx, TILT_RY, brz, speed=20)
        time.sleep(0.5)

        # 12) restore original upright orientation at place
        self._move(bx, by, bz, brx, bry, brz, speed=25)
        time.sleep(0.5)

        # 13) lift away
        self._move(bx, by, self.Z_APPROACH, brx, bry, brz, speed=60)

    # ------------------------------
    # shutdown
    # ------------------------------
    def destroy_node(self):
        self.get_logger().info("Disconnecting xArm...")
        try:
            self.arm.disconnect()
        except Exception:
            pass
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
