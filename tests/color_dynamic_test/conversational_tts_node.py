import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import signal

from elevenlabs import ElevenLabs
from elevenlabs.client import Conversation
from elevenlabs.audio import DefaultAudioInterface


class ConversationalTTSNode(Node):
    def __init__(self):
        super().__init__("conversational_tts_node")

        # Store CV detected color
        self.detected_color = ""

        # Subscriber → from CV node
        self.color_sub = self.create_subscription(
            String,
            "/cv_detected_color",
            self.color_callback,
            10
        )

        # Publisher → send user request to CV
        self.color_pub = self.create_publisher(String, "/requested_bottle_color", 10)

        # Also subscribe to CV feedback (found/not found)
        self.feedback_sub = self.create_subscription(
            String,
            "/tts_feedback",
            self.feedback_callback,
            10
        )

        # ElevenLabs setup
        api_key = 'your_api_key_here'  # replace with your ElevenLabs API key
        agent_id = 'your_agent_id_here'  # replace with your agent ID

        self.elevenlabs = ElevenLabs(api_key=api_key)

        # Start conversation
        self.conversation = Conversation(
            self.elevenlabs,
            agent_id,
            requires_auth=bool(api_key),
            audio_interface=DefaultAudioInterface(),
            callback_user_transcript=self.evaluate_user_transcript,
        )

        self.get_logger().info("Conversational TTS Node started. Listening...")

        signal.signal(signal.SIGINT, lambda sig, frame: self.conversation.end_session())

    # --------------------------
    # CV DETECTED COLOR HANDLING
    # --------------------------
    def color_callback(self, msg):
        self.detected_color = msg.data
        self.get_logger().info(f"CV detected bottle color: {self.detected_color}")

    # --------------------------
    # CV FEEDBACK HANDLING
    # --------------------------
    def feedback_callback(self, msg):
        feedback = msg.data
        self.get_logger().info(f"Got CV feedback: {feedback}")

        # Speak with dynamic variable injection
        try:
            self.conversation.say(
                feedback,
                dynamic_variables={"bottle_color": self.detected_color}
            )
        except Exception as e:
            self.get_logger().error(f"TTS error: {e}")

    # --------------------------
    # HANDLE USER SPEECH INPUT
    # --------------------------
    def evaluate_user_transcript(self, text: str):
        self.get_logger().info(f"User said: {text}")

        # Simple color extraction
        color = None
        for c in ["red", "green", "blue"]:
            if c in text.lower():
                color = c
                break

        if color:
            self.get_logger().info(f"Publishing color request: {color}")
            self.color_pub.publish(String(data=color))

            # Also talk with dynamic variable
            self.conversation.say(
                f"Let me check for a {color} bottle.",
                dynamic_variables={"bottle_color": self.detected_color}
            )

        else:
            self.conversation.say(
                text,
                dynamic_variables={"bottle_color": self.detected_color}
            )


def main(args=None):
    rclpy.init(args=args)
    node = ConversationalTTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
