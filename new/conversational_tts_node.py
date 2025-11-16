# conversational_tts_node.py
import os
import signal
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface


class ConversationalTTSNode(Node):
    def __init__(self):
        super().__init__('conversational_tts_node')

        # Setup publisher to CV node
        self.color_pub = self.create_publisher(String, '/requested_bottle_color', 10)

        # NEW: Subscriber to receive feedback from CV node
        self.feedback_sub = self.create_subscription(
            String,
            '/tts_feedback',
            self.feedback_callback,
            10
        )

        # ElevenLabs setup
        api_key = 'sk_80dd87c8c862e73011f0bcfe3f6950a053e77ef1dc72962e'
        self.get_logger().info(api_key)
        agent_id = 'agent_2401ka6c0s1pe8v9xge3nvmjpr87'

        self.elevenlabs = ElevenLabs(api_key=api_key)

        # Start conversational session
        self.conversation = Conversation(
            self.elevenlabs,
            agent_id,
            requires_auth=bool(api_key),
            audio_interface=DefaultAudioInterface(),
            callback_user_transcript=self.evaluate_user_transcript,
        )

        signal.signal(signal.SIGINT, lambda sig, frame: self.conversation.end_session())

        self.get_logger().info("Starting conversation session...")
        self.conversation.start_session()
        self.conversation.wait_for_session_end()

    # NEW: Speak responses coming from bottle detector
    def feedback_callback(self, msg: String):
        feedback = msg.data
        self.get_logger().info(f"Received CV feedback: {feedback}")

        try:
            self.conversation.say(feedback)
        except Exception as e:
            self.get_logger().error(f"Failed to speak feedback: {e}")

    # Detect color request from user speech
    def evaluate_user_transcript(self, transcript):
        self.get_logger().info(f"User said: {transcript}")
        color = None

        t = transcript.lower()

        if 'red bottle' in t:
            color = 'red'
        elif 'blue bottle' in t:
            color = 'blue'
        elif 'green bottle' in t:
            color = 'green'

        if color:
            msg = String()
            msg.data = color
            self.color_pub.publish(msg)
            self.get_logger().info(f"Published color request: {color}")
        else:
            self.get_logger().info("No color detected.")

def main(args=None):
    rclpy.init(args=args)
    node = ConversationalTTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
