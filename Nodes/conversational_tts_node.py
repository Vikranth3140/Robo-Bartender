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

        # Cocktail-color mapping
        self.cocktails = {
            'martini': ['red', 'blue'],
            'mojito': ['green', 'blue'],
            'bloody mary': ['red'],
            'blue lagoon': ['blue'],
            'cosmopolitan': ['red', 'green']
        }

        # Setup publisher to CV node
        self.color_pub = self.create_publisher(String, '/requested_bottle_color', 10)

        # Subscriber for feedback from CV node
        self.feedback_sub = self.create_subscription(
            String,
            '/tts_feedback',
            self.feedback_callback,
            10
        )

        # ElevenLabs setup
        api_key = 'sk_de44b5d768232e11b1a644e83b93c77701274225f2b3ab13'
        agent_id = 'agent_3801kaes6nvae0ht7qcbmtd2f2q4'

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

    def feedback_callback(self, msg: String):
        feedback = msg.data
        self.get_logger().info(f"Received CV feedback: {feedback}")
        try:
            self.conversation.say(feedback)
        except Exception as e:
            self.get_logger().error(f"Failed to speak feedback: {e}")

    def evaluate_user_transcript(self, transcript):
        self.get_logger().info(f"User said: {transcript}")
        t = transcript.lower()

        # Find matching cocktail
        cocktail_name = None
        for c in self.cocktails.keys():
            if c in t:
                cocktail_name = c
                break

        if cocktail_name:
            colors = self.cocktails[cocktail_name]
            msg = String()
            msg.data = " ".join(colors)  # publish all colors
            self.color_pub.publish(msg)
            self.get_logger().info(f"Published requested colors for cocktail: {msg.data}")
        else:
            self.get_logger().info("No known cocktail detected.")

def main(args=None):
    rclpy.init(args=args)
    node = ConversationalTTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
