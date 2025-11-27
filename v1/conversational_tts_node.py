# conversational_tts_node.py
import os
import signal
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from threading import Thread

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface


class ConversationalTTSNode(Node):
    def __init__(self):
        super().__init__('conversational_tts_node')

        # -------------------------------
        # Cocktail → required bottle colors
        # -------------------------------
        self.cocktails = {
            'martini': ['red', 'blue'],
            'mojito': ['green', 'blue'],
            'bloody mary': ['red'],
            'blue lagoon': ['blue'],
            'cosmopolitan': ['red', 'green']
        }

        # -----------------------------------------
        # Publisher → CV node (requested colors)
        # -----------------------------------------
        self.color_pub = self.create_publisher(String, '/requested_bottle_color', 10)

        # -----------------------------------------
        # Subscriber → TTS feedback from xArm node
        # -----------------------------------------
        self.feedback_sub = self.create_subscription(
            String,
            '/tts_feedback',
            self.feedback_callback,
            10
        )

        # ----------------------------
        # ElevenLabs initialization
        # ----------------------------
        api_key = 'your_api_key_here'  # replace with your ElevenLabs API key
        agent_id = 'your_agent_id_here'  # replace with your agent ID

        self.elevenlabs = ElevenLabs(api_key=api_key)

        # ----------------------------------------------
        # Start a conversation session in a background thread
        # ----------------------------------------------
        self.conversation_thread = Thread(target=self.start_conversation, daemon=True)
        self.conversation_thread.start()

        self.get_logger().info("Conversational TTS node initialized.")

    # -----------------------------------------------------------------
    # ElevenLabs conversation runner (runs in background thread)
    # -----------------------------------------------------------------
    def start_conversation(self):
        self.conversation = Conversation(
            self.elevenlabs,
            "agent_3801kaes6nvae0ht7qcbmtd2f2q4",
            requires_auth=True,
            audio_interface=DefaultAudioInterface(),
            callback_user_transcript=self.evaluate_user_transcript,
        )

        # signal.signal(signal.SIGINT, lambda sig, frame: self.conversation.end_session())

        self.get_logger().info("Starting ElevenLabs conversation session...")
        self.conversation.start_session()
        self.conversation.wait_for_session_end()

    # ---------------------------------------------------------
    # Feedback from xArm (CV-validated) → spoken out
    # ---------------------------------------------------------
    def feedback_callback(self, msg: String):
        feedback = msg.data
        self.get_logger().info(f"Received CV feedback: {feedback}")

        try:
            self.conversation.say(feedback)
        except Exception as e:
            self.get_logger().error(f"Failed to speak feedback: {e}")

    # ---------------------------------------------------------
    # When user speaks → ElevenLabs gives transcript here
    # ---------------------------------------------------------
    def evaluate_user_transcript(self, transcript):
        self.get_logger().info(f"User said: {transcript}")
        text = transcript.lower()

        # Find cocktail
        cocktail = None
        for name in self.cocktails:
            if name in text:
                cocktail = name
                break

        if not cocktail:
            self.get_logger().info("No known cocktail detected.")
            return

        # Publish required bottle colors
        colors = self.cocktails[cocktail]
        msg = String()
        msg.data = " ".join(colors)
        self.color_pub.publish(msg)

        self.get_logger().info(f"Published requested colors for cocktail: {msg.data}")

        # Also inform user via voice
        try:
            self.conversation.say(
                f"Okay, I will prepare a {cocktail}. I need the colors {msg.data}."
            )
        except:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ConversationalTTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
