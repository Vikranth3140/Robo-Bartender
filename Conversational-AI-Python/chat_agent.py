import os
import signal
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ELEVENLABS_API_KEY")
AGENT_ID = os.getenv("AGENT_ID")

elevenlabs = ElevenLabs(api_key=API_KEY)

def evaluate_user_transcript(transcript):
    if 'red bottle' in transcript.lower():
        print("User requested for a red bottle.")
    elif 'blue bottle' in transcript.lower():
        print("User requested for a blue bottle.")
    else:
        print("User requested for a green bottle.")
    print(f"You said: {transcript}")

conversation = Conversation(
    elevenlabs,
    AGENT_ID,
    requires_auth=bool(API_KEY),
    audio_interface=DefaultAudioInterface(),
    callback_agent_response=lambda response: print(f"Agent: {response}"),
    callback_agent_response_correction=lambda original, corrected: print(f"Agent: {original} -> {corrected}"),
    callback_user_transcript=evaluate_user_transcript
)

signal.signal(signal.SIGINT, lambda sig, frame: conversation.end_session())

conversation.start_session()
conversation_id = conversation.wait_for_session_end()
print(f"Conversation ID: {conversation_id}")
