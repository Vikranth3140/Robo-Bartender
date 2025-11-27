#!/usr/bin/env python3
import os
import io
import json
import base64
import time
import threading
import traceback
from typing import List

import numpy as np
import sounddevice as sd
import websockets
import asyncio

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from groq import Groq

# -----------------------
# CONFIG (set via env or edit here)
# -----------------------
ELEVEN_API_KEY = "your_api_key_here"  # required
GROQ_API_KEY = "your_groq_api_key_here"          # required

# STT settings
ELEVEN_STT_MODEL_ID = "scribe_v2_realtime"  # or the realtime model you prefer
STT_SAMPLE_RATE = 16000
STT_CHUNK_SEC = 0.5

# TTS settings
TTS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"   # replace with your voice id
TTS_MODEL_ID = "eleven_flash_v2_5"
TTS_OUTPUT_SR = 22050                    # sample rate of TTS PCM returned

# Groq settings
GROQ_MODEL = "moonshotai/kimi-k2-instruct"

# Cocktail mapping
COCKTAILS = {
    'martini': ['red', 'blue'],
    'mojito': ['green', 'blue'],
    'bloody mary': ['red'],
    'blue lagoon': ['blue'],
    'cosmopolitan': ['red', 'green']
}

# -----------------------
# Helpers: audio capture
# -----------------------
def record_pcm16_chunk(seconds=STT_CHUNK_SEC, samplerate=STT_SAMPLE_RATE, channels=1):
    """Record chunk and return raw PCM16 bytes. Added detailed debugging."""
    try:
        frames = sd.rec(
            int(seconds * samplerate),
            samplerate=samplerate,
            channels=channels,
            dtype='int16',
            blocking=True
        )
        pcm = frames.tobytes()

        # LOGGING
        print(f"[STT] Captured audio chunk: {len(pcm)} bytes")

        if len(pcm) == 0:
            print("[STT] WARNING: microphone returned 0 bytes!")

        return pcm

    except Exception as e:
        print(f"[STT] ERROR: Failed to record audio: {e}")
        return b""

# -----------------------
# Node
# -----------------------
class ConversationalTTSNode(Node):
    def __init__(self):
        super().__init__('conversational_tts_node')

        # --- API KEYS ---
        if not ELEVEN_API_KEY:
            self.get_logger().error("ELEVENLABS_API_KEY not set. Exiting.")
            raise RuntimeError("Set ELEVENLABS_API_KEY")

        if not GROQ_API_KEY:
            self.get_logger().error("GROQ_API_KEY not set. Exiting.")
            raise RuntimeError("Set GROQ_API_KEY")

        self.current_bottle_colors = []

        # --- Vision: current bottle seen by camera ---
        self.current_bottle_color = "none"  # "red", "green", "blue", "none", "other"

        # Subscribe to CV detection result
        self.create_subscription(
            String,
            '/bottle_detection_stream',
            self._bottle_detection_callback,
            10
        )

        # --- ROS topics ---
        self.color_pub = self.create_publisher(String, '/requested_bottle_color', 10)
        self.create_subscription(String, '/order_status', self._order_status_cb, 10)

        # --- Groq client ---
        self.groq = Groq(api_key=GROQ_API_KEY)

        self.system_prompt_template = {
            "role": "system",
            "content": (
                "You are a friendly robot bartender.\n\n"
                "Your camera detects bottle colors in real time. The currently visible "
                "bottle colors are provided in THIS LIST - {{current_bottle}}, which is always a Python-style list.\n"
                "There are only three valid bottle colors: red, green, blue.\n"
                "NEVER hallucinate bottles. ONLY trust {{current_bottle}}.\n\n"
                "Cocktail mapping (each cocktail requires ALL listed colors to be visible):\n"
                "If you see red and blue in {{current_bottle}}, tell martini is available\n"
                "If you see green and blue in {{current_bottle}}, tell mojito is available\n"
                "If you see red in {{current_bottle}}, tell bloody mary is available\n"
                "If you see blue in {{current_bottle}}, tell blue lagoon is available\n"
                "If you see red and green in {{current_bottle}}, tell cosmopolitan is available\n\n"
                "Keep your replies within 30 words.\n"
            )
        }
    

        # --- Conversation history (starts fresh) ---
        self.conversation_history = [
            {"role": "assistant", "content": "Hello! I'm your robot bartender. What can I get you today?"}
        ]

        # --- STT control ---
        self.active = True
        self.stt_running = True
        self._stt_loop = asyncio.new_event_loop()
        self._stt_paused = False
        self._stt_lock = threading.Lock()

        # Start STT in background
        threading.Thread(target=self._start_stt_loop, daemon=True).start()

        self.get_logger().info("Conversational TTS node initialized with vision awareness")
    # -----------------------
    # STT loop runner (thread)
    # -----------------------

    def _bottle_detection_callback(self, msg: String):
        """
        Called every time CV node publishes to /bottle_detection_result
        Now supports multiple bottles visible at once!
        """
        text = msg.data.strip().lower()

        if text.startswith("visible:"):
            # Extract all detected colors
            detected_colors = text.replace("visible:", "", 1).strip().split()
            # Filter only valid colors (safety)
            valid_colors = {'red', 'green', 'blue'}
            detected_colors = [c for c in detected_colors if c in valid_colors]
            print("Detected colors from CV:", detected_colors)
            if detected_colors:
                self.current_bottle_colors = detected_colors  # list: e.g. ['red', 'blue']
                colors_str = " and ".join(detected_colors)
                self.get_logger().info(f"Camera sees: {colors_str.upper()} bottle(s)")
            else:
                self.current_bottle_colors = []
        else:
            # No full match → nothing relevant visible
            self.current_bottle_colors = []

        # Always update the singular fallback for old logic (optional)
        # self.current_bottle_color = self.current_bottle_colors[0] if self.current_bottle_colors else "none"
    def _start_stt_loop(self):
        asyncio.set_event_loop(self._stt_loop)
        try:
            self._stt_loop.run_until_complete(self._realtime_stt_connect())
        except Exception as e:
            self.get_logger().error(f"STT loop ended with error: {e}\n{traceback.format_exc()}")

    async def _realtime_stt_connect(self):
        url = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id={ELEVEN_STT_MODEL_ID}&include_timestamps=true"
        headers = {"xi-api-key": ELEVEN_API_KEY}

        self.get_logger().info("Connecting to ElevenLabs Realtime STT...")
        try:
            async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
                self.get_logger().info("Connected to STT websocket. Start speaking.")

                # ---- Supervisor loop for _stt_audio_sender ----
                async def stt_sender_supervisor():
                    while self.active and self.stt_running:
                        try:
                            self.get_logger().info("[STT] Starting _stt_audio_sender")
                            await self._stt_audio_sender(ws)
                        except Exception as e:
                            self.get_logger().error(f"[STT] _stt_audio_sender crashed: {e}\nRestarting in 0.5s...")
                            await asyncio.sleep(0.5)  # short delay before restart

                sender_supervised = asyncio.create_task(stt_sender_supervisor())
                receiver_task = asyncio.create_task(self._stt_receiver(ws))

                done, pending = await asyncio.wait([sender_supervised, receiver_task],
                                                return_when=asyncio.FIRST_EXCEPTION)
                for p in pending:
                    p.cancel()

        except Exception as e:
            self.get_logger().error(f"Could not open STT websocket: {e}\n{traceback.format_exc()}")


    async def _stt_audio_sender(self, ws):
        """
        Capture mic → send to websocket with SILENCE DETECTION.
        Sends audio chunks until silence lasts > SILENCE_DURATION_SEC.
        """

        SILENCE_THRESHOLD = 7000         # adjust to your mic
        SILENCE_DURATION_SEC = 2.0
        CHUNK_SEC = STT_CHUNK_SEC

        last_sound_time = time.time()
        sending_enabled = True  # true while sending speech audio

        while self.active and self.stt_running:

            if self._stt_paused:
                self.get_logger().info("[STT] Sender paused, waiting for TTS to finish")
                await asyncio.sleep(0.1)
                continue

            pcm = record_pcm16_chunk()
            if not pcm:
                await asyncio.sleep(0.05)
                continue

            # --- compute volume ---
            arr = np.frombuffer(pcm, dtype=np.int16)
            volume = float(np.abs(arr).mean())

            # --- sound detected ---
            if volume > SILENCE_THRESHOLD:
                last_sound_time = time.time()

            # --- silence → commit ---
            if sending_enabled and (time.time() - last_sound_time > SILENCE_DURATION_SEC):

                self.get_logger().info("[STT] Silence detected → committing speech")

                # IMPORTANT: wait a moment for last chunks to flush
                await asyncio.sleep(0.25)

                await ws.send(json.dumps({
                    "message_type": "input_audio_buffer.commit"
                }))

                sending_enabled = False
                await asyncio.sleep(0.1)
                continue

            # --- new speech after commit ---
            elif not sending_enabled and volume > SILENCE_THRESHOLD:
                sending_enabled = True
                last_sound_time = time.time()
                self.get_logger().info("[STT] Speech resumed")

            # --- only send chunks while sending_enabled ---
            if sending_enabled:
                b64 = base64.b64encode(pcm).decode("ascii")
                msg = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": b64,
                    "sample_rate": STT_SAMPLE_RATE,
                    "commit": False
                }
                await ws.send(json.dumps(msg))
                self.get_logger().info(
                    f"[STT] Sending chunk → PCM {len(pcm)} bytes | Volume {volume:.1f}"
                )

            await asyncio.sleep(0.001)  # yield control


    async def _stt_receiver(self, ws):
        async for raw in ws:
            self.get_logger().info(f"[STT] Received raw event: {raw[:200]}...")

            try:
                data = json.loads(raw)
            except Exception:
                self.get_logger().warning("[STT] Non-JSON packet received")
                continue

            msg_type = data.get("message_type") or data.get("type") or data.get("event")
            self.get_logger().info(f"[STT] Event type: {msg_type}")

            # ---- handle first partial transcript ----
            if msg_type == "partial_transcript":
                text = data.get("text", "").strip()
                if text and not self._stt_paused:
                    self.get_logger().info(f"[STT] First partial received → sending to LLM: '{text}'")

                    # pause STT sender
                    with self._stt_lock:
                        self._stt_paused = True

                    # handle transcript (LLM + TTS) in background thread
                    threading.Thread(
                        target=self._handle_committed_transcript,
                        args=(text,),
                        daemon=True
                    ).start()

            # COMMITTED transcript (if needed)
            elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps", "transcript", "final"):
                text = data.get("text", "").strip()
                if text:
                    threading.Thread(
                        target=self._handle_committed_transcript,
                        args=(text,),
                        daemon=True
                    ).start()

    # -----------------------
    # Process committed transcript (worker thread)
    # -----------------------
    def _handle_committed_transcript(self, text: str):
        try:
            self.get_logger().info(f"Handling transcript: {text}")

            # 1) append user message
            self.conversation_history.append({"role": "user", "content": text})

            # 2) detect cocktail
            cocktail = self._detect_cocktail(text)
            if cocktail:
                colors = COCKTAILS.get(cocktail, [])
                msg = String()
                msg.data = " ".join(colors)
                self.color_pub.publish(msg)
                self.get_logger().info(f"Published requested colors for '{cocktail}': {msg.data}")

            # 3) call Groq LLM
            reply = self._call_groq()
            if not reply:
                self.get_logger().warning("Groq returned no reply.")
                reply = "Sorry, I didn't understand that."

            # 4) append assistant reply and speak via TTS
            self.conversation_history.append({"role": "assistant", "content": reply})
            self.get_logger().info(f"Assistant reply: {reply}")
            self._tts_say_sync(reply)

        finally:
            # ---- resume STT after TTS ----
            with self._stt_lock:
                self._stt_paused = False
                self.get_logger().info("[STT] Resuming STT capture after TTS")

    def _detect_cocktail(self, text: str):
        t = text.lower()
        for name in COCKTAILS.keys():
            if name in t:
                return name
        return None

    def _call_groq(self) -> str:
        try:
            colors = self.current_bottle_colors  # e.g. ["red", "green"]
            print("Detected bottle colors:\n", colors)

            # ---------------------------------------------------------
            # 1. Create human-readable description ONLY for debugging
            # ---------------------------------------------------------
            if not colors or colors == ["none"]:
                readable = "no bottles"
            elif len(colors) == 1:
                readable = f"a {colors[0]} bottle"
            else:
                # Multi-color readable format
                if len(colors) == 2:
                    readable = f"{colors[0]} and {colors[1]} bottles"
                else:
                    readable = ", ".join(colors[:-1]) + f", and {colors[-1]} bottles"

            # Print for debugging only
            print(f"Readable (debug only): {readable}")

            # ---------------------------------------------------------
            # 2. LLM prompt MUST receive an EXACT Python list string
            # ---------------------------------------------------------
            # Examples:
            #   []                  → "[]"
            #   ["red"]             → "['red']"
            #   ["green","red"]     → "['green', 'red']"
            #
            # repr() creates EXACTLY that format
            current_list_str = repr(colors)
            print("Python list passed to LLM:", current_list_str)

            # ---------------------------------------------------------
            # 3. Build correct system prompt with the list injected
            # ---------------------------------------------------------
            system_msg = self.system_prompt_template.copy()
            system_msg["content"] = system_msg["content"].replace("{{current_bottle}}",
                                                                current_list_str)

            print("*** GROQ SYSTEM PROMPT ***\n")
            print(system_msg["content"])

            messages = [system_msg] + self.conversation_history

            # ---------------------------------------------------------
            # 4. Call the model
            # ---------------------------------------------------------
            resp = self.groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )

            return resp.choices[0].message.content.strip()

        except Exception as e:
            self.get_logger().error(f"GROQ error: {e}")
            return "Sorry, I had trouble understanding that."

    # -----------------------
    # TTS WebSocket helper (async) — collects audio chunks and returns bytes
    # -----------------------
    async def _tts_ws_play_async(self, text: str):
        """Connect to ElevenLabs TTS WS, stream text, collect audio, and play it."""
        ws_url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{TTS_VOICE_ID}/stream-input"
            f"?model_id={TTS_MODEL_ID}&output_format=pcm_22050"
        )

        try:
            async with websockets.connect(ws_url, max_size=None) as ws:
                self.get_logger().info(f"[TTS] Connected to WebSocket")
                
                # STEP 1: Initialize connection with voice settings (REQUIRED FIRST MESSAGE)
                init_msg = {
                    "text": " ",  # Space character for initialization
                    "voice_settings": {
                        "stability": 0.5, 
                        "similarity_boost": 0.7,
                        "use_speaker_boost": False
                    },
                    "generation_config": {
                        "chunk_length_schedule": [120, 160, 250, 290]
                    },
                    "xi_api_key": ELEVEN_API_KEY  # Include API key in message
                }
                await ws.send(json.dumps(init_msg))
                self.get_logger().info("[TTS] Sent initialization message")

                # STEP 2: Send the actual text
                text_msg = {
                    "text": text,
                    "xi_api_key": ELEVEN_API_KEY
                }
                await ws.send(json.dumps(text_msg))
                self.get_logger().info(f"[TTS] Sent text: '{text}'")

                # STEP 3: Send empty string to signal end and close connection
                end_msg = {
                    "text": "",
                    "xi_api_key": ELEVEN_API_KEY
                }
                await ws.send(json.dumps(end_msg))
                self.get_logger().info("[TTS] Sent end signal")

                # Collect audio chunks
                all_audio = bytearray()
                
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                    except Exception:
                        self.get_logger().warning("[TTS] Non-JSON message received")
                        continue

                    # Debug what we're receiving
                    self.get_logger().debug(f"[TTS] WS message keys: {list(data.keys())}")

                    # Check for errors
                    if data.get("error"):
                        self.get_logger().error(f"[TTS] Server error: {data['error']}")
                        break

                    # Collect audio chunk
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        try:
                            chunk = base64.b64decode(audio_b64)
                            all_audio.extend(chunk)
                            self.get_logger().info(f"[TTS] Received audio chunk: {len(chunk)} bytes (total {len(all_audio)} bytes)")
                        except Exception as e:
                            self.get_logger().error(f"[TTS] Failed to decode audio chunk: {e}")
                            continue

                    # Check for completion
                    if data.get("isFinal"):
                        self.get_logger().info("[TTS] Received final signal")
                        break

                # Play audio if we received any
                if not all_audio:
                    self.get_logger().warning("[TTS] No audio received from WebSocket")
                    return

                try:
                    # Ensure even number of bytes for int16
                    if len(all_audio) % 2 != 0:
                        all_audio.extend(b'\x00')
                    
                    audio_np = np.frombuffer(bytes(all_audio), dtype=np.int16)
                    self.get_logger().info(f"[TTS] Playing audio: {len(audio_np)} samples")
                    sd.play(audio_np, samplerate=TTS_OUTPUT_SR)
                    sd.wait()
                    self.get_logger().info("[TTS] Playback complete")
                except Exception as e:
                    self.get_logger().error(f"[TTS] Playback error: {e}")

        except Exception as e:
            self.get_logger().error(f"[TTS] WebSocket connection error: {e}")


    def _tts_say_sync(self, text: str):
        """Synchronous wrapper to run the async TTS websocket and play audio."""
        try:
            # run in a temporary asyncio loop for TTS
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._tts_ws_play_async(text))
            loop.close()
        except Exception as e:
            self.get_logger().error(f"TTS sync error: {e}")
    
    # -----------------------
    # order_status subscriber (reset history on "done")
    # -----------------------
    def _order_status_cb(self, msg: String):
        if msg.data.strip().lower() == "done":
            self.get_logger().info("Order done received — resetting conversation history.")
            self.conversation_history = [
                {"role": "system", "content": "You are a friendly bartender. Keep replies short and helpful."},
                {"role": "assistant", "content": "Hello! What can I get you to drink?"}
            ]
            # speak confirmation
            try:
                self._tts_say_sync("Your drink is ready!")
            except Exception:
                pass

    # -----------------------
    # shutdown
    # -----------------------
    def destroy_node(self):
        self.get_logger().info("Shutting down conversational node...")
        self.active = False
        self.stt_running = False
        # try stop stt loop
        try:
            if self._stt_loop.is_running():
                self._stt_loop.call_soon_threadsafe(self._stt_loop.stop)
        except Exception:
            pass
        super().destroy_node()


# -----------------------
# MAIN
# -----------------------
def main(args=None):
    rclpy.init(args=args)
    node = ConversationalTTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
