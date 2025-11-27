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

from groq import Groq

# -----------------------
# CONFIG (set via env or edit here)
# -----------------------
ELEVEN_API_KEY = "your_api_key_here"            # required
GROQ_API_KEY = "your_groq_api_key_here"         # required

# STT settings
ELEVEN_STT_MODEL_ID = "scribe_v2_realtime"
STT_SAMPLE_RATE = 16000
STT_CHUNK_SEC = 0.5

# TTS settings
TTS_VOICE_ID = "your_voice_id_here"
TTS_MODEL_ID = "eleven_flash_v2_5"
TTS_OUTPUT_SR = 22050

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
    """Record chunk and return raw PCM16 bytes."""
    try:
        frames = sd.rec(
            int(seconds * samplerate),
            samplerate=samplerate,
            channels=channels,
            dtype='int16',
            blocking=True
        )
        pcm = frames.tobytes()
        print(f"[STT] Captured audio chunk: {len(pcm)} bytes")
        return pcm
    except Exception as e:
        print(f"[STT] ERROR: Failed to record audio: {e}")
        return b""


# ===========================================================
#  NON-ROS Conversational Class
# ===========================================================

class ConversationalTTS:
    def __init__(self):
        # --- API KEYS ---
        if not ELEVEN_API_KEY:
            raise RuntimeError("ELEVENLABS_API_KEY not set.")
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set.")

        self.current_bottle_colors = []   # now updated manually, not from ROS

        self.groq = Groq(api_key=GROQ_API_KEY)

        self.system_prompt_template = {
            "role": "system",
            "content": (
                "You are a friendly robot bartender.\n\n"
                "The currently visible bottle colors are provided in THIS LIST - {{current_bottle}}.\n"
                "Valid bottle colors: red, green, blue.\n"
                "NEVER hallucinate bottles — ONLY trust {{current_bottle}}.\n\n"
                "Cocktail mapping:\n"
                "red + blue → martini available\n"
                "green + blue → mojito available\n"
                "red → bloody mary available\n"
                "blue → blue lagoon available\n"
                "red + green → cosmopolitan available\n\n"
                "Keep replies ≤ 30 words.\n"
            )
        }

        self.conversation_history = [
            {"role": "assistant", "content": "Hello! I'm your robot bartender. What can I get you today?"}
        ]

        # STT control
        self.active = True
        self.stt_running = True
        self._stt_loop = asyncio.new_event_loop()
        self._stt_paused = False
        self._stt_lock = threading.Lock()

        # Start STT thread
        threading.Thread(target=self._start_stt_loop, daemon=True).start()


    # ===========================================================
    #  PUBLIC FUNCTION to update bottle colors (replaces ROS callback)
    # ===========================================================
    def update_visible_bottles(self, color_list):
        """
        Example call:
        >>> bot.update_visible_bottles(["red", "blue"])
        """
        valid = {"red", "green", "blue"}
        self.current_bottle_colors = [c for c in color_list if c in valid]
        print("Updated bottle colors:", self.current_bottle_colors)

    # ===========================================================
    #  STT loop
    # ===========================================================
    def _start_stt_loop(self):
        asyncio.set_event_loop(self._stt_loop)
        try:
            self._stt_loop.run_until_complete(self._realtime_stt_connect())
        except Exception as e:
            print(f"STT loop ended with error: {e}\n{traceback.format_exc()}")

    async def _realtime_stt_connect(self):
        url = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id={ELEVEN_STT_MODEL_ID}&include_timestamps=true"
        headers = {"xi-api-key": ELEVEN_API_KEY}

        print("Connecting to ElevenLabs Realtime STT...")
        try:
            async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
                print("Connected to STT websocket.")

                async def stt_sender_supervisor():
                    while self.active and self.stt_running:
                        try:
                            await self._stt_audio_sender(ws)
                        except Exception as e:
                            print(f"[STT] sender crashed: {e}")
                            await asyncio.sleep(0.5)

                sender_supervised = asyncio.create_task(stt_sender_supervisor())
                receiver_task = asyncio.create_task(self._stt_receiver(ws))

                await asyncio.wait([sender_supervised, receiver_task], return_when=asyncio.FIRST_EXCEPTION)

        except Exception as e:
            print(f"Could not open STT websocket: {e}")

    async def _stt_audio_sender(self, ws):
        SILENCE_THRESHOLD = 7000
        SILENCE_DURATION_SEC = 2.0

        last_sound_time = time.time()
        sending_enabled = True

        while self.active and self.stt_running:

            if self._stt_paused:
                await asyncio.sleep(0.1)
                continue

            pcm = record_pcm16_chunk()
            if not pcm:
                await asyncio.sleep(0.05)
                continue

            arr = np.frombuffer(pcm, dtype=np.int16)
            volume = float(np.abs(arr).mean())

            if volume > SILENCE_THRESHOLD:
                last_sound_time = time.time()

            if sending_enabled and (time.time() - last_sound_time > SILENCE_DURATION_SEC):
                await asyncio.sleep(0.25)
                await ws.send(json.dumps({"message_type": "input_audio_buffer.commit"}))
                sending_enabled = False
                continue

            elif not sending_enabled and volume > SILENCE_THRESHOLD:
                sending_enabled = True
                last_sound_time = time.time()

            if sending_enabled:
                b64 = base64.b64encode(pcm).decode("ascii")
                msg = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": b64,
                    "sample_rate": STT_SAMPLE_RATE,
                    "commit": False
                }
                await ws.send(json.dumps(msg))

            await asyncio.sleep(0.001)

    async def _stt_receiver(self, ws):
        async for raw in ws:
            try:
                data = json.loads(raw)
            except:
                continue

            msg_type = data.get("message_type")

            if msg_type == "partial_transcript":
                text = data.get("text", "").strip()
                if text and not self._stt_paused:
                    with self._stt_lock:
                        self._stt_paused = True

                    threading.Thread(
                        target=self._handle_committed_transcript,
                        args=(text,),
                        daemon=True
                    ).start()

    # ===========================================================
    #  Process transcript → LLM → TTS
    # ===========================================================
    def _handle_committed_transcript(self, text: str):
        try:
            print("User:", text)
            self.conversation_history.append({"role": "user", "content": text})

            cocktail = self._detect_cocktail(text)
            if cocktail:
                print(f"User ordered: {cocktail}, needs colors:", COCKTAILS[cocktail])

            reply = self._call_groq()
            if not reply:
                reply = "Sorry, I didn't understand that."

            print("Assistant:", reply)
            self.conversation_history.append({"role": "assistant", "content": reply})

            self._tts_say_sync(reply)

        finally:
            with self._stt_lock:
                self._stt_paused = False

    # Cocktail detection
    def _detect_cocktail(self, text: str):
        t = text.lower()
        for name in COCKTAILS:
            if name in t:
                return name
        return None

    # ===========================================================
    #  Groq LLM
    # ===========================================================
    def _call_groq(self):
        try:
            colors = self.current_bottle_colors
            current_list_str = repr(colors)

            system_msg = self.system_prompt_template.copy()
            system_msg["content"] = system_msg["content"].replace(
                "{{current_bottle}}", current_list_str
            )

            messages = [system_msg] + self.conversation_history

            resp = self.groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )

            return resp.choices[0].message.content.strip()

        except Exception as e:
            print("Groq error:", e)
            return None

    # ===========================================================
    #  TTS
    # ===========================================================
    async def _tts_ws_play_async(self, text: str):
        ws_url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{TTS_VOICE_ID}/stream-input"
            f"?model_id={TTS_MODEL_ID}&output_format=pcm_22050"
        )

        try:
            async with websockets.connect(ws_url, max_size=None) as ws:

                init_msg = {
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.7,
                        "use_speaker_boost": False
                    },
                    "generation_config": {"chunk_length_schedule": [120, 160, 250, 290]},
                    "xi_api_key": ELEVEN_API_KEY
                }
                await ws.send(json.dumps(init_msg))

                await ws.send(json.dumps({"text": text, "xi_api_key": ELEVEN_API_KEY}))
                await ws.send(json.dumps({"text": "", "xi_api_key": ELEVEN_API_KEY}))

                all_audio = bytearray()

                async for raw in ws:
                    data = json.loads(raw)
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        chunk = base64.b64decode(audio_b64)
                        all_audio.extend(chunk)

                    if data.get("isFinal"):
                        break

                if not all_audio:
                    print("[TTS] No audio received.")
                    return

                if len(all_audio) % 2 != 0:
                    all_audio.extend(b"\x00")

                audio_np = np.frombuffer(bytes(all_audio), dtype=np.int16)
                sd.play(audio_np, samplerate=TTS_OUTPUT_SR)
                sd.wait()

        except Exception as e:
            print("[TTS] WebSocket error:", e)

    def _tts_say_sync(self, text: str):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._tts_ws_play_async(text))
        loop.close()


# ===========================================================
#  MAIN
# ===========================================================

def main():
    bot = ConversationalTTS()
    print("System running…\n")
    print("Call bot.update_visible_bottles(['red','blue']) from anywhere to update camera colors.\n")
    print("Speak to the microphone to talk to the bartender.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.active = False
        bot.stt_running = False
        print("Shutting down...")

if __name__ == "__main__":
    main()
