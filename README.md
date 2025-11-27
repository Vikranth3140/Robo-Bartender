# 🤖🍹 Conversational Robo Bartender

An intelligent robotic bartender system that combines advanced computer vision, natural language processing, and precise robotic manipulation to create cocktails through voice commands. The system integrates real-time speech recognition, AI-powered conversation, color-based bottle detection, and robotic arm control to deliver a seamless bartending experience.

## System Overview

The Conversational Robo Bartender is a sophisticated ROS2-based system that enables users to order cocktails through natural conversation. The system uses computer vision to detect colored bottles, processes voice commands through AI, and executes precise robotic movements to create drinks.

### Key Features

- **Real-time Voice Interaction**: Natural conversation using ElevenLabs STT/TTS and Groq LLM
- **Intelligent Bottle Detection**: YOLO-based computer vision for multi-color bottle recognition
- **Precision Robotics**: xArm robotic arm control for accurate bottle manipulation and pouring
- **Smart Recipe Management**: Cocktail recipes mapped to bottle color combinations
- **Multi-Modal Integration**: Seamless coordination between speech, vision, and motion systems

## System Architecture

The system consists of three main integrated nodes that communicate via ROS2 topics:

```
┌──────────────────────────────────────────────────────────────┐
│                    ROS2 Communication Layer                  │
├──────────────────┬─────────────────┬─────────────────────────┤
│ Conversational   │   Computer      │      Robotic Arm        │
│    TTS Node      │  Vision Node    │        Node             │
│                  │                 │                         │
│ • STT/TTS        │ • Bottle        │ • xArm Control          │
│ • AI Conversation│  Detection      │ • Motion Planning       │
│ • Recipe Logic   │ • Color ID      │ • Pour Operations       │
│ • Order Status   │ • Multi-bottle  │ • Safety Management     │
└──────────────────┴─────────────────┴─────────────────────────┘
```

### Communication Flow

1. **User speaks** → STT converts to text
2. **AI processes** request and identifies cocktail
3. **Vision system** detects available bottles
4. **Robot arm** executes picking, pouring, and serving sequence
5. **TTS confirms** completion to user

## Core Components

📚 **Detailed Documentation**: For comprehensive setup and usage instructions for individual components, see:
- [Conversational AI Module](Conversational-AI/README.md)
- [Bottle Detection System](Bottle-Detection/README.md) 
- [xArm Manipulation Control](xArm-Manipulation/README.md)

### 1. Conversational TTS Node (`conversational_tts_node.py`)

**Primary Functions:**
- Real-time speech-to-text using ElevenLabs WebSocket API
- AI-powered conversation with Groq LLM (Moonshot Kimi model)
- Text-to-speech synthesis for natural responses
- Cocktail recipe management and order processing

**Key Technologies:**
- ElevenLabs Realtime STT with silence detection
- Groq API for conversational AI
- WebSocket-based audio streaming
- ROS2 message coordination

**Cocktail Recipes:**
```python
COCKTAILS = {
    'martini': ['red', 'blue'],
    'mojito': ['green', 'blue'], 
    'bloody mary': ['red'],
    'blue lagoon': ['blue'],
    'cosmopolitan': ['red', 'green']
}
```

**Current LLM System Prompt:**
The AI bartender uses the following system prompt to ensure accurate, context-aware responses:

```
You are a friendly robot bartender.

Your camera detects bottle colors in real time. The currently visible 
bottle colors are provided in THIS LIST - {{current_bottle}}, which is always a Python-style list.
There are only three valid bottle colors: red, green, blue.
NEVER hallucinate bottles. ONLY trust {{current_bottle}}.

Cocktail mapping (each cocktail requires ALL listed colors to be visible):
If you see red and blue in {{current_bottle}}, tell martini is available
If you see green and blue in {{current_bottle}}, tell mojito is available
If you see red in {{current_bottle}}, tell bloody mary is available
If you see blue in {{current_bottle}}, tell blue lagoon is available
If you see red and green in {{current_bottle}}, tell cosmopolitan is available

Keep your replies within 30 words.
```

This prompt ensures the AI only recommends cocktails based on actually detected bottles and maintains concise, natural interactions.

### 2. Computer Vision Node (`cv_bottle_detector_node.py`)

**Primary Functions:**
- Real-time bottle detection using YOLO11
- HSV-based color classification (red, green, blue)
- Multi-bottle tracking and validation
- Integration with order requirements

**Detection Capabilities:**
- Multiple bottle types (classes: 39, 41, 40, 45, 75)
- Robust color classification with HSV thresholding
- Real-time video processing with OpenCV
- Confidence-based filtering

**Color Ranges (HSV):**
- **Red**: (0-10, 170-180) hue with 50+ saturation
- **Green**: (30-90) hue with 40+ saturation  
- **Blue**: (80-140) hue with 40+ saturation

### 3. Robotic Arm Node (`xarm_node.py`)

**Primary Functions:**
- Precise xArm robotic arm control
- Multi-step bottle manipulation sequences
- Safety management and error handling
- Coordinated pick-pour-return operations

**Workspace Configuration:**
```python
BOTTLES = {
    "green": {"pos": (418.9, 30.8, 153),  "rpy": (90.7, -68.2, 88.2)},
    "red":   {"pos": (380, -140.2, 153),  "rpy": (90.7, -68.2, 88.2)}, 
    "blue":  {"pos": (225.5, -140.2, 153), "rpy": (90.7, -68.2, 88.2)}
}
CUP_POS = (240.4, 19.1, 153)
```

**Movement Sequence:**
1. **Approach** bottle at safe height (220mm)
2. **Pick** bottle with gripper engagement
3. **Transport** to cup position with tilt control
4. **Pour** with precise angle adjustment
5. **Return** bottle to original position

## Installation & Setup

### Prerequisites

- **Operating System**: Windows/Linux with ROS2 Humble
- **Python**: 3.8+
- **Hardware**: 
  - USB Camera for bottle detection
  - Microphone for voice input
  - Speakers for audio output
  - xArm robotic arm (IP: 192.168.1.152)

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Vikranth3140/Robo-Bartender.git
cd Robo-Bartender

# Install dependencies
pip install -r requirements.txt

# Source ROS2 environment
source /opt/ros/humble/setup.bash
```

### 2. API Configuration

Create environment variables or update the configuration in `conversational_tts_node.py`:

```python
ELEVEN_API_KEY = "your_elevenlabs_api_key"
GROQ_API_KEY = "your_groq_api_key" 
TTS_VOICE_ID = "your_elevenlabs_voice_id"
```

### 3. Hardware Calibration

**Camera Setup:**
- Connect USB camera as device 0
- Ensure proper lighting for color detection
- Test bottle visibility in camera frame

**Robot Arm Setup:**
- Connect xArm to network (IP: 192.168.1.152)
- Calibrate workspace positions
- Verify safety boundaries

## Running the System

### Quick Start

```bash
# Setup ROS2 workspace and package
cd ~/ros2_ws
ros2 pkg create --build-type ament_python robo-bartender
colcon build --symlink-install
source install/setup.bash

# Terminal 1: Launch Conversational Node
ros2 run robo-bartender conversational_tts_node

# Terminal 2: Launch Computer Vision Node
ros2 run robo-bartender cv_bottle_detector_node

# Terminal 3: Launch Robotic Arm Node
ros2 run robo-bartender xarm_node
```

### System Verification

1. **Vision System**: Verify bottle detection in OpenCV window
2. **Audio System**: Test microphone input and speaker output
3. **Robot Arm**: Confirm connection and home position
4. **Integration**: Place bottles and attempt voice order

## 📡 ROS2 Communication Topics

| Topic | Publisher | Subscriber | Message Type | Purpose |
|-------|-----------|------------|--------------|---------|
| `/bottle_detection_result` | CV Node | TTS Node | String | Detected bottle colors |
| `/bottle_detection_stream` | CV Node | TTS Node | String | Real-time detection stream |
| `/requested_bottle_color` | TTS Node | CV/Arm Nodes | String | Required colors for cocktail |
| `/order_status` | Arm Node | TTS Node | String | Order completion status |

### Message Formats

**Bottle Detection:**
```
"visible: red blue"           # Multiple bottles detected
"detected: red blue"          # Order-specific detection
```

**Order Status:**
```
"done"                        # Order completed successfully
```

## User Interaction

### Voice Commands Examples

- *"I'd like a martini please"* → Requests red and blue bottles
- *"Can I get a mojito?"* → Requests green and blue bottles  
- *"What drinks can you make?"* → Lists available cocktails based on visible bottles
- *"Make me a blue lagoon"* → Requests blue bottle only

### System Responses

The AI bartender provides contextual responses based on:
- Currently visible bottles detected by camera
- Available cocktail recipes
- Order processing status
- Error conditions or missing ingredients

## Configuration

### Audio Settings

```python
STT_SAMPLE_RATE = 16000        # Speech recognition sample rate
TTS_OUTPUT_SR = 22050          # Text-to-speech output rate
STT_CHUNK_SEC = 0.5            # Audio processing chunk size
SILENCE_DURATION_SEC = 2.0     # Silence detection threshold
```

### Vision Parameters

```python
confidence_threshold = 0.1     # YOLO detection confidence
color_threshold = 0.02         # HSV color matching threshold
```

### Robot Parameters

```python
Z_APPROACH = 220               # Safe approach height (mm)
Z_PICK = 150                   # Bottle picking height (mm)  
Z_POUR = 180                   # Pouring height (mm)
speed = 50                     # Movement speed (mm/s)
```

## Troubleshooting

### Common Issues

**Audio Problems:**
- Check microphone permissions and device selection
- Verify ElevenLabs API key and quota
- Test audio devices with `sounddevice`

**Vision Issues:**
- Ensure adequate lighting for color detection
- Calibrate HSV color ranges for your bottles
- Check camera device index (usually 0)

**Robot Communication:**
- Verify xArm IP address and network connectivity
- Check arm initialization and safety status
- Calibrate workspace coordinates

**ROS2 Integration:**
- Confirm all nodes are running with `ros2 node list`
- Monitor topics with `ros2 topic echo <topic_name>`
- Check for error messages in node logs

## License

This project is developed for research and educational purposes under [MIT License](LICENSE).
