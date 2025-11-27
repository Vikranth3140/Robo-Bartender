# xArm Manipulation Module

A ROS2 node for controlling an xArm robotic manipulator to perform automated bottle pouring operations based on color detection and voice commands.

## Overview

This module implements a ROS2 node (`XArmNode`) that coordinates with computer vision and conversational AI systems to:
- Detect bottle colors using CV input
- Receive bottle color requests from conversational AI
- Execute precise pick-and-pour operations with an xArm robot
- Return bottles to their original positions after pouring

## Features

- **Multi-color bottle handling**: Supports green, red, and blue bottles
- **Precision movement**: Pre-configured workspace positions for bottles and cup
- **Safety mechanisms**: Approach heights and gradual movements
- **Real-time coordination**: Integrates with CV detection and voice command systems
- **Status reporting**: Publishes completion status for downstream processes

## Hardware Requirements

- xArm robotic manipulator
- Network connectivity to xArm controller
- Bottles positioned at predefined workspace coordinates
- Target cup/container at designated position

## Dependencies

- ROS2 (Robot Operating System 2)
- `rclpy` - ROS2 Python client library
- `xarm` - Official xArm Python SDK (see [xArm-Python-SDK](https://github.com/xArm-Developer/xArm-Python-SDK))
- `std_msgs` - Standard ROS2 message types

**Note**: For comprehensive ROS2 integration with xArm, also refer to the official [xarm_ros2](https://github.com/xArm-Developer/xarm_ros2) package.

## Installation

1Install ROS2and xArm dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your xArm is properly calibrated and positioned

## Configuration

### Network Setup
- Default xArm IP: `192.168.1.152`
- Modify the IP parameter in the node if your xArm uses a different address

### Workspace Positions
The module uses predefined coordinates for optimal operation:

**Bottle Positions:**
- Green bottle: (418.9, 30.8, 153mm) with rotation (90.7°, -68.2°, 88.2°)
- Red bottle: (380, -140.2, 153mm) with rotation (90.7°, -68.2°, 88.2°)
- Blue bottle: (225.5, -140.2, 153mm) with rotation (90.7°, -68.2°, 88.2°)

**Cup Position:**
- Position: (240.4, 19.1, 153mm) with rotation (89°, -40.2°, 91.4°)

**Movement Heights:**
- Approach height: 220mm
- Pick/place height: 150mm
- Pour height: 180mm

## Usage

### Running the Node

```bash
ros2 run xarm_manipulation xarm_node
```

### ROS2 Topics

**Subscribers:**
- `/requested_bottle_color` (String): Receives color requests from conversational AI
  - Format: Space-separated color names (e.g., "green red blue")
- `/bottle_detection_result` (String): Receives CV detection results
  - Format: "detected: color1 color2 color3"

**Publishers:**
- `/order_status` (String): Publishes completion status ("done" when finished)

## Operation Workflow

1. **Initialization**: Node connects to xArm and initializes arm parameters
2. **Request Reception**: Waits for both voice command and CV detection inputs
3. **Validation**: Ensures requested colors match detected bottles
4. **Execution**: Performs pick-pour-return sequence for each requested bottle
5. **Completion**: Returns arm to home position and publishes status

## Movement Sequence

For each bottle, the robot performs:

1. **Pick Phase:**
   - Move to approach height above bottle
   - Descend to pick height
   - Tilt gripper for optimal grip
   - Grasp bottle
   - Lift to approach height

2. **Pour Phase:**
   - Move to approach height above cup
   - Descend to pour height
   - Tilt bottle to pour contents
   - Return to upright position
   - Lift to approach height

3. **Return Phase:**
   - Move to approach height above original bottle position
   - Descend to place height
   - Place bottle in original position
   - Return gripper to normal orientation
   - Lift to approach height

## Safety Features

- **Error Handling**: Comprehensive exception catching and logging
- **Arm Initialization**: Clears warnings and errors before operation
- **Threading**: Non-blocking execution prevents system freezes
- **State Management**: Prevents concurrent operations

## References

- [xArm ROS2 Package](https://github.com/xArm-Developer/xarm_ros2) - Official ROS2 integration for xArm robots
- [xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK) - Official Python SDK for xArm control

## Integration

This module is designed to work with:
- **CV Bottle Detector**: Provides bottle color detection
- **Conversational AI**: Processes voice commands and converts to color requests
- **Overall System**: Part of a larger robotic bartender/service system
