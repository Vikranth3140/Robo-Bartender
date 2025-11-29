#!/usr/bin/env python3

from xarm.wrapper import XArmAPI
import time

# Replace with your xArm IP
ARM_IP = '192.168.1.152'

print(f"Trying to connect to xArm at {ARM_IP} ...")
arm = XArmAPI(ARM_IP, baud_checkset=False)

# Wait a few seconds for connection
time.sleep(2)

print("Connected?", arm.connected)

if arm.connected:
    print("xArm firmware version:", arm.get_version())
else:
    print("Failed to connect. Check IP, network, and arm power.")
