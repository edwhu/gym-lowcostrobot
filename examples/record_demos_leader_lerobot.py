import argparse
import os

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_lowcostrobot # Import the low-cost robot environments

from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from low_cost_robot.robot import Robot
from low_cost_robot.dynamixel import Dynamixel


def do_sim(args):

    leader_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name=args.device).instantiate()
    leader = Robot(leader_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
    leader.name = 'leader'
    leader.set_trigger_torque()
    leader_dynamixel.disconnect()
    del leader

    env = gym.make(args.env_name, render_mode="human")
    # env = gym.make(args.env_name, render_mode="rgb_array")

    # offsets = [0, 0, np.pi/2, np.pi, -np.pi/2, 0]  # leader1
    offsets = [0, 0, np.pi/2, np.pi, 0, 0]  # leader2 (white joint)
    counts_to_radians = np.pi * 2. / 4096.
    start_pos = [2072, 2020, 1063, 3966, 3053, 1938] # get the start pos from .cache/calibration directory in your local lerobot
    axis_direction = [-1, -1, -1, 1, -1, -1]
    joint_commands = [0,0,0,0,0,0]
    leader_arm = DynamixelMotorsBus(
        port=args.device,
        motors={
            # name: (index, model)
            "shoulder_pan": (1, "xl330-m077"),
            "shoulder_lift": (2, "xl330-m077"),
            "elbow_flex": (3, "xl330-m077"),
            "wrist_flex": (4, "xl330-m077"),
            "wrist_roll": (5, "xl330-m077"),
            "gripper": (6, "xl330-m077"),
        },
    )

    if not leader_arm.is_connected:
        leader_arm.connect()
    
    env.reset()
    rewards = []
    timesteps = []
    demos_data = {
        'obs/joints': [],
        'obs/wrist_cam': [],
        'obs/gripper_stuck': [],
        'action': [],
        'reward': [],
        'terminated': [],
        'truncated': [],
    }

    # Create a folder to save demo data
    demo_folder = args.demo_folder
    if not os.path.exists(demo_folder):
        os.makedirs(demo_folder)
    
    demo_count = 0
  
    # Main tele-operation loop
    while env.unwrapped.viewer.is_running():
        # Read current joint positions
        positions = leader_arm.read("Present_Position")
        # Make sure joint commands = number of positions
        assert len(joint_commands) == len(positions)
        # For each of the joints
        for i in range(len(joint_commands)):
            # move in the axis direction * (position change in radians) + offset
            joint_commands[i] = axis_direction[i] * \
                (positions[i] - start_pos[i]) * counts_to_radians + offsets[i]
        
        # send joint commands to simulated environment
        ret = env.step(joint_commands)

        # record rewards, timesteps
        rewards.append(ret[1])
        timesteps.append(env.unwrapped.data.time)
            
    plt.plot(timesteps, rewards)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose between 5dof and 6dof lowcost robot simulation.")
    parser.add_argument('--device', type=str, default='/dev/ttyACM0', help='Port name (e.g., COM1, /dev/ttyUSB0, /dev/tty.usbserial-*)')
    # parser.add_argument('--env-name', type=str, default='PushCubeLoop-v0', help='Specify the gym-lowcost robot env to test.')
    parser.add_argument('--env-name', type=str, default='LiftCubeCamera-v0', help='Specify the gym-lowcost robot env to test.')
    parser.add_argument('--demo_folder', type=str, default='demos', help='Specify the local folder to save demos to')
    args = parser.parse_args()

    do_sim(args)
