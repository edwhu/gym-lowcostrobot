import argparse
from collections import defaultdict
import os
import time
import cv2

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_lowcostrobot # Import the low-cost robot environments
from tqdm import trange, tqdm

from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from low_cost_robot.robot import Robot
from low_cost_robot.dynamixel import Dynamixel


def do_sim(args):
    """Tele-op method, use for debugging"""
    leader_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name=args.device).instantiate()
    leader = Robot(leader_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
    leader.name = 'leader'
    leader.set_trigger_torque()
    leader_dynamixel.disconnect()
    del leader

    env = gym.make(args.env_name, render_mode="human", observation_mode="both", action_mode="joint")

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
        joint_commands[0] = 0.0 # disable movement along joint1
        
        # print("POSITIONS", positions) # [0, 4096]
        # print("QPOS", env.unwrapped.data.qpos) # [radians]
        # print("JOINT COMMANDS", joint_commands) # [radians]

        # send joint commands to simulated environment
        obs, rew, terminated, truncated, info = env.step(joint_commands)
        # print("REWARD", rew)
        print("CUBE POS", obs['cube_pos'])

        # record rewards, timesteps
        rewards.append(rew)
        timesteps.append(env.unwrapped.data.time)
            
    plt.plot(timesteps, rewards)
    plt.show()


def collect_demos(args):

    leader_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name=args.device).instantiate()
    leader = Robot(leader_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
    leader.name = 'leader'
    leader.set_trigger_torque()
    leader_dynamixel.disconnect()
    del leader

    env = gym.make(args.env_name, render_mode="human", observation_mode="both")

    # offsets = [0, 0, np.pi/2, np.pi, -np.pi/2, 0]  # leader1
    offsets = [0, 0, np.pi/2, np.pi, 0, 0]  # leader2 (white joint)
    counts_to_radians = np.pi * 2. / 4096.
    start_pos = [2072, 2020, 1063, 3966, 3053, 1938] # get the start pos from .cache/calibration directory in your local lerobot
    axis_direction = [-1, -1, -1, 1, -1, -1]
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

    # Create a folder to save demo data
    demo_folder = args.demo_folder
    if not os.path.exists(demo_folder):
        os.makedirs(demo_folder)
    
    demo_length = 400 # in steps
    reset_seconds = 0.5 # in seconds
    num_demos = 20
    demos_collected = 0
  
    while demos_collected < num_demos:
        ep_dict = defaultdict(list)
        obs, info = env.reset()
        for k, v in obs.items():
            ep_dict['obs/' + k].append(v)
        
        print("Clean up the environment.")
        # Here, we would give the user some time to clean up the environment and the robot. 
        start_time = time.time()
        with tqdm(total=reset_seconds, desc="Waiting for reset...") as pbar:
            last_time = time.time()
            while time.time() - start_time < reset_seconds:
                positions = leader_arm.read("Present_Position")
                joint_commands = [0] * len(positions)
                # Compute action (joint commands) based on the position changes
                for i in range(len(joint_commands)):
                    joint_commands[i] = axis_direction[i] * \
                        (positions[i] - start_pos[i]) * counts_to_radians + offsets[i]
                joint_commands[0] = 0.0 # disable movement along joint1

                ret = env.step(joint_commands)
                current_time = time.time()
                pbar.update(current_time - last_time)
                last_time = current_time

        input(f"Demo {demos_collected + 1}/{num_demos}, Press Enter to start the collection.")

        for timestep in trange(demo_length, desc="Collecting demo"):
            start_time = time.time()
            positions = leader_arm.read("Present_Position")
            joint_commands = [0] * len(positions)

            # Compute action (joint commands) based on the position changes
            for i in range(len(joint_commands)):
                joint_commands[i] = axis_direction[i] * \
                    (positions[i] - start_pos[i]) * counts_to_radians + offsets[i]
            joint_commands[0] = 0.0 # disable movement along joint1

            ep_dict['action'].append(np.asarray(joint_commands, dtype=np.float32))
            
            obs, rew, terminated, truncated, info = env.step(joint_commands)
            print("REWARD", rew)
            print("CUBE POS", obs['cube_pos'])
            
            # doesn't work on mac
            # img = obs['image_top']
            # cv2.imshow('top', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # img = obs['image_wrist']
            # cv2.imshow('wrist', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)

            for k, v in obs.items():
                ep_dict['obs/' + k].append(v)
            ep_dict['reward'].append(rew)
            ep_dict['terminated'].append(terminated)
            ep_dict['truncated'].append(truncated)
            for k, v in info.items():
                ep_dict['info/' + k].append(v)
        
        save_demo = input("Save the demo? enter y/n")
        if save_demo.lower() == 'y':
            demos_collected += 1
            demo_path = os.path.join(demo_folder, f'demo_{demos_collected}.npz')
            np.savez_compressed(demo_path, **ep_dict)
        else:
            print("Demo not saved.")

        for key in ep_dict:
            print(key, ep_dict[key][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose between 5dof and 6dof lowcost robot simulation.")
    parser.add_argument('--device', type=str, default='/dev/ttyACM0', help='Port name (e.g., COM1, /dev/ttyUSB0, /dev/tty.usbserial-*)')
    parser.add_argument('--env-name', type=str, default='LiftCubeCamera-v0', help='Specify the gym-lowcost robot env to test.')
    parser.add_argument('--demo_folder', type=str, default='demos', help='Specify the local folder to save demos to')
    args = parser.parse_args()

    # do_sim(args)
    collect_demos(args)
