"""
Data Collection Script for Koch Robot

This script provides functionality to collect robot interaction data using either human-operated
gamepad control or policy execution. The collected data is stored in a LeRobotDataset
format.

Features:
- Human-operated data collection using gamepad control
- Random policy data collection (for baseline)
- Learned policy data collection (placeholder for future implementation)
- Support for both simulation and real robot environments
- Automatic episode collection and storage
- Gamepad debugging mode for input testing
- Camera-based object targeting for scripted policy

Usage:
    python collect_episodes.py [--mode {gamepad,random,learned,debug_gamepad}] [--sim]

Arguments:
    --mode: Collection mode (default: gamepad)
        - gamepad: Human-operated collection using gamepad
        - random: Random policy collection (baseline)
        - learned: Learned policy collection (placeholder)
        - debug_gamepad: Test gamepad inputs
    --sim: Run in simulation mode instead of real robot

The collected data includes:
- Robot joint positions
- Camera images
- Actions taken
- Rewards received
- Episode termination states
"""
import gymnasium as gym
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import gym_lowcostrobot
from inputs import get_gamepad
import threading
import time
import argparse
from typing import Dict, Any, Callable, Tuple, Optional
import os
import cv2
import yaml
from gym_lowcostrobot.camera.d455 import D455Camera
# Type aliases for better readability
Observation = Dict[str, Any]
Action = np.ndarray
Policy = Callable[[Observation], Tuple[Action, bool]]


# Global variables for mouse callback
clicked_point = None
rgb_frame = None
depth_frame = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at pixel coordinates: ({x}, {y})")

def prepare_frame_data(
    obs: Observation,
    action: Action,
    reward: float,
    done: bool,
    terminated: bool,
    task_name: str = "Robot Task",
) -> Dict[str, Any]:
    """Prepare a single frame of data for the dataset.
    
    Args:
        obs: Current observation from the environment
        action: Action taken by the policy
        reward: Reward received from the environment
        done: Whether the episode is done
        terminated: Whether the episode was terminated (vs truncated)
        task_name: Description of the task being performed
        color_segmentation_config: Configuration for color segmentation
        
    Returns:
        Dictionary containing the frame data
    """
    frame_data = {
        "task": task_name,
        "action": action,
        "reward": np.array([reward], dtype=np.float32),
        "done": np.array([done], dtype=bool),
        "terminated": np.array([terminated], dtype=bool),
        "observation.arm_qpos": obs["arm_qpos"],
        "observation.rgb": obs["rgb"],
        "observation.depth": obs["depth"],
        # assumed use_camera is true
        "observation.target_eepos": obs["target_eepos"].astype(np.float32),
        "observation.gripper_blocked": np.array([obs["gripper_blocked"]], dtype=np.float32).reshape(-1),
        "observation.ee_pos": obs["ee_pos"].astype(np.float32),
        "observation.log_is_success": np.array([obs["log_is_success"]], dtype=np.float32).reshape(-1),  
        "observation.segmentation": obs["segmentation"],
        "observation.estimated_target_pos": obs["estimated_target_pos"].astype(np.float32),
    }
    return frame_data

def collect_episodes(
    env: gym.Env,
    policy: Policy,
    dataset: LeRobotDataset,
    num_episodes: int,
    task_name: str = "Robot Task",
    color_segmentation_config: Optional[Dict] = None
) -> None:
    """Collect episodes using the given policy and store them in a LeRobotDataset.
    
    Args:
        env: Gymnasium environment
        policy: Policy function that takes observations and returns (action, reset) tuple
        dataset: LeRobotDataset instance to store the episodes
        num_episodes: Number of episodes to collect
        task_name: Description of the task being performed
        color_segmentation_config: Configuration for color segmentation
    """
    print("TODO: need to cframeollect terminal observation, currently not doing that.")
    for ep_idx in range(num_episodes):
        print(f"Collecting episode {ep_idx+1}/{num_episodes}")
        # reset the noise if running scripted policy
        if isinstance(policy, ScriptedLiftPolicy):
            policy.reset_noise()
        obs, _ = env.reset()
        done = False
        frame_idx = 0
        
        while not done:
            action, should_reset = policy(obs)
            print(f"step: {frame_idx}, act: {action}")
            if should_reset:
                print(f"Episode {ep_idx+1} reset and skipped.")
                break

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = prepare_frame_data(obs, action, reward, done, terminated, task_name)
            dataset.add_frame(frame)
            
            obs = next_obs
            frame_idx += 1
    
        if not should_reset:
            dataset.save_episode()
            print(f"Episode {ep_idx+1} completed with {frame_idx} frames")
        dataset.clear_episode_buffer()

def create_or_load_dataset(
    repo_id: str,
    root_path: Optional[Path] = None,
    fps: int = 30,
    rgb_shape: Optional[Tuple[int, int, int]] = None,
    depth_shape: Optional[Tuple[int, int]] = None,
    segmentation_shape: Optional[Tuple[int, int]] = None
) -> LeRobotDataset:
    """Create a new LeRobotDataset or load an existing one.
    
    Args:
        repo_id: Repository ID for the dataset
        root_path: Local path to store/load the dataset (if None, uses default cache)
        fps: Frames per second for the dataset
        rgb_shape: Shape of RGB images
        depth_shape: Shape of depth images
        segmentation_shape: Shape of segmentation masks
        
    Returns:
        LeRobotDataset instance
    """
    # Check if the dataset already exists, if so create dataset = it
    if os.path.exists(root_path):
        dataset = LeRobotDataset(repo_id=repo_id, root=root_path)
        print(f"Loaded existing dataset from {root_path}")
        return dataset
    else:
        print(f"Creating new dataset at {root_path}")
    features = {
        "observation.rgb": {
            "dtype": "video",
            "shape": rgb_shape,
            "names": ["height", "width", "channels"]
        },
        "observation.depth": {
            "dtype": "uint16",
            "shape": depth_shape,
            "names": ["height", "width"]
        },
        "observation.arm_qpos": {
            "dtype": "float32",
            "shape": (6,),
            "names": None
        },
        "observation.segmentation": {
            "dtype": "video",
            "shape": rgb_shape,
            "names": ["height", "width", "channels"]
        },
        "observation.target_eepos": {
            "dtype": "float32",
            "shape": (3,),
            "names": None
        },
        "observation.gripper_blocked": {
            "dtype": "float32", # should it be boolean?
            "shape": (1,),
            "names": None
        },
        "observation.ee_pos": {
            "dtype": "float32",
            "shape": (4,),
            "names": None
        },
        "observation.log_is_success": {
            "dtype": "float32",
            "shape": (1,),
            "names": None
        },
        "observation.estimated_target_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": None
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": None
        },
        "reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None
        },
        "done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None
        },
        "terminated": {
            "dtype": "bool",
            "shape": (1,),
            "names": None
        },
    }
    
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root_path,
        features=features,
        use_videos=True,
        video_backend='pyav'
    )

class GamepadController:
    """Controller class for handling gamepad input and mapping it to robot actions."""
    
    def __init__(self, pos_sensitivity: float = 1.0, gripper_sensitivity: float = 1.0, rate_limit: float = 0.1):
        """Initialize the gamepad controller.
        
        Args:
            pos_sensitivity: Sensitivity multiplier for position controls
            gripper_sensitivity: Sensitivity multiplier for gripper controls
        """
        self.action = np.zeros(4, dtype=np.float32)  # [x, y, z, gripper]
        self.reset = False
        self._running = True
        self.pos_sensitivity = pos_sensitivity
        self.gripper_sensitivity = gripper_sensitivity
        self.rate_limit = rate_limit
        self.thread = threading.Thread(target=self._update_gamepad, daemon=True)
        self.thread.start()
    
    def _rescale_action(self, state: int, state_min: int = 0, state_max: int = 255, 
                       min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Rescale gamepad input to action range."""
        return np.float32((state - state_min) / (state_max - state_min) * (max_val - min_val) + min_val)
    
    def _update_gamepad(self) -> None:
        """Background thread to continuously poll gamepad inputs."""
        while self._running:
            try:
                events = get_gamepad()
                for event in events:
                    if event.ev_type == 'Absolute':
                        self._handle_absolute_event(event)
                    elif event.ev_type == 'Key':
                        self._handle_key_event(event)
            except Exception as e:
                print(f"Gamepad error: {e}")
                time.sleep(0.1)
    
    def _handle_absolute_event(self, event: Any) -> None:
        """Handle absolute axis events from the gamepad."""
        if event.code == 'ABS_X':
            self.action[1] = self._rescale_action(event.state) * self.pos_sensitivity
        elif event.code == 'ABS_Y':
            self.action[0] = self._rescale_action(event.state) * self.pos_sensitivity
        elif event.code == 'ABS_RZ':
            self.action[2] = self._rescale_action(event.state, min_val=1.0, max_val=-1.0) * self.pos_sensitivity
    
    def _handle_key_event(self, event: Any) -> None:
        """Handle key events from the gamepad."""
        if event.code == 'BTN_BASE4':
            self.reset = event.state == 1
        elif event.code == 'BTN_BASE2' and event.state == 1:
            self.action[3] = 1.0
        elif event.code == 'BTN_BASE' and event.state == 1:
            self.action[3] = -1.0
    
    def get_action(self, obs: Observation) -> Tuple[Action, bool]:
        """Get the current action and reset state.
        
        Returns:
            Tuple of (action array, reset flag)
        """
        time.sleep(self.rate_limit)  # Rate limiting
        action = np.where(np.abs(self.action) < 0.05, 0, self.action)
        action = action.clip(-1.0, 1.0).astype(np.float32)
        # print(f"Action: {action}, Reset: {self.reset}")
        return action.copy(), self.reset
    
    def stop(self) -> None:
        """Stop the gamepad polling thread."""
        self._running = False
        self.thread.join()

def run_gamepad_control(env: gym.Env, color_segmentation_config: Optional[Path] = None) -> None:
    """Run the environment with gamepad control for human testing/playing."""
    controller = GamepadController(pos_sensitivity=0.2, gripper_sensitivity=1.0, rate_limit=0.0)
    
    try:
        # Load color segmentation config
        segmentation_config = None
        rgb_shape, depth_shape, segmentation_shape = None, None, None
        
        if color_segmentation_config is not None and color_segmentation_config.exists():
            with open(color_segmentation_config, 'r') as f:
                segmentation_config = yaml.safe_load(f)
                crop_region = segmentation_config['crop_region']
                rgb_shape = (crop_region[3] - crop_region[1], crop_region[2] - crop_region[0], 3)
                depth_shape = (crop_region[3] - crop_region[1], crop_region[2] - crop_region[0])
                segmentation_shape = (crop_region[3] - crop_region[1], crop_region[2] - crop_region[0])
                print(f"RGB shape: {rgb_shape}, Depth shape: {depth_shape}, Segmentation shape: {segmentation_shape}")
        
        dataset_path = Path("./data/koch_robot_dataset_human")
        dataset = create_or_load_dataset(
            repo_id="gym-lowcostrobot/koch_robot_dataset_human",
            root_path=dataset_path,
            fps=30,
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
            segmentation_shape=segmentation_shape
        )
        
        collect_episodes(env, controller.get_action, dataset, num_episodes=5, 
                        task_name="Lift cube task (human)",
                        color_segmentation_config=segmentation_config)
        
        print(f"Dataset collected and saved to {dataset_path}")
    
    finally:
        controller.stop()
        env.close()

def run_random_collection(env: gym.Env, color_segmentation_config: Optional[Path] = None) -> None:
    """Run the environment with a random policy to collect baseline data."""
    def random_policy(obs: Observation) -> Tuple[Action, bool]:
        """Simple random policy for baseline data collection."""
        return np.random.uniform(-1, 1, size=4), False
   
    try:
        # Load color segmentation config
        segmentation_config = None
        rgb_shape, depth_shape, segmentation_shape = None, None, None
        
        if color_segmentation_config is not None and color_segmentation_config.exists():
            with open(color_segmentation_config, 'r') as f:
                segmentation_config = yaml.safe_load(f)
                crop_region = segmentation_config['crop_region']
                rgb_shape = (crop_region[3] - crop_region[1], crop_region[2] - crop_region[0], 3)
                depth_shape = (crop_region[3] - crop_region[1], crop_region[2] - crop_region[0])
                segmentation_shape = (crop_region[3] - crop_region[1], crop_region[2] - crop_region[0])
                print(f"RGB shape: {rgb_shape}, Depth shape: {depth_shape}, Segmentation shape: {segmentation_shape}")
        
        dataset_path = Path("./data/koch_robot_dataset_random")
        dataset = create_or_load_dataset(
            repo_id="gym-lowcostrobot/koch_robot_dataset_random",
            root_path=dataset_path,
            fps=30,
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
            segmentation_shape=segmentation_shape
        )
        
        collect_episodes(env, random_policy, dataset, num_episodes=10,
                        task_name="Lift cube task (random baseline)",
                        color_segmentation_config=segmentation_config)
        
        print(f"Dataset collected and saved to {dataset_path}")
    
    finally:
        env.close()

class ScriptedLiftPolicy:
    """A scripted policy for the lift cube task with camera-based object targeting."""

    def __init__(self, env):
        # Store environment for action scaling
        self.env = env
        
        # Control parameters
        self.x_error = 0.008
        self.y_error = 0.004
        # self.x_offset = 0.01
        self.x_offset_noisy = 0.0
        self.reset_noise()
        self.z_limit = 0.038
        self.z_offset = 0.01
        self.z_step = 0.01
        self.z_desc_limit = 0.01
        self.z_desc_error = -0.003
        self.g_grasp = 0.9
        self.g_close = 0.1
        self.target = None
        self.noise_updated = True

    def reset_noise(self):
        self.x_offset_noisy = np.clip(np.random.normal(0, 0.03), -0.02, 0.03)
        print(f"x_offset_noisy::::::::::::: {self.x_offset_noisy}")

    def __call__(self, obs: Observation) -> Tuple[Action, bool]:
        """Generate actions based on current observation.
        uniform
        Args:
            obs: Current observation from the environment
            
        Returns:
            Tuple of (action array, reset flag)
        """
        self.target = obs['target_eepos']
        
        blocked = obs["gripper_blocked"]
        ee_pos = obs['ee_pos']
        target = self.target.copy()  # Use the camera-set target
        # target[0] += self.x_offset
        target[0] += self.x_offset_noisy

        x, y, z, g = ee_pos[0], ee_pos[1], ee_pos[2], ee_pos[3]
        dx, dy, dz = target - ee_pos[:3]
        
        action = np.zeros(4, dtype=np.float32)
       
        if (abs(dx) <= self.x_error and abs(dy) <= self.y_error) or blocked:
            if self.z_desc_limit - z < self.z_desc_error and not blocked:
                # Above target - approaching
                action[3] = max(0, self.g_grasp - g)
                action[2] = min(0, np.sign(self.z_desc_limit - z) * self.z_step)
                # add a small random noise in x direction
                action[0] += np.random.uniform(-0.003, 0.005)
            else:
                # At target - gripping or lifting
                if blocked:
                    # Lifting
                    action[3] = min(0, self.g_close - g)
                    action[2] = 3 * self.z_step
                elif self.g_close-g > -0.03:
                    action = np.array([0, 0, 0.05, 0], dtype=np.float32)
                    # add a small random noise to the xy of the action 
                    action[:2] += np.random.uniform(-0.005, 0.005, size=2)
                    action[2] += 0.01
                    print("!!! gripping failed. Resetting position and noise")
                    self.reset_noise()
                else:
                    # Gripping
                    action[3] = min(0, self.g_close - g)
                    print(action)
        else:
            # Moving to target xy position
            if abs(dx) > self.x_error:
                action[0] = dx
            if abs(dy) > self.y_error:
                action[1] = dy
            if z < self.z_limit:
                action[2] = self.z_offset

        # Scale the action before returning it
        scaled_action = self.env.unwrapped.get_scaled_action(action)
        return scaled_action, False
    

def run_scripted_policy_collection(env: gym.Env, color_segmentation_config: Optional[Path] = None) -> None:
    """Run the environment with a scripted policy to collect data."""
    try:
        # Load color segmentation config
        segmentation_config = None
        if color_segmentation_config is not None and color_segmentation_config.exists():
            with open(color_segmentation_config, 'r') as f:
                segmentation_config = yaml.safe_load(f)
                crop_region = segmentation_config['crop_region']
                rgb_shape = (crop_region[3] - crop_region[1], crop_region[2] - crop_region[0], 3)
                depth_shape = (crop_region[3] - crop_region[1], crop_region[2] - crop_region[0])
                segmentation_shape = (crop_region[3] - crop_region[1], crop_region[2] - crop_region[0])
                print(f"RGB shape: {rgb_shape}, Depth shape: {depth_shape}, Segmentation shape: {segmentation_shape}")

        dataset_path = Path("./data/koch_robot_dataset_scripted_v1.2")
        dataset = create_or_load_dataset(
            repo_id="edwhu/koch_robot_dataset_scripted_v1.2",
            root_path=dataset_path,
            fps=30,
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
            segmentation_shape=segmentation_shape
        )
        
        policy = ScriptedLiftPolicy(env=env)
        try:
            collect_episodes(env, policy, dataset, num_episodes=10,
                            task_name="Lift cube task (scripted policy)",
                            color_segmentation_config=segmentation_config)
            
            print(f"Dataset collected and saved to {dataset_path}")

        finally:
            # policy.cleanup()
            pass
    
    finally:
        env.close()

def transform_cropped_to_full_image(pixel_x: int, pixel_y: int, x1: int, y1: int) -> Tuple[int, int]:
    """Transform the x, y position of a pixel in the cropped image to the x, y position of the pixel in the full image."""
    return pixel_x + x1, pixel_y + y1

def run_debug_gamepad() -> None:
    """Debug mode to print gamepad events and their values."""
    print("Gamepad test mode - Press controls to see their codes and values")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            events = get_gamepad()
            for event in events:
                print(f"Event Type: {event.ev_type}")
                print(f"Event Code: {event.code}")
                print(f"Event State: {event.state}")
                print("------------------------")
                
    except KeyboardInterrupt:
        print("\nExiting gamepad test...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect robot data using gamepad, random policy, or learned policy.')
    parser.add_argument('--mode', type=str, 
                      choices=['gamepad', 'random', 'scripted', 'debug_gamepad'], 
                      default='gamepad', 
                      help='Collection mode: gamepad for human control, random for baseline, learned for policy')   
    parser.add_argument('--sim', action='store_true', help='Run in simulation mode')
    parser.add_argument('--koch_device', type=str, default='/dev/ttyACM0', help='Koch device to use: auto, real, sim')
    parser.add_argument('--color_config', type=str, default='./color_segmentation_results/color_segmentation_config.yaml',
                      help='Path to color segmentation configuration file')
    args = parser.parse_args()

    os.environ['KOCH_DEVICE_NAME'] = args.koch_device
    
    env_id = ("LiftCubeStateGamepadHumanRender-v0" if args.mode == 'gamepad' 
              else "LiftCubeStateNoisyHumanRender-v0") if args.sim else "LiftCubeStateReal-v0"

    # Set up color segmentation config path
    color_segmentation_config = Path(args.color_config)
    if not color_segmentation_config.exists():
        print(f"Warning: Color segmentation config file not found at {color_segmentation_config}")
        print("Proceeding without color segmentation and cropping.")
        color_segmentation_config = None

    if args.mode == 'debug_gamepad':
        run_debug_gamepad()
    else:
        print(f"env_id: {env_id}")
        env = gym.make(env_id)
        if args.mode == 'gamepad':
            print("Running gamepad control mode...")
            run_gamepad_control(env, color_segmentation_config)
        elif args.mode == 'random':
            print("Running random policy collection mode (baseline)...")
            run_random_collection(env, color_segmentation_config)
        elif args.mode == 'scripted':
            print("Running scripted policy collection mode...")
            run_scripted_policy_collection(env, color_segmentation_config)
