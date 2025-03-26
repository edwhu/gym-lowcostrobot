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
# Type aliases for better readability
Observation = Dict[str, Any]
Action = np.ndarray
Policy = Callable[[Observation], Tuple[Action, bool]]

def prepare_frame_data(
    obs: Observation,
    action: Action,
    reward: float,
    done: bool,
    terminated: bool,
    task_name: str = "Robot Task"
) -> Dict[str, Any]:
    """Prepare a single frame of data for the dataset.
    
    Args:
        obs: Current observation from the environment
        action: Action taken by the policy
        reward: Reward received from the environment
        done: Whether the episode is done
        terminated: Whether the episode was terminated (vs truncated)
        task_name: Description of the task being performed
        
    Returns:
        Dictionary containing the frame data
    """
    return {
        "task": task_name,
        "action": action,
        "reward": np.array([reward], dtype=np.float32),
        "done": np.array([done], dtype=bool),
        "terminated": np.array([terminated], dtype=bool),
        # "observation.images.log_image_front": obs["log_image_front"],
        "observation.arm_qpos": obs["arm_qpos"],
    }

def collect_episodes(
    env: gym.Env,
    policy: Policy,
    dataset: LeRobotDataset,
    num_episodes: int,
    task_name: str = "Robot Task"
) -> None:
    """Collect episodes using the given policy and store them in a LeRobotDataset.
    
    Args:
        env: Gymnasium environment
        policy: Policy function that takes observations and returns (action, reset) tuple
        dataset: LeRobotDataset instance to store the episodes
        num_episodes: Number of episodes to collect
        task_name: Description of the task being performed
    """
    for ep_idx in range(num_episodes):
        print(f"Collecting episode {ep_idx+1}/{num_episodes}")
        
        obs, _ = env.reset()
        done = False
        frame_idx = 0
        
        while not done:
            action, should_reset = policy(obs)
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

def create_dataset(
    repo_id: str,
    root_path: Optional[Path] = None,
    fps: int = 30
) -> LeRobotDataset:
    """Create a LeRobotDataset for storing episodes.
    
    Args:
        repo_id: Repository ID for the dataset
        root_path: Local path to store the dataset (if None, uses default cache)
        fps: Frames per second for the dataset
        
    Returns:
        LeRobotDataset instance
    """
    features = {
        # "observation.images.log_image_front": {
        #     "dtype": "video",
        #     "shape": (64, 64, 3),
        #     "names": ["height", "width", "channels"]
        # },
        "observation.arm_qpos": {
            "dtype": "float32",
            "shape": (6,),
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

def run_gamepad_control(env: gym.Env) -> None:
    """Run the environment with gamepad control for human testing/playing."""
    controller = GamepadController(pos_sensitivity=0.2, gripper_sensitivity=1.0, rate_limit=0.0)
    
    try:
        dataset_path = Path("./data/koch_robot_dataset_human")
        dataset = create_dataset(
            repo_id="gym-lowcostrobot/koch_robot_dataset_human",
            root_path=dataset_path,
            fps=30
        )
        
        collect_episodes(env, controller.get_action, dataset, num_episodes=5, 
                        task_name="Lift cube task (human)")
        
        print(f"Dataset collected and saved to {dataset_path}")
    
    finally:
        controller.stop()
        env.close()

def run_random_collection(env: gym.Env) -> None:
    """Run the environment with a random policy to collect baseline data."""
    def random_policy(obs: Observation) -> Tuple[Action, bool]:
        """Simple random policy for baseline data collection."""
        return np.random.uniform(-1, 1, size=4), False
   
    try:
        dataset_path = Path("./data/koch_robot_dataset_random")
        dataset = create_dataset(
            repo_id="gym-lowcostrobot/koch_robot_dataset_random",
            root_path=dataset_path,
            fps=30
        )
        
        collect_episodes(env, random_policy, dataset, num_episodes=10,
                        task_name="Lift cube task (random baseline)")
        
        print(f"Dataset collected and saved to {dataset_path}")
    
    finally:
        env.close()

def run_learned_policy_collection(env: gym.Env) -> None:
    """Run the environment with a learned policy to collect data.
    
    Note: This is a placeholder function. The actual learned policy implementation
    should be added here when available.
    """
    def learned_policy(obs: Observation) -> Tuple[Action, bool]:
        """Placeholder for the learned policy.
        
        This function should be replaced with the actual learned policy implementation.
        Currently returns random actions as a placeholder.
        """
        # TODO: Replace with actual learned policy implementation
        print("Warning: Using placeholder learned policy (random actions)")
        return np.random.uniform(-1, 1, size=4), False
   
    try:
        dataset_path = Path("./data/koch_robot_dataset_learned")
        dataset = create_dataset(
            repo_id="gym-lowcostrobot/koch_robot_dataset_learned",
            root_path=dataset_path,
            fps=30
        )
        
        collect_episodes(env, learned_policy, dataset, num_episodes=10,
                        task_name="Lift cube task (learned policy)")
        
        print(f"Dataset collected and saved to {dataset_path}")
    
    finally:
        env.close()

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
                      choices=['gamepad', 'random', 'learned', 'debug_gamepad'], 
                      default='gamepad', 
                      help='Collection mode: gamepad for human control, random for baseline, learned for policy')
    parser.add_argument('--sim', action='store_true', help='Run in simulation mode')
    parser.add_argument('--koch_device', type=str, default='/dev/ttyACM0', help='Koch device to use: auto, real, sim')
    args = parser.parse_args()

    os.environ['KOCH_DEVICE_NAME'] = args.koch_device
    
    env_id = ("LiftCubeStateGamepadHumanRender-v0" if args.mode == 'gamepad' 
              else "LiftCubeStateNoisyHumanRender-v0") if args.sim else "LiftCubeStateReal-v0"
    
    if args.mode == 'debug_gamepad':
        run_debug_gamepad()
    else:
        env = gym.make(env_id)
        if args.mode == 'gamepad':
            print("Running gamepad control mode...")
            run_gamepad_control(env)
        elif args.mode == 'random':
            print("Running random policy collection mode (baseline)...")
            run_random_collection(env)
        else:  # learned
            print("Running learned policy collection mode (placeholder)...")
            run_learned_policy_collection(env)

