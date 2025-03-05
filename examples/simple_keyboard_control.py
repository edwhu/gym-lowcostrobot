import os
import time
import numpy as np
import gymnasium as gym
import gym_lowcostrobot
from gym_lowcostrobot.envs.lift_cube_state_env_twin import LiftCubeStateEnv

def main():
    """
    Simple keyboard control script for the LiftCubeStateEnv robot environment
    
    Controls:
    - w/s: Move forward/backward (X-axis)
    - a/d: Move left/right (Y-axis)
    - q/e: Move up/down (Z-axis)
    - z/x: Close/open gripper
    - r: Reset environment
    - Any other key: Exit
    """
    # Create environment
    env = LiftCubeStateEnv(
        observation_mode="both",
        render_mode="human", 
        action_mode="nullspace", 
        use_action_noise=False
    )
    
    # Define keyboard mapping
    key_action_map = {
        'w': np.array([-1.0, 0.0, 0.0, 0.0]),  # Forward (negative X)
        's': np.array([1.0, 0.0, 0.0, 0.0]),   # Backward (positive X)
        'a': np.array([0.0, -1.0, 0.0, 0.0]),  # Left (negative Y)
        'd': np.array([0.0, 1.0, 0.0, 0.0]),   # Right (positive Y)
        'q': np.array([0.0, 0.0, 1.0, 0.0]),   # Up (positive Z)
        'e': np.array([0.0, 0.0, -1.0, 0.0]),  # Down (negative Z)
        'z': np.array([0.0, 0.0, 0.0, 1.0]),   # Close gripper
        'x': np.array([0.0, 0.0, 0.0, -1.0]),  # Open gripper
    }
    
    # Sensitivity settings
    pos_sensitivity = 0.1  # Position control sensitivity
    gripper_sensitivity = 1.0  # Gripper control sensitivity
    
    print("\n=== Low-Cost Robot Keyboard Control ===")
    print("Controls:")
    print("  w/s: Move forward/backward (X-axis)")
    print("  a/d: Move left/right (Y-axis)")
    print("  q/e: Move up/down (Z-axis)")
    print("  z/x: Close/open gripper")
    print("  r: Reset environment")
    print("  Any other key: Exit")
    print("======================================\n")
    
    # Initialize environment
    obs, info = env.reset()
    print(f"Initial end-effector position: {obs['ee_pos'][:3]}")
    env.render()
    
    try:
        while True:
            # Get keyboard input
            raw_key = input("Enter control key (w/a/s/d/q/e/z/x, r to reset, any other key to exit): ")
            
            # Check for reset command
            if raw_key.lower() == 'r':
                obs, info = env.reset()
                print(f"Environment reset. End-effector position: {obs['ee_pos'][:3]}")
                env.render()
                continue
                
            # Check for valid action key
            if raw_key.lower() in key_action_map:
                # Get and scale action
                action = key_action_map[raw_key.lower()].copy()
                action[:3] *= pos_sensitivity  # Scale position actions
                action[-1] *= gripper_sensitivity  # Scale gripper action
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Display information
                print(f"Action: {raw_key.lower()}")
                print(f"End-effector position: {obs['ee_pos'][:3]}")
                print(f"Gripper position: {obs['ee_pos'][3]}")
                
                if 'goal_pos' in info:
                    print(f"Goal position: {info['goal_pos']}")
                
                # Render environment
                env.render()
                
                # Check if completed
                if terminated:
                    print("Success! Task completed.")
                    choice = input("Continue with new episode? (y/n): ")
                    if choice.lower() != 'y':
                        break
                    obs, info = env.reset()
                    print(f"New episode started. End-effector position: {obs['ee_pos'][:3]}")
            else:
                # Exit on any other key
                print("Need valid key...")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
    finally:
        # Clean up
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main() 