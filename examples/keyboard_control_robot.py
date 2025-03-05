import os
import time
import argparse
import numpy as np
import gymnasium as gym
import gym_lowcostrobot  # Import the low-cost robot environments

def keyboard_control():
    """
    Control the robot using keyboard inputs.
    
    Key mappings:
    - w/s: Move forward/backward (X-axis)
    - a/d: Move left/right (Y-axis)
    - q/e: Move up/down (Z-axis)
    - z/x: Close/open gripper
    - r: Reset environment
    - ESC: Exit
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Keyboard control for low-cost robot.")
    parser.add_argument('--env-name', type=str, default='LiftCubeState-v0', 
                        help='Specify the gym-lowcost robot env to test.')
    parser.add_argument('--render-mode', type=str, default='human',
                        help='Render mode (human or rgb_array).')
    parser.add_argument('--action-mode', type=str, default='nullspace',
                        help='Action mode (nullspace or joint).')
    args = parser.parse_args()

    # Set up environment
    env = gym.make(args.env_name, 
                   observation_mode="both",
                   render_mode=args.render_mode, 
                   action_mode=args.action_mode, 
                   use_action_noise=False)
    
    # Define key-to-action mapping
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
    
    print("\n=== Keyboard Control for Low-Cost Robot ===")
    print("Controls:")
    print("  w/s: Move forward/backward (X-axis)")
    print("  a/d: Move left/right (Y-axis)")
    print("  q/e: Move up/down (Z-axis)")
    print("  z/x: Close/open gripper")
    print("  r: Reset environment")
    print("  ESC or any other key: Exit")
    print("=======================================\n")
    
    try:
        # Initial reset
        obs, info = env.reset()
        print(f"Initial end-effector position: {obs['ee_pos'][:3]}")
        env.render()
        
        while True:
            # Get keyboard input
            raw_key = input("Enter action key (w/a/s/d/q/e/z/x, r to reset, any other key to exit): ")
            
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
                
                print(f"Reward: {reward}")
                
                # Render the environment
                env.render()
                
                # Check if episode is done
                if terminated:
                    print("Success! Task completed.")
                    choice = input("Continue with new episode? (y/n): ")
                    if choice.lower() != 'y':
                        break
                    obs, info = env.reset()
                    print(f"New episode started. End-effector position: {obs['ee_pos'][:3]}")
            else:
                # Exit on any other key
                print("Exiting...")
                break
                
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
    finally:
        # Clean up
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    keyboard_control() 