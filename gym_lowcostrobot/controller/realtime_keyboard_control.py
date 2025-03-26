import os
import time
import numpy as np
import gymnasium as gym
from gym_lowcostrobot.envs.lift_cube_state_env_real import LiftCubeStateEnv
from pynput import keyboard

# Global variables
current_action = np.zeros(4)  # [x, y, z, gripper]
is_running = True
env = None
obs = None
info = None
ACTION_MAGNITUDE = 1.0

"""
    real-time keyboard control of the robot.
    
    Controls:
    - W/S: Move forward/backward (X-axis)
    - A/D: Move left/right (Y-axis)
    - U/I: Move up/down (Z-axis)
    - J/K: Close/open gripper
    - Right Ctrl: Reset environment
    - ESC: Exit program

    Tony 03-22-2025 V1
"""


def on_press(key):
    """Handle key press events"""
    global current_action, is_running, env, obs, info
    
    try:
        if hasattr(key, 'char'):
            k = key.char.lower()
            
            if k == 'w':  # Forward (negative X)
                current_action[0] = -ACTION_MAGNITUDE
            elif k == 's':  # Backward (positive X)
                current_action[0] = ACTION_MAGNITUDE
            elif k == 'a':  # Left (negative Y)
                current_action[1] = -ACTION_MAGNITUDE
            elif k == 'd':  # Right (positive Y)
                current_action[1] = ACTION_MAGNITUDE
            elif k == 'u':  # Up (positive Z)
                current_action[2] = ACTION_MAGNITUDE
            elif k == 'i':  # Down (negative Z)
                current_action[2] = -ACTION_MAGNITUDE
            
            # Gripper controls
            elif k == 'j':  # Close gripper
                current_action[3] = ACTION_MAGNITUDE
            elif k == 'k':  # Open gripper
                current_action[3] = -ACTION_MAGNITUDE
            
  
    except AttributeError:
        # Check for special keys
        if key == keyboard.Key.esc:
            print("Exiting...")
            is_running = False
            return False  # Stop listener
        elif key == keyboard.Key.ctrl_r:
            print("Resetting environment...")
            obs, info = env.reset()
            print(f"Environment reset. End-effector position: {obs['ee_pos'][:3]}")

def on_release(key):
    """Handle key release events"""
    global current_action
    if hasattr(key, 'char'):
        k = key.char.lower()
        
        # Reset movement controls
        if k == 'w' or k == 's':
            current_action[0] = 0.0
        elif k == 'a' or k == 'd':
            current_action[1] = 0.0
        elif k == 'i' or k == 'u':
            current_action[2] = 0.0
        # Reset gripper controls
        elif k == 'j' or k == 'k':
            current_action[3] = 0.0

def main():

    global current_action, is_running, env, obs, info
    
    # Create environment
    env = LiftCubeStateEnv(
        observation_mode="both",
        render_mode="human", 
        action_mode="nullspace", 
        use_action_noise=False
    )
    
    # Sensitivity settings
    pos_sensitivity = 0.05  # Position control sensitivity (reduced for smoother control)
    gripper_sensitivity = 0.5  # Gripper control sensitivity
    
    print("\n=== Real-time Keyboard Control for Robot ===")
    print("Controls:")
    print("  W/S: Move forward/backward (X-axis)")
    print("  A/D: Move left/right (Y-axis)")
    print("  I/K: Move up/down (Z-axis)")
    print("  J/L: Close/open gripper")
    print("  Right Ctrl + R: Reset environment")
    print("  ESC or Ctrl C: Exit")
    print("==========================================\n")
    
    # Initialize environment
    obs, info = env.reset()
    print(f"Initial end-effector position: {obs['ee_pos'][:3]}")
    env.render()
    
    # Start keyboard listener in a non-blocking way
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    listener.start()
    
    try:
        # Main control loop
        while is_running:
            # Scale action
            scaled_action = current_action.copy()
            scaled_action[:3] *= pos_sensitivity
            scaled_action[3] *= gripper_sensitivity
            
            # Only take action if there's any input
            if np.any(scaled_action != 0):
                # Execute action
                obs, reward, terminated, truncated, info = env.step(scaled_action)
                
                # Display information (less frequently to avoid console spam)
                if np.random.random() < 0.1:  # Only print 10% of the time
                    print(f"End-effector position: {obs['ee_pos'][:3]}")
                    if 'goal_pos' in info:
                        print(f"Goal position: {info['goal_pos']}")
                
                # Check if completed
                if terminated:
                    print("Success! Task completed.")
                    print("Resetting environment in 3 seconds...")
                    time.sleep(3)
                    obs, info = env.reset()
                    print(f"New episode started. End-effector position: {obs['ee_pos'][:3]}")
            
            # Render environment
            env.render()
            
            # Small sleep to prevent CPU overload
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
    finally:
        # Clean up
        listener.stop()
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main() 