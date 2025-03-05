import time
import numpy as np
from pynput import keyboard
from gym_lowcostrobot.envs.lift_cube_state_env_twin import LiftCubeStateEnv

# Global variables
action = np.zeros(4)  # [x, y, z, gripper]
running = True
env = None
obs = None

def on_press(key):
    """Handle key press events"""
    global action, running, env, obs
    
    try:
        # Regular keys
        if hasattr(key, 'char'):
            k = key.char.lower()
            
            # WASD for XY movement
            if k == 'w':
                action[0] = -1.0
            elif k == 's':
                action[0] = 1.0
            elif k == 'a':
                action[1] = -1.0
            elif k == 'd':
                action[1] = 1.0
            
            # RF for Z movement
            elif k == 'r':
                action[2] = 1.0
            elif k == 'f':
                action[2] = -1.0
            
            # EC for gripper
            elif k == 'e':
                action[3] = 1.0
            elif k == 'c':
                action[3] = -1.0
    
    except AttributeError:
        pass
        
    # Special keys
    if key == keyboard.Key.esc:
        print("Exiting...")
        running = False
        return False
    elif key == keyboard.Key.ctrl:
        print("Resetting environment...")
        obs, _ = env.reset()

def on_release(key):
    """Handle key release events"""
    global action
    
    try:
        if hasattr(key, 'char'):
            k = key.char.lower()
            
            # Reset controls on key release
            if k == 'w' or k == 's':
                action[0] = 0.0
            elif k == 'a' or k == 'd':
                action[1] = 0.0
            elif k == 'r' or k == 'f':
                action[2] = 0.0
            elif k == 'e' or k == 'c':
                action[3] = 0.0
    except:
        pass

def main():
    """
    Minimal keyboard control for robot
    
    Controls:
    - W/S: Forward/Backward
    - A/D: Left/Right
    - R/F: Up/Down
    - E/C: Close/Open gripper
    - Ctrl: Reset
    - ESC: Exit
    """
    global action, running, env, obs
    
    # Create environment
    env = LiftCubeStateEnv(
        observation_mode="both",
        render_mode="human", 
        action_mode="nullspace", 
        use_action_noise=False
    )
    
    # Settings
    pos_speed = 0.05
    grip_speed = 0.5
    
    print("\n=== Robot Keyboard Control ===")
    print("W/S: Forward/Backward")
    print("A/D: Left/Right")
    print("R/F: Up/Down")
    print("E/C: Close/Open gripper")
    print("Ctrl: Reset")
    print("ESC: Exit")
    print("============================\n")
    
    # Initialize
    obs, _ = env.reset()
    print(f"Ready! Position: {obs['ee_pos'][:3]}")
    
    # Start keyboard listener
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    listener.start()
    
    try:
        # Main loop
        last_print = time.time()
        
        while running:
            # Scale action
            scaled = action.copy()
            scaled[:3] *= pos_speed
            scaled[3] *= grip_speed
            
            # Execute action if any input
            if np.any(scaled != 0):
                obs, reward, done, _, _ = env.step(scaled)
                
                # Print position occasionally
                now = time.time()
                if now - last_print > 2.0:
                    print(f"Position: {obs['ee_pos'][:3]}")
                    last_print = now
                
                # Auto-reset on success
                if done:
                    print("Success! Resetting...")
                    obs, _ = env.reset()
            
            # Render and sleep
            env.render()
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        listener.stop()
        env.close()

if __name__ == "__main__":
    main() 