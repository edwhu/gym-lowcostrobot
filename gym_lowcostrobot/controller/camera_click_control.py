import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.envs.registration')


import os
import time
import numpy as np
import cv2
import gymnasium as gym
import gym_lowcostrobot
from gym_lowcostrobot.envs.lift_cube_state_env_real import LiftCubeStateRealEnv
from gym_lowcostrobot.envs.lift_cube_state_env import LiftCubeStateEnv
from gym_lowcostrobot.camera.d455 import D455Camera

# Global variables for mouse callback
clicked_point = None
rgb_frame = None
depth_frame = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at pixel coordinates: ({x}, {y})")

def main():
    """
    Camera-based robot control program
    
    Click on a point in the camera view to move the robot to that position.
    
    Controls:
    - Mouse click: Select target position
    - z/x: Close/open gripper
    - r: Reset environment
    - q: Exit
    """
    global clicked_point, rgb_frame, depth_frame
    
    # Load calibration matrix
    calibration_path = os.path.join('results', 'calibration_matrix.npy')
    if not os.path.exists(calibration_path):
        print(f"Calibration matrix not found at {calibration_path}")
        return
    
    T_base_camera = np.load(calibration_path)
    print(f"Loaded calibration matrix:\n{T_base_camera}")
    
    # Initialize camera
    camera = D455Camera(
        enable_rgb=True,
        enable_depth=True,
        rgb_resolution=(848, 480),
        depth_resolution=(848, 480),
        fps=30,
        align_frames=True
    )
    
    camera.set_calibration_matrix(T_base_camera)
    
    if not camera.start():
        print("Failed to start camera")
        return
    
    env = LiftCubeStateRealEnv(
        observation_mode="both",
        render_mode="human", 
        action_mode="nullspace", 
    )
    
    obs, info = env.reset()
    print(f"Initial end-effector position: {obs['ee_pos'][:3]}")
    env.render()
    
    cv2.namedWindow("Camera View")
    cv2.setMouseCallback("Camera View", mouse_callback)
    
    print("\n=== Camera-Based Robot Control ===")
    print("Instructions:")
    print("  Click on a point in the camera view to move the robot to that position")
    print("  z/x: Close/open gripper")
    print("  r: Reset environment")
    print("  q: Exit")
    print("======================================\n")
    
    # Position control sensitivity
    pos_sensitivity = 0.1
    gripper_sensitivity = 1.0
    
    try:
        while True:
            rgb_frame, depth_frame = camera.get_frames()
            
            if rgb_frame is None or depth_frame is None:
                print("Failed to get frames")
                time.sleep(0.1)
                continue
            
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            combined = np.hstack((rgb_frame, depth_colormap))
            
            # Draw current end-effector position if available
            if 'ee_pos' in obs:
                cv2.putText(combined, 
                           f"EE Position: {obs['ee_pos'][0]:.3f}, {obs['ee_pos'][1]:.3f}, {obs['ee_pos'][2]:.3f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Camera View", combined)
            
            if clicked_point is not None:
                x, y = clicked_point
                clicked_point = None  
                
                # Check if click is in the RGB frame (left half of combined image)
                if x < rgb_frame.shape[1]:
                    point_base = camera.get_3d_point_robot_base(x, y, depth_frame)
                    
                    if point_base is not None:
                        print(f"3D point in robot base frame: {point_base}")
                        
                        # Calculate movement vector (from current position to target)
                        current_pos = obs['ee_pos'][:3]
                        movement_vector = point_base - current_pos
                        
                        # Scale movement for safety
                        movement_vector = movement_vector 
                        
                        action = np.zeros(4)
                        action[:3] = movement_vector
                        action = env.get_scaled_action(action)
                        
                        obs, reward, terminated, truncated, info = env.step(action)
                        
                        print(f"New end-effector position: {obs['ee_pos']}")
                        env.render()
                    else:
                        print("Invalid depth at clicked point")
            
            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                # Reset environment
                obs, info = env.reset()
                print(f"Environment reset. End-effector position: {obs['ee_pos'][:3]}")
                env.render()
            elif key == ord('z'):
                # Close gripper
                action = np.array([0.0, 0.0, 0.0, 1.0 * gripper_sensitivity])  # TODO check if it is delta
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Closing gripper. Position: {obs['ee_pos'][3]}")
                env.render()
            elif key == ord('x'):
                # Open gripper
                action = np.array([0.0, 0.0, 0.0, -1.0 * gripper_sensitivity]) # TODO check if it is delta
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Opening gripper. Position: {obs['ee_pos'][3]}")
                env.render()
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        camera.stop()
        env.close()
        cv2.destroyAllWindows()
        print("Program closed.")

if __name__ == "__main__":
    main() 