import os
import time
import numpy as np
import cv2
import gymnasium as gym
import gym_lowcostrobot
from gym_lowcostrobot.envs.lift_cube_state_env_twin import LiftCubeStateEnv
from gym_lowcostrobot.camera.d455 import D455Camera

# Global variables for mouse callback
clicked_point = None
rgb_frame = None
depth_frame = None
camera = None  # Will hold the camera instance
env = None     # Will hold the environment instance

# Transformation correction factors (can be adjusted during runtime)
depth_scale = 1.0
depth_offset = 0.0
transform_correction = np.eye(4)  # Identity transformation as default

def mouse_callback(event, x, y, flags, param):
    """Callback function for mouse events"""
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at pixel coordinates: ({x}, {y})")
        
        # Display depth information at clicked point
        if camera is not None and depth_frame is not None:
            # Get raw depth value
            raw_depth = depth_frame[y, x] if x < depth_frame.shape[1] and y < depth_frame.shape[0] else 0
            
            # Get 3D point in camera frame
            point_camera = camera.get_3d_point(x, y)
            
            # Get 3D point in robot base frame
            point_base = camera.get_3d_point_robot_base(x, y)
            
            print(f"Raw depth value: {raw_depth} mm")
            if point_camera is not None:
                print(f"3D point in camera frame: {point_camera}")
            if point_base is not None:
                print(f"3D point in robot base frame: {point_base}")
                
                # Apply correction factors for debugging
                corrected_point = apply_correction(point_base)
                print(f"Corrected point in robot base frame: {corrected_point}")

def apply_correction(point):
    """Apply correction factors to the point"""
    # Apply depth scaling and offset
    corrected_point = point.copy()
    corrected_point[2] = point[2] * depth_scale + depth_offset
    
    # Apply additional transformation correction if needed
    point_homogeneous = np.append(corrected_point, 1.0)
    corrected_homogeneous = transform_correction @ point_homogeneous
    return corrected_homogeneous[:3]

def update_correction_factors(key):
    """Update correction factors based on key press"""
    global depth_scale, depth_offset, transform_correction
    
    # Scale adjustments
    if key == ord('1'):
        depth_scale *= 0.95  # Decrease scale by 5%
        print(f"Depth scale decreased to {depth_scale:.3f}")
    elif key == ord('2'):
        depth_scale *= 1.05  # Increase scale by 5%
        print(f"Depth scale increased to {depth_scale:.3f}")
    
    # Offset adjustments
    elif key == ord('3'):
        depth_offset -= 0.01  # Decrease offset by 1cm
        print(f"Depth offset decreased to {depth_offset:.3f}")
    elif key == ord('4'):
        depth_offset += 0.01  # Increase offset by 1cm
        print(f"Depth offset increased to {depth_offset:.3f}")
    
    # X translation adjustments
    elif key == ord('5'):
        transform_correction[0, 3] -= 0.01  # Decrease X by 1cm
        print(f"X translation decreased to {transform_correction[0, 3]:.3f}")
    elif key == ord('6'):
        transform_correction[0, 3] += 0.01  # Increase X by 1cm
        print(f"X translation increased to {transform_correction[0, 3]:.3f}")
    
    # Y translation adjustments
    elif key == ord('7'):
        transform_correction[1, 3] -= 0.01  # Decrease Y by 1cm
        print(f"Y translation decreased to {transform_correction[1, 3]:.3f}")
    elif key == ord('8'):
        transform_correction[1, 3] += 0.01  # Increase Y by 1cm
        print(f"Y translation increased to {transform_correction[1, 3]:.3f}")
    
    # Z translation adjustments
    elif key == ord('9'):
        transform_correction[2, 3] -= 0.01  # Decrease Z by 1cm
        print(f"Z translation decreased to {transform_correction[2, 3]:.3f}")
    elif key == ord('0'):
        transform_correction[2, 3] += 0.01  # Increase Z by 1cm
        print(f"Z translation increased to {transform_correction[2, 3]:.3f}")
    
    # Reset corrections
    elif key == ord('c'):
        depth_scale = 1.0
        depth_offset = 0.0
        transform_correction = np.eye(4)
        print("Correction factors reset to default")
    
    # Save current correction factors
    elif key == ord('s'):
        save_correction_factors()

def save_correction_factors():
    """Save the current correction factors to a file"""
    correction_data = {
        'depth_scale': depth_scale,
        'depth_offset': depth_offset,
        'transform_correction': transform_correction.tolist()
    }
    
    # Save as numpy file
    np.save('omni_depth/results/correction_factors.npy', correction_data)
    print(f"Correction factors saved to omni_depth/results/correction_factors.npy")

def main():
    """
    Camera-based robot control program with transformation debugging
    
    Click on a point in the camera view to move the robot to that position.
    
    Controls:
    - Mouse click: Select target position
    - z/x: Close/open gripper
    - r: Reset environment
    - 1/2: Decrease/increase depth scale
    - 3/4: Decrease/increase depth offset
    - 5/6: Decrease/increase X translation
    - 7/8: Decrease/increase Y translation
    - 9/0: Decrease/increase Z translation
    - c: Reset correction factors
    - s: Save correction factors
    - m: Move to clicked point (with corrections)
    - q: Exit
    """
    global clicked_point, rgb_frame, depth_frame, camera, env
    
    # Load calibration matrix
    calibration_path = os.path.join('omni_depth', 'results', 'calibration_matrix.npy')
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
    
    # Set the calibration matrix
    camera.set_calibration_matrix(T_base_camera)
    
    # Start camera
    if not camera.start():
        print("Failed to start camera")
        return
    
    # Create environment
    env = LiftCubeStateEnv(
        observation_mode="both",
        render_mode="human", 
        action_mode="nullspace", 
        use_action_noise=False
    )
    
    # Initialize environment
    obs, info = env.reset()
    print(f"Initial end-effector position: {obs['ee_pos'][:3]}")
    env.render()
    
    # Create display window
    cv2.namedWindow("Camera View")
    cv2.setMouseCallback("Camera View", mouse_callback)
    
    print("\n=== Camera-Based Robot Control with Transformation Debugging ===")
    print("Instructions:")
    print("  Click on a point in the camera view to see its coordinates")
    print("  m: Move robot to clicked point (with corrections)")
    print("  z/x: Close/open gripper")
    print("  r: Reset environment")
    print("  1/2: Decrease/increase depth scale")
    print("  3/4: Decrease/increase depth offset")
    print("  5/6: Decrease/increase X translation")
    print("  7/8: Decrease/increase Y translation")
    print("  9/0: Decrease/increase Z translation")
    print("  c: Reset correction factors")
    print("  s: Save correction factors")
    print("  q: Exit")
    print("======================================\n")
    
    # Position control sensitivity
    pos_sensitivity = 0.1
    gripper_sensitivity = 1.0
    
    # Last clicked point (for move command)
    last_clicked_point_base = None
    
    try:
        while True:
            # Get frames from camera
            rgb_frame, depth_frame = camera.get_frames()
            
            if rgb_frame is None or depth_frame is None:
                print("Failed to get frames")
                time.sleep(0.1)
                continue
            
            # Convert depth to color map for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # Combine images side by side
            combined = np.hstack((rgb_frame, depth_colormap))
            
            # Draw current end-effector position if available
            if 'ee_pos' in obs:
                cv2.putText(combined, 
                           f"EE Position: {obs['ee_pos'][0]:.3f}, {obs['ee_pos'][1]:.3f}, {obs['ee_pos'][2]:.3f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw correction factors
            cv2.putText(combined, 
                       f"Scale: {depth_scale:.2f} Offset: {depth_offset:.2f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show images
            cv2.imshow("Camera View", combined)
            
            # Process mouse clicks
            if clicked_point is not None:
                x, y = clicked_point
                clicked_point = None  # Reset click
                
                # Check if click is in the RGB frame (left half of combined image)
                if x < rgb_frame.shape[1]:
                    # Get 3D point in robot base frame
                    point_base = camera.get_3d_point_robot_base(x, y)
                    
                    if point_base is not None:
                        # Apply correction factors
                        corrected_point = apply_correction(point_base)
                        last_clicked_point_base = corrected_point
            
            # Check for key press
            key = cv2.waitKey(1)
            
            # Update correction factors
            update_correction_factors(key)
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset environment
                obs, info = env.reset()
                print(f"Environment reset. End-effector position: {obs['ee_pos'][:3]}")
                env.render()
            elif key == ord('z'):
                # Close gripper
                action = np.array([0.0, 0.0, 0.0, 1.0 * gripper_sensitivity])
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Closing gripper. Position: {obs['ee_pos'][3]}")
                env.render()
            elif key == ord('x'):
                # Open gripper
                action = np.array([0.0, 0.0, 0.0, -1.0 * gripper_sensitivity])
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Opening gripper. Position: {obs['ee_pos'][3]}")
                env.render()
            elif key == ord('m'):
                # Move to last clicked point (with corrections)
                if last_clicked_point_base is not None:
                    print(f"Moving to corrected target: {last_clicked_point_base}")
                    
                    # Calculate movement vector (from current position to target)
                    current_pos = obs['ee_pos'][:3]
                    movement_vector = last_clicked_point_base - current_pos
                    
                    # Scale movement for safety
                    movement_vector = movement_vector * pos_sensitivity
                    
                    # Create action (x, y, z, gripper)
                    action = np.zeros(4)
                    action[:3] = movement_vector
                    
                    # Execute action
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Display information
                    print(f"New end-effector position: {obs['ee_pos'][:3]}")
                    env.render()
                else:
                    print("No valid point selected yet")
    
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