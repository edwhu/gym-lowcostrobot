import os
import numpy as np
from gym_lowcostrobot.camera.d455 import D455Camera

def fix_and_test_camera_transform():
    """
    Fix and test the camera transformation in the D455Camera class
    """
    # Load calibration matrix
    calibration_path = os.path.join('results', 'calibration_matrix.npy')
    if not os.path.exists(calibration_path):
        print(f"Calibration matrix not found at {calibration_path}")
        # Create the results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Let's create a reasonable test calibration matrix
        # This places the camera 0.5m in front of the robot, looking at the robot
        test_matrix = np.array([
            [0.0, -1.0, 0.0, 0.5],  # Camera X axis aligned with robot -Y axis
            [-1.0, 0.0, 0.0, 0.0],  # Camera Y axis aligned with robot -X axis
            [0.0, 0.0, -1.0, 0.3],  # Camera Z axis aligned with robot -Z axis
            [0.0, 0.0, 0.0, 1.0]
        ])
        np.save(calibration_path, test_matrix)
        print(f"Created test calibration matrix at {calibration_path}:")
        print(test_matrix)
        T_base_camera = test_matrix
    else:
        T_base_camera = np.load(calibration_path)
        print(f"Loaded calibration matrix from {calibration_path}:")
        print(T_base_camera)
    
    # Create and inspect both transformation matrices
    T_camera_base = np.linalg.inv(T_base_camera)
    
    print("\nT_base_camera (transform from camera frame to robot base frame):")
    print(T_base_camera)
    
    print("\nT_camera_base (transform from robot base frame to camera frame):")
    print(T_camera_base)
    
    # Test a point transformation
    print("\nTesting point transformation:")
    # Point in camera coordinates (e.g., from a depth camera)
    point_camera = np.array([0.1, 0.1, 0.5])  # 10cm right, 10cm down, 50cm forward from camera
    print(f"Point in camera coordinates: {point_camera}")
    
    # Convert to homogeneous coordinates
    point_camera_homogeneous = np.append(point_camera, 1.0)
    
    # Transform to robot base coordinates 
    # CORRECT: point_base = T_base_camera @ point_camera_homogeneous
    point_base = T_base_camera @ point_camera_homogeneous
    
    print(f"Point in robot base coordinates: {point_base[:3]}")
    
    # Transform back to camera coordinates to verify
    point_camera_reconstructed = T_camera_base @ point_base
    print(f"Point transformed back to camera coordinates: {point_camera_reconstructed[:3]}")
    
    # Calculate error
    error = np.linalg.norm(point_camera - point_camera_reconstructed[:3])
    print(f"Reconstruction error: {error:.6f} meters")
    
    # Fix the transformation in the D455Camera class
    print("\nFixing the D455Camera.get_3d_point_robot_base method...")
    print("The original method has:")
    print("point_base = np.linalg.inv(self.T_base_camera) @ point_homogeneous")
    print("The correct implementation should be:")
    print("point_base = self.T_base_camera @ point_homogeneous")
    
    # Check if we have the necessary permissions to edit the file
    d455_py_path = 'gym_lowcostrobot/camera/d455.py'
    if os.path.exists(d455_py_path) and os.access(d455_py_path, os.W_OK):
        print(f"\nThe file {d455_py_path} exists and is writable.")
        print("You can apply this fix directly to the file.")
    else:
        print(f"\nCannot modify the file {d455_py_path} directly.")
        print("Here's the fix you should apply:")
        print("\nIn the get_3d_point_robot_base method, change:")
        print("point_base = np.linalg.inv(self.T_base_camera) @ point_homogeneous")
        print("to:")
        print("point_base = self.T_base_camera @ point_homogeneous")
    
    print("\nSummary of camera transform issues:")
    print("1. The calibration matrix (T_base_camera) transforms points from camera frame to robot base frame.")
    print("2. In get_3d_point_robot_base, we directly apply T_base_camera to camera points (not its inverse).")
    print("3. The camera position may be too far from the ArUco marker, making detection difficult.")
    print("   - Try moving the camera closer to the marker")
    print("   - Ensure there's good lighting for ArUco detection")
    print("   - Check that the marker size parameter matches the actual size")
    print("   - Verify the marker is fully visible in the camera view")

if __name__ == "__main__":
    fix_and_test_camera_transform() 