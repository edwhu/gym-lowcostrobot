import os
import numpy as np

def test_calibration_matrix():
    """
    Test the calibration matrix by transforming points between 
    robot base frame and camera frame
    """
    # Load the calibration matrix
    calibration_path = os.path.join('results', 'calibration_matrix.npy')
    if not os.path.exists(calibration_path):
        print(f"Calibration matrix not found at {calibration_path}")
        return

    T_base_camera = np.load(calibration_path)
    print(f"Loaded calibration matrix:\n{T_base_camera}")
    
    # Calculate inverse transformation (camera to base)
    T_camera_base = np.linalg.inv(T_base_camera)
    
    # Test points in robot base frame
    test_points_base = [
        np.array([0.0, 0.0, 0.0, 1.0]),  # Origin
        np.array([0.1, 0.0, 0.0, 1.0]),  # 10cm along X axis
        np.array([0.0, 0.1, 0.0, 1.0]),  # 10cm along Y axis
        np.array([0.0, 0.0, 0.1, 1.0]),  # 10cm along Z axis
        np.array([0.1, 0.1, 0.1, 1.0])   # 10cm in all directions
    ]
    
    print("\nTransforming points from robot base frame to camera frame and back:")
    print("-------------------------------------------------------------------")
    for i, point_base in enumerate(test_points_base):
        # Transform from base to camera
        point_camera = T_camera_base @ point_base
        
        # Transform back to base
        point_base_reconstructed = T_base_camera @ point_camera
        
        # Calculate error
        error = np.linalg.norm(point_base[:3] - point_base_reconstructed[:3])
        
        print(f"\nTest Point {i+1}:")
        print(f"  Base frame:              {point_base[:3]}")
        print(f"  Camera frame:            {point_camera[:3]}")
        print(f"  Reconstructed base frame: {point_base_reconstructed[:3]}")
        print(f"  Error:                   {error:.6f} meters")
    
    # Test a specific point that corresponds to the ArUco marker position
    marker_pos_base = np.array([-0.01, 0, 0, 1.0])  # As per your description
    marker_pos_camera = T_camera_base @ marker_pos_base
    
    print("\nArUco Marker Position:")
    print(f"  Base frame:   {marker_pos_base[:3]}")
    print(f"  Camera frame: {marker_pos_camera[:3]}")
    
    # Check if the camera can see the marker
    # Assuming Z axis is the viewing direction in camera frame
    if marker_pos_camera[2] > 0:
        print("  The marker is in front of the camera (positive Z in camera frame)")
    else:
        print("  The marker is behind the camera (negative Z in camera frame)")
        print("  This suggests an issue with the calibration or marker placement")
    
    # Estimate distance from camera to marker
    distance = np.linalg.norm(marker_pos_camera[:3])
    print(f"  Distance from camera to marker: {distance:.3f} meters")
    
    # Check if T_base_camera makes sense for a typical camera setup
    # Camera should be outside the robot's workspace and looking at it
    camera_pos_base = T_base_camera[:3, 3]
    print("\nCamera Position in Robot Base Frame:")
    print(f"  Position: {camera_pos_base}")
    
    distance_to_origin = np.linalg.norm(camera_pos_base)
    print(f"  Distance from robot base origin: {distance_to_origin:.3f} meters")
    
    # Check camera orientation
    z_axis = T_base_camera[:3, 2]  # 3rd column is the Z axis
    print(f"  Camera Z axis (viewing direction) in base frame: {z_axis}")
    
    # The Z axis should point somewhat toward the origin for a typical setup
    dot_product = np.dot(-z_axis, camera_pos_base / np.linalg.norm(camera_pos_base))
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
    print(f"  Angle between camera-to-origin vector and viewing direction: {angle:.1f} degrees")
    if angle < 45:
        print("  Camera is approximately pointed toward the robot (good)")
    else:
        print("  Camera may not be pointed toward the robot (check calibration)")

if __name__ == "__main__":
    test_calibration_matrix() 