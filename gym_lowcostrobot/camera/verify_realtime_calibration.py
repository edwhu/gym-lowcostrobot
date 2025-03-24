import os
import numpy as np
from omni_depth.scripts.realtime_calibrate import RealtimeCalibrator

def verify_calibration_transform():
    """
    Verify the calibration transformation in the RealtimeCalibrator class
    """
    print("Verifying calibration transformations in RealtimeCalibrator")
    
    # Create the RealtimeCalibrator instance
    calibrator = RealtimeCalibrator()
    
    # Print the T_base_marker matrix
    print("\nT_base_marker (transformation from robot base to ArUco marker):")
    print(calibrator.T_base_marker)
    
    # The actual marker position according to the description is at (-0.01, 0, 0)
    # But in the code it's set to [-0.0243, -0.0, 0.0001]
    print("\nMarker position in code:", calibrator.T_base_marker[:3, 3])
    print("Marker position should be: [-0.01, 0.0, 0.0] (according to your description)")
    
    # Create a simulated T_camera_marker as if detected from ArUco
    # This represents a camera 0.5m in front of the marker, looking at it
    T_camera_marker = np.array([
        [0.0, -1.0, 0.0, 0.0],  # Camera X axis aligned with marker -Y axis
        [-1.0, 0.0, 0.0, 0.0],  # Camera Y axis aligned with marker -X axis
        [0.0, 0.0, -1.0, 0.5],  # Camera Z axis aligned with marker -Z axis, 0.5m away
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    print("\nSimulated T_camera_marker (as if detected by ArUco):")
    print(T_camera_marker)
    
    # Calculate T_base_camera = T_base_marker * inv(T_camera_marker)
    T_base_camera = calibrator.T_base_marker @ np.linalg.inv(T_camera_marker)
    
    print("\nCalculated T_base_camera:")
    print(T_base_camera)
    
    # Verify with some test points
    print("\nVerifying transformation with test points:")
    
    # Test point: 10cm in front of the camera
    point_camera = np.array([0.0, 0.0, 0.1, 1.0])
    point_base = T_base_camera @ point_camera
    
    print(f"Point 10cm in front of camera:")
    print(f"  Camera frame: {point_camera[:3]}")
    print(f"  Base frame:   {point_base[:3]}")
    
    # Test point: origin in marker frame should map to marker position in base frame
    point_marker = np.array([0.0, 0.0, 0.0, 1.0])
    point_camera_from_marker = np.linalg.inv(T_camera_marker) @ point_marker
    point_base_from_marker = T_base_camera @ point_camera_from_marker
    
    print(f"\nMarker origin:")
    print(f"  Marker frame: {point_marker[:3]}")
    print(f"  Camera frame: {point_camera_from_marker[:3]}")
    print(f"  Base frame:   {point_base_from_marker[:3]}")
    print(f"  Expected base frame: {calibrator.T_base_marker[:3, 3]}")
    
    # Save corrected values to file
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Update T_base_marker to match the description
    corrected_T_base_marker = np.array([
        [1, 0, 0, -0.01],  # x translation -1cm as described
        [0, 1, 0, 0.0],    # no y translation
        [0, 0, 1, 0.0],    # no z translation
        [0, 0, 0, 1]
    ])
    
    # Calculate corrected T_base_camera
    corrected_T_base_camera = corrected_T_base_marker @ np.linalg.inv(T_camera_marker)
    
    print("\nCorrected T_base_marker to match description:")
    print(corrected_T_base_marker)
    
    print("\nCorrected T_base_camera with updated marker position:")
    print(corrected_T_base_camera)
    
    # Save corrected calibration matrix
    np.save(os.path.join(results_dir, 'corrected_calibration_matrix.npy'), corrected_T_base_camera)
    print(f"\nSaved corrected calibration matrix to {os.path.join(results_dir, 'corrected_calibration_matrix.npy')}")
    
    print("\nRecommendations for realtime_calibrate.py:")
    print("1. Update T_base_marker to match your actual marker position:")
    print("   Change:\n   [1, 0, 0, -0.0243],  # x translation from doc")
    print("   [0, 1, 0, -0.0],    # y translation from doc")
    print("   [0, 0, 1, 0.0001],  # z translation from doc")
    print("   To:\n   [1, 0, 0, -0.01],  # x translation -1cm")
    print("   [0, 1, 0, 0.0],     # no y translation")
    print("   [0, 0, 1, 0.0],     # no z translation")
    print("2. To improve ArUco marker detection:")
    print("   - Ensure the camera is close enough to the marker (ideally 20-50cm)")
    print("   - Make sure the marker is well-lit with even lighting")
    print("   - Confirm the marker size parameter matches the actual size of your marker")
    print("   - Try increasing the marker size or using a different ArUco dictionary")

if __name__ == "__main__":
    verify_calibration_transform() 