import os
import numpy as np
import mujoco
import mujoco.viewer
import time
from gym_lowcostrobot.envs.lift_cube_state_env_twin import LiftCubeStateEnv

def visualize_camera_pose():
    """
    Visualize the camera pose in MuJoCo using the calibration matrix
    from ./results/calibration_matrix.npy
    """
    # Load the calibration matrix
    calibration_path = os.path.join('results', 'calibration_matrix.npy')
    if not os.path.exists(calibration_path):
        print(f"Calibration matrix not found at {calibration_path}")
        return

    T_base_camera = np.load(calibration_path)
    print(f"Loaded calibration matrix:\n{T_base_camera}")

    # Create the environment
    env = LiftCubeStateEnv(
        observation_mode="both",
        render_mode="human", 
        action_mode="nullspace", 
        use_action_noise=False
    )
    
    # Get the MuJoCo model and data
    model = env.model
    data = env.data
    
    # Add a visual marker for the camera pose
    add_camera_marker(model, data, T_base_camera)
    
    # Reset the environment
    obs, info = env.reset()
    print(f"Initial end-effector position: {obs['ee_pos'][:3]}")
    
    # Render the environment
    while True:
        env.render()
        time.sleep(0.1)
        # Check for user input to exit
        key = input("Press Enter to continue, 'q' to quit: ")
        if key.lower() == 'q':
            break
    
    env.close()

def add_camera_marker(model, data, T_base_camera):
    """
    Add a visual marker for the camera pose in the MuJoCo simulation
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        T_base_camera: 4x4 transformation matrix from camera to robot base
    """
    # Get the position and orientation from the transformation matrix
    position = T_base_camera[:3, 3]
    rotation_matrix = T_base_camera[:3, :3]
    
    # Add a site to visualize the camera position
    # Note: This is a simplification. In a real implementation, you would modify the XML
    # to add a site for the camera. Here we're just printing the values.
    print(f"Camera position: {position}")
    print(f"Camera orientation (rotation matrix):\n{rotation_matrix}")
    
    # Draw camera position and orientation in the viewer
    # This will be drawn every frame in the viewer
    def camera_marker_callback(model, data):
        # Draw a sphere at the camera position
        radius = 0.02  # 2cm radius
        # Draw coordinate axes to show orientation
        axis_length = 0.1  # 10cm axes
        
        # Calculate axis endpoints
        x_axis = position + rotation_matrix[:, 0] * axis_length
        y_axis = position + rotation_matrix[:, 1] * axis_length
        z_axis = position + rotation_matrix[:, 2] * axis_length
        
        print("To visualize the camera pose:")
        print(f"  - Position: {position}")
        print(f"  - X axis endpoint: {x_axis} (red)")
        print(f"  - Y axis endpoint: {y_axis} (green)")
        print(f"  - Z axis endpoint: {z_axis} (blue)")
        
        # Note: In a full implementation, you would use mujoco.mjvScene
        # to add these visual elements to the scene
    
    # Call the callback once (in a full implementation, this would be attached to the render loop)
    camera_marker_callback(model, data)
    
    print("\nInstructions to interpret the camera pose:")
    print("1. The camera position is shown as a point in the robot's base frame")
    print("2. The camera orientation is shown as three axes:")
    print("   - X axis (red): points to the right in the camera view")
    print("   - Y axis (green): points down in the camera view")
    print("   - Z axis (blue): points forward from the camera (viewing direction)")
    print("3. The transformation matrix describes how to convert points from camera frame to robot base frame")

def create_camera_xml_snippet(T_base_camera):
    """
    Create XML snippet to add a visual camera model to the MuJoCo simulation
    
    Args:
        T_base_camera: 4x4 transformation matrix from camera to robot base
    
    Returns:
        XML string to add to the MuJoCo model
    """
    position = T_base_camera[:3, 3]
    # Convert rotation matrix to quaternion
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(T_base_camera[:3, :3])
    quat = r.as_quat()  # x, y, z, w format
    # Reorder to w, x, y, z for MuJoCo
    quat_mujoco = np.array([quat[3], quat[0], quat[1], quat[2]])
    
    xml = f"""
    <body name="camera_visual" pos="{position[0]} {position[1]} {position[2]}" quat="{quat_mujoco[0]} {quat_mujoco[1]} {quat_mujoco[2]} {quat_mujoco[3]}">
        <geom type="box" size="0.03 0.02 0.01" rgba="0.1 0.1 0.1 1"/>
        <geom type="cylinder" pos="0 0 0.02" size="0.01 0.01" rgba="0.1 0.1 0.1 1"/>
        <site name="camera_frame_origin" pos="0 0 0" size="0.01"/>
        <site name="camera_frame_x" pos="0.05 0 0" size="0.005" rgba="1 0 0 1"/>
        <site name="camera_frame_y" pos="0 0.05 0" size="0.005" rgba="0 1 0 1"/>
        <site name="camera_frame_z" pos="0 0 0.05" size="0.005" rgba="0 0 1 1"/>
    </body>
    """
    
    print("XML snippet to add to MuJoCo model:")
    print(xml)
    
    return xml

if __name__ == "__main__":
    visualize_camera_pose() 