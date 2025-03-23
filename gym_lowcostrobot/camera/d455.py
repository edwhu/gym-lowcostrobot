import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.envs.registration')

import numpy as np
import cv2
import pyrealsense2 as rs
import time
import os
import json
from threading import Thread
from typing import Tuple, Optional, Dict, List, Union


class D455Camera:
    """
    Driver for Intel RealSense D455 camera.
    Provides RGB and depth streaming with point cloud generation.
    
    Features:
    - RGB and depth video streaming
    - Point cloud generation
    - Frame capture and saving
    - Calibration support
    """
    
    def __init__(
        self,
        enable_rgb: bool = True,
        enable_depth: bool = True,
        rgb_resolution: Tuple[int, int] = (848, 480),
        depth_resolution: Tuple[int, int] = (848, 480),
        fps: int = 30,
        align_frames: bool = True,
        device_id: Optional[str] = None,
        output_dir: str = 'camera_output'
    ):
        """
        Initialize D455 camera driver.
        
        Args:
            enable_rgb: Enable RGB stream
            enable_depth: Enable depth stream
            rgb_resolution: Resolution for RGB stream (width, height)
            depth_resolution: Resolution for depth stream (width, height)
            fps: Frames per second
            align_frames: Align depth frames to RGB frames
            device_id: Specific device ID to use (None for any available)
            output_dir: Directory to save captured frames and data
        """
        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.rgb_resolution = rgb_resolution
        self.depth_resolution = depth_resolution
        self.fps = fps
        self.align_frames = align_frames
        self.device_id = device_id
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize camera state
        self.pipeline = None
        self.config = None
        self.align = None
        self.profile = None
        self.is_running = False
        self.is_recording = False
        self.recording_thread = None
        self.frame_count = 0
        
        # Calibration data
        self.T_base_camera = np.eye(4)  # Identity matrix as default
        
        # Initialize streaming
        self.initialize_streaming()
        
    def initialize_streaming(self):
        """Initialize the RealSense pipeline and configuration."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Try to get the device
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices connected")
        
        # Select device
        selected_device = None
        if self.device_id:
            for device in devices:
                if device.get_info(rs.camera_info.serial_number) == self.device_id:
                    selected_device = device
                    break
            if not selected_device:
                raise ValueError(f"Device with ID {self.device_id} not found")
        else:
            selected_device = devices[0]
            self.device_id = selected_device.get_info(rs.camera_info.serial_number)
            print(f"Using RealSense device: {self.device_id}")
        
        # Enable the device
        self.config.enable_device(self.device_id)
        
        # Configure streams
        if self.enable_rgb:
            self.config.enable_stream(
                rs.stream.color,
                self.rgb_resolution[0],
                self.rgb_resolution[1],
                rs.format.bgr8,
                self.fps
            )
        
        if self.enable_depth:
            self.config.enable_stream(
                rs.stream.depth,
                self.depth_resolution[0],
                self.depth_resolution[1],
                rs.format.z16,
                self.fps
            )
            
        # Create alignment object
        if self.align_frames and self.enable_rgb and self.enable_depth:
            self.align = rs.align(rs.stream.color)
    
    def start(self):
        """Start the camera streaming."""
        if not self.is_running:
            try:
                # Start streaming
                self.profile = self.pipeline.start(self.config)
                
                # Wait for auto-exposure to stabilize
                time.sleep(1.0)
                
                self.is_running = True
                print("Camera streaming started")
                
                # Get depth sensor if available
                if self.enable_depth:
                    depth_sensor = self.profile.get_device().first_depth_sensor()
                    # Set depth units to millimeters (1mm)
                    depth_sensor.set_option(rs.option.depth_units, 0.001)
                
                return True
            except Exception as e:
                print(f"Error starting camera: {e}")
                return False
        else:
            print("Camera is already running")
            return True
    
    def stop(self):
        """Stop the camera streaming."""
        if self.is_running:
            if self.is_recording:
                self.stop_recording()
            
            self.pipeline.stop()
            self.is_running = False
            print("Camera stopped")
            return True
        else:
            print("Camera is not running")
            return False
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the current RGB and depth frames.
        
        Returns:
            Tuple containing (rgb_frame, depth_frame) as numpy arrays
        """
        if not self.is_running:
            print("Camera is not running")
            return None, None
        
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Align frames if enabled
            if self.align_frames and self.enable_rgb and self.enable_depth:
                frames = self.align.process(frames)
            
            # Extract color and depth frames
            rgb_frame = None
            depth_frame = None
            
            if self.enable_rgb:
                color_frame = frames.get_color_frame()
                if color_frame:
                    rgb_frame = np.asanyarray(color_frame.get_data())
            
            if self.enable_depth:
                depth_frame_raw = frames.get_depth_frame()
                if depth_frame_raw:
                    depth_frame = np.asanyarray(depth_frame_raw.get_data())
            
            return rgb_frame, depth_frame
        
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None
    
    def get_point_cloud(self, rgb_frame: np.ndarray, depth_frame: np.ndarray) -> np.ndarray:
        """
        Generate a point cloud from RGB and depth frames.
        
        Args:
            rgb_frame: RGB frame as numpy array
            depth_frame: Depth frame as numpy array
            
        Returns:
            Nx6 numpy array with [x, y, z, r, g, b] points
        """
        if not (self.is_running and self.enable_depth):
            print("Camera is not running or depth is not enabled")
            return None
        
        try:
            # Get depth intrinsics
            depth_profile = self.profile.get_stream(rs.stream.depth)
            intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            
            # Create empty point cloud
            height, width = depth_frame.shape
            points = np.zeros((height * width, 6), dtype=np.float32)
            
            # Generate point cloud
            pixel_idx = 0
            for y in range(height):
                for x in range(width):
                    depth_value = depth_frame[y, x]
                    
                    # Skip invalid depth values
                    if depth_value == 0:
                        continue
                    
                    # Deproject from pixel to 3D point
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_value)
                    
                    # Add point to point cloud
                    points[pixel_idx, 0:3] = point
                    
                    # Add RGB color if available
                    if rgb_frame is not None:
                        points[pixel_idx, 3:6] = rgb_frame[y, x] / 255.0
                    
                    pixel_idx += 1
            
            # Only keep valid points
            return points[:pixel_idx]
            
        except Exception as e:
            print(f"Error generating point cloud: {e}")
            return None
    
    def get_intrinsics(self) -> Dict:
        """
        Get camera intrinsic parameters.
        
        Returns:
            Dictionary containing intrinsic parameters
        """
        if not self.is_running:
            print("Camera is not running")
            return None
        
        try:
            intrinsics_dict = {}
            
            if self.enable_rgb:
                color_profile = self.profile.get_stream(rs.stream.color)
                color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
                
                intrinsics_dict['rgb'] = {
                    'width': color_intrinsics.width,
                    'height': color_intrinsics.height,
                    'fx': color_intrinsics.fx,
                    'fy': color_intrinsics.fy,
                    'cx': color_intrinsics.ppx,
                    'cy': color_intrinsics.ppy,
                    'distortion_model': str(color_intrinsics.model),
                    'coeffs': list(color_intrinsics.coeffs)
                }
            
            if self.enable_depth:
                depth_profile = self.profile.get_stream(rs.stream.depth)
                depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
                
                intrinsics_dict['depth'] = {
                    'width': depth_intrinsics.width,
                    'height': depth_intrinsics.height,
                    'fx': depth_intrinsics.fx,
                    'fy': depth_intrinsics.fy,
                    'cx': depth_intrinsics.ppx,
                    'cy': depth_intrinsics.ppy,
                    'distortion_model': str(depth_intrinsics.model),
                    'coeffs': list(depth_intrinsics.coeffs)
                }
            
            return intrinsics_dict
            
        except Exception as e:
            print(f"Error getting intrinsics: {e}")
            return None
    
    def capture_frame(self, save: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture a single frame and optionally save it.
        
        Args:
            save: Whether to save the frame to disk
            
        Returns:
            Tuple containing (rgb_frame, depth_frame)
        """
        rgb_frame, depth_frame = self.get_frames()
        
        if save and (rgb_frame is not None or depth_frame is not None):
            timestamp = int(time.time())
            
            if rgb_frame is not None:
                rgb_path = os.path.join(self.output_dir, f"rgb_{timestamp}.png")
                cv2.imwrite(rgb_path, rgb_frame)
                print(f"Saved RGB frame to {rgb_path}")
            
            if depth_frame is not None:
                # Normalize depth for visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_frame, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                depth_path = os.path.join(self.output_dir, f"depth_{timestamp}.png")
                cv2.imwrite(depth_path, depth_colormap)
                
                # Also save raw depth data
                depth_raw_path = os.path.join(self.output_dir, f"depth_{timestamp}.npy")
                np.save(depth_raw_path, depth_frame)
                print(f"Saved depth frame to {depth_path} and {depth_raw_path}")
        
        return rgb_frame, depth_frame
    
    def _record_frames(self, duration: float, interval: float):
        """
        Background thread function for recording frames.
        
        Args:
            duration: Duration to record in seconds (0 for indefinite)
            interval: Interval between frames in seconds
        """
        start_time = time.time()
        self.frame_count = 0
        
        print(f"Recording started. Duration: {duration if duration > 0 else 'indefinite'} seconds")
        
        while self.is_recording:
            # Check if duration has elapsed
            if duration > 0 and (time.time() - start_time) >= duration:
                break
            
            # Capture frame
            rgb_frame, depth_frame = self.capture_frame(save=True)
            
            if rgb_frame is not None or depth_frame is not None:
                self.frame_count += 1
            
            # Wait for next frame
            time.sleep(interval)
        
        self.is_recording = False
        print(f"Recording stopped. Captured {self.frame_count} frames")
    
    def start_recording(self, duration: float = 0, interval: float = 0.1):
        """
        Start recording frames at regular intervals.
        
        Args:
            duration: Duration to record in seconds (0 for indefinite)
            interval: Interval between frames in seconds
            
        Returns:
            True if recording started, False otherwise
        """
        if not self.is_running:
            print("Camera is not running")
            return False
        
        if self.is_recording:
            print("Already recording")
            return False
        
        self.is_recording = True
        self.recording_thread = Thread(
            target=self._record_frames,
            args=(duration, interval),
            daemon=True
        )
        self.recording_thread.start()
        return True
    
    def stop_recording(self):
        """Stop the current recording."""
        if self.is_recording:
            self.is_recording = False
            if self.recording_thread:
                self.recording_thread.join(timeout=2.0)
            print("Recording stopped")
            return True
        else:
            print("Not currently recording")
            return False
    
    def get_depth_at_point(self, x: int, y: int) -> float:
        """
        Get depth value at a specific pixel coordinate.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Depth value in meters (or None if invalid)
        """
        if not (self.is_running and self.enable_depth):
            print("Camera is not running or depth is not enabled")
            return None
        
        _, depth_frame = self.get_frames()
        
        if depth_frame is None:
            return None
        
        height, width = depth_frame.shape
        
        if x < 0 or x >= width or y < 0 or y >= height:
            print(f"Point ({x}, {y}) is outside of depth frame bounds ({width}x{height})")
            return None
        
        depth_value = depth_frame[y, x]
        
        # Convert depth to meters (assuming it's in millimeters)
        if depth_value == 0:
            return None  # Invalid depth
        
        return depth_value / 1000.0  # Convert to meters
    
    def get_3d_point(self, x: int, y: int) -> np.ndarray:
        """
        Get 3D point at a specific pixel coordinate.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            3D point in camera coordinate system [x, y, z] in meters
        """
        if not (self.is_running and self.enable_depth):
            print("Camera is not running or depth is not enabled")
            return None
        
        _, depth_frame = self.get_frames()
        
        if depth_frame is None:
            return None
        
        height, width = depth_frame.shape
        
        if x < 0 or x >= width or y < 0 or y >= height:
            print(f"Point ({x}, {y}) is outside of depth frame bounds ({width}x{height})")
            return None
        
        depth_value = depth_frame[y, x]
        
        if depth_value == 0:
            return None  # Invalid depth
        
        # Get depth intrinsics
        depth_profile = self.profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        
        # Deproject from pixel to 3D point
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_value)
        
        # Convert to meters
        return np.array(point) / 1000.0
    
    def get_3d_point_robot_base(self, x: int, y: int) -> np.ndarray:
        """
        Get 3D point at pixel coordinate in robot base frame.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            3D point in robot base coordinate system [x, y, z] in meters
        """
        # Get point in camera coordinates
        point_camera = self.get_3d_point(x, y)
        
        if point_camera is None:
            return None
        
        # Convert to homogeneous coordinates
        point_homogeneous = np.append(point_camera, 1.0)
        
        # Transform to robot base coordinates
        point_base = self.T_base_camera @ point_homogeneous
        
        # Return just the 3D coordinates
        return point_base[:3]
    
    def set_calibration_matrix(self, T_base_camera: np.ndarray):
        """
        Set the calibration matrix for transforming from camera to robot base.
        
        Args:
            T_base_camera: 4x4 transformation matrix
        """
        if T_base_camera.shape != (4, 4):
            raise ValueError("Calibration matrix must be 4x4")
        
        self.T_base_camera = T_base_camera.copy()
        
        # Save calibration matrix
        np.save(os.path.join(self.output_dir, 'calibration_matrix.npy'), self.T_base_camera)
        print(f"Calibration matrix saved to {self.output_dir}/calibration_matrix.npy")
    
    def load_calibration_matrix(self, file_path: Optional[str] = None):
        """
        Load calibration matrix from file.
        
        Args:
            file_path: Path to calibration matrix file (.npy)
                       If None, tries to load from output_dir
        """
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'calibration_matrix.npy')
        
        if not os.path.exists(file_path):
            print(f"Calibration file not found: {file_path}")
            return False
        
        try:
            self.T_base_camera = np.load(file_path)
            print(f"Loaded calibration matrix from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading calibration matrix: {e}")
            return False
    
    def calibrate_with_points(self, points_camera: List[np.ndarray], points_robot: List[np.ndarray]):
        """
        Calibrate camera to robot base using corresponding points.
        
        Args:
            points_camera: List of points in camera coordinates
            points_robot: List of points in robot base coordinates
            
        Returns:
            Calibration error (RMSE)
        """
        if len(points_camera) != len(points_robot):
            raise ValueError("Number of camera and robot points must be the same")
        
        if len(points_camera) < 3:
            raise ValueError("At least 3 points are needed for calibration")
        
        # Convert to numpy arrays
        points_camera = np.array(points_camera)
        points_robot = np.array(points_robot)
        
        # Calculate rotation and translation using Kabsch algorithm
        centroid_camera = np.mean(points_camera, axis=0)
        centroid_robot = np.mean(points_robot, axis=0)
        
        # Center the points
        points_camera_centered = points_camera - centroid_camera
        points_robot_centered = points_robot - centroid_robot
        
        # Calculate covariance matrix
        H = points_camera_centered.T @ points_robot_centered
        
        # Singular value decomposition
        U, _, Vt = np.linalg.svd(H)
        
        # Calculate rotation matrix
        R = Vt.T @ U.T
        
        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation
        t = centroid_robot - R @ centroid_camera
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        # Set calibration matrix
        self.set_calibration_matrix(T)
        
        # Calculate error
        errors = []
        for i in range(len(points_camera)):
            point_camera_homogeneous = np.append(points_camera[i], 1.0)
            point_robot_estimated = T @ point_camera_homogeneous
            error = np.linalg.norm(point_robot_estimated[:3] - points_robot[i])
            errors.append(error)
        
        rmse = np.sqrt(np.mean(np.square(errors)))
        print(f"Calibration RMSE: {rmse:.6f} meters")
        
        return rmse

    def visualize_point_cloud(self, rgb_frame: np.ndarray, depth_frame: np.ndarray):
        """
        Visualize the point cloud using Open3D (if available).
        This method requires Open3D to be installed.
        
        Args:
            rgb_frame: RGB frame
            depth_frame: Depth frame
        """
        try:
            import open3d as o3d
            
            # Generate point cloud
            points = self.get_point_cloud(rgb_frame, depth_frame)
            
            if points is None or len(points) == 0:
                print("No valid points in point cloud")
                return
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
            
            # Visualize
            o3d.visualization.draw_geometries([pcd])
            
        except ImportError:
            print("Open3D not available. Install with: pip install open3d")


def test_camera():
    """Test the D455 camera functionality."""
    print("\n=== Testing RealSense D455 Camera ===")
    
    try:
        # Create camera object
        print("\n1. Initializing camera...")
        camera = D455Camera(
            enable_rgb=True,
            enable_depth=True,
            rgb_resolution=(848, 480),
            depth_resolution=(848, 480),
            fps=30,
            align_frames=True
        )
        
        # Start camera
        print("\n2. Starting camera stream...")
        if not camera.start():
            print("Failed to start camera")
            return
        
        # Get and display camera info
        print("\n3. Getting camera information...")
        intrinsics = camera.get_intrinsics()
        if intrinsics:
            print("\nCamera intrinsics:")
            for stream_name, params in intrinsics.items():
                print(f"\n{stream_name.upper()} Stream:")
                for param_name, value in params.items():
                    print(f"  {param_name}: {value}")
        
        # Create display window
        print("\n4. Starting display loop...")
        print("   Press 'C' to capture frame")
        print("   Press 'R' to start/stop recording")
        print("   Press 'ESC' to exit")
        
        cv2.namedWindow("RGB-D Stream", cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Get frames
            rgb_frame, depth_frame = camera.get_frames()
            
            if rgb_frame is None or depth_frame is None:
                print("Failed to get frames")
                time.sleep(0.1)
                continue
            
            # Update FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                print(f"\rFPS: {fps:.1f}", end="")
                frame_count = 0
                start_time = time.time()
            
            # Convert depth to color map for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # Add depth range text
            min_depth = depth_frame[depth_frame > 0].min() if depth_frame.size > 0 else 0
            max_depth = depth_frame.max()
            cv2.putText(depth_colormap, 
                       f"Depth range: {min_depth/1000.0:.2f}m - {max_depth/1000.0:.2f}m",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Combine images side by side
            combined = np.hstack((rgb_frame, depth_colormap))
            
            # Show images
            cv2.imshow("RGB-D Stream", combined)
            
            # Check for key press
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
            elif key == ord('c'):
                # Capture frame
                print("\nCapturing frame...")
                camera.capture_frame(save=True)
            elif key == ord('r'):
                # Start/stop recording
                if camera.is_recording:
                    print("\nStopping recording...")
                    camera.stop_recording()
                else:
                    print("\nStarting recording (10 seconds)...")
                    camera.start_recording(duration=10, interval=0.1)
    
    except Exception as e:
        print(f"\nError during camera test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        print("\n5. Cleaning up...")
        if 'camera' in locals():
            camera.stop()
        cv2.destroyAllWindows()
        print("\nCamera test completed.")


if __name__ == "__main__":
    test_camera()
