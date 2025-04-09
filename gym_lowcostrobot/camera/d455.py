import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.envs.registration')

import numpy as np
import cv2
import pyrealsense2 as rs
import time
import os
import json
from threading import Thread
from typing import Tuple, Optional, Dict, List, Union, Any


class D455Camera:
    """
    Driver for Intel RealSense D455 camera.
    Provides RGB and depth streaming with point cloud generation.
    
    Features:
    - RGB and depth video streaming
    - Point cloud generation
    - Frame capture and saving
    - Calibration support
    - Post-processing filters for improved depth quality
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
        output_dir: str = 'camera_output',
        enable_filters: bool = True,
        filter_config: Optional[Dict[str, Any]] = None
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
            enable_filters: Enable depth post-processing filters
            filter_config: Configuration dictionary for depth filters
        """
        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.rgb_resolution = rgb_resolution
        self.depth_resolution = depth_resolution
        self.fps = fps
        self.align_frames = align_frames
        self.device_id = device_id
        self.output_dir = output_dir
        self.video_dir = os.path.join('realsense', 'videos')
        self.enable_filters = enable_filters
        
        # Default filter configuration
        self.default_filter_config = {
            'decimation': {
                'enable': True,
                'magnitude': 2  # Decimation factor (1=no effect, 2=half res, etc.)
            },
            'spatial': {
                'enable': True,
                'magnitude': 2,  # Filter magnitude (1-5)
                'smooth_alpha': 0.5,  # Alpha value for smoothing (0-1)
                'smooth_delta': 20,  # Delta value for edge preservation
                'hole_fill': 1  # Hole filling mode (0-5)
            },
            'temporal': {
                'enable': True,
                'smooth_alpha': 0.4,  # Alpha value for smoothing
                'smooth_delta': 20,  # Delta value for edge preservation
                'persistence_control': 3  # Persistence (0-8)
            },
            'hole_filling': {
                'enable': True,
                'mode': 1  # Hole filling mode (0-2)
            },
            'threshold': {
                'enable': False,
                'min_dist': 0.1,  # Minimum distance in meters
                'max_dist': 4.0  # Maximum distance in meters
            },
            'disparity': {
                'enable': True
            }
        }
        
        # Apply user-provided filter config on top of defaults
        # TODO: let xingfang review this
        if self.enable_filters:
            self.filter_config = self.default_filter_config.copy()
        
        if filter_config:
                # Update config with provided settings
            for filter_name, settings in filter_config.items():
                if filter_name in self.filter_config:
                    self.filter_config[filter_name].update(settings)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        
        # Initialize camera state
        self.pipeline = None
        self.config = None
        self.align = None
        self.profile = None
        self.is_running = False
        self.is_recording = False
        self.recording_thread = None
        self.frame_count = 0
        
        # Initialize filters
        self.filters = {}
        
        # Calibration data
        self.T_base_camera = np.eye(4)  
        
        # Video buffer
        self.frame_buffer = []
        self.is_buffering = False
        self.max_buffer_size = 1000  # Maximum frames to store in buffer
        
        self.initialize_streaming()
        
    def initialize_streaming(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices connected")
        
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
        
        self.config.enable_device(self.device_id)
        
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
            
        if self.align_frames and self.enable_rgb and self.enable_depth:
            self.align = rs.align(rs.stream.color)
            
        # Initialize post-processing filters
        if self.enable_filters and self.enable_depth:
            self._initialize_filters()
    # TODO double check with doc later
    def _initialize_filters(self):
        """Initialize RealSense post-processing filters"""
        # Decimation filter (reduces resolution)
        if self.filter_config['decimation']['enable']:
            decimate = rs.decimation_filter()
            decimate.set_option(rs.option.filter_magnitude, 
                                self.filter_config['decimation']['magnitude'])
            self.filters['decimation'] = decimate
        
        # Spatial filter (smooths and fills small holes)
        if self.filter_config['spatial']['enable']:
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, 
                              self.filter_config['spatial']['magnitude'])
            spatial.set_option(rs.option.filter_smooth_alpha, 
                              self.filter_config['spatial']['smooth_alpha'])
            spatial.set_option(rs.option.filter_smooth_delta, 
                              self.filter_config['spatial']['smooth_delta'])
            spatial.set_option(rs.option.holes_fill, 
                              self.filter_config['spatial']['hole_fill'])
            self.filters['spatial'] = spatial
        
        # Temporal filter (reduces temporal noise)
        if self.filter_config['temporal']['enable']:
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, 
                               self.filter_config['temporal']['smooth_alpha'])
            temporal.set_option(rs.option.filter_smooth_delta, 
                               self.filter_config['temporal']['smooth_delta'])
            temporal.set_option(rs.option.holes_fill, 
                               self.filter_config['temporal']['persistence_control'])
            self.filters['temporal'] = temporal
        
        # Hole filling filter
        if self.filter_config['hole_filling']['enable']:
            hole_filling = rs.hole_filling_filter()
            hole_filling.set_option(rs.option.holes_fill, 
                                   self.filter_config['hole_filling']['mode'])
            self.filters['hole_filling'] = hole_filling
        
        # Threshold filter (removes points outside distance range)
        if self.filter_config['threshold']['enable']:
            threshold = rs.threshold_filter()
            threshold.set_option(rs.option.min_distance, 
                                self.filter_config['threshold']['min_dist'])
            threshold.set_option(rs.option.max_distance, 
                                self.filter_config['threshold']['max_dist'])
            self.filters['threshold'] = threshold
        
        # Disparity transform (better preserves edges)
        if self.filter_config['disparity']['enable']:
            self.filters['disparity_to_depth'] = rs.disparity_transform(False)
            self.filters['depth_to_disparity'] = rs.disparity_transform(True)
            
        print(f"Initialized {len(self.filters)} depth post-processing filters")
    
    def start(self):
        if not self.is_running:
            try:
                self.profile = self.pipeline.start(self.config)
                # Wait for auto-exposure to stabilize
                time.sleep(1.0)
                
                self.is_running = True
                print("Camera streaming started")
                
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
    
    def _apply_filters(self, depth_frame):
        """Apply post-processing filters to depth frame"""
        if not self.enable_filters or not self.filters:
            return depth_frame
        
        filtered_frame = depth_frame
        
        # Convert to disparity for better filtering (if enabled)
        if 'depth_to_disparity' in self.filters:
            filtered_frame = self.filters['depth_to_disparity'].process(filtered_frame)
        
        # Decimate (reduce resolution, smooth)
        if 'decimation' in self.filters:
            filtered_frame = self.filters['decimation'].process(filtered_frame)
        
        # Apply threshold filter to remove distant points
        if 'threshold' in self.filters:
            filtered_frame = self.filters['threshold'].process(filtered_frame)
        
        # Apply spatial filter (edge-preserving)
        if 'spatial' in self.filters:
            filtered_frame = self.filters['spatial'].process(filtered_frame)
        
        # Apply temporal filter (smooth over time)
        if 'temporal' in self.filters:
            filtered_frame = self.filters['temporal'].process(filtered_frame)
        
        # Convert back from disparity to depth (if needed)
        if 'disparity_to_depth' in self.filters:
            filtered_frame = self.filters['disparity_to_depth'].process(filtered_frame)
        
        # Apply hole filling after filters
        if 'hole_filling' in self.filters:
            filtered_frame = self.filters['hole_filling'].process(filtered_frame)
        
        return filtered_frame
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the current RGB and depth frames with optional post-processing.
        
        Returns:
            Tuple containing (rgb_frame, depth_frame) as numpy arrays
        """
        if not self.is_running:
            print("Camera is not running")
            return None, None
        
        try:
            frames = self.pipeline.wait_for_frames()
            
            if self.align_frames and self.enable_rgb and self.enable_depth:
                frames = self.align.process(frames)
            
            rgb_frame = None
            depth_frame = None
            
            if self.enable_rgb:
                color_frame = frames.get_color_frame()
                if color_frame:
                    rgb_frame = np.asanyarray(color_frame.get_data())
            
            if self.enable_depth:
                depth_frame_raw = frames.get_depth_frame()
                
                if depth_frame_raw:
                    # Apply post-processing filters if enabled
                    if self.enable_filters and self.filters:
                        depth_frame_filtered = self._apply_filters(depth_frame_raw)
                        depth_frame_filtered.keep()  # important for memory management
                        depth_frame = np.asanyarray(depth_frame_filtered.get_data())
                    else:
                        depth_frame_raw.keep()  # important if you are storing depth_frame in a list later.
                        depth_frame = np.asanyarray(depth_frame_raw.get_data())
            
            return rgb_frame, depth_frame
        
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None
    
    def get_point_cloud(self, rgb_frame: np.ndarray, depth_frame: np.ndarray) -> np.ndarray:
        """         
        Returns:  Nx6 numpy array with [x, y, z, r, g, b] points
        """
        if not (self.is_running and self.enable_depth):
            print("Camera is not running or depth is not enabled")
            return None
        
        try:
            depth_profile = self.profile.get_stream(rs.stream.depth)
            intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            
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
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(rgb_path, bgr_frame)
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
        
        # Reset frame buffer when starting a new recording
        self.frame_buffer = []
        self.is_buffering = True
        
        self.is_recording = True
        self.recording_thread = Thread(
            target=self._record_frames,
            args=(duration, interval),
            daemon=True
        )
        self.recording_thread.start()
        return True
    
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.recording_thread:
                self.recording_thread.join(timeout=2.0)
            
            # Save buffered video after recording stops
            if self.is_buffering and len(self.frame_buffer) > 0:
                self.save_buffered_video()
                self.is_buffering = False
            
            print("Recording stopped")
            return True
        else:
            print("Not currently recording")
            return False
    
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
            rgb_frame, depth_frame = self.get_frames()
            
            if rgb_frame is not None:
                self.frame_count += 1
                # Add RGB frame to buffer if buffering is enabled
                if self.is_buffering and len(self.frame_buffer) < self.max_buffer_size:
                    self.frame_buffer.append(rgb_frame.copy())
            
            # Save individual frames if needed
            self.capture_frame(save=True)
            
            # Wait for next frame
            time.sleep(interval)
        
        self.is_recording = False
        print(f"Recording stopped. Captured {self.frame_count} frames")
    
    def add_to_buffer(self, frame):
        """
        Add a frame to the video buffer.
        
        Args:
            frame: RGB frame to add to buffer
        """
        if self.is_buffering and len(self.frame_buffer) < self.max_buffer_size:
            self.frame_buffer.append(frame.copy())
    
    def clear_buffer(self):
        """Clear the video buffer."""
        self.frame_buffer = []
    
    def save_buffered_video(self, filename=None):
        """
        Save the buffered frames as a video file.
        
        Args:
            filename: Optional filename (without extension)
                     If None, a timestamp-based filename will be used
        
        Returns:
            Path to the saved video file or None if failed
        """
        if not self.frame_buffer or len(self.frame_buffer) == 0:
            print("No frames in buffer to save")
            return None
        
        try:
            # Create timestamp-based filename if not provided
            if filename is None:
                timestamp = int(time.time())
                filename = f"rgb_video_{timestamp}"
            
            # Ensure video directory exists
            os.makedirs(self.video_dir, exist_ok=True)
            
            # Create full file path
            video_path = os.path.join(self.video_dir, f"{filename}.mp4")
            
            # Get frame properties
            height, width = self.frame_buffer[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                video_path, fourcc, self.fps, (width, height)
            )
            
            # Write all frames
            for frame in self.frame_buffer:
                video_writer.write(frame)
            
            # Release the writer
            video_writer.release()
            
            print(f"Saved {len(self.frame_buffer)} frames to video: {video_path}")
            return video_path
            
        except Exception as e:
            print(f"Error saving video: {e}")
            return None
        finally:
            # Clear buffer after saving
            self.clear_buffer()
    
    def get_depth_at_point(self, x: int, y: int) -> float:
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
        
        depth_value = depth_frame[y, x] /1000.0
        
        # Convert depth to meters 
        if depth_value == 0:
            return None  # Invalid depth
        
        return depth_value    # Convert to meters
    
    def get_3d_point(self, x: int, y: int, depth_frame: np.ndarray, thickness: float = 0) -> np.ndarray:
        if not (self.is_running and self.enable_depth):
            print("Camera is not running or depth is not enabled")
            return None
        
        # _, depth_frame = self.get_frames()
        
        # if depth_frame is None:
        #     return None
        
        height, width = depth_frame.shape
        
        if x < 0 or x >= width or y < 0 or y >= height:
            print(f"Point ({x}, {y}) is outside of depth frame bounds ({width}x{height})")
            return None
        
        depth_value = depth_frame[y, x] + thickness # NOTE: this is a hack to get the depth value of the pixel of the candy
        
        if depth_value == 0:
            return None  # Invalid depth
        
        # Get depth intrinsics
        depth_profile = self.profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        
        # Deproject from pixel to 3D point
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_value)
        
        # Convert to meters
        return np.array(point) / 1000.0
    
    def get_3d_point_robot_base(self, x: int, y: int, depth_frame: np.ndarray, thickness: float = 0) -> np.ndarray:
        """
        Get 3D point at pixel coordinate in robot base frame.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            3D point in robot base coordinate system [x, y, z] in meters
        """
        # Get point in camera coordinates
        point_camera = self.get_3d_point(x, y, depth_frame, thickness)
        
        if point_camera is None:
            return None
        
        # Convert to homogeneous coordinates
        point_homogeneous = np.append(point_camera, 1.0)
        
        # Transform to robot base coordinates
        point_base = self.T_base_camera @ point_homogeneous # TODO may be an error!
        
        # Return just the 3D coordinates
        return point_base[:3]
    
    def set_calibration_matrix(self, T_base_camera: np.ndarray):
        """
        Set extristics for transforming from camera to robot base.
        """
        if T_base_camera.shape != (4, 4):
            raise ValueError("Calibration matrix must be 4x4")
        
        self.T_base_camera = T_base_camera.copy()
        if np.array_equal(T_base_camera, np.eye(4)):
            print("Error: set to identity")
        else:
            print("Regsited extrinsic ")

    def load_calibration_matrix(self, file_path: Optional[str] = None):
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
    
    def update_filter_parameter(self, filter_name: str, option_name: str, value: float) -> bool:
        """
        Update a parameter for a specific filter at runtime.
        
        Args:
            filter_name: Name of the filter ('spatial', 'temporal', etc.)
            option_name: Name of the option to update
            value: New value for the option
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_filters or not self.filters:
            print("Filters are not enabled")
            return False
            
        if filter_name not in self.filters:
            print(f"Filter '{filter_name}' not found")
            return False
            
        try:
            # Map from our config names to RealSense option names
            option_mapping = {
                'magnitude': rs.option.filter_magnitude,
                'smooth_alpha': rs.option.filter_smooth_alpha,
                'smooth_delta': rs.option.filter_smooth_delta,
                'hole_fill': rs.option.holes_fill,
                'min_dist': rs.option.min_distance,
                'max_dist': rs.option.max_distance
            }
            
            # Map the user-friendly name to RealSense option
            if option_name not in option_mapping:
                print(f"Option '{option_name}' not recognized")
                return False
                
            rs_option = option_mapping[option_name]
            
            # Update the filter
            self.filters[filter_name].set_option(rs_option, value)
            
            # Also update our config dict for consistency
            if filter_name in self.filter_config and option_name in self.filter_config[filter_name]:
                self.filter_config[filter_name][option_name] = value
                
            print(f"Updated {filter_name}.{option_name} to {value}")
            return True
            
        except Exception as e:
            print(f"Error updating filter parameter: {e}")
            return False
    
    def save_filter_config(self, file_path: Optional[str] = None) -> bool:
        """
        Save the current filter configuration to a JSON file.
        
        Args:
            file_path: Path to save the configuration (None for default path)
            
        Returns:
            True if successful, False otherwise
        """
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'filter_config.json')
            
        try:
            with open(file_path, 'w') as f:
                json.dump(self.filter_config, f, indent=4)
            print(f"Saved filter configuration to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving filter configuration: {e}")
            return False
    
    def load_filter_config(self, file_path: Optional[str] = None) -> bool:
        """
        Load filter configuration from a JSON file and reinitialize filters.
        
        Args:
            file_path: Path to the configuration file (None for default path)
            
        Returns:
            True if successful, False otherwise
        """
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'filter_config.json')
            
        if not os.path.exists(file_path):
            print(f"Filter configuration file not found: {file_path}")
            return False
            
        try:
            with open(file_path, 'r') as f:
                new_config = json.load(f)
                
            # Update config
            for filter_name, settings in new_config.items():
                if filter_name in self.filter_config:
                    self.filter_config[filter_name].update(settings)
            
            # Reinitialize filters with new config
            self.filters = {}
            self._initialize_filters()
            
            print(f"Loaded filter configuration from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading filter configuration: {e}")
            return False
    
    def get_filter_info(self) -> Dict:
        """
        Get information about current filter settings.
        
        Returns:
            Dictionary with filter information
        """
        info = {
            'enabled': self.enable_filters,
            'filters': {}
        }
        
        if not self.enable_filters:
            return info
            
        for filter_name, filter_obj in self.filters.items():
            # Skip disparity transforms as they don't have configurable options
            if filter_name in ['depth_to_disparity', 'disparity_to_depth']:
                continue
                
            filter_info = {}
            
            # Try to get standard options for each filter
            try:
                if hasattr(rs.option, 'filter_magnitude'):
                    filter_info['magnitude'] = filter_obj.get_option(rs.option.filter_magnitude)
            except:
                pass
                
            try:
                if hasattr(rs.option, 'filter_smooth_alpha'):
                    filter_info['smooth_alpha'] = filter_obj.get_option(rs.option.filter_smooth_alpha)
            except:
                pass
                
            try:
                if hasattr(rs.option, 'filter_smooth_delta'):
                    filter_info['smooth_delta'] = filter_obj.get_option(rs.option.filter_smooth_delta)
            except:
                pass
                
            try:
                if hasattr(rs.option, 'holes_fill'):
                    filter_info['hole_fill'] = filter_obj.get_option(rs.option.holes_fill)
            except:
                pass
                
            info['filters'][filter_name] = filter_info
            
        return info
    
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
            align_frames=True,
            enable_filters=False,
            filter_config=None
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
            
            if rgb_frame is None and depth_frame is None:
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

def test_camera_under_latency():
    """Test the D455 camera functionality with intentional latency between frame queries."""
    print("\n=== Testing RealSense D455 Camera with Latency ===")
    
    try:
        # Create camera object
        camera = D455Camera(
            enable_rgb=True,
            enable_depth=True,
            rgb_resolution=(848, 480),
            depth_resolution=(848, 480),
            fps=30,
            align_frames=True
        )
        
        # Start camera
        print("Starting camera stream...")
        if not camera.start():
            print("Failed to start camera")
            return
            
        # Create display window
        print("Starting display loop with 0.5s delay between frames...")
        print("Press 'ESC' or 'q' to exit")
        
        cv2.namedWindow("Camera Test with Latency", cv2.WINDOW_AUTOSIZE)
        
        # Test with intentional delay
        for i in range(100):  # Capture 100 frames or until user exits
            # Add intentional delay
            time.sleep(5)
            
            # Get frames
            start_time = time.time()
            rgb_frame, depth_frame = camera.get_frames()
            frame_time = time.time() - start_time
            
            if rgb_frame is None or depth_frame is None:
                print(f"Frame {i}: Failed to get frames")
                continue
                
            # Convert depth to color map for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # Combine images side by side
            combined = np.hstack((rgb_frame, depth_colormap))
            
            # Add frame information
            cv2.putText(combined, 
                       f"Frame: {i} | Capture time: {frame_time*1000:.1f}ms",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show images
            cv2.imshow("Camera Test with Latency", combined)
            print(f"Frame {i} captured (took {frame_time*1000:.1f}ms)")
            
            # Check for key press to exit
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # ESC or q key
                print("User interrupted test")
                break
                
    except Exception as e:
        print(f"\nError during camera test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        print("Cleaning up...")
        if 'camera' in locals():
            camera.stop()
        cv2.destroyAllWindows()
        print("Latency test completed.")

def tune_filters():
    """
    Interactive utility to tune depth filters for optimal performance.
    
    This tool allows users to:
    1. See real-time comparison between filtered and unfiltered depth
    2. Adjust filter parameters with live feedback
    3. Save optimized filter configuration
    """
    print("\n=== RealSense D455 Filter Tuning Utility ===")
    
    try:
        # Create camera with default filter settings
        camera = D455Camera(
            enable_rgb=True,
            enable_depth=True,
            rgb_resolution=(848, 480),
            depth_resolution=(848, 480),
            fps=30,
            align_frames=True,
            enable_filters=True
        )
        
        # Start camera
        if not camera.start():
            print("Failed to start camera")
            return
            
        print("\nFilter Tuning Instructions:")
        print("  Press 'T' to toggle filters on/off")
        print("  Press '1-5' to select active filter:")
        print("    1 - Decimation")
        print("    2 - Spatial")
        print("    3 - Temporal")
        print("    4 - Hole Filling")
        print("    5 - Threshold")
        print("  Press 'UP/DOWN' to increase/decrease parameter value")
        print("  Press 'LEFT/RIGHT' to select parameter")
        print("  Press 'S' to save current configuration")
        print("  Press 'L' to load saved configuration")
        print("  Press 'ESC' to exit")
        
        cv2.namedWindow("Filter Tuning", cv2.WINDOW_AUTOSIZE)
        
        # State variables
        active_filter = 'spatial'
        active_param = 'magnitude'
        filters_enabled = True
        
        # Parameter maps for each filter type
        parameters = {
            'decimation': ['magnitude'],
            'spatial': ['magnitude', 'smooth_alpha', 'smooth_delta', 'hole_fill'],
            'temporal': ['smooth_alpha', 'smooth_delta', 'persistence_control'],
            'hole_filling': ['mode'],
            'threshold': ['min_dist', 'max_dist']
        }
        
        # Step sizes for each parameter
        step_sizes = {
            'magnitude': 1.0,
            'smooth_alpha': 0.05,
            'smooth_delta': 1.0,
            'hole_fill': 1.0,
            'mode': 1.0,
            'min_dist': 0.1,
            'max_dist': 0.1,
            'persistence_control': 1.0
        }
        
        # Main loop
        while True:
            # Get frames
            rgb_frame, depth_frame = camera.get_frames()
            
            if rgb_frame is None or depth_frame is None:
                print("Failed to get frames")
                time.sleep(0.1)
                continue
            
            # Get unfiltered frame for comparison
            camera.enable_filters = False
            _, unfiltered_depth = camera.get_frames()
            camera.enable_filters = filters_enabled
            
            # Create visualizations
            depth_color = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            unfiltered_color = cv2.applyColorMap(
                cv2.convertScaleAbs(unfiltered_depth, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # Calculate hole percentages
            filtered_holes = np.sum(depth_frame == 0) / depth_frame.size * 100
            unfiltered_holes = np.sum(unfiltered_depth == 0) / unfiltered_depth.size * 100
            
            # Draw filter status
            cv2.putText(depth_color, 
                       f"Filtered (holes: {filtered_holes:.1f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(unfiltered_color, 
                       f"Unfiltered (holes: {unfiltered_holes:.1f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add filter info
            y_pos = 70
            cv2.putText(depth_color, 
                       f"Active filter: {active_filter} | Parameter: {active_param}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display current filter settings
            if active_filter in camera.filter_config:
                for param, value in camera.filter_config[active_filter].items():
                    y_pos += 30
                    highlight = param == active_param
                    color = (0, 255, 255) if highlight else (255, 255, 255)
                    cv2.putText(depth_color, 
                               f"{param}: {value}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if highlight else 1)
            
            # Combine images
            comparison = np.hstack((depth_color, unfiltered_color))
            
            # Show image
            cv2.imshow("Filter Tuning", comparison)
            
            # Handle key events
            key = cv2.waitKey(1)
            
            if key == 27:  # ESC
                break
            elif key == ord('t'):
                # Toggle filters
                filters_enabled = not filters_enabled
                camera.enable_filters = filters_enabled
                print(f"Filters {'enabled' if filters_enabled else 'disabled'}")
            elif key == ord('s'):
                # Save configuration
                camera.save_filter_config()
            elif key == ord('l'):
                # Load configuration
                camera.load_filter_config()
            elif key == ord('1'):
                active_filter = 'decimation'
                active_param = parameters[active_filter][0]
                print(f"Selected filter: {active_filter}")
            elif key == ord('2'):
                active_filter = 'spatial'
                active_param = parameters[active_filter][0]
                print(f"Selected filter: {active_filter}")
            elif key == ord('3'):
                active_filter = 'temporal'
                active_param = parameters[active_filter][0]
                print(f"Selected filter: {active_filter}")
            elif key == ord('4'):
                active_filter = 'hole_filling'
                active_param = parameters[active_filter][0]
                print(f"Selected filter: {active_filter}")
            elif key == ord('5'):
                active_filter = 'threshold'
                active_param = parameters[active_filter][0]
                print(f"Selected filter: {active_filter}")
            elif key == 82:  # Up arrow
                # Increase parameter value
                if active_filter in camera.filter_config and active_param in camera.filter_config[active_filter]:
                    current_value = camera.filter_config[active_filter][active_param]
                    step = step_sizes.get(active_param, 1.0)
                    new_value = current_value + step
                    camera.update_filter_parameter(active_filter, active_param, new_value)
            elif key == 84:  # Down arrow
                # Decrease parameter value
                if active_filter in camera.filter_config and active_param in camera.filter_config[active_filter]:
                    current_value = camera.filter_config[active_filter][active_param]
                    step = step_sizes.get(active_param, 1.0)
                    new_value = max(0, current_value - step)  # Prevent negative values
                    camera.update_filter_parameter(active_filter, active_param, new_value)
            elif key == 81:  # Left arrow
                # Previous parameter
                if active_filter in parameters:
                    param_list = parameters[active_filter]
                    current_idx = param_list.index(active_param) if active_param in param_list else 0
                    new_idx = (current_idx - 1) % len(param_list)
                    active_param = param_list[new_idx]
                    print(f"Selected parameter: {active_param}")
            elif key == 83:  # Right arrow
                # Next parameter
                if active_filter in parameters:
                    param_list = parameters[active_filter]
                    current_idx = param_list.index(active_param) if active_param in param_list else 0
                    new_idx = (current_idx + 1) % len(param_list)
                    active_param = param_list[new_idx]
                    print(f"Selected parameter: {active_param}")
            
    except Exception as e:
        print(f"Error in filter tuning: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'camera' in locals():
            camera.stop()
        cv2.destroyAllWindows()
        print("Filter tuning completed")


if __name__ == "__main__":
    test_camera()
    # test_camera_under_latency()
    # tune_filters()
