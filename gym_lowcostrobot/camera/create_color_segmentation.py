"""
This script creates a color segmentation mask for the realsense camera. It has a GUI for the user to create the color segmentation mask.
The user can click on an object to sample its color and adjust HSV thresholds.
The user can also define a crop region by clicking two points for the top-left and bottom-right corners.
"""

import numpy as np
import cv2
import time
import os
import yaml  # Add YAML import
from gym_lowcostrobot.camera.d455 import D455Camera

class ColorSegmentation:
    def __init__(self, output_dir='results'):
        # Initialize camera
        self.camera = D455Camera(
            enable_rgb=True,
            enable_depth=False,
            rgb_resolution=(848, 480),
            depth_resolution=(848, 480),
            fps=30,
            align_frames=True
        )
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize colors
        self.clicked_colors = []
        self.hsv_lower = np.array([0, 0, 0])
        self.hsv_upper = np.array([179, 255, 255])
        
        # Initialize crop region
        self.crop_mode = False
        self.crop_points = []
        self.crop_region = None  # (x1, y1, x2, y2)
        
        # Window name
        self.window_name = "Color Segmentation"
        
        # Mouse click flag
        self.click_count = 0
        
    def start(self):
        """Start camera streaming"""
        if not self.camera.start():
            print("Failed to start camera")
            return False
        return True
        
    def stop(self):
        """Stop camera streaming"""
        self.camera.stop()
        cv2.destroyAllWindows()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events to sample colors or define crop region"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get current frame
            frame = param
            if frame is not None:
                if self.crop_mode:
                    # In crop mode, record points for the crop region
                    self.crop_points.append((x, y))
                    print(f"Crop point {len(self.crop_points)}: ({x}, {y})")
                    
                    # If we have two points, define the crop region
                    if len(self.crop_points) == 2:
                        x1 = min(self.crop_points[0][0], self.crop_points[1][0])
                        y1 = min(self.crop_points[0][1], self.crop_points[1][1])
                        x2 = max(self.crop_points[0][0], self.crop_points[1][0])
                        y2 = max(self.crop_points[0][1], self.crop_points[1][1])
                        
                        self.crop_region = (x1, y1, x2, y2)
                        self.crop_mode = False  # Exit crop mode
                        print(f"Crop region defined: {self.crop_region}")
                else:
                    # Normal color sampling mode
                    # Convert BGR to HSV
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    # Sample color at the clicked point
                    color = hsv_frame[y, x]
                    self.clicked_colors.append(color)
                    self.click_count += 1
                    print(f"Sampled color {self.click_count}: HSV = {color}")
                    
                    # Update color ranges based on clicked points
                    if len(self.clicked_colors) > 0:
                        # Calculate min and max for each HSV component
                        hsv_values = np.array(self.clicked_colors)
                        # Add tolerance to the min/max values
                        tolerance_h = 10  # Hue tolerance
                        tolerance_s = 50  # Saturation tolerance
                        tolerance_v = 50  # Value tolerance
                        
                        # Calculate min/max with tolerance
                        h_min = max(0, np.min(hsv_values[:, 0]) - tolerance_h)
                        s_min = max(0, np.min(hsv_values[:, 1]) - tolerance_s)
                        v_min = max(0, np.min(hsv_values[:, 2]) - tolerance_v)
                        
                        h_max = min(179, np.max(hsv_values[:, 0]) + tolerance_h)
                        s_max = min(255, np.max(hsv_values[:, 1]) + tolerance_s)
                        v_max = min(255, np.max(hsv_values[:, 2]) + tolerance_v)
                        
                        # Update trackbar positions
                        cv2.setTrackbarPos('HMin', self.window_name, int(h_min))
                        cv2.setTrackbarPos('SMin', self.window_name, int(s_min))
                        cv2.setTrackbarPos('VMin', self.window_name, int(v_min))
                        cv2.setTrackbarPos('HMax', self.window_name, int(h_max))
                        cv2.setTrackbarPos('SMax', self.window_name, int(s_max))
                        cv2.setTrackbarPos('VMax', self.window_name, int(v_max))

    def create_trackbars(self):
        """Create trackbars for adjusting HSV thresholds"""
        def nothing(x):
            pass
        
        # Create trackbars
        cv2.createTrackbar('HMin', self.window_name, 0, 179, nothing)
        cv2.createTrackbar('SMin', self.window_name, 0, 255, nothing)
        cv2.createTrackbar('VMin', self.window_name, 0, 255, nothing)
        cv2.createTrackbar('HMax', self.window_name, 179, 179, nothing)
        cv2.createTrackbar('SMax', self.window_name, 255, 255, nothing)
        cv2.createTrackbar('VMax', self.window_name, 255, 255, nothing)
    
    def get_threshold_values(self):
        """Get current threshold values from trackbars"""
        h_min = cv2.getTrackbarPos('HMin', self.window_name)
        s_min = cv2.getTrackbarPos('SMin', self.window_name)
        v_min = cv2.getTrackbarPos('VMin', self.window_name)
        h_max = cv2.getTrackbarPos('HMax', self.window_name)
        s_max = cv2.getTrackbarPos('SMax', self.window_name)
        v_max = cv2.getTrackbarPos('VMax', self.window_name)
        
        self.hsv_lower = np.array([h_min, s_min, v_min])
        self.hsv_upper = np.array([h_max, s_max, v_max])
        
        return self.hsv_lower, self.hsv_upper
    
    def apply_mask(self, frame):
        """Apply color segmentation mask to the frame"""
        # Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get current threshold values
        lower, upper = self.get_threshold_values()
        
        # Create mask
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        # Apply mask
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        return mask, result
    
    def apply_crop(self, frame, mask=None, result=None):
        """Apply crop to frame and optionally to mask and result"""
        if self.crop_region is None:
            return frame, mask, result
        
        x1, y1, x2, y2 = self.crop_region
        cropped_frame = frame[y1:y2, x1:x2].copy()
        
        if mask is not None:
            cropped_mask = mask[y1:y2, x1:x2].copy()
        else:
            cropped_mask = None
            
        if result is not None:
            cropped_result = result[y1:y2, x1:x2].copy()
        else:
            cropped_result = None
            
        return cropped_frame, cropped_mask, cropped_result
    
    def draw_crop_region(self, frame):
        """Draw the crop region or crop points on the frame"""
        vis_frame = frame.copy()
        
        # Draw crop region if it exists
        if self.crop_region is not None:
            x1, y1, x2, y2 = self.crop_region
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw crop points if in crop mode
        if self.crop_mode and len(self.crop_points) > 0:
            for i, point in enumerate(self.crop_points):
                cv2.circle(vis_frame, point, 5, (0, 0, 255), -1)
                cv2.putText(vis_frame, f"P{i+1}", (point[0]+10, point[1]+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # If one point is selected, draw a rectangle from that point to current mouse position
            if len(self.crop_points) == 1 and 'mouse_pos' in dir(self):
                cv2.rectangle(vis_frame, self.crop_points[0], self.mouse_pos, (0, 0, 255), 2)
        
        return vis_frame
    
    def run(self):
        """Run the color segmentation tool"""
        if not self.start():
            return
        
        print("Starting color segmentation tool...")
        print("Click on objects to sample colors")
        print("Adjust sliders to fine-tune the color range")
        print("Press 'C' to define crop region (requires 2 clicks for corners)")
        print("Press 'S' to save current configuration")
        print("Press 'R' to reset")
        print("Press 'ESC' to exit")
        
        # Create window and trackbars
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.create_trackbars()
        
        # Set mouse callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback, None)
        
        try:
            while True:
                # Get frame
                rgb_frame, _ = self.camera.get_frames()
                
                if rgb_frame is None:
                    print("Failed to get frames")
                    time.sleep(0.1)
                    continue
                
                # Update mouse callback parameter
                cv2.setMouseCallback(self.window_name, self.mouse_callback, rgb_frame)
                
                # Apply mask
                mask, result = self.apply_mask(rgb_frame)
                
                # Draw crop region
                vis_rgb = self.draw_crop_region(rgb_frame)
                vis_result = self.draw_crop_region(result)
                
                # Show cropped version if crop region exists
                if self.crop_region is not None and not self.crop_mode:
                    # Apply crop
                    cropped_frame, cropped_mask, cropped_result = self.apply_crop(rgb_frame, mask, result)
                    
                    # Create small visualization of cropped view
                    # Add a small inset of the cropped region
                    h, w = cropped_frame.shape[:2]
                    max_inset_size = 200
                    scale = min(max_inset_size / w, max_inset_size / h)
                    inset_w, inset_h = int(w * scale), int(h * scale)
                    
                    cropped_resized = cv2.resize(cropped_frame, (inset_w, inset_h))
                    cropped_result_resized = cv2.resize(cropped_result, (inset_w, inset_h))
                    
                    # Create a border around the inset
                    cropped_resized = cv2.copyMakeBorder(cropped_resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                    cropped_result_resized = cv2.copyMakeBorder(cropped_result_resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                    
                    # Overlay the insets at the bottom right
                    h_main, w_main = vis_rgb.shape[:2]
                    h_inset, w_inset = cropped_resized.shape[:2]
                    
                    # Position the insets
                    x_offset = w_main - w_inset - 10
                    y_offset = h_main - h_inset - 10
                    
                    # Add the insets
                    vis_rgb[y_offset:y_offset+h_inset, x_offset:x_offset+w_inset] = cropped_resized
                    vis_result[y_offset:y_offset+h_inset, x_offset:x_offset+w_inset] = cropped_result_resized
                
                # Create visualization
                # Display original and masked images side by side
                vis_image = np.hstack([vis_rgb, vis_result])
                
                # Add text showing current HSV ranges
                text = f"HSV Range: [{self.hsv_lower[0]},{self.hsv_lower[1]},{self.hsv_lower[2]}] - [{self.hsv_upper[0]},{self.hsv_upper[1]},{self.hsv_upper[2]}]"
                cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show clicked points count
                cv2.putText(vis_image, f"Clicked points: {self.click_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show crop mode status
                crop_status = "CROP MODE ACTIVE - Click two points" if self.crop_mode else ""
                if self.crop_region is not None and not self.crop_mode:
                    x1, y1, x2, y2 = self.crop_region
                    crop_status = f"Crop region: ({x1},{y1}) to ({x2},{y2})"
                
                cv2.putText(vis_image, crop_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show result
                cv2.imshow(self.window_name, vis_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # Save configuration
                    # Save HSV ranges
                    hsv_ranges = {
                        'lower': self.hsv_lower.tolist(),
                        'upper': self.hsv_upper.tolist()
                    }
                    
                    # Add crop region if defined
                    if self.crop_region is not None:
                        hsv_ranges['crop_region'] = self.crop_region
                    
                    # Save as YAML file
                    yaml_path = os.path.join(self.output_dir, 'color_segmentation_config.yaml')
                    with open(yaml_path, 'w') as f:
                        yaml.dump(hsv_ranges, f, default_flow_style=False)
                    
                    print(f"Saved configuration to {yaml_path}")
                    
                    # Also save as NumPy file for backward compatibility
                    np.save(os.path.join(self.output_dir, 'hsv_ranges.npy'), hsv_ranges)
                    
                    # Save sample images
                    cv2.imwrite(os.path.join(self.output_dir, 'original.jpg'), rgb_frame)
                    cv2.imwrite(os.path.join(self.output_dir, 'mask.jpg'), mask)
                    cv2.imwrite(os.path.join(self.output_dir, 'result.jpg'), result)
                    
                    # Save cropped images if crop region exists
                    if self.crop_region is not None:
                        cropped_frame, cropped_mask, cropped_result = self.apply_crop(rgb_frame, mask, result)
                        cv2.imwrite(os.path.join(self.output_dir, 'cropped_original.jpg'), cropped_frame)
                        cv2.imwrite(os.path.join(self.output_dir, 'cropped_mask.jpg'), cropped_mask)
                        cv2.imwrite(os.path.join(self.output_dir, 'cropped_result.jpg'), cropped_result)
                    
                    print(f"Saved images to {self.output_dir}")
                elif key == ord('r'):  # Reset
                    self.clicked_colors = []
                    self.click_count = 0
                    self.crop_mode = False
                    self.crop_points = []
                    self.crop_region = None
                    cv2.setTrackbarPos('HMin', self.window_name, 0)
                    cv2.setTrackbarPos('SMin', self.window_name, 0)
                    cv2.setTrackbarPos('VMin', self.window_name, 0)
                    cv2.setTrackbarPos('HMax', self.window_name, 179)
                    cv2.setTrackbarPos('SMax', self.window_name, 255)
                    cv2.setTrackbarPos('VMax', self.window_name, 255)
                    print("Reset color selection and crop region")
                elif key == ord('c'):  # Enter crop mode
                    self.crop_mode = True
                    self.crop_points = []
                    print("Entering crop mode. Click two points to define crop region.")
        finally:    
            self.stop()

if __name__ == "__main__":
    import sys
    import os
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'color_segmentation_results'
    
    segmentation = ColorSegmentation(output_dir)
    segmentation.run()