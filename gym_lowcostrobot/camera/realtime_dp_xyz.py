import numpy as np
import cv2
import time
import os
import torch
import depth_pro
import warnings
from gym_lowcostrobot.camera.d455 import D455Camera

# Filter out the specific PyTorch warning about weights_only
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

class Realtime_DP_XYZ():
    def __init__(self, output_dir='results'):
        # Initialize camera
        self.camera = D455Camera(
            enable_rgb=True,
            enable_depth=True,
            rgb_resolution=(848, 480),
            depth_resolution=(848, 480),
            fps=2,   
            align_frames=True
        )
        
        # Load calibration matrix
        self.T_base_camera = np.load('results/calibration_matrix.npy')
        
        # Camera matrix (should match your camera calibration)
        self.camera_matrix = np.array([[424, 0, 424],
                                     [0, 424, 240],
                                     [0, 0, 1]], dtype=np.float32)
        
        # Initialize DepthPro model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device,
            precision=torch.half,
        )
        self.model.eval()
        
        # Visualization parameters
        self.selected_point = None
        self.show_dp_depth = True
        self.show_scaled_depth = False
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def start(self):
        if not self.camera.start():
            print("Failed to start camera")
            return False
        return True
        
    def stop(self):
        self.camera.stop()
        cv2.destroyAllWindows()
        
    def get_dp_depth(self, image, f_px=322.5):
        with torch.no_grad():
            image = self.transform(image)
            prediction = self.model.infer(image, f_px=torch.tensor(f_px))
            dp_depth = prediction["depth"]
            return dp_depth.cpu().numpy() * 1000  # Convert to mm
        
    def get_scaled_depth(self, rs_depth, dp_depth):
        return dp_depth * np.median(rs_depth / dp_depth)
        
    def depth_to_point_cloud(self, depth_map):
        """Convert depth map to point cloud in camera frame"""
        height, width = depth_map.shape
        fx = self.camera_matrix[0,0]
        fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]
        cy = self.camera_matrix[1,2]
        
        # Create grid of pixel coordinates
        v, u = np.mgrid[0:height, 0:width]
        
        # Convert to 3D points in camera frame
        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack to 3D points
        points = np.stack([x, y, z], axis=-1)
        return points
    
 
    def process_frame(self, rgb_frame, depth_frame):
        """Process a single frame for DepthPro estimation"""
        # Convert RGB frame from BGR to RGB
        rgb_frame_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        
        # Get DepthPro estimation
        dp_depth = self.get_dp_depth(rgb_frame_rgb)
        scaled_depth = self.get_scaled_depth(depth_frame, dp_depth)
        
        # Create visualizations
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_frame, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        dp_depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(dp_depth, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        scaled_depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(scaled_depth, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        # Stack images horizontally
        vis_images = [rgb_frame, depth_colormap]
        
        if self.show_dp_depth:
            vis_images.append(dp_depth_colormap)
        if self.show_scaled_depth:
            vis_images.append(scaled_depth_colormap)
            
        vis_image = np.hstack(vis_images)
        
        # Draw selected point if exists
        if self.selected_point is not None:
            x, y = self.selected_point
            color = (0, 255, 0)  # Green
            
            # Draw cross on all images
            for i in range(len(vis_images)):
                offset = i * rgb_frame.shape[1]
                cv2.line(vis_image, (offset + x - 10, y), (offset + x + 10, y), color, 2)
                cv2.line(vis_image, (offset + x, y - 10), (offset + x, y + 10), color, 2)
                
            # Get point coordinates in base frame
            points_camera = self.depth_to_point_cloud(depth_frame)
            points_base = self.transform_points_to_base(points_camera)
            point_base = points_base[y, x]
            
            # Display coordinates
            coord_text = f"Base XYZ: {point_base[0]:.1f}, {point_base[1]:.1f}, {point_base[2]:.1f} mm"
            cv2.putText(vis_image, coord_text, (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display depth values
            depth_text = f"RS: {depth_frame[y, x]:.1f}mm | DP: {dp_depth[y, x]:.1f}mm | Scaled: {scaled_depth[y, x]:.1f}mm"
            cv2.putText(vis_image, depth_text, (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
        
    def run(self):
        if not self.start():
            return
            
        print("Starting realtime DepthPro estimation...")
        print("Instructions:")
        print(" - Click on image to select a point")
        print(" - Press 'd' to toggle DepthPro depth view")
        print(" - Press 's' to toggle scaled depth view")
        print(" - Press 'c' to clear selected point")
        print(" - Press 'ESC' to exit")
        
        cv2.namedWindow("Realtime DepthPro XYZ Estimation", cv2.WINDOW_AUTOSIZE)
        
        # Set mouse callback for point selection
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Adjust x coordinate based on which image was clicked
                img_width = self.camera.rgb_resolution[0]
                img_idx = x // img_width
                x = x % img_width
                
                # Only allow selection on the first image (RGB)
                if img_idx == 0:
                    self.selected_point = (x, y)
                    print(f"Selected point: ({x}, {y})")
        
        cv2.setMouseCallback("Realtime DepthPro XYZ Estimation", mouse_callback)
        
        try:
            while True:
                # Get frames
                rgb_frame, depth_frame = self.camera.get_frames()
                
                if rgb_frame is None or depth_frame is None:
                    print("Failed to get frames")
                    time.sleep(0.1)
                    continue
                
                vis_image = self.process_frame(rgb_frame, depth_frame)
                
                # Show result
                cv2.imshow("Realtime DepthPro XYZ Estimation", vis_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == ord('d'):
                    self.show_dp_depth = not self.show_dp_depth
                elif key == ord('s'):
                    self.show_scaled_depth = not self.show_scaled_depth
                elif key == ord('c'):
                    self.selected_point = None
                elif key == ord('p'):  # Save point cloud data
                    if self.selected_point is not None:
                        x, y = self.selected_point
                        points_camera = self.depth_to_point_cloud(depth_frame)
                        points_base = self.transform_points_to_base(points_camera)
                        point_base = points_base[y, x]
                        
                        timestamp = int(time.time())
                        np.save(os.path.join(self.output_dir, f'point_{timestamp}.npy'), point_base)
                        print(f"Saved point data: {point_base}")
        finally:
            self.stop()

if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    
    dp_xyz = Realtime_DP_XYZ(output_dir)
    dp_xyz.run()