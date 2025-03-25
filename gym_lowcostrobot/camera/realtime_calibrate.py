import numpy as np
import cv2
import time
from gym_lowcostrobot.camera.d455 import D455Camera
from simpe_calibrate import SimpleCalibrator


'''
Stream version of realtime calibrator
Feature
    - realtime calibration
    - marker to base translation
    - store the calibration as camera extrinsic to the robot base
Usage:
    - Place the marker near the robot base
    - Measure the marker position relative to the robot base
    - Measure the marker size and input
    
TODO:
    - Clean the code
Update:
    - Tony, 02-09-2025 V1
'''

class RealtimeCalibrator(SimpleCalibrator):
    def __init__(self, output_dir='results'):
        super().__init__(output_dir)
        # Initialize camera
        self.camera = D455Camera(
            enable_rgb=True,
            enable_depth=True,
            rgb_resolution=(848, 480),
            depth_resolution=(848, 480),
            fps=30,
            align_frames=True
        )
        
        # Known transformation from robot base to ArUco marker
        # This is the position where we placed the marker relative to robot base
        # Assuming marker is placed on XY plane of robot base
        # NOTICE: you need to read the marker position via moving the robot eef to it
        self.T_base_marker = np.array([
            [1, 0, 0, -0.025],  # x translation from  
            [0, 1, 0, 0.025],  # y translation from  
            [0, 0, 1, 0.0],   # z translation from  
            [0, 0, 0, 1]
        ])
        
        # Initialize transformation from camera to robot base
        self.T_base_camera = np.eye(4)
        
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
        
    def process_frame(self, rgb_frame, depth_frame, marker_size=0.02):
        """Process a single frame for calibration"""
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250) # TODO expand dictionary to arbitrary size
        detector = cv2.aruco.ArucoDetector(dictionary)
        
        corners, ids, _ = detector.detectMarkers(gray)
        
        vis_image = rgb_frame.copy()
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis_image, corners, ids)
            
            for i in range(len(corners)):
                # 3D points of the marker
                objPoints = np.array([[-marker_size/2, marker_size/2, 0],
                                    [marker_size/2, marker_size/2, 0],
                                    [marker_size/2, -marker_size/2, 0],
                                    [-marker_size/2, -marker_size/2, 0]], dtype=np.float32)
                
                # Pose estimation
                success, rvec, tvec = cv2.solvePnP(objPoints, 
                                                 corners[i].reshape(4,2), 
                                                 self.camera_matrix, 
                                                 self.dist_coeffs)
                
                if success:
                    # Draw axes on marker
                    axis_length = marker_size
                    cv2.drawFrameAxes(vis_image, self.camera_matrix, 
                                    self.dist_coeffs, rvec, tvec, axis_length)
                    
                    # Get rotation matrix
                    R, _ = cv2.Rodrigues(rvec)
                    
                    # Construct T_camera_marker
                    T_camera_marker = np.eye(4)
                    T_camera_marker[:3, :3] = R
                    T_camera_marker[:3, 3] = tvec.reshape(3)
                    
                    # Method 1: Translate from marker to base
                    #  T_base_camera = T_base_marker * T_marker_camera 
                    #                = T_base_marker * inv(T_camera_marker)
                    self.T_base_camera = self.T_base_marker @ np.linalg.inv(T_camera_marker)

                    # Method 2: Directly set marker at same position as base
                    # self.T_base_camera = np.linalg.inv(T_camera_marker)
                    
                    # Draw some info on image
                    pos = self.T_base_camera[:3, 3]
                    cv2.putText(vis_image, 
                              f"Camera position: {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}m",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if depth_frame is not None:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03),
                cv2.COLORMAP_JET
            )
            vis_image = np.hstack((vis_image, depth_colormap))
        
        return vis_image
        
    def run(self, marker_size=0.02):
        if not self.start():
            return
            
        print("Starting realtime calibration...")
        print("Press 'S' to save current calibration")
        print("Press 'ESC' to exit")
        
        cv2.namedWindow("Realtime Calibration", cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                # Get frames
                rgb_frame, depth_frame = self.camera.get_frames()
                
                if rgb_frame is None or depth_frame is None:
                    print("Failed to get frames")
                    time.sleep(0.1)
                    continue
                
                vis_image = self.process_frame(rgb_frame, depth_frame, marker_size)
                
                # Show result
                cv2.imshow("Realtime Calibration", vis_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # Save calibration
                    matrix_path = os.path.join(self.output_dir, 'calibration_matrix.npy')
                    np.save(matrix_path, self.T_base_camera)
                    print(f"Saved calibration matrix to {matrix_path}")
                    cv2.imwrite(os.path.join(self.output_dir, 'calibration_view.jpg'), vis_image)
                    print(f"camera extrinstic: \n{self.T_base_camera}")
        finally:
            self.stop()

if __name__ == "__main__":
    import sys
    import os
    
    marker_size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'results'
    
    calibrator = RealtimeCalibrator(output_dir)
    calibrator.run(marker_size) 