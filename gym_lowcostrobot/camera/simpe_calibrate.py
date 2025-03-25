import numpy as np
import cv2
import argparse
import os

'''
Base class for realtime calibrator
Feature
    - init camera metrix & output dir
    - single frame calibration
    - marker generation
TODO:
    - Clean the code
Update:
    - Tony, 02-09-2025 V1
'''

class SimpleCalibrator:
    def __init__(self, output_dir='results'):
        self.T_base_camera = np.eye(4)
        # RealSense D455   (848x480)
        self.camera_matrix = np.array([[424, 0, 424],
                                     [0, 424, 240],
                                     [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def calibrate(self, image, marker_size=0.02):
        """Simple ArUco marker calibration"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        detector = cv2.aruco.ArucoDetector(dictionary)
        
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is None:
            print("No markers detected")
            return False
            
        vis_image = image.copy()
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
                axis_length = marker_size/8
                cv2.drawFrameAxes(vis_image, self.camera_matrix, 
                                self.dist_coeffs, rvec, tvec, axis_length)
                
                R, _ = cv2.Rodrigues(rvec)
                self.T_base_camera[:3, :3] = R
                self.T_base_camera[:3, 3] = tvec.reshape(3)
                
                print(f"Marker {ids[i][0]} detected and used for calibration")
        
        cv2.imwrite(os.path.join(self.output_dir, 'calibration_result.jpg'), vis_image)
        return True

def generate_marker(marker_id=23, size_pixels=1000, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker = np.zeros((size_pixels, size_pixels), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dictionary, marker_id, size_pixels, marker, 1)
    
    output_path = os.path.join(output_dir, f"aruco_marker_{marker_id}.png")
    cv2.imwrite(output_path, marker)
    print(f"Generated marker {marker_id} at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--marker_size', type=float, default=0.01, help='Marker size in meters')
    parser.add_argument('--generate', action='store_true', help='Generate marker')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    args = parser.parse_args()
    
    if args.generate:
        generate_marker(output_dir=args.output_dir)
    else:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Could not read image: {args.image}")
            exit(1)
            
        calibrator = SimpleCalibrator(output_dir=args.output_dir)
        if calibrator.calibrate(image, args.marker_size):
            print("\nCalibration matrix:")
            print(calibrator.T_base_camera)
            
            matrix_path = os.path.join(args.output_dir, 'calibration_matrix.npy')
            np.save(matrix_path, calibrator.T_base_camera)
            print(f"Saved calibration matrix to {matrix_path}")