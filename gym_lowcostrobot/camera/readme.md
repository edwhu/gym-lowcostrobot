# Camera to Robot Calibration Guide
Tony Wang
2025-03-26

---

Here is how Koch1.1 is calibrated with RealSense D455 camera.

The target is to transform the RGBD xyz into the robot frame xyz.

## File 

- `lowcostrobot/camera/d455.py`: RealSense D455 camera class, check as it can be wrong (Written by Tony)
- `lowcostrobot/camera/realtime_calibrate.py`: Real-time calibration script
- `lowcostrobot/camera/realtime_dp_xyz.py`: Real-time Depth Pro Model estimation script



## Transformation Chain

1. **Robot base → Robot gripper**: Internal robot kinematics
2. **Marker → Robot base**: Known transformation, marker placed at (-0.01, 0, 0) relative to robot base
3. **Camera → Marker**: Detected using ArUco markers
4. **Camera → Robot base**: Calculated using the above transformations

## Coordinate Systems

- **Robot Base Frame**: The origin of the robot is at front center
   - X-axis points backward
   - Y-axis points right
   - Z-axis points upward

- **Camera Frame**: 
  - Origin at the camera's optical center
  - Z-axis points forward (viewing direction)
  - X-axis points to the right
  - Y-axis points down

## Calibration Matrix

`T_base_camera`, which transforms points from the camera frame to the robot base frame:

```
point_base = T_base_camera @ point_camera
```

```
T_base_camera = T_base_marker @ inv(T_camera_marker)
```


## Calibration Process

1. Place the ArUco marker, clearly and in front of robot base

2. Record position of the ArUco marker in `T_base_marker` within start of `realtime_calibrate.py`

3. Run the realtime calibration script:
   ```
   python lowcostrobot/camera/realtime_calibrate.py 0.05 # optional, 5cm is default
   ```

4. If the xyz axis of the marker is aligned with the robot base frame, press 'S' to save the calibration matrix.

5. Test the calibration by running `python lowcostrobot/controller/camera_click_control.py`
   - Click the object and see the pos difference between the goal position and the end position

---
> Below are some common issues and their solutions

## Troubleshooting

### ArUco Marker Detection Issues

If the ArUco marker isn't being detected:

1. **Distance**: Position the camera 20-50cm from the marker. If too far, the marker will be too small to detect.

2. **Lighting**: Ensure even lighting without glare on the marker.

3. **Marker Size**: Make sure the `marker_size` parameter matches your actual marker size in meters.

4. **Visibility**: Ensure the entire marker is in the camera's field of view. Avoid occlusions caused by robot. 

### Transformation Issues

If points aren't mapping correctly between camera and robot space:

1. **Marker Position**: Verify that `T_base_marker` in `realtime_calibrate.py` matches the actual position of your marker relative to the robot base.

2. **Matrix Application**: When transforming points from camera to robot base, directly apply `T_base_camera` without inverting:

3. **Calibration File**: Check that you're using the most recent calibration file. The file should be in `results/calibration_matrix.npy`.

