import os
import time
from collections import deque
from typing import Tuple

import gymnasium as gym
from gymnasium import Env, spaces
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import websockets
import asyncio
import json
import cv2
import yaml

from gym_lowcostrobot import ASSETS_PATH, BASE_LINK_NAME
from gym_lowcostrobot.envs.dynamixel import Dynamixel
from gym_lowcostrobot.envs.robot import Robot

import cv2
from gym_lowcostrobot.camera.d455 import D455Camera



# DEVICE_NAME='/dev/ttyACM0'
# DEVICE_NAME='/dev/tty.usbmodem58760435361'
# DEVICE_NAME='/dev/tty.usbmodem585A0085321'
MOTOR_3_BIAS = 30
ACTION_SLEEP_SEC = 0.5

class LiftCubeStateRealEnv(Env):
    """
    ## Description

    The robot has to lift a cube with its end-effector.

    ## Action space

    Two action modes are available: "joint" and "nullspace". In the "joint" mode, the action space is a 6-dimensional box
    representing the target joint angles.

    | Index | Action              | Type (unit) | Min  | Max |
    | ----- | ------------------- | ----------- | ---- | --- |
    | 0     | Shoulder pan joint  | Float (rad) | -1.0 | 1.0 |
    | 1     | Shoulder lift joint | Float (rad) | -1.0 | 1.0 |
    | 2     | Elbow flex joint    | Float (rad) | -1.0 | 1.0 |
    | 3     | Wrist flex joint    | Float (rad) | -1.0 | 1.0 |
    | 4     | Wrist roll joint    | Float (rad) | -1.0 | 1.0 |
    | 5     | Gripper joint       | Float (rad) | -1.0 | 1.0 |

    In the "nullspace" mode, the action space is a 4-dimensional box representing the target end-effector velocity and the
    gripper position. The actions are normalized to -1, 1, see the bounds in _initialize_action_space.

    | Index | Action        | Type (unit) | Min  | Max |
    | ----- | ------------- | ----------- | ---- | --- |
    | 0     | X             | Float       | -1.0 | 1.0 |
    | 1     | Y             | Float       | -1.0 | 1.0 |
    | 2     | Z             | Float       | -1.0 | 1.0 |
    | 5     | Gripper joint | Float       | -1.0 | 1.0 |

    ## Observation space

    The observation space is a dictionary containing multiple keys. See _initialize_observation_space for the full list of keys.

    ## Reward

    The reward is the sum of two terms: the height of the cube above the threshold and the negative distance between the
    end effector and the cube.

    ## Arguments

    - `observation_mode (str)`: the observation mode, can be "image", "state", or "both", default is "image", see
        section "Observation space".
    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is None.
    """

    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 200}

    def __init__(self, observation_mode="state", action_mode="nullspace", render_mode=None, render_obs=True, include_initial_obj_pose=False, use_camera=False, use_auto_target = False):
        self.use_camera = use_camera
        self.use_auto_target = use_auto_target
        self._initialize_dynamixel()
        self._initialize_mujoco()
        self._initialize_action_space(action_mode)
        self._initialize_observation_space(observation_mode, include_initial_obj_pose)
        self._initialize_renderer(render_mode, render_obs)
        self._initialize_task_variables()
    
    def _initialize_dynamixel(self):
        self.real_qpos_min = np.array([906, 0, 0, 0, 0, 2040]) # you may need to tune per robot
        self.real_qpos_max = np.array([3202, 4095,4095,4095,4095,2877])
        self.qpos_min = self.position_to_radian(self.real_qpos_min)
        self.qpos_max = self.position_to_radian(self.real_qpos_max)
        if 'KOCH_DEVICE_NAME' not in os.environ:
            raise ValueError("Please set the KOCH_DEVICE_NAME environment variable to the serial port of the Dynamixel device")
        DEVICE_NAME = os.environ['KOCH_DEVICE_NAME']
        self.dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name=DEVICE_NAME).instantiate()
        self.realrobot = Robot(self.dynamixel)
        self.initial_qpos = np.array([-0.017867, 0.005605, -0.131519, -1.433267, 1.552938, 0.8])
        # self.initial_qpos_before_camera = np.array([-1.2, 0.005605, -0.131519, -1.433267, 1.552938, 0.8])
        self.initial_qpos_before_camera = np.array([1.0, 0.005605, -0.131519, -1.433267, 1.552938, 0.8])

        assert np.all(self.initial_qpos >= self.qpos_min) and np.all(self.initial_qpos <= self.qpos_max)

        self.motor_3_bias = MOTOR_3_BIAS
        real_qpos = self.radian_to_position(self.initial_qpos)
        real_qpos[2] += self.motor_3_bias
        real_qpos = np.clip(real_qpos, self.real_qpos_min, self.real_qpos_max)
        self.realrobot.set_goal_pos(real_qpos)
        time.sleep(ACTION_SLEEP_SEC)

    def radian_to_position(self, values):
        scale = 4095 / (3.14 - (-3.14))
        return [round((v + 3.14) * scale) for v in values]

    def position_to_radian(self, positions):
        scale = 4095 / (3.14 - (-3.14))
        return [(p / scale) - 3.14 for p in positions]

    def _initialize_mujoco(self):
        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS_PATH, "lift_cube_camera.xml"), {})
        self.data = mujoco.MjData(self.model)
        # Enable gravity compensation. Set to 0.0 to disable.
        gravity_compensation = False
        for body in ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]:
            body_id = self.model.body(body).id
            self.model.body_gravcomp[body_id] = float(gravity_compensation)
        self.model.body_gravcomp[:] = float(gravity_compensation)
        # self.dt: float = 0.002
        # self.model.opt.timestep = self.dt
        self.nb_dof = 6
    
    def _initialize_action_space(self, action_mode):
        # Set the action space
        self.action_mode = action_mode
        action_shape = {"joint": 6, "ee": 4, "nullspace": 4}[action_mode]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_shape,), dtype=np.float32)
        # used for bounding the nullspace controller
        self.action_min = np.array([-0.1, -0.1, -0.1, -0.3], dtype=np.float32)
        self.action_max = np.array([0.1, 0.1, 0.1, 0.3], dtype=np.float32)

    def _initialize_observation_space(self, observation_mode, include_initial_obj_pose):
        # Set the observations space
        self.observation_mode = observation_mode
        self.observation_subspaces = {
            "arm_qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(6,)),
            "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),
            "log_is_success": spaces.Box(low=-np.inf, high=np.inf, dtype="float32"),
            "gripper_blocked": spaces.Box(low=-np.inf, high=np.inf, dtype="float32"),
        }

        # Only add camera-related spaces if use_camera is True
        if self.use_camera:
            self.observation_subspaces.update({
                "target_eepos": spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),
                "rgb": spaces.Box(low=0, high=255, shape=(480, 848, 3), dtype=np.uint8),
                "depth": spaces.Box(low=0, high=65535, shape=(480, 848), dtype=np.uint16),
                "estimated_target_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            })
            # initialize the camera
            self.init_camera()
        else:
            self.camera = None

        self.include_initial_obj_pose = include_initial_obj_pose
        if include_initial_obj_pose:
            self.initial_obj_pose = np.zeros((7,), dtype=np.float32)
            self.observation_subspaces["initial_obj_pose"] = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))

        if self.observation_mode in ["image", "both"]:
            # observation_subspaces["image_wrist"] = spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
            self.observation_subspaces["log_image_front"] = spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
            self.renderer = mujoco.Renderer(self.model, height=256, width=256)

        self.observation_space = gym.spaces.Dict(self.observation_subspaces)
    
    def _initialize_renderer(self, render_mode, render_obs):
        # Set the render utilities
        self.render_obs = render_obs
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = -75
            self.viewer.cam.distance = 1
            self.rgb_array_renderer = mujoco.Renderer(self.model, height=64, width=64)
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer = mujoco.Renderer(self.model, height=64, width=64)
    
    def _initialize_task_variables(self,):
        # Set variables used for the task
        self.threshold_height = 0.07
        ## modified to make the cube generate in the boundaries.
        # self.cube_low = np.array([-0.14, -0.03, 0.01])  # move the cube closer to the robot
        # self.cube_high = np.array([-0.08, 0.03, 0.01])

        # move the cube out of the boundaries
        self.cube_low = np.array([-2.14, -0.03, 0.01])  # move the cube closer to the robot
        self.cube_high = np.array([-2.08, 0.03, 0.01])

        # get dof addresses
        self.cube_dof_id = self.model.body("cube").dofadr[0]
        self.arm_dof_id = self.model.body(BASE_LINK_NAME).dofadr[0]
        self.arm_dof_vel_id = self.arm_dof_id
        # if the arm is not at address 0 then the cube will have 7 states in qpos and 6 in qvel
        if self.arm_dof_id != 0:
            self.arm_dof_id = self.arm_dof_vel_id + 1
        self.joint_names = [f"joint_{i}" for i in range(1, 7)]

        # workspace bounds for the ee 
        self.ee_min = np.array([-0.15, -0.1, 0.008]) # z-axis should be lower, like 0.010 (likely not impacting the data)
        self.ee_max = np.array([-0.06, 0.1, 0.1])

    def diffik_nullspace(
        self,
        goal_pos,
        goal_quat,
        site_id,
    ):
        model = self.model
        data = self.data
        # Spatial velocity (aka twist).
        dx = goal_pos - data.site(site_id).xpos
        self.twist[:3] = self.Kpos * dx / self.integration_dt
        mujoco.mju_mat2Quat(self.site_quat, data.site(site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, goal_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= self.Kori / self.integration_dt

        # Jacobian.
        temp_jac = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, temp_jac[:3], temp_jac[3:], site_id)
        self.jac = temp_jac[:, self.dof_ids]

        # Damped least squares.
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.twist)

        # Nullspace control biasing joint velocities towards the home configuration.
        dq += (self.eye - np.linalg.pinv(self.jac) @ self.jac) @ (self.Kn * (self.q0 - data.qpos[self.dof_ids]))

        # Clamp maximum joint velocity.
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > self.max_angvel:
            dq *= self.max_angvel / dq_abs_max
        # Integrate joint velocities to obtain joint positions.
        temp_dq = np.concatenate([dq, np.zeros(6)])
        q = data.qpos.copy()  # Note the copy here is important.
        mujoco.mj_integratePos(model, q, temp_dq, self.integration_dt)
        q =  q[self.dof_ids]
        np.clip(q, *model.jnt_range[self.dof_ids].T, out=q)
        return q

    def apply_action(self, action):
        """
        Step the simulation forward based on the action

        Action shape
        - nullspace EE mode: [dx, dy, dz, gripper]
        - Joint mode: [q1, q2, q3, q4, q5, q6, gripper]
        """
        info = {}
        if self.action_mode == "nullspace":
            assert action.min() >= -1.0 and action.max() <= 1.0
            # assume actions are relative and normalized to [-1, 1]
            raw_action = self.get_raw_action(action)
            ee_action, gripper_action = raw_action[:3], raw_action[-1]
            # print('gripper_action', gripper_action)
            # print('gripper_position', self.radian_to_position([gripper_action]))

            goal_pos = ee_action + self.data.site("attachment_site").xpos
            # clip the goal pos to ee bounds
            # warn if goal_pos is outside the ee bounds
            if np.any(goal_pos < self.ee_min) or np.any(goal_pos > self.ee_max):
                print(f"Goal pos is outside the ee bounds: {goal_pos}")
            goal_pos = np.clip(goal_pos, self.ee_min, self.ee_max)

            # goal_quat = np.array([0.7071, 0.7071, 0, 0]) # rotate 90 on x axis to make gripper point downwards.
            goal_quat = np.array([0.5, 0.5, 0.5, 0.5]) # rotate 90 on x axis to make gripper point downwards

            site_id = self.model.site("attachment_site").id

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            target_qpos = self.diffik_nullspace(
                goal_pos,
                goal_quat,
                site_id,
            )

            # TODO: figure out the gripper action
            # import ipdb; ipdb.set_trace()
            # print(target_qpos[-1:], gripper_action)
            target_qpos[-1:] += gripper_action
            target_real_qpos = self.radian_to_position(target_qpos)

        elif self.action_mode == "joint":
            # target_low = np.array([-3.14159, -1.5708, -1.48353, -1.91986, -2.96706, -1.74533])
            # target_high = np.array([3.14159, 1.22173, 1.74533, 1.91986, 2.96706, 0.0523599])
            # target_qpos = np.array(action).clip(target_low, target_high)
            target_qpos = np.array(action)
        else:
            raise ValueError("Invalid action mode, must be 'nullspace' or 'joint'")

        # Set the new position for the real robot with naive grav comp term
        target_real_qpos[2] += self.motor_3_bias
        target_real_qpos = np.clip(target_real_qpos, self.real_qpos_min, self.real_qpos_max)
        self.realrobot.set_goal_pos(target_real_qpos)
        time.sleep(ACTION_SLEEP_SEC) # give the robot time to move.
        info = {
            'goal_pos': goal_pos, # the goal eef position (xyz)
            'target_qpos': target_qpos, # the target joint positions from IK in radians
            'target_real_qpos': target_real_qpos, # the target joint positions in dynamixel units
            'raw_action': raw_action, # the action converted into metric space
        }
        if self.render_mode == "human":
            self.viewer.sync()
        return info

    def get_scaled_action(self, raw_action):
        # go from raw action space to (-1, 1) actions
        scaled_min, scaled_max = -1, 1
        raw_action = np.clip(raw_action, self.action_min, self.action_max)
        scaled_action = (raw_action - self.action_min) / (self.action_max - self.action_min) * (scaled_max - scaled_min) + scaled_min
        return scaled_action
    
    def get_raw_action(self, scaled_action):
        # go from (-1, 1) actions to raw action space
        scaled_min, scaled_max = -1, 1
        scaled_action = np.clip(scaled_action, scaled_min, scaled_max)
        raw_action = (scaled_action - scaled_min) / (scaled_max - scaled_min) * (self.action_max - self.action_min) + self.action_min
        return raw_action

    def get_observation(self):
        # qpos is [x, y, z, qw, qx, qy, qz, q1, q2, q3, q4, q5, q6, gripper]
        # qvel is [vx, vy, vz, wx, wy, wz, dq1, dq2, dq3, dq4, dq5, dq6, dgripper]
        try:
            real_qpos = np.asarray(self.realrobot.read_position(), dtype=np.float32)
            qpos = np.array(self.position_to_radian(real_qpos), dtype=np.float32)
        except Exception as e: 
            print(f"Failed to read position:, using last known and valid position")
            qpos = np.array(self.last_qpos.copy(), dtype=np.float32)
            real_qpos = self.radian_to_position(qpos)

        # if positions are all within the limits, then cache them
        threshold = 10
        if np.all(real_qpos >= (self.real_qpos_min - threshold)) and np.all(real_qpos <= (self.real_qpos_max + threshold)):
            self.last_qpos = self.position_to_radian(real_qpos)
        else:
            print(f"qpos_min: {self.real_qpos_min}")
            print(f"qpos_real {real_qpos} ")
            print(f"qpos max: {self.real_qpos_max}")
            print(f"You may need to tune the real_qpos_min and real_qpos_max values")
            qpos = np.clip(qpos, self.real_qpos_min, self.real_qpos_max)

        # set the sim robot qpos to the real robot qpos
        self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof] = qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        observation = {
            "arm_qpos": qpos,
            "ee_pos": self.get_ee_pos(),
        }

        if self.observation_mode in ["image", "both"]:
            if self.render_obs:
                self.rgb_array_renderer.update_scene(self.data, camera="camera_front")
                img = self.rgb_array_renderer.render()
            else:
                img = np.zeros((64, 64, 3), dtype=np.uint8)
            observation["log_image_front"] = img
        return observation
    
    def get_camera_observations(self, observation):
        if self.use_camera:
            rgb_img, depth_img = self.camera.get_frames()
            # Extract crop region and HSV thresholds from config
            crop_region = self.color_segmentation_config.get('crop_region')
            hsv_lower = np.array(self.color_segmentation_config.get('lower'))
            hsv_upper = np.array(self.color_segmentation_config.get('upper'))
            
            # Crop RGB and depth images 
            x1, y1, x2, y2 = crop_region
            rgb = rgb_img[y1:y2, x1:x2]
            depth = depth_img[y1:y2, x1:x2]
            assert rgb.shape[0] == depth.shape[0] and rgb.shape[1] == depth.shape[1]

            # Create segmentation mask using HSV thresholds
            hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV) # Note, rgb is actually in BGR.
            # Create mask using HSV thresholds
            segmentation = cv2.inRange(hsv, hsv_lower, hsv_upper)
            segmentation = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2RGB)
            observation['segmentation'] = segmentation

            # convert back to RGB
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            observation["rgb"] = rgb
            observation["depth"] = depth
            if observation["gripper_blocked"]:
                print("Gripper blocked, using ee pos as estimated target pos")
                observation["estimated_target_pos"] = observation["ee_pos"][:3]
            else:
                # segmentation shape: (61, 168, 3) because it's cropped
                # rgb shape: (480, 848, 3) because it's not cropped
                # the get 3dpoint method takes in a pixel coordinate and 
                # compute the x, y position of the pixel as the geometric center of the pixels in the cropped image where value is 255
                y, x, _ = np.where(segmentation == 255)
                x_center = np.mean(x).astype(np.int32)
                y_center = np.mean(y).astype(np.int32)
                # transform the x, y position of the pixel in the cropped image to the x, y position of the pixel in the full image
                x_full, y_full = transform_cropped_to_full_image(x_center, y_center, x1, y1)
                # print(f"x_center: {x_center}, y_center: {y_center}")
                # print(f"x_full: {x_full}, y_full: {y_full}")
                estimated_target_pos = self.camera.get_3d_point_robot_base(x_full, y_full, depth_img)
                if estimated_target_pos is not None:
                    observation["estimated_target_pos"] = estimated_target_pos
                else:
                    print(f"Object position is not in the camera view, using ee pos as estimated target pos")
                    # import ipdb; ipdb.set_trace()
                    observation["estimated_target_pos"] = observation["ee_pos"][:3]

        return observation

    def mouse_callback(self, event, x, y, flags, param):
        global clicked_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)
            print(f"Clicked at pixel coordinates: ({x}, {y})")


    def init_camera(self):
        """Initialize the camera for object targeting."""
        # Load calibration matrix
        # make the path relative to the gym-lowcostrobot directory
        calibration_path = os.path.join(os.path.dirname(__file__), '../../', 'results', 'calibration_matrix.npy')
        if not os.path.exists(calibration_path):
            raise RuntimeError(f"Calibration matrix not found at {calibration_path}. Using camera without calibration.")
            # print(f"Calibration matrix not found at {calibration_path}. Using camera without calibration.")
            # self.T_base_camera = None
        else:
            self.T_base_camera = np.load(calibration_path)
            print(f"Loaded calibration matrix:\n{self.T_base_camera}")
        
        # Load color segmentation config 
        self.color_segmentation_config = None
        color_segmentation_config_path = os.path.join(os.path.dirname(__file__), '../../', 'color_segmentation_results/', 'color_segmentation_config.yaml')
        if not os.path.exists(color_segmentation_config_path):
            raise RuntimeError(f"Color segmentation config not found at {color_segmentation_config_path}. Using camera without color segmentation.")
        else:
            with open(color_segmentation_config_path, 'r') as f:
                self.color_segmentation_config = yaml.safe_load(f)
        
        # Initialize camera
        self.camera = D455Camera(
            enable_rgb=True,
            enable_depth=True,
            rgb_resolution=(848, 480),
            depth_resolution=(848, 480),
            fps=30,
            align_frames=True,
            enable_filters=True
        )
        
        if self.T_base_camera is not None:
            self.camera.set_calibration_matrix(self.T_base_camera)
        
        if not self.camera.start():
            raise RuntimeError("Failed to start camera. Cannot proceed without camera for object targeting.")
    
    def get_target_from_user(self):
        """Show camera feed and let user click on the object to set target position.
        
        Raises:
            RuntimeError: If camera is not available or target cannot be detected
        """
        global clicked_point, rgb_frame, depth_frame
        
        if self.camera is None:
            raise RuntimeError("Camera not available. Cannot proceed without camera for object targeting.")
        
        clicked_point = None
        cv2.namedWindow("Click on the object")
        cv2.setMouseCallback("Click on the object", self.mouse_callback)
        
        print("\n=== Camera-Based Object Targeting ===")
        print("Click on the object in the camera view")
        print("Press 'q' to cancel")
        print("======================================\n")
        
        while clicked_point is None:
            rgb_frame, depth_frame = self.camera.get_frames()
            
            if rgb_frame is None or depth_frame is None:
                print("Failed to get frames")
                time.sleep(0.1)
                continue
            
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            combined = np.hstack((rgb_frame, depth_colormap))
            cv2.putText(combined, 
                "Click on the object to set target position",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Click on the object", combined)
            
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                cv2.destroyWindow("Click on the object")
                raise RuntimeError("User canceled target selection.")
        
        # User clicked on the object
        x, y = clicked_point
        cv2.destroyWindow("Click on the object")
        
        # Check if click is in the RGB frame (left half of combined image)
        if x < rgb_frame.shape[1]:
            point_base = self.camera.get_3d_point_robot_base(x, y, depth_frame)
            
            if point_base is not None:
                print(f"3D target point in robot base frame: {point_base}")
                return point_base
        
        raise RuntimeError("Invalid depth at clicked point or point outside RGB frame. Cannot determine target position.")

    def move_robot_to_pre_camera_position(self):
        real_qpos_pre_camera = self.radian_to_position(self.initial_qpos_before_camera)
        real_qpos_pre_camera[2] += self.motor_3_bias
        # print(f'Moving robot to pre-camera position: {real_qpos_pre_camera}')
        self.realrobot.set_goal_pos(real_qpos_pre_camera)
        time.sleep(ACTION_SLEEP_SEC)


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)
        self.target = None
        observation = {}
        observation['gripper_blocked'] = np.zeros((1,), dtype=np.float32)
        

        # Reset the robot to the initial position and sample the cube position
        cube_pos = self.np_random.uniform(self.cube_low, self.cube_high)
        cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
        # robot_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #### CHANGING THIS POSITION IN WHAT WAS USED IN REAL ROBOT TESTING
        # robot_qpos = np.array([0, 0, 0, -1.44, -1.57, -1.5])
        qpos = self.initial_qpos
        real_qpos = self.radian_to_position(qpos)

        # add naive grav comp term
        real_qpos[2] += self.motor_3_bias
        real_qpos = np.clip(real_qpos, self.real_qpos_min, self.real_qpos_max)
        self.realrobot.set_goal_pos(real_qpos)
        time.sleep(ACTION_SLEEP_SEC)
        try:
            real_qpos = self.realrobot.read_position()
            qpos = np.asarray(self.position_to_radian(real_qpos), dtype=np.float32)
        except Exception as e:
            print(f"Reset: Failed to read position: {e}")
            raise e
        self.last_qpos = np.asarray(qpos.copy(), dtype=np.float32)
        # Set a better resting position initially
        # robot_qpos = np.array([0.0,  7.30210626e-01,  1.37570755e+00,  1.60038381e-01,\
        #     1.64550541e+00, -1.30162992e+00])
        self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof] = qpos
        self.data.qpos[self.cube_dof_id:self.cube_dof_id+7] = np.concatenate([cube_pos, cube_rot])
        self.data.qvel[:] = 0


        # nullspace action space setup
        model = self.model
        self.dof_ids = np.array([model.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array([model.actuator(name).id for name in self.joint_names])
        self.q0 = np.array([0,0,0,-1.44,-1.57,-1.5])

        # Integration timestep in seconds. This corresponds to the amount of time the joint
        # velocities will be integrated for to obtain the desired joint positions.
        self.integration_dt: float = 1.0

        # Damping term for the pseudoinverse. This is used to prevent joint velocities from
        # becoming too large when the Jacobian is close to singular.
        self.damping: float = 1e-4

        # Gains for the twist computation. These should be between 0 and 1. 0 means no
        # movement, 1 means move the end-effector to the target in one integration step.
        # self.Kpos: float = 0.95
        # self.Kori: float = 0.95
        self.Kpos: float = 1.0
        self.Kori: float = 1.0

        # Nullspace P gain.
        self.Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 10.0, 0.0])
        self.Kn *= 100

        # Maximum allowable joint velocity in rad/s.
        # self.max_angvel = 0.785
        self.max_angvel = 100

        self.jac = np.zeros((6, 6))
        self.diag = self.damping * np.eye(6)
        self.eye = np.eye(6)
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        robot_observation = self.get_observation()
        observation["log_is_success"] = np.zeros((1,), dtype=np.float32)
        observation['gripper_blocked'] = np.zeros((1,), dtype=np.float32)
        observation.update(robot_observation)
        if self.use_camera:
            # Get camera frames
            # rgb_frame, depth_frame = self.camera.get_frames()
            # if rgb_frame is None or depth_frame is None:
            #     raise RuntimeError("Failed to get camera frames during reset")
            # self.move_robot_to_pre_camera_position()
            if self.use_auto_target:
                input("Using auto target. Press Enter to continue...")
                observation = self.get_camera_observations(observation)
                observation['target_eepos'] = observation['estimated_target_pos']
                self.target = observation['target_eepos']
                print(f"Auto target: {self.target}")
            else:
                time.sleep(1)
                self.target = self.get_target_from_user()
                print(f"Target: {self.target}")
                observation = self.get_camera_observations(observation)
                observation['target_eepos'] = self.target
 
        info = {
            'qpos': qpos,
            'real_qpos': real_qpos,
        }
        return observation, info

    def step(self, action):
        # Perform the action and step the simulation
        gripper_before_action = self.last_qpos[-1]
        action_info = self.apply_action(action)
        gripper_displacement = action_info['raw_action'][-1]
        # Get the new observation
        observation = self.get_observation()
        gripper_after_action = observation['arm_qpos'][-1]  

        # Check if the gripper is holding the object, by comparing the expected gripper position
        # after the action and the actual gripper position after the action.
        expected_gripper_pos = gripper_before_action + gripper_displacement
        # clip the expected gripper pos to the limits
        expected_gripper_pos = np.clip(expected_gripper_pos, self.qpos_min[-1], self.qpos_max[-1])
        gripper_blocked = np.abs(gripper_after_action - expected_gripper_pos) > 0.02
        observation['gripper_blocked'] = np.array([float(gripper_blocked)], dtype=np.float32)
        reward = 0.0
        if self.use_camera:
            observation['target_eepos'] = self.target
            observation = self.get_camera_observations(observation)
            goal_point = observation['estimated_target_pos'] + np.array([0.01, 0, 0])
            # add reward for proximity to target and gripper blocked
            # 0 to 0.1. 
            dist_obj_robot = np.linalg.norm(observation['ee_pos'][:3] - goal_point[:3])
            # now print out dx dy dz to debug
            print(f"dx: {observation['ee_pos'][0] - goal_point[0]}, dy: {observation['ee_pos'][1] - goal_point[1]}, dz: {observation['ee_pos'][2] - goal_point[2]}")
            # print(f"dist_obj_robot: {dist_obj_robot}")
            dist_rew_term = np.clip(dist_obj_robot, 0, 0.1) * -2
            gripper_block_term = observation['gripper_blocked'] * 2.0
            reward += dist_rew_term + gripper_block_term

        # print('gripper before action', gripper_before_action)
        # print('gripper displacement', gripper_displacement)
        # print('gripper after action', gripper_after_action)
        # print('expected gripper pos', expected_gripper_pos)
        # print('gripper blocked', gripper_blocked)

        # print(f"gripper blocked: {gripper_blocked}, ee_pos: {observation['ee_pos']}")
        success = gripper_blocked and observation['ee_pos'][2] >= 0.05
        observation["log_is_success"] = np.array([float(success)], dtype=np.float32) 
        reward += float(success) * 100
        reward = reward.item()
        terminated = success
        truncated = False
        info = {
            'dist_rew_term': dist_rew_term,
            'gripper_block_term': gripper_block_term,
        }
        print(f"reward: {reward}, dist:{dist_obj_robot:.2f}, rew_dist: {dist_rew_term:.2f}, rew_block: {gripper_block_term}")
        info["qpos"] = self.data.qpos.copy()
        info["qvel"] = self.data.qvel.copy()
        info.update(action_info)
        return observation, reward, terminated, truncated, info


    def render(self):
        if self.render_mode == "human":
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # self.rgb_array_renderer.update_scene(self.data, camera="camera_vizu")
            self.renderer.update_scene(self.data, camera="camera_front")
            front_img = self.renderer.render()
            self.renderer.update_scene(self.data, camera="camera_wrist")
            wrist_img = self.renderer.render()
            # concatenate the images.
            combined_img = np.concatenate([wrist_img, front_img], 1)
            return combined_img

    def close(self):
        for renderer in ["viewer", "renderer", "rgb_array_renderer"]:
            if hasattr(self, renderer):
                getattr(self, renderer).close()
    
    def get_ee_pos(self):
        ee_id = self.model.site("end_effector").id
        ee_pos = np.array([0.0, 0.0, 0.0, 0.0])
        ee_pos[:3] = self.data.site_xpos[ee_id]
        ee_pos[-1] = self.data.qpos[self.arm_dof_id+self.nb_dof-1]
        return ee_pos.copy()

    def get_cube_pos(self):
        cube_pos = self.data.qpos[self.cube_dof_id:self.cube_dof_id+3]
        return cube_pos.copy()

# Global variable to store the latest gamepad action
latest_gamepad_action = np.array([0.0, 0.0, 0.0, 0.0])
reset = False

async def receive_gamepad_input():
    """
    WebSocket client to receive gamepad inputs from the browser.
    """
    global latest_gamepad_action
    global reset
    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("Connected to WebSocket server.")
            while True:
                message = await websocket.recv()
                # print("Received message from server:", message)  # Log the received message
                inputs = json.loads(message)
                axes = inputs.get("axes", [0.0] * 4)
                buttons = inputs.get("buttons", [0.0] * 16)

                # Map axes to X, Y, Z displacements
                x_displacement = axes[0] * 0.2  # Left joystick X axis
                y_displacement = -axes[1] * 0.2  # Left joystick Y axis (inverted)
                z_displacement = -axes[3] * 0.2  # Right joystick Y axis (inverted)

                # Map buttons to gripper control (e.g., right trigger - left trigger)
                gripper_action = (buttons[7] - buttons[6]) * 1.0

                # Update the latest gamepad action
                latest_gamepad_action = np.array([-y_displacement, x_displacement, z_displacement, gripper_action])

                reset = buttons[9] == 1

                # print("Updated gamepad action:", latest_gamepad_action)  # Log the updated action
    except Exception as e:
        print(f"WebSocket connection error: {e}")

def get_gamepad_action():
    """
    Get the latest gamepad action.
    """
    return latest_gamepad_action, reset

def start_gamepad_listener():
    """
    Start the WebSocket client to listen for gamepad inputs.
    """
    print("Starting gamepad listener thread...")
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(receive_gamepad_input())

def transform_cropped_to_full_image(pixel_x: int, pixel_y: int, x1: int, y1: int) -> Tuple[int, int]:
    """Transform the x, y position of a pixel in the cropped image to the x, y position of the pixel in the full image."""
    return pixel_x + x1, pixel_y + y1

if __name__ == "__main__":
    import threading

    # Start the gamepad listener in a separate thread
    gamepad_thread = threading.Thread(target=start_gamepad_listener, daemon=True)
    gamepad_thread.start()

    np.set_printoptions(precision=4, suppress=True)
    env = LiftCubeStateRealEnv(observation_mode="both",render_mode="human", action_mode="nullspace")
        
    key_action_map = {
        'w': np.array([-1.0, 0.0, 0.0, 0.0]),
        's': np.array([1.0, 0.0, 0.0, 0.0]),
        'a': np.array([0.0, -1.0, 0.0, 0.0]),
        'd': np.array([0.0, 1.0, 0.0, 0.0]),
        'q': np.array([0.0, 0.0, 1.0, 0.0]),
        'e': np.array([0.0, 0.0, -1.0, 0.0]),
        'z': np.array([0.0, 0.0, 0.0, 1.0]), # close gripper
        'x': np.array([0.0, 0.0, 0.0, -1.0]),# open gripper
    }
    pos_sensitivity = 0.1
    gripper_sensitivity = 1.0
    # obs, info = env.reset()
    # # print(f"qpos: {info['qpos']}")
    # print(f"eef pos: {obs['ee_pos']}")
    # env.render()
    # while True:
    #     raw_key = input("Enter action: ")
    #     if raw_key in key_action_map:
    #         action = key_action_map[raw_key].copy()
    #         action[:3] *= pos_sensitivity
    #         action[-1] *= gripper_sensitivity
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         print(f"goal pos: {info['goal_pos']}")
    #         print(f"eef pos: {obs['ee_pos']}")
    #         # print(f"Reward: {reward}")
    #         env.render()
    #         if terminated:
    #             print("Terminated")
    #             break
    #     else:
    #         break
    # env.close()
    while True:
        obs, info = env.reset()
        # print(f"qpos: {info['qpos']}")
        # print(f"eef pos: {obs['ee_pos']}")
        env.render()
        while True:
            # raw_key = input("Enter action: ")
            # if raw_key in key_action_map:
            # action = key_action_map[raw_key].copy()
            # action[:3] *= pos_sensitivity
            # action[-1] *= gripper_sensitivity
            action, reset = get_gamepad_action()
            # if any action is close to zero, set it to zero
            action = np.where(np.abs(action) < 0.01, 0.0, action)
            print("Gamepad action:", action)
            # print(action)
            obs, reward, terminated, truncated, info = env.step(action)
            # print(f"goal pos: {info['goal_pos']}")
            # print(f"eef pos: {obs['ee_pos']}")
            # print(f"Reward: {reward}")
            env.render()
            if terminated or reset:
                print("Terminated or reset")
                break
            # else: break
    env.close()
