import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import Env, spaces
from collections import deque

from gym_lowcostrobot import ASSETS_PATH, BASE_LINK_NAME


class LiftCubeStateEnv(Env):
    """
    ## Description

    The robot has to lift a cube with its end-effector.

    ## Action space

    Two action modes are available: "joint" and "ee". In the "joint" mode, the action space is a 6-dimensional box
    representing the target joint angles.

    | Index | Action              | Type (unit) | Min  | Max |
    | ----- | ------------------- | ----------- | ---- | --- |
    | 0     | Shoulder pan joint  | Float (rad) | -1.0 | 1.0 |
    | 1     | Shoulder lift joint | Float (rad) | -1.0 | 1.0 |
    | 2     | Elbow flex joint    | Float (rad) | -1.0 | 1.0 |
    | 3     | Wrist flex joint    | Float (rad) | -1.0 | 1.0 |
    | 4     | Wrist roll joint    | Float (rad) | -1.0 | 1.0 |
    | 5     | Gripper joint       | Float (rad) | -1.0 | 1.0 |

    In the "ee" mode, the action space is a 4-dimensional box representing the target end-effector position and the
    gripper position.

    | Index | Action        | Type (unit) | Min  | Max |
    | ----- | ------------- | ----------- | ---- | --- |
    | 0     | X             | Float (m)   | -1.0 | 1.0 |
    | 1     | Y             | Float (m)   | -1.0 | 1.0 |
    | 2     | Z             | Float (m)   | -1.0 | 1.0 |
    | 5     | Gripper joint | Float (rad) | -1.0 | 1.0 |

    ## Observation space

    The observation space is a dictionary containing the following subspaces:

    - `"arm_qpos"`: the joint angles of the robot arm in radians, shape (6,)
    - `"arm_qvel"`: the joint velocities of the robot arm in radians per second, shape (6,)
    - `"image_front"`: the front image of the camera of size (240, 320, 3)
    - `"image_top"`: the top image of the camera of size (240, 320, 3)
    - `"cube_pos"`: the position of the cube, as (x, y, z)
    - `"ee_pos"`: the position of the ee, as (x, y, z)

    Three observation modes are available: "image" (default), "state", and "both".

    | Key             | `"image"` | `"state"` | `"both"` |
    | --------------- | --------- | --------- | -------- |
    | `"arm_qpos"`    | ✓         | ✓         | ✓        |
    | `"arm_qvel"`    | ✓         | ✓         | ✓        |
    | `"image_front"` | ✓         |           | ✓        |
    | `"image_top"`   | ✓         |           | ✓        |
    | `"cube_pos"`    |           | ✓         | ✓        |
    | `"ee_pos"`      |           | ✓         | ✓        |

    ## Reward

    The reward is the sum of two terms: the height of the cube above the threshold and the negative distance between the
    end effector and the cube.

    ## Arguments

    - `observation_mode (str)`: the observation mode, can be "image", "state", or "both", default is "image", see
        section "Observation space".
    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is None.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, observation_mode="state", action_mode="joint", render_mode=None, render_obs=True, include_initial_obj_pose=False):
        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS_PATH, "lift_cube_camera.xml"), {})
        self.data = mujoco.MjData(self.model)

        # Enable gravity compensation. Set to 0.0 to disable.
        gravity_compensation = True
        for body in ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]:
            body_id = self.model.body(body).id
            self.model.body_gravcomp[body_id] = float(gravity_compensation)


        self.model.body_gravcomp[:] = float(gravity_compensation)
        # self.dt: float = 0.002
        # self.model.opt.timestep = self.dt

        # Set the action space
        self.action_mode = action_mode
        action_shape = {"joint": 6, "ee": 4, "nullspace": 4}[action_mode]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_shape,), dtype=np.float32)
        # used for bounding the nullspace controller
        self.action_min = np.array([-0.1, -0.1, -0.1, -2.0])
        self.action_max = np.array([0.1, 0.1, 0.1, 11.0])

        self.nb_dof = 6

        # Set the observations space
        self.observation_mode = observation_mode
        self.observation_subspaces = {
            "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(13,)),
            "qvel": spaces.Box(low=-np.inf, high=np.inf, shape=(12,)),
            "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),
            "touch": spaces.Box(low=-10.0, high=10.0, shape=(2,)),
            "arm_qpos": spaces.Box(low=-np.pi, high=np.pi, shape=(6,)),
            "log_is_success": spaces.Box(low=-np.inf, high=np.inf, dtype="float32"),
        }
        self.include_initial_obj_pose = include_initial_obj_pose
        if include_initial_obj_pose:
            self.initial_obj_pose = np.zeros((7,), dtype=np.float32)
            self.observation_subspaces["initial_obj_pose"] = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))

        if self.observation_mode in ["image", "both"]:
            # observation_subspaces["image_wrist"] = spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
            self.observation_subspaces["log_image_front"] = spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
            self.renderer = mujoco.Renderer(self.model, height=256, width=256)


        self.observation_space = gym.spaces.Dict(self.observation_subspaces)

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

        # Set additional utils
        self.threshold_height = 0.07
        self.cube_low = np.array([-0.03, 0.08, 0.01])  # move the cube closer to the robot
        self.cube_high = np.array([0.03, 0.14, 0.01])

        # get dof addresses
        self.cube_dof_id = self.model.body("cube").dofadr[0]
        self.arm_dof_id = self.model.body(BASE_LINK_NAME).dofadr[0]
        self.arm_dof_vel_id = self.arm_dof_id
        # if the arm is not at address 0 then the cube will have 7 states in qpos and 6 in qvel
        if self.arm_dof_id != 0:
            self.arm_dof_id = self.arm_dof_vel_id + 1

        self.control_decimation = 40 # number of simulation steps per control step
        self.frames = deque(maxlen=3)

    def inverse_kinematics(self, ee_target_pos, step=0.2, joint_name="end_effector", nb_dof=6, regularization=1e-6):
        """
        Computes the inverse kinematics for a robotic arm to reach the target end effector position.

        :param ee_target_pos: numpy array of target end effector position [x, y, z]
        :param step: float, step size for the iteration
        :param joint_name: str, name of the end effector joint
        :param nb_dof: int, number of degrees of freedom
        :param regularization: float, regularization factor for the pseudoinverse computation
        :return: numpy array of target joint positions
        """
        try:
            # Get the site ID from the name
            site_id = self.model.site(joint_name).id
        except KeyError:
            raise ValueError(f"Site name '{joint_name}' not found in the model.")

        # Get the current end effector position from the site
        ee_pos = self.data.site_xpos[site_id]

        # Compute the Jacobian for the end effector site
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, site_id)

        # Compute the difference between target and current end effector positions
        delta_pos = ee_target_pos - ee_pos

        # Compute the pseudoinverse of the Jacobian with regularization
        jac_reg = jac[:, :nb_dof].T @ jac[:, :nb_dof] + regularization * np.eye(nb_dof)
        jac_pinv = np.linalg.inv(jac_reg) @ jac[:, :nb_dof].T

        # Compute target joint velocities
        qdot = jac_pinv @ delta_pos

        # Normalize joint velocities to avoid excessive movements
        qdot_norm = np.linalg.norm(qdot)
        if qdot_norm > 1.0:
            qdot /= qdot_norm

        # Read the current joint positions
        qpos = self.data.qpos[self.arm_dof_id:self.arm_dof_id + nb_dof]

        # Compute the new joint positions
        q_target_pos = qpos + qdot * step

        return q_target_pos

    def diffik_nullspace(
        self,
        goal_pos,
        goal_quat,
        site_id,
    ):
        model = self.model
        data = self.data
        # Spatial velocity (aka twist).
        # dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
        dx = goal_pos - data.site(site_id).xpos
        self.twist[:3] = self.Kpos * dx / self.integration_dt
        mujoco.mju_mat2Quat(self.site_quat, data.site(site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        # mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
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
        - EE mode: [dx, dy, dz, gripper]
        - Joint mode: [q1, q2, q3, q4, q5, q6, gripper]
        """
        if self.action_mode == "ee":
            if len(action) == 4:
                ee_action, gripper_action = action[:3], action[-1]

                # Update the robot position based on the action
                ee_id = self.model.site("end_effector").id
                # ee_target_pos = self.data.site_xpos[ee_id] + ee_action
                ee_target_pos = ee_action

                # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
                target_qpos = self.inverse_kinematics(ee_target_pos=ee_target_pos)
                target_qpos[-1:] = gripper_action
            else:
                target_low = np.array([-3.14159, -1.5708, -1.48353, -1.91986, -2.96706, -1.74533])
                target_high = np.array([3.14159, 1.22173, 1.74533, 1.91986, 2.96706, 0.0523599])
                target_qpos = np.array(action).clip(target_low, target_high)
        elif self.action_mode == "nullspace":
            # actions are relative and normalized to [-1, 1]
            raw_action = self.get_raw_action(action)
            ee_action, gripper_action = raw_action[:3], raw_action[-1]

            goal_pos = ee_action + self.data.site("attachment_site").xpos
            goal_quat = np.array([0.7071, 0.7071, 0, 0]) # rotate 90 on x axis to make gripper point downwards.

            site_id = self.model.site("attachment_site").id

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            target_qpos = self.diffik_nullspace(
                goal_pos,
                goal_quat,
                site_id,
            )
            target_qpos[-1:] += gripper_action


        elif self.action_mode == "joint":
            # target_low = np.array([-3.14159, -1.5708, -1.48353, -1.91986, -2.96706, -1.74533])
            # target_high = np.array([3.14159, 1.22173, 1.74533, 1.91986, 2.96706, 0.0523599])
            # target_qpos = np.array(action).clip(target_low, target_high)
            target_qpos = np.array(action)
        else:
            raise ValueError("Invalid action mode, must be 'ee' or 'joint'")

        # Set the target position
        self.data.ctrl = target_qpos
        info = {"target_qpos": target_qpos}

        # Step the simulation forward
        # for _ in range(self.control_decimation):
        #     mujoco.mj_step(self.model, self.data)
        #     if self.render_mode == "human":
        #         self.viewer.sync()
        mujoco.mj_step(self.model, self.data, self.control_decimation)
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
        observation = {
            "qpos": self.data.qpos.copy().astype(np.float32),
            "qvel": self.data.qvel.copy().astype(np.float32),
            "arm_qpos": self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof].copy().astype(np.float32),
            "ee_pos": self.get_ee_pos().astype(np.float32),
        }
        if self.include_initial_obj_pose:
            observation["initial_obj_pose"] = self.initial_obj_pose.copy().astype(np.float32)

        touch_left_finger = False
        touch_right_finger = False
        obj = "cube"
        l_finger_geom_id = self.model.geom("link_6_pad").id
        r_finger_geom_id = self.model.geom("link_5_pad").id
        for j in range(self.data.ncon):
            c = self.data.contact[j]
            body1 = self.model.geom_bodyid[c.geom1]
            body2 = self.model.geom_bodyid[c.geom2]
            body1_name = self.model.body(body1).name
            body2_name = self.model.body(body2).name

            if c.geom1 == l_finger_geom_id and body2_name == obj:
                touch_left_finger = True
            if c.geom2 == l_finger_geom_id and body1_name == obj:
                touch_left_finger = True

            if c.geom1 == r_finger_geom_id and body2_name == obj:
                touch_right_finger = True
            if c.geom2 == r_finger_geom_id and body1_name == obj:
                touch_right_finger = True
        observation["touch"] = np.array([int(touch_left_finger), int(touch_right_finger)]).astype(np.float32)

        if self.observation_mode in ["image", "both"]:
            if self.render_obs:
                self.rgb_array_renderer.update_scene(self.data, camera="camera_front")
                img = self.rgb_array_renderer.render()
            else:
                img = np.zeros((64, 64, 3), dtype=np.uint8)
            observation["log_image_front"] = img

            # if self.render_obs:
            #     # self.rgb_array_renderer.update_scene(self.data, camera="camera_front")
            #     # observation["image_front"] = self.rgb_array_renderer.render()
            #     self.rgb_array_renderer.update_scene(self.data, camera="camera_wrist")
            #     wrist_img = self.rgb_array_renderer.render()
            #     # convert rgb to grayscale
            #     wrist_img = np.dot(wrist_img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            # else:
            #     # observation["image_front"] = np.zeros((84, 84, 3), dtype=np.uint8)
            #     wrist_img = np.zeros((84, 84), dtype=np.uint8)
            # if len(self.frames) == 0:
            #     self.frames.append(wrist_img)
            #     self.frames.append(wrist_img)

            # self.frames.append(wrist_img)
            # # stack the frames together into 84,84,3
            # wrist_frames = np.stack(self.frames, axis=-1)
            # observation["image_wrist"] = wrist_frames

        return observation

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)
        
        if options is None:
            # Reset the robot to the initial position and sample the cube position
            cube_pos = self.np_random.uniform(self.cube_low, self.cube_high)
            cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
            # robot_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            robot_qpos = np.array([0, 0, 0, -1.44, -1.57, -1.5])

            # Set a better resting position initially
            # robot_qpos = np.array([0.0,  7.30210626e-01,  1.37570755e+00,  1.60038381e-01,\
            #     1.64550541e+00, -1.30162992e+00])
            self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof] = robot_qpos
            self.data.qpos[self.cube_dof_id:self.cube_dof_id+7] = np.concatenate([cube_pos, cube_rot])
            self.data.qvel[:] = 0
        else:
            self.data.qpos = options["qpos"].copy()
            self.data.qvel = options["qvel"].copy()
        
        if self.include_initial_obj_pose:
            self.initial_obj_pose = self.data.qpos[self.cube_dof_id:self.cube_dof_id+7].copy()

        # nullspace action space setup
        model = self.model
        joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
        self.dof_ids = np.array([model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([model.actuator(name).id for name in joint_names])
        self.q0 = np.array([0,0,0,-1.44,-1.57,-1.5])

        # Integration timestep in seconds. This corresponds to the amount of time the joint
        # velocities will be integrated for to obtain the desired joint positions.
        self.integration_dt: float = 0.1

        # Damping term for the pseudoinverse. This is used to prevent joint velocities from
        # becoming too large when the Jacobian is close to singular.
        self.damping: float = 1e-4

        # Gains for the twist computation. These should be between 0 and 1. 0 means no
        # movement, 1 means move the end-effector to the target in one integration step.
        self.Kpos: float = 0.95
        self.Kori: float = 0.95


        # Nullspace P gain.
        self.Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 10.0, 0.0])
        # self.Kn /= 100.0

        # Maximum allowable joint velocity in rad/s.
        self.max_angvel = 0.785

        self.jac = np.zeros((6, 6))
        self.diag = self.damping * np.eye(6)
        self.eye = np.eye(6)
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)


        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        self.frames.clear()
        
        observation = self.get_observation()
        observation["log_is_success"] = np.zeros((1,), dtype=np.float32)
        # info = {'image_front': observation['image_front']}
        info = {'qpos': self.data.qpos.copy()}
        return observation, info

    def step(self, action):
        # Perform the action and step the simulation
        action_info = self.apply_action(action)

        # Get the new observation
        observation = self.get_observation()

        # Get the position of the cube and the distance between the end effector and the cube
        cube_pos = self.data.qpos[self.cube_dof_id:self.cube_dof_id+3]
        cube_z = cube_pos[2]
        ee_id = self.model.site("end_effector").id
        ee_pos = self.data.site_xpos[ee_id]
        ee_to_cube = np.linalg.norm(ee_pos - cube_pos)

        terminated = cube_z >= self.threshold_height and ee_to_cube < 0.05
        observation["log_is_success"] = np.ones((1,), dtype=np.float32) * terminated
        reward = 0
        if terminated:
            msg = "success phase"
            reward = 300
        else:
            dist = ee_to_cube
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward
            msg = "reaching phase"

            # grasping reward
            if observation["touch"].all():
                reward += 0.25
                dist = np.abs(cube_z - self.threshold_height)
                picking_reward = 1 - np.tanh(10.0 * dist)
                reward += picking_reward
                msg = "picking phase"

        # print(f"{msg}: {reward}, {observation['touch'].all()}")
        # penalize closed gripper when not close to the cube.
        is_close = ee_to_cube < 0.05
        gripper_closing = self.data.qpos[self.arm_dof_id+self.nb_dof-1] >= -1.5
        gripper_penalty = 0.5 * gripper_closing * np.tanh(10 * ee_to_cube) * ~is_close

        # penalize noisy actions using action norm
        action_penalty = 0.1 * np.linalg.norm(action)

        # print(f"task reward: {reward:.3f}, gripper_penalty: {gripper_penalty:.3f}, ee_to_cube: {ee_to_cube:.3f}, action_penalty: {action_penalty:.3f}")

        reward -= gripper_penalty
        reward -= action_penalty

        info = {}
        # Store the correct (x,y,z,gripper_joint) action that WOULD have been taken
        ee_id = self.model.site("end_effector").id
        action_ee = np.array([0.0, 0.0, 0.0, 0.0])
        action_ee[:3] = self.data.site_xpos[ee_id]
        action_ee[-1] = self.data.qpos[self.arm_dof_id+self.nb_dof-1]

        info["action_ee"] = action_ee
        info["qpos"] = self.data.qpos.copy()
        info["qvel"] = self.data.qvel.copy()
        info["target_qpos"] = action_info["target_qpos"]

        return observation, reward, terminated, False, info


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

class LiftCubeStateDreamerV4Env(LiftCubeStateEnv):
    # modify observation space to just have arm_qpos
    # and rename log_image_front to log/image_front
    def __init__(self, observation_mode="state", action_mode="joint", render_mode=None, render_obs=True, include_initial_obj_pose=False):
        super().__init__(observation_mode, action_mode, render_mode, render_obs, include_initial_obj_pose)
        self.observation_subspaces = {k: v for k, v in self.observation_subspaces.items() if k in ["arm_qpos", "log_image_front", "log_is_success"]}
        if "log_image_front" in self.observation_subspaces:
            self.observation_subspaces["log/image_front"] = self.observation_subspaces.pop("log_image_front")
        self.observation_space = gym.spaces.Dict(self.observation_subspaces)
    
    def observation(self, observation):
        # pop everything except for keys in self.observation_space
        new_observation = {k: v for k, v in observation.items() if k in self.observation_space.spaces.keys()}
        new_observation["log/is_success"] = observation.pop("log_is_success")
        if "log_image_front" in observation and "log/image_front" in self.observation_space.spaces.keys():
            new_observation["log/image_front"] = observation.pop("log_image_front")
        return new_observation
    
    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        return self.observation(observation), info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return self.observation(observation), reward, terminated, truncated, info


if __name__ == "__main__":
    # env = LiftCubeStateEnv(observation_mode="both",render_mode="human")
    env = LiftCubeStateDreamerV4Env(observation_mode="both",render_mode="human")
    while True:
        obs, info = env.reset()
        import ipdb; ipdb.set_trace()
        env.render()
    # env.reset()
    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    env.close()