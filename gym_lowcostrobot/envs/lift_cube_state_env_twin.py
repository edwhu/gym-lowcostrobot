import os
import time
from collections import deque
import warnings
# Filter out Gymnasium environment registration warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.envs.registration')

import gymnasium as gym
from gymnasium import Env, spaces
import mujoco
import mujoco.viewer
import numpy as np

from gym_lowcostrobot import ASSETS_PATH, BASE_LINK_NAME
from gym_lowcostrobot.envs.dynamixel import Dynamixel
from gym_lowcostrobot.envs.robot import Robot

# Constants
MOTOR_3_BIAS = 30
ACTION_SLEEP_SEC = 1.0
THRESHOLD_HEIGHT = 0.07
INTERPOLATION_STEPS = 10

class LiftCubeStateEnv(Env):
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

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, observation_mode="state", action_mode="nullspace", render_mode=None, render_obs=True, include_initial_obj_pose=False, use_action_noise=False):
        self._initialize_dynamixel()
        self._initialize_mujoco()
        self._initialize_action_space(action_mode)
        self._initialize_observation_space(observation_mode, include_initial_obj_pose)
        self._initialize_renderer(render_mode, render_obs)
        self._initialize_task_variables(use_action_noise)
    
    def _initialize_dynamixel(self):
        """Initialize the Dynamixel hardware interface"""
        self.real_qpos_min = [906, 0, 0, 0, 0, 2040]  # Position limits in Dynamixel units
        self.real_qpos_max = [3202, 4095, 4095, 4095, 4095, 2877]
        self.qpos_min = self.position_to_radian(self.real_qpos_min)
        self.qpos_max = self.position_to_radian(self.real_qpos_max)
        
        # if 'KOCH_DEVICE_NAME' not in os.environ:
        #     raise ValueError("Please set the KOCH_DEVICE_NAME environment variable to the serial port of the Dynamixel device")
        
        DEVICE_NAME = 'COM6' # os.environ['KOCH_DEVICE_NAME']
        self.dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name=DEVICE_NAME).instantiate()
        self.realrobot = Robot(self.dynamixel)
        
        # Set via manual calculation
        self.initial_qpos = np.array([-0.017867, 0.005605, -0.131519, -1.433267, 1.552938, 0.8])
        
        # dynamixel_values = [2100, 1700, 1800, 2000, 3200, 3000]

        # self.initial_qpos = self.position_to_radian(dynamixel_values)

        
        assert np.all(self.initial_qpos >= self.qpos_min) and np.all(self.initial_qpos <= self.qpos_max)

        # Initialize robot position
        real_qpos = self.radian_to_position(self.initial_qpos)
        real_qpos[2] += MOTOR_3_BIAS
        real_qpos = np.clip(real_qpos, self.real_qpos_min, self.real_qpos_max)
        self.realrobot.set_goal_pos(real_qpos)
        time.sleep(ACTION_SLEEP_SEC)

    def _initialize_mujoco(self):
        """Initialize the MuJoCo simulation"""
        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS_PATH, "lift_cube_camera.xml"), {})
        self.data = mujoco.MjData(self.model)
        
        # Enable gravity compensation. Set to 0.0 to disable.
        gravity_compensation = False
        for body in ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]:
            body_id = self.model.body(body).id
            self.model.body_gravcomp[body_id] = float(gravity_compensation)
        self.model.body_gravcomp[:] = float(gravity_compensation)
        
        self.nb_dof = 6
    
    def _initialize_action_space(self, action_mode):
        """Initialize the action space based on the specified mode"""
        self.action_mode = action_mode
        action_shape = {"joint": 6, "ee": 4, "nullspace": 4}[action_mode]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_shape,), dtype=np.float32)
        
        # Used for bounding the nullspace controller
        self.action_min = np.array([-0.3, -0.3, -0.3, -0.3])
        self.action_max = np.array([0.3, 0.3, 0.3, 0.3])

    def _initialize_observation_space(self, observation_mode, include_initial_obj_pose):
        """Initialize the observation space based on the specified mode"""
        self.observation_mode = observation_mode
        self.observation_subspaces = {
            "arm_qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(6,)),
            "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),
            "log_is_success": spaces.Box(low=-np.inf, high=np.inf, dtype="float32"),
        }
        
        self.include_initial_obj_pose = include_initial_obj_pose
        if include_initial_obj_pose:
            self.initial_obj_pose = np.zeros((7,), dtype=np.float32)
            self.observation_subspaces["initial_obj_pose"] = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))

        if self.observation_mode in ["image", "both"]:
            self.observation_subspaces["log_image_front"] = spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
            self.renderer = mujoco.Renderer(self.model, height=256, width=256)

        self.observation_space = gym.spaces.Dict(self.observation_subspaces)
    
    def _initialize_renderer(self, render_mode, render_obs):
        """Initialize the renderer based on the specified mode"""
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
    
    def _initialize_task_variables(self, use_action_noise):
        """Initialize variables used for the task"""
        # Cube position boundaries
        self.cube_low = np.array([-2.14, -0.03, 0.01])
        self.cube_high = np.array([-2.08, 0.03, 0.01])

        # Get DOF addresses
        self.cube_dof_id = self.model.body("cube").dofadr[0]
        self.arm_dof_id = self.model.body(BASE_LINK_NAME).dofadr[0]
        self.arm_dof_vel_id = self.arm_dof_id
        
        # If the arm is not at address 0 then the cube will have 7 states in qpos and 6 in qvel
        if self.arm_dof_id != 0:
            self.arm_dof_id = self.arm_dof_vel_id + 1
            
        self.joint_names = [f"joint_{i}" for i in range(1, 7)]

        # Domain randomization variables
        self._dr_noise = {
            # Add onto data.ctrl, range is in radians
            "joint_ctrl_min": np.array([-0.01, -0.01, -0.1, -0.01, -0.01, -0.01]),
            "joint_ctrl_max": np.array([0.01, 0.01, 0.001, 0.01, 0.01, 0.01]),
        }
        
        if not use_action_noise:
            self._dr_noise = {k: np.zeros_like(v) for k, v in self._dr_noise.items()}

        # Koch 1.1的工作空间限制
        self.ee_min = np.array([-0.5, -0.5, 0.01])  # 更大的工作空间
        self.ee_max = np.array([0.5, 0.5, 0.5])

    def radian_to_position(self, values):
        """Convert radian values to Dynamixel position values"""
        scale = 4095 / (3.14 - (-3.14))
        return [round((v + 3.14) * scale) for v in values]

    def position_to_radian(self, positions):
        """Convert Dynamixel position values to radian values"""
        scale = 4095 / (3.14 - (-3.14))
        return [(p / scale) - 3.14 for p in positions]

    def diffik_nullspace(self, goal_pos, goal_quat, site_id):
        """
        Differential inverse kinematics with nullspace control
        
        Args:
            goal_pos: Target position for the end effector
            goal_quat: Target orientation for the end effector
            site_id: Site ID for the end effector
            
        Returns:
            Target joint positions
        """
        model = self.model
        data = self.data
        
        # Spatial velocity (aka twist)
        dx = goal_pos - data.site(site_id).xpos
        self.twist[:3] = self.Kpos * dx / self.integration_dt
        
        mujoco.mju_mat2Quat(self.site_quat, data.site(site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, goal_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= self.Kori / self.integration_dt

        # Jacobian
        temp_jac = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, temp_jac[:3], temp_jac[3:], site_id)
        self.jac = temp_jac[:, self.dof_ids]

        # Damped least squares
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.twist)

        # Nullspace control biasing joint velocities towards the home configuration
        dq += (self.eye - np.linalg.pinv(self.jac) @ self.jac) @ (self.Kn * (self.q0 - data.qpos[self.dof_ids]))

        # Clamp maximum joint velocity
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > self.max_angvel:
            dq *= self.max_angvel / dq_abs_max
            
        # Integrate joint velocities to obtain joint positions
        temp_dq = np.concatenate([dq, np.zeros(6)])
        q = data.qpos.copy()
        mujoco.mj_integratePos(model, q, temp_dq, self.integration_dt)
        q = q[self.dof_ids]
        np.clip(q, *model.jnt_range[self.dof_ids].T, out=q)
        
        return q

    def apply_action(self, action):
        """
        Apply the action to the robot
        
        Args:
            action: Action to apply
            
        Returns:
            Dictionary with information about the action
        """
        info = {}
        
        if self.action_mode == "nullspace":
            assert action.min() >= -5.0 and action.max() <= 5.0
            
            # Convert normalized action to raw action
            raw_action = self.get_raw_action(action)
            ee_action, gripper_action = raw_action[:3], raw_action[-1]
            
            # Calculate goal position
            goal_pos = ee_action + self.data.site("attachment_site").xpos
            goal_pos = np.clip(goal_pos, self.ee_min, self.ee_max)
            
            # Set goal orientation (pointing downwards)
            goal_quat = np.array([0.5, 0.5, 0.5, 0.5])
            site_id = self.model.site("attachment_site").id
            
            # Use inverse kinematics to get joint positions
            target_qpos = self.diffik_nullspace(goal_pos, goal_quat, site_id)
            
            # Apply gripper action
            target_qpos[-1:] += gripper_action
            target_real_qpos = self.radian_to_position(target_qpos)
            
        elif self.action_mode == "joint":
            target_qpos = np.array(action)
            target_real_qpos = self.radian_to_position(target_qpos)
        else:
            raise ValueError("Invalid action mode, must be 'nullspace' or 'joint'")

        # Apply action to the real robot with interpolation for smoother motion
        target_real_qpos[2] += MOTOR_3_BIAS
        target_real_qpos = np.clip(target_real_qpos, self.real_qpos_min, self.real_qpos_max)
        
        # Get current position for interpolation
        try:
            current_real_qpos = np.array(self.realrobot.read_position(), dtype=np.int32)
        except Exception as e:
            # If we can't read the position, use the last known position
            current_real_qpos = np.array(self.radian_to_position(self.last_qpos), dtype=np.int32)
        
        # Ensure target position is also integer array
        target_real_qpos = np.array(target_real_qpos, dtype=np.int32)
        
        # Interpolate between current and target position
        for i in range(INTERPOLATION_STEPS):
            # Linear interpolation
            alpha = (i + 1) / INTERPOLATION_STEPS
            interp_real_qpos = current_real_qpos * (1 - alpha) + target_real_qpos * alpha
            # Convert to integers and clip to valid range
            interp_real_qpos = np.round(interp_real_qpos).astype(np.int32)
            interp_real_qpos = np.clip(interp_real_qpos, self.real_qpos_min, self.real_qpos_max)
            
            # Send interpolated position to robot
            self.realrobot.set_goal_pos(interp_real_qpos.tolist())
            time.sleep(ACTION_SLEEP_SEC / INTERPOLATION_STEPS)
            
            if self.render_mode == "human":
                self.viewer.sync()
        
        # Store information about the action
        info = {
            'goal_pos': goal_pos if self.action_mode == "nullspace" else None,
            'target_qpos': target_qpos,
            'target_real_qpos': target_real_qpos,
            'raw_action': raw_action if self.action_mode == "nullspace" else action,
        }
        
        return info

    def get_scaled_action(self, raw_action):
        """Convert raw action to scaled action in range [-1, 1]"""
        scaled_min, scaled_max = -1, 1
        raw_action = np.clip(raw_action, self.action_min, self.action_max)
        scaled_action = (raw_action - self.action_min) / (self.action_max - self.action_min) * (scaled_max - scaled_min) + scaled_min
        return scaled_action
    
    def get_raw_action(self, scaled_action):
        """Convert scaled action in range [-1, 1] to raw action"""
        scaled_min, scaled_max = -1, 1
        scaled_action = np.clip(scaled_action, scaled_min, scaled_max)
        raw_action = (scaled_action - scaled_min) / (scaled_max - scaled_min) * (self.action_max - self.action_min) + self.action_min
        return raw_action

    def get_observation(self):
        """Get the current observation"""
        try:
            real_qpos = np.asarray(self.realrobot.read_position(), dtype=np.float32)
            qpos = np.array(self.position_to_radian(real_qpos), dtype=np.float32)
        except Exception as e: 
            print(f"Failed to read position:, using last known and valid position")
            qpos = np.array(self.last_qpos.copy(), dtype=np.float32)
            real_qpos = self.radian_to_position(qpos)

        # If positions are all within the limits, then cache them
        if np.all(real_qpos >= self.real_qpos_min) and np.all(real_qpos <= self.real_qpos_max):
            self.last_qpos = self.position_to_radian(real_qpos)
        else:
            print(f"Real robot qpos out of bounds: {real_qpos} ")
            print(f"You may need to tune the real_qpos_min and real_qpos_max values")
            qpos = np.clip(qpos, self.real_qpos_min, self.real_qpos_max)

        # Set the sim robot qpos to the real robot qpos
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

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed, options=options)
        
        # Sample cube position
        cube_pos = self.np_random.uniform(self.cube_low, self.cube_high)
        cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Set robot to initial position
        qpos = self.initial_qpos
        real_qpos = self.radian_to_position(qpos)
        real_qpos[2] += MOTOR_3_BIAS
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
        
        # Set simulation state
        self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof] = qpos
        self.data.qpos[self.cube_dof_id:self.cube_dof_id+7] = np.concatenate([cube_pos, cube_rot])
        self.data.qvel[:] = 0

        # Setup nullspace controller parameters
        model = self.model
        self.dof_ids = np.array([model.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array([model.actuator(name).id for name in self.joint_names])
        self.q0 = np.array([0, 0, 0, -1.44, -1.57, -1.5])

        # Integration timestep in seconds
        self.integration_dt = 1.0

        # Damping term for the pseudoinverse
        self.damping = 1e-4

        # Gains for the twist computation
        self.Kpos = 1.0
        self.Kori = 1.0

        # Nullspace P gain
        self.Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 10.0, 0.0]) * 100

        # Maximum allowable joint velocity in rad/s
        self.max_angvel = 100

        # Initialize controller variables
        self.jac = np.zeros((6, 6))
        self.diag = self.damping * np.eye(6)
        self.eye = np.eye(6)
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)
        
        # Get observation and info
        observation = self.get_observation()
        observation["log_is_success"] = np.zeros((1,), dtype=np.float32)
        
        info = {
            'qpos': qpos,
            'real_qpos': real_qpos,
        }
        
        # For telling if the object is in the hand
        self.gripper_history = deque(maxlen=4)
        self.gripper_history.append(False)
        
        return observation, info

    def step(self, action):
        """
        Step the environment
        
        Args:
            action: Action to apply
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Record gripper position before action
        gripper_before_action = self.last_qpos[-1]
        
        # Apply action
        action_info = self.apply_action(action)
        gripper_displacement = action_info['raw_action'][-1]
        
        # Get new observation
        observation = self.get_observation()
        gripper_after_action = observation['arm_qpos'][-1]  

        # Check if the gripper is holding the object
        expected_gripper_pos = gripper_before_action + gripper_displacement
        expected_gripper_pos = np.clip(expected_gripper_pos, self.qpos_min[-1], self.qpos_max[-1])
        gripper_blocked = np.abs(gripper_after_action - expected_gripper_pos) > 0.1

        # Determine success and reward
        if 0:
            success = gripper_blocked and observation['ee_pos'][2] >= THRESHOLD_HEIGHT
            reward = float(success) 
            terminated = success
            truncated = False
        else:
            success = False
            reward = 0.0
            terminated = False
            truncated = False
        
        # Prepare info dictionary
        info = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }
        info.update(action_info)

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data, camera="camera_front")
            front_img = self.renderer.render()
            self.renderer.update_scene(self.data, camera="camera_wrist")
            wrist_img = self.renderer.render()
            # Concatenate the images
            combined_img = np.concatenate([wrist_img, front_img], 1)
            return combined_img

    def close(self):
        """Close the environment"""
        for renderer in ["viewer", "renderer", "rgb_array_renderer"]:
            if hasattr(self, renderer):
                getattr(self, renderer).close()
    
    def get_ee_pos(self):
        """Get the end effector position"""
        ee_id = self.model.site("end_effector").id
        ee_pos = np.array([0.0, 0.0, 0.0, 0.0])
        ee_pos[:3] = self.data.site_xpos[ee_id]
        ee_pos[-1] = self.data.qpos[self.arm_dof_id+self.nb_dof-1]
        return ee_pos.copy()

    def get_cube_pos(self):
        """Get the cube position"""
        cube_pos = self.data.qpos[self.cube_dof_id:self.cube_dof_id+3]
        return cube_pos.copy()

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    env = LiftCubeStateEnv(observation_mode="both", render_mode="human", action_mode="nullspace", use_action_noise=False)
        
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
    
    obs, info = env.reset()
    print(f"eef pos: {obs['ee_pos']}")
    env.render()
    
    while True:
        raw_key = input("Enter action: ")
        if raw_key in key_action_map:
            action = key_action_map[raw_key].copy()
            action[:3] *= pos_sensitivity
            action[-1] *= gripper_sensitivity
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"goal pos: {info['goal_pos']}")
            print(f"eef pos: {obs['ee_pos']}")
            env.render()
            if terminated:
                print("Terminated")
                break
        else:
            break
    
    env.close()
