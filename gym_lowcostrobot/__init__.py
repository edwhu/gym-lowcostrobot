import os

from gymnasium.envs.registration import register

__version__ = "0.0.1"

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets", "low_cost_robot_6dof")
BASE_LINK_NAME = "link_1"

def register_robotics_envs():
    register(
        id="LiftCube-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeEnv",
        max_episode_steps=500,
    )

    register(
        id="LiftCubeCamera-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeCameraEnv",
        max_episode_steps=500,
    )

    register(
        id="LiftCubeCameraPrivileged-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeCameraEnv",
        max_episode_steps=50,
        kwargs={"observation_mode":"both", "action_mode":"nullspace", "use_action_noise": False, "render_mode":"rgb_array"}
    )

    # for each environment, we have 3 types
    # the normal environment for training
    # the evaluation environment with render mode set to rgb array
    # the desktop visualization environment with render mode set to human
    register(
        id="LiftCubeState-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeStateEnv",
        max_episode_steps=100,
        kwargs={"observation_mode":"state", "action_mode":"nullspace"}
    )
    register(
        id="LiftCubeStateEval-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeStateEnv",
        max_episode_steps=100,
        kwargs={"observation_mode":"both", "action_mode":"nullspace", "render_mode":"rgb_array"}
    )
    register(
        id="LiftCubeStateHumanRender-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeStateEnv",
        max_episode_steps=100,
        kwargs={"observation_mode":"both", "action_mode":"nullspace", "render_mode":"human"}
    )

    # action noise variant
    register(
        id="LiftCubeStateNoisy-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeStateEnv",
        max_episode_steps=100,
        kwargs={"observation_mode":"state", "action_mode":"nullspace", "use_action_noise":True}
    )
    register(
        id="LiftCubeStateNoisyEval-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeStateEnv",
        max_episode_steps=100,
        kwargs={"observation_mode":"both", "action_mode":"nullspace", "render_mode":"rgb_array", "use_action_noise":True}
    )
    register(
        id="LiftCubeStateNoisyHumanRender-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeStateEnv",
        max_episode_steps=100,
        kwargs={"observation_mode":"both", "action_mode":"nullspace", "render_mode":"human", "use_action_noise":True}
    )

    register(
        id="LiftCubeStateOpenLoop-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeStateEnv",
        max_episode_steps=100,
        kwargs={"observation_mode":"state", "action_mode":"nullspace", "include_initial_obj_pose":True}
    )
    register(
        id="LiftCubeStateOpenLoopEval-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeStateEnv",
        max_episode_steps=100,
        kwargs={"observation_mode":"both", "action_mode":"nullspace", "render_mode":"rgb_array", "include_initial_obj_pose":True}
    )
    register(
        id="LiftCubeStateOpenLoopHumanRender-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeStateEnv",
        max_episode_steps=100,
        kwargs={"observation_mode":"both", "action_mode":"nullspace", "render_mode":"human", "include_initial_obj_pose":True}
    )

    register(
        id="LiftCubeStateDreamerV4-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeStateDreamerV4Env",
        max_episode_steps=100,
        kwargs={"observation_mode":"state", "action_mode":"nullspace"}
    )
    # used for eval, obs dict contains an image for visualization.
    register(
        id="LiftCubeStateDreamerV4Eval-v0",
        entry_point="gym_lowcostrobot.envs:LiftCubeStateDreamerV4Env",
        max_episode_steps=100,
        kwargs={"observation_mode":"both", "action_mode":"nullspace", "render_mode":"rgb_array"}
    )



    register(
        id="PickPlaceCube-v0",
        entry_point="gym_lowcostrobot.envs:PickPlaceCubeEnv",
        max_episode_steps=500,
    )

    register(
        id="PushCube-v0",
        entry_point="gym_lowcostrobot.envs:PushCubeEnv",
        max_episode_steps=500,
    )

    register(
        id="ReachCube-v0",
        entry_point="gym_lowcostrobot.envs:ReachCubeEnv",
        max_episode_steps=500,
    )

    register(
        id="StackTwoCubes-v0",
        entry_point="gym_lowcostrobot.envs:StackTwoCubesEnv",
        max_episode_steps=500,
    )

    register(
        id="PushCubeLoop-v0",
        entry_point="gym_lowcostrobot.envs:PushCubeLoopEnv",
        max_episode_steps=500,
    )

register_robotics_envs()