import os

from gymnasium.envs.registration import register

__version__ = "0.0.1"

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets", "low_cost_robot_6dof")
BASE_LINK_NAME = "link_1"

register(
    id="LiftCube-v0",
    entry_point="gym_lowcostrobot.envs:LiftCubeEnv",
    max_episode_steps=500,
)

register(
    id="LiftCubeCamera-v0",
    entry_point="gym_lowcostrobot.envs:LiftCubeCameraEnv",
    max_episode_steps=600,
)

register(
    id="LiftCubeCameraPrivileged-v0",
    entry_point="gym_lowcostrobot.envs:LiftCubeCameraEnv",
    max_episode_steps=500,
    kwargs={"observation_mode":"both"}
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
    id="PushCubeSimple-v0",
    entry_point="gym_lowcostrobot.envs:PushCubeSimpleEnv",
    max_episode_steps=100,
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
