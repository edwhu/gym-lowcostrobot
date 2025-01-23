import gymnasium as gym
import gym_lowcostrobot

env = gym.make("LiftCubeCameraPrivileged-v0", render_mode="rgb_array", disable_env_checker=True)
obs, info = env.reset()
while True:
    obs, info = env.reset()
    print('rendering is working')

