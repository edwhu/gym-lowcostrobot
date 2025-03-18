import gym_lowcostrobot # Import the low-cost robot environments
import numpy as np
import gymnasium as gym
from copy import deepcopy
import imageio 
import os
# from collections import deque
from tensordict import TensorDict
# import torchvision.transforms.functional as F
import torch

np.set_printoptions(precision=3, suppress=True)
demo_folder = "/Users/edward/projects/gym-lowcostrobot/demos/grayscale_lift"
os.makedirs(demo_folder, exist_ok=True)
# format of episodic buffer should be:
# a list of dictionaries, each dictionary contains the following keys:
# observations, next_observations, actions, rewards, dones.

demos = []
new_episode = {
    'observations': {
        'rgb': [],
        'state': []
    },
    'actions': [],
    'rewards': [],
    'dones': [],
    'terminated': [],
}
env = gym.make('LiftCubeCameraPrivileged-v0', render_mode="rgb_array", observation_mode="both", action_mode="nullspace")
# env = gym.make('LiftCubeStateNoisy-v0', render_mode="human")
env.reset()



pos_action_noise = 0.005

def store_transition(demo, obs, abs_action, rel_action, reward, term, trunc, info):
    demo['observations']['rgb'].append(obs['image'])
    state = np.concatenate([obs['qpos'], obs['qvel']], -1)
    demo['observations']['state'].append(state)

    demo['actions'].append(rel_action)
    demo['rewards'].append(reward)
    demo['dones'].append(term or trunc)
    demo['terminated'].append(term)

    return demo

def collect_episode(demo, env, ep):
    obs, info = env.reset()
    demo['observations']['rgb'].append(obs['image'])
    state = np.concatenate([obs['qpos'], obs['qvel']], -1)
    demo['observations']['state'].append(state)

    i = 0
    desired_pos = obs['qpos'][env.unwrapped.cube_dof_id: env.unwrapped.cube_dof_id + 3]
    # pos_diff = np.array([0.02, 0, 0.025])
    pos_diff = np.array([0.01, 0.00, 0.025])
    desired_pos = pos_diff + desired_pos
    action = np.array([*desired_pos, 0.8])

    print("Descending down to the box")
    # go right over the box.
    ee_pos = env.unwrapped.get_ee_pos()
    while np.linalg.norm(ee_pos[:3] - desired_pos) > 0.01:
        noise = np.random.normal(0, pos_action_noise, size=3)
        rel_action = action - ee_pos
        rel_action[:3] += noise
        abs_action = action.copy()
        abs_action[:3] += noise

        rel_action = env.unwrapped.get_scaled_action(rel_action)
        obs, reward, term, trunc, info = env.step(rel_action)
        ee_pos = env.unwrapped.get_ee_pos()
        demo = store_transition(demo, obs, abs_action, rel_action, reward, term, trunc, info)

        # print(f"desired ee: {desired_pos}")
        # print(f't:{i+1}', 'ee', obs['ee_pos'], end='\n')
        i += 1
        if term:
            print(f'ep {ep} terminated with {i} actions')
            break
        if trunc:
            print(f'ep {ep} truncated with {i} actions')
            break
    if term or trunc:
        return demo
    # go down over the box
    print('Going down for picking')
    desired_pos = info['qpos'][env.unwrapped.cube_dof_id: env.unwrapped.cube_dof_id + 3]
    pos_diff = np.array([0.01, 0.00, -0.01])
    desired_pos = pos_diff + desired_pos
    action = np.array([*desired_pos, 0.8])
    ee_pos = env.unwrapped.get_ee_pos()
    while np.linalg.norm(ee_pos[:3] - desired_pos) > 0.02:
        noise = np.random.normal(0, pos_action_noise, size=3)
        rel_action = action - ee_pos
        rel_action[:3] += noise
        abs_action = action.copy()
        abs_action[:3] += noise

        rel_action = env.unwrapped.get_scaled_action(rel_action)
        obs, reward, term, trunc, info = env.step(rel_action)
        ee_pos = env.unwrapped.get_ee_pos()
        demo = store_transition(demo, obs, abs_action, rel_action, reward, term, trunc, info)
        # print(f"desired ee: {desired_pos}")
        # print(f't:{i+1}', 'ee', obs['ee_pos'], end='\n')
        i += 1
        # print(np.linalg.norm(obs['ee_pos'][:3] - desired_pos))
        if term:
            print(f'ep {ep} terminated with {i} actions')
            break
        if trunc:
            print(f'ep {ep} truncated with {i} actions')
            break
    if term or trunc:
        return demo
    # close the gripper.
    # print("Closing the gripper")
    ee_pos = env.unwrapped.get_ee_pos()
    desired_pos = ee_pos[:3]
    action = np.array([*desired_pos, -12])
    for _ in range(10):
        noise = np.random.normal(0, pos_action_noise, size=3)
        rel_action = action - ee_pos
        rel_action[:3] += noise
        abs_action = action.copy()
        abs_action[:3] += noise

        rel_action = env.unwrapped.get_scaled_action(rel_action)
        obs, reward, term, trunc, info = env.step(rel_action)
        ee_pos = env.unwrapped.get_ee_pos()
        demo = store_transition(demo, obs, abs_action, rel_action, reward, term, trunc, info)

        i += 1
        if term:
            print(f'ep {ep} terminated with {i} actions')
            break
        if trunc:
            print(f'ep {ep} truncated with {i} actions')
            break
    if term or trunc:
        return demo
    # lift up 
    print("Lifting up")
    desired_pos = info['qpos'][env.unwrapped.cube_dof_id: env.unwrapped.cube_dof_id + 3]
    pos_diff = np.array([0.00, 0, 0.12])
    desired_pos = pos_diff + desired_pos
    action = np.array([*desired_pos, -12])
    ee_pos = env.unwrapped.get_ee_pos()
    while np.linalg.norm(ee_pos[:3] - desired_pos) > 0.01:
        noise = np.random.normal(0, pos_action_noise, size=3)
        rel_action = action - ee_pos
        rel_action[:3] += noise
        abs_action = action.copy()
        abs_action[:3] += noise

        rel_action = env.unwrapped.get_scaled_action(rel_action)
        obs, reward, term, trunc, info = env.step(rel_action)
        ee_pos = env.unwrapped.get_ee_pos()
        demo = store_transition(demo, obs, abs_action, rel_action, reward, term, trunc, info)
        # print(f"desired ee: {desired_pos}")
        # print(f't:{i+1}', 'ee', obs['ee_pos'], end='\n')
        i += 1
        if term:
            print(f'ep {ep} terminated with {i} actions')
            break
        if trunc:
            print(f'ep {ep} truncated with {i} actions')
            break

    return demo
episodic_return = []
episodic_success = []

demos = []
for ep in range(100):
    print(ep)
    demo = deepcopy(new_episode)
    demo = collect_episode(demo, env, ep)
    episode = TensorDict(demo)
    demos.append(episode)
    success = np.sum(demo['rewards']) > 300
    episodic_success.append(success)
    episodic_return.append(np.sum(demo['rewards']))
    print(f"Running success rate: {np.mean(episodic_success):.2f}, {ep} episodes", end='\n')

print('\nfinal demo dataset')
for k, v in demos[0].items():
    if isinstance(v, dict):
        for k2, v2 in v.items():
            print(k, k2, v2.shape)
    else:
        print(k, v.shape)

# save the statistics into a metadata dict
metadata = {
    'success': np.mean(episodic_success),
    'return_avg': np.mean(episodic_return),
    'return_min': np.min(episodic_return),
    'return_max': np.max(episodic_return),
    'action_min': np.ones_like(demos[0]['actions'][0]) * -1,
    'action_max': np.ones_like(demos[0]['actions'][0]),
}

data = {}
data['metadata'] = metadata
data['episodes'] = demos

print('\nstatistics:')
for k, v in metadata.items():
    print(k, v)

imageio.mimwrite(os.path.join(demo_folder, 'demos.mp4'), demos[0]['observations']['rgb'][:1000], fps=5)

torch.save(data, os.path.join(demo_folder, 'buffer.pkl'))   