import gym_lowcostrobot # Import the low-cost robot environments
import numpy as np
import gymnasium as gym
from copy import deepcopy
import imageio 
from collections import deque

np.set_printoptions(precision=3, suppress=True)

# env = gym.make('LiftCubeStateCameraPrivileged-v0', render_mode="human", observation_mode="both", action_mode="nullspace")
env = gym.make('LiftCubeState-v0', render_mode="human")
env.reset()

demo_dict = {
    'observations': {
        'rgb': [],
        'state': []
    },
    'next_observations': {
        'rgb': [],
        'state': []
    },
    'actions': [],
    'abs_pos_actions': [],
    'joint_actions': [],
    'rewards': [],
    'dones': [],
}

pos_action_noise = 0.0001

def store_transition(demo, obs, abs_action, rel_action, reward, term, trunc, info):
    # demo['observations']['rgb'].append(obs['image_wrist'])
    # state = np.concatenate([obs['arm_qpos'], obs['ee_pos'], obs['cube_pos']], -1)
    # demo['observations']['state'].append(state)

    demo['actions'].append(rel_action)
    demo['abs_pos_actions'].append(abs_action)
    demo['joint_actions'].append(info['target_qpos'])

    demo['rewards'].append(reward)
    demo['dones'].append(term or trunc)

    return demo

def collect_episode(demo, env, ep):

    # while True:
    obs, info = env.reset()
    # env.render()

    # demo['observations']['rgb'].append(obs['image_wrist'])
    # state = np.concatenate([obs['arm_qpos'], obs['ee_pos'], obs['cube_pos']], -1)
    state = np.concatenate([obs['qpos'], obs['qvel']], -1)
    demo['observations']['state'].append(state)

    i = 0
    print("Going up")
    # first go up to middle of the workspace so everything is in the field of view.
    desired_pos = np.array([0, 0.07, 0.13])
    action = np.array([*desired_pos, -2])
    ee_pos = env.unwrapped.get_ee_pos()
    while np.linalg.norm(ee_pos[:3][-1] - desired_pos[-1]) > 0.02:
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
        # print(f't:{i+1}', 'ee', ee_pos, end='\n')
        i += 1
        if term:
            print(f'ep {ep} terminated with {i} actions')
            break
        if trunc:
            print(f'ep {ep} truncated with {i} actions')
            break
    if term or trunc:
        return demo



    desired_pos = info['qpos'][env.unwrapped.cube_dof_id: env.unwrapped.cube_dof_id + 3]
    # pos_diff = np.array([0.02, 0, 0.025])
    pos_diff = np.array([0.00, 0, 0.025])
    desired_pos = pos_diff + desired_pos
    action = np.array([*desired_pos, -1])

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
    pos_diff = np.array([0.00, 0, -0.01])
    desired_pos = pos_diff + desired_pos
    action = np.array([*desired_pos, -1])
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
    print("Closing the gripper")
    ee_pos = env.unwrapped.get_ee_pos()
    desired_pos = ee_pos[:3]
    action = np.array([*desired_pos, 10])
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
    action = np.array([*desired_pos, 10])
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

demos = deepcopy(demo_dict)
for ep in range(1):
    demo = deepcopy(demo_dict)
    demo = collect_episode(demo, env, ep)