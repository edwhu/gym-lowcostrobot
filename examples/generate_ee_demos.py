import gym_lowcostrobot # Import the low-cost robot environments
import numpy as np
import gymnasium as gym
from copy import deepcopy
import imageio 
from collections import deque

np.set_printoptions(precision=2, suppress=True)

env = gym.make('LiftCubeCameraPrivileged-v0', render_mode="rgb_array", observation_mode="both", action_mode="nullspace")

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

pos_action_noise = 0.01

def store_transition(demo, obs, abs_action, rel_action, reward, term, trunc, info):
    demo['observations']['rgb'].append(obs['image_wrist'])
    state = np.concatenate([obs['arm_qpos'], obs['ee_pos'], obs['cube_pos']], -1)
    demo['observations']['state'].append(state)

    demo['actions'].append(rel_action)
    demo['abs_pos_actions'].append(abs_action)
    demo['joint_actions'].append(info['target_qpos'])

    demo['rewards'].append(reward)
    demo['dones'].append(term or trunc)

    return demo

def collect_episode(demo, env):
    obs, info = env.reset()

    demo['observations']['rgb'].append(obs['image_wrist'])
    state = np.concatenate([obs['arm_qpos'], obs['ee_pos'], obs['cube_pos']], -1)
    demo['observations']['state'].append(state)

    desired_pos = info['qpos'][env.unwrapped.cube_dof_id: env.unwrapped.cube_dof_id + 3]
    pos_diff = np.array([0.02, 0, 0.025])
    desired_pos = pos_diff + desired_pos
    action = np.array([*desired_pos, -1])

    # go right over the box.
    i = 0
    while np.linalg.norm(obs['ee_pos'][:3] - desired_pos) > 0.01:
        noise = np.random.normal(0, pos_action_noise, size=3)
        rel_action = action - obs['ee_pos']
        rel_action[:3] += noise
        abs_action = action.copy()
        abs_action[:3] += noise

        obs, reward, term, trunc, info = env.step(rel_action)
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
    # print('go down')
    desired_pos = info['qpos'][env.unwrapped.cube_dof_id: env.unwrapped.cube_dof_id + 3]
    pos_diff = np.array([0.02, 0, -0.01])
    desired_pos = pos_diff + desired_pos
    action = np.array([*desired_pos, -1])
    while np.linalg.norm(obs['ee_pos'][:3] - desired_pos) > 0.01:
        noise = np.random.normal(0, pos_action_noise, size=3)
        rel_action = action - obs['ee_pos']
        rel_action[:3] += noise
        abs_action = action.copy()
        abs_action[:3] += noise

        obs, reward, term, trunc, info = env.step(rel_action)
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
    desired_pos = obs['ee_pos'][:3]
    action = np.array([*desired_pos, 10])
    for _ in range(10):
        noise = np.random.normal(0, pos_action_noise, size=3)
        rel_action = action - obs['ee_pos']
        rel_action[:3] += noise
        abs_action = action.copy()
        abs_action[:3] += noise

        obs, reward, term, trunc, info = env.step(rel_action)
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
    # print('lift up')
    desired_pos = info['qpos'][env.unwrapped.cube_dof_id: env.unwrapped.cube_dof_id + 3]
    pos_diff = np.array([0.00, 0, 0.12])
    desired_pos = pos_diff + desired_pos
    action = np.array([*desired_pos, 10])
    while np.linalg.norm(obs['ee_pos'][:3] - desired_pos) > 0.01:
        noise = np.random.normal(0, pos_action_noise, size=3)
        rel_action = action - obs['ee_pos']
        rel_action[:3] += noise
        abs_action = action.copy()
        abs_action[:3] += noise

        obs, reward, term, trunc, info = env.step(rel_action)
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
for ep in range(100):
    # desired_pos = goals[ep]
    demo = deepcopy(demo_dict)
    demo = collect_episode(demo, env)
    for k, v in demo.items():
        if k != 'next_observations' and isinstance(v, dict):
            for k2, v2 in v.items():
                demo[k][k2] = np.stack(v2)

    # episode is done. finalize the demo dictionary. 
    # framestacking
    rgb = demo['observations']['rgb'].transpose(0,3,1,2)
    resized_rgb = rgb.transpose(0,2,3,1) # (T, H, W, C)
    # do framestacking, so that we transform (T,H,W,C) to (T,H,W,C * Frame stack)
    num_frames = 2
    frames = deque(maxlen=num_frames)
    framestacked_rgb = np.ones((resized_rgb.shape[0], resized_rgb.shape[1], resized_rgb.shape[2], 3), dtype=np.uint8)
    for _ in range(num_frames-1):
        frames.append(resized_rgb[0])
    
    for t in range(resized_rgb.shape[0]):
        frames.append(resized_rgb[t])
        _all_6_frames = np.concatenate(frames, axis=-1)
        framestacked_rgb[t] = _all_6_frames[:, :, 1::2]

    resized_rgb = framestacked_rgb

    # create the next observations array.
    demo['next_observations']['rgb'] = resized_rgb[1:]
    demo['next_observations']['state'] = demo['observations']['state'][1:]
    # chop off the last observation
    demo['observations']['rgb'] = resized_rgb[:-1]
    demo['observations']['state'] = demo['observations']['state'][:-1]

    episodic_return.append(np.sum(demo['rewards']))
    episodic_success.append(episodic_return[-1] > 0)

    # add the demo to the list of demos
    for k, v in demo.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                demos[k][k2].append(np.stack(v2))
        else:
            demos[k].append(v)

# flatten the demos
for k, v in demos.items():
    if isinstance(v, dict):
        for k2, v2 in v.items():
            demos[k][k2] = np.concatenate(v2, axis=0)
    else:
        demos[k] = np.concatenate(v, axis=0)

print('\nfinal demo dataset')
for k, v in demos.items():
    if isinstance(v, dict):
        for k2, v2 in v.items():
            print(k, k2, v2.shape, v2.dtype)
    else:
        print(k, v.shape, v2.dtype)

# save the statistics into a metadata dict
metadata = {
    'success': np.mean(episodic_success),
    'return_avg': np.mean(episodic_return),
    'return_min': np.min(episodic_return),
    'return_max': np.max(episodic_return),
    'action_min': np.min(demos['actions'], axis=0),
    'action_max': np.max(demos['actions'], axis=0),
    'joint_action_min': np.min(demos['joint_actions'], axis=0),
    'joint_action_max': np.max(demos['joint_actions'], axis=0),
}
demos['metadata'] = metadata

print('\nstatistics:')
for k, v in metadata.items():
    print(k, v)

imageio.mimwrite('demos.mp4', demos['observations']['rgb'][:1000], fps=100)
# store as a pickle file.
import pickle 
with open('buffer.pkl', 'wb') as f:
    pickle.dump(demos, f)