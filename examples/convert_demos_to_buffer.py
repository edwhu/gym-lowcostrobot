"""Used to convert npz demos into a single buffer.pkl for FoWM codebase"""
import gymnasium as gym
import imageio
import numpy as np
import os
import torch
import torchvision.transforms.functional as F

demo_folder = "/home/edward/projects/gym-lowcostrobot/demos/terminate_demos"

demos = {
    'observations': {
        'rgb': [],
        'state': []
    },
    'next_observations': {
        'rgb': [],
        'state': []
    },
    'actions': [],
    'rewards': [],
    'dones': [],
}

episodic_return = []
episodic_success = []
for entry in os.scandir(demo_folder):
    if entry.is_file() and entry.name.endswith('.npz'):
        print(entry.name)
        demo = np.load(entry.path)
        rgb = torch.from_numpy(demo['obs/image_front'].transpose(0,3,1,2))
        # resized_rgb = F.resize(rgb, size=(84,84)).numpy().transpose(0,2,3,1)
        resized_rgb = rgb.numpy().transpose(0,2,3,1)
        demos['observations']['rgb'].append(resized_rgb[:-1])
        demos['next_observations']['rgb'].append(resized_rgb[1:])

        state = np.concatenate([demo['obs/arm_qpos'], demo['obs/ee_pos'], demo['obs/cube_pos']], -1)
        demos['observations']['state'].append(state[:-1])
        demos['next_observations']['state'].append(state[1:])
        
        demos['actions'].append(demo['action'])
        demos['rewards'].append(demo['reward'])
        terminated = demo['terminated']   
        truncated = demo['truncated']
        done = (terminated | truncated).astype(np.bool)
        demos['dones'].append(done)

        episodic_return.append(np.sum(demo['reward']))
        episodic_success.append(episodic_return[-1] > 0)

# convert the demos to numpy arrays
for k, v in demos.items():
    if isinstance(v, dict):
        for k2, v2 in v.items():
            demos[k][k2] = np.concat(v2, axis=0)
    else:
        demos[k] = np.concat(v, axis=0)

print('\nfinal demo dataset')
for k, v in demos.items():
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
    'action_min': np.min(demos['actions']),
    'action_max': np.max(demos['actions']),
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