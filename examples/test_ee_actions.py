import gym_lowcostrobot # Import the low-cost robot environments
import numpy as np
import gymnasium as gym
np.set_printoptions(2)
np.set_printoptions(suppress=True)


env = gym.make('LiftCubeCamera-v0', render_mode="human", observation_mode="both", action_mode="nullspace")

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

for ep in range(100):
    # desired_pos = goals[ep]
    obs, info = env.reset()
    desired_pos = info['qpos'][env.unwrapped.cube_dof_id: env.unwrapped.cube_dof_id + 3]
    pos_diff = np.array([0.02, 0, 0.025])
    desired_pos = pos_diff + desired_pos
    action = np.array([*desired_pos, -1])

    # go right over the box.
    i = 0
    while np.linalg.norm(obs['ee_pos'][:3] - desired_pos) > 0.01:
        obs, reward, term, trunc, info = env.step(action)
        # print(f"desired ee: {desired_pos}")
        # print(f't:{i+1}', 'ee', obs['ee_pos'], end='\n')
        i += 1
        if term:
            print('episode terminated')
            break
        if trunc:
            print('episode truncated')
            break
    
    # go down over the box
    # print('go down')
    desired_pos = info['qpos'][env.unwrapped.cube_dof_id: env.unwrapped.cube_dof_id + 3]
    pos_diff = np.array([0.02, 0, -0.01])
    desired_pos = pos_diff + desired_pos
    action = np.array([*desired_pos, -1])
    while np.linalg.norm(obs['ee_pos'][:3] - desired_pos) > 0.01:
        obs, reward, term, trunc, info = env.step(action)
        # print(f"desired ee: {desired_pos}")
        # print(f't:{i+1}', 'ee', obs['ee_pos'], end='\n')
        i += 1
        # print(np.linalg.norm(obs['ee_pos'][:3] - desired_pos))
        if term:
            print('episode terminated')
            break
        if trunc:
            print('episode truncated')
            break
    
    # close the gripper.
    desired_pos = obs['ee_pos'][:3]
    action = np.array([*desired_pos, 10])
    for _ in range(10):
        obs, reward, term, trunc, info = env.step(action)
        i += 1
        if term:
            print('episode terminated')
            break
        if trunc:
            print('episode truncated')
            break

    # lift up 
    # print('lift up')
    desired_pos = info['qpos'][env.unwrapped.cube_dof_id: env.unwrapped.cube_dof_id + 3]
    pos_diff = np.array([0.00, 0, 0.12])
    desired_pos = pos_diff + desired_pos
    action = np.array([*desired_pos, 10])
    while np.linalg.norm(obs['ee_pos'][:3] - desired_pos) > 0.01:
        obs, reward, term, trunc, info = env.step(action)
        # print(f"desired ee: {desired_pos}")
        # print(f't:{i+1}', 'ee', obs['ee_pos'], end='\n')
        i += 1
        if term:
            print('episode terminated')
            break
        if trunc:
            print('episode truncated')
            break