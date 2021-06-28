import sys
sys.path.append("..") 
import numpy as np
import os
import csv
import torch
from agent.a2c_ppo_acktr.envs import make_vec_envs


def evaluate(agent, env_name, seed, num_processes, device, eval_steps):
    """
    Evaluate current policy on global simulator
    """
    # torch.manual_seed(seed)
    envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                         None, './logs', device, True)
    obs = envs.reset()
    recurrent_hidden_states = torch.zeros(
        num_processes, agent.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)
    n_steps = 0
    print('Evaluating policy on global simulator...')
    episodic_returns = []
    episodic_return = 0
    while n_steps < eval_steps//num_processes:
        n_steps += 1
        with torch.no_grad():
            _, action, _, recurrent_hidden_states = agent.act(
                obs,
                recurrent_hidden_states,
                masks,
                deterministic=True)
        # Obser reward and next obs
        obs, reward, done, _ = envs.step(action)
        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)
        episodic_return += np.mean(reward.detach().numpy())
        # Think what to do if episodes are not same length
        if done[0]:
            episodic_returns.append(episodic_return)
            episodic_return = 0
    print('Done!')
    # envs.close()
    return np.mean(episodic_returns)