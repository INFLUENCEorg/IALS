import sys
sys.path.append("..") 
import numpy as np
import os
import csv
import torch
from agent.a2c_ppo_acktr.envs import make_vec_envs


def collect_data(agent, env_name, seed, num_processes, log_dir,
                 device, num_samples, data_path):
    """
    Collects data from global simulator
    """
    generate_path(data_path)
    inputs_file = data_path + str('inputs.csv')
    targets_file = data_path + str('targets.csv')
    envs = make_vec_envs(env_name, seed + num_processes, 8,
                         None, log_dir, device, True)
    obs = envs.reset()
    recurrent_hidden_states = torch.zeros(
        num_processes, agent.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)
    n_steps = 0
    done = [False]*num_processes
    dset = []
    infs = []
    print('Collecting data from global simulator...')
    while n_steps < num_samples//num_processes:
        n_steps += 1
        # NOTE: Episodes might not be same length
        if done[0]:
            log(dset, infs, inputs_file, targets_file)
            dset = []
            infs = []
        with torch.no_grad():
            _, action, _, recurrent_hidden_states = agent.act(
                obs,
                recurrent_hidden_states,
                masks,
                deterministic=True)
        # Obser reward and next obs
        obs, _, done, info = envs.step(action)
        dset.append(np.array([i['dset'] for i in info]))
        infs.append(np.array([i['infs'] for i in info]))
        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)
    envs.close()
    print('Done!')

def log(dset, infs, inputs_file, targets_file):
    """
    Log influence dataset
    """
    dset = np.reshape(np.swapaxes(dset, 0, 1), (-1, np.shape(dset)[2]))
    infs = np.reshape(np.swapaxes(infs, 0, 1), (-1, np.shape(infs)[2]))
    with open(inputs_file,'a') as file:
        writer = csv.writer(file)
        for element in dset:
            writer.writerow(element)
    with open(targets_file,'a') as file:
        writer = csv.writer(file)
        for element in infs:
            writer.writerow(element)

def generate_path(data_path):
    """
    Generate a path to store infs-dset pairs. Check if
    all needed subpaths exist, and if not, create them.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)