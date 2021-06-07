import sys
sys.path.append("..") 
import numpy as np
import os
import csv
import torch
from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr import utils
from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.envs import make_vec_envs


def collect_data(agent, env_name, seed, num_processes, log_dir,
                 device, num_samples, data_path):
    
    print('Collecting data...')
    generate_path(data_path)
    inputs_file = data_path + str('inputs.csv')
    targets_file = data_path + str('targets.csv')
    envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                         None, log_dir, device, True)
    obs = envs.reset()
    recurrent_hidden_states = torch.zeros(
        num_processes, agent.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)
    n_steps = 0
    done = [False]*num_processes
    while n_steps < num_samples//num_processes:
        n_steps += 1
        # NOTE: Episodes might not be same length
        if done[0]:
            log(dset, infs, inputs_file, targets_file)
            dset = []
            infs = []
        dset.append(np.array(envs.get_dset()))
        infs.append(np.array(envs.get_infs()))
        with torch.no_grad():
            _, action, _, recurrent_hidden_states = agent.act(
                obs,
                recurrent_hidden_states,
                masks,
                deterministic=True)
        # Obser reward and next obs
        obs, _, done, _ = envs.step(action)
        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)
    envs.close()
    print('Done!')

def log(dset, infs, inputs_file, targets_file):
    """
    Log dataset
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
    Generate a path to store e.g. logs, models and plots. Check if
    all needed subpaths exist, and if not, create them.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)