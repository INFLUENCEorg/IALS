import os
import sys
sys.path.append("..") 
from simulators.distributed_simulation import DistributedSimulation
from simulators.simulation import Simulation
import argparse
import yaml
import time
import torch
import numpy as np
import csv


class DataCollector(object):
    """
    Creates experimentor object to store interact with the environment and
    the agent and log results.
    """

    def __init__(self, agent, env, num_workers, influence, data_path, seed):
        """
        """
        self.generate_path(data_path)
        self.inputs_file = data_path + str('inputs.csv')
        self.targets_file = data_path + str('targets.csv')
        self.agent = agent
        self.influence = influence
        self.num_workers = num_workers
        self.env = env
        self.seed = seed
        self.sim = DistributedSimulation(self.env, 'global', self.num_workers, self.influence, self.seed)

    def generate_path(self, data_path):
        """
        Generate a path to store e.g. logs, models and plots. Check if
        all needed subpaths exist, and if not, create them.
        """
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def run(self, num_steps, log=False, load=False):
        """
        Runs the data collection process.
        """
        print('collecting data...')
        # if self.num_workers > 1:
        if load:
            self.sim.load_influence_model()
        # else:
            # sim = Simulation(self.env, 'global', self.influence, self.seed)
        step = 0
        step_output = self.sim.reset()
        episodic_returns = []
        episodic_return = 0
        infs = []
        dset = []
        while step <= num_steps//self.num_workers:
            if log:
                # Think what to do if episodes are not same length
                if step_output['done'][0]:
                    self.log(dset, infs)
                    dset = []
                    infs = []
                dset.append(np.array(step_output['dset']))
                infs.append(np.array(step_output['infs']))
            action = self.agent.take_action(step_output, 'eval')
            step += 1
            step_output = self.sim.step(action)
            episodic_return += np.mean(step_output['reward'])
            # Think what to do if episodes are not same length
            if step_output['done'][0]:
                episodic_returns.append(episodic_return)
                episodic_return = 0
        print('Done!')
        self.seed += self.num_workers # Changing seed for the next run. Last iteration seed was self.seed+[0:self.num_workers]
        mean_episodic_return = np.mean(episodic_returns)
        return mean_episodic_return
    
    def log(self, dset, infs):
        dset = np.reshape(np.swapaxes(dset, 0, 1), (-1, np.shape(dset)[2]))
        infs = np.reshape(np.swapaxes(infs, 0, 1), (-1, np.shape(infs)[2]))
        with open(self.inputs_file,'a') as file:
            writer = csv.writer(file)
            for element in dset:
                writer.writerow(element)
        with open(self.targets_file,'a') as file:
            writer = csv.writer(file)
            for element in infs:
                writer.writerow(element)

def read_parameters(config_file):
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters['parameters']

if __name__ == "__main__":
    sys.path.append("..") 
    from agents.random_agent import RandomAgent
    from simulators.warehouse.warehouse import Warehouse
    from influence_dummy import InfluenceDummy
    agent = RandomAgent(2)
    parameters = {'n_sources': 4, 'output_size': 1, 'aug_obs': False}
    influence = InfluenceDummy(parameters)
    exp = DataCollector(agent, 'traffic', 8, influence, './data/traffic/', 0)
    exp.run(10000, log=True)