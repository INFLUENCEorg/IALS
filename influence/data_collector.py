import os
import sys
sys.path.append("..") 
from agents.random_agent import RandomAgent
from simulators.warehouse.warehouse import Warehouse
import argparse
import yaml
import time
import torch
import numpy as np
from einops import rearrange
import csv


class DataCollector(object):
    """
    Creates experimentor object to store interact with the environment and
    the agent and log results.
    """

    def __init__(self, agent, simulator, num_workers, influence, data_path):
        """
        """
        self.generate_path(data_path)
        self.inputs_file = data_path + str('inputs.csv')
        self.targets_file = data_path + str('targets.csv')
        self.sim = simulator
        self.agent = agent
        self.influence = influence
        self.num_workers = num_workers

    def generate_path(self, data_path):
        """
        Generate a path to store e.g. logs, models and plots. Check if
        all needed subpaths exist, and if not, create them.
        """
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def run(self, num_steps, log=False):
        """
        Runs the data collection process.
        """
        print('collecting data...')
        step = 0
        step_output = self.sim.reset()
        episodic_return = 0
        episodic_returns = []
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
            if step_output['done']:
                episodic_returns.append(episodic_return)
                episodic_return = 0
        self.sim.close()
        print('Done!')
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
    simulator = Warehouse()
    
    agent = RandomAgent(simulator.action_space.n)
    exp = DataCollector(agent, simulator)
    exp.run()