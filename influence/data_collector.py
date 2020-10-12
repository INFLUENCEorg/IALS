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


class DataCollector(object):
    """
    Creates experimentor object to store interact with the environment and
    the agent and log results.
    """

    def __init__(self, agent, simulator, influence, data_path):
        """
        """
        self.generate_path(data_path)
        self.inputs_file = data_path + str('inputs.csv')
        self.targets_file = data_path + str('targets.csv')
        self.sim = simulator
        self.agent = agent
        self.influence = influence

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
        # reset environment
        obs = self.sim.reset()
        done = True
        episodic_return = 0
        episodic_returns = []
        while step < num_steps:
            if log:
                self.sim.log(self.inputs_file, 'dset')
                self.sim.log(self.targets_file, 'infs')
            if self.influence.aug_obs:
                if done:
                    self.influence.reset()
                dset = self.sim.get_dset()
                probs = self.influence.predict(dset)
                obs = np.append(obs, np.concatenate([prob[:-1] for prob in probs]))
            action = self.agent.take_action({'obs': [obs], 'done': [done]}, 'eval')[0]
            step += 1
            obs, reward, done, _ = self.sim.step(action)
            episodic_return += reward
            if done:
                episodic_returns.append(episodic_return)
                episodic_return = 0
        self.sim.close()
        print('Done!')
        mean_episodic_return = np.mean(episodic_returns)
        return mean_episodic_return

def read_parameters(config_file):
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters['parameters']

if __name__ == "__main__":
    simulator = Warehouse()
    
    agent = RandomAgent(simulator.action_space.n)
    exp = DataCollector(agent, simulator)
    exp.run()