import os
import sys
sys.path.append("..") 
from agents.random_agent import RandomAgent
from simulators.warehouse.warehouse import Warehouse
import argparse
import yaml
import time


class DataCollector(object):
    """
    Creates experimentor object to store interact with the environment and
    the agent and log results.
    """

    def __init__(self, agent, simulator):
        """
        """
        self.parameters = read_parameters('../influence/configs/data_collection.yaml')
        self.data_file = self.generate_path()
        self.sim = simulator
        self.agent = agent

    def generate_path(self):
        """
        Generate a path to store e.g. logs, models and plots. Check if
        all needed subpaths exist, and if not, create them.
        """
        name = self.parameters['name']
        data_path = os.path.join('../influence/data', name)
        data_file = os.path.join(data_path, 'data.csv')
        if os.path.exists(data_file):
            os.remove(data_file)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        return data_file

    def run(self):
        """
        Runs the data collection process.
        """
        print('collecting data...')
        self.maximum_time_steps = int(self.parameters["max_steps"])
        step = 0
        # reset environment
        obs = self.sim.reset()
        done = True
        while step < self.maximum_time_steps:
            self.sim.log_obs(self.data_file)
            action = self.agent.take_action({'obs': [obs], 'done': [done]}, 'eval')[0]
            step += 1
            obs, _, done, _ = self.sim.step(action)
        self.sim.close()
        print('Done!')

def read_parameters(config_file):
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters['parameters']

if __name__ == "__main__":
    simulator = Warehouse()
    
    agent = RandomAgent(simulator.action_space.n)
    exp = DataCollector(agent, simulator)
    exp.run()