import os
import sys
sys.path.append("..") 
from agents.random_agent import RandomAgent
from environments.warehouse.warehouse import Warehouse
import argparse
import yaml
import time


class DataCollector(object):
    """
    Creates experimentor object to store interact with the environment and
    the agent and log results.
    """

    def __init__(self, parameters):
        """
        Initializes the experiment by extracting the parameters
        @param parameters a dictionary with many obligatory elements
        <ul>
        <li> "env_type" (SUMO, atari, grid_world),
        <li> algorithm (DQN, PPO)
        <li> maximum_time_steps
        <li> maximum_episode_time
        <li> skip_frames
        <li> save_frequency
        <li> step
        <li> episodes
        and more TODO
        </ul>
        """
        self.parameters = parameters
        self.generate_path(self.parameters)
        self.env = Warehouse()
        self.agent = RandomAgent(self.parameters, self.env.action_space.n)

    def generate_path(self, parameters):
        """
        Generate a path to store e.g. logs, models and plots. Check if
        all needed subpaths exist, and if not, create them.
        """
        path = self.parameters['name']
        data_path = os.path.join("../data", path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def run(self):
        """
        Runs the data collection process.
        """
        print('collecting data...')
        self.maximum_time_steps = int(self.parameters["max_steps"])
        step = 0
        # reset environment
        obs = self.env.reset()
        while step < self.maximum_time_steps:
            self.env.log_obs('../data/warehouse/data.csv')
            action = self.agent.take_action({'obs': [obs], 'done': [False]})
            step += 1
            obs, _, _, _ = self.env.step(action)
        self.env.close()
        print('Done!')

def get_parameters():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config', default=None, help='config file')
    args = parser.parse_args()
    return args

def read_parameters(config_file):
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters['parameters']

if __name__ == "__main__":
    args = get_parameters()
    parameters = read_parameters(args.config)
    exp = DataCollector(parameters)
    exp.run()