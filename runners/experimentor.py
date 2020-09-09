import os
import tensorflow as tf
import sys
sys.path.append("..") 
from agents.PPO.PPOAgent import PPOAgent
from agents.random_agent import RandomAgent
from simulators.distributed_simulation import DistributedSimulation
from influence.influence import Influence
from simulators.warehouse.warehouse import Warehouse
import argparse
import yaml
import time


class Experimentor(object):
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
        self.path = self.generate_path(self.parameters)
        self.agent = PPOAgent(self.parameters, 4)
        self.train_frequency = self.parameters["train_frequency"]
        if self.parameters['simulator'] == 'partial':
            global_simulator = Warehouse()
            self.influence = Influence(self.agent, global_simulator)
        else:
            self.influence = None
        self.sim = DistributedSimulation(self.parameters, self.influence)
        tf.reset_default_graph()

    def generate_path(self, parameters):
        """
        Generate a path to store e.g. logs, models and plots. Check if
        all needed subpaths exist, and if not, create them.
        """
        path = self.parameters['name']
        result_path = os.path.join("../results", path)
        model_path = os.path.join("../models", path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return path

    def print_results(self, info, n_steps=0):
        """
        Prints results to the screen.
        """
        print(("Train step {} of {}".format(self.step,
                                            self.maximum_time_steps)))
        print(("-"*30))
        print(("Episode {} ended after {} steps.".format(self.agent.episodes,
                                                         n_steps)))
        print(("- Total reward: {}".format(info)))
        print(("-"*30))

    def run(self):
        """
        Runs the experiment.
        """
        self.maximum_time_steps = int(self.parameters["max_steps"])
        self.step = max(self.parameters["iteration"], 0)
        # reset environment
        step_output = self.sim.reset()
        reward = 0
        n_steps = 0
        start = time.time()
        while self.step < self.maximum_time_steps:
            if self.parameters['simulator'] == 'partial' and \
              self.step % self.parameters['influence_train_frequency'] == 0:
                self.influence.train()
            # Select the action to perform
            action = self.agent.take_action(step_output)
            # Increment step
            self.step += 1
            # Get new state and reward given actions a
            step_output = self.sim.step(action)
            
            reward += step_output['reward'][0]
            n_steps += 1
            if step_output['done'][0]:
                end = time.time()
                print('Time: ', end - start)
                start = end
                self.print_results(reward, n_steps)
                reward = 0
                n_steps = 0

        self.sim.close()

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
    exp = Experimentor(parameters)
    exp.run()