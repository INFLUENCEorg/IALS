import os
import tensorflow as tf
import sys
sys.path.append("..") 
from agents.PPO.PPOAgent import PPOAgent
from agents.random_agent import RandomAgent
from simulators.distributed_simulation import DistributedSimulation
from influence.influence_network import InfluenceNetwork
from influence.influence_uniform import InfluenceUniform
from influence.influence_dummy import InfluenceDummy
from influence.data_collector import DataCollector
from simulators.warehouse.warehouse import Warehouse
import argparse
import yaml
import time
import sacred
from sacred.observers import MongoObserver
import pymongo

class Experiment(object):
    """
    Creates experiment object to interact with the environment and
    the agent and log results.
    """

    def __init__(self, parameters, _run, seed):
        """
        """
        self.parameters = parameters['main']
        self.parameters_influence = parameters['influence']
        self.path = self.generate_path(self.parameters['name'])
        self.agent = PPOAgent(4, parameters['main'])
        self.train_frequency = self.parameters['train_frequency']
        data_path = parameters['influence']['data_path'] + str(_run._id) + '/'
        if self.parameters['simulator'] == 'partial':
            if self.parameters['influence_model'] == 'nn':
                self.influence = InfluenceNetwork(parameters['influence'], data_path, _run._id)
            else:
                self.influence = InfluenceUniform(parameters['influence'])
        else:
            self.influence = InfluenceDummy(parameters['influence'])
        self.sim = DistributedSimulation(self.parameters['env'], self.parameters['simulator'], 
                                         self.parameters['num_workers'], self.influence, seed)
        self.data_collector = DataCollector(self.agent, self.parameters['env'], self.parameters['num_workers'], 
                                            self.influence, data_path, seed)
        tf.reset_default_graph()
        self._run = _run

    def generate_path(self, path):
        """
        Generate a path to store e.g. logs, models and plots. Check if
        all needed subpaths exist, and if not, create them.
        """
        result_path = os.path.join("../results", path)
        model_path = os.path.join("../models", path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return path

    def print_results(self, episode_return, episode_step, global_step):
        """
        Prints results to the screen.
        """
        print(("Train step {} of {}".format(global_step,
                                            self.maximum_time_steps)))
        print(("-"*30))
        print(("Episode {} ended after {} steps.".format(self.agent.episodes,
                                                         episode_step)))
        print(("- Total reward: {}".format(episode_return)))
        print(("-"*30))

    def run(self):
        """
        Runs the experiment.
        """
        self.maximum_time_steps = int(self.parameters["max_steps"])
        global_step = max(self.parameters["iteration"], 0)
        # reset environment
        # self.sim = DistributedSimulation(self.parameters, self.influence)
        # step_output = self.sim.reset()
        episode_return = 0
        episode_step = 0
        start = time.time()
        step_output = self.sim.reset()
        while global_step <= self.maximum_time_steps:
            if global_step % self.parameters_influence['train_freq']  == 0 and self.parameters['simulator'] == 'partial':
                mean_episodic_return = self.data_collector.run(self.parameters_influence['dataset_size'], log=True)
                self.influence.train()
                # influence model parameters need to be loaded every time they are updated because 
                # each process keeps a separate copy of the influence model
                self.sim.load_influence_model()
                self._run.log_scalar("mean episodic return", mean_episodic_return, global_step)
            elif global_step % self.parameters['eval_freq'] == 0:
                mean_episodic_return = self.data_collector.run(self.parameters['eval_steps'], log=False)
                self._run.log_scalar("mean episodic return", mean_episodic_return, global_step)
            # Select the action to perform
            action = self.agent.take_action(step_output)
            # Increment step
            episode_step += 1
            global_step += 1
            # Get new state and reward given actions a
            step_output = self.sim.step(action)
            episode_return += step_output['reward'][0]
            if step_output['done'][0]:
                end = time.time()
                print('Time: ', end - start)
                start = end
                self.print_results(episode_return, episode_step, global_step)
                episode_return = 0
                episode_step = 0

        self.sim.close()

def add_mongodb_observer():
    """
    connects the experiment instance to the mongodb database
    """
    db_uri = 'mongodb://localhost:27017/scalable-simulations'
    db_name = 'scalable-simulations'
    try:
        print("Trying to connect to mongoDB '{}'".format(db_uri))
        client = pymongo.MongoClient(db_uri, ssl=False)
        client.server_info()
        ex.observers.append(MongoObserver.create(db_uri, db_name=db_name, ssl=False))
        print("Added MongoDB observer on {}.".format(db_uri))
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print(e)
        print("ONLY FILE STORAGE OBSERVER ADDED")
    from sacred.observers import FileStorageObserver
    ex.observers.append(FileStorageObserver.create('saved_runs'))
    
ex = sacred.Experiment('scalable-simulations')
ex.add_config('configs/warehouse/default.yaml')
add_mongodb_observer()

@ex.automain
def main(parameters, seed, _run):
    exp = Experiment(parameters, _run, seed)
    exp.run()