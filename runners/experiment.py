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
from sshtunnel import SSHTunnelForwarder

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
                if global_step == 0:
                    dataset_size = self.parameters_influence['dataset_size1']
                    num_epochs = self.parameters_influence['n_epochs1']
                else:
                    dataset_size = self.parameters_influence['dataset_size2']
                    num_epochs = self.parameters_influence['n_epochs2']
                mean_episodic_return = self.data_collector.run(dataset_size, log=True)
                self.influence.train(num_epochs)
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
        # server.stop()

def add_mongodb_observer():
    """
    connects the experiment instance to the mongodb database
    """
    MONGO_HOST = 'TUD-tm2'
    MONGO_DB = 'scalable-simulations'
    PKEY = '~/.ssh/id_rsa'
    global server
    try:
        print("Trying to connect to mongoDB '{}'".format(MONGO_DB))
        server = SSHTunnelForwarder(
            MONGO_HOST,
            ssh_pkey=PKEY,
            remote_bind_address=('127.0.0.1', 27017)
            )
        server.start()
        DB_URI = 'mongodb://localhost:{}/scalable-simulations'.format(server.local_bind_port)
        # pymongo.MongoClient('127.0.0.1', server.local_bind_port)
        ex.observers.append(MongoObserver.create(DB_URI, db_name=MONGO_DB, ssl=False))
        print("Added MongoDB observer on {}.".format(MONGO_DB))
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print(e)
        print("ONLY FILE STORAGE OBSERVER ADDED")
        from sacred.observers import FileStorageObserver
        ex.observers.append(FileStorageObserver.create('saved_runs'))
    
ex = sacred.Experiment('scalable-simulations')
ex.add_config('configs/warehouse/default.yaml')
# add_mongodb_observer()

@ex.automain
def main(parameters, seed, _run):
    exp = Experiment(parameters, _run, seed)
    exp.run()