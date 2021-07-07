import os
import sys
sys.path.append("..")
from influence.influence_network import InfluenceNetwork
from influence.influence_uniform import InfluenceUniform
# from stable_baselines3.common import set_global_seeds
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from recurrent_policies.PPO import Agent, FNNPolicy
import gym
import sacred
from sacred.observers import MongoObserver
import pymongo
from sshtunnel import SSHTunnelForwarder
import numpy as np
import csv
import os
import time

def generate_path(path):
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


def log(dset, infs, data_path):
    """
    Log influence dataset
    """
    generate_path(data_path)
    inputs_file = data_path + str('inputs.csv')
    targets_file = data_path + str('targets.csv')
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

def make_env(env_id, rank, seed=0, influence=None):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        if 'local' in env_id:
            env = gym.make(env_id, influence=influence)
        else:
            env = gym.make(env_id)
        # env = Monitor(env, './logs')
        env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init

def add_mongodb_observer():
    """
    connects the experiment instance to the mongodb database
    """
    MONGO_HOST = 'TUD-tm2'
    MONGO_DB = 'scalable-simulations'
    PKEY = '~/.ssh/id_rsa'
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

class Experiment(object):
    """
    Creates experiment object to interact with the environment and
    the agent and log results.
    """

    def __init__(self, parameters, _run, seed):
        """
        """
        self._run = _run
        self._seed = seed
        self.parameters = parameters['main']

        policy = FNNPolicy(self.parameters['obs_size'], 
            self.parameters['num_actions'], 
            self.parameters['num_workers']
            )
        self.agent = Agent(
            policy=policy,
            memory_size=self.parameters['memory_size'],
            batch_size=self.parameters['batch_size'],
            seq_len=self.parameters['seq_len'],
            num_epoch=self.parameters['num_epoch'],
            learning_rate=self.parameters['learning_rate'],
            total_steps=self.parameters['total_steps'],
            clip_range=self.parameters['epsilon'],
            entropy_coef=self.parameters['beta']
            )

        global_env_name = self.parameters['env']+ ':' + self.parameters['env'] + '-v0'
        self.global_env = SubprocVecEnv(
            [make_env(global_env_name, i, seed) for i in range(self.parameters['num_workers'])]
            )
        # self.global_env = VecNormalize(self.global_env, norm_obs=True, norm_reward=True)

        if self.parameters['simulator'] == 'local':
            data_path = parameters['influence']['data_path'] + str(_run._id) + '/'

            if self.parameters['influence_model'] == 'nn':
                influence = InfluenceNetwork(parameters['influence'], data_path, _run._id)
                self.collect_data(parameters['influence']['dataset_size1'], data_path)
                loss = influence.learn()
                self._run.log_scalar('influence loss', loss, 0)
            
            else:
                influence = InfluenceUniform(parameters['influence'], data_path)

            local_env_name = self.parameters['env']+ ':local-' + self.parameters['env'] + '-v0'
            self.env = SubprocVecEnv(
                [make_env(local_env_name, i, seed, influence) for i in range(self.parameters['num_workers'])]
                )
        else:
            self.env = self.global_env

    def run(self):

        obs = self.env.reset()
        step = 0
        episode_reward = 0
        episode_step = 0
        episode = 1
        start = time.time()
        while step < self.parameters['total_steps']:

            rollout_step = 0
            while rollout_step < self.parameters['rollout_steps']:
                if step % self.parameters['eval_freq'] == 0:
                   mean_return = self.evaluate()
                   self._run.log_scalar('mean episodic return', mean_return, step)
                action, value, log_prob, hidden_memory = self.agent.choose_action(obs)
                new_obs, reward, done, info = self.env.step(action)
                self.agent.add_to_memory(obs, action, reward, done, value, log_prob, hidden_memory)
                obs = new_obs
                rollout_step += 1
                step += 1
                episode_step += 1
                episode_reward += np.mean(reward)
                if done[0]:
                    end = time.time()
                    print('Time: ', end - start)
                    start = end
                    self.print_results(episode_reward, episode_step, step, episode)
                    episode_reward = 0
                    episode_step = 0
                    episode += 1
            
            self.agent.bootstrap(
                obs, 
                self.parameters['rollout_steps'], 
                self.parameters['gamma'], 
                self.parameters['lambda']
                )

            if self.agent.buffer.is_full:
                self.agent.update()
            

    def collect_data(self, agent, dataset_size, data_path):
        """Collect data from global simulator"""
        print('Collecting data from global simulator...')
        episode_rewards =[]
        n_steps = 0
        while n_steps < dataset_size//self.parameters['num_workers']:
            reward_sum = 0
            done = [False]*self.parameters['num_workers']
            obs = self.global_env.reset()
            dset = []
            infs = []
            # NOTE: Episodes in all envs must terminate at the same time 
            while not done[0]:
                n_steps += 1
                action, _, _, _ = agent.choose_action(obs)
                obs, reward, done, info = self.global_env.step(action)
                dset.append(np.array([i['dset'] for i in info]))
                infs.append(np.array([i['infs'] for i in info]))
                reward_sum += reward
            episode_rewards.append(reward_sum)
            log(dset, infs, data_path)
        print('Done!')
        return np.mean(episode_rewards)

    def evaluate(self):
        """Return mean sum of episodic rewards) for given model"""
        episode_rewards = []
        n_steps = 0
        print('Evaluating policy on global simulator...')
        while n_steps < self.parameters['eval_steps']//self.parameters['num_workers']:
            reward_sum = 0
            done = [False]*self.parameters['num_workers']
            obs = self.global_env.reset()
            # NOTE: Episodes in all envs must terminate at the same time
            while not done[0]:
                n_steps += 1
                action, _, _, _ = self.agent.choose_action(obs)
                obs, reward, done, _ = self.global_env.step(action)
                reward_sum += reward
            episode_rewards.append(reward_sum)
        print('Done!')
        return np.mean(episode_rewards)
        

    def print_results(self, episode_return, episode_step, global_step, episode):
        """
        Prints results to the screen.
        """
        print(("Train step {} of {}".format(global_step,
                                            self.parameters['total_steps'])))
        print(("-"*30))
        print(("Episode {} ended after {} steps.".format(episode,
                                                         episode_step)))
        print(("- Total reward: {}".format(episode_return)))
        print(("-"*30))
        
ex = sacred.Experiment('scalable-simulations')
ex.add_config('configs/default.yaml')
add_mongodb_observer()

@ex.automain
def main(parameters, seed, _run):
    exp = Experiment(parameters, _run, seed)
    exp.run()
