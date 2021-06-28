import os
import sys
sys.path.append("..")
from influence.influence_network import InfluenceNetwork
from influence.influence_uniform import InfluenceUniform
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import FeedForwardPolicy, LstmPolicy
from stable_baselines.bench import Monitor
import gym
from stable_baselines import PPO2
import sacred
from sacred.observers import MongoObserver
import pymongo
from sshtunnel import SSHTunnelForwarder
import numpy as np
import csv
import os

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
        env = Monitor(env, './logs')
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
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

class CustomMlpPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[512, 256],
                                           feature_extraction="mlp")

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[512, 'lstm'],
                         layer_norm=False, feature_extraction="mlp", **_kwargs)


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
        global_env_name = self.parameters['env']+ ':' + self.parameters['env'] + '-v0'
        self.global_env = SubprocVecEnv(
            [make_env(global_env_name, i, seed) for i in range(self.parameters['num_workers'])]
            )
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

        # learning_rate = LinearSchedule(
        #         self.parameters['max_steps']*self.parameters['num_workers'], 
        #         final_p = 0, initial_p = self.parameters['learning_rate']
        #         )
        
        self.model = PPO2(CustomLSTMPolicy, self.env, n_steps=8, verbose=1, learning_rate=self.parameters['learning_rate'])
        # mean_return = self.evaluate(self.parameters['eval_steps'])
        # self._run.log_scalar('mean episodic return', mean_return, 0)

    def run(self):
        eval_steps = self.parameters['eval_steps']//self.parameters['num_workers']
        evaluate_callback = EvaluateCallback(self.parameters['eval_freq'], self.global_env, 
            self.model, eval_steps, self.parameters['num_workers'], self._run)
        self.model.learn(
            total_timesteps=int(self.parameters['max_steps']*self.parameters['num_workers']), 
            callback=evaluate_callback
            )

    def collect_data(self, dataset_size, data_path):
        """Collect data from global simulator"""
        print('Collecting data from global simulator...')
        model = PPO2(MlpPolicy, self.global_env)
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
                action, _ = model.predict(obs)
                obs, reward, done, info = self.global_env.step(action)
                dset.append(np.array([i['dset'] for i in info]))
                infs.append(np.array([i['infs'] for i in info]))
                reward_sum += reward
            episode_rewards.append(reward_sum)
            log(dset, infs, data_path)
        print('Done!')
        return np.mean(episode_rewards)

class EvaluateCallback(BaseCallback):
    """
    Callback for evaluating the model on the global simulator
    """
    def __init__(self, check_freq: int, global_env, 
        model, eval_steps, num_workers, run, verbose=0, 
        ):
        super(EvaluateCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.global_env = global_env
        self.eval_steps = eval_steps
        self.num_workers = num_workers
        self.model = model
        self.run = run

    def _on_step(self) -> bool:
        if (self.n_calls-1) % self.check_freq == 0:
            print('Evaluating...')
            self.evaluate()
            print('Done!')


    def evaluate(self):
        """Return mean sum of episodic rewards) for given model"""
        episode_rewards = []
        n_steps = 0
        while n_steps < self.eval_steps:
            reward_sum = 0
            done = [False]*self.num_workers
            obs = self.global_env.reset()
            # NOTE: Episodes in all envs must terminate at the same time
            while not done[0]:
                n_steps += 1
                action, _ = self.model.predict(obs)
                obs, reward, done, _ = self.global_env.step(action)
                reward_sum += reward
            episode_rewards.append(reward_sum)
        mean_episode_rewards = np.mean(episode_rewards)
        self.run.log_scalar('mean episodic return', mean_episode_rewards, self.n_calls-1)


    
ex = sacred.Experiment('scalable-simulations')
ex.add_config('configs/default.yaml')
add_mongodb_observer()

@ex.automain
def main(parameters, seed, _run):
    exp = Experiment(parameters, _run, seed)
    exp.run()