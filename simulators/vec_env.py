from simulators.worker import Worker
import multiprocessing as mp
import numpy as np
import random

class VecEnv(object):
    """
    Creates multiple instances of an environment to run in parallel.
    Each of them contains a separate worker (actor) all of them following
    the same policy
    """

    def __init__(self, env, simulator, num_workers, seed, influence_model=None):
        print('Total number of CPUs {}'.format(mp.cpu_count()))
        if num_workers > mp.cpu_count():
            num_workers = mp.cpu_count()
        print("Number of workers {}. ".format(num_workers))        
        # Random seed needs to be set different for each worker (seed + worker_id). Otherwise multiprocessing takes 
        # the current system time, which is the same for all workers!
        self.workers = [Worker(env, simulator, seed + worker_id, influence_model) for worker_id in range(num_workers)]

    def reset(self):
        """
        Resets each of the environment instances
        """
        for worker in self.workers:
            worker.child.send(('reset', None))
        obs = []
        for worker in self.workers:
            o =  worker.child.recv()
            obs.append(o)
        return obs

    def step(self, actions):
        """
        Takes an action in each of the enviroment instances
        """
        for worker, action in zip(self.workers, actions):
            worker.child.send(('step', action))
        output = {'obs': [], 'reward': [], 'done': [], 'prev_action': [],
                  'dset': [], 'infs': []}
        obs = []
        reward = []
        done = []
        inf = []
        for worker in self.workers:
            o, r, d, i = worker.child.recv()
            obs.append(o)
            reward.append(r)
            done.append(d)
            inf.append(i)
        return obs, reward, done, inf

    def action_space(self):
        """
        Returns the dimensions of the environment's action space
        """
        self.workers[0].child.send(('action_space', None))
        action_space = self.workers[0].child.recv()
        return action_space

    def close(self):
        """
        Closes each of the threads in the multiprocess
        """
        for worker in self.workers:
            worker.child.send(('close', None))

    def load_influence_model(self):
        """
        Loads the newest influence model
        """
        for worker in self.workers:
            worker.child.send(('load', None))
        