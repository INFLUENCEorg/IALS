from simulators.worker import Worker
import multiprocessing as mp
import numpy as np
import random

class DistributedSimulation(object):
    """
    Creates multiple instances of an environment to run in parallel.
    Each of them contains a separate worker (actor) all of them following
    the same policy
    """

    def __init__(self, env, simulator, num_workers, influence_model, seed):
        print('Total number of CPUs {}'.format(mp.cpu_count()))
        if num_workers > mp.cpu_count():
            num_workers = mp.cpu_count()
        print("Number of workers {}. ".format(num_workers))        
        # Random seed is different for each worker (seed + worker_id). Otherwise multiprocessing takes the current system time
        # which is the same for all workers!  
        self.workers = [Worker(env, simulator, influence_model, seed + worker_id) for worker_id in range(num_workers)]

    def reset(self):
        """
        Resets each of the environment instances
        """
        for worker in self.workers:
            worker.child.send(('reset', None))
        output = {'obs': [], 'prev_action': [], 'done': [], 'reward': [], 'dset': [], 'infs': []}
        for worker in self.workers:
            obs, reward, done, dset, infs = worker.child.recv()
            output['obs'].append(obs)
            output['done'].append(done)
            output['reward'].append(reward)
            output['dset'].append(dset)
            output['infs'].append(infs)
        return output

    def step(self, actions):
        """
        Takes an action in each of the enviroment instances
        """
        for worker, action in zip(self.workers, actions):
            worker.child.send(('step', action))
        output = {'obs': [], 'reward': [], 'done': [], 'prev_action': [],
                  'dset': [], 'infs': []}

        for worker in self.workers:
            obs, reward, done, dset, infs = worker.child.recv()
            output['obs'].append(obs)
            output['reward'].append(reward)
            output['done'].append(done)
            output['dset'].append(dset)
            output['infs'].append(infs)
        return output

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
        