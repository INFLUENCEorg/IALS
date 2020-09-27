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

    def __init__(self, parameters, influence):
        print('cpu count', mp.cpu_count())
        if parameters['num_workers'] < mp.cpu_count():
            self.num_workers = parameters['num_workers']
        else:
            self.num_workers = mp.cpu_count()
        self.workers = [Worker(parameters, i, influence) for i in range(self.num_workers)]

    def reset(self):
        """
        Resets each of the environment instances
        """
        for worker in self.workers:
            worker.child.send(('reset', None))
        output = {'obs': [], 'prev_action': [], 'done': []}
        for worker in self.workers:
            obs = worker.child.recv()
            output['obs'].append(obs)
            output['prev_action'].append(-1)
            output['done'].append(False)
        return output

    def step(self, actions):
        """
        Takes an action in each of the enviroment instances
        """
        for worker, action in zip(self.workers, actions):
            worker.child.send(('step', action))
        output = {'obs': [], 'reward': [], 'done': [], 'prev_action': [],
                  'info': []}
        i = 0
        for worker in self.workers:
            obs, reward, done, info = worker.child.recv()
            output['obs'].append(obs)
            output['reward'].append(reward)
            output['done'].append(done)
            output['info'].append(info)
            i += 1
        output['prev_action'] = actions
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
