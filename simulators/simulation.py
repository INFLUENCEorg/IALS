from simulators.worker import Worker
import multiprocessing as mp
import numpy as np
import random

class Simulation(object):
    """
    Creates multiple instances of an environment to run in parallel.
    Each of them contains a separate worker (actor) all of them following
    the same policy
    """

    def __init__(self, env_type, simulator, influence_model, seed):
        if env_type == 'warehouse':
            if simulator == 'partial':
                from simulators.warehouse.partial_warehouse import PartialWarehouse
                self.sim = PartialWarehouse(influence_model, seed)
            else:
                from simulators.warehouse.warehouse import Warehouse
                self.sim = Warehouse(influence_model, seed)
        if env_type == 'traffic':
            if simulator == 'partial':
                from simulators.traffic.partial_traffic import PartialTraffic
                self.sim = PartialTraffic(influence_model, seed)
            else:
                from simulators.traffic.global_traffic import GlobalTraffic
                self.sim = GlobalTraffic(seed)

    def reset(self):
        """
        Resets each of the environment instances
        """
        output = {'obs': [], 'prev_action': [], 'done': [], 'reward': [], 'dset': [], 'infs': []}
        obs, reward, done, dset, infs = self.sim.reset()
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
        output = {'obs': [], 'reward': [], 'done': [], 'prev_action': [],
                  'dset': [], 'infs': []}
        obs, reward, done, dset, infs = self.sim.step(actions[0])
        if done:
                obs, _, _, dset, infs = self.sim.reset()
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
        action_space = self.sim.action_space
        return action_space

    def close(self):
        """
        Closes each of the threads in the multiprocess
        """    
        self.sim.close()

    def load_influence_model(self):
        """
        Loads the newest influence model
        """
        self.sim.load_influence_model()
        