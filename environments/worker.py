import multiprocessing
import multiprocessing.connection
from environments.warehouse.warehouse import Warehouse
from environments.warehouse.partial_warehouse import PartialWarehouse
import os


def worker_process(remote: multiprocessing.connection.Connection, parameters,
                   worker_id):
    """
    This function is used as target by each of the threads in the multiprocess
    to build environment instances and define the commands that can be executed
    by each of the workers.
    """
    # The Atari wrappers are now imported from openAI baselines
    # https://github.com/openai/baselines
    log_dir = './log'
    if parameters['env'] == 'warehouse':
        if parameters['simulator'] == 'partial':
            env = PartialWarehouse()
        else:
            env = Warehouse()
        
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, info = env.step(data)
            if parameters['env'] == 'atari':
                done = False
                if 'episode' in info.keys():
                    done = True
                    obs = env.reset()
            else:
                if done:
                    obs = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == 'reset':
            remote.send(env.reset())
        elif cmd == 'action_space':
            remote.send(env.action_space.n)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker(object):
    """
    Creates workers (actors) and starts single parallel threads in the
    multiprocess. Commands can be send and outputs received by calling
    child.send() and child.recv() respectively
    """
    def __init__(self, parameters, worker_id):

        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process,
                                               args=(parent, parameters,
                                                     worker_id))
        self.process.start()
