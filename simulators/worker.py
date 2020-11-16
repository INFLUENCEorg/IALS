import multiprocessing as mp
import multiprocessing.connection
from simulators.warehouse.warehouse import Warehouse
from simulators.warehouse.partial_warehouse import PartialWarehouse
from simulators.traffic.global_traffic import GlobalTraffic
from simulators.traffic.partial_traffic import PartialTraffic


def worker_process(remote: multiprocessing.connection.Connection, env_type,
                   simulator, worker_id, influence, seed):
    """
    This function is used as target by each of the threads in the multiprocess
    to build environment instances and define the commands that can be executed
    by each of the workers.
    """
    if env_type == 'warehouse':
        if simulator == 'partial':
            env = PartialWarehouse(influence, seed+worker_id)
        else:
            env = Warehouse(influence, seed+worker_id)
    if env_type == 'traffic':
        if simulator == 'partial':
            env = PartialTraffic(influence, seed+worker_id)
        else:
            env = GlobalTraffic(seed+worker_id)
        
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, dset, infs = env.step(data)
            if done:
                obs, _, _, dset, infs = env.reset()
            remote.send((obs, reward, done, dset, infs))
        elif cmd == 'reset':
            remote.send(env.reset())
        elif cmd == 'action_space':
            remote.send(env.action_space.n)
        elif cmd == 'load':
            env.load_influence_model()
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
    def __init__(self, env, simulator, worker_id, influence, seed):

        self.child, parent = mp.Pipe()
        self.process = mp.Process(target=worker_process, args=(parent, env, simulator, worker_id, influence, seed))
        self.process.start()
