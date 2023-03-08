import multiprocessing as mp
import multiprocessing.connection

def worker_process(remote: multiprocessing.connection.Connection, env_type,
                   simulator, seed, influence=None):
    """
    This function is used as target by each of the threads in the multiprocess
    to build environment instances and define the commands that can be executed
    by each of the workers.
    """
    if env_type == 'warehouse':
        if simulator == 'local':
            from simulators.warehouse.warehouse.envs.local_warehouse import LocalWarehouse
            env = LocalWarehouse(influence)
        else:
            from simulators.warehouse.warehouse.envs.warehouse import Warehouse
            env = Warehouse()
    if env_type == 'traffic':
        if simulator == 'local':
            from simulators.traffic.partial_traffic import PartialTraffic
            env = PartialTraffic(influence)
        else:
            from simulators.traffic.global_traffic import GlobalTraffic
            env = GlobalTraffic()
    env.seed(seed)
        
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, inf = env.step(data)
            if done:
                obs = env.reset()
            remote.send((obs, reward, done, inf))
        elif cmd == 'reset':
            remote.send(env.reset())
        elif cmd == 'action_space':
            remote.send(env.action_space.n)
        elif cmd == 'load':
            env.load_influence_model()
        elif cmd == 'close':
            env.close()
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
    def __init__(self, env, simulator, influence, seed):

        self.child, parent = mp.Pipe()
        self.process = mp.Process(target=worker_process, args=(parent, env, simulator, influence, seed))
        print(self.process)
        self.process.start()
