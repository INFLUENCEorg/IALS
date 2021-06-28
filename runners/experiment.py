import os
import sys
sys.path.append("..")
from influence.influence_network import InfluenceNetwork
from influence.influence_uniform import InfluenceUniform
from agent.a2c_ppo_acktr.model import Policy
from agent.a2c_ppo_acktr import algo, utils
from agent.a2c_ppo_acktr.storage import RolloutStorage
from simulators.envs import *
import time
import sacred
from collections import deque
from gym import spaces
from collect_data import collect_data
from evaluate import evaluate
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
        self._run = _run
        self._seed = seed
        self.parameters = parameters['main']
        self.actor_critic = Policy((37,), 
                              spaces.Discrete(self.parameters['num_actions']), 
                              base_kwargs={'recurrent': self.parameters['recurrent']})
        data_path = parameters['influence']['data_path'] + str(_run) + '/'
        self.global_env_name = self.parameters['env']+ ':' + self.parameters['env'] + '-v0'
        self.path = self.generate_path(self.parameters['env'])
        if self.parameters['simulator'] == 'local':
            self.local_env_name = self.parameters['env']+ ':local-' + self.parameters['env'] + '-v0'

            if self.parameters['influence_model'] == 'nn':
                self.influence = InfluenceNetwork(parameters['influence'], data_path, _run)
                collect_data(self.actor_critic, self.global_env_name, seed, self.parameters['num_workers'], 
                             None, 'cpu', 1000, data_path)
                loss = self.influence.train()
                self._run.log_scalar('influence loss', loss, 0)
            else:
                self.influence = InfluenceUniform(parameters['influence'], data_path)

            self.sim = make_vec_envs(self.local_env_name, seed, self.parameters['num_workers'], 
                self.parameters['gamma'], './logs', 'cpu', True, influence=self.influence)

        else:
            env_name = self.parameters['env']+ ':' + self.parameters['env'] + '-v0'
            self.sim = make_vec_envs(env_name, seed, self.parameters['num_workers'], 
                self.parameters['gamma'], './logs', 'cpu', True)

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

    def run(self):
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        torch.set_num_threads(1)
        device = 'cpu'
        self.actor_critic.to(device)

        
        agent = algo.PPO(self.actor_critic, self.parameters['epsilon'], 
            self.parameters['num_epoch'], 4, self.parameters['c1'],
            self.parameters['beta'], self.parameters['learning_rate'],
            1e-5, max_grad_norm=0.5)

        rollouts = RolloutStorage(self.parameters['time_horizon'], self.parameters['num_workers'],
                self.sim.observation_space.shape, self.sim.action_space,
                self.actor_critic.recurrent_hidden_state_size)

        obs = self.sim.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        episode_rewards = deque(maxlen=10)

        start = time.time()
        num_updates = int(self.parameters['max_steps']) // self.parameters['time_horizon']
        for j in range(num_updates):

            utils.update_linear_schedule(agent.optimizer, j, num_updates, 
            self.parameters['learning_rate'])

            for step in range(self.parameters['time_horizon']):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

                # Obser reward and next obs
                obs, reward, done, infos = self.sim.step(action)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, True, self.parameters['gamma'],
                                    self.parameters['lambda'], False)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            # save for every interval-th episode or for the last epoch
            if (j % self.parameters['save_frequency'] == 0
                    or j == num_updates - 1):
                try:
                    os.makedirs(self.path)
                except OSError:
                    pass

                torch.save([
                    self.actor_critic,
                    getattr(utils.get_vec_normalize(self.sim), 'obs_rms', None)
                ], os.path.join(self.path + ".pt"))

            if j % 10 == 0 and len(episode_rewards) > 1:
                total_num_steps = j*self.parameters['time_horizon']
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

            if j % self.parameters['eval_freq'] == 0:
                mean_episodic_return = evaluate(self.actor_critic, 
                                        self.global_env_name, self._seed,
                                        self.parameters['num_workers'], 'cpu',
                                        self.parameters['eval_steps'])
                self._run.log_scalar('mean episodic return', mean_episodic_return, self.parameters['time_horizon']*j)
        # self.sim.close()

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
    
ex = sacred.Experiment('scalable-simulations')
ex.add_config('configs/default.yaml')
add_mongodb_observer()

@ex.automain
def main(parameters, seed, _run):
    exp = Experiment(parameters, _run, seed)
    exp.run()