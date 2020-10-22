import sys
sys.path.append("..") 
from agents.PPO.PPOmodel import PPOmodel
from agents.PPO.buffer import SerialSampling
import numpy as np
import os
import tensorflow as tf


class PPOAgent(object):
    """
    PPOAgent
    """

    def __init__(self, action_map, parameters):
        self.parameters = parameters
        self.num_actions = action_map
        self.model = PPOmodel(self.parameters,
                              self.num_actions)
        self.buffer = SerialSampling(self.parameters,
                                     self.num_actions)
        self.cumulative_rewards = 0
        self.episode_step = 0
        self.episodes = 0
        self.t = 0
        self.stats = {"cumulative_rewards": [],
                      "episode_length": [],
                      "value": [],
                      "learning_rate": [],
                      "entropy": [],
                      "policy_loss": [],
                      "value_loss": []}
        if self.parameters['influence']:
            self.seq_len = self.parameters['inf_seq_len']
        elif self.parameters['recurrent']:
            self.seq_len = self.parameters['seq_len']
        else:
            self.seq_len = 1
        tf.reset_default_graph()
        self.step = 0
        summary_path = '../summaries/' + self.parameters['name']
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        self.summary_writer = tf.summary.FileWriter(summary_path)
        self._step_output = None
        self._prev_action = [-1]*self.parameters['num_workers']

    def take_action(self, step_output, mode='train'):
        """
        Get each factor's action based on its local observation. Append the given
        state to the factor's replay memory.
        """
        take_action_output = self.model.evaluate_policy(step_output['obs'], self._prev_action)
        if mode == 'train':
            if self.step != 0:
                # Store experiences in buffer.
                self._add_to_memory(self._prev_step_output, step_output,
                                    self._prev_action_output, self._prev_action)
                # Estimate the returns using value function when time
                # horizon has been reached
                self._bootstrap(step_output['obs'], self._prev_action)    
                self._update()
                self._write_summary()
                self._save_graph()
            self._increment_step()
        self._prev_step_output = step_output
        self._prev_action_output = take_action_output
        # copying values using np.copy otherwise the prev action object mutates
        self._prev_action = [-1]*self.parameters['num_workers']#np.copy(take_action_output['action'])
        for worker in range(self.parameters['num_workers']):
            if step_output['done'][worker]:
                # setting prev_action to -1 for next episode
                self._prev_action[worker] = -1
        return take_action_output['action']


    ######################### Private Functions ###########################

    def _add_to_memory(self, step_output, next_step_output, get_actions_output, prev_action):
        """
        Append the last transition to buffer and to stats
        """
        self.buffer['obs'].append(step_output['obs'])
        self.buffer['rewards'].append(next_step_output['reward'])
        self.buffer['dones'].append(next_step_output['done'])
        self.buffer['actions'].append(get_actions_output['action'])
        self.buffer['values'].append(get_actions_output['value'])
        self.buffer['action_probs'].append(get_actions_output['action_probs'])
        # This mask is added so we can ignore experiences added when
        # zero-padding incomplete sequences
        self.buffer['masks'].append([1]*self.parameters['num_workers'])
        self.cumulative_rewards += next_step_output['reward'][0]
        self.episode_step += 1
        self.stats['value'].append(get_actions_output['value'][0])
        self.stats['entropy'].append(get_actions_output['entropy'][0])
        self.stats['learning_rate'].append(get_actions_output['learning_rate'])

        if self.parameters['recurrent']:
            self.buffer['states_in'].append(
                    np.transpose(get_actions_output['state_in'], (1,0,2)))
            self.buffer['prev_actions'].append(prev_action)
        if self.parameters['influence']:
            self.buffer['inf_states_in'].append(
                    np.transpose(get_actions_output['inf_state_in'], (1,0,2)))
            self.buffer['inf_prev_actions'].append(prev_action)
        if next_step_output['done'][0]:
            self.episodes += 1
            self.stats['cumulative_rewards'].append(self.cumulative_rewards)
            self.stats['episode_length'].append(self.episode_step)
            self.cumulative_rewards = 0
            self.episode_step = 0
        if self.parameters['recurrent'] or self.parameters['influence']:
            for worker, done in enumerate(next_step_output['done']):
                if done and self.parameters['num_workers'] != 1:
                    # reset worker's internal state
                    self.model.reset_state_in(worker)
                    # zero padding incomplete sequences
                    remainder = len(self.buffer['masks']) % self.seq_len
                    # NOTE: we need to zero-pad all workers to keep the
                    # same buffer dimensions even though only one of them has
                    # reached the end of the episode.
                    if remainder != 0:
                        missing = self.seq_len - remainder
                        self.buffer.zero_padding(missing, worker)
                        self.t += missing

    def _bootstrap(self, obs, prev_action):
        """
        Computes GAE and returns for a given time horizon
        """
        # TODO: consider the case where the episode is over because the maximum
        # number of steps in an episode has been reached.
        self.t += 1
        if self.t >= self.parameters['time_horizon']:
            evaluate_value_output = self.model.evaluate_value(obs, prev_action)
            next_value = evaluate_value_output['value']
            batch = self.buffer.get_last_entries(self.t, ['rewards', 'values',
                                                          'dones'])
            advantages = self._compute_advantages(np.array(batch['rewards']),
                                                  np.array(batch['values']),
                                                  np.array(batch['dones']),
                                                  next_value,
                                                  self.parameters['gamma'],
                                                  self.parameters['lambda'])
            self.buffer['advantages'].extend(advantages)
            returns = advantages + batch['values']
            self.buffer['returns'].extend(returns)
            self.t = 0

    def _update(self):
        """
        Runs multiple epoch of mini-batch gradient descent to update the model
        using experiences stored in buffer.
        """
        if self.step % self.parameters['train_frequency'] == 0 and self._full_memory():
            policy_loss = 0
            value_loss = 0
            n_sequences = self.parameters['batch_size'] // self.seq_len
            n_batches = self.parameters['memory_size'] // \
                self.parameters['batch_size']
            for _ in range(self.parameters['num_epoch']):
                self.buffer.shuffle()
                for b in range(n_batches):
                    batch = self.buffer.sample(b, n_sequences)
                    update_model_output = self.model.update_model(batch)
                    policy_loss += update_model_output['policy_loss']
                    value_loss += update_model_output['value_loss']
            self.buffer.empty()
            self.stats['policy_loss'].append(np.mean(policy_loss))
            self.stats['value_loss'].append(np.mean(value_loss))
        
    def _compute_advantages(self, rewards, values, dones, last_value, gamma,
                            lambd):
        """
        Calculates advantages using genralized advantage estimation (GAE)
        """
        last_advantage = 0
        advantages = np.zeros((self.parameters['time_horizon'],
                               self.parameters['num_workers']),
                              dtype=np.float32)
        for t in reversed(range(self.parameters['time_horizon'])):
            mask = 1.0 - dones[t, :]
            last_value = last_value*mask
            last_advantage = last_advantage*mask
            delta = rewards[t, :] + gamma*last_value - values[t, :]
            last_advantage = delta + gamma*lambd*last_advantage
            advantages[t, :] = last_advantage
            last_value = values[t, :]
        return advantages

    def _increment_step(self):
        self.model.increment_step()
        self.step = self.model.get_current_step()

    def _write_summary(self):
        """
        Saves training statistics to Tensorboard.
        """
        if self.step % self.parameters['summary_frequency'] == 0 and \
           self.parameters['tensorboard']:
            summary = tf.Summary()
            for key in self.stats.keys():
                if len(self.stats[key]) > 0:
                    stat_mean = float(np.mean(self.stats[key]))
                    summary.value.add(tag='{}'.format(key), simple_value=stat_mean)
                    self.stats[key] = []
            self.summary_writer.add_summary(summary, self.step)
            self.summary_writer.flush()

    def _store_memory(self, path):
        # Create factor path if it does not exist.
        path = os.path.join(os.environ['APPROXIMATOR_HOME'], path)
        if not os.path.exists(path):
            os.makedirs(path)
        # Store the replay memory
        self.buffer.store(path)

    def _full_memory(self):
        """
        Check if the replay memories are filled.
        """
        return self.buffer.full()

    def _save_graph(self):
        """
        Store all the networks and replay memories.
        """
        if self.step % self.parameters['save_frequency'] == 0:
            # Create factor path if it does not exist.
            path = os.path.join('models', self.parameters['name'])
            if not os.path.exists(path):
                os.makedirs(path)
            self.model.save_graph(self.step)