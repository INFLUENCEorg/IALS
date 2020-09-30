import logging
import random
from gym.spaces import Dict


class RandomAgent(object):
    """
    A simple agent component represents a single agent
    This implementation selects random actions
    """

    def __init__(self, action_space, parameters=None):
        self.action_space = action_space
        self.episodes = 0

    def take_action(self, step_output, mode):
        """
        Selects a random action
        """
        obs = step_output['obs']
        action = [random.randint(0, self.action_space - 1) for _ in range(len(obs))]
        if step_output['done'][0] == True:
            self.episodes += 1
        return action
