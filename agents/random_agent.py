from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.Environment import Env
from aienvs.gym.DecoratedSpace import DecoratedSpace
import logging
import random
from gym.spaces import Dict


class RandomAgent(AtomicAgent):
    """
    A simple agent component represents a single agent
    This implementation selects random actions
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def step(self, obs):
        """
        Selects a random action
        """
        action = random.randint(0, self.action_space - 1)
        return action
