import gym
import numpy as np
from gym import spaces

class Tmaze(gym.Env):
    
    CORRIDOR_LENGTH = 7
    ACTIONS = {0: 'UP',
               1: 'DOWN',
               2: 'LEFT',
               3: 'RIGHT'}
    OBS_SIZE = CORRIDOR_LENGTH + 1

    def __init__(self, seed):
        self.seed(seed)
        self.max_steps = 100

    def reset(self):
        self.value = np.random.choice(2,1)
        self.location = np.zeros(self.CORRIDOR_LENGTH)
        self.location[0] = 1
        obs = np.append(self.location, self.value)
        self.steps = 0
        return obs
    
    def step(self, action):
        location_idx = np.where(self.location==1)[0][0]
        reward = 0.0
        done = False
        self.steps += 1
        if location_idx == (self.CORRIDOR_LENGTH - 1):
            if action == 0:
                done = True
                if self.value == 0:
                    reward = 1.0
            if action == 1:
                done = True
                if self.value == 1:
                    reward = 1.0
        if action == 2:
            if location_idx > 0:
                self.location = np.zeros(self.CORRIDOR_LENGTH)
                self.location[location_idx-1] = 1
        if action == 3:
            if location_idx < (self.CORRIDOR_LENGTH - 1):
                self.location = np.zeros(self.CORRIDOR_LENGTH)
                self.location[location_idx+1] = 1
        if self.steps >= self.max_steps:
            done = True
        obs = np.append(self.location, 0)
        return obs, reward, done, {}

    def render(self):
        pass
        
    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(self.OBS_SIZE,))

    @property
    def action_space(self):
        """
        Returns A gym dict containing the number of action choices for all the
        agents in the environment
        """
        return spaces.Discrete(len(self.ACTIONS))

            
                

