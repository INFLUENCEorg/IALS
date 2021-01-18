from influence.data_collector import DataCollector
import os

class InfluenceUniform(object):
    """
    """
    def __init__(self, parameters):
        """
        """
        self.n_sources = parameters['n_sources']
        self.output_size = parameters['output_size']
        self.aug_obs = parameters['aug_obs']
        self.strength = 1
        self.probs = parameters['probs']

    def train(self):
        pass

    def predict(self, obs):
        if self.probs != 0:
            return self.probs
        self.probs = [[1/self.output_size]*self.output_size]*self.n_sources
        return self.probs
    
    def reset(self):
        pass

    def _load_model(self):
        pass
