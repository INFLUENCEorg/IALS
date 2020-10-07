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
        self.strength = 1

    def train(self, step):
        pass

    def predict(self, obs):
        probs = []
        for s in range(self.n_sources):
            probs.append([1/self.output_size[s]]*self.output_size[s])
        return probs
    
    def reset(self):
        pass

    def _load_model(self):
        pass
