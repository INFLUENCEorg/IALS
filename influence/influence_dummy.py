class InfluenceDummy(object):
    """
    """
    def __init__(self, parameters):
        """
        """
        self.n_sources = parameters['n_sources']
        self.output_size = parameters['output_size']
        self.aug_obs = parameters['aug_obs']
        self.strength = 1

    def train(self):
        pass

    def predict(self, obs):
        probs = []
        return probs
    
    def reset(self):
        pass

    def _load_model(self):
        pass
