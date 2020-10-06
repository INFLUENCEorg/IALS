from influence.data_collector import DataCollector

class InfluenceUniform(object):
    """
    """
    def __init__(self, agent, simulator, parameters, run_id):
        """
        """
        self.n_sources = parameters['n_sources']
        self.output_size = parameters['output_size']
        self.aug_obs = parameters['influence_aug_obs']
        self.strength = 1
        self.data_collector = DataCollector(agent, simulator, self, self.aug_obs,
                                            run_id, parameters['dataset_size'])

    def train(self):
        mean_episodic_return = self.data_collector.run()
        return mean_episodic_return

    def predict(self, obs):
        probs = []
        for s in range(self.n_sources):
            probs.append([1/self.output_size[s]]*self.output_size[s])
        return probs
    
    def reset(self):
        pass

    def _load_model(self):
        pass
