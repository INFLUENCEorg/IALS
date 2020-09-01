from pomegranate import *

class BayesianNetworkModel(object):
    """
    Bayesian Network model
    """
    def __init__(self, data):
        self.model = BayesianNetwork.from_samples(data, algorithm='exact')
