from aiagents.single.WarehouseNaiveAgent import WarehouseNaiveAgent
from aiagents.single.ConstantAgent import ConstantAgent
from aiagents.single.RandomAgent import RandomAgent
from aiagents.multi.BasicComplexAgent import BasicComplexAgent
from aienvs.runners.Episode import Episode
from aienvs.runners.Experiment import Experiment
from aienvs.Warehouse import Warehouse
from pomegranate import *
import argparse
import numpy as np
import csv

class LearnBayesianNetwork(object):
    """
    """
    def __init__(self):
        parameters = self._parse_args()
        self._episode_len = parameters.episode_len
        self._data_file = parameters.data_file

    def learn(self):
        # self._generate_data()
        data = self._read_data()
        X = self._form_dataset(data)
        model = self._define_network_structure(X)
        # model = BayesianNetwork.from_samples(X, algorithm='exact')
        model.bake()
        model.fit(X)
        # model.bake()
        breakpoint()
        import seaborn, time
        seaborn.set_style('whitegrid')
        model.plot('./file')

    def _generate_data(self):
        print('Generating data...')
        env = Warehouse()
        agents = []
        env.reset()
        for agent_id in env.action_space.spaces.keys():
            agents.append(ConstantAgent(agentId=agent_id, environment=env, parameters={'action':3}))
        agents[20] = RandomAgent(agentId=20, environment=env)
        complexAgent = BasicComplexAgent(agents)
        experiment = Experiment(complexAgent, env, maxSteps=1e4, render=False,
                                seedlist=np.arange(0,30000, 10), renderDelay=0.5)
        experiment.run()
        print('Done')

    def _read_data(self):
        data = []
        with open('obs_data.csv') as data_file:
            csv_reader = csv.reader(data_file, delimiter=',')
            for row in csv_reader:
                data.append([int(element) for element in row])
        return data

    def _form_dataset(self, data):
        n_episodes = len(data)//self._episode_len
        X = []
        for episode in range(n_episodes):
            for i in range(self._episode_len - 1):
                idx = episode*self._episode_len + i
                loc = data[idx][0]
                next_loc = data[idx+1][0]
                action = data[idx][1]
                items = []
                for item, next_item in zip(data[idx][2:], data[idx+1][2:]):
                    items += [item, next_item]
                X.append([loc] + [action] + [next_loc] + items)
        return X

    def _define_network_structure(self, X):
        robot_locs = set(list(np.array(X)[:,0]))
        actions = set(list(np.array(X)[:,1]))
        loc_prob = DiscreteDistribution({loc: 1/len(robot_locs) for loc in robot_locs})
        action_prob = DiscreteDistribution({action: 1/len(actions) for action in actions})
        prob_table = []
        prob = 1/len(robot_locs)
        for loc in robot_locs:
            for action in actions:
                for next_loc in robot_locs:
                    prob_table.append([loc, action, next_loc, prob])
        next_loc_prob = ConditionalProbabilityTable(prob_table, [loc_prob, action_prob])
        prob = 0.5
        loc = State(loc_prob, 'loc')
        action = State(action_prob, 'action')
        next_loc = State(next_loc_prob, 'next_loc')
        model = BayesianNetwork()
        model.add_states(loc, action, next_loc)
        model.add_transition(loc, next_loc)
        model.add_transition(action, next_loc)
        item = []
        item_prob = []
        next_item = []
        next_item_prob = []
        for i in range(10):
            item_prob.append(DiscreteDistribution({0: 0.5, 1: 0.5}))
            item.append(State(item_prob[i], 'item' + str(i)))
            prob_table = []
            for l in robot_locs:
                for a in actions:
                    for item_tf in range(2):
                        for next_item_tf in range(2):
                            prob_table.append([l, a, item_tf, next_item_tf, 0.5])
            next_item_prob.append(ConditionalProbabilityTable(prob_table, [loc_prob, action_prob, item_prob[i]]))
            next_item.append(State(next_item_prob[i], 'next_item' + str(i)))
            model.add_states(item[i], next_item[i])
            model.add_transition(item[i], next_item[i])
            model.add_transition(loc, next_item[i])
            model.add_transition(action, next_item[i])
        return model
        


    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--episode_len', type=int, default=100,
                            help='episode length')
        parser.add_argument('--data_file', type=str, default='obs_data.csv',
                            help='path to the file where observations are stored')
        args = parser.parse_args()
        return args

if __name__ == '__main__':
    BN = LearnBayesianNetwork()
    BN.learn()