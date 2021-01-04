import torch
import argparse
import numpy as np
import csv
import sys
sys.path.append("..") 
# from influence.data_collector import DataCollector
# from agents.random_agent import RandomAgent
# from simulators.warehouse.warehouse import Warehouse
import random
import matplotlib.pyplot as plt
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_memory_size, n_sources, output_size, recurrent):
        super().__init__()
        # self.fc = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        # self.lstm = nn.LSTM(input_size, hidden_memory_size, batch_first=True)
        self.recurrent = recurrent
        if self.recurrent:
            self.gru = nn.GRU(input_size, hidden_memory_size, batch_first=True)
        else:
            self.linear1 = nn.Linear(input_size, hidden_memory_size)
        self.linear2 = nn.ModuleList()
        # self.linear3 = nn.ModuleList()
        self.n_sources = n_sources
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.hidden_memory_size = hidden_memory_size
        for _ in range(self.n_sources):
            self.linear2.append((nn.Linear(hidden_memory_size, output_size)))
            # self.linear3.append((nn.Linear(hidden_memory_size, output_size)))
        self.reset()

    def forward(self, input_seq):
        if self.recurrent:
            out, self.hidden_cell = self.gru(input_seq, self.hidden_cell)
        else:
            out = self.relu(self.linear1(input_seq))
        logits = []
        probs = []
        for k in range(self.n_sources):
            # linear2_out = self.relu(self.linear2[k](out))
            linear2_out = self.linear2[k](out)
            logits.append(linear2_out)
            if np.shape(linear2_out[:, -1, :])[1] > 1: 
                probs.append(self.softmax(linear2_out[:, -1, :]).detach().numpy())
            else:
                probs.append(self.sigmoid(linear2_out[:, -1, :]).detach().numpy())
        return logits, probs
    
    def reset(self):
        # self.hidden_cell = (torch.zeros(1,1,self.hidden_memory_size),
        #                     torch.zeros(1,1,self.hidden_memory_size))
        self.hidden_cell = torch.zeros(1,1,self.hidden_memory_size)

class InfluenceNetwork(object):
    """
    """
    def __init__(self, parameters, data_path, run_id):
        """
        """
        # parameters = read_parameters('../influence/configs/influence.yaml')
        self._seq_len = parameters['seq_len']
        self._episode_length = parameters['episode_length']
        self._lr = parameters['lr']
        self._hidden_memory_size = parameters['hidden_memory_size']
        self._batch_size = parameters['batch_size']
        self.n_sources = parameters['n_sources']
        self.input_size = parameters['input_size']
        self.output_size = parameters['output_size']
        self.curriculum = parameters['curriculum']
        self.aug_obs = parameters['aug_obs']
        self.parameters = parameters
        self.inputs_file = data_path + 'inputs.csv'
        self.targets_file = data_path + 'targets.csv'
        self.recurrent = self._seq_len > 1
        self.model = Network(self.input_size, self._hidden_memory_size, 
                             self.n_sources, self.output_size, self.recurrent)
        if self.output_size > 1:
            self.loss_function = nn.CrossEntropyLoss()
        else:
            self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=0.001)
        self.checkpoint_path = parameters['checkpoint_path'] + str(run_id)
        if parameters['load_model']:
            self._load_model()
        if self.curriculum:
            self.strength = 0.5
            self.strength_increment = 0.025
        else:
            self.strength = 1

    def train(self, n_epochs):
        inputs = self._read_data(self.inputs_file)
        targets = self._read_data(self.targets_file)
        input_seqs, target_seqs = self._form_sequences(inputs, targets)
        train_input_seqs, train_target_seqs, test_input_seqs, test_target_seqs = self._split_train_test(input_seqs, target_seqs)
        loss = self._train(train_input_seqs, train_target_seqs, test_input_seqs, test_target_seqs, n_epochs)
        self._save_model()
        if self.curriculum:
            self.strength += self.strength_increment
        os.remove(self.inputs_file)
        os.remove(self.targets_file)
        return loss

    def test(self):
        inputs = self._read_data(self.inputs_file)
        targets = self._read_data(self.targets_file)
        input_seqs, target_seqs = self._form_sequences(inputs, targets)
        loss = self._test(input_seqs, target_seqs)
        print(f'Test loss: {loss:10.8f}')
        os.remove(self.inputs_file)
        os.remove(self.targets_file)
        return loss

    
    def predict(self, obs):
        obs_tensor = torch.reshape(torch.FloatTensor(obs), (1,1,-1))
        _, probs = self.model(obs_tensor)
        probs = [prob[0] for prob in probs]
        return probs
    
    def reset(self):
        self.model.reset()
    
    def get_hidden_state(self):
        # return self.model.hidden_cell[0].detach().numpy()[0][0]
        return self.model.hidden_cell.detach().numpy()


    ### Private methods ###        

    def _read_data(self, data_file):
        data = []
        with open(data_file) as data_file:
            csv_reader = csv.reader(data_file, delimiter=',')
            for row in csv_reader:
                data.append([int(element) for element in row])
        return data

    def _form_sequences(self, inputs, targets):
        n_episodes = len(inputs)//self._episode_length
        input_seq = []
        target_seq = []
        for episode in range(n_episodes):
            for seq in range(self._episode_length - (self._seq_len - 1)):
                start = episode*self._episode_length+seq
                end = episode*self._episode_length+seq+self._seq_len
                input_seq.append(inputs[start:end])
                target_seq.append(targets[start:end])
        return input_seq, target_seq

    def _split_train_test(self, inputs, targets):
        test_size = int(0.1*len(inputs))
        train_inputs, train_targets = inputs[:-test_size], targets[:-test_size] 
        test_inputs, test_targets = inputs[-test_size:], targets[-test_size:]
        return train_inputs, train_targets, test_inputs, test_targets

    def _train(self, train_inputs, train_targets, test_inputs, test_targets, n_epochs):
        seqs = torch.FloatTensor(train_inputs)
        targets = torch.FloatTensor(train_targets)
        for e in range(n_epochs):
            permutation = torch.randperm(len(seqs))
            if e % 50 == 0:
                test_loss = self._test(test_inputs, test_targets)
                print(f'epoch: {e:3} test loss: {test_loss:10.8f}')
            for i in range(0, len(seqs) - len(seqs) % self._batch_size, self._batch_size):
                indices = permutation[i:i+self._batch_size]
                seqs_batch = seqs[indices]
                targets_batch = targets[indices]
                # self.model.hidden_cell = (torch.randn(1, self._batch_size, self._hidden_memory_size),
                #                           torch.randn(1, self._batch_size, self._hidden_memory_size))
                self.model.hidden_cell = torch.zeros(1, self._batch_size, self._hidden_memory_size)
                logits, probs = self.model(seqs_batch)
                end = 0
                self.optimizer.zero_grad()
                loss = 0
                for s in range(self.n_sources):
                    start = end 
                    end += self.output_size
                    target = targets_batch[:, :, start:end]
                    logit =  logits[s]
                    if self.output_size > 1:
                        logit = logit.view(-1, self.output_size)
                        target = torch.argmax(target, dim=2).view(-1)
                    else:
                        logit = logit.view(-1)
                        target = target.view(-1)
                    loss += self.loss_function(logit, target)
                loss.backward()
                self.optimizer.step()
        test_loss = self._test(test_inputs, test_targets)
        print(f'epoch: {e+1:3} test loss: {test_loss:10.8f}')
        self.model.reset()
        return test_loss

    def _test(self, inputs, targets):
        inputs = torch.FloatTensor(inputs)
        targets = torch.FloatTensor(targets)
        loss = 0
        # self.model.hidden_cell = (torch.randn(1, len(inputs), self._hidden_memory_size),
        #                           torch.randn(1, len(inputs), self._hidden_memory_size))
        self.model.hidden_cell = torch.zeros(1, len(inputs), self._hidden_memory_size)
        logits, probs = self.model(inputs)
        self.img1 = None
        end = 0
        targets_counts = []
        for s in range(self.n_sources):
            start = end
            end += self.output_size
            # loss += self.loss_function[s % 2](logits[s][:,-1,:], torch.argmax(targets[:, start:end], dim=1))
            target = targets[:, :, start:end]
            logit =  logits[s].view(-1, self.output_size)
            if self.output_size > 1:
                target = torch.argmax(target, dim=2).view(-1)
            else:
                target = target.view(-1, 1)
            loss += self.loss_function(logit, target)
            # from collections import Counter
            # targets_counts = Counter(torch.argmax(targets[:, start:end], dim=1).detach().numpy())
            # print(targets_counts)
            # probs_counts = np.sum(probs[s], axis=0)
            # print(probs_counts)
            # for i in range(len(inputs)):
                # self._plot_prediction(probs[s][i], targets[i, start:end])
        return loss.item()

    def _plot_prediction(self, prediction, target):
        prediction = prediction.detach().numpy()
        prediction = np.reshape(np.append(prediction, [prediction[5]]*19), (5,5))
        target = target.detach().numpy()
        target = np.reshape(np.append(target, [target[5]]*19), (5,5))
        if self.img1 is None:
            fig = plt.figure(figsize=(10,6))
            sub1 = fig.add_subplot(1, 2, 2)
            self.img1 = sub1.imshow(prediction, vmin=0, vmax=1)
            sub2 = fig.add_subplot(1, 2, 1)
            self.img2 = sub2.imshow(target, vmin=0, vmax=1)
            plt.tight_layout()
        else:
            self.img1.set_data(prediction)
            self.img2.set_data(target)
        plt.pause(0.5)
        plt.draw()

    def _save_model(self):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, 
                    os.path.join(self.checkpoint_path, 'checkpoint'))
    
    def _load_model(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_path, 'checkpoint'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        torch.set_grad_enabled(False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def read_parameters(config_file):
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters['parameters']


if __name__ == '__main__':
    sys.path.append("..") 
    from agents.random_agent import RandomAgent
    from simulators.warehouse.warehouse import Warehouse
    from influence_dummy import InfluenceDummy
    from data_collector import DataCollector
    agent = RandomAgent(2)
    parameters = {'n_sources': 4, 'output_size': 1, 'aug_obs': False}
    parameters = read_parameters('./configs/influence.yaml')
    influence = InfluenceNetwork(parameters, './data/traffic/', None)
    # data_collector = DataCollector(agent, 'warehouse', 8, influence, './data/warehouse/', 0)
    # data_collector.run(parameters['dataset_size'], log=True)
    influence.train(2000)
