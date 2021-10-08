import os
import torch
import torch.nn as nn
import csv

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
        self._episode_length = parameters['episode_length']
        self._seq_len = parameters['seq_len']
        self.loss_function = nn.CrossEntropyLoss()
        self.truncated = self._seq_len < self._episode_length
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
    

    def test(self, inputs_file, targets_file):
        inputs = self._read_data(inputs_file)
        targets = self._read_data(targets_file)
        input_seqs, target_seqs = self._form_sequences(inputs, targets)
        loss = self._test(input_seqs, target_seqs)
        print(f'Test loss: {loss:10.8f}')
        os.remove(inputs_file)
        os.remove(targets_file)
        return loss

    def _test(self, inputs, targets):
        inputs = torch.FloatTensor(inputs)
        targets = torch.FloatTensor(targets)
        logits = torch.log(torch.FloatTensor(self.probs*inputs.shape[0]))
        if targets.shape[-1] == self.n_sources*self.output_size:
            targets = torch.argmax(targets.view(-1, self.n_sources, self.output_size), dim=2).long().flatten()
        else:
            targets = targets.long().flatten()
        # logits = logits.flatten(end_dim=1)
        loss = self.loss_function(logits, targets)
        return loss.item()
    
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
                if self.truncated:
                    target_seq.append(targets[end-1])
                else:
                    target_seq.append(targets[start:end])
        return input_seq, target_seq
