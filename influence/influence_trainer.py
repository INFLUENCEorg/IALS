import torch
from aiagents.single.WarehouseNaiveAgent import WarehouseNaiveAgent
from aiagents.multi.BasicComplexAgent import BasicComplexAgent
from aienvs.runners.Episode import Episode
from aienvs.runners.Experiment import Experiment
from aienvs.Warehouse import Warehouse
import argparse
import numpy as np
import csv
from influence_model import InfluenceModel
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import os

class InfluenceTrainer(object):
    def __init__(self):
        parameters = self._parse_args()
        self._seq_len = parameters.seq_len
        self._episode_len = parameters.episode_len
        self._data_file = parameters.data_file
        self._lr = parameters.lr
        self._n_epochs = parameters.n_epochs
        self._hidden_layer_size = parameters.hidden_layer_size
        self._batch_size = parameters.batch_size
        self.n_sources = parameters.n_sources
        self.input_size = parameters.input_size
        self.output_size = parameters.output_size
        self.model = InfluenceModel(self.input_size, self._hidden_layer_size, self.n_sources, self.output_size)
        weights1 = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.05])
        weights2 = torch.FloatTensor([1.0, 0.04])
        self.loss_function = [nn.CrossEntropyLoss(weight=weights1),  nn.CrossEntropyLoss(weight=weights2)]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=0.001)
        self.checkpoint_path = parameters.checkpoint_path
        if parameters.load_model:
            self._load_model(self.model, self.optimizer, self.checkpoint_path)

    def learn(self):
        data = self._read_data(self._data_file)
        inputs, targets = self._form_sequences(np.array(data))
        train_inputs, train_targets, test_inputs, test_targets = self._split_train_test(inputs, targets)
        self._train(train_inputs, train_targets, test_inputs, test_targets)
        self._test(test_inputs, test_targets)
        self._save_model(self.model, self.optimizer, self.checkpoint_path)

    def _read_data(self, data_file):
        data = []
        with open(data_file) as data_file:
            csv_reader = csv.reader(data_file, delimiter=',')
            for row in csv_reader:
                data.append([int(element) for element in row])
        return data

    def _form_sequences(self, data):
        n_episodes = len(data)//self._episode_len
        inputs = []
        targets = []
        for episode in range(n_episodes):
            for seq in range(self._episode_len - self._seq_len - 1):
                start = episode*self._episode_len+seq
                end = episode*self._episode_len+seq+self._seq_len
                inputs.append(data[start:end, :41])
                targets.append(data[end-1, 41:])
        return inputs, targets

    def _split_train_test(self, inputs, targets):
        train_inputs, train_targets = inputs[:-self._episode_len], targets[:-self._episode_len] 
        test_inputs, test_targets = inputs[-self._episode_len:], targets[-self._episode_len:]
        return train_inputs, train_targets, test_inputs, test_targets

    def _train(self, train_inputs, train_targets, test_inputs, test_targets):
        seqs = torch.FloatTensor(train_inputs)
        targets = torch.FloatTensor(train_targets)
        for e in range(self._n_epochs):
            permutation = torch.randperm(len(seqs))
            loss = 0
            test_loss = self._test(test_inputs, test_targets)
            print(f'epoch: {e:3} test loss: {test_loss.item():10.8f}')
            for i in range(0, len(seqs) - len(seqs) % self._batch_size, self._batch_size):
                indices = permutation[i:i+self._batch_size]
                seqs_batch = seqs[indices]
                targets_batch = targets[indices]
                self.model.hidden_cell = (torch.zeros(1, self._batch_size, self._hidden_layer_size),
                                          torch.zeros(1, self._batch_size, self._hidden_layer_size))
                logits, _ = self.model(seqs_batch)
                end = 0
                for s in range(self.n_sources):
                    self.optimizer.zero_grad()
                    start = end
                    end += self.output_size[s]
                    single_loss = self.loss_function[s % 2](logits[s][:,-1,:], torch.argmax(targets_batch[:, start:end], dim=1))
                    single_loss.backward(retain_graph=True)
                    self.optimizer.step()
                loss += single_loss
        test_loss = self._test(test_inputs, test_targets)
        print(f'epoch: {e+1:3} test loss: {test_loss.item():10.8f}')

    def _test(self, inputs, targets):
        inputs = torch.FloatTensor(inputs)
        targets = torch.FloatTensor(targets)
        loss = 0
        self.model.hidden_cell = (torch.zeros(1, len(inputs), self._hidden_layer_size),
                                  torch.zeros(1, len(inputs), self._hidden_layer_size))
        logits, probs = self.model(inputs)
        self.img1 = None
        end = 0
        for s in range(self.n_sources):
            start = end
            end += self.output_size[s]
            loss += self.loss_function[s % 2](logits[s][:,-1,:], torch.argmax(targets[:, start:end], dim=1))
            # for i in range(len(inputs)):
                # self._plot_prediction(probs[k][i], targets[i, start:end])
        return loss

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

    def _save_model(self, model, optimizer, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    os.path.join(path, 'checkpoint'))
    
    def _load_model(self, model, optimizer, path):
        checkpoint = torch.load(os.path.join(path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seq_len', type=int, default=20,
                            help='length of the sequences used for prediction')
        parser.add_argument('--episode_len', type=int, default=100,
                            help='episode length')
        parser.add_argument('--data_file', type=str, default='../data/warehouse/data.csv',
                            help='path to the file where observations are stored')
        parser.add_argument('--lr', type=float, default=1.0e-3,
                            help='learning rate')
        parser.add_argument('--n_epochs', type=int, default=15,
                            help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=500,
                            help='batch size')
        parser.add_argument('--hidden_layer_size', type=int, default=32,
                            help='recurrent hidden layer size')
        parser.add_argument('--input_size', type=int, default=41,
                            help='input size')
        parser.add_argument('--output_size', type=int, default=[6,2,6,2,6,2,6,2],
                            help='size of each influence source')
        parser.add_argument('--n_sources', type=int, default=8,
                            help='number of influence sources')
        parser.add_argument('--checkpoint_path', type=str, default='../models/influence',
                            help='checkpoint path')
        parser.add_argument('--load_model', type=bool, default=False,
                            help='load model yes/no')
        args = parser.parse_args()
        return args

if __name__ == '__main__':
    trainer = InfluenceTrainer()
    trainer.learn()
