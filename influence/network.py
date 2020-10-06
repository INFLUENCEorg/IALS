import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class InfluenceModel(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_layer_size, n_sources, output_size):
        super().__init__()
        # self.fc = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.n_sources = n_sources
        self.softmax = nn.Softmax(dim=1)
        self.hidden_layer_size = hidden_layer_size
        for s in range(self.n_sources):
            self.linear1.append((nn.Linear(hidden_layer_size, hidden_layer_size)))
            self.linear2.append(nn.Linear(hidden_layer_size, output_size[s]))
        self.reset()

    def forward(self, input_seq):
        # linear_out = self.relu(self.fc(input_seq))
        # print(self.hidden_cell)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        # print(self.hidden_cell)
        logits = []
        probs = []
        for k in range(self.n_sources):
            linear1_out = self.relu(self.linear1[k](lstm_out[:,-1,:]))
            linear2_out = self.linear2[k](linear1_out)
            logits.append(linear2_out)
            probs.append(self.softmax(logits[-1]).detach().numpy())
        return logits, probs
    
    def reset(self):
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))