import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class InfluenceModel(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_layer_size, n_sources, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.ModuleList()
        self.n_sources = n_sources
        self.softmax = nn.Softmax(dim=1)
        self.hidden_layer_size = hidden_layer_size
        for s in range(self.n_sources):
            self.linear.append(nn.Linear(hidden_layer_size, output_size[s]))
        self.reset()

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        logits = []
        probs = []
        for k in range(self.n_sources):
            logits.append(self.linear[k](lstm_out))
            probs.append(self.softmax(logits[-1][:, -1, :])[0].detach().numpy())
        return logits, probs
    
    def reset(self):
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))