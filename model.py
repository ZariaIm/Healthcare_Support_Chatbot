#LINEAR 
import torch
import torch.nn as nn
from torch.nn.modules import dropout

class LSTM_CNN(nn.Module):
    def __init__(self, vocab_size, input_size, num_classes, embedding_vector_length, lstm_size, num_layers, filter_num):
        super(LSTM_CNN, self).__init__()
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.embed = nn.Embedding((vocab_size), (embedding_vector_length),(input_size))
        self.conv1 = nn.Conv1d(input_size, filter_num, 9) 
        self.maxpool = nn.MaxPool1d(2) 
        self.LSTM = nn.LSTM(lstm_size, lstm_size, num_layers = num_layers, dropout = 0.2)
        self.fc1 = nn.Linear(lstm_size*filter_num,num_classes)
        self.relu = nn.ReLU()
    def forward(self, x, prev_state):
        out = self.embed((x.long()))
        out = self.relu(self.conv1(out))
        out = self.maxpool(out)
        out, state = self.LSTM(out, prev_state)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        # no activation and no softmax at the end
        return out, state
    def init_state(self, sequence_length):
        return torch.zeros(self.num_layers, sequence_length, self.lstm_size), torch.zeros(self.num_layers, sequence_length, self.lstm_size)
