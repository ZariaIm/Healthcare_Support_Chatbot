#LINEAR 
import torch.nn as nn

class LSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM_CNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        # no activation and no softmax at the end
        return out

class LSTM_CNN_Dropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM_CNN_Dropout, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        # no activation and no softmax at the end
        return out