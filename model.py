#LINEAR 
import torch
import torch.nn as nn
from torch.nn.modules import dropout
embedding_vector_length = 256
# cnn_model.add(Embedding(input_dim=top_words, output_dim=embedding_vector_length, input_length=max_review_length))
# cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# cnn_model.add(MaxPooling1D(pool_size=2))
# cnn_model.add(LSTM(100))
# cnn_model.add(Dense(units=1, activation='sigmoid'))

class LSTM_CNN(nn.Module):
    def __init__(self, vocab_size, input_size, num_classes):
        super(LSTM_CNN, self).__init__()
        self.embed = nn.Embedding((num_classes), (embedding_vector_length),(input_size))
        self.conv1 = nn.Conv1d(input_size, 32, 9) 
        self.maxpool = nn.MaxPool1d(2) 
        self.LSTM = nn.LSTM(124, 124, dropout = 0.2)
        self.fc1 = nn.Linear(124,num_classes)
        self.relu = nn.ReLU()
    def forward(self, x, prev_state):
        print("1", x.shape, type(x))
        out = self.embed((x.long()))
        print("2", out.shape, type(out))
        out = self.relu(self.conv1(out))
        print("3", out.shape, type(out))
        out = self.maxpool(out)
        print("4", out.shape, type(out))
        out, state = self.LSTM(out, prev_state)
        print("5", out, type(out))
        out = out.view(out.shape[0], -1)
        print("6", out.shape, type(out))
        out = self.fc1(out)
        print("7", out.shape, type(out))
        # no activation and no softmax at the end
        return out, state

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