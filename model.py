#LINEAR 
import torch.nn as nn
embedding_vector_length = 32
# cnn_model.add(Embedding(input_dim=top_words, output_dim=embedding_vector_length, input_length=max_review_length))
# cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# cnn_model.add(MaxPooling1D(pool_size=2))
# cnn_model.add(LSTM(100))
# cnn_model.add(Dense(units=1, activation='sigmoid'))

class LSTM_CNN(nn.Module):
    def __init__(self, vocab_size, input_size, num_classes):
        super(LSTM_CNN, self).__init__()
        self.embed = nn.Embedding(int(vocab_size), int(embedding_vector_length),int(input_size))
        self.conv1 = nn.Conv1d(embedding_vector_length, 32, 3) 
        self.maxpool = nn.MaxPool1d(2) 
        self.LSTM = nn.LSTM(1000, 100)
        self.fc1 = nn.Linear(180,1000)
        self.fc2 = nn.Linear(1000,num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.embed(x)
        out = self.conv1(out)
        out = self.maxpool(out)
        out = self.LSTM(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
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