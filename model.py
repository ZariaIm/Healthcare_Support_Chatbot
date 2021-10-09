#LINEAR 
import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

class LinearNetDropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
    #dropout after non linear activation function
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

#using 1d convolutions
#nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
class ConvolutionalNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ConvolutionalNet, self).__init__()
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size = 5) 
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size = 5) 
        self.pool = nn.MaxPool1d(5)
        self.globalPool = nn.MaxPool1d(128)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        #out = self.embedding(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        #out = out.view(out.shape[0],-1)
        out = (self.fc1(out))
        out = self.relu(out)
        out = self.fc2(out)
        # no activation and no softmax at the end
        return out

# class ShallowCNN(nn.Module):
#   def __init__(self):
#     #call __init__ of parent nn.Module
#     super(ShallowCNN, self).__init__()

#     #Define Convolutional layers - Channels in = 3
#     self.conv1 = nn.Conv2d(3,96,7,2,0)
#     self.conv2 = nn.Conv2d(96,64,5,2,0)
#     self.conv3 = nn.Conv2d(64,128,3,2,0)
#     #Define fully connected layers (Classifier)
#     self.fc1 = nn.Linear(1152,128)
#     self.fc2 = nn.Linear(128,10)
#     #Define Max Pooling layer
#     self.max_pool = nn.MaxPool2d(3)

#   def forward(self,x):
    
#     #Pass to convolutional layers (F.relu activation func)
#     #conv1 -> relu -> conv2 ->relu -> conv3 -> relu 
#     out1 = F.relu(self.conv1(x))
#     out2 = F.relu(self.conv2(out1))
#     out3 = F.relu(self.conv3(out2))

#     #pass to maxpool
#     #-> maxpool 
#     out4 = self.max_pool(out3)
    
#     #pass to linear layers after reshaping
#     #-> fc1 -> relu -> fc2
#     out4_reshape = out4.view(out4.shape[0],-1)
#     out5 = F.relu(self.fc1(out4_reshape))
#     out6 = self.fc2(out5)

#     return out6

# input1 = Input(shape=(33,))
# x = Embedding(input_dim=vocab_size, output_dim=32, input_length=33)(input1)
# x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
# x = MaxPooling1D(pool_size=2)(x)
# x = Dropout(0.5)(x)
# x = Flatten()(x)
# x = Dense(30, activation='sigmoid')(x)
# x = Dropout(0.5)(x)
# x = Dense(5, activation='sigmoid')(x)
# x = Dropout(0.5)(x)
# output1 = Dense(1, activation='sigmoid')(x)