import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
import pandas as pd
from createAllWords import all_symptoms, disease_labels, disease_symptoms, df_test, df_train

class SymptomDataset(Dataset):

    def __init__(self):
        self.x_data = np.array(df_train.iloc[:,1:18])
        #get dummies creates one hot encoding from pandas data
        self.y_data = np.array(pd.get_dummies(df_train["Disease"]))
        #print(self.y_data)

        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, idx):
        list_of_symptoms = []
        for i in range(17):
            list_of_symptoms.extend(tokenize(' '.join(self.x_data[idx,i].split("_"))))
        x_bagged = bag_of_words(list_of_symptoms, all_symptoms)
        #print(x_bagged)
        y_bagged = self.y_data[idx]
        #print(y_bagged.shape)
        return x_bagged, y_bagged

    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.x_data)

dataset = SymptomDataset()
print("Created Dataset")
# Hyper-parameters 
num_epochs = 40
batch_size = 100
learning_rate = 0.001
input_size = len(all_symptoms)
hidden_size = 8
output_size = len(disease_labels)

print("Preparing to set up the neural network")
print(f" --- input size: {input_size}; output_size: {output_size} --- ")

train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print("Model Initialised. Entering Training Loop.")
#Train the model
for epoch in range(num_epochs):
    for (x,y) in train_loader:
        # Forward pass
        y_hat = model(x) #[batch_size,41]
        y = y.squeeze(0)
        # for one hot encoding
        y = torch.max(y, 1)[1] #[41]
        loss = criterion(y_hat, y)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_symptoms,
"labels": disease_labels,
"disease_symptoms": disease_symptoms
}

FILE = "model_symptoms.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')