import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
import pandas as pd
from createSymptomAllWords import all_symptoms, disease_labels, X_train, y_train

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
print("Created Dataset")
# Hyper-parameters 
num_epochs = 300
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
print("Model initialised. Entering Training Loop.")
for epoch in range(num_epochs):
    for (x, y) in train_loader:
        x = x.to(device)
        y = y.to(dtype=torch.long).to(device)
        #print("words", word)
        # Forward pass
        y_hat = model(x)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(y_hat, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch%50 == 0):    
        print (f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_symptoms,
"labels": disease_labels
#"disease_symptoms": disease_symptoms
}

FILE = "model_symptoms.pth"
torch.save(data, FILE)

print(f'symptom classifier training complete. file saved to {FILE}')