import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
import pandas as pd

#column 1: labels/diseases/Y-train
#column 2: input/symptoms/X-train
#print(df[['Symptom_2','Symptom_3']]) # indexes both columns by name
#print(type(df.Symptom_2[3]))#4th row of symptom 2 - retrieves the class 'str'


#read dataset and cast values as strings
df = pd.read_csv("datasets/dataset.csv", dtype = "string") #4920 rows  x 18 cols
df = df.fillna(" ")

#Create all words array, diseases array
all_words = [] #list type
diseases = []
ignore_words = ['?', '.', '!']


for i in range(1,18):
    c_name = f"Symptom_{i}"
    for word in df[c_name]:
        all_words.extend(tokenize(' '.join(word.split("_"))))
all_words = [stem(w) for w in all_words if w not in ignore_words]
#remove duplicates from list and sort
#sets are easier for comparing too
all_words = sorted(set(all_words))
#print(all_words)
print("Collected all symptoms")


for value in df["Disease"].str.split(","):
    diseases.extend(value)
# all_words.pop(0)#remove " "
diseases = sorted(set(diseases))
#list comprehension to remove letters
temp = [] # temporary array
[temp.append(x) for x in diseases if len(x)>1]
diseases = (temp)
print(len(diseases))
print("Collected all diseases")

#Loop through each list of symptoms for the label -  treat the row almost like a sentence
#create training data from random rows using random.choice (set the seed)
#create df_test by dropping rows from original - make results reproducible
np.random.seed(42)
# sample without replacement
train_ix = np.random.choice(df.index, 354, replace=False)
df_training = df.iloc[train_ix] # 354 rows x 18 cols
df_test = df.drop(train_ix) # 4566 rows x 18 cols


class SymptomDataset(Dataset):

    def __init__(self):
        self.x_data = np.array(df_training.iloc[:,1:18])
        #get dummies creates one hot encoding from pandas data
        self.y_data = np.array(pd.get_dummies(df_training["Disease"]))
        #print(self.y_data)

        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, idx):
        list_of_symptoms = []
        for i in range(17):
            list_of_symptoms.extend(tokenize(' '.join(self.x_data[idx,i].split("_"))))
        x_bagged = bag_of_words(list_of_symptoms, all_words)
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
num_epochs = 20
batch_size = 100
learning_rate = 0.0001
input_size = len(all_words)
hidden_size = 8
output_size = len(diseases)

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
    for (words,labels) in train_loader:
        #words is a list
        #labels is a tuple
        #print(words.shape) # should be [8,137]
        #print(labels)
        # Forward pass
        outputs = model(words)
        labels = labels.squeeze(0)
        # for one hot encoding
        labels = torch.max(labels, 1)[1]
        #print(outputs.shape) # should be [8,41]
        #print(labels.shape, labels) # should [41]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #if (epoch+1) % 100 == 0:
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"labels": labels
}

FILE = "model_symptoms.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')