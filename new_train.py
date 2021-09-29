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
#stem_all_words = []
for i in range(1,18):
    c_name = f"Symptom_{i}"
    #print(c_name)
    for value in df[c_name].str.split("_"):
        #print(value)
        all_words.extend(value)
print("Collected all symptoms")
for value in df["Disease"].str.split(","):
    diseases.extend(value)
#remove duplicates from list and sort
#sets are easier for comparing too
all_words = sorted(set(all_words))
all_words.pop(0)#remove " "
diseases = sorted(set(diseases))
#list comprehension to remove letters
temp = [] # temporary array
[temp.append(x) for x in diseases if len(x)>1]
diseases = temp
print("Collected all diseases")


#Loop through each list of symptoms for the label -  treat the row almost like a sentence
#create training data from random rows using random.choice (set the seed)
#create df_test by dropping rows from original - make results reproducible
np.random.seed(42)
# sample without replacement
train_ix = np.random.choice(df.index, 354, replace=False)
df_training = df.iloc[train_ix] # 354 rows x 18 cols
df_test = df.drop(train_ix) # 4566 rows x 18 cols

# x_set = df_training.iloc[:,1:18]
# index = x_set.index
# #print(x_set.loc[4916,"Symptom_2"]) # prints symptom
# list_of_symptoms = []
# for i in range(1,18):
#     c_name = f"Symptom_{i}"
#     list_of_symptoms.append(x_set.loc[index[1],c_name])
# x_return = bag_of_words(list_of_symptoms, all_words)
# #print(list_of_symptoms)     

class SymptomDataset(Dataset):

    def __init__(self):
        self.n_samples = df_training.shape[0]
        self.x_data = df_training.iloc[:,1:18]
        self.y_data = df_training["Disease"]
        self.index = self.x_data.index
        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, i):
        list_of_symptoms = []
        for i in range(1,18):
            c_name = f"Symptom_{i}"
            list_of_symptoms.append(self.x_data.loc[self.index[i],c_name].split("_"))
        return (list_of_symptoms), (self.y_data.iloc[i])

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = SymptomDataset()
print("Created Dataset")
# Hyper-parameters 
num_epochs = 1000
batch_size = 1
learning_rate = 0.001
input_size = (df_training.shape[0])
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
        print(words)
        print(labels)
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

# data = {
# "model_state": model.state_dict(),
# "input_size": input_size,
# "hidden_size": hidden_size,
# "output_size": output_size,
# "all_words": all_words,
# "tags": tags
# }

# FILE = "model.pth"
# torch.save(data, FILE)

# print(f'training complete. file saved to {FILE}')

