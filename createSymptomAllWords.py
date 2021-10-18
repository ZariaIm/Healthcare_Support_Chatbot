import numpy as np
import pandas as pd
import torch
from nltk_utils import bag_of_words, tokenize, stem
#####################################################################
class ChatDataset():
    def __init__(self, X_train, y_train):
        self.x_data = X_train
        self.y_data = y_train
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.X_train.items()}
        item['labels'] = torch.tensor(self.y_data[index])
        return item 
    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.y_data)

#Create all words array, diseases array
all_symptoms = [] #list type
disease_labels = []
xy = []
#####################################################################
#read dataset and cast values as strings
df = pd.read_csv("datasets/dataset.csv", dtype = "string") #4920 rows  x 18 cols
df = df.fillna(" ")
df = df.iloc[4879:4922] # the dataset kinda repeats at this point
#####################################################################
#Loop through each list of symptoms for the label -  treat the row almost like a sentence
#####################################################################
#Collect disease labels
for row in range(len(df["Disease"])):
    value =  df.iloc[row,0]
    disease_labels.append(value)
    temp_symptoms = []
    for i in range(1,18):
        word = df.iloc[row, i]
        all_symptoms.extend(tokenize(' '.join(word.split("_"))))
        temp_symptoms.extend(tokenize(' '.join(word.split("_"))))
    xy.append((temp_symptoms, value))    
ignore_words = ['in', ', ', 'like', 'feel', 'from', 'and', 'of', 'on', 'the']
all_symptoms = [stem(w) for w in all_symptoms if w not in ignore_words]
#print("all symptoms", all_symptoms)
#remove duplicates from list and sort
#sets are easier for comparing too
all_symptoms = sorted(set(all_symptoms))
disease_labels = sorted(set(disease_labels))
#print("Collected all diseases and symptom ALL Words")
##################################################################
disease_symptoms = []
for disease in disease_labels:
    symptoms = []
    for col in range(1,17):
        symptom_num = f'Symptom_{col}'
        for value in df[symptom_num][df["Disease"] == f"{disease}"].drop_duplicates():
            if len(value)>1:
                symptoms.append(value)
    symptoms = sorted(set(symptoms))
    disease_symptoms.append(symptoms)
#print("Collected all symptoms related to each disease")
###################################################################
required_symptoms = []
emergency_symptoms = []
#Trying to do the emergency thing
for i in range(len(disease_labels)):
    if ("Hypertension" in disease_labels[i]):
        required_symptoms.extend(disease_symptoms[i])
    if ("Heart attack" in disease_labels[i]):
        required_symptoms.extend(disease_symptoms[i])
for word in required_symptoms:
    emergency_symptoms.extend(tokenize(' '.join(word.split("_"))))
ignore_words = ['in', ', ', 'like', 'feel', 'from', 'and', 'of', 'on', 'the', 'lack','loss','sweat']
emergency_symptoms = [stem(w) for w in emergency_symptoms if w not in ignore_words]
#print("Hypertension and Heart attack symptoms classed as emergency")
##################################################################
data = {
"all_words": all_symptoms,
"labels": disease_labels,
"disease_symptoms":disease_symptoms,
"emergency_symptoms":emergency_symptoms
}
torch.save(data, "disease.pth")
#print("symptoms and diseases saved to disease.pth")
##################################################################

# create training data
X_train_symptom = []
y_train_symptom = []
for (pattern_sentence, label) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_symptoms, 400)
    X_train_symptom.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    tag = disease_labels.index(label)
    y_train_symptom.append(tag)

X_train_symptom = torch.Tensor(X_train_symptom)
y_train_symptom = torch.Tensor(y_train_symptom)
print("Training data created for symptom classifier")