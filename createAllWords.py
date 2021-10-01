import numpy as np
import random
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
chat_labels = []

xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    label = intent['labels']
    # add to tag list
    chat_labels.append(label)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        #print(w)
        #print(label)
        xy.append((w, label))
#print(labels)
# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))

#print(labels)

# create training data
X_train = []
y_train = []

for (pattern_sentence, label) in xy:
    # X: bag of words for each pattern_sentence
    #print(type(pattern_sentence[0]))
    #print(all_words)
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    tag = chat_labels.index(label)
    y_train.append(tag)

X_train = np.array(X_train)
y_train = np.array(y_train)
#print(type(X_train))
#print(X_train.shape)


#Create all words array, diseases array
all_symptoms = [] #list type

#read dataset and cast values as strings
df = pd.read_csv("datasets/dataset.csv", dtype = "string") #4920 rows  x 18 cols
df = df.fillna(" ")

#Loop through each list of symptoms for the label -  treat the row almost like a sentence
#create training data from random rows using random.choice (set the seed)
#create df_test by dropping rows from original - make results reproducible
np.random.seed(42)
# sample without replacement
test_ix = np.random.choice(df.index, 354, replace=False)
df_test = df.iloc[test_ix] # 354 rows x 18 cols
df_train = df.drop(test_ix) # 4566 rows x 18 cols

for i in range(1,18):
    for word in df[f"Symptom_{i}"]:
        all_symptoms.extend(tokenize(' '.join(word.split("_"))))
all_symptoms = [stem(w) for w in all_symptoms]
#remove duplicates from list and sort
#sets are easier for comparing too
all_symptoms = (set(all_symptoms))
#print(all_words)
print("Collected all symptoms")

disease_labels = []
disease_symptoms = torch.FloatTensor([])
temp_symptoms = torch.FloatTensor([])

for value in df["Disease"].str.split(", "):
    disease_labels.extend(value)
disease_labels = (set(disease_labels))

###############################################################
#Need to fix this

# for value in disease_labels:
#     #collect symptoms for specific diseases
#     print(value)
#     temp_name = f"{value}_symptoms"
#     for word in df.loc[f"{value}"]:
#         print(word)
#         temp_symptoms.extend(word)
#         temp_symptoms = set(temp_symptoms)
#     print(temp_symptoms)
#     temp_name = temp_symptoms
# #list comprehension to remove letters
# #temp = [] # temporary array
# #[temp.append(x) for x in disease_labels if len(x)>1]
# #disease_labels = (temp)
# print(disease_symptoms)
# print(disease_labels)
print("Collected all diseases")



