import numpy as np
import pandas as pd
import torch
from nltk_utils import bag_of_words, tokenize, stem

#Create all words array, diseases array
all_symptoms = [] #list type
disease_labels = []
xy = []
#read dataset and cast values as strings
df = pd.read_csv("datasets/dataset.csv", dtype = "string") #4920 rows  x 18 cols
df = df.fillna(" ")
df = df.iloc[0:2460] # the dataset kinda changes at this point

#Loop through each list of symptoms for the label -  treat the row almost like a sentence
#create training data from random rows using random.choice (set the seed)
#create df_test by dropping rows from original - make results reproducible
np.random.seed(42)
# sample without replacement
test_ix = np.random.choice(df.index, 354, replace=False)
df_test = df.iloc[test_ix] # 354 rows x 18 cols
df_train = df.drop(test_ix) # remianiig rows x 18 cols
test_ix = np.random.choice(df_train.index, 354, replace=False)
df_val = df.iloc[test_ix] # 354 rows x 18 cols
df_train = df.drop(test_ix) # remaining rows x 18 cols

disease_symptoms = []

#Collect disease labels
for row in range(len(df_train["Disease"])):
    value =  df_train.iloc[row,0]
    disease_labels.append(value)
    temp_symptoms = []
    for i in range(1,18):
        word = df_train.iloc[row, i]
        all_symptoms.extend(tokenize(' '.join(word.split("_"))))
        temp_symptoms.extend(tokenize(' '.join(word.split("_"))))
    xy.append((temp_symptoms, value))    

all_symptoms = [stem(w) for w in all_symptoms]
#remove duplicates from list and sort
#sets are easier for comparing too
all_symptoms = sorted(set(all_symptoms))
disease_labels = list(set(disease_labels))

print("Collected all diseases and symptom All Words")

# create training data
X_train_symptom = []
y_train_symptom = []

for (pattern_sentence, label) in xy:
    # X: bag of words for each pattern_sentence
    #print(type(pattern_sentence[0]))
    #print(all_words)
    bag = bag_of_words(pattern_sentence, all_symptoms)
    X_train_symptom.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    tag = disease_labels.index(label)
    y_train_symptom.append(tag)

X_train_symptom = np.array(X_train_symptom)
y_train_symptom = np.array(y_train_symptom)


data = {
"all_words": all_symptoms,
"labels": disease_labels,
}
torch.save(data, "disease.pth")
print("symptoms and diseases saved to disease.pth")



X_val = []
y_val = []
for row in range(len(df_val["Disease"])):
    value =  df_val.iloc[row,0]
    temp_symptoms = []
    for i in range(1,18):
        word = df_val.iloc[row, i]
        temp_symptoms.extend(tokenize(' '.join(word.split("_"))))
    xy.append((temp_symptoms, value))    

for (pattern_sentence, label) in xy:
    # X: bag of words for each pattern_sentence
    #print(type(pattern_sentence[0]))
    #print(all_words)
    bag = bag_of_words(pattern_sentence, all_symptoms)
    X_val.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    tag = disease_labels.index(label)
    y_val.append(tag)
X_val = np.array(X_val)
y_val = np.array(y_val)


X_test = []
y_test =[]
for row in range(len(df_test["Disease"])):
    value =  df_test.iloc[row,0]
    temp_symptoms = []
    for i in range(1,18):
        word = df_test.iloc[row, i]
        temp_symptoms.extend(tokenize(' '.join(word.split("_"))))
    xy.append((temp_symptoms, value))    

for (pattern_sentence, label) in xy:
    # X: bag of words for each pattern_sentence
    #print(type(pattern_sentence[0]))
    #print(all_words)
    bag = bag_of_words(pattern_sentence, all_symptoms)
    X_test.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    tag = disease_labels.index(label)
    y_test.append(tag)
X_test = np.array(X_test)
y_test = np.array(y_test)
