import numpy as np
import json
from transformers import DistilBertTokenizerFast
import createIntents
import torch
##################################################################
with open('intents.json', 'r') as f:
    intents = json.load(f)
##################################################################
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
        w = DistilBertTokenizerFast.tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, label))
##################################################################
##################################################################
# create training data
#print("xy", xy)
#print(xy[0])
X_train_chat = []
y_train_chat = []
for (pattern_sentence, label) in xy:
    # X: bag of words for each pattern_sentence
    bag = (pattern_sentence, all_words)
    X_train_chat.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    tag = chat_labels.index(label)
    y_train_chat.append(tag)
X_train_chat = torch.Tensor(X_train_chat)
y_train_chat = torch.Tensor(y_train_chat)
print("Training data created for intent classifier")
##################################################################
#print(X_train_chat.shape, type(X_train_chat))

