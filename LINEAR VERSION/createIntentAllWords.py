import numpy as np
import json
from nltk_utils import bag_of_words, tokenize, stem
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
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, label))
#print(labels)
# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
print("Created All Words for Intents")

#split into train and testing data
train_xy = []
test_xy = []
np.random.seed(42)
ix = np.random.rand(len(xy)) <= 0.98
for idx in range(len(xy)):
    if ix[idx]:
        train_xy.append(xy[idx])
    else:
        test_xy.append(xy[idx])

##################################################################
data = {"all_words": all_words, "labels": chat_labels}
torch.save(data, "chat.pth")
print("all words and labels saved to chat.pth")
##################################################################
# create training data
X_train_chat = []
y_train_chat = []
for (pattern_sentence, label) in train_xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train_chat.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    tag = chat_labels.index(label)
    y_train_chat.append(tag)
X_train_chat = np.array(X_train_chat)
y_train_chat = np.array(y_train_chat)
print("Training data created for intent classifier")
##################################################################
# create test data
X_test_chat = []
y_test_chat = []
for (pattern_sentence, label) in test_xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_test_chat.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    tag = chat_labels.index(label)
    y_test_chat.append(tag)
X_test_chat = np.array(X_test_chat)
y_test_chat = np.array(y_test_chat)
print("Testing data created for intent classifier")

