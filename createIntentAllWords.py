import numpy as np
import json
from nltk_utils import bag_of_words, tokenize, stem
import createIntents
import torch

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
X_train_chat = []
y_train_chat = []

for (pattern_sentence, label) in xy:
    # X: bag of words for each pattern_sentence
    #print(type(pattern_sentence[0]))
    #print(all_words)
    bag = bag_of_words(pattern_sentence, all_words)
    X_train_chat.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    tag = chat_labels.index(label)
    y_train_chat.append(tag)

X_train_chat = np.array(X_train_chat)
y_train_chat = np.array(y_train_chat)
#print(type(X_train))
#print(X_train.shape)

print("Created All Words for Intents")


data = {
"all_words": all_words,
"labels": chat_labels,
}
torch.save(data, "chat.pth")
print("all words and labels saved to chat.pth")