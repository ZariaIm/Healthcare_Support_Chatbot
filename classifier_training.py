import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from model import LinearNet
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#####################################################################


class ChatDataset():

    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
#####################################################################


def train(net, device, loader, optimizer, loss_fun):
    loss = 0
    # Set Network in train mode
    net.train()
    # Perform a single epoch of training on the input dataloader, logging the loss at every step
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(dtype=torch.long).to(device)

        y_hat = net(x)

        loss = loss_fun(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # return the logger array
    return loss


def evaluate(net, device, loader):
    # initialise counter
    epoch_acc = 0
    # Set network in evaluation mode
    net.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(dtype=torch.long).to(device)
            y_hat = net(x)
            # log the cumulative sum of the acc
            epoch_acc += (y_hat.argmax(1) == y.to(device)).sum().item()
    # return the accuracy from the epoch
    return epoch_acc / len(loader.dataset)


#####################################################################
# read dataset and cast values as strings
np.random.seed(42)

# 4920 rows  x 18 cols
df = pd.read_csv("datasets/dataset.csv", dtype="string")
df = df.fillna(" ")
df_train = df.iloc[4879:4922]  # the dataset kinda repeats at this point
ix = np.random.rand(4878) > 0.8
df_test = df.iloc[0:4878][ix]
#####################################################################
##################################################################
# create training data and validation data
# #################TO DO##################

##################################################################
#####################################################################
# Create all words array, diseases array
all_symptoms = []  # list type
disease_labels = []
train_xy = []
# Collect disease labels
for row in range(len(df_train["Disease"])):
    value = df_train.iloc[row, 0]
    disease_labels.append(value)
    temp_symptoms = []
    for i in range(1, 18):
        word = df_train.iloc[row, i]
        all_symptoms.extend(tokenize(' '.join(word.split("_"))))
        temp_symptoms.extend(tokenize(' '.join(word.split("_"))))
    train_xy.append((temp_symptoms, value))
ignore_words = ['in', ', ', 'like', 'feel', 'from', 'and', 'of', 'on', 'the']
all_symptoms = [stem(w) for w in all_symptoms if w not in ignore_words]
# remove duplicates from list and sort
# sets are easier for comparing too
all_symptoms = sorted(set(all_symptoms))
disease_labels = sorted(set(disease_labels))
print("Collected all diseases and symptom ALL Words")
#####################################################################
#### testing set
test_xy = []
# Collect disease labels
for row in range(len(df_test["Disease"])):
    value = df_test.iloc[row, 0]
    temp_symptoms = []
    for i in range(1, 18):
        word = df_test.iloc[row, i]
        temp_symptoms.extend(tokenize(' '.join(word.split("_"))))
    test_xy.append((temp_symptoms, value))

##################################################################
disease_symptoms = []
for disease in disease_labels:
    symptoms = []
    for col in range(1, 17):
        symptom_num = f'Symptom_{col}'
        for value in df_train[symptom_num][df_train["Disease"] == f"{disease}"].drop_duplicates():
            if len(value) > 1:
                symptoms.append(value)
    symptoms = sorted(set(symptoms))
    disease_symptoms.append(symptoms)
print("Collected all symptoms related to each disease")
###################################################################
required_symptoms = []
emergency_symptoms = []
# Trying to do the emergency thing
for i in range(len(disease_labels)):
    if ("Hypertension" in disease_labels[i]):
        required_symptoms.extend(disease_symptoms[i])
    if ("Heart attack" in disease_labels[i]):
        required_symptoms.extend(disease_symptoms[i])
for word in required_symptoms:
    emergency_symptoms.extend(tokenize(' '.join(word.split("_"))))
ignore_words = ['in', ', ', 'like', 'feel', 'from',
                'and', 'of', 'on', 'the', 'lack', 'loss', 'sweat']
emergency_symptoms = [stem(w)
                      for w in emergency_symptoms if w not in ignore_words]
print("Hypertension and Heart attack symptoms classed as emergency")
##################################################################
data = {
    "all_symptoms": all_symptoms,
    "disease_labels": disease_labels,
    "disease_symptoms": disease_symptoms,
    "emergency_symptoms": emergency_symptoms
}
torch.save(data, "disease.pth")
print("symptoms and diseases saved to disease.pth")
##################################################################
# create training data
X_train = []
y_train = []
for (pattern_sentence, label) in train_xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_symptoms)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    tag = disease_labels.index(label)
    y_train.append(tag)

X_train = np.array(X_train)
y_train = np.array(y_train)
print("Training data created for symptom classifier")
##################################################################
# create training data
X_test = []
y_test = []
for (pattern_sentence, label) in test_xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_symptoms)
    X_test.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    tag = disease_labels.index(label)
    y_test.append(tag)

X_test = np.array(X_test)
y_test = np.array(y_test)
print("Training data created for symptom classifier")
##################################################################
num_epochs = 20
batch_size = 41  # higher than 41 won't do anything because there are only 41 samples
learning_rate = 0.1
input_size = len(X_train[0])
hidden_size = 8
output_size = len(y_train)
##################################################################
print(f" --- input size: {input_size}; output_size: {output_size} --- ")

train_dataset = ChatDataset(X_train, y_train)

train_loader = DataLoader(dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0)

test_dataset = ChatDataset(X_test, y_test)

test_loader = DataLoader(dataset=test_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0)


model = LinearNet(input_size, hidden_size, output_size).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
##################################################################

training_loss_logger = []
training_acc_logger = []
testing_acc_logger = []
for epoch in range(num_epochs):
    training_loss = train(model, device, train_loader, optimizer, criterion)
    train_acc = evaluate(model, device, train_loader)
    test_acc = evaluate(model, device, test_loader)
    training_acc_logger.append(train_acc)
    testing_acc_logger.append(test_acc)
    training_loss_logger.append(training_loss.item())
    if (epoch % 5 == 0):
        print(
            f'| Epoch: {epoch:02} |  Train Loss: {training_loss.item():.4f} | Train Acc: {train_acc*100:05.2f}% | Test Acc: {test_acc*100:05.2f}% |')

chat_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
}

torch.save(chat_data, "model_symptoms.pth")
