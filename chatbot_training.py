from model import FineTunedModel
import json
from transformers import DistilBertTokenizerFast
from time import process_time
from torch.utils.data import Dataset, DataLoader
import createIntents
import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "distilbert-base-uncased"
import matplotlib.pyplot as plt
##################################################################

with open('intents.json', 'r') as f:
    intents = json.load(f)

##################################################################

chat_text = []
chat_labels_str = []
chat_labels = []


for idx, intent in enumerate(intents['intents']):
    label = intent['labels']
    chat_labels_str.append(label)
    i = 0
    for pattern in intent['patterns']:
        chat_text.append(pattern)
        chat_labels.append(idx)

#split into train and testing data
train_chat_text = []
test_chat_text = []
train_chat_labels = []
test_chat_labels = []

np.random.seed(42)
ix = np.random.rand(len(chat_text)) <= 0.98

for idx in range(len(chat_text)):
    if ix[idx]:
        train_chat_text.append(chat_text[idx])
        train_chat_labels.append(chat_labels[idx])
    else:
        test_chat_text.append(chat_text[idx])
        test_chat_labels.append(chat_labels[idx])


##################################################################
########################################################################################
# This function should perform a single training epoch using our training data


def train(net, device, loader, optimizer, loss_func):
    loss = 0
    # Set Network in train mode
    net.train()
    # Perform a single epoch of training on the input dataloader, logging the loss at every step
    for batch in (loader):
        x = batch['input_ids'].to(device)
        x = x.to(dtype=torch.long)
        attn = batch['attention_mask'].to(dtype=torch.long).to(device)
        logits = net(x, attn)

        y = batch['labels'].to(device)

        loss = loss_func(logits, y.to(dtype=torch.long))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss
########################################################################################
# This function should perform a single evaluation epoch, it WILL NOT be used to train our model


def evaluate(net, device, loader):
    # initialise counter
    epoch_acc = []
    # Set network in evaluation mode
    net.eval()
    with torch.no_grad():
        for batch in (loader):
            x = batch['input_ids'].to(device)
            x = x.to(dtype=torch.long)
            y = batch['labels'].to(device)
            attn = batch['attention_mask'].to(dtype=torch.long).to(device)
            logits = net(x, attn)
            predicted = torch.argmax(logits, dim=1).flatten()
            epoch_acc.append((predicted == y).numpy().mean())
    # return the accuracy from the epoch
    return np.mean(epoch_acc)
##################################################################
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
# tokenize each word in the sentence
train_chat_encodings = tokenizer(
    text=train_chat_text,
    truncation=True,
    padding=True,
    return_tensors='pt')

test_chat_encodings = tokenizer(
    text=test_chat_text,
    truncation=True,
    padding=True,
    return_tensors='pt')
##################################################################
class ChatDataset():
    def __init__(self, X_train, y_train):
        super()
        self.x_data = X_train
        self.y_data = y_train
    # support indexing such that dataset[i] can be used to get i-th sample

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index])
                for key, val in self.x_data.items()}
        item['labels'] = torch.tensor(self.y_data[index])
        return item
    # we can call len(dataset) to return the size

    def __len__(self):
        return len(self.y_data)

##################################################################
##################################################################

chat_train_dataset = ChatDataset(train_chat_encodings, train_chat_labels)
chat_test_dataset = ChatDataset(test_chat_encodings, test_chat_labels)

##################################################################

#batch_size = 50
batch_size = 500
#learning_rate = 5e-5
learning_rate = 1e-2
#num_train_epochs = 2
num_epochs = 250

##################################################################

train_loader = DataLoader(dataset=chat_train_dataset,
                          batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=chat_test_dataset,
                          batch_size=20, shuffle=False, num_workers=0)

##################################################################

#chat_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels = len(chat_labels_str))
chat_model = FineTunedModel(len(chat_labels_str), model_name)
# print(chat_model)
optimizer = torch.optim.AdamW(chat_model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

### TRAINING LOOP ###
training_loss_logger = []
training_acc_logger = []
testing_acc_logger = []
for epoch in range(num_epochs):
    training_loss = train(chat_model, device, train_loader, optimizer, loss_func)
    train_acc = evaluate(chat_model, device, train_loader)
    test_acc = evaluate(chat_model, device, test_loader)
    training_acc_logger.append(train_acc)
    testing_acc_logger.append(test_acc)
    training_loss_logger.append(training_loss.item())
    if (epoch % 5 == 0):
        print(
            f'| Epoch: {epoch:02} |  Train Loss: {training_loss.item():.4f} | Train Acc: {train_acc*100:05.2f}% | Test Acc: {test_acc*100:05.2f}% |')
##################################################################
plt.plot(training_loss_logger)
plt.title('model training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

plt.title('model accuracy')
plt.plot(training_acc_logger)
plt.plot(testing_acc_logger)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
##################################################################
data = {
    "model_state": chat_model.state_dict(),
    "output_size": len(chat_labels_str),
    "model_name": model_name}

torch.save(data, "model_chatbot.pth")
