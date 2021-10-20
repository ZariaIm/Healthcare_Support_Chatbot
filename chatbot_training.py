import createIntents
import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import Dataset, DataLoader
from time import process_time
from transformers import DistilBertTokenizerFast
import json
from model import FineTunedModel
model_name = "distilbert-base-uncased"
##################################################################

with open('intents.json', 'r') as f:
    intents = json.load(f)

##################################################################

chat_text =[]
chat_labels_str = []
chat_labels = []


for idx, intent in enumerate(intents['intents']):
    label = intent['labels']
    chat_labels_str.append(label)
    i = 0
    for pattern in intent['patterns']:
        chat_text.append(pattern)
        chat_labels.append(idx)

##################################################################
########################################################################################
#This function should perform a single training epoch using our training data
def train(net, device, loader, optimizer, loss_func):
    loss = 0
    #Set Network in train mode
    net.train()
    #Perform a single epoch of training on the input dataloader, logging the loss at every step 
    for batch in (loader):
        x = batch['input_ids'].to(device)
        x = x.to(dtype=torch.long)
        attn = batch['attention_mask'].to(dtype=torch.long).to(device)
        logits = net(x,attn)
        
        y = batch['labels'].to(device)
        # y_onehot = y.numpy()
        # y_onehot = (np.arange(len(chat_labels_str)) == y_onehot[:,None]).astype(np.float32)
        # y = torch.from_numpy(y_onehot)
        loss = loss_func(logits, y.to(dtype=torch.long))
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()      
    return loss
########################################################################################
#This function should perform a single evaluation epoch, it WILL NOT be used to train our model
def evaluate(net, device, loader):
    #initialise counter
    epoch_acc = 0
    #Set network in evaluation mode
    net.eval()
    with torch.no_grad():
        for batch in (loader):
            x = batch['input_ids'].to(device)
            x = x.to(dtype=torch.long)
            y = batch['labels'].to(device)
            attn = batch['attention_mask'].to(dtype=torch.long).to(device)
            logits = net(x,attn)
            
            predicted = torch.argmax(logits, dim=1).flatten()
            #print(y)
            #print(predicted.argmax(1))
            ## Need to fix acc calculation
            epoch_acc =(predicted ==y).numpy().mean()*100
    #return the accuracy from the epoch 
    return np.mean(epoch_acc)
##################################################################

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
# tokenize each word in the sentence
chat_encodings = tokenizer(
    text = chat_text,
    truncation = True, 
    padding = True,
    return_tensors = 'pt')

##################################################################

class ChatDataset():
    def __init__(self, X_train, y_train):
        super()
        self.x_data = X_train
        self.y_data = y_train
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.x_data.items()}
        item['labels'] = torch.tensor(self.y_data[index])
        return item 
    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.y_data)

##################################################################
# create training data and validation data 
# #################TO DO##################

##################################################################
chat_train_dataset = ChatDataset(chat_encodings, chat_labels)
chat_val_dataset = ChatDataset(chat_encodings, chat_labels)
##################################################################

#batch_size = 50
batch_size = 10
#learning_rate = 5e-5
learning_rate = 2e-10
#num_train_epochs = 2
num_train_epochs=10

##################################################################

train_loader = DataLoader(dataset=chat_train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

##################################################################

#chat_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels = len(chat_labels_str))
chat_model = FineTunedModel(len(chat_labels_str), model_name)
#print(chat_model)
optimizer = torch.optim.Adam(chat_model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

for epoch in range(num_train_epochs):
    loss = train(chat_model, device, train_loader, optimizer, loss_func)
    acc = evaluate(chat_model, device, train_loader)
    print("epoch:",epoch+1,"loss:", loss.item(), "acc:", acc)

data = {
    "model_state": chat_model.state_dict(),
    "output_size": len(chat_labels_str),
    "model_name": model_name}

torch.save(data, "model_chatbot.pth")


