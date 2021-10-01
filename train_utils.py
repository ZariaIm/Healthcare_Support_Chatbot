import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
from createIntentAllWords import all_words, chat_labels


class ChatDataset(Dataset):

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

#This function should perform a single training epoch using our training data
def train(net, device, loader, optimizer, loss_fun):
    loss = 0
    #Set Network in train mode
    net.train()
    #Perform a single epoch of training on the input dataloader, logging the loss at every step 
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(dtype=torch.long).to(device) 
        y_hat = net(x)
        loss = loss_fun(y_hat, y)
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
    #return the logger array       
    return loss

#This function should perform a single evaluation epoch, it WILL NOT be used to train our model
def evaluate(net, device, loader):
    
    #initialise counter
    epoch_acc = 0
    
    #Set network in evaluation mode
    net.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(dtype=torch.long).to(device) 
            y_hat = net(x)
            #log the cumulative sum of the acc
            epoch_acc += (y_hat.argmax(1) == y.to(device)).sum().item()
            
    #return the accuracy from the epoch 
    return epoch_acc / len(loader.dataset)    

def initialise(device, X_train, y_train, batch_size, learning_rate, input_size, hidden_size, output_size):
    print(f" --- input size: {input_size}; output_size: {output_size} --- ")

    dataset = ChatDataset(X_train, y_train)
    loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer, loader
    

def training_loop(device, num_epochs, model, loader,optimizer, criterion):
    chat_training_loss_logger = []
    chat_training_acc_logger = []

    for epoch in range(num_epochs):
        training_loss = train(model, device, loader, optimizer, criterion)
        train_acc = evaluate(model, device, loader)
        chat_training_acc_logger.append(train_acc)
        chat_training_loss_logger.append(training_loss.item())
        if (epoch%50 == 0):    
            #print(f'| Epoch: {epoch+1:.4f} | Train Acc: {train_acc*100:.4f}% | Train Loss: {training_loss.item():.4f}')
            print (f'Epoch [{epoch}/{num_epochs}], Loss: {training_loss.item():.4f}, Acc: {train_acc}')
    return model

def save_model(FILE, model, input_size, hidden_size, output_size):
    chat_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "labels": chat_labels
    }
    #print(labels)
    
    torch.save(chat_data, FILE)
