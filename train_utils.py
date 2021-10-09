import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import LSTM_CNN, LSTM_CNN_Dropout
from createIntentAllWords import all_words, chat_labels
Model_Chatbot = LSTM_CNN
Model_Classifier = LSTM_CNN

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
    model = Model_Chatbot(input_size, hidden_size, output_size).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer, loader

def initialise_with_val(device, X_train, y_train, X_val, y_val, X_test, y_test, batch_size, learning_rate, input_size, hidden_size, output_size):
    print(f" --- input size: {input_size}; output_size: {output_size} --- ")

    dataset = ChatDataset(X_train, y_train)
    trainloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    dataset = ChatDataset(X_val, y_val)
    valloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    dataset = ChatDataset(X_test, y_test)
    testloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    model = Model_Classifier(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer, trainloader, valloader, testloader

def training_loop(device, num_epochs, model, loader,optimizer, criterion):
    training_loss_logger = []
    training_acc_logger = []
    for epoch in range(num_epochs):
        training_loss = train(model, device, loader, optimizer, criterion)
        train_acc = evaluate(model, device, loader)
        training_acc_logger.append(train_acc)
        training_loss_logger.append(training_loss.item())
        if (epoch%50 == 0):    
            print(f'| Epoch: {epoch:02} | Train Acc: {train_acc*100:05.2f}% | Train Loss: {training_loss.item():.4f}')
    return model, training_loss_logger,training_acc_logger

def training_loop_with_val_loader(device, num_epochs, model, trainloader,valloader,optimizer, criterion):
    training_loss_logger = []
    training_acc_logger = []
    validation_acc_logger = []
    for epoch in range(num_epochs):
        training_loss = train(model, device, trainloader, optimizer, criterion)
        train_acc = evaluate(model, device, trainloader)
        val_acc = evaluate(model, device, valloader)
        training_acc_logger.append(train_acc)
        validation_acc_logger.append(val_acc)
        training_loss_logger.append(training_loss.item())
        if (epoch%50 == 0):    
            print (f'| Epoch: {epoch:02} |Train Loss: {training_loss:.4f}| Train Acc: {train_acc*100:05.2f}% | Val. Acc: {val_acc*100:05.2f}% |')
    return model, training_loss_logger,training_acc_logger,validation_acc_logger

def save_model(FILE, model, input_size, hidden_size, output_size):
    chat_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "labels": chat_labels
    }    
    torch.save(chat_data, FILE)