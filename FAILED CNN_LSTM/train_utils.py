import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import LSTM_CNN
########################################################################################
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
########################################################################################
#This function should perform a single training epoch using our training data
def train(net, device, loader, optimizer, loss_fun, filter_num):
    loss = 0
    #Set Network in train mode
    net.train()
    #Perform a single epoch of training on the input dataloader, logging the loss at every step 
    state_h, state_c = net.init_state(filter_num)
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(dtype=torch.long).to(device) 
        y_hat, (state_h, state_c) = net(x, (state_h, state_c))
        loss = loss_fun(y_hat, y)
        state_h = state_h.detach()
        state_c = state_c.detach()
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
    #return the logger array       
    return loss
########################################################################################
#This function should perform a single evaluation epoch, it WILL NOT be used to train our model
def evaluate(net, device, loader, filter_num):
    
    #initialise counter
    epoch_acc = 0
    
    #Set network in evaluation mode
    net.eval()
    state_h, state_c = net.init_state(filter_num)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(dtype=torch.long).to(device) 
            y_hat, (state_h, state_c) = net(x, (state_h, state_c))
            state_h = state_h.detach()
            state_c = state_c.detach()
            #log the cumulative sum of the acc
            epoch_acc += (y_hat.argmax(1) == y.to(device)).sum().item()
            
    #return the accuracy from the epoch 
    return epoch_acc / len(loader.dataset)    
########################################################################################
########################################################################################
########################################################################################
########################################################################################
def initialise(device, X_train, y_train, batch_size, learning_rate, dict_size, input_size, output_size, kernel_size,embedding_vector_length,num_layers,filter_num):
    print(f" --- input size: {input_size}; output_size: {output_size} --- ")

    dataset = ChatDataset(X_train, y_train)
    loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    

    model = LSTM_CNN(dict_size, input_size, output_size, kernel_size,embedding_vector_length,num_layers,filter_num).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer, loader
########################################################################################

########################################################################################
def training_loop(device, num_epochs, model, loader,optimizer, criterion, filter_num):
    training_loss_logger = []
    training_acc_logger = []
    for epoch in range(num_epochs):
        training_loss = train(model, device, loader, optimizer, criterion, filter_num)
        train_acc = evaluate(model, device, loader, filter_num)
        training_acc_logger.append(train_acc)
        training_loss_logger.append(training_loss.item())  
        if ((epoch+1)%10) == 0: 
            print(f'| Epoch: {epoch+1:02}/{num_epochs} | Train Acc: {train_acc*100:05.2f}% | Train Loss: {training_loss.item():.4f}')
    return model, training_loss_logger,training_acc_logger
########################################################################################

########################################################################################
def save_model(FILE, model, dict_size, input_size, output_size, kernel_size,embedding_vector_length,num_layers,filter_num):
    data = {
    "model_state": model.state_dict(),
    "dict_size": dict_size,
    "input_size": input_size,
    "output_size": output_size,
    "kernel_size": kernel_size,
    "embedding_vector_length": embedding_vector_length,
    "num_layers": num_layers,
    "filter_num": filter_num
    }    
    torch.save(data, FILE)
    ########################################################################################