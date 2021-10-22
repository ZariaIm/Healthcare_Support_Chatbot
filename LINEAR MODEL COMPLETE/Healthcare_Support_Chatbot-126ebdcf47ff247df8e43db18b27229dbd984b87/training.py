from time import process_time
from createIntentAllWords import X_train_chat, y_train_chat, X_test_chat, y_test_chat, all_words, chat_labels
from createSymptomAllWords import X_train_symptom, y_train_symptom,X_test_symptom, y_test_symptom, all_symptoms, disease_labels
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import LinearNet
##########################################################################

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

def save_model(FILE, model, input_size, hidden_size, output_size, words, labels):
    chat_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": words,
    "labels": labels
    }
    #print(labels)
    
    torch.save(chat_data, FILE)
##########################################################################
#Hyperparameters for chatbot training
num_epochs = 300
batch_size = 100
learning_rate = 0.001
input_size = len(X_train_chat[0])
hidden_size = 8
output_size = len(y_train_chat)

X_train = X_train_chat
y_train = y_train_chat
X_test = X_test_chat
y_test = y_test_chat

dataset = ChatDataset(X_train, y_train)
trainloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)
dataset = ChatDataset(X_test, y_test)
testloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)
model = LinearNet(input_size, hidden_size, output_size).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

training_loss_logger = []
training_acc_logger = []
testing_acc_logger = []
start_time = process_time()
for epoch in range(num_epochs):
    training_loss = train(model, device, trainloader, optimizer, criterion)
    train_acc = evaluate(model, device, trainloader)
    test_acc = evaluate(model, device, testloader)
    training_acc_logger.append(train_acc)
    testing_acc_logger.append(test_acc)
    training_loss_logger.append(training_loss.item())
    if (epoch%50 == 0):    
        print (f'| Epoch: {epoch:02} |Train Loss: {training_loss:.4f}| Train Acc: {train_acc*100:05.2f}% | Test. Acc: {test_acc*100:05.2f}% |')
end_time = process_time()
print(f"Ave training time per epoch: {(start_time-end_time)/num_epochs} seconds")
FILE = "model_chatbot.pth"
save_model(FILE, model, input_size, hidden_size, output_size, all_words, chat_labels)
print(f'chatbot training complete. file saved to {FILE}')
##################################################################
plt.plot(training_loss_logger)
plt.title('Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
##################################################################
plt.title('Model Accuracy')
plt.plot(training_acc_logger)
plt.plot(testing_acc_logger)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
##########################################################################
##########################################################################
##########################################################################
##########################################################################
# Hyper-parameters for symptom classifier training
num_epochs = 300
batch_size = 100
learning_rate = 0.001
input_size = len(X_train_symptom[0])
hidden_size = 8
output_size = len(y_train_symptom)

X_train = X_train_symptom
y_train = y_train_symptom
X_test = X_test_symptom
y_test = y_test_symptom

dataset = ChatDataset(X_train, y_train)
trainloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)

dataset = ChatDataset(X_test, y_test)
testloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)
model = LinearNet(input_size, hidden_size, output_size).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

training_loss_logger = []
training_acc_logger = []
testing_acc_logger = []
for epoch in range(num_epochs):
    training_loss = train(model, device, trainloader, optimizer, criterion)
    train_acc = evaluate(model, device, trainloader)
    test_acc = evaluate(model, device, testloader)
    training_acc_logger.append(train_acc)
    testing_acc_logger.append(test_acc)
    training_loss_logger.append(training_loss.item())
    if (epoch%50 == 0):    
        print (f'| Epoch: {epoch:02} |Train Loss: {training_loss:.4f}| Train Acc: {train_acc*100:05.2f}% | Test. Acc: {test_acc*100:05.2f}% |')

FILE = "model_symptoms.pth"
save_model(FILE, model, input_size, hidden_size, output_size, all_symptoms, disease_labels)
print(f'classifier training complete. file saved to {FILE}')

##################################################################
plt.plot(training_loss_logger)
plt.title('Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
##################################################################
plt.title('Model Accuracy')
plt.plot(training_acc_logger)
plt.plot(testing_acc_logger)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
##########################################################################
##########################################################################