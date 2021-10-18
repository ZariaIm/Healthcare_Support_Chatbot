from transformers.utils.dummy_pt_objects import DistilBertForSequenceClassification
import createIntents
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from time import process_time
from transformers import DistilBertTokenizerFast
from transformers import Trainer, TrainingArguments
import numpy as np
import json

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

##################################################################
with open('intents.json', 'r') as f:
    intents = json.load(f)
##################################################################
chat_text =[]
chat_labels = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    label = intent['labels']
    # add to tag list
    chat_labels.append(label)
    for pattern in intent['patterns']:
        # add to our words list
        chat_text.append(pattern)
#### Need to split training and validation sets
##################################################################

class ChatDataset():
    def __init__(self, X_train, y_train):
        self.x_data = X_train
        self.y_data = y_train
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.X_train.items()}
        item['labels'] = torch.tensor(self.y_data[index])
        return item 
    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.y_data)

##################################################################

# tokenize each word in the sentence
chat_encodings = DistilBertTokenizerFast.tokenize(chat_text, truncation = True, padding = True)
# create training data
chat_train_dataset = ChatDataset(chat_encodings, chat_labels)
chat_val_dataset = ChatDataset(chat_encodings, chat_labels)
##################################################################
training_args = TrainingArguments(
    output_dir = './results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps = 500,
    learning_rate = 5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = chat_train_dataset,
    eval_dataset = chat_val_dataset
)
trainer.train()

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
##########################################################################
##########################################################################

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
learning_rate = 1e-5
batch_size = 32
warmup = 600
max_seq_length = 128
num_train_epochs = 3.0

##########################################################################
##########################################################################
#Hyperparameters for chatbot training
#training set size...1320
num_epochs = 1000
batch_size = 300
learning_rate = 2e-6
input_size = len(X_train_chat[0])
output_size = len(chat_labels)
embedding_vector_length = 64
num_layers = 2
filter_num = 32
kernel_size = 9
dict_size = len(all_words)

X_train = X_train_chat
y_train = y_train_chat
print("Preparing to set up the neural network")
[model, criterion, optimizer, trainloader] = initialise(device, X_train, y_train, batch_size, learning_rate, dict_size, input_size, output_size, kernel_size,embedding_vector_length,num_layers,filter_num)
print()
print("chatbot model",model)
print()
print(f"Chatbot Model initialised. Entering Training Loop. \nBatch size: {batch_size}\nLearning rate:{learning_rate}")
Start_time_chatbot =  process_time()
[model, training_loss_logger,training_acc_logger] = training_loop(device, num_epochs, model, trainloader,optimizer, criterion, filter_num)
End_time_chatbot =  process_time()
FILE = "model_chatbot.pth"
save_model(FILE, model, dict_size, input_size, output_size, kernel_size,embedding_vector_length,num_layers,filter_num)
print(f'chatbot training complete. file saved to {FILE}')
print("time taken to train chatbot per epoch = ", (End_time_chatbot-Start_time_chatbot)/num_epochs, "seconds")
print(f"total time: {(End_time_chatbot-Start_time_chatbot)/60} minutes")
##########################################################################
##########################################################################

##########################################################################
##########################################################################
# Hyper-parameters for symptom classifier training
#num_epochs = 10
batch_size = 20
#learning_rate = 0.3
input_size = len(X_train_symptom[0])
output_size = len(disease_labels)
embedding_vector_length = 8
num_layers = 1
filter_num = 12
kernel_size = 3
dict_size = len(all_symptoms)

X_train = X_train_symptom
y_train = y_train_symptom
print("Preparing to set up the neural network")
[model, criterion, optimizer, trainloader] = initialise(device, X_train, y_train, batch_size, learning_rate, dict_size, input_size, output_size, kernel_size,embedding_vector_length,num_layers,filter_num)
print("classifier model",model)
print(f"Classifier Model initialised. Entering Training Loop.\nBatch size: {batch_size}\nLearning rate:{learning_rate}")
Start_time_classifier =  process_time()
[model, training_loss_logger,training_acc_logger] = training_loop(device, num_epochs, model, trainloader,optimizer, criterion, filter_num)
End_time_classifier =  process_time()
FILE = "model_symptoms.pth"
save_model(FILE, model, dict_size, input_size, output_size, kernel_size,embedding_vector_length,num_layers,filter_num)
print(f'classifier training complete. file saved to {FILE}')
print("time taken to train classifier per epoch = ", (End_time_classifier-Start_time_classifier)/num_epochs, "seconds")
print(f"total time: {(End_time_classifier-Start_time_classifier)/60} minutes")
