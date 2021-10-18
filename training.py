import createIntents
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import Dataset, DataLoader
from time import process_time
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
##################################################################
with open('intents.json', 'r') as f:
    intents = json.load(f)
##################################################################
chat_text =[]
chat_labels_str = []
chat_labels = []
id2label_dict = {}
label2id_dict = {}

for idx, intent in enumerate(intents['intents']):
    label = intent['labels']
    chat_labels_str.append(label)
    i = 0
    for pattern in intent['patterns']:
        chat_text.append(pattern)
        chat_labels.append(idx)
        if i == 0:
            id2label_dict[idx] = f'{label}'
            label2id_dict[f'{label}'] = idx
            i += 1
##################################################################
########################################################################################
#This function should perform a single training epoch using our training data
def train(net, device, loader, optimizer):
    loss = 0
    #Set Network in train mode
    net.train()
    #Perform a single epoch of training on the input dataloader, logging the loss at every step 
    for batch in (loader):
        x = batch['input_ids'].to(device)
        y = batch['labels'].to(device)
        attn = batch['attention_mask'].to(device)
        y_hat = net(x,attn,y)
        loss = y_hat[0]
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
    #return the logger array       
    #return loss
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
            y = batch['labels'].to(device)
            attn = batch['attention_mask'].to(device)
            y_hat = net(x,attn,y)
            epoch_acc += (y_hat.argmax(1) == y.to(device)).sum().item()
            
    #return the accuracy from the epoch 
    return epoch_acc / len(loader.dataset)    
##################################################################
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
# tokenize each word in the sentence
chat_encodings = tokenizer(text = chat_text, truncation = True, padding = True)
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

batch_size = 50
output_dir = './results'
num_train_epochs=2
per_device_train_batch_size=16
per_device_eval_batch_size=64
warmup_steps = 100
learning_rate = 5e-5
weight_decay=0.01
logging_dir='./logs'
logging_steps=10

##################################################################
class FineTunedModel(nn.Module):
    def __init__(self):
        super(FineTunedModel, self).__init__()
        self.base_model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, len(chat_labels_str))
        
    def forward(self, input_ids, attn_mask, labels):
        outputs = self.base_model(input_ids, attention_mask=attn_mask, labels = labels)
        print(outputs)
        outputs = self.dropout(outputs[0])
        outputs = self.linear(outputs)
        
        return outputs
##################################################################
train_loader = DataLoader(dataset=chat_train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
##################################################################
chat_model = FineTunedModel()
optimizer = torch.optim.Adam(chat_model.parameters(), lr=learning_rate)
for epoch in range(num_train_epochs):
    train(chat_model, device, train_loader, optimizer)

# print(f'| Epoch: {epoch+1:02}/{num_epochs} | Train Acc: {train_acc*100:05.2f}% | Train Loss: {training_loss.item():.4f}')

# def save_model(FILE, model, dict_size, input_size, output_size, kernel_size,embedding_vector_length,num_layers,filter_num):
#     data = {
#     "model_state": model.state_dict(),
#     "dict_size": dict_size,
#     "input_size": input_size,
#     "output_size": output_size,
#     "kernel_size": kernel_size,
#     "embedding_vector_length": embedding_vector_length,
#     "num_layers": num_layers,
#     "filter_num": filter_num
#     }    
#     torch.save(data, FILE)
##########################################################################
##########################################################################
