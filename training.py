import numpy as np
import torch
import torch.nn as nn


from createIntentAllWords import all_words, chat_labels, X_train_chat, y_train_chat
from createSymptomAllWords import all_symptoms, disease_labels, X_train_symptom, y_train_symptom
from train_utils import initialise, training_loop, save_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##########################################################################
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
print("Preparing to set up the neural network")
[model, criterion, optimizer, loader] = initialise(device, X_train, y_train, batch_size, learning_rate, input_size, hidden_size, output_size)
print("Chatbot Model initialised. Entering Chatbot Training Loop.")
model = training_loop(device, num_epochs, model, loader,optimizer, criterion)
FILE = "chatbot.pth"
save_model(FILE, model, input_size, hidden_size, output_size)
print(f'chatbot training complete. file saved to {FILE}')
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
print("Preparing to set up the neural network")
[model, criterion, optimizer, loader] = initialise(device, X_train, y_train, batch_size, learning_rate, input_size, hidden_size, output_size)
print("Classifier Model initialised. Entering Chatbot Training Loop.")
model = training_loop(device, num_epochs, model, loader,optimizer, criterion)
FILE = "model_symptoms.pth"
save_model(FILE, model, input_size, hidden_size, output_size)
print(f'classifier training complete. file saved to {FILE}')
##########################################################################
##########################################################################


#     #If this model has the highest performace on the validation set 
#     #then save a checkpoint
#     #{} define a dictionary, each entry of the dictionary is indexed with a string
#     if (valid_acc > best_valid_acc):
#         best_valid_acc = valid_acc
#         print("Saving Model")
#         torch.save({
#             'epoch':                 epoch,
#             'model_state_dict':      net.state_dict(),
#             'optimizer_state_dict':  optimizer.state_dict(), 
#             'train_acc':             train_acc,
#             'valid_acc':             valid_acc,
#         }, save_path)
    
#     #clear_output(True)