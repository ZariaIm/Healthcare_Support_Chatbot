import numpy as np
import torch
import torch.nn as nn


from createIntentAllWords import X_train_chat, y_train_chat
from createSymptomAllWords import X_train_symptom, y_train_symptom, X_val, y_val, X_test, y_test
from train_utils import evaluate, initialise, initialise_with_val,training_loop, save_model, training_loop_with_val_loader
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
[model, criterion, optimizer, trainloader] = initialise(device, X_train, y_train, batch_size, learning_rate, input_size, hidden_size, output_size)
print("Chatbot Model initialised. Entering Training Loop.")
[model, training_loss_logger,training_acc_logger] = training_loop(device, num_epochs, model, trainloader,optimizer, criterion)
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
[model, criterion, optimizer, trainloader, valloader, testloader] = initialise_with_val(device, X_train, y_train, X_val, y_val, X_test, y_test, batch_size, learning_rate, input_size, hidden_size, output_size)
print("Classifier Model initialised. Entering Training Loop.")
[model,symptom_training_loss_logger, sypmtom_training_acc_logger,symptom_validation_acc_logger] = training_loop_with_val_loader(device, num_epochs, model, trainloader, valloader,optimizer, criterion)
FILE = "model_symptoms.pth"
test_acc = evaluate(model, device, testloader)
print(f"test accuracy: {test_acc*100:.4f}%")
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