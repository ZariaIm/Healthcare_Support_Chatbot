import torch
from createIntentAllWords import X_train_chat, y_train_chat, chat_labels, all_words
from createSymptomAllWords import X_train_symptom, y_train_symptom, disease_labels, all_symptoms
from train_utils import initialise, training_loop, save_model
from time import process_time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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