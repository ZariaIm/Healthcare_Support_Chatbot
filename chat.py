import random
import json
from tkinter import Label

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from train_symptomclassifier import all_words as sympt

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "chatbot.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
labels = data['labels']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def write_json(new_data, filename='storedSymptoms.json'):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["storedSymptoms"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

def get_response(msg):
    
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    
    label = labels[predicted.item()]
    label = [label == i for i in labels]
    label = [label.index(i) for i in label if i == True]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
   
  

    
    if prob.item() > 0.6:
        ctr = 0
        for intent in intents['intents']:
            if label == [ctr]:
                 return random.choice(intent['responses'])
            ctr +=1
    return "I do not understand..."

    if Label == sympt:
        write_json(Label, filename='storedSymptoms.json')
        

    #  if Label == sympt:
    #    write_json(Label, filename='storedSymptoms.json')
    