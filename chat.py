import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
model_c = NeuralNet(input_size, hidden_size, output_size).to(device)
model_c.load_state_dict(model_state)
model_c.eval()

FILE = "model_symptoms.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
labels = data['labels']
model_state = data["model_state"]
model_s = NeuralNet(input_size, hidden_size, output_size).to(device)
model_s.load_state_dict(model_state)
model_s.eval()

bot_name = "Sam"

def write_json(new_data, filename='storedSymptoms.json'):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["intents"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

def get_response(msg):
    
    sentence = tokenize(msg)
    check = bag_of_words(sentence, full_list_of_symptoms)
    if sum(check)>0:
        print("symptom identified")

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model_c(X)
    _, predicted = torch.max(output, dim=1)
    # print(predicted)
    # print(labels)
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
