import random
import json
from tkinter import Label
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from createAllWords import labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('storedSymptoms.json', "w") as outfile:
        outfile.write("")
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "chatbot.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
labels_c = data['labels']
model_state = data["model_state"]
model_c = NeuralNet(input_size, hidden_size, output_size).to(device)
model_c.load_state_dict(model_state)
model_c.eval()

FILE = "model_symptoms.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_symptoms = data['all_words']
labels_s = data['labels']
model_state = data["model_state"]
model_s = NeuralNet(input_size, hidden_size, output_size).to(device)
model_s.load_state_dict(model_state)
model_s.eval()

bot_name = "Sam"

def write_json(new_data, filename='storedSymptoms.json'):
    # Serializing json 
    json_object = json.dumps(new_data,separators = (",", "."))
    # Writing to sample.json
    with open(filename, "a") as outfile:
        outfile.write(json_object)

def get_response(msg):
    
    sentence = tokenize(msg)
    check = bag_of_words(sentence, all_symptoms)
    for word in sentence:
        if word in all_symptoms:
            write_json(word)
            print("symptom saved")
   
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model_c(X)
    _, predicted = torch.max(output, dim=1)
    
    label = labels_c[predicted.item()]
    label = [label == i for i in labels_c]
    label = [label.index(i) for i in label if i == True]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    #list_of_symptoms = ['ulcer'] #testing instead of loading from file
    if prob.item() > 0.6:
        ctr = 0
        if label == [0]:
            with open('storedSymptoms.json', 'r') as json_data:
                list_of_symptoms = json.load(json_data)
            symptom_bag = bag_of_words(list_of_symptoms, all_symptoms)
            output_s = model_s(torch.FloatTensor(symptom_bag).unsqueeze(0))
            _, predicted_s = torch.max(output_s, dim=1)
            disease = labels[predicted_s]
            return f"You may have {disease}."
        for intent in intents['intents']:
            if label == [ctr]:
                 return random.choice(intent['responses'])
            ctr +=1
    return "I do not understand..."
    