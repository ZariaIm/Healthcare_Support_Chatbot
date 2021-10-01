import random
import json
from tkinter import Label
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Clearing previous symptoms")
with open("storedSymptoms.json",'w') as file:
        json.dump({"symptoms":[]}, file)

FILE = "chatbot.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
chat_labels = data['labels']
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
disease_labels = data['labels']
disease_symptoms = data['disease_symptoms']
model_state = data["model_state"]
model_s = NeuralNet(input_size, hidden_size, output_size).to(device)
model_s.load_state_dict(model_state)
model_s.eval()

bot_name = "Sam"
user_name = "You"

def write_json(new_data, filename='storedSymptoms.json'):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["symptoms"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

def get_response(msg):
    
    sentence = tokenize(msg)
    check = bag_of_words(sentence, all_symptoms)
    for word in sentence:
        if word in all_symptoms:
            add = {"symptom":f"{word}"}
            write_json(add)
            print("Symptom was stored")
   
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    #Predicting intent for chatbot
    output = model_c(X)
    _, predicted = torch.max(output, dim=1)
    label = chat_labels[predicted.item()]
    label = [label == i for i in chat_labels]
    label = [label.index(i) for i in label if i == True]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    #list_of_symptoms = ['ulcer'] #testing instead of loading from file
    if prob.item() > 0.75:
        ctr = 0
        if label == [0]:
            list_of_symptoms = []
            with open('storedSymptoms.json', 'r') as json_data:
                file_data = json.load(json_data)
                symptoms = file_data["symptoms"]
                for pair in symptoms:
                    list_of_symptoms.append(pair["symptom"])
            ############To do: remove duplicates before passing
            symptom_bag = bag_of_words(list_of_symptoms, all_symptoms)
            output_s = model_s(torch.FloatTensor(symptom_bag).unsqueeze(0))
            _, predicted_s = torch.max(output_s, dim=1)
            print(predicted_s)
            print(disease_labels)
            disease = disease_labels[predicted_s]
            return f"You may have {disease}. The symptoms of {disease} are ____"
        for intent in intents['intents']:
            if label == [ctr]:
                 return random.choice(intent['responses'])
            ctr +=1
    return "I do not understand..."
    