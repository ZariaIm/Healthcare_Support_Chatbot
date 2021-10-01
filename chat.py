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
#disease_symptoms = data['disease_symptoms']
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

def predict_intent(X, y):
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model_c(X)
    _, predicted = torch.max(output, dim=1)
    label = y[predicted.item()]
    label = [label == i for i in y]
    label = [label.index(i) for i in label if i == True]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return label, prob

def store_symptom(word):
    add = {"symptom":f"{word}"}
    write_json(add)
    print(f"{word} was stored")

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

def predict_disease(y):
    list_of_symptoms = []
    with open('storedSymptoms.json', 'r') as json_data:
        file_data = json.load(json_data)
    symptoms = file_data["symptoms"]
    for pair in symptoms:
        list_of_symptoms.append(pair["symptom"])
    ############To do: make sure we aren't adding duplicates
    symptom_bag = bag_of_words(list_of_symptoms, all_symptoms)
    output = model_s(torch.FloatTensor(symptom_bag).unsqueeze(0))
    _, predicted = torch.max(output, dim=1)
    disease = y[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return disease, prob


def get_response(msg):
    sentence = tokenize(msg)
    for word in sentence:
        if word in all_symptoms:
            store_symptom(word)
    #Predicting intent for chatbot
    X = bag_of_words(sentence, all_words)
    [label_intent, prob_intent] = predict_intent(X,chat_labels)
    if prob_intent.item() > 0.75:
        ctr = 0
        
        if label_intent == [0]:
            [disease,prob_disease] = predict_disease(disease_labels)
            if prob_disease.item() > 0.15:
                return f"You may have {disease}. I'm {torch.round(prob_disease*100)}% confident in my prediction. The symptoms of {disease} are ____"
            else:
                return f"I think I need more symptoms to be sure.. I'm only {torch.round(prob_disease*100)}% confident in my prediction."
        for intent in intents['intents']:
            if label_intent == [ctr]:
                 return random.choice(intent['responses'])
            ctr +=1
            #print(label_intent)

    return "I do not understand..."
    