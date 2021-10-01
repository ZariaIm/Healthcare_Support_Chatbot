import random
import json
from tkinter import Label
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from chat_utils import load_saved_model, load_saved_words, predict_disease, predict_intent, store_symptom

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Clearing previous symptoms")
with open("storedSymptoms.json",'w') as file:
        json.dump({"symptoms":[]}, file)

FILE = "chatbot.pth"
chatbot_model = load_saved_model(device, FILE)

FILE = "model_symptoms.pth"
classifier_model = load_saved_model(device, FILE)
FILE = "chat.pth"
[all_words, chat_labels] = load_saved_words(FILE)
FILE = "disease.pth"
[all_symptoms, disease_labels] = load_saved_words(FILE)

bot_name = "Sam"
user_name = "You"

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

def get_response(msg):
    sentence = tokenize(msg)
    for word in sentence:
        if word in all_symptoms:
            store_symptom(word)
    #Predicting intent for chatbot
    X = bag_of_words(sentence, all_words)
    [label_intent, prob_intent] = predict_intent(device, X,chat_labels, chatbot_model)
    if prob_intent.item() > 0.75:
        ctr = 0
        
        if label_intent == [0]:
            [disease,prob_disease] = predict_disease(disease_labels, classifier_model, all_symptoms)
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
    