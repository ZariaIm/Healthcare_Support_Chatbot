import random
import json
import torch
from nltk_utils import bag_of_words, tokenize
from chat_utils import load_saved_model, load_saved_symptoms, load_saved_words, predict_disease, predict_intent, store_symptom

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Clearing previous symptoms")
with open("storedSymptoms.json",'w') as file:
        json.dump({"symptoms":[]}, file)

FILE = "model_chatbot.pth"
chatbot_model = load_saved_model(device, FILE)
FILE = "model_symptoms.pth"
classifier_model = load_saved_model(device, FILE)
FILE = "chat.pth"
[all_words, chat_labels] = load_saved_words(FILE)
FILE = "disease.pth"
[all_symptoms, disease_labels, disease_symptoms] = load_saved_symptoms(FILE)

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
            [index, prob_disease, list_of_symptoms] = predict_disease(disease_labels, classifier_model, all_symptoms)
            if prob_disease.item() > 0.15:
                return f"You may have {disease_labels[index]}. I'm {torch.round(prob_disease*100)}% confident in my prediction. The symptoms I used to make the prediction are {list_of_symptoms}. The symptoms of {disease_labels[index]} are {disease_symptoms[index]}"
            else:
                return f"I think I need more symptoms to be sure.. I'm only {torch.round(prob_disease*100)}% confident in my prediction."
        for intent in intents['intents']:
            if label_intent == [ctr]:
                 return random.choice(intent['responses'])
            ctr +=1
            #print(label_intent)

    return "I do not understand..."
    