import random
import json
import torch
import nltk
from nltk_utils import bag_of_words, tokenize, stem
from chat_utils import load_saved_model, load_saved_symptoms, load_saved_words, predict_disease, predict_intent, store_symptom
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
############################################################################################
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
[all_symptoms, disease_labels, disease_symptoms, emergency_symptoms] = load_saved_symptoms(FILE)
bot_name = "Sam"
user_name = "You"
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
############################################################################################
############################################################################################
############################################################################################
def get_response(msg):
    sentence = tokenize(msg)
    #Predicting intent for chatbot
    X = bag_of_words(sentence, all_words)
    [label_intent, prob_intent] = predict_intent(device, X,chat_labels, chatbot_model)
    #########################################################################################
    if prob_intent.item() > 0.75:
        ctr = 0
        ###########################################################################
        ###########################################################################
        for intent in intents['intents']:
            if label_intent == [ctr]:
                if intent["context"] == "predicted disease":
                    [index, prob_disease, list_of_symptoms] = predict_disease(classifier_model, all_symptoms)
                    if prob_disease.item() > 0.15:
                        return f"You may have {disease_labels[index]}. I'm {torch.round(prob_disease*100)}% confident in my prediction. The symptoms I used to make the prediction are {list_of_symptoms}. The symptoms of {disease_labels[index]} are {disease_symptoms[index]}"
                    else:
                        return f"I think I need more symptoms.. I'm only {torch.round(prob_disease*100)}% confident in my prediction."
                if intent["context"] == "experiencing symptoms":
                    for word in sentence:
                        word = stem(word)
                        if word in all_symptoms:
                            store_symptom(word)
                        if word in emergency_symptoms:
                            return random.choice(intent['responses']) + "\n Please call 000 if you are experiencing severe symptoms"
                if intent["context"] == "asking symptoms":
                    temp = []
                    temp_labels = [((disease).split()) for disease in disease_labels]
                    for i in temp_labels:
                        temp_temp = []
                        for j in range(len(i)):
                            temp_temp.append(stem(i[j]))
                        temp.append(temp_temp)
                    for word in sentence:
                        disease_ctr = 0
                        for disease in temp:
                            if stem(word) in disease:
                                i = disease.index(stem(word))
                                return f"The symptoms of {disease_labels[disease_ctr]} are {disease_symptoms[disease_ctr]}"
                            disease_ctr += 1
                    return "I didn't quite catch which disease that was"
                return random.choice(intent['responses'])
        ###########################################################################
            ctr +=1
    return "I do not understand..."
    #########################################################################################