from transformers import DistilBertTokenizerFast
from nltk_utils import bag_of_words, tokenize, stem
from model import FineTunedModel, LinearNet
import random
import json
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = torch.load("model_symptoms.pth")
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
model_state = data["model_state"]

classifier_model = LinearNet(input_size, hidden_size, output_size).to(device)
classifier_model.load_state_dict(model_state)
classifier_model.eval()

data = torch.load("model_chatbot.pth", map_location=torch.device('cpu'))
output_size = data["output_size"]
model_name = data["model_name"]
model_state = data["model_state"]
hidden_size = data["hidden_size"]
chat_model = FineTunedModel(output_size, model_name, hidden_size).to(device)
chat_model.load_state_dict(model_state)
chat_model.eval()

data = torch.load("disease.pth")
all_symptoms = data["all_symptoms"]
disease_labels = data["disease_labels"]
disease_symptoms = data["disease_symptoms"]
emergency_symptoms = data["emergency_symptoms"]


def write_json(new_data, filename='storedSymptoms.json'):
    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data["symptoms"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)

# NEED TO MODIFY


def predict_intent(device, msg, model):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    # tokenize each word in the sentence
    chat_encodings = tokenizer(
        text=msg,
        truncation=True,
        padding=True,
        return_tensors='pt').to(device)
    #print(x)
    logits = model(chat_encodings['input_ids'], chat_encodings['attention_mask'])
    label_intent = torch.argmax(logits, dim=1).flatten()
    probs = torch.softmax(logits, dim=1)
    prob_intent = probs[0][label_intent]
    return label_intent, prob_intent


def store_symptom(word):
    add = {"symptom": f"{word}"}
    write_json(add)
    print(f"{word} was stored")


def predict_disease(model, all_symptoms):
    list_of_symptoms = []
    with open('storedSymptoms.json', 'r') as json_data:
        file_data = json.load(json_data)
    symptoms = file_data["symptoms"]
    for symptom in symptoms:
        list_of_symptoms.append(symptom["symptom"])
        list_of_symptoms = list(set(list_of_symptoms))
    symptom_bag = bag_of_words(list_of_symptoms, all_symptoms)
    output = model(torch.FloatTensor(symptom_bag).unsqueeze(0))
    _, predicted = torch.max(output, dim=1)
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return predicted.item(), prob, list_of_symptoms


############################################################################################
print("Clearing previous symptoms")
with open("storedSymptoms.json", 'w') as file:
    json.dump({"symptoms": []}, file)

bot_name = "Ayla"
user_name = "You"

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
############################################################################################
############################################################################################
############################################################################################


def get_response(msg):
    # Predicting intent for chatbot
    [label_intent, prob_intent] = predict_intent(device, msg, chat_model)
    if prob_intent.item() > 0:  # 0.75
        intent = intents['intents'][label_intent]
        if intent['context'] == "asking symptoms":
            disease_words = []
            temp_labels = [((disease).split()) for disease in disease_labels]
            for raw_word in temp_labels:
                stemmed_words = []
                for j in range(len(raw_word)):
                    stemmed_words.append(stem(raw_word[j]))
                disease_words.append(stemmed_words)
            for word in tokenize(msg):
                disease_ctr = 0
                for disease_word in disease_words:
                    if stem(word) in disease_word:
                        raw_word = disease_word.index(stem(word))
                        return f"The symptoms of {disease_labels[disease_ctr]} are {disease_symptoms[disease_ctr]}"
                    disease_ctr += 1
        if intent["context"] == "predicted disease":
            [index, prob_disease, list_of_symptoms] = predict_disease(
                classifier_model, all_symptoms)
            if prob_disease.item() > 0.15:
                return f"You may have {disease_labels[index]}. I'm {torch.round(prob_disease*100)}% confident in my prediction. The symptoms I used to make the prediction are {list_of_symptoms}. The symptoms of {disease_labels[index]} are {disease_symptoms[index]}"
            else:
                return f"I think I need more symptoms.. I'm only {torch.round(prob_disease*100)}% confident in my prediction."
        if intent["context"] == "experiencing symptoms":
            for word in tokenize(msg):
                word = stem(word)
                if word in all_symptoms:
                    store_symptom(word)
                if word in emergency_symptoms:
                    return random.choice(intent['responses']) + "\n Please call 000 if you are experiencing severe symptoms"
        return random.choice(intent['responses'])
    return f"I do not understand..."
