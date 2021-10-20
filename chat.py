import random
import json
import torch
import nltk
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from model import FineTunedModel, LinearNet
from nltk_utils import bag_of_words, tokenize, stem

data = torch.load("model_symptoms.pth")
input_size = data["input_size"]
output_size = data["output_size"]
model_state = data["model_state"]
embedding_vector_length = data["embedding_vector_length"]
kernel_size = data["kernel_size"]
num_layers = data["num_layers"]
filter_num = data["filter_num"]
classifier_model = LinearNet(input_size, output_size, embedding_vector_length, num_layers, filter_num).to(device)
classifier_model.load_state_dict(model_state)
classifier_model.eval()

data = torch.load("model_chatbot.pth")
output_size = data["output_size"]
model_name = data["model_name"]
chat_model = FineTunedModel(output_size, model_name).to(device)
chat_model.load_state_dict(model_state)
chat_model.eval()

data = torch.load("disease.pth")
all_words = data["all_words"]
labels = data["labels"]
disease_symptoms = data["disease_symptoms"]
emergency_symptoms = data["emergency_symptoms"]

def write_json(new_data, filename='storedSymptoms.json'):
    with open(filename,'r+') as file:
        file_data = json.load(file)
        file_data["symptoms"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent = 4)

####################################NEED TO MODIFY
def predict_intent(device, X, y, model):
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
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
with open("storedSymptoms.json",'w') as file:
        json.dump({"symptoms":[]}, file)

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
    ### NEED TO MODIFY
    #X = bag_of_words(sentence, all_words)
    #[label_intent, prob_intent] = predict_intent(device, X,chat_labels, chatbot_model)
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
                    disease_words = []
                    temp_labels = [((disease).split()) for disease in disease_labels]
                    for raw_word in temp_labels:
                        stemmed_words = []
                        for j in range(len(raw_word)):
                            stemmed_words.append(stem(raw_word[j]))
                        disease_words.append(stemmed_words)
                    for word in sentence:
                        disease_ctr = 0
                        for disease_word in disease_words:
                            if stem(word) in disease_word:
                                raw_word = disease_word.index(stem(word))
                                return f"The symptoms of {disease_labels[disease_ctr]} are {disease_symptoms[disease_ctr]}"
                            disease_ctr += 1
                    return "I didn't quite catch which disease that was"
                return random.choice(intent['responses'])
        ###########################################################################
            ctr +=1
    return "I do not understand..."
    #########################################################################################