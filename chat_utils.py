
import json
import torch
from model import LSTM_CNN
from nltk_utils import bag_of_words
Model = LSTM_CNN

def load_saved_model(device, FILE):
    data = torch.load(FILE)
    input_size = data["input_size"]
    output_size = data["output_size"]
    model_state = data["model_state"]
    embedding_vector_length = data["embedding_vector_length"]
    kernel_size = data["kernel_size"]
    num_layers = data["num_layers"]
    filter_num = data["filter_num"]
    model = Model(input_size, output_size, embedding_vector_length, num_layers, filter_num).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model

def load_saved_words(FILE):
    data = torch.load(FILE)
    all_words = data["all_words"]
    labels = data["labels"]
    return all_words, labels

def load_saved_symptoms(FILE):
    data = torch.load(FILE)
    all_words = data["all_words"]
    labels = data["labels"]
    disease_symptoms = data["disease_symptoms"]
    emergency_symptoms = data["emergency_symptoms"]
    return all_words, labels,disease_symptoms, emergency_symptoms

def write_json(new_data, filename='storedSymptoms.json'):
    with open(filename,'r+') as file:
        file_data = json.load(file)
        file_data["symptoms"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent = 4)

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