import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
import pandas as pd

#column 1: labels/diseases/Y-train
#column 2: input/symptoms/X-train
#print(df[['Symptom_2','Symptom_3']]) # indexes both columns by name
#print(type(df.Symptom_2[3]))#4th row of symptom 2 - retrieves the class 'str'

print("Clearing previous intents")
with open("intents.json",'w') as file:
        json.dump({"intents":[]}, file)

#read dataset and cast values as strings
df_description = pd.read_csv("datasets/symptom_description.csv", dtype = "string")
df_precaution = pd.read_csv("datasets/symptom_precaution.csv", dtype = "string")
df = pd.read_csv("datasets/dataset.csv", dtype = "string") #4920 rows  x 18 cols
df = df.fillna(" ")

#each intent in intents has labels, patterns and responses
# function to add to JSON
def write_json(new_data, filename='intents.json'):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["intents"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)
 

#################################################################
#Make sure this one is first (it doesn't need responses)
#responses edited directly in chat.py
add = {
            "labels": "ask disease",
            "patterns": [
                "what do you think i have?",
                "what is it?",
                "why am i sick?",
                "I feel sick",
                "am i sick?"
            ]
        }
write_json(add)
#################################################################
add = {
            "labels": "greeting",
            "patterns": [
                "hi",
                "hey",
                "how are you",
                "hello"
            ],
            "responses": [
                "Hello there! My name is Sam. What can I do for you?",
                "Hi, what can I do for you?"
            ]
        }
write_json(add)

add = {
            "labels": "intro",
            "patterns": [
                "tell me about yourself",
                "what do you do",
                "who are you"
            ],
            "responses": [
                "I help you figure out what might possibly be the problem",
                "Someone to talk to to learn about diseases"
            ]
        }
write_json(add)

##################################################################
#using the disease descriptions and precautions data
#we should probably edit some of the actual data to make it more readable
for row in range(len(df_description)):
    disease = (df_description.Disease[row])
    description = (df_description.Description[row])
    precaution1 = df_precaution.Precaution_1[row]
    precaution2 = df_precaution.Precaution_2[row]
    precaution3 = df_precaution.Precaution_3[row]
    precaution4 = df_precaution.Precaution_4[row]
    add = {"labels":f"describe_{disease}",
            "patterns": [
                f"can you tell me about {disease}?",
                f"what do you know about {disease}?",
                f"what is {disease}?",
                f"explain {disease}?"
                ],
            "responses": [f"{description}",f"{description}"]
            }
    write_json(add)
    add = {"labels":f"identify_{disease}",
            "patterns": [f"{description}",f"{description}"],
            "responses": [f"Are you talking about {disease}",f"I think that's {disease}"]
            }
    write_json(add)
    add = {"labels":f"precaution_{disease}",
            "patterns": [
                f"what should i do about {disease}?",
                f"what can i do if i have {disease}?", 
                f"what do i do if i have {disease}",
                f"i have {disease}",
                ],
            "responses": [
            f"If you have {disease}, you could {precaution1}",
            f"If you have {disease}, you could {precaution2}",
            f"If you have {disease}, you could {precaution3}",
            f"If you have {disease}, you could {precaution4}"]
            }
    write_json(add)
##################################################################