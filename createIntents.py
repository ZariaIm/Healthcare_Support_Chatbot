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


#read dataset and cast values as strings
df_description = pd.read_csv("datasets/symptom_description.csv", dtype = "string")
df_precaution = pd.read_csv("datasets/symptom_precaution.csv", dtype = "string")

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
 
    # python object to be appended

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
    add = {"labels":f"precaution_{disease}",
            "patterns": [
                f"what should i do about {disease}?",
                f"what can i do if i have {disease}?", 
                f"what do i do if i have {disease}",
                f"i have {disease}",
                ],
            "responses": [f"You could {precaution1}",f"You could {precaution2}",f"You could {precaution3}",f"You could {precaution4}"]
            }
    write_json(add)

   