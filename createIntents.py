import json
import torch
from nltk_utils import bag_of_words, tokenize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pandas as pd
#################################################################
#################################################################
print("Clearing previous intents")
with open("intents.json",'w') as file:
        json.dump({"intents":[]}, file)
#################################################################
#################################################################
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
 #################################################################


#################################################################
print("adding first intent")
#Make sure this one is first (it doesn't need responses)
#responses edited directly in chat.py
add = {
            "labels": "ask disease",
            "patterns": [
                "what do you think i have?",
                "why am i sick?"
            ]
        }
write_json(add)
##################################################################
print("adding symptom intents")
# using the dataset of diseases related to which symptoms
df = pd.read_csv("datasets/dataset.csv", dtype = "string") #4920 rows  x 18 cols
df = df.fillna(" ")
temp = []
for i in range(1,18):
    for word in df[f"Symptom_{i}"]:
        symptom = ((' '.join(word.split("_"))))
        if symptom not in temp:
            temp.append(symptom)
            add = {"labels":f"talking about {symptom}",
                "patterns": [
                    f"{symptom}",
                    f"I'm in {symptom}", 
                    f"what do i do if i have {symptom}",
                    f"i have {symptom}",
                    ],
                "responses": [
                    f"ah...{symptom}. I'll add it to the list, ask me 'Why am I sick?' if you want to see what I predict.",
                    f"ah.... {symptom}. I'll add it to the list, ask me 'Why am I sick?' if you want to see what I predict."
                    ]
                }
            write_json(add)

##################################################################
print("adding description/precaution intents")
#using the disease descriptions and precautions data
df_description = pd.read_csv("datasets/symptom_description.csv", dtype = "string")
df_precaution = pd.read_csv("datasets/symptom_precaution.csv", dtype = "string")
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
    add = {"labels":f"precaution_{disease}",
            "patterns": [
                f"what should i do about {disease}?",
                f"what can i do if i have {disease}?", 
                f"what do i do if i have {disease}",
                f"i have {disease}"
                ],
            "responses": [
                f"If you have {disease}, you could {precaution1}",
                f"If you have {disease}, you could {precaution2}",
                f"If you have {disease}, you could {precaution3}",
                f"If you have {disease}, you could {precaution4}"
                ]
            }
    write_json(add)
#################################################################
print("Adding last few intents")
#other intents
add = {
            "labels": "greeting",
            "patterns": [
                "hi",
                "hey",
                "hello"
            ],
            "responses": [
                "Hello there! My name is Sam. What can I do for you?",
                "Hi, what can I do for you?"
            ]
        }
write_json(add)
add = {
            "labels": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
            "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
        }
write_json(add)
add = {
            "labels": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
            "responses": ["Happy to help!", "Any time!", "My pleasure"],
        }
write_json(add)
add = {
            "labels": "options",
            "patterns": ["How you could help me?", "What help you provide?", "How you can be helpful?"],
            "responses": [
                "I will help you figure out what might possibly be the problem",
                "I am someone to talk to to learn about diseases, what to try if you have a particular disease and give you a possible idea of what disease you may have if you are experiencing symptoms."
            ]
        }
write_json(add)
add = {
    "labels": "asking wellbeing",
    "patterns": ["how are you?", "how you doing?"],
    "responses":["thanks for asking, I'm great.","I'm doing well, hope you are too."]
}
###################################################################
print("Finished creating intents.json")