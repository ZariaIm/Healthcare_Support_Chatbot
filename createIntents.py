import pandas as pd
import json
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#################################################################
#################################################################
print("Clearing previous intents")
with open("intents.json", 'w') as file:
    json.dump({"intents": []}, file)
#################################################################
#################################################################
# each intent in intents has labels, patterns and responses
# function to add to JSON


def write_json(new_data, filename='intents.json'):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["intents"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)
#################################################################
#################################################################


##################################################################
# using the dataset of diseases related to which symptoms
# 4920 rows  x 18 cols
df = pd.read_csv("datasets/dataset.csv", dtype="string")
df = df.fillna(" ")
temp = []
ignore_words = ["", " "]

for i in range(1, 18):
    for word in df[f"Symptom_{i}"]:
        symptom = ((' '.join(word.split("_"))))
        if symptom not in (temp or ignore_words):
            temp.append(symptom)
            add = {"labels": f"talking about {symptom}",
                   "patterns": [
                       f"{symptom}",
                       f"I'm in {symptom}",
                       f"I'm {symptom}",
                       f"I've got a {symptom}",
                       f"i have {symptom}",
                       f"I have a {symptom}"
                   ],
                   "responses": [
                       f"ah...{symptom}. I'll add it to the list.",
                       f"ah.... {symptom}. I'll add it to the list."
                   ],
                   "context": "experiencing symptoms"
                   }
            write_json(add)
##################################################################
# using the disease descriptions and precautions data
df_description = pd.read_csv(
    "datasets/symptom_description.csv", dtype="string")
df_precaution = pd.read_csv("datasets/symptom_precaution.csv", dtype="string")
# we should probably edit some of the actual data to make it more readable
for row in range(len(df_description)):
    disease = (df_description.Disease[row])
    description = (df_description.Description[row])
    precaution1 = df_precaution.Precaution_1[row]
    precaution2 = df_precaution.Precaution_2[row]
    precaution3 = df_precaution.Precaution_3[row]
    precaution4 = df_precaution.Precaution_4[row]
    add = {"labels": f"describe_{disease}",
           "patterns": [
               f"can you tell me about {disease}?",
               f"what do you know about {disease}?",
               f"what is {disease}?",
               f"explain {disease}?",
               f"{disease}"
           ],
           "responses": [f"{description}", f"{description}"],
           "context": "about diseases"}
    write_json(add)
    add = {"labels": f"precaution_{disease}",
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
           ],
           "context": "disease precautions"
           }
    write_json(add)
    add = {"labels": f"symptoms_of_{disease}",
           "patterns": [
               f"what are the symptoms of {disease}?",
               f"how would i know if i have {disease}?",
               f"how to tell if its {disease}?",
           ],
           "context": "asking symptoms"
           }
    write_json(add)
#################################################################
# other intents
add = {
    "labels": "greeting",
    "patterns": [
        "hi",
        "hey",
        "hello",
        "yo",  # :D
        "G'day"
    ],
    "responses": [
        "How are you feeling today?", "How are you going?"
    ],
    "context": "none"
}
write_json(add)
add = {
    "labels": "good",
    "patterns": ["I'm good", "I'm great", "Feeling great", "Always good"],
    "responses": ["I'm glad, let me know if I can help with anything else", "That's good to hear, can I help in some other way"],
    "context": "none"
}
write_json(add)
add = {
    "labels": "bad",
    "patterns": ["I feel sick", "I am sick", "I'm sick", "I'm not well", "I'm feeling bad", "Feeling down", "Not doing great", "Not well"],
    "responses": ["That's not good, do you have any symptoms to tell me?", "Are you experiencing any symptoms lately that I should know about?", "If you tell me some symptoms, I might be able to help find the reason why you're unwell"],
    "context": "none"
}
write_json(add)
add = {
    "labels": "goodbye",
    "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
    "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
    "context": "none"
}
write_json(add)
add = {
    "labels": "thanks",
    "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
    "responses": ["Happy to help!", "Any time!", "My pleasure"],
    "context": "none"
}
write_json(add)
add = {
    "labels": "options",
    "patterns": ["What do you do?", "What help do you provide?", "How are you helpful?"],
    "responses": [
        "I will help you figure out what might possibly be the problem",
        "I am someone to talk to to learn about diseases, what to try if you have a particular disease and give you a possible idea of what disease you may have if you are experiencing symptoms."
    ],
    "context": "none"
}
write_json(add)
add = {
    "labels": "asking wellbeing",
    "patterns": ["how are you?", "how you doing?"],
    "responses": ["thanks for asking, I'm great.", "I'm doing well, hope you are too."],
    "context": "none"
}
write_json(add)
add = {
    "labels": "ask disease",
    "patterns": [
        "what do you think i have",
        "why am i sick",
        "What is wrong with me",
        "can you tell me what is wrong"
    ],
    "context": "predicted disease"
}
write_json(add)
###################################################################
print("Finished creating intents.json")
