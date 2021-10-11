from shutil import which
import numpy as np
import nltk
#nltk.download('punkt')
# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    tokens = sentence.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [''.join(gram)
                         for gram in list(find_ngrams(list(hashed_token), 3))]
    return final_tokens



def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    #return stemmer.stem(word.lower())
    #maybe no need to stem anymore
    return word


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(1000, dtype=np.float32)
    for idx, w in enumerate(words):
        for idx2, word in enumerate(sentence_words):
            if w == word: 
                bag[idx2] = idx
    return bag

# tokenized = tokenize("no hey you")
# print(tokenized)
# test_bag = bag_of_words(tokenized, ["#he", "ou#","you","hey"])
# print(test_bag)