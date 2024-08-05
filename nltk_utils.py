import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    #Tokenize a sentence into words/tokens
    return nltk.word_tokenize(sentence)


def stem(word):
    #Stem a word to its root form
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    # Stem each word in the tokenized sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize a bag with zeros
    bag = np.zeros(len(words), dtype=np.float32)
    # Set the corresponding position to 1 if the word is in the tokenized sentence
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
"""
Create a bag-of-words vector:
- tokenized_sentence: List of words/tokens in the sentence.
- words: List of known words (vocabulary).
"""