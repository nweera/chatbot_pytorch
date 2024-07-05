import nltk
#nltk.download('punkt') # Uncomment this line to download the 'punkt' package if not already downloaded
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenize(s):
    return nltk.word_tokenize(s)

def stem(w):
    return porter.stem(w.lower())

def bag_of_words(tokenized_s, all_w):
    pass

"""a = "How long does the shipping take"
print(a)
a = tokenize(a)
print(a)"""
