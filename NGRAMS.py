from nltk import ngrams
from nltk.tokenize import word_tokenize
import string

def generate_ngrams(text, n):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    n_grams = list(ngrams(tokens, n))
    for gram in n_grams:
        print(gram)
text = "I am learning how to generate N-grams in Python."
n = 3  
generate_ngrams(text, n)
