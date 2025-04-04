import spacy 
from nltk.stem import PorterStemmer
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_sm")
stemmer=PorterStemmer()
text="sathvika is going to college  which is located at melumoi."
tokens=word_tokenize(text)
print(tokens)
docs=nlp(text)
print("------------pos tagging-------------")
for token in docs:
    print(token.text,token.pos_,token.tag_,token.dep_)
print("------------lemmatization-----------")
for token in docs:
    print(token.text,token.lemma_)
print("------------stemming-----------")
tokenizer = nltk.word_tokenize(text)
for token in tokenizer:
    print(token,stemmer.stem(token))
for ent in docs.ents:
    print(ent.text,ent.label_)
