import spacy 
from nltk.stem import PorterStemmer
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordNERTagger
nltk.download("punkt")
nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_sm")
stemmer=PorterStemmer()
text="the runner was running quickly towards the finish line o claim the prize."
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
print("------------NER-----------")
stanford_ner_tagger = StanfordNERTagger(
    '/path/to/stanford-ner/stanford-ner.jar',
    '/path/to/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'
)

tokens = nltk.word_tokenize(text)
classified_text = stanford_ner_tagger.tag(tokens)

print(classified_text)
