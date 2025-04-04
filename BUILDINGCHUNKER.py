import re
from nltk.tokenize import sent_tokenize
import spacy
nlp = spacy.load("en_core_web_sm")
def chunk_text(text, max_words=50):
    chunks, chunk, count = [], [], 0
    for sentence in sent_tokenize(text):
        words = len(re.findall(r'\w+', sentence))
        if count + words > max_words:
            chunks.append(" ".join(chunk))
            chunk, count = [], 0
        chunk.append(sentence)
        count += words
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

text = "Natural language processing is fascinating. It helps machines understand human language. Applications include chatbots and translation."
print(chunk_text(text, max_words=10))
doc = nlp(text)
print("Noun Phrases (NPs):")
for chunk in doc.noun_chunks:
    print(f"Chunk: {chunk.text}, Root: {chunk.root.text}")
for token in doc:
    print(f"Word: {token.text}, Dep: {token.dep_}, POS: {token.pos_}")

