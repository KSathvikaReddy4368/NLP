import spacy
nlp = spacy.load("en_core_web_sm")
sentence = "The quick brown fox jumps over the lazy dog."
doc = nlp(sentence)
print("Noun Phrases (NPs):")
for chunk in doc.noun_chunks:
    print(f"Chunk: {chunk.text}, Root: {chunk.root.text}")
for token in doc:
    print(f"Word: {token.text}, Dep: {token.dep_}, POS: {token.pos_}")
