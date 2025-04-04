import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
from nltk.probability import LaplaceProbDist
nltk.download('treebank')
nltk.download('punkt')
train_sents = treebank.tagged_sents()[:3000]
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_sents)
test_sentence = "This is a simple sentence".split()
tagged_sentence = hmm_tagger.tag(test_sentence)
print(tagged_sentence)
test_data = treebank.tagged_sents()[3000:3100]
accuracy = hmm_tagger.evaluate(test_data)
print( f"Accuracy: {accuracy * 100:.2f}%")
