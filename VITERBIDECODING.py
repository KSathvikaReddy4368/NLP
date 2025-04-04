import nltk
from nltk.corpus import treebank
from nltk.tag import hmm

# Train HMM tagger on Treebank corpus
nltk.download('treebank')
train_sents = treebank.tagged_sents()[:3000]
tagger = hmm.HiddenMarkovModelTrainer().train(train_sents)

# Define Viterbi Decoding function
def viterbi_decode(sentence, tagger):
    states = tagger._states
    transitions = tagger._transitions
    emissions = tagger._emissions
    V = [{}]
    path = {}

    for state in states:
        V[0][state] = transitions['START'].get(state, 0) * emissions[state].get(sentence[0], 0)
        path[state] = [state]

    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}
        for state in states:
            (prob, state_max) = max(
                (V[t-1][prev_state] * transitions[prev_state].get(state, 0) * emissions[state].get(sentence[t], 0), prev_state)
                for prev_state in states
            )
            V[t][state] = prob
            new_path[state] = path[state_max] + [state]
        path = new_path

    (prob, state_max) = max((V[len(sentence)-1][state], state) for state in states)
    return path[state_max]

# Test sentence
test_sentence = "This is a simple sentence".split()

# Apply Viterbi Decoding
tagged_sentence = viterbi_decode(test_sentence, tagger)
print(list(zip(test_sentence, tagged_sentence)))
