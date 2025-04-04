import nltk
from collections import defaultdict

def n_grams(text, n):
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and lowercase
    return list(nltk.ngrams(tokens, n))

def count_n_grams(corpus, n):

    counts = defaultdict(lambda: defaultdict(int))  # For n-grams and their contexts
    for sentence in corpus:
      for ngram in n_grams(sentence, n):
        context = ngram[:-1] # the n-1 gram context
        token = ngram[-1] # the last word of n-gram
        counts[tuple(context)][token] += 1
    return counts


def smoothed_probability(ngram, counts, n, vocabulary_size, smoothing_factor=1):

    context = ngram[:-1]
    token = ngram[-1]

    if n == 1: # unigram case
        numerator = counts[()][token] + smoothing_factor
        denominator = sum(counts[()].values()) + smoothing_factor * vocabulary_size
    else: # n-gram case
        numerator = counts[tuple(context)][token] + smoothing_factor
        denominator = sum(counts[tuple(context)].values()) + smoothing_factor * vocabulary_size
        
    return numerator / denominator


def main():
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "The brown fox is quick.",
        "The lazy dog is slow.",
        "The fox jumps."
    ]

    vocabulary = set()
    for sentence in corpus:
        for token in nltk.word_tokenize(sentence.lower()):
            vocabulary.add(token)
    vocabulary_size = len(vocabulary)

    n = 2  # Example: bigrams
    counts = count_n_grams(corpus, n)

    ngram_to_check = ("the", "fox")
    probability = smoothed_probability(ngram_to_check, counts, n, vocabulary_size, smoothing_factor=0.5) # Example smoothing factor

    print(f"N-gram: {ngram_to_check}")
    print(f"Smoothed Probability: {probability}")

    ngram_to_check = ("the",) # unigram
    probability = smoothed_probability(ngram_to_check, counts, 1, vocabulary_size, smoothing_factor=0.5) # Example smoothing factor

    print(f"N-gram: {ngram_to_check}")
    print(f"Smoothed Probability: {probability}")


if __name__ == "__main__":
    main()
