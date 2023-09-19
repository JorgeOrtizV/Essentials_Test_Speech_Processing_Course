from nltk.util import ngrams
from collections import Counter

def createVocabulary(text):
    # Naive implementation:
    # vocabulary = set()
    # for char in text:
    #     if char not in vocabulary:
    #         vocabulary.add(char)
    # return vocabulary
    
    # better:
    vocabulary = set(text)
    return vocabulary

def sort_bigrams(text):
    tokens = text.split(" ")
    bigrams=list(ngrams(tokens,2))
    c = Counter(bigrams)
    return c.most_common()[0]

if __name__ == "__main__":
    # Define s
    v_size = 13
    # Create vocabulary
    training_data = "The methane lane is sane".replace(' ','') #Alternatively take it from an input
    vocabulary = createVocabulary(training_data)
    # Separate all characters with a white space for easier processing
    processing_text = ' '.join(list(training_data))
    while len(vocabulary) < v_size:
        # Obrain most common bigram
        sym1, sym2 = sort_bigrams(processing_text)[0]
        # Add it to the vocabulary
        vocabulary.add(sym1+sym2)
        # Merge most common bigrams
        processing_text = processing_text.replace(sym1+' '+sym2, sym1+sym2)
    print(vocabulary)
    print(processing_text)

