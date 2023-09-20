from sklearn import metrics
import spacy

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm") # alternatively use en_core_web_lg since it gives better results for small models
                                       # en_core_web_sm seems to be not the best option for similarity
    tokens = nlp("cat dog cammel")

    dog = tokens[0].vector
    cat = tokens[1].vector
    cammel = tokens[2].vector

    # Similarity cat and dog
    print('Similarity cat and dog: {}'.format(metrics.pairwise.cosine_similarity([dog], [cat])))

    # Similarity cammel and dog
    print('Similarity cammel and dog: {}'.format(metrics.pairwise.cosine_similarity([dog], [cammel])))

    # Similarity cammel and cat
    print('Similarity cat and cammel: {}'.format(metrics.pairwise.cosine_similarity([cammel], [cat])))

    # Similarity method
    print(tokens[0].similarity(tokens[1]))