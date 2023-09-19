from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def add_noise(y, noise_level):
    y = y.copy()
    n = int(noise_level * len(y))
    indices = np.random.choice(len(y), n, replace=False)
    y[indices] = np.random.choice(len(np.unique(y)), n)
    return y

if __name__ == "__main__":
    #data
    category_names = ['comp.graphics', 'sci.space']
    training_data = fetch_20newsgroups(subset="train",categories=category_names)
    test_data = fetch_20newsgroups(subset='test',categories=category_names)
    
    # preprocessing
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(training_data.data)
    X_test = vectorizer.transform(test_data.data)
    
    y_train = training_data.target
    y_test = test_data.target

    # Classification
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    # Add some noise to see behavior
    results = []
    noise_levels = np.linspace(0,1,11)

    for noise_level in noise_levels:
        y_train = add_noise(y_train, noise_level)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"noise_level={noise_level}, accuracy={accuracy_score(y_test, y_pred)}")
        results.append(accuracy_score(y_test, y_pred))

    plt.plot(noise_levels, results)
    plt.xlabel("noise level")
    plt.ylabel("accuracy")
    plt.show()

    

    