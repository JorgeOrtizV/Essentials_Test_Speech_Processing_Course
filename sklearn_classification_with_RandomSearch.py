# Exercise 2.3

from sklearn.datasets import load_iris
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline




if __name__ == "__main__":
    X, y = load_iris(return_X_y=True) # Pandas formatting - DataFrames
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, stratify=y)

    # NN
    nca = NeighborhoodComponentsAnalysis(random_state=23)
    knn = KNeighborsClassifier()
    

    # Random Search
    distributions = dict(leaf_size=list(range(5,35,5)), n_neighbors=list(range(1,11)))
    random_search = RandomizedSearchCV(knn, distributions, random_state=0)
    search = random_search.fit(X_train, y_train)
    print(search.best_params_)

    # Train using results
    knn_optimized = KNeighborsClassifier(n_neighbors=7, leaf_size=15)
    nca_pipe = Pipeline([('nca', nca), ('knn', knn_optimized)])
    nca_pipe.fit(X_train, y_train)

    # Results
    print(nca_pipe.score(X_test, y_test))