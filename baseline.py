import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Splitting
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# Preprocessing
from sklearn.preprocessing import OneHotEncoder

# Metrics
from sklearn.metrics import accuracy_score

# Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Dense, Dropout

from collections import Counter


def get_naive_predictions(y_train, y_test):
    c = Counter(y_train)
    most_common = c.most_common()[0][0]
    predictions = [most_common]*len(y_test)
    
    print("Naive accuracy on the test set: {}".format(accuracy_score(y_test, predictions)))


def get_logreg_results(X_train, X_test, y_train, y_test):
    clf = LogisticRegression()
    skf = StratifiedKFold(n_splits=5)
    
    scores = []
    f = 0
    for train_index, test_index in skf.split(X_train, y_train):
        X_tr, X_t = X_train[train_index], X_train[test_index]
        y_tr, y_t = y_train[train_index], y_train[test_index]
        
        clf.fit(X_tr, y_tr)
        scores.append(clf.score(X_t, y_t))
        print("Fold {}: {}".format(f+1, scores[-1]))
        f+=1
    print("Logistic cross-validation accuracy: {}".format(np.mean(scores)))
    
    clf.fit(X_train, y_train)
    print("Logistic accuracy on the test set: {}".format(accuracy_score(y_test, clf.predict(X_test))))


def get_svm_results(X_train, X_test, y_train, y_test):
    clf = SVC(kernel='linear')
    skf = StratifiedKFold(n_splits=5)
    
    scores = []
    f = 0
    for train_index, test_index in skf.split(X_train, y_train):
        X_tr, X_t = X_train[train_index], X_train[test_index]
        y_tr, y_t = y_train[train_index], y_train[test_index]
        
        clf.fit(X_tr, y_tr)
        scores.append(clf.score(X_t, y_t))
        print("Fold {}: {}".format(f+1, scores[-1]))
        f+=1
    print("SVM cross-validation accuracy: {}".format(np.mean(scores)))
    
    clf.fit(X_train, y_train)
    print("SVM accuracy on the test set: {}".format(accuracy_score(y_test, clf.predict(X_test))))


def get_neural_results(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(10000, input_dim=X_train.shape[1], init='normal', activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(y_train.shape[1], init='normal', activation='linear'))

    # Compile model     #logarithmic  loss     #method
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

    skf = KFold(n_splits=5)

    scores = []
    f = 0
    for train_index, test_index in skf.split(X_train, y_train):
        X_tr, X_t = X_train[train_index], X_train[test_index]
        y_tr, y_t = y_train[train_index], y_train[test_index]

        # Fit the model
        model.fit(X_tr, y_tr, epochs=10, verbose=1)
        scores.append(accuracy_score(np.argmax(y_t,axis=-1), 
                                     np.argmax(model.predict(X_t),axis=-1)))
        print("Fold {}: {}".format(f+1, scores[-1]))
        f+=1
    print("Neural Network cross-validation accuracy: {}".format(np.mean(scores)))

    model.fit(X_train, y_train, verbose=0)
    print("Neural Network accuracy on the test set: {}".format(accuracy_score(np.argmax(y_test,axis=-1), 
                                                               np.argmax(model.predict(X_test),axis=-1))))


if __name__ == "__main__":
    features = "tfidf"  # bow or tfidf

    print("Reading {} dataset".format(features))
    df_train = pd.read_csv("data/{}_train.csv".format(features))
    df_test = pd.read_csv("data/{}_test.csv".format(features))

    X_train = df_train.drop(["label"], axis=1).as_matrix()
    X_test = df_test.drop(["label"], axis=1).as_matrix()

    y_train = df_train["label"].as_matrix()
    y_test = df_test["label"].as_matrix()

    get_naive_predictions(y_train, y_test)

    get_logreg_results(X_train, X_test, y_train, y_test)

    get_svm_results(X_train, X_test, y_train, y_test)

    # Neural network takes very long to train

    # enc = OneHotEncoder()
    # enc.fit(y_train.reshape(-1, 1))
    # y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    # y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # get_neural_results(X_train, X_test, y_train, y_test)

