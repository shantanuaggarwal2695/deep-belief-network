"""
==============================================================
Deep Belief Network features for digit classification
==============================================================

Adapted from http://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html#sphx-glr-auto-examples-neural-networks-plot-rbm-logistic-classification-py

This example shows how to build a classification pipeline with a UnsupervisedDBN
feature extractor and a :class:`LogisticRegression
<sklearn.linear_model.LogisticRegression>` classifier. The hyperparameters
of the entire model (learning rate, hidden layer size, regularization)
were optimized by grid search, but the search is not reproduced here because
of runtime constraints.

Logistic regression on raw pixel values is presented for comparison. The
example shows that the features extracted by the UnsupervisedDBN help improve the
classification accuracy.
"""

from __future__ import print_function

print(__doc__)

import numpy as np

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dbn.models import UnsupervisedDBN # use "from dbn.tensorflow import SupervisedDBNClassification" for computations on TensorFlow
import pandas as pd

###############################################################################
# Setting up


def prepare_data():
    train = pd.read_csv("../data/experiments/validation/set_5/train.csv", index_col=False, header=0)
    test = pd.read_csv("../data/experiments/validation/set_5/test.csv", index_col=False, header=0)
    cols = ["glcm_contrast_Scaled", "glcm_dissimilarity_Scaled", "glcm_homogeneity_Scaled", "glcm_energy_Scaled",
            "glcm_correlation_Scaled", "glcm_ASM_Scaled"]

    train_X = train[cols]
    test_X = test[cols]
    train_Y = train['Class']
    test_Y = test['Class']

    # train_X.shape
    train_feat_new = np.repeat(np.array(train_X), 26, axis=1)
    test_feat_new = np.repeat(np.array(test_X), 26, axis=1)

    return train_feat_new, test_feat_new, train_Y, test_Y

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

# Load Data
# digits = datasets.load_digits()
# X = np.asarray(digits.data, 'float32')
# # X, Y = nudge_dataset(X, digits.target)
# X, Y =

X_train, Y_train, X_test, Y_test = prepare_data()
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)  # 0-1 scaling
Y_train = (Y_train - np.min(Y_train, 0)) / (np.max(Y_train, 0) + 0.0001)  # 0-1 scaling

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                     test_size=0.2,
#                                                     random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression()
dbn = UnsupervisedDBN(hidden_layers_structure=[256, 512],
                      batch_size=10,
                      learning_rate_rbm=0.06,
                      n_epochs_rbm=20,
                      activation_function='sigmoid')

classifier = Pipeline(steps=[('dbn', dbn),
                             ('logistic', logistic)])

###############################################################################
# Training
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X_train, Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)

###############################################################################
# Evaluation

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))

###############################################################################
