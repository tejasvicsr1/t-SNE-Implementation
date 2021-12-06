import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib as mpl
from sklearn.manifold import Isomap
from sklearn.datasets import fetch_olivetti_faces

# Set global parameters
NUM_POINTS = 10000            # Number of samples from MNIST
PERPLEXITY = 20
SEED = 42                    # Random seed
MOMENTUM = 0.9
LEARNING_RATE = 10.
NUM_ITERS = 500             # Num iterations to train for
NUM_PLOTS = 5               # Num. times to plot in training
PCA_DIM_NUMBER = 30

def run_PCA(data, labels, n_components=2):
    """
    Runs PCA on the data.
    """
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = principalComponents)
    return principalDf

def load_from_csv(filename):
    """
    Loads a CSV file into a pandas dataframe.
    """
    dataset = pd.read_csv(filename)
    train_data = dataset.sample(n=NUM_POINTS, random_state=SEED)

    y = train_data['label']

    X = train_data.drop(columns=['label'])
    X = run_PCA(train_data, y, PCA_DIM_NUMBER)
    X = pd.DataFrame(X).to_numpy()

    return X, y
