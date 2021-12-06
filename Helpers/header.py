import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sammon import sammon
from sklearn.decomposition import PCA
import matplotlib as mpl
from sklearn.manifold import Isomap
from sklearn.datasets import fetch_olivetti_faces

def load_from_csv(filename):
    """
    Loads a CSV file into a pandas dataframe.
    """
    return pd.read_csv(filename)

def load_from_npy(filename):
    """
    Loads a NPY file into a pandas dataframe.
    """
    return np.load(filename)