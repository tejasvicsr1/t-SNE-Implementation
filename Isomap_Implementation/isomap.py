from Helpers.header import *

def run_isomap(data, labels, n_components=2):
    embedding = Isomap(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    X_transformed = pd.DataFrame(X_transformed)
    X_transformed['label'] = labels
    return X_transformed