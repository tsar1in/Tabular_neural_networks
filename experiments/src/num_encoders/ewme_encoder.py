import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ElementWiseMultiplEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_dim=8):
        """Энкодер с поэлементным умножением"""
        self.embedding_dim = embedding_dim
        self.weights_ = None
        self.biases_ = None
        self.feature_names_ = None

    def fit(self, X):
        """Инициализирует веса и смещения для численных признаков"""
        n_features = len(X.columns)
        self.feature_names_ = X.columns.tolist()

        self.weights_ = np.random.randn(n_features, self.embedding_dim) * 0.01
        self.biases_ = np.zeros((1, self.embedding_dim))

        return self

    def transform(self, X):
        """Преобразует признаки в эмбеддинги """
        if self.weights_ is None:
            raise ValueError("Encoder not fitted yet")

        X_array = X[self.feature_names_].values

        embeddings = []
        feature_names = []

        for i, feature in enumerate(self.feature_names_):
            emb = self.biases_[0] + X_array[:, [i]] * self.weights_[i]
            embeddings.append(emb)

            feature_names.extend([f"{feature}_emb_{j}" for j in range(self.embedding_dim)])

        return pd.DataFrame(np.hstack(embeddings), columns=feature_names, index=X.index)