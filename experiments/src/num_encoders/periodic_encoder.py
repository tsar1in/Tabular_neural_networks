import numpy as np
import pandas as pd


class PeriodicEncoder:
    def __init__(self, n_frequencies=3, initialization='log-linear', sigma=1.0, trainable=False):
        """Периодический энкодер с адаптивными частотами"""
        self.n_frequencies = n_frequencies
        self.initialization = initialization
        self.sigma = sigma
        self.trainable = trainable
        self.coefficients_ = None

    def fit(self, X, y=None):
        """Инициализация коэффициентов на основе данных"""
        self.n_features_ = X.shape[1]
        
        if self.initialization == 'log-linear':
            coefficients = self.sigma ** (np.arange(self.n_frequencies) / self.n_frequencies)
        elif self.initialization == 'normal':
            coefficients = np.random.normal(0, self.sigma, size=self.n_frequencies)
        
        self.coefficients_ = np.tile(coefficients, (self.n_features_, 1))
        
        return self

    def transform(self, X):
        """Преобразование признаков в периодическое пространство"""
        encoded_features = []
        feature_names = []
        
        for i, col in enumerate(X.columns):
            coeffs = self.coefficients_[i]
            
            v = 2 * np.pi * X[col].values[:, None] * coeffs
            
            sin_features = np.sin(v)
            cos_features = np.cos(v)
            
            names = [
                f"{col}_sin_f{j+1}" if self.initialization == 'log-linear' 
                else f"{col}_sin_{coeff:.2f}"
                for j, coeff in enumerate(coeffs)
            ] + [
                f"{col}_cos_f{j+1}" if self.initialization == 'log-linear' 
                else f"{col}_cos_{coeff:.2f}"
                for j, coeff in enumerate(coeffs)
            ]
            
            encoded_features.append(np.hstack([sin_features, cos_features]))
            feature_names.extend(names)
            
        return pd.DataFrame(
            np.hstack(encoded_features),
            columns=feature_names,
            index=X.index
        )