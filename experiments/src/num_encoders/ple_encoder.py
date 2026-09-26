import numpy as np
import pandas as pd


class PiecewiseLinearEncoder:
    def __init__(self, n_bins):
        """Инициализация энкодера"""
        self.n_bins = n_bins
        self.feature_name_to_borders = {}

    def fit(self, x):
        """Вычисляет границы интервалов для каждого признака"""
        for feature_name in x.columns:
            values = x[feature_name].values
            min_value, max_value = values.min(), values.max()

            # Если все значения одинаковы, создаем искусственные границы
            if min_value == max_value:
                borders = [min_value - 1, min_value, max_value + 1]
            else:
                borders = np.quantile(
                    values,
                    np.linspace(0, 1, self.n_bins + 1)
                ).tolist()

            self.feature_name_to_borders[feature_name] = borders

    def transform(self, x):
        """Преобразует признаки в кусочно-линейное представление"""
        encoded_features = []

        for feature in x.columns:
            values = x[feature].values
            borders = self.feature_name_to_borders[feature]
            n_bins = len(borders) - 1

            encoded = np.zeros((len(values), n_bins))

            if len(set(borders)) == 2:
                encoded[:, 0] = 1
            else:
                bin_indices = np.digitize(values, borders[1:-1], right=True)

                for i, val in enumerate(values):
                    val_clipped = np.clip(val, borders[0], borders[-1])
                    bin_idx = min(bin_indices[i], n_bins - 1)

                    left, right = borders[bin_idx], borders[bin_idx + 1]
                    if right != left:
                        encoded[i, bin_idx] = (val_clipped - left) / (right - left)

                    encoded[i, bin_idx + 1:] = 1

            encoded_features.append(encoded)

        return pd.DataFrame(np.hstack(encoded_features))