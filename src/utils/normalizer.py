import torch
import numpy as np


class Normalizer:
    def __init__(self, dataloader):
        self.means, self.stds = self.calculate_statistics(dataloader)

    def calculate_statistics(self, dataloader):
        means = {}
        stds = {}

        for batched_data in dataloader:
            for key, value in batched_data.items():
                if isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        inner_value = inner_value.numpy().ravel()
                        means.setdefault(inner_key, []).append(inner_value.mean())
                        stds.setdefault(inner_key, []).append(inner_value.std())
                else:
                    value = value.numpy().ravel()
                    means.setdefault(key, []).append(value.mean())
                    stds.setdefault(key, []).append(value.std())

        # Calculate overall mean and standard deviation
        means = {key: np.mean(values) for key, values in means.items()}
        stds = {key: np.std(values) for key, values in stds.items()}

        return means, stds

    def normalize(self, data, var_name):
        return (data - self.means[var_name]) / self.stds[var_name]

    def denormalize(self, normalized_data, var_name):
        return normalized_data * self.stds[var_name] + self.means[var_name]
