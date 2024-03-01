import torch
import numpy as np


class Normalizer:
    def __init__(self, dataset):
        self.means, self.stds = self.calculate_statistics(dataset)

    def calculate_statistics(self, dataset):
        means = {}
        stds = {}

        for key in dataset.x_states:
            if key not in means:
                means[key] = []
                stds[key] = []

            data = dataset.x_states[key].ravel()
            means[key].append(np.mean(data))
            stds[key].append(np.std(data))

        for key in dataset.z_states:
            if key not in means:
                means[key] = []
                stds[key] = []

            data = dataset.z_states[key].ravel()
            means[key].append(np.mean(data))
            stds[key].append(np.std(data))

        if dataset.input_trajectories is not None:
            key = 'input_trajectories'
            if key not in means:
                means[key] = []
                stds[key] = []

            data = dataset.input_trajectories.ravel()
            means[key].append(np.mean(data))
            stds[key].append(np.std(data))

        return means, stds

    def normalize(self, data, var_name):
        return (data - self.means[var_name]) / self.stds[var_name]

    def denormalize(self, normalized_data, var_name):
        return normalized_data * self.stds[var_name] + self.means[var_name]
