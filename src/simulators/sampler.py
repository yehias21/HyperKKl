# Library imports
from smt.sampling_methods import LHS, Random
import numpy as np
from typing import List


# Sampling functions
def get_sampler(sampler_type: str, seed: int = None, sample_space: List[List[int]] = None, delta: float = None):
    match sampler_type.lower():
        case "lhs":
            sample_space = np.array(sample_space)
            assert sample_space.ndim == 2, "sample_space should be a 2D array"
            return LHS(xlimits=sample_space, random_state=seed)
        case "circular":
            assert delta is not None, "delta should be provided for circular sampler"
            return CircularSampler(delta)
        case "spherical":
            assert delta is not None, "delta should be provided for spherical sampler"
            return SphericalSampler(delta)
        case "uniform":
            sample_space = np.array(sample_space)
            assert sample_space.ndim == 2, "sample_space should be a 2D array"
            return UniformSampler(xlimits=sample_space, seed=seed)
        case "normal":
            sample_space = np.array(sample_space)
            assert sample_space.ndim == 2, "sample_space should be a 2D array"
            return NormalSampler(xlimits=sample_space)
        case _:
            raise NotImplementedError(f"{sampler_type} is not a valid sampler")


class CircularSampler:
    def __init__(self, delta: np.ndarray):
        self.delta = delta

    def __call__(self, num_samples: int) -> np.ndarray:
        ic = []
        for distance in self.delta:
            r = distance + np.sqrt(2)
            angles = np.arange(0, 2 * np.pi, 2 * np.pi / num_samples)
            x = r * np.cos(angles, np.zeros([1, num_samples])).T
            y = r * np.sin(angles, np.zeros([1, num_samples])).T
            init_cond = np.concatenate((x, y), axis=1)
            ic.append(np.expand_dims(init_cond, axis=1))
        return np.array(ic)


class SphericalSampler:
    def __init__(self, delta: np.ndarray):
        self.delta = delta

    def __call__(self, num_samples: int) -> np.ndarray:
        r = self.delta + np.sqrt(0.02)
        theta = np.arange(0, 2 * np.pi, 2 * np.pi / num_samples)
        phi = np.arange(0, np.pi, (np.pi) / num_samples)

        x = lambda r, theta, phi: r * np.cos(theta) * np.sin(phi)
        y = lambda r, theta, phi: r * np.sin(theta) * np.sin(phi)
        z = lambda r, phi: r * np.cos(phi)

        sphere = []
        for radius in r:
            circles = []
            for angle in phi:
                x_coord = x(radius, theta, np.ones(len(theta)) * angle).reshape(-1, 1)
                y_coord = y(radius, theta, np.ones(len(theta)) * angle).reshape(-1, 1)
                z_coord = z(radius, np.ones(len(theta)) * angle).reshape(-1, 1)
                circle_coord = np.concatenate((x_coord, y_coord, z_coord), axis=1)
                circles.append(circle_coord)
            sphere.append(circles)

        return np.array(sphere)


class UniformSampler:
    def __init__(self, xlimits, seed=0):
        self.xlimits = np.array(xlimits)
        self.random = np.random.RandomState(seed)

    def __call__(self, num_samples):
        return self.random.uniform(low=self.xlimits[:, 0], high=self.xlimits[:, 1],
                                   size=(num_samples, len(self.xlimits)))


class NormalSampler:
    def __init__(self, xlimits, seed=0):
        self.xlimits = np.array(xlimits)
        self.random = np.random.RandomState(seed)

    def __call__(self, num_samples):
        mean = 0.5 * (self.xlimits[:, 0] + self.xlimits[:, 1])
        std_dev = 0.5 * (self.xlimits[:, 1] - self.xlimits[:, 0])
        samples = mean + std_dev * self.random.randn(num_samples, len(self.xlimits))
        return samples
