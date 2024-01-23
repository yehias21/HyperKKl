#Library imports
from smt.sampling_methods import LHS
import numpy as np

# Sampling functions

lhs = lambda ss, seed, samples: LHS(xlimits = ss, random_state = seed)(samples)

def sample_circular(delta:np.ndarray, num_samples:int) -> np.ndarray:
    """
    Sampling of initial conditions from concentric cricles. Returns a 4D array
    of shape (Number of circles, Number of sampeles in each, 1, 2)
    """
    ic = []
    for distance in delta:
        r = distance + np.sqrt(2)
        angles = np.arange(0, 2*np.pi, 2*np.pi / num_samples)
        x = r*np.cos(angles, np.zeros([1, num_samples])).T
        y = r*np.sin(angles, np.zeros([1, num_samples])).T
        init_cond = np.concatenate((x,y), axis = 1)
        ic.append(np.expand_dims(init_cond, axis = 1))
    return np.array(ic)

def sample_spherical(delta:np.ndarray, num_samples:int) -> np.ndarray:
    """
    Sampling of initial conditions from expanding spherical shells. Returns a 4D array of
    shape (Number of spheres, Number of circles in each sphere, Number of points in each circle, 3)
    """
    r = delta + np.sqrt(0.02)
    theta = np.arange(0, 2*np.pi, 2*np.pi/num_samples)
    phi = np.arange(0, np.pi, (np.pi)/num_samples)
    x = lambda r, theta, phi: r*np.cos(theta)*np.sin(phi)
    y = lambda r, theta, phi: r*np.sin(theta)*np.sin(phi)
    z = lambda r, phi: r*np.cos(phi)

    sphere = []
    for radius in r:
        circles = []
        for angle in phi:
            x_coord = x(radius, theta, np.ones(len(theta))*angle).reshape(-1,1)
            y_coord = y(radius, theta, np.ones(len(theta))*angle).reshape(-1,1)
            z_coord =z(radius, np.ones(len(theta))*angle).reshape(-1,1)
            circle_coord = np.concatenate((x_coord, y_coord, z_coord), axis = 1)
            circles.append(circle_coord)
        sphere.append(circles)

    return np.array(sphere) 