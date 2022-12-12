# coding="utf-8"

from typing import Tuple
import numpy as np

def make_primitive_vectors(
    a: float, 
    b: float, 
    c: float, 
    alpha_deg: float=90.0, 
    beta_deg: float=90.0, 
    gamma_deg: float=90.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    alpha: float = np.deg2rad(alpha_deg)
    beta: float = np.deg2rad(beta_deg)
    gamma: float = np.deg2rad(gamma_deg)

    cos_al: float = np.cos(alpha)
    sin_al: float = np.sin(alpha)

    cos_be: float = np.cos(beta)
    sin_be: float = np.sin(beta)

    cos_ga: float = np.cos(gamma)
    sin_ga: float = np.sin(gamma)

    avec: np.ndarray = np.array([a, 0.0, 0.0])
    bvec: np.ndarray = np.array([b*cos_ga, b*sin_ga, 0.0])

    c1: float = c*cos_be
    c2: float = c*(cos_al - cos_be*cos_ga)/sin_ga
    c3: float = np.sqrt(c*c - c1*c1 - c2*c2)

    cvec: np.ndarray = np.array([c1, c2, c3])

    return avec, bvec, cvec

def make_reciprocal_primitive_vectors(
    avec: np.ndarray, bvec: np.ndarray, cvec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    pref: float = 2.0*np.pi*np.dot(avec, np.outer(bvec, cvec))

    return pref*np.outer(bvec, cvec), pref*np.outer(cvec, avec), pref*np.outer(avec, bvec)

def make_reciprocal_lattice_vector(
    h: int, k: int, l: int, 
    astar: np.ndarray, bstar: np.ndarray, cstar: np.ndarray
    ) -> np.ndarray:

    return h*astar + k*bstar + l*cstar

if __name__ == "__main__":
    pass