# coding="utf-8"

from typing import List, IO, Tuple
import numpy as np

import math

def calc_polar_map(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, 
                   mask: np.ndarray, 
                   qmin: float, qmax: float, 
                   N_q: int, N_azi: int) \
                   -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    @param qx: x component of the q vector
    @param qy: y component of the q vector
    @param qz: z component of the q vector
    @param mask:
    @param qmax:
    @param N_q:
    @param N_azi:

    @return map_q, map_azi, density, ax_q, ax_azi:
    map_q: array of indices in q-axis and phi-axis.
    map_azi: array of indices in phi-axis.
    density: number of pixel in the section.
    ax_q: q-axis.
    ax_azi = phi-axis.
    '''

    q: float = 0.0
    azi: float =0.0

    map_q: np.ndarray = np.zeros_like(qx, dtype=int)
    map_azi: np.ndarray = np.zeros_like(qx, dtype=int)
    density: np.ndarray = np.zeros((N_azi, N_q), dtype=int)

    height: int = qx.shape[0]
    width: int = qx.shape[1]

    i_q: int = 0
    i_azi: int = 0

    delta_q: float = (qmax-qmin)/float(N_q)
    delta_azi: float = 2.0*np.pi/float(N_azi)

    ax_q: np.ndarray = qmin + (np.arange(N_q).astype(float) + 0.5) * delta_q
    ax_azi: np.ndarray = (np.arange(N_azi).astype(float) + 0.5) * delta_azi

    j: int = 0
    k: int = 0

    for j in range(height):
        for k in range(width):
            if (mask[j,k] == 1):
                map_q[j,k] = -1
                map_azi[j,k] = -1
                continue

            q = math.sqrt(qx[j,k]*qx[j,k] + qy[j,k]*qy[j,k] + qz[j,k]*qz[j,k]) # q = |q|
            i_q = int(math.floor((q-qmin)/delta_q)) # get index (fast)
            if (i_q >= N_q) or (q < qmin):      # if the index is out of bound,
                map_q[j,k] = -1
                map_azi[j,k] = -1
                continue

            if qx[j,k] == 0.0:           # very special cases
                if qy[j,k] > 0.0:
                    azi = np.pi/2.0
                if qy[j,k] < 0.0:
                    azi = 3.0*np.pi/2.0
                else:
                    azi = 0.0
            else:
                azi = math.atan(qy[j,k]/qx[j,k])       # azimuthal angle of the q vector projected to the x-y plane
                if qx[j,k] < 0.0:                  # convert limit of azi to [0, 2pi)
                    azi = azi + np.pi
                elif qy[j,k] < 0.0:
                    azi = azi + 2.0*np.pi
            i_azi = int(math.floor(azi/delta_azi))
            if i_azi >= N_azi:               # if the index is out of bound,
                map_q[j,k] = -1
                map_azi[j,k] = -1
                continue

            map_q[j,k] = i_q
            map_azi[j,k] = i_azi
            density[i_azi, i_q] = density[i_azi, i_q] + 1 # count up only for non-masked & valid pixels
    
    return map_q, map_azi, density, ax_q, ax_azi


def circular_average(i: np.ndarray, e: np.ndarray, 
                     map_q: np.ndarray, map_azi: np.ndarray, 
                     density: np.ndarray) \
                     -> Tuple[np.ndarray, np.ndarray]:
    '''
    Do circular-average using given maps.

    @param i: scattered intensity.
    @param map_q: indices in the q-axis.
    @param map_azi: indices in the phi-axis.
    @param density: number of pixels in the section.

    @note: Masking should be done beforehand, i.e., either in calc_q_vector() or calc_polar_map().

    @return ave_i, ave_e: 2d arrays of the reduced intensity and its error.
    '''

    ave_i: np.ndarray = np.zeros_like(density, dtype=float)
    ave_e2: np.ndarray = np.zeros_like(density, dtype=float)
    densityf: np.ndarray = density.astype(float) # float version of density
    densityf2: np.ndarray = np.power(densityf, 2)

    height: int = i.shape[0]
    width: int = i.shape[1]

    j: int = 0
    k: int = 0

    i_q: int = 0
    i_azi: int = 0

    # N_q: int  = densityf.shape[1]
    # N_azi: int = densityf.shape[0]

    # intensities
    for j in range(height):
        for k in range(width):
            i_q = map_q[j,k]
            i_azi = map_azi[j,k]

            if (i_q == -1) or (i_azi == -1) or (density[i_azi,i_q] == 0.0): # these pixels are masked
                pass
            else:
                try:
                    ave_i[i_azi, i_q] += i[j,k]/densityf[i_azi,i_q]          # normalized by the number of pixels
                    ave_e2[i_azi, i_q] += e[j,k]*e[j,k]/densityf2[i_azi,i_q] # from the law of error propagation
                except ZeroDivisionError:
                    pass

    return ave_i, np.sqrt(ave_e2)

if __name__ == "__main__":
    pass