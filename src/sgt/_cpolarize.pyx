# coding="utf-8"

import numpy as np
cimport numpy as np
cimport cython

cimport libc.math
from libc.math cimport round as roundc
from libc.math cimport atan2 as atan2c
from libc.math cimport atan as atanc
from libc.math cimport hypot as hypotc
from libc.math cimport ceil as ceilc
from libc.math cimport floor as floorc
from libc.math cimport fabs as fabsc
from libc.math cimport sqrt as sqrtc
from libc.math cimport M_PI
from libc.math cimport M_PI_2

@cython.boundscheck(False)
def calc_polar_map(np.float64_t[:,:] qx, np.float64_t[:,:] qy, np.float64_t[:,:] qz, 
                   np.uint8_t[:,:] mask, 
                   float qmin, float qmax, 
                   int N_q, int N_azi):
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

    cdef np.float64_t q = 0.0
    cdef np.float64_t azi = 0.0

    map_q = np.zeros_like(qx, dtype=np.int64)
    map_azi = np.zeros_like(qx, dtype=np.int64)
    density = np.zeros((N_azi, N_q), dtype=np.int64)

    cdef np.int64_t[:,:] map_q_memview = map_q
    cdef np.int64_t[:,:] map_azi_memview = map_azi
    cdef np.int64_t[:,:] density_memview = density

    cdef int height = qx.shape[0]
    cdef int width = qx.shape[1]

    cdef int i_q = 0
    cdef int i_azi = 0

    cdef np.float64_t delta_q = 0.0
    cdef np.float64_t delta_azi = 0.0
    
    delta_q = (qmax-qmin)/float(N_q)
    delta_azi = 2.0*np.pi/float(N_azi)

    ax_q = qmin + (np.arange(N_q).astype(float) + 0.5) * delta_q
    ax_azi = (np.arange(N_azi).astype(float) + 0.5) * delta_azi

    j: int = 0
    k: int = 0

    for j in range(height):
        for k in range(width):
            if (mask[j,k] == 1):
                map_q_memview[j,k] = -1
                map_azi_memview[j,k] = -1
                continue

            q = sqrtc(qx[j,k]*qx[j,k] + qy[j,k]*qy[j,k] + qz[j,k]*qz[j,k]) # q = |q|
            i_q = int(floorc((q-qmin)/delta_q)) # get index (fast)
            if (i_q >= N_q) or (q < qmin):      # if the index is out of bound,
                map_q_memview[j,k] = -1
                map_azi_memview[j,k] = -1
                continue

            if qx[j,k] == 0.0:           # very special cases
                if qy[j,k] > 0.0:
                    azi = np.pi/2.0
                if qy[j,k] < 0.0:
                    azi = 3.0*np.pi/2.0
                else:
                    azi = 0.0
            else:
                azi = atanc(qy[j,k]/qx[j,k])       # azimuthal angle of the q vector projected to the x-y plane
                if qx[j,k] < 0.0:                  # convert limit of azi to [0, 2pi)
                    azi = azi + np.pi
                elif qy[j,k] < 0.0:
                    azi = azi + 2.0*np.pi
            i_azi = int(floorc(azi/delta_azi))
            if i_azi >= N_azi:               # if the index is out of bound,
                map_q_memview[j,k] = -1
                map_azi_memview[j,k] = -1
                continue

            map_q_memview[j,k] = i_q
            map_azi_memview[j,k] = i_azi
            density_memview[i_azi, i_q] = density_memview[i_azi, i_q] + 1 # count up only for non-masked & valid pixels
    
    return map_q, map_azi, density, ax_q, ax_azi

@cython.boundscheck(False)
def circular_average(np.float64_t[:,:] i, 
                     np.float64_t[:,:] e, 
                     np.int64_t[:,:] map_q, np.int64_t[:,:] map_azi, 
                     np.int64_t[:,:] density):
    '''
    Do circular-average using given maps.

    @param i: scattered intensity.
    @param map_q: indices in the q-axis.
    @param map_azi: indices in the phi-axis.
    @param density: number of pixels in the section.

    @note: Masking should be done beforehand, i.e., either in calc_q_vector() or calc_polar_map().

    @return ave_i, ave_e: 2d arrays of the reduced intensity and its error.
    '''

    ave_i = np.zeros_like(density, dtype=np.float64)
    ave_e2 = np.zeros_like(density, dtype=np.float64)
    densityf = np.asarray(density).astype(np.float64)
    densityf2 = np.power(densityf, 2)

    cdef np.float64_t[:,:] ave_i_memview = ave_i
    cdef np.float64_t[:,:] ave_e2_memview = ave_e2
    cdef np.float64_t[:,:] densityf_memview = densityf
    cdef np.float64_t[:,:] densityf2_memview = densityf2

    cdef int height = i.shape[0]
    cdef int width = i.shape[1]

    cdef int j = 0
    cdef int k = 0

    cdef int i_q = 0
    cdef int i_azi = 0

    for j in range(height):
        for k in range(width):
            i_q = map_q[j,k]
            i_azi = map_azi[j,k]

            if (i_q == -1) or (i_azi == -1) or (density[i_azi,i_q] == 0.0): # these pixels are masked
                pass
            else:
                try:
                    ave_i_memview[i_azi, i_q] += i[j,k]/densityf_memview[i_azi,i_q]          # normalized by the number of pixels
                    ave_e2_memview[i_azi, i_q] += e[j,k]*e[j,k]/densityf2_memview[i_azi,i_q] # from the law of error propagation
                except ZeroDivisionError:
                    pass

    return ave_i, np.sqrt(ave_e2)