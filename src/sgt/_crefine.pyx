# coding="utf-8"

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sqrt as sqrtc

@cython.boundscheck(False)
def proximity(np.float64_t[:,:] intensity,
              np.float64_t[:,:] qx,
              np.float64_t[:,:] qy,
              np.float64_t[:,:] qz,
              np.float64_t[:,:] factor,
              np.uint8_t[:,:] mask,
              float qt, float qmin, float qmax, float qhwhm):

    cdef float q = 0.0
    cdef float qshift = 0.0
    cdef int height = intensity.shape[0]
    cdef int width = intensity.shape[1]
    cdef int j=0
    cdef int k=0

    cdef float val = 0.0
    cdef float qhwhm2 = qhwhm * qhwhm
    cdef float w = 0.0
    cdef float summation = 0.0
    cdef int N = 0

    for j in range(height):
        for k in range(width):
            if mask[j,k] == 1:
                continue

            q = sqrtc(qx[j,k]*qx[j,k] + qy[j,k]*qy[j,k] + qz[j,k]*qz[j,k]) # |q|

            if (qmin <= q) and (q <= qmax):
                qshift = q - qt
                w = qhwhm/(qhwhm2 + qshift*qshift) # Lorentz w/ FWHM = delta_q/2
                val += intensity[j,k]*factor[j,k]*w
                summation += intensity[j,k]*factor[j,k]
                N += 1

    if summation == 0.0:
        return 0.0
    else:
        return val/float(N)/summation
