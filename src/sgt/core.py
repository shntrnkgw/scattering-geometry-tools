# coding="utf-8"
"""Core routines for scattering geometry calculations. 
"""

import enum
import numpy as np
from numpy.typing import NDArray
import math

class Axis(enum.Enum):
    """Spatial axis identifiers
    """
    X = 0
    Y = 1
    Z = 2

def make_coords_in_detector_system(px_numbers: tuple[int,int], 
                                   px_sizes: tuple[float,float], 
                                   center_coord: tuple[float,float]) -> tuple[NDArray[np.float64], 
                                                                              NDArray[np.float64]]:
    
    """Makes the matrices of coordinates of each pixel on the 2D detector coordinate system. 

    The detector coordinate system is a 2D Cartesian coordinate system 
    whose origin is at the image center (= where the direct beam hits the detector plane). 
    Two axes, denoted as u and v, are defined to be parallel to the horizontal and vertical 
    edges of the detector, respectively. 

    Args:
        px_numbers: Numbers of pixels along the horizontal and vertical axes (hor, ver). 
        px_sizes: Size of a single pixel along the horizontal and vertical axes (hor, ver). 
        center_coord: (x, y) coordinate of the image center, measured from the center of a pixel at index `[0,0]`. 

    Returns:
        2D numpy arrays of the horizontal and vertical coordinates. 

    Example:
        >>> u, v = make_coords_in_detector_system( (1475, 1679), (0.172, 0.172), (132.35, 134.39) )
    """

    # u coordinates = (index of the pixel along u axis)*(pixel width) - (u coord at the center)
    u: NDArray[np.float64] = np.arange(px_numbers[0]).astype(float)*px_sizes[0] - center_coord[0]

    # v coordinates = (index of the pixel along v axis)*(pixel width) - (v coord at the center)
    v: NDArray[np.float64] = np.arange(px_numbers[1]).astype(float)*px_sizes[1] - center_coord[1]

    # matrix of u coordinates & v coordinates
    uu: NDArray[np.float64]
    vv: NDArray[np.float64]
    uu, vv = np.meshgrid(u, v, indexing="xy")

    return uu, vv

def make_rotation_matrix_around_lab_axis(axis: Axis, angle_deg: float) -> NDArray[np.float64]:
    """Makes a rotation matrix representing a rotation around an axis in the lab coordinate system. 
    
    Args:
        axis: the axis around which the rotation is performed. 
            Rotation around x means to rotate y axis toward z axis. 
            Rotation around y means to rotate z axis toward x axis. 
            Rotation around z means to rotate x axis toward y axis. 
        angle_deg: Rotation angle in degrees. 

    Returns:
        A 3x3 rotation matrix. 
    
    Example:
        >>> R = make_rotation_matrix_around_lab_axis(sgt.Axis.Y, 19.0)
    """
    ang: float = np.deg2rad(angle_deg)
    cos: float = np.cos(ang)
    sin: float = np.sin(ang)
    
    if axis == Axis.X:
        return np.array([[ 1.0, 0.0, 0.0], 
                         [ 0.0, cos,-sin], 
                         [ 0.0, sin, cos]])
    elif axis == Axis.Y:
        return np.array([[ cos, 0.0, sin], 
                         [ 0.0, 1.0, 0.0], 
                         [-sin, 0.0, cos]])
    elif axis == Axis.Z:
        return np.array([[ cos,-sin, 0.0], 
                         [ sin, cos, 0.0], 
                         [ 0.0, 0.0, 1.0]])

def make_rotation_matrix_euler_xyx(alpha_deg: float, 
                                   beta_deg: float, 
                                   gamma_deg: float) -> NDArray[np.float64]:
    """Makes a rotation matrix that defines orientation of the detector. 

    The rotation matrix is calculated based on the given Euler angles. 
    The Euler angles here is the x-y-x type, that is, 
    
    1. The initial axes :math:`(x, y, z)` are rotated by `alpha_deg` around the :math:`x` axis. 
    2. The resultant axes :math:`(x', y', z')` are rotated by `beta_deg` around the :math:`y'` axis. 
    3. The resultant axes :math:`(x'', y'', z'')` are rotated by `gamma_deg` around the :math:`x''` axis. 

    Args:
        alpha_deg: Rotation around the :math:`x` axis, in degrees. 
        beta_deg: Rotation around the :math:`y'` axis, in degrees. 
        gamma_deg: Rotation around the :math:`x''` axis, in degrees. 

    Returns:
        A 3x3 numpy array. 

    Example: 
        >>> R = make_rotation_matrix(10.0, 10.0, 10.0)
        >>> vec = np.array([1.0, 2.0, 3.0])
        >>> rotated = np.matmul(R, vec)
    """
    r1: float = math.radians(alpha_deg)
    r2: float = math.radians(beta_deg)
    r3: float = math.radians(gamma_deg)
    
    c1: float = math.cos(r1)
    c2: float = math.cos(r2)
    c3: float = math.cos(r3)
    s1: float = math.sin(r1)
    s2: float = math.sin(r2)
    s3: float = math.sin(r3)
    
    return np.array([[c2,     s2*s3,          c3*s2], 
                     [s1*s2,  c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1], 
                     [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]]) # xyx

def make_default_mask(hor_px_num: int, ver_px_num: int) -> NDArray[np.int8]:
    """Makes a default mask array. 

    A mask array is a 2D array of the same shape as the scattering image
    but of `numpy.uint8` type. Pixels to be masked are assigned with 1 
    and unmasked pixels are assigned with zero. 

    Args:
        hor_px_num: Number of pixels along the horizontal edge. 
        ver_px_num: Number of pixels along the horizontal edge. 

    Returns:
        A 2D numpy array of the dtype `numpy.uint8`. 
    """
    return np.zeros((ver_px_num, hor_px_num), dtype=np.uint8)

def make_default_array(hor_px_num: int, ver_px_num: int) -> NDArray[np.float64]:
    """Makes a default float array. 

    Args:
        hor_px_num: Number of pixels along the horizontal edge. 
        ver_px_num: Number of pixels along the horizontal edge. 

    Returns:
        A 2D numpy array of the float type. 
    """
    return np.zeros((ver_px_num, hor_px_num), dtype=np.float64)

def make_detector_basis_vectors_in_lab_system(rotation_matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], 
                                                                                             NDArray[np.float64], 
                                                                                             NDArray[np.float64]]:
    """Makes the basis vectors of the detector coordinate system, 
    expressed in the lab coordinate system. 

    For the definition of the detector coordinate system, 
    refer to :py:func:`make_coords_in_detector_system`. 
    The basis vectors of the detector coordinate system 
    is basically the basis vectors of the lab coordinate system
    being rotated by the rotation matrix that defines 
    the detector orientation. 
    The input rotation matrix can be created by 
    :py:func:`make_rotation_matrix`. 
    (but actually can be any SO(3) matrix)

    Args:
        rotation_matrix: A 3x3 numpy array representing 
            the detector orientation. 

    Returns:
        Three numpy arrays `eu`, `ev`, and `en` representing the basis vectors. 
        `eu` and `ev` are the basis vector along the horizontal and vertical
        edge of the detector, respectively, and `en` is the one 
        perpendicular to the detector plane. 
    """

    # basis vectors on the detector plane, expressed in the lab coordinate system
    eu: NDArray[np.float64] = np.matmul(rotation_matrix, np.array([1.0, 0.0, 0.0])) # in-plane basis vector of the detector plane
    ev: NDArray[np.float64] = np.matmul(rotation_matrix, np.array([0.0, 1.0, 0.0])) # in-plane basis vector of the detector plane
    en: NDArray[np.float64] = np.matmul(rotation_matrix, np.array([0.0, 0.0, 1.0])) # plane normal

    return eu, ev, en

def transform_detector_to_lab(u: NDArray[np.float64], 
                              v: NDArray[np.float64], 
                              eu: NDArray[np.float64], 
                              ev: NDArray[np.float64], 
                              sdd: float) -> tuple[NDArray[np.float64], 
                                                   NDArray[np.float64], 
                                                   NDArray[np.float64]]:
    """Make the lab system coordinate from the detector system coordinate

    Args: 
        u: horizontal coordinates in the detector system. Must be of the same shape as `v`. 
        v: vertical coordinates in the detector system. Must be of the same shape as `u`. 
        eu: basis vector of u axis on the detector, expressed in the lab system. 
        ev: basis vector of v axis on the detector, expressed in the lab system. 
        sdd: sample to detector distance. 

    Returns:
        three arrays, each with the same shape as u and v, representing coordinates in the lab system. 
    """

    x: NDArray[np.float64] = u*eu[0] + v*ev[0]
    y: NDArray[np.float64] = u*eu[1] + v*ev[1]
    z: NDArray[np.float64] = u*eu[2] + v*ev[2] + sdd

    return x, y, z

def calc_shortest_distance_to_detector(eu: NDArray[np.float64], 
                                       ev: NDArray[np.float64], 
                                       en: NDArray[np.float64], 
                                       sdd: float) -> float:
    """Computes the shortest distance from the sample to the detector plane. 

    Let P be the point on the detector such that OP is the shortest distance 
    between the origin and the detector plane. 
    The vector OP must be perpendicular to the detector plane, 
    so the vector :math:`\\vec{OP}` is proportional 
    to the detector plane normal vector :math:`\\vec{e}_n`. 
    That is, 
    
    .. math::
       \\vec{OP} = (OP)\\vec{e}_n

    The vector OP can also be expressed as

    .. math::
       \\vec{OP} = u_P\\vec{e}_u + v_P\\vec{e}_v + L\\vec{e}_z

    where :math:`(u_P, v_P)` is the coordinate of point P 
    on the detector coordinate system
    and vector :math:`\\vec{e}_z` is the z basis vector. 
    :math:`L` is the sample-to-detector distance. 

    Equating the two expressions, 

    .. math::
       k\\vec{e}_n = u_P\\vec{e}_u + v_P\\vec{e}_v + L\\vec{e}_z

    which reads

    .. math::
       u_P (\\vec{e}_u)_x + v_P (\\vec{e}_v)_x - (OP) (\\vec{e}_n)_x &= 0 \\\\
       u_P (\\vec{e}_u)_y + v_P (\\vec{e}_v)_y - (OP) (\\vec{e}_n)_y &= 0 \\\\
       u_P (\\vec{e}_u)_z + v_P (\\vec{e}_v)_z - (OP) (\\vec{e}_n)_z &= -L
    
    By defining a matrix

    .. math::
       \\mathbf{M} = 
       \\begin{pmatrix}
       (\\vec{e}_u)_x & (\\vec{e}_v)_x & -(\\vec{e}_n)_x \\\\
       (\\vec{e}_u)_y & (\\vec{e}_v)_y & -(\\vec{e}_n)_y \\\\
       (\\vec{e}_u)_z & (\\vec{e}_v)_z & -(\\vec{e}_n)_z \\\\
       \\end{pmatrix}

    the equations are simplified to

    .. math::
       \\mathbf{M} \\vec{s} &= -L \\vec{e}_z \\\\
       \\vec{s} &= -L \\mathbf{M}^{-1}\\vec{e}_z

    where :math:`\\vec{s} = (u_P, v_P, OP)`. 
    This method computes :math:`\\vec{s}` using the above equation
    and returns its third component, :math:`OP`. 

    Args:
        eu: basis vector of the detector coordinate system 
            along the horizontal edge of the detector. 
        ev: basis vector of the detector coordinate system
            along the vertical edge of the detector.  
        en: basis vector of the detector coordinate system
            along the plane normal of the detector. 
        sdd: sample-to-detector distance. 
    
    Returns:
        The distance in the float value. 
    
    Note: 
        The input vectors should be expressed 
        in the lab coordinate system. 
        Use :py:func:`make_basis_vectors_on_detector_in_lab_system`
        to generate the basis vectors. 
    """

    M: NDArray[np.float64] = np.array([[eu[0], ev[0], -en[0]],
                                       [eu[1], ev[1], -en[1]],
                                       [eu[2], ev[2], -en[2]]])
    ez: NDArray[np.float64] = np.array([0.0, 0.0, 1.0])
    s: NDArray[np.float64] = -sdd*np.matmul(np.linalg.inv(M), ez)

    return float(s[2])

def make_solid_angle_coverage_correction_factors(x: NDArray[np.float64], 
                                                 y: NDArray[np.float64], 
                                                 z: NDArray[np.float64], 
                                                 shortest_dist_to_detector: float) -> NDArray[np.float64]:
    """Makes an array with the correction factor for solid angle coverage of each pixel. 

    Based on Equation 28 in Pauw J Phys.: Condens. Matter 25, 383201. 
    DOI: 10.1088/0953-8984/25/38/383201. 

    Args:
        x: x coordinates of pixels in the lab system. 
        y: y coordinates of pixels in the lab system. 
        z: z coordinates of pixels in the lab system. 
        shortest_dist_to_detector: shortest distance from the sample to the detector plane. 
            see :py:func:`calc_shortest_dist_to_detector`. 
    
    Returns:
        A 2D array of correction factors for each pixel. 
        The correction factor is normalized at the beam center. 
        The correction can be done by multiplying this array to the intensity array. 
    """

    # Lp = (x^2 + y^2 + z^2)^(1/2)
    # Lp^3 = (x^2 + y^2 + z^2)^(3/2)

    return np.power(x*x + y*y + z*z, 3.0/2.0) / np.power(shortest_dist_to_detector, 3.0)

def make_q_vector(x: NDArray[np.float64], 
                  y: NDArray[np.float64], 
                  z: NDArray[np.float64], 
                  wavelength: float) -> tuple[NDArray[np.float64], 
                                              NDArray[np.float64], 
                                              NDArray[np.float64]]:
    '''Computes q vector. 
    
    By definition, 

    .. math::
       \\vec{q} = \\dfrac{2 \\pi}{\\lambda}(\\vec{e}_\\mathrm{s} - \\vec{e}_\\mathrm{i})

    where :math:`\\lambda` is the wavelength, 
    :math:`\\vec{e}_\\mathrm{s}` is the basis vector along the scattered ray, 
    and :math:`\\vec{e}_\\mathrm{i}` is the basis vector along the incident ray. 

    Here, :math:`\\vec{e}_\\mathrm{i}` is fixed to (0, 0, 1). 
    Since the sample is placed at the origin, 
    
    .. math::
       \\vec{e}_\\mathrm{s} = \\dfrac{\\vec{r}}{|\\vec{r}|}

    where :math:`\\vec{r}` is the coordinate of the pixel in the lab system. 

    Args:
        x: x coordinates of pixels in the lab system. 
        y: y coordinates of pixels in the lab system. 
        z: z coordinates of pixels in the lab system. 
        wavelength: wavelength of the incident beam. 

    Returns:
        Three arrays representing x, y, and z components of the q vectors. 
    '''

    ei_z: float = 1.0
    pre: float = 2.0*np.pi/wavelength

    Lp: float = np.sqrt(x*x + y*y + z*z)

    es_x: NDArray[np.float64] = x/Lp
    es_y: NDArray[np.float64] = y/Lp
    es_z: NDArray[np.float64] = z/Lp

    qx: NDArray[np.float64] = pre * es_x
    qy: NDArray[np.float64] = pre * es_y
    qz: NDArray[np.float64] = pre * (es_z - ei_z)

    return qx, qy, qz

def make_qabs(qx: NDArray[np.float64], 
              qy: NDArray[np.float64], 
              qz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes the magnitude of the q vectors. 

    Args:
        qx: x components of the q vectors. 
        qy: y components of the q vectors. 
        qz: z components of the q vectors. 
    
    Returns:
        An array of the same shape as qx, qy, and qz, 
    """

    return np.sqrt(qx*qx + qy*qy + qz*qz)

def make_azimuthal_angle_rad(qx: NDArray[np.float64], 
                             qy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Make azimuthal angle of the q vectors. 

    Args:
        qx: x components of the q vectors. 
        qy: y components of the q vectors. 
    
    Returns:
        An array of the same shape as qx and qy. 
    """

    return np.atan2(qy, qx)

def make_bin_indices(bin_edges: NDArray[np.float64], 
                     binning_coords: NDArray[np.float64], 
                     masked_pixels: NDArray[any]) -> tuple[NDArray[np.intp], 
                                                           NDArray[np.intp]]:
    """Make index mapping for binning. 

    Args:
        bin_edges: Edges of each bin. Must be sorted in an ascending order. 
        binning_coords: Coordinates (positions) used for binning. 
        masked_pixels: Truth array or index array that specifies the pixels to mask. 
    
    Returns:
        Two int arrays, `indices_in_bin` and `bin_counts`. 
        `indices_in_bin` is an array with the same size as `binning_coords`. 
        Each element is an index (starting from 1) in the bin to which the pixel belongs. 
        The index 0 means that the pixel is masked or out of the bin range. 
        `bin_counts` is an array with the same size as `bin_edges`. 
        `bin_counts[i+1]` is the number of pixels in `i`th bin. 
        `bin_counts[0]` is the number of pixels that was masked or out of the bin range. 
    """

    # r_bins contain indices in the r bin array, for each pixel. 
    # 0 means the pixel is outside the bin edges (too small)
    indices_in_bin: NDArray[np.intp] = np.searchsorted(bin_edges, binning_coords)

    # masking: setting the r_bins element to zero means that it is practically out of the bin range
    indices_in_bin[masked_pixels] = 0

    # r_bins value equal to (or larger than) len(r_edges) means that those pixels are out of bin range (too large)
    indices_in_bin[indices_in_bin == len(bin_edges)] = 0

    # count the number of pixels in each bin
    # bin_px_counts[1] = # of pixels in the 1st bin, etc. 
    # bin_px_counts[0] = # of pixels that are out of bin range. 
    # this is supposed to have the same length as the bin_edges. 
    # if the range of the bin is wider than the binning coords, 
    # there will be zeros near the end of bin_px_counts. 
    bin_counts: NDArray[np.intp] = np.bincount(indices_in_bin.ravel(), minlength=len(bin_edges))

    return indices_in_bin, bin_counts

def binning(indices_in_bin: NDArray[np.intp], 
            bin_counts: NDArray[np.intp], 
            intensities: NDArray[np.float64], 
            errors: NDArray[np.float64]) -> tuple[NDArray[np.float64], 
                                                  NDArray[np.float64]]:
    """Bin intensity & error using binning indices

    Args:
        indices_in_bin: Binning index array, that can be generated by :py:func:`make_bin_indices`. 
        bin_counts: Pixel counts in each bin, that can be generated by :py:func:`make_bin_indices`. 
        intensities: Intensity array. 
        errors: Errors of intensity. Can be zero-filled array in case the error output is not necessary. 
    
    Returns:
        Binned arrays of intensity and error. 
    """

    i_binned: NDArray[np.float64] = np.zeros_like(bin_counts, dtype=np.float64)
    e_i_binned: NDArray[np.float64] = np.zeros_like(bin_counts, dtype=np.float64)

    for bin_index in range(1, len(bin_counts)):
        
        # find pixels that belongs to the bin
        pixel_inds: NDArray[any] = indices_in_bin == bin_index
        
        # if no points in bin, insert nan
        if bin_counts[bin_index] == 0:
            i_binned[bin_index] = np.nan
            e_i_binned[bin_index] = np.nan
        # otherwise, do summation
        else:
            w = 1.0/float(bin_counts[bin_index])

            i_binned[bin_index] = np.sum(intensities[pixel_inds]) * w

            e_i_binned[bin_index] = np.sqrt(np.sum(errors[pixel_inds]**2)) * w

            # print(v)
    
    return i_binned[1:], e_i_binned[1:]

if __name__ == "__main__":
    pass