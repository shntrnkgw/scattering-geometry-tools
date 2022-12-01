# coding="utf-8"
"""Core tools for scattering geometry calculations. 
"""

from typing import List, IO, Tuple
import numpy as np

import math

from sgt import _cpolarize

def make_rotation_matrix(alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
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

def make_pixel_coords_in_detector_system(
    hor_px_num: int, ver_px_num: int, 
    px_width: float, px_height: float, 
    center_coord_hor: float, center_coord_ver: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Makes the matrices of coordinates of each pixel on the 2D detector coordinate system. 

    The detector coordinate system is a 2D Cartesian coordinate system 
    whose origin is at the image center (= where the direct beam hits the detector plane). 
    Two axes, denoted as u and v, are defined to be parallel to the horizontal and vertical 
    edges of the detector, respectively. 

    Args:
        hor_px_num: Number of pixels along the horizontal axis. 
        ver_px_num: Number of pixels along the vertical axis. 
        px_width: Size of a single pixel along the horizontal axis. 
        px_height: Size of a single pixel along the horizontal axis. 
        center_coord_hor: Horizontal coordinate of the image center
            measured from the center of pixel at index `[0,0]`. 
        center_coord_ver: Vertical coordinate of the image center. 

    Returns:
        2D numpy arrays of the horizontal and vertical coordinates. 

    Example:
        >>> u, v = make_pixel_coords_in_detector_system(1475, 1679, 0.172, 0.172, 132.35, 134.39)
    """
    # u coordinates = (index of the pixel along u axis)*(pixel width) - (u coord at the center)
    u: np.ndarray = np.arange(hor_px_num).astype(float)*px_width - center_coord_hor

    # v coordinates = (index of the pixel along v axis)*(pixel width) - (v coord at the center)
    v: np.ndarray = np.arange(ver_px_num).astype(float)*px_height - center_coord_ver

    # matrix of u coordinates & v coordinates
    uu: np.ndarray = np.array([])
    vv: np.ndarray = np.array([])
    uu, vv = np.meshgrid(u, v, indexing="xy")

    return uu, vv

def make_default_mask(hor_px_num: int, ver_px_num: int) -> np.ndarray:
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

def make_default_array(hor_px_num: int, ver_px_num: int) -> np.ndarray:
    """Makes a default float array. 

    Args:
        hor_px_num: Number of pixels along the horizontal edge. 
        ver_px_num: Number of pixels along the horizontal edge. 

    Returns:
        A 2D numpy array of the float type. 
    """
    return np.zeros((ver_px_num, hor_px_num), dtype=float)

def make_basis_vectors_on_detector_in_lab_system(rotation_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Makes the basis vectors of the detector coordinate system, 
    expressed in the lab coordinate system. 

    For the definition of the detector coordinate system, 
    refer to :py:func:`make_pixel_coords_in_detector_system`. 
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
        Three numpy arrays `a`, `b`, and `n` representing the basis vectors. 
        `a` and `b` are the basis vector along the horizontal and vertical
        edge of the detector, respectively, and `n` is the one 
        perpendicular to the detector plane. 
    """

    # basis vectors on the detector plane, expressed in the lab coordinate system
    a: np.ndarray = np.matmul(rotation_matrix, np.array([1.0, 0.0, 0.0])) # in-plane basis vector of the detector plane
    b: np.ndarray = np.matmul(rotation_matrix, np.array([0.0, 1.0, 0.0])) # in-plane basis vector of the detector plane
    n: np.ndarray = np.matmul(rotation_matrix, np.array([0.0, 0.0, 1.0])) # plane normal

    return a, b, n


def make_pixel_coords_in_lab_system(
    xcoords_det: np.ndarray, ycoords_det: np.ndarray, 
    a: np.ndarray, b: np.ndarray, n: np.ndarray, sdd: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x: np.ndarray = xcoords_det*a[0] + ycoords_det*b[0]
    y: np.ndarray = xcoords_det*a[1] + ycoords_det*b[1]
    z: np.ndarray = xcoords_det*a[2] + ycoords_det*b[2] + sdd

    return x, y, z

def calc_shortest_dist_to_detector(a: np.ndarray, b: np.ndarray, n: np.ndarray, sdd: float) -> float:
    """Computes the shortest distance from the sample to the detector plane. 

    Let P be the point on the detector such that OP is the shortest distance 
    between the origin and the detector plane. 
    The vector OP must be perpendicular to the detector plane, 
    so the vector :math:`\\vec{\\mathrm{OP}}` is proportional 
    to the detector plane normal vector :math:`\\vec{n}`. 
    That is, 
    
    .. math::
       \\vec{\\mathrm{OP}} = \\mathrm{OP} \\vec{n}

    The vector OP can also be expressed as

    .. math::
       \\vec{\\mathrm{OP}} = u\\vec{a} + v\\vec{b} + L\\vec{e}_z

    where :math:`(u, v)` is the coordinate of point P 
    on the detector coordinate system
    and vector :math:`\\vec{e}_z` is the z basis vector. 
    :math:`L` is the sample-to-detector distance. 

    Equating the two expressions, 

    .. math::
       k\\vec{n} = u\\vec{a} + v\\vec{b} + L\\vec{e}_z

    which reads

    .. math::
       u a_x + v b_x - \\mathrm{OP} n_x &= 0 \\\\
       u a_y + v b_y - \\mathrm{OP} n_y &= 0 \\\\
       u a_z + v b_z - \\mathrm{OP} n_z &= -L
    
    By defining a matrix

    .. math::
       \\mathbf{M} = 
       \\begin{pmatrix}
       a_x & b_x & -n_x \\\\
       a_y & b_y & -n_y \\\\
       a_z & b_z & -n_z \\\\
       \\end{pmatrix}

    the equations are simplified to

    .. math::
       \\mathbf{M} \\vec{s} &= -L \\vec{e}_z \\\\
       \\vec{s} &= -L \\mathbf{M}^{-1}\\vec{e}_z

    where :math:`\\vec{s} = (u, v, \\mathrm{OP})`. 
    This method computes :math:`\\vec{s}` using the above equation
    and returns its third component, :math:`\mathrm{OP}`. 

    Args:
        a: basis vector of the detector coordinate system 
            along the horizontal edge of the detector. 
        b: basis vector of the detector coordinate system
            along the vertical edge of the detector.  
        n: basis vector of the detector coordinate system
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

    M: np.ndarray = np.array([[a[0], b[0], -n[0]],
                              [a[1], b[1], -n[1]],
                              [a[2], b[2], -n[2]]])
    ez: np.ndarray = np.array([0.0, 0.0, 1.0])
    s: np.ndarray = -sdd*np.matmul(np.linalg.inv(M), ez)

    return s[2]

def make_solid_angle_coverage_correction_factors(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, shortest_dist_to_detector: float
    ) -> np.ndarray:
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
    Lp3: np.ndarray = np.power(x*x + y*y + z*z, 3.0/2.0)

    return Lp3/np.power(shortest_dist_to_detector, 3.0)


def make_q(x: np.ndarray, y: np.ndarray, z: np.ndarray, wavelength: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes q vector. 
    
    By definition, 

    .. math::
       \\vec{q} = \dfrac{2 \\pi}{\\lambda}(\\vec{e}_\\mathrm{s} - \\vec{e}_\\mathrm{i})

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
        Three 2D arrays representing x, y, and z components of the q vector. 
    """

    ei_z: float = 1.0
    pre: float = 2.0*np.pi/wavelength

    Lp: float = np.sqrt(x*x + y*y + z*z)

    es_x: np.ndarray = x/Lp
    es_y: np.ndarray = y/Lp
    es_z: np.ndarray = z/Lp

    qx: np.ndarray = pre * es_x
    qy: np.ndarray = pre * es_y
    qz: np.ndarray = pre * (es_z - ei_z)

    return qx, qy, qz

def calc_polar_map(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, 
    mask: np.ndarray, 
    qmin: float, qmax: float,
    N_q: int, N_azi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates mapping to the polar coordinate system. 

    Args:
        qx: x component of q vector. 
        qy: y component of q vector. 
        qz: z component of q vector. 
        mask: mask array. 
        qmin: lower boundary of q. 
        qmax: upper boundary of q. 
        N_q: number of bins along the q axis. 
        N_azi: number of bins along the azimuthal axis. 
            360 deg is divided into `N_azi` sections. 
    
    Returns:
        Five numpy arrays, `map_q`, `map_azi`, `density`, `ax_q`, and `ax_azi`. 
    """

    assert qx.dtype == np.float64
    assert qy.dtype == np.float64
    assert qz.dtype == np.float64
    assert mask.dtype == np.uint8

    return _cpolarize.calc_polar_map(qx, qy, qz, mask, qmin, qmax, N_q, N_azi)


def circular_average(
    i: np.ndarray, e: np.ndarray, 
    map_q: np.ndarray, map_azi: np.ndarray, 
    density: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    assert i.dtype == np.float64
    assert e.dtype == np.float64
    assert map_q.dtype == np.int64
    assert map_azi.dtype == np.int64
    assert density.dtype == np.int64

    return _cpolarize.circular_average(i, e, map_q, map_azi, density)


if __name__ == "__main__":
    pass