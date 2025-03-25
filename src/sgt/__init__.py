# coding="utf-8"
"""Higher-level interface for manipulation of scattering geometry. 
"""

import numpy as np
import json
from sgt import core
from numpy.typing import NDArray
import io
import copy

# for easy access
from sgt.core import Axis

__version__ = "1.0.0"

class PlanarRectangularDetector:
    """Calculates coordinates for a planar rectangular detector. 

    This class is designed to be immutable. 
    All calculations are done upon instantiation 
    based on the parameters given to the initializer. 
    If any of the parameters need to be modified, 
    create a new instance. 

    Attributes:
        u: Horizontal (u) coordinates of each pixel in the detector coordinate system
        v: Vertical (v) coordinates of each pixel in the detector coordinate system
        eu: Basis vector along the u axis in the lab coordinate system
        ev: Basis vector along the v axis in the lab coordinate system
        en: Basis vector along the n axis in the lab coordinate system
        x: x coordinates of each pixel in the lab coordinate system
        y: y coordinates of each pixel in the lab coordinate system
        z: z coordinates of each pixel in the lab coordinate system
        xyz_center: Coordinate of the detector center point in the lab coordinate system
        shortest_distance: The shortest distance from the sample to the detector plane
        pixel_solid_angle_coverage_correction_factors: correction factors for each pixel accounting for pixel solid angle coverage
    """

    def __init__(self, px_numbers: tuple[int,int], 
                       px_sizes: tuple[float,float], 
                       center_offset: tuple[float,float], 
                       rotation_matrix: NDArray[np.float64], 
                       sample_to_detector_distance: float) -> None:
        """Initializer

        Args:
            px_numbers: Numbers of pixels along the horizontal and vertical axes (hor, ver)
            px_sizes: Size of a single pixel along the horizontal and vertical axes (hor, ver)
            center_coord: (x, y) coordinate of the image center, 
                measured from the center of a pixel at index `[0,0]`
            rotation_matrix: Rotation matrix that defines the orientation of the detector
            sample_to_detector_distance: Sample to detector distance
        
        Note:
            `px_sizes`, `center_coord`, and `sample_to_detector_distance` 
            should be of the same unit (typically mm). 
        """
        
        self.px_numbers: tuple[int, int] = copy.deepcopy(px_numbers)
        self.px_sizes: tuple[float, float] = copy.deepcopy(px_sizes)
        self.center_offset: tuple[float, float] = copy.deepcopy(center_offset)
        self.rotation_matrix: NDArray[np.float64] = np.copy(rotation_matrix)
        self.sample_to_detector_distance: float = copy.copy(sample_to_detector_distance)

        # Pixel coordinates in the detector system (2D)
        self.u: NDArray[np.float64]
        self.v: NDArray[np.float64]
        self.u, self.v = core.make_coords_in_detector_system(self.px_numbers, 
                                                             self.px_sizes, 
                                                             self.center_offset)

        # Detector basis vectors in the lab system
        self.eu: NDArray[np.float64]
        self.ev: NDArray[np.float64]
        self.en: NDArray[np.float64]
        self.eu, self.ev, self.en = \
            core.make_detector_basis_vectors_in_lab_system(self.rotation_matrix)
        
        # Pixel coordinates in the lab system
        self.x: NDArray[np.float64]
        self.y: NDArray[np.float64]
        self.z: NDArray[np.float64]
        self.x, self.y, self.z = \
            core.transform_detector_to_lab(self.u, self.v, 
                                           self.eu, self.ev, self.sample_to_detector_distance)
        
        # Center coordinate in the lab system
        xc: NDArray[np.float64]
        yc: NDArray[np.float64]
        zc: NDArray[np.float64]
        xc, yc, zc = \
            core.transform_detector_to_lab(np.array([0.0]), np.array([0.0]), 
                                           self.eu, self.ev, self.sample_to_detector_distance)
        self.xyz_center: NDArray[np.float64] = np.array([xc[0], yc[0], zc[0]])

        # Shortest distance from the sample to the detector plane
        self.shortest_distance: float = \
            core.calc_shortest_distance_to_detector(self.eu, self.ev, self.en, self.sample_to_detector_distance)

        # Pixel solid angle coverage correction factors
        self.pixel_solid_angle_coverage_correction_factors: NDArray[np.float64] = \
            core.make_solid_angle_coverage_correction_factors(self.x, self.y, self.z, self.shortest_distance)

class ReciprocalGeometry:
    """Calculates q vector. 
    
    This class is designed to be immutable. 
    All calculations are done upon instantiation 
    based on the parameters given to the initializer. 
    If any of the parameters need to be modified, 
    create a new instance. 

    Attributes:
        qx: x component of the scattering vector q
        qy: y component of the scattering vector q
        qz: z component of the scattering vector q
        qabs: absolute value of the scattering vector q
        azimuthal_angle_rad: Azimuthal angle in radians (= atan(qy/qx))
        wavelength: wavelength of the incident beam
    """

    def __init__(self, x: NDArray[np.float64], 
                       y: NDArray[np.float64], 
                       z: NDArray[np.float64], 
                       wavelength: float) -> None:
        """Initializer

        Args:
            x: x coordinates of each pixel in the lab coordinate system
            y: y coordinates of each pixel in the lab coordinate system
            z: z coordinates of each pixel in the lab coordinate system
            wavelength: wavelength of the incident beam
        """
    
        self.qx: NDArray[np.float64]
        self.qy: NDArray[np.float64]
        self.qz: NDArray[np.float64]
        self.wavelength: float = copy.copy(wavelength)

        self.qx, self.qy, self.qz = core.make_q_vector(x, y, z, wavelength)
        self.qabs: NDArray[np.float64] = core.make_qabs(self.qx, self.qy, self.qz)
        self.azimuthal_angle_rad: NDArray[np.float64] = core.make_azimuthal_angle_rad(self.qx, self.qy)

class PixelSorter:
    """Circular averager
    
    Attributes:
        bin_edges: Edges of each bin 
        binning_coords: Coordinates (positions) used for binning
        masked_pixels: Truth array or index array that specifies the pixels to mask
        bin_midpoints: Midpoints of bin edges
        indices_in_bin: Binning index array
        bin_counts: Pixel counts in each bin
    """

    def __init__(self, bin_edges: NDArray[np.float64], 
                       binning_coords: NDArray[np.float64], 
                       masked_pixels: NDArray[any]) -> None:
        """Initializer

        Args:
            bin_edges: Edges of each bin. Must be sorted in an ascending order. 
            binning_coords: Coordinates (positions) used for binning (usually qabs)
            masked_pixels: Truth array or index array that specifies the pixels to mask
        """
        
        self.bin_edges: NDArray[np.float64] = np.copy(bin_edges)
        self.binning_coords: NDArray[np.float64] = np.copy(binning_coords)
        self.masked_pixels: NDArray[any] = np.copy(masked_pixels)

        self.bin_midpoints: NDArray[np.float64] = (bin_edges[:-1] + bin_edges[1:])/2.0

        self.indices_in_bin: NDArray[np.intp]
        self.bin_counts: NDArray[np.intp]
        self.indices_in_bin, self.bin_counts = core.make_bin_indices(self.bin_edges, 
                                                                     self.binning_coords, 
                                                                     self.masked_pixels)
        
    def bin(self, intensities: NDArray[np.float64], 
                  errors: NDArray[np.float64]) -> tuple[NDArray[np.float64], 
                                                        NDArray[np.float64]]:
        """Bin intensity & error using binning indices

        Args:
            intensities: Intensity array
            errors: Errors of intensity. Can be zero-filled array in case the error output is not necessary
        
        Returns:
            Binned arrays of intensity and error
        """
        
        return core.binning(self.indices_in_bin, 
                            self.bin_counts, 
                            intensities, 
                            errors)

REQUIRED_KEYS_GEOMETRY = [
    "type",
    "width_px",
    "height_px",
    "px_width_mm",
    "px_height_mm",
    "u0_mm",
    "v0_mm",
    "L0_mm",
    "lam_ang",
    "version"
]
ROTATION_TYPE_EULER_XYX = "euler_xyx"
ROTATION_TYPE_AROUND_AXES = "around_axes"

def load_geometry_from_dict(d: dict) -> tuple[PlanarRectangularDetector,ReciprocalGeometry]:

    for k in REQUIRED_KEYS_GEOMETRY:
        assert k in d.keys()
    
    assert d["version"] == "1.0.0"

    assert d["type"] == "PlanarRectangularDetector"

    # make rotation matrix
    R: NDArray[np.float64]
    if "rotation" not in d.keys():
        R = np.eye(3, dtype=np.float64)
    elif d["rotation"]["type"] == ROTATION_TYPE_EULER_XYX:
        R = core.make_rotation_matrix_euler_xyx(d["rotation"]["alpha_deg"], 
                                                d["rotation"]["beta_deg"], 
                                                d["rotation"]["gamma_deg"])
    elif d["rotation"]["type"] == ROTATION_TYPE_AROUND_AXES:
        R = np.eye(3, dtype=np.float64)
        ax: Axis
        # rotation matrix formed from multiple rotations
        for rotspec in d["rotation"]["rotations"]:
            match rotspec["axis"]:
                case "X": ax = Axis.X
                case "Y": ax = Axis.Y
                case "Z": ax = Axis.Z
            Radd: NDArray[np.float64] = core.make_rotation_matrix_around_lab_axis(ax, rotspec["angle_deg"])
            R = np.matmul(Radd, R)

    # detector
    det = PlanarRectangularDetector((d["width_px"], d["height_px"]), 
                                    (d["px_width_mm"], d["px_height_mm"]),
                                    (d["u0_mm"], d["v0_mm"]),
                                    R, 
                                    d["L0_mm"])
    
    # scattering geometry
    sc = ReciprocalGeometry(det.x, det.y, det.z, d["lam_ang"])

    return  det, sc


from sgt.legacy_geometry import geometry

if __name__ == "__main__":
    pass