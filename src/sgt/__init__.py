# coding="utf-8"
"""Higher-level interface for manipulation of scattering geometry. 

"""

from typing import List, IO, Tuple
import numpy as np
import json
from sgt import core, _cpolarize

__version__ = "0.1.2"

class geometry(object):
    """Interface class for scattering geometry manipulation. 

    Example:
        To create a `geometry` instance, 
        
            >>> import sgt
            >>> g = sgt.geometry()
        
        Although all necessary parameters may be supplied 
        as the args of the initializer, 
        the easiest way is to load the specs from a file. 

            >>> g.load_specs("Geometry_AgBh.txt")

        Make sure to call refresh functions 
        when you make changes to the geometry. 
        
            >>> g.refresh_q()
        
        Mask can be supplied later. 
        For example, suppose that we already have the mask array
        `maskarray`, 
        
            >>> g.mask = maskarray

        Make sure to refresh polar maps whenever you made
        any changes to the geometry. 
            
            >>> g.refresh_polar_map()

        Finally, it can perform circular averaging. 
        Suppose that we have the 2D intensity array `i`
        and associated error array `e_i`, 

            >>> i_av, e_i_av = g.circular_average(i, e_i)
        
        The output arrays are also 2D. 
        For example, ``i_av[k]`` is the q-profile at ``k`` th 
        azimuthal section. 
        The arrays along the q and azimuthal angle axes 
        are stored as ``g.ax_q`` and ``g.ax_azi``, 
        respectively. 
    """

    minimal_spec_keys: Tuple = \
        ("width_px", 
         "height_px", 
         "px_width_mm", 
         "px_height_mm", 
         "u0_mm", 
         "v0_mm",   
         "alpha_deg", 
         "beta_deg", 
         "gamma_deg", 
         "L0_mm", 
         "lam_ang", 
         "qmin_anginv", 
         "qmax_anginv", 
         "q_number", 
         "azi_number")

    def __init__(self, 
    width_px: int=1, height_px: int=1, 
    px_width_mm: float=1.0, px_height_mm: float=1.0, 
    u0_mm: float=0.0, v0_mm: float=0.0, 
    alpha_deg: float=0.0, beta_deg: float=0.0, gamma_deg: float=0.0, 
    L0_mm: float=1.0, lam_ang: float=1.0, 
    qmin: float=0.0, qmax: float=1.0, N_q: int=100, N_azi: int=1, 
    mask: np.ndarray|None=None) -> None:

        self.specs: dict = {}

        self.specs["width_px"]     = width_px
        self.specs["height_px"]    = height_px
        self.specs["px_width_mm"]  = px_width_mm
        self.specs["px_height_mm"] = px_height_mm
        self.specs["u0_mm"]        = u0_mm
        self.specs["v0_mm"]        = v0_mm 
        self.specs["alpha_deg"]    = alpha_deg 
        self.specs["beta_deg"]     = beta_deg 
        self.specs["gamma_deg"]    = gamma_deg 
        self.specs["L0_mm"]        = L0_mm 
        self.specs["lam_ang"]      = lam_ang
        self.specs["qmin_anginv"]  = qmin
        self.specs["qmax_anginv"]  = qmax
        self.specs["q_number"]     = N_q
        self.specs["azi_number"]   = N_azi 

        if mask is None:
            self.mask = core.make_default_mask(width_px, height_px)
        else:
            self.mask = mask

        # vars to be calculated
        self._R: np.ndarray = np.zeros((3,3))
        self._u: np.ndarray = np.empty((0,0))
        self._v: np.ndarray = np.empty((0,0))
        self._a: np.ndarray = np.zeros((3,))
        self._b: np.ndarray = np.zeros((3,))
        self._n: np.ndarray = np.zeros((3,))
        self._x: np.ndarray = np.empty((0,0))
        self._y: np.ndarray = np.empty((0,0))
        self._z: np.ndarray = np.empty((0,0))
        self._qx: np.ndarray = np.empty((0,0))
        self._qy: np.ndarray = np.empty((0,0))
        self._qz: np.ndarray = np.empty((0,0))
        self._solid_angle_factor: np.ndarray = np.empty((0,0))
        self._map_q:   np.ndarray = np.empty((0,0), dtype=int)
        self._map_azi: np.ndarray = np.empty((0,0), dtype=int)
        self._density: np.ndarray = np.empty((0,0), dtype=int)
        self._ax_q:    np.ndarray = np.empty((0,0))
        self._ax_azi:  np.ndarray = np.empty((0,0))
        self._normal_incidence_dist: float = 0.0

    def load_specs(self, fp: str|IO) -> None:
        """Loads and applies parameters from a file
        
        It loads parameters from a geometry specification file, 
        which is a JSON-formatted text file. 
        The file must contain all keys listed in ``minimal_spec_keys``. 

        Args:
            fp: file-like or path to a JSON-formatted file. 
        """

        h: dict = {}
        missing_keys: List[str] = []

        if isinstance(fp, str):
            with open(fp, "r") as f:
                lines = f.readlines()
        else:
            lines = fp.readlines()
        
        # for backward compatibility
        # older geometry file contains # at every line head
        if lines[0].startswith("#"):
            jsonfeed = "".join([l.lstrip("#") for l in lines])
        else:
            jsonfeed = "".join(lines)

        h = json.loads(jsonfeed)

        missing_keys = [k for k in self.minimal_spec_keys if k not in h]

        if missing_keys:
            raise ValueError("missing key(s): " + ", ".join(missing_keys))

        self.specs.update(h)
    
    def save_specs(self, fp: str|IO) -> None:
        h: dict = {k: v for k, v in self.specs.items()}
        h["version"] = __version__

        hstr: str = json.dumps(h, indent=4, ensure_ascii=False)

        if isinstance(fp, str):
            with open(fp, "w") as f:
                f.write(hstr)
        else:
            fp.write(hstr)

    def refresh_q(self) -> None:
        """
        
        Note:
            ``self.mask`` is not used in this method. 
        """
        self._R = core.make_rotation_matrix(
            self.specs["alpha_deg"], 
            self.specs["beta_deg"], 
            self.specs["gamma_deg"])

        self._u, self._v = core.make_pixel_coords_in_detector_system(
            self.specs["width_px"], self.specs["height_px"], 
            self.specs["px_width_mm"], self.specs["px_height_mm"], 
            self.specs["u0_mm"], self.specs["v0_mm"]
            )

        self._a, self._b, self._n = core.make_basis_vectors_on_detector_in_lab_system(self._R)

        self._x, self._y, self._z = core.make_pixel_coords_in_lab_system(
            self._u, self._v, 
            self._a, self._b, self._n, self.specs["L0_mm"]
            )

        self._normal_incidence_dist = core.calc_shortest_dist_to_detector(
            self._a, self._b, self._n, self.specs["L0_mm"]
            )

        self._solid_angle_factor = \
            core.make_solid_angle_coverage_correction_factors(
                self._x, self._y, self._z, self._normal_incidence_dist
            )

        self._qx, self._qy, self._qz = core.make_q(
            self._x, self._y, self._z, self.specs["lam_ang"]
            )

    def refresh_polar_map(self) -> None:
        """
        
        Note: 
            This method uses ``self.mask``. 
        """

        assert self._is_ready_for_polar_map() # check

        self._map_q, self._map_azi, self._density, self._ax_q, self._ax_azi = \
            _cpolarize.calc_polar_map(
                self._qx, self._qy, self._qz, 
                self.mask, 
                self.specs["qmin_anginv"], self.specs["qmax_anginv"], 
                self.specs["q_number"], self.specs["azi_number"]
            )
    
    def circular_average(self, intensity: np.ndarray, e_intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs circular averaging. 

        Args:
            intensity: intensity array. 
            e_intensity: intensity error array. 
        
        Returns:
            Two 2D numpy arrays of the intensity and the error, 
            both with the shape (``azi_number``, ``q_number``). 
        """

        return _cpolarize.circular_average(
            intensity.astype(np.float64), 
            e_intensity.astype(np.float64), 
            self._map_q, self._map_azi, self._density
            )
    
    def _is_ready_for_polar_map(self) -> bool:

        shape: Tuple[int, int] = (self.specs["height_px"], self.specs["width_px"])
        
        # check mask size
        if self.mask.shape != shape:
            return False
        
        return True

    def get_qabs(self) -> np.ndarray: 
        return np.sqrt(self.qx*self.qx + self.qy*self.qy + self.qz*self.qz)

    @property
    def R(self) -> np.ndarray: return self._R

    @property
    def u(self) -> np.ndarray: return self._u

    @property
    def v(self) -> np.ndarray: return self._v

    @property
    def x(self) -> np.ndarray: return self._x

    @property
    def y(self) -> np.ndarray: return self._y

    @property
    def z(self) -> np.ndarray: return self._z

    @property
    def qx(self) -> np.ndarray: return self._qx

    @property
    def qy(self) -> np.ndarray: return self._qy

    @property
    def qz(self) -> np.ndarray: return self._qz

    @property
    def solid_angle_factor(self) -> np.ndarray: return self._solid_angle_factor

    @property
    def map_q(self) -> np.ndarray: return self._map_q

    @property
    def map_azi(self) -> np.ndarray: return self._map_azi

    @property
    def density(self) -> np.ndarray: return self._density

    @property
    def ax_q(self) -> np.ndarray: return self._ax_q

    @property
    def ax_azi(self) -> np.ndarray: return self._ax_azi

    @property
    def normal_incidence_dist(self) -> float: return self._normal_incidence_dist

if __name__ == "__main__":
    pass