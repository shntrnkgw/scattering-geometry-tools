# coding="utf-8"

import io
import json
import numpy as np
from numpy.typing import NDArray
from sgt import core, PlanarRectangularDetector, ReciprocalGeometry, PixelSorter, __version__

class geometry:
    """Interface class for scattering geometry manipulation. 

    Note:
        This is a deprecated class and may not be maintained in the future. 

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

    minimal_spec_keys: tuple[str] = \
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
                 mask: NDArray[np.int8]|None=None) -> None:

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

        self._detector: PlanarRectangularDetector
        self._scattering_geometry: ReciprocalGeometry
        self._binners: list[PixelSorter]

        self._ax_azi: NDArray[np.float64]

        if mask is None:
            self.mask: NDArray[np.int8] = core.make_default_mask(width_px, height_px)
        else:
            self.mask: NDArray[np.int8] = mask

        self.refresh_q()
        self.refresh_polar_map()

    def refresh_q(self) -> None:
        """
        
        Note:
            ``self.mask`` is not used in this method. 
        """

        R: NDArray[np.float64] = \
            core.make_rotation_matrix_euler_xyx(self.specs["alpha_deg"], 
                                                self.specs["beta_deg"], 
                                                self.specs["gamma_deg"])

        self._detector = \
            PlanarRectangularDetector((self.specs["width_px"],    self.specs["height_px"]), 
                                      (self.specs["px_width_mm"], self.specs["px_height_mm"]), 
                                      (self.specs["u0_mm"],       self.specs["v0_mm"]), 
                                      R, 
                                      self.specs["L0_mm"])
        
        self._scattering_geometry = \
            ReciprocalGeometry(self._detector.x, 
                               self._detector.y, 
                               self._detector.z, 
                               self.specs["lam_ang"])

    def _is_ready_for_polar_map(self) -> bool:
        
        # check mask size
        if self.mask.shape[1] == self._detector.px_numbers[0] and self.mask.shape[0] == self._detector.px_numbers[1]:
            return True
        else:
            return False

    def refresh_polar_map(self) -> None:
        """
        
        Note: 
            This method uses ``self.mask``. 
        """

        azi_deg = np.rad2deg(self._scattering_geometry.azimuthal_angle_rad)

        # convert limit of azi to [0, 360)
        inds = self._scattering_geometry.qx < 0.0
        azi_deg[inds] = azi_deg[inds] + 180.0

        inds = self._scattering_geometry.qy < 0.0
        azi_deg[inds] = azi_deg[inds] + 360.0

        bin_edges_azi = np.linspace(0.0, 360.0, self.specs["azi_number"] + 1, endpoint=True, dtype=np.float64)

        self._ax_azi = (bin_edges_azi[:-1] + bin_edges_azi[1:])/2.0

        self._binners = []
        for i in range(self.specs["azi_number"]):
            mask_add = np.logical_or(azi_deg <= bin_edges_azi[i], bin_edges_azi[i+1] < azi_deg)
            ma = np.logical_or(self.mask == 1, mask_add)
            binner = \
                PixelSorter(np.linspace(self.specs["qmin_anginv"], self.specs["qmax_anginv"], self.specs["q_number"]+1, endpoint=True), 
                                   self._scattering_geometry.qabs, 
                                   ma)
            self._binners.append(binner)

    def load_specs(self, fp: str|io.FileIO|io.BytesIO|io.StringIO) -> None:
        """Loads and applies parameters from a file
        
        It loads parameters from a geometry specification file, 
        which is a JSON-formatted text file. 
        The file must contain all keys listed in ``minimal_spec_keys``. 

        Args:
            fp: file-like or path to a JSON-formatted file. 
        """

        h: dict = {}
        missing_keys: list[str] = []

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
    
    def save_specs(self, fp: str|io.FileIO|io.BytesIO|io.StringIO) -> None:
        h: dict = {k: v for k, v in self.specs.items()}
        h["version"] = __version__

        hstr: str = json.dumps(h, indent=4, ensure_ascii=False)

        if isinstance(fp, str):
            with open(fp, "w") as f:
                f.write(hstr)
        else:
            fp.write(hstr)
    
    def circular_average(self, 
                         intensity: NDArray[np.float64], 
                         e_intensity: NDArray[np.float64]) -> tuple[NDArray[np.float64], 
                                                                    NDArray[np.float64]]:
        """Performs circular averaging. 

        Args:
            intensity: intensity array. 
            e_intensity: intensity error array. 
        
        Returns:
            Two 2D numpy arrays of the intensity and the error, 
            both with the shape (``azi_number``, ``q_number``). 
        """

        ibs = []
        eibs = []

        for binner in self._binners:
            ib, eib = binner.bin(intensity, e_intensity)

            ibs.append(ib)
            eibs.append(eib)
        
        return np.array(ibs), np.array(eibs)

    def get_qabs(self) -> NDArray[np.float64]:
        return self._scattering_geometry.qabs

    @property
    def R(self) -> NDArray[np.float64]: return self._detector.rotation_matrix

    @property
    def u(self) -> NDArray[np.float64]: return self._detector.u

    @property
    def v(self) -> NDArray[np.float64]: return self._detector.v

    @property
    def x(self) -> NDArray[np.float64]: return self._detector.x

    @property
    def y(self) -> NDArray[np.float64]: return self._detector.y

    @property
    def z(self) -> NDArray[np.float64]: return self._detector.z

    @property
    def qx(self) -> NDArray[np.float64]: return self._scattering_geometry.qx

    @property
    def qy(self) -> NDArray[np.float64]: return self._scattering_geometry.qy

    @property
    def qz(self) -> NDArray[np.float64]: return self._scattering_geometry.qz

    @property
    def solid_angle_factor(self) -> NDArray[np.float64]: return self._detector.pixel_solid_angle_coverage_correction_factors

    @property
    def map_q(self) -> None:
        raise NotImplementedError

    @property
    def map_azi(self) -> None:
        raise NotImplementedError

    @property
    def density(self) -> None:
        raise NotImplementedError

    @property
    def ax_q(self) -> NDArray[np.float64]: return self._binners[0].bin_midpoints

    @property
    def ax_azi(self) -> NDArray[np.float64]: return self._ax_azi

    @property
    def normal_incidence_dist(self) -> float: return self._detector.shortest_distance

if __name__ == "__main__":
    pass