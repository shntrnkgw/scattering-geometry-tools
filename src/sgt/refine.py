# coding="utf-8"
from typing import List, Tuple, Callable, Any
import sgt
from sgt import _crefine
import numpy as np
import scipy.optimize

def score_by_rings(
    geom: sgt.geometry, 
    i: np.ndarray, 
    qtargets: List[float], 
    qlimwidths: List[float], 
    qfuncwidths: List[float], 
    verbose: bool = False
    ) -> float:
    """Evaluates the score of the geometry. 
    """

    ret: float = 0.0
    qtarget: float = 0.0
    qlimwidth: float = 0.0
    qfuncwidth: float = 0.0
    qmin: float = 0.0
    qmax: float = 0.0
    
    # for every ring, 
    for qtarget, qlimwidth, qfuncwidth in zip(qtargets, qlimwidths, qfuncwidths):
        
        # q limits of the ring
        qmin = qtarget - qlimwidth
        qmax = qtarget + qlimwidth

        # calculate the score and add it to ret
        ret = ret + _crefine.proximity(
            i.astype(np.float64), 
            geom.qx, geom.qy, geom.qz, 
            geom.solid_angle_factor, 
            geom.mask, qtarget, qmin, qmax, qfuncwidth)

    # return 1.0/ret
    if verbose:
        print(-ret)

    return -ret

def refine(
    geom: sgt.geometry, 
    score_func: Callable[..., float], 
    score_args: Tuple[Any, ...], 
    refine_specnames: Tuple[str, ...], 
    score_requires_polarmap: bool = False, 
    optimize_method: str = "Nelder-Mead", 
    callback: Callable[[int, float], None]|None = None
    ) -> None:
    """Refines a geometry. 

    The parameters in ``specs`` of ``geom``
    are optimized so that ``score_func`` is minimized. 

    The ``score_func`` should be a callable that takes 
    a ``geometry`` object as the first argument. 
    It will be called as
    ```python
    score_func(geom, *score_args)
    ```
    during the optimization. 

    ``geom`` will be updated on every iteration of optimization. 
    The names of the parameters in ``specs`` to be optimized 
    must be specified by ``refine_specnames``. 

    Args:
        geom: A geometry object to be optimized. 
        score_func: A scoring function with the call signature 
            ``score_func(geom, *score_args)``. 
        score_args: Args for ``score_func``. 
        refine_specnames: Names of the parameters to be optimized. 
        score_requires_polarmap: If True, ``geom.refresh_polarmap()`` 
            will be called on every iteration. 
            If False, only ``geom.refresh_q()`` will be called. 
        optimize_method: Optimization method to be passed to 
            ``scipy.optimize.minimize``. 
        callback: A callback function to be called 
    
    """

    ncalls: int = 0
    score: float = 0.0

    # initial values of the parameters to be optimized
    p0: List[Any] = [geom.specs[k] for k in refine_specnames]

    # define scoring function that takes the parameters to be optimized as the args
    def _get_score(params: List[Any]) -> float:
        nonlocal ncalls

        # update the geometry
        geom.specs.update({k: v for k, v in zip(refine_specnames, params)})
        geom.refresh_q()
        if score_requires_polarmap:
            geom.refresh_polar_map()

        score = score_func(geom, *score_args)
        ncalls = ncalls + 1
        callback(ncalls, score)

        return score

    res = scipy.optimize.minimize(_get_score, p0, method=optimize_method)

if __name__ == "__main__":
    pass