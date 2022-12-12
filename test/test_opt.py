# coding="utf-8"

import snpl
import snpl.image
import sgt
from sgt import pilatus, refine
import numpy as np

if __name__ == "__main__":
    Q_AgBh_1 = 2.0*np.pi/58.38

    mask = snpl.image.NpzImage("test data for refine/Mask_EmptyBeam.npz").get_layer("mask")
    geom = sgt.geometry()
    geom.load_specs("test data for refine/Geometry_AgBh.txt")
    geom.mask = mask
    geom.refresh_q()

    geom.specs["u0_mm"] = 73.0
    geom.specs["v0_mm"] = 95.0

    geom.refresh_q()

    q = geom.get_qabs()

    intensity = pilatus.ReadTiff("test data for refine/AgBh.tif").astype(np.float64)

    qtargets = [Q_AgBh_1*1.0, Q_AgBh_1*2.0, Q_AgBh_1*3.0]
    qfuncwidths = [0.005, 0.005, 0.005]
    qlimwidths = [v*4.0 for v in qfuncwidths]

    qfuncmins = [qt - qw for qt, qw in zip(qtargets, qfuncwidths)]
    qfuncmaxs = [qt + qw for qt, qw in zip(qtargets, qfuncwidths)]
    qlimmins = [qt - qw for qt, qw in zip(qtargets, qlimwidths)]
    qlimmaxs = [qt + qw for qt, qw in zip(qtargets, qlimwidths)]

    snpl.figsize(6,6)
    snpl.gca().set_aspect("equal")
    snpl.pyplot.pcolormesh(geom.u, geom.v, intensity, shading="nearest", vmin=0.0, vmax=10000)
    pmask = np.ones_like(mask, dtype=float)
    pmask[mask==0] = np.nan
    snpl.pyplot.pcolormesh(geom.u, geom.v, pmask, shading="nearest", vmin=0, vmax=1, cmap="Greys")

    snpl.pyplot.contour(geom.u, geom.v, q, qtargets, linewidths=0.2, colors="r", linestyles="-")
    snpl.pyplot.contour(geom.u, geom.v, q, qfuncmins, linewidths=0.2, colors="r", linestyles=":")
    snpl.pyplot.contour(geom.u, geom.v, q, qfuncmaxs, linewidths=0.2, colors="r", linestyles=":")
    snpl.pyplot.contour(geom.u, geom.v, q, qlimmins, linewidths=0.1, colors="r", linestyles="-")
    snpl.pyplot.contour(geom.u, geom.v, q, qlimmaxs, linewidths=0.1, colors="r", linestyles="-")


    snpl.axvline(0.0, color="w", lw=0.2)
    snpl.axhline(0.0, color="w", lw=0.2)

    snpl.savefig("test_opt_image.png", dpi=300)
    snpl.clf()


    def callback(ncalls, score):
        print("Calls = {0} Score = {1}".format(ncalls, score))

    refine.refine(geom, 
    refine.score_by_rings, 
    (intensity, qtargets, qlimwidths, qfuncwidths), 
    ("u0_mm", "v0_mm"), 
    callback=callback)

    geom.refresh_q()

    q = geom.get_qabs()

    snpl.gca().set_aspect("equal")
    snpl.pyplot.pcolormesh(geom.u, geom.v, intensity, shading="nearest", vmin=0.0, vmax=10000)
    pmask = np.ones_like(mask, dtype=float)
    pmask[mask==0] = np.nan
    snpl.pyplot.pcolormesh(geom.u, geom.v, pmask, shading="nearest", vmin=0, vmax=1, cmap="Greys")

    snpl.pyplot.contour(geom.u, geom.v, q, qtargets, linewidths=0.2, colors="r", linestyles="-")
    snpl.pyplot.contour(geom.u, geom.v, q, qfuncmins, linewidths=0.2, colors="r", linestyles=":")
    snpl.pyplot.contour(geom.u, geom.v, q, qfuncmaxs, linewidths=0.2, colors="r", linestyles=":")
    snpl.pyplot.contour(geom.u, geom.v, q, qlimmins, linewidths=0.1, colors="r", linestyles="-")
    snpl.pyplot.contour(geom.u, geom.v, q, qlimmaxs, linewidths=0.1, colors="r", linestyles="-")

    snpl.axvline(0.0, color="w", lw=0.2)
    snpl.axhline(0.0, color="w", lw=0.2)

    snpl.savefig("test_opt_image_after.png", dpi=300)