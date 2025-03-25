# coding="utf-8"

import pytest
from sgt import hdf5image, legacy_geometry, pilatus
import numpy as np
import os

def test_legacy_geometry_circular_average():

    mask = hdf5image.HDF5Image("tests/data/20240223_PF_BL6A/r1_2400mm_RSC_WAXS/common/Mask.hdf5").get_layer("mask")
    geom = legacy_geometry.geometry()
    geom.load_specs("tests/data/20240223_PF_BL6A/r1_2400mm_RSC_WAXS/common/Geometry.txt")
    geom.mask = mask
    geom.specs["azi_number"] = 36
    geom.refresh_q()
    geom.refresh_polar_map()

    i = pilatus.ReadTiff("tests/data/20240223_PF_BL6A/1_2400mm_RSC_WAXS/AgBh_2.tif").astype(np.float64)
    i[i < 0] = 0.0
    e = np.sqrt(i)

    ai, ae = geom.circular_average(i, e)

    # validation data
    vd = hdf5image.HDF5Image("tests/data/20240223_PF_BL6A/r1_2400mm_RSC_WAXS/common/AgBh_validation.hdf5")

    assert np.all(np.isclose(ai, vd.get_layer("i"), equal_nan=True))
    assert np.all(np.isclose(ae, vd.get_layer("e"), equal_nan=True))