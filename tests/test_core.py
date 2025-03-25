# coding="utf-8"

import pytest
from sgt import core
import numpy as np

def test_make_rotation_matrix():

    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])

    Rx = core.make_rotation_matrix_around_lab_axis(core.Axis.X, 90.0)
    Ry = core.make_rotation_matrix_around_lab_axis(core.Axis.Y, 90.0)
    Rz = core.make_rotation_matrix_around_lab_axis(core.Axis.Z, 90.0)

    assert np.all(np.isclose(np.matmul(Rx, ex), np.array([ 1.0, 0.0, 0.0]))) # Rx ex -> no change. (1 0 0)")
    assert np.all(np.isclose(np.matmul(Rx, ey), np.array([ 0.0, 0.0, 1.0]))) # Rx ey -> y to z. (0 0 1)")
    assert np.all(np.isclose(np.matmul(Rx, ez), np.array([ 0.0,-1.0, 0.0]))) # Rx ez -> z to -y.  (0 -1 0)")

    assert np.all(np.isclose(np.matmul(Ry, ex), np.array([ 0.0, 0.0,-1.0]))) # Ry ex -> z to x, x to -z. (0 0 -1)")
    assert np.all(np.isclose(np.matmul(Ry, ey), np.array([ 0.0, 1.0, 0.0]))) # Ry ey -> no change. (0 1 0)")
    assert np.all(np.isclose(np.matmul(Ry, ez), np.array([ 1.0, 0.0, 0.0]))) # Ry ez -> z to x. (1 0 0)")

    assert np.all(np.isclose(np.matmul(Rz, ex), np.array([ 0.0, 1.0, 0.0]))) # Rz ex -> x to y. (0 1 0)")
    assert np.all(np.isclose(np.matmul(Rz, ey), np.array([-1.0, 0.0, 0.0]))) # Rz ey -> x to y, y to -x. (-1 0 0)")
    assert np.all(np.isclose(np.matmul(Rz, ez), np.array([ 0.0, 0.0, 1.0]))) # Rz ez -> no change. (0 0 1)")

def test_make_detector_basis_vectors_in_lab_system():

    angle_deg = 30.0
    angle_rad = np.deg2rad(angle_deg)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)

    Rx = core.make_rotation_matrix_around_lab_axis(core.Axis.X, angle_deg)
    eu, ev, en = core.make_detector_basis_vectors_in_lab_system(Rx)

    assert np.all(np.isclose(eu, np.array([1.0, 0.0, 0.0]))) # eu = (1 0 0), rot. y to z. no change. (1 0 0)
    assert np.all(np.isclose(ev, np.array([0.0, cos, sin]))) # ev = (0 1 0), rot. y to z. (0 cos sin)
    assert np.all(np.isclose(en, np.array([0.0,-sin, cos]))) # en = (0 0 1), rot. z to -y. (0 -sin cos)

    Ry = core.make_rotation_matrix_around_lab_axis(core.Axis.Y, angle_deg)
    eu, ev, en = core.make_detector_basis_vectors_in_lab_system(Ry)

    assert np.all(np.isclose(eu, np.array([cos, 0.0,-sin]))) # eu = (1 0 0), rot. x to -z. (cos 0 -sin)
    assert np.all(np.isclose(ev, np.array([0.0, 1.0, 0.0]))) # ev = (0 1 0), rot. z to x. no change. 
    assert np.all(np.isclose(en, np.array([sin, 0.0, cos]))) # en = (0 0 1), rot. z to x. (sin 0 cos)

    Rz = core.make_rotation_matrix_around_lab_axis(core.Axis.Z, angle_deg)
    eu, ev, en = core.make_detector_basis_vectors_in_lab_system(Rz)

    assert np.all(np.isclose(eu, np.array([cos, sin, 0.0]))) # eu = (1 0 0), rot. x to y. (cos sin 0)
    assert np.all(np.isclose(ev, np.array([-sin, cos, 0.0]))) # ev = (0 1 0), rot. y to -x. (-sin cos 0)
    assert np.all(np.isclose(en, np.array([0.0, 0.0, 1.0]))) # en = (0 0 1), rot. x to y. no change