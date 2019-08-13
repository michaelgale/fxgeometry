# Sample Test passing with nose and pytest

# system modules
import math, os.path
import sys
import pytest
from math import pi
# my modules

from fxgeometry.fxgeometry import *


def test_radpoint():
    a = RadialPoint(3, 1, 0)
    x, y = a.mid_xy()
    v0 = Vector(x, y, 0.0)
    v1 = Vector(0.0, 0.0, 0.0)
    assert v0.almost_same_as(v1)
    xi, yi = a.inner_xy()
    v0 = Vector(xi, yi, 0.0)
    v1 = Vector(-0.5, 0.0, 0.0)
    assert v0.almost_same_as(v1)
    xo, yo = a.outer_xy()
    v0 = Vector(xo, yo, 0.0)
    v1 = Vector(0.5, 0.0, 0.0)
    assert v0.almost_same_as(v1)
    x, y, z = a.outer_3d()
    v0 = Vector(x, y, z)
    assert v0.almost_same_as(v1)

def test_radpoint_angle():
    a = RadialPoint(3, 1, 45)
    x, y = a.mid_xy()
    v0 = Vector(x, y, 0.0)
    v1 = Vector(-0.879, 2.121, 0.0)
    assert v0.almost_same_as(v1)
    xi, yi = a.inner_xy()
    v0 = Vector(xi, yi, 0.0)
    v1 = Vector(-1.232, 1.768, 0.0)
    assert v0.almost_same_as(v1)
    xo, yo = a.outer_xy()
    v0 = Vector(xo, yo, 0.0)
    v1 = Vector(-0.525, 2.475, 0.0)
    assert v0.almost_same_as(v1)
    assert a.angle() == -45

def test_radpoint_angleneg():
    a = RadialPoint(3, 1, -5)
    x, y = a.mid_xy()
    v0 = Vector(x, y, 0.0)
    v1 = Vector(-0.011, -0.261, 0.0)
    assert v0.almost_same_as(v1)
    xi, yi = a.inner_xy()
    v0 = Vector(xi, yi, 0.0)
    v1 = Vector(-0.510, -0.218, 0.0)
    assert v0.almost_same_as(v1)
    xo, yo = a.outer_xy()
    v0 = Vector(xo, yo, 0.0)
    v1 = Vector(0.487, -0.305, 0.0)
    assert v0.almost_same_as(v1)
    assert a.angle() == 5

def test_radpoint_offset():
    a = RadialPoint(25, 4, 10)
    x, y = a.mid_xy()
    v0 = Vector(x, y, 0.0)
    v1 = Vector(-0.380, 4.341, 0.0)
    assert v0.almost_same_as(v1)
    a.lin_offset = 1.0
    x, y = a.mid_xy()
    v0 = Vector(x, y, 0.0)
    v1 = Vector(-0.553, 5.326, 0.0)
    assert v0.almost_same_as(v1)
    a.lin_offset = -1.5
    x, y = a.mid_xy()
    v0 = Vector(x, y, 0.0)
    v1 = Vector(-0.119, 2.864, 0.0)
    assert v0.almost_same_as(v1)
