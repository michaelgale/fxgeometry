# Sample Test passing with nose and pytest

# system modules
import math, os.path
import sys
import pytest
from math import pi
# my modules

from fxgeometry.fxgeometry import *


def test_point_length():
    a = Point(3, 4)
    assert a.length() == 5

def test_point_func():
    a = Point(5, 8)
    (bx, by) = a.swapped()
    assert a.x == by
    assert a.y == bx
    a.move_to(-3, 2)
    assert a.x == -3
    assert a.y == 2

def test_point_rotate():
    a = Point(1, 0)
    b = a.rotate(pi/2.0)
    b.integerize()
    assert b.x == 0
    assert b.y == 1

def test_grid_2d():
    pts = grid_points_2d(10, 20, 3)
    assert len(pts) == 9
    assert pts[0] == (-5, -10)
    assert pts[1] == (-5, 0)
    assert pts[2] == (-5, 10)
    assert pts[3] == (0, -10)
    assert pts[4] == (0, 0)
    assert pts[5] == (0, 10)
    assert pts[6] == (5, -10)
    assert pts[7] == (5, 0)
    assert pts[8] == (5, 10)
    pts = grid_points_2d(12, -1, 3, 1)
    assert len(pts) == 3
    assert pts[0] == (-6, -1)
    assert pts[1] == (0, -1)
    assert pts[2] == (6, -1)
    pts = grid_points_at_height(20, 4, 3, 3, 2)
    assert len(pts) == 6
    assert pts[0] == (-10, -2, 3)
    assert pts[1] == (-10, 2, 3)
    assert pts[2] == (0, -2, 3)
    assert pts[3] == (0, 2, 3)
    assert pts[4] == (10, -2, 3)
    assert pts[5] == (10, 2, 3)
