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
