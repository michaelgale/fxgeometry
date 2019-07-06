# Sample Test passing with nose and pytest

# system modules
import math, os.path
import sys
import pytest
from math import pi
# my modules

from fxgeometry.fxgeometry import *


def test_rect_size():
    a = Rect(10, 4)
    assert a.left == -5
    assert a.right == 5
    assert a.top == 2
    assert a.bottom == -2
    a.bottom_up = True
    a.move_to(Point(0, 0))
    assert a.top == -2
    assert a.bottom == 2

def test_contains():
    a = Rect(5, 4)
    b = Point(1, 1.5)
    c = Point(-3, 10)
    assert a.contains(b)
    assert a.contains(c) == False

def test_overlap():
    a = Rect(10, 5)
    b = Rect(2, 3)
    c = copy.copy(b)
    c.move_to(Point(10, -7))
    assert a.overlaps(c) == False

def test_move():
    a = Rect(4, 8)
    a.move_top_left_to(Point(-10, -7))
    assert a.left == -10
    assert a.top == -7
    assert a.right == -6
    assert a.bottom == -15
    (x, y) = a.get_centre()
    assert x == -8
    assert y == -11
