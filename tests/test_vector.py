# Sample Test passing with nose and pytest

# system modules
import math, os.path
import sys
import pytest

# my modules

from fxgeometry.fxgeometry import *


def test_vector_add():
    a = Vector(1, 2, 3)
    b = Vector(4, 5, 6)
    c = a + b
    assert c.x == 5
    assert c.y == 7
    assert c.z == 9

def test_vector_sub():
    a = Vector(9, 8, 7)
    b = Vector(4, 5, 6)
    c = a - b
    assert 5 == c.x
    assert 3 == c.y
    assert 1 == c.z

def test_equality():
    a = Vector(1.0, 2.5, -5.2)
    b = Vector(1.0, 2.5, -5.2)
    assert a == b
    c = Vector(1.02, 2.5, -5.2)
    assert a.almost_same_as(c, 0.1)
    assert a.almost_same_as(c, 1e-3) == False
