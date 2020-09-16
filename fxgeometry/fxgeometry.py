
import copy
import math
from math import sin, cos, radians, sqrt, atan, degrees, atan2, hypot
from numbers import Number
from functools import reduce


class MatrixError(Exception):
    pass


class Axis(object):
    pass


class XAxis(Axis):
    pass


class YAxis(Axis):
    pass


class ZAxis(Axis):
    pass


class AngleUnits(object):
    pass


class Radians(AngleUnits):
    pass


class Degrees(AngleUnits):
    pass


def _rows_multiplication(r1, r2):
    rows = [
        [
            r1[0][0] * r2[0][0] + r1[0][1] * r2[1][0] + r1[0][2] * r2[2][0],
            r1[0][0] * r2[0][1] + r1[0][1] * r2[1][1] + r1[0][2] * r2[2][1],
            r1[0][0] * r2[0][2] + r1[0][1] * r2[1][2] + r1[0][2] * r2[2][2],
        ],
        [
            r1[1][0] * r2[0][0] + r1[1][1] * r2[1][0] + r1[1][2] * r2[2][0],
            r1[1][0] * r2[0][1] + r1[1][1] * r2[1][1] + r1[1][2] * r2[2][1],
            r1[1][0] * r2[0][2] + r1[1][1] * r2[1][2] + r1[1][2] * r2[2][2],
        ],
        [
            r1[2][0] * r2[0][0] + r1[2][1] * r2[1][0] + r1[2][2] * r2[2][0],
            r1[2][0] * r2[0][1] + r1[2][1] * r2[1][1] + r1[2][2] * r2[2][1],
            r1[2][0] * r2[0][2] + r1[2][1] * r2[1][2] + r1[2][2] * r2[2][2],
        ],
    ]
    return rows


class Matrix(object):
    """ a transformation matrix """

    def __init__(self, rows):
        self.rows = rows

    def __repr__(self):
        values = reduce(lambda x, y: x + y, self.rows)
        format_string = "((%f, %f, %f),\n" " (%f, %f, %f),\n" " (%f, %f, %f))"
        return format_string % tuple(values)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            r1 = self.rows
            r2 = other.rows
            return Matrix(_rows_multiplication(r1, r2))
        elif isinstance(other, Vector):
            r = self.rows
            x, y, z = other.x, other.y, other.z
            return Vector(
                r[0][0] * x + r[0][1] * y + r[0][2] * z,
                r[1][0] * x + r[1][1] * y + r[1][2] * z,
                r[2][0] * x + r[2][1] * y + r[2][2] * z,
            )
        else:
            raise MatrixError

    def __rmul__(self, other):
        if isinstance(other, Matrix):
            r1 = other.rows
            r2 = self.rows
            return Matrix(_rows_multiplication(r1, r2))
        elif isinstance(other, Vector):
            r = self.rows
            x, y, z = other.x, other.y, other.z
            return Vector(
                x * r[0][0] + y * r[1][0] + z * r[2][0],
                x * r[0][1] + y * r[1][1] + z * r[2][1],
                x * r[0][2] + y * r[1][2] + z * r[2][2],
            )
        else:
            raise MatrixError

    def copy(self):
        """ make a copy of this matrix """
        return Matrix(copy.deepcopy(self.rows))

    def rotate(self, angle, axis, units=Degrees):
        """ rotate the matrix by an angle around an axis """
        if units == Degrees:
            c = math.cos(angle / 180.0 * math.pi)
            s = math.sin(angle / 180.0 * math.pi)
        else:
            c = math.cos(angle)
            s = math.sin(angle)
        if axis == XAxis:
            rotation = Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == YAxis:
            rotation = Matrix([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        elif axis == ZAxis:
            rotation = Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise MatrixError("Invalid axis specified.")
        return self * rotation

    def scale(self, sx, sy, sz):
        """ scale the matrix by a number"""
        return Matrix([[sx, 0, 0], [0, sy, 0], [0, 0, sz]]) * self

    def transpose(self):
        """ transpose """
        r = self.rows
        return Matrix(
            [
                [r[0][0], r[1][0], r[2][0]],
                [r[0][1], r[1][1], r[2][1]],
                [r[0][2], r[1][2], r[2][2]],
            ]
        )

    def det(self):
        """ determinant of the matrix """
        r = self.rows
        terms = [
            r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]),
            r[0][1] * (r[1][2] * r[2][0] - r[1][0] * r[2][2]),
            r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]),
        ]
        return sum(terms)

    def flatten(self):
        """ flatten the matrix """
        return tuple(reduce(lambda x, y: x + y, self.rows))

    def fix_diagonal(self):
        """ Some applications do not like matrices with zero diagonal elements. """
        corrected = False
        for i in range(3):
            if self.rows[i][i] == 0.0:
                self.rows[i][i] = 0.001
                corrected = True
        return corrected

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        return self.rows == other.rows


def Identity():
    """ a transformation matrix representing Identity """
    return Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


class Vector(object):
    """ a Vector in 3D"""

    def __init__(self, x, y=None, z=None):
        if isinstance(x, tuple):
            self.x = x[0]
            self.y = x[1]
            self.z = x[2]
        elif y is not None and z is not None:
            self.x, self.y, self.z = x, y, z
        else:
            self.x = 0
            self.y = 0
            self.z = 0

    @property
    def repr(self):
        return "%f, %f, %f" % (self.x, self.y, self.z)

    def __repr__(self):
        return "<Vector: (%s)>" % (self.repr)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        # Return a new object.
        return Vector(x, y, z)

    __radd__ = __add__

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        # Return a new object.
        return Vector(x, y, z)

    def __rsub__(self, other):
        x = other.x - self.x
        y = other.y - self.y
        z = other.z - self.z
        # Return a new object.
        return Vector(x, y, z)

    def __cmp__(self, other):
        # This next expression will only return zero (equals) if all
        # expressions are false.
        return self.x != other.x or self.y != other.y or self.z != other.z

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Vector(self.x * other, self.y * other, self.z * other)
        raise ValueError("Cannot multiply %s with %s" % (self.__class__, type(other)))

    def __div__(self, other):
        if isinstance(other, Number):
            return Vector(self.x / other, self.y / other, self.z / other)
        raise ValueError("Cannot divide %s with %s" % (self.__class__, type(other)))

    def copy(self):
        """vector = copy(self)
        Copy the vector so that new vectors containing the same values
        are passed around rather than references to the same object.
        """
        return Vector(self.x, self.y, self.z)

    def cross(self, other):
        """ cross product """
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def dot(self, other):
        """ dot product"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self):
        """ normalized """
        _length = abs(self)
        self.x = self.x / _length
        self.y = self.y / _length
        self.z = self.z / _length

    def polar_xy(self, r_offset=0.0):
        r = ((self.x + r_offset) * (self.x + r_offset) + self.y * self.y) ** 0.5
        t = degrees(atan2(self.y, (self.x + r_offset)))
        return (r, t)

    def offset_xy(self, xo, yo):
        self.x += xo
        self.y += yo

    def polar_quad(self, r_offset=0.0):
        r, t = self.polar_xy(r_offset=0.0)
        if t > 0:
            if t > 90.0:
                return "TL"
            else:
                return "TR"
        else:
            if t < -90.0:
                return "BL"
            else:
                return "BR"

    def almost_same_as(self, other, tolerance=1e-3):
        if not isinstance(other, Vector):
            return False
        if abs(self.x - other.x) > tolerance:
            return False
        if abs(self.y - other.y) > tolerance:
            return False
        if abs(self.z - other.z) > tolerance:
            return False
        return True


class Vector2D(object):
    """ a Vector in 2D """

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        return "<Vector2D: (%f, %f) >" % (self.x, self.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        # Return a new object.
        return Vector2D(x, y)

    __radd__ = __add__

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        # Return a new object.
        return Vector2D(x, y)

    def __rsub__(self, other):
        x = other.x - self.x
        y = other.y - self.y
        # Return a new object.
        return Vector2D(x, y)

    def __cmp__(self, other):
        # This next expression will only return zero (equals) if all
        # expressions are false.
        return self.x != other.x or self.y != other.y

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Vector2D(self.x * other, self.y * other)
        raise ValueError("Cannot multiply %s with %s" % (self.__class__, type(other)))

    def __div__(self, other):
        if isinstance(other, Number):
            return Vector2D(self.x / other, self.y / other)
        raise ValueError("Cannot divide %s with %s" % (self.__class__, type(other)))

    def copy(self):
        """
        vector = copy(self)
        Copy the vector so that new vectors containing the same values
        are passed around rather than references to the same object.
        """
        return Vector2D(self.x, self.y)

    def dot(self, other):
        """ dot product """
        return self.x * other.x + self.y * other.y


class CoordinateSystem(object):
    def __init__(
        self, x=Vector(1.0, 0.0, 0.0), y=Vector(0.0, 1.0, 0.0), z=Vector(0.0, 0.0, 1.0)
    ):
        self.x = x
        self.y = y
        self.z = z

    def project(self, p):
        return Vector(p.dot(self.x), p.dot(self.y), p.dot(self.z))


class Point:
    def __init__(self, x=0.0, y=0.0):
        if isinstance(x, tuple):
            self.x = x[0]
            self.y = x[1]
        elif isinstance(x, list):
            if isinstance(x[0], tuple):
                self.x = x[0][0]
                self.y = x[0][1]
            else:
                self.x = x[0]
                self.y = x[1]
        else:
            self.x = x
            self.y = y

    def __add__(self, p):
        """Point(x1+x2, y1+y2)"""
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        """Point(x1-x2, y1-y2)"""
        return Point(self.x - p.x, self.y - p.y)

    def __mul__(self, scalar):
        """Point(x1*x2, y1*y2)"""
        return Point(self.x * scalar, self.y * scalar)

    def __div__(self, scalar):
        """Point(x1/x2, y1/y2)"""
        return Point(self.x / scalar, self.y / scalar)

    def __str__(self):
        if isinstance(self.x, float):
            return "(%.2f, %.2f)" % (self.x, self.y)
        else:
            return "(%s, %s)" % (self.x, self.y)

    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.x, self.y)

    def strspc(self):
        if isinstance(self.x, float):
            return "(%.3f %.3f)" % (self.x, self.y)
        else:
            return "(%s %s)" % (self.x, self.y)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def distance_to(self, p):
        """Calculate the distance between two points."""
        return (self - p).length()

    def as_tuple(self):
        """(x, y)"""
        return (self.x, self.y)

    def swapped(self):
        return (self.y, self.x)

    def clone(self):
        """Return a full copy of this point."""
        return Point(self.x, self.y)

    def integerize(self):
        """Convert co-ordinate values to integers."""
        self.x = int(self.x)
        self.y = int(self.y)

    def floatize(self):
        """Convert co-ordinate values to floats."""
        self.x = float(self.x)
        self.y = float(self.y)

    def move_to(self, x, y):
        """Reset x & y coordinates."""
        self.x = x
        self.y = y

    def slide(self, p):
        """Move to new (x+dx,y+dy).

        Can anyone think up a better name for this function?
        slide? shift? delta? move_by?
        """
        self.x = self.x + p.x
        self.y = self.y + p.y

    def slide_xy(self, dx, dy):
        """Move to new (x+dx,y+dy).

        Can anyone think up a better name for this function?
        slide? shift? delta? move_by?
        """
        self.x = self.x + dx
        self.y = self.y + dy

    def offset(self, xoffset=0.0, yoffset=None):
        if yoffset is not None:
            return (self.x + xoffset, self.y + yoffset)
        else:
            return (self.x + xoffset, self.y + xoffset)

    def mirror_y(self):
        self.y = -self.y

    def mirror_x(self):
        self.x = -self.x

    def rotate(self, rad):
        """Rotate counter-clockwise by rad radians.

        Positive y goes *up,* as in traditional mathematics.

        Interestingly, you can use this in y-down computer graphics, if
        you just remember that it turns clockwise, rather than
        counter-clockwise.

        The new position is returned as a new Point.
        """
        s, c = [f(rad) for f in (math.sin, math.cos)]
        x, y = (c * self.x - s * self.y, s * self.x + c * self.y)
        return Point(x, y)

    def rotate_about(self, p, theta):
        """Rotate counter-clockwise around a point, by theta degrees.

        Positive y goes *up,* as in traditional mathematics.

        The new position is returned as a new Point.
        """
        result = self.clone()
        result.slide(-p.x, -p.y)
        result.rotate(theta)
        result.slide(p.x, p.y)
        return result


class Size:
    """ Container class for 2D sizes """
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height

    def __str__(self):
        return "%s, %s" % (self.width, self.height)

    def swapped(self):
        return (self.height, self.width)


class Rect:
    """ 2D Rectangle class """

    def __init__(self, width=2.0, height=2.0, bottomUp=False):
        self.bottom_up = bottomUp
        self.left = -width / 2.0
        self.right = width / 2.0
        if bottomUp:
            self.top = -height / 2.0
            self.bottom = height / 2.0
        else:
            self.top = height / 2.0
            self.bottom = -height / 2.0
        self.width = abs(self.right - self.left)
        self.height = abs(self.top - self.bottom)


    def __str__(self):
        return "<Rect (%.2f,%.2f)-(%.2f,%.2f)>" % (self.left, self.top, self.right, self.bottom)

    def __repr__(self):
        return "%s(%r, %r)" % (
            self.__class__.__name__,
            Point(self.left, self.top),
            Point(self.right, self.bottom),
        )

    def copy(self):
        r = Rect(self.width, self.height)
        r.left, r.right = self.left, self.right
        r.top, r.bottom = self.top, self.bottom
        r.bottom_up = self.bottom_up
        return r

    def get_size(self):
        self.width = abs(self.right - self.left)
        self.height = abs(self.top - self.bottom)
        return self.width, self.height

    def get_centre(self):
        x = self.left + self.width / 2
        if self.bottom_up:
            y = self.top + self.height / 2
        else:
            y = self.top - self.height / 2
        return x, y

    def get_pts(self):
        return [ (self.left, self.top), (self.right, self.top),
        (self.left, self.bottom), (self.right, self.bottom)
        ]
    def get_pts_3d(self, height=0):
        return [ (self.left, self.top, height), (self.right, self.top, height),
        (self.left, self.bottom, height), (self.right, self.bottom, height)
        ]

    def move_to(self, pt, py=None):
        if isinstance(pt, Point):
            (x, y) = pt.as_tuple()
        elif isinstance(pt, tuple):
            x, y = pt[0], pt[1]
        else:
            x, y = pt, py
        self.left = x - self.width / 2
        self.right = x + self.width / 2
        if self.bottom_up:
            self.top = y - self.height / 2
            self.bottom = y + self.height / 2
        else:
            self.top = y + self.height / 2
            self.bottom = y - self.height / 2

    def get_top_left(self):
        return (self.left, self.top)

    def get_bottom_left(self):
        return (self.left, self.bottom)

    def get_top_right(self):
        return (self.right, self.top)

    def get_bottom_right(self):
        return (self.right, self.bottom)

    def get_anchor_pt(self, anchor_pt):
        if "left" in anchor_pt:
            x = self.left
        elif "right" in anchor_pt:
            x = self.right
        else:
            x = self.left + self.width/2
        if "top" in anchor_pt:
            y = self.top
        elif "bottom" in anchor_pt:
            y = self.bottom
        else:
            y = self.top - self.height/2
        return x, y
 
    def _xy_from_pt(self, pt):
        if isinstance(pt, Point):
            (x, y) = pt.as_tuple()
        else:
            x, y = pt[0], pt[1]
        return x, y

    def move_top_left_to(self, pt):
        x, y = self._xy_from_pt(pt)
        self.left = x
        self.right = x + self.width
        self.top = y
        if self.bottom_up:
            self.bottom = y + self.height
        else:
            self.bottom = y - self.height

    def move_top_right_to(self, pt):
        x, y = self._xy_from_pt(pt)
        self.right = x
        self.left = x - self.width
        self.top = y
        if self.bottom_up:
            self.bottom = y + self.height
        else:
            self.bottom = y - self.height

    def move_bottom_left_to(self, pt):
        x, y = self._xy_from_pt(pt)
        self.left = x
        self.right = x + self.width
        self.bottom = y
        if self.bottom_up:
            self.top = y - self.height
        else:
            self.top = y + self.height

    def move_bottom_right_to(self, pt):
        x, y = self._xy_from_pt(pt)
        self.right = x
        self.left = x - self.width
        self.bottom = y
        if self.bottom_up:
            self.top = y - self.height
        else:
            self.top = y + self.height

    def set_points(self, pt1, pt2):
        """Reset the rectangle coordinates."""
        x1, y1 = self._xy_from_pt(pt1)
        x2, y2 = self._xy_from_pt(pt2)
        self.left = min(x1, x2)
        self.right = max(x1, x2)
        if self.bottom_up:
            self.top = min(y1, y2)
            self.bottom = max(y1, y2)
        else:
            self.top = max(y1, y2)
            self.bottom = min(y1, y2)
        self.width = abs(x2 - x1)
        self.height = abs(y2 - y1)

    def bounding_rect(self, pts):
        """Makes a bounding rect from the extents of a list of points 
        or a list of rects """
        if len(pts) == 0:
            return
        bx = []
        by = []
        for pt in pts:
            if isinstance(pt, Point):
                (x, y) = pt.as_tuple()
            elif isinstance(pt, Rect):
                x, y = pt.left, pt.top
                bx.append(x)
                by.append(y)
                x, y = pt.right, pt.bottom
            else:
                x, y = pt[0], pt[1]
            bx.append(x)
            by.append(y)
        self.left = min(bx)
        self.right = max(bx)
        if self.bottom_up:
            self.top = min(by)
            self.bottom = max(by)
        else:
            self.top = max(by)
            self.bottom = min(by)
        self.width = abs(self.right - self.left)
        self.height = abs(self.top - self.bottom)

    def set_size(self, width, height):
        """Sets a new size for the rectangle."""
        self.left = -width / 2
        self.right = width / 2
        if self.bottom_up:
            self.top = -height / 2
            self.bottom = height / 2
        else:
            self.top = height / 2
            self.bottom = -height / 2
        self.width = width
        self.height = height

    def set_size_anchored(self, width, height, anchor_pt="centre centre"):
        """Sets a new size for the rectangle and optionally anchors the 
        rectangle to any one of 10 points specified with a string containing
        anchor point description, e.g. 'top left', 'right', 'bottom centre' """
        if "left" in anchor_pt:
            x1 = self.left
            x2 = self.left + width
        elif "right" in anchor_pt:
            x1 = self.right
            x2 = self.right - width
        else:
            x1 = self.left + self.width/2 - width/2
            x2 = self.right - self.width/2 + width/2
        if "top" in anchor_pt:
            y1 = self.top
            y2 = self.top - height
        elif "bottom" in anchor_pt:
            y1 = self.bottom
            y2 = self.bottom + height
        else:
            y1 = self.top - self.height/2 + height/2
            y2 = self.bottom + self.height/2 - height/2
        if self.bottom_up:
            y1, y2 = y2, y1
        self.set_points((x1, y1), (x2, y2))

    def anchor_to_pt(self, rect, from_pt="centre centre", to_pt="centre centre"):
        """Moves a rectangle from its anchor point to another rectangle's 
        anchor point. Example: "top right" to "bottom left" """
        x, y = rect.get_anchor_pt(to_pt)
        if "left" in from_pt:
            x1 = x
            x2 = max(x, self.right) if "resize" in to_pt else x1 + self.width
        elif "right" in from_pt:
            x2 = x
            x1 = min(self.left, x) if "resize" in to_pt else x2 - self.width
        elif "centre" in from_pt or "center" in from_pt:
            x1 = x - self.width/2
            x2 = x1 + self.width
        else:
            x1 = self.left
            x2 = self.right
        if "top" in from_pt:
            y1 = y
            y2 = min(y, self.bottom) if "resize" in to_pt else y1 - self.height
        elif "bottom" in from_pt:
            y2 = y
            y1 = max(self.top, y) if "resize" in to_pt else y2 + self.height
        elif "centre" in from_pt or "center" in from_pt:
            y1 = y + self.height / 2
            y2 = y1 - self.height
        else:
            y1 = self.top
            y2 = self.bottom
        if self.bottom_up:
            y1, y2 = min(y2, y1), max(y2, y1)
        self.set_points((x1, y1), (x2, y2))

    def anchor_to_rect(self, rect, anchor_pt= "centre centre"):
        """Moves rectangle to an anchor reference of another rectangle.
        'top left' moves this rectangle to the other rectangle's top left
        for example."""
        self.anchor_to_pt(rect, anchor_pt, anchor_pt)

    def anchor_with_constraint(self, rect, constraint):
        """Moves a rectangle from its anchor point to another rectangle's 
        anchor point. Example: "top right to bottom left" or "below" """
        c = constraint.lower()
        if c == "below":
            self.anchor_to_pt(rect, from_pt="top", to_pt="bottom")
        elif c == "above":
            self.anchor_to_pt(rect, from_pt="bottom", to_pt="top")
        elif c == "rightof":
            self.anchor_to_pt(rect, from_pt="left", to_pt="right")
        elif c == "leftof":
            self.anchor_to_pt(rect, from_pt="right", to_pt="left")
        elif c == "middleof":
            self.anchor_to_pt(rect, from_pt="centre", to_pt="centre")
        else:
            c = constraint.split()
            cu = []
            for e in c:
                if e.lower() == "to":
                    cu.append("TO")
                else:
                    cu.append(e.lower())
            c = " ".join(cu)
            c = c.split("TO")
            if len(c) == 2:
                self.anchor_to_pt(rect, from_pt=c[0], to_pt=c[1])

    def contains(self, pt):
        """Return true if a point is inside the rectangle."""
        x, y = self._xy_from_pt(pt)
        if self.left <= x <= self.right:
            if not self.bottom_up:
                if self.bottom <= y <= self.top:
                    return True
            else:
                if self.top <= y <= self.bottom:
                    return True
        return False

    def overlaps(self, other):
        """Return true if a rectangle overlaps this rectangle."""
        return (
            self.right > other.left
            and self.left < other.right
            and self.top < other.bottom
            and self.bottom > other.top
        )

    def expanded_by(self, n):
        """Return a rectangle with extended borders.

        Create a new rectangle that is wider and taller than the
        immediate one. All sides are extended by "n" points.
        """
        p1 = Point(self.left - n, self.top - n)
        p2 = Point(self.right + n, self.bottom + n)
        r = Rect()
        r.set_points(p1, p2)
        return r

    @staticmethod
    def bounding_rect_from_rects(rects):
        r = Rect()
        r.bounding_rect(rects)
        return r

    @staticmethod
    def layout_rects(rects, bounds, row_wise=True, vert_align="top", horz_align="left", auto_adjust=True):
        def dict_idx(row, col):
            return "%d_%d" % (row, col)        
    
        def compute_wasted_space(rd, row_wise=True):
            rows, cols = 0, 0
            for k, v in rd.items():
                row, col = k.split("_")
                rows = max(rows, int(row))
                cols = max(cols, int(col))
            rows += 1
            cols += 1
            rws = []
            if row_wise:
                for row in range(rows):
                    rw = 0
                    for col in range(cols):
                        if dict_idx(row, col) in rd:
                            r = rd["%d_%d" % (row, col)]
                            rw += r.width
                    rws.append(rw)
            else:
                for col in range(cols):
                    ch = 0
                    for row in range(rows):
                        if dict_idx(row, col) in rd:
                            r = rd["%d_%d" % (row, col)]
                            ch += r.height
                    rws.append(ch)
            if abs(max(rws)) > 0:
                ws = (max(rws) - min(rws)) / max(rws)
            else:
                ws = 0
            return ws

        wasted_space = 1.0
        
        times = 0 if auto_adjust else 9
        last_wasted_space = 1.0
        while (wasted_space > 0.25) and (times < 10):
            rw, rh = 0, 0
            cx, cy = bounds.left, bounds.top
            rd = {}
            row, col = 0, 0
            if row_wise:
                for r in rects:
                    cw, ch = r.width, r.height
                    rh = max(rh, ch)
                    if rw + cw <= bounds.width:
                        rw += cw
                        r.move_top_left_to((cx, cy))
                        rd[dict_idx(row, col)] = r
                        cx += cw
                        col += 1
                    else:
                        # overflowed width, go to next row
                        rw, col = 0, 0
                        row += 1
                        cx, cy = bounds.left, cy - rh
                        r.move_top_left_to((cx, cy))
                        rd[dict_idx(row, col)] = r
                        col += 1
                        cx += cw
                        rw, rh = cw, ch
            else:
                for r in rects:
                    cw, ch = r.width, r.height
                    rw = max(rw, cw)
                    if rh + ch <= bounds.height:
                        rh += ch
                        r.move_top_left_to((cx, cy))
                        rd[dict_idx(row, col)] = r
                        cy -= ch
                        row += 1
                    else:
                        # overflowed height, go to next col
                        rh, row = 0, 0
                        col += 1
                        cx, cy = cx + rw, bounds.top
                        r.move_top_left_to((cx, cy))
                        rd[dict_idx(row, col)] = r
                        row += 1
                        cy -= ch
                        rw, rh = cw, ch
            last_wasted_space = wasted_space
            wasted_space = compute_wasted_space(rd, row_wise=row_wise)
            if wasted_space > last_wasted_space:
                if row_wise:
                    bounds.right += 0.05 * bounds.width
                    bounds.width = bounds.right - bounds.left
                else:
                    bounds.bottom -= 0.05 * bounds.height
                    bounds.height = abs(bounds.bottom - bounds.top)
                times = 9
            else:
                if row_wise:
                    bounds.right -= 0.05 * bounds.width
                    bounds.width = bounds.right - bounds.left
                else:
                    bounds.bottom += 0.05 * bounds.height
                    bounds.height = abs(bounds.bottom - bounds.top)
                times += 1


        # Re-align each row or column based on vert_align and horz_align respectively
        rows, cols = 0, 0
        for k, v in rd.items():
            row, col = k.split("_")
            rows = max(rows, int(row))
            cols = max(cols, int(col))
        rows += 1
        cols += 1
        new_rects = []
        if row_wise:
            for row in range(rows):
                rh = 0
                for col in range(cols):
                    if dict_idx(row, col) in rd:
                        rh = max(rh, rd[dict_idx(row, col)].height)
                for col in range(cols):
                    if dict_idx(row, col) in rd:
                        r = copy.copy(rd[dict_idx(row, col)])
                        if vert_align == "bottom":
                            r.move_bottom_left_to((r.left, r.top - rh))
                        elif vert_align == "centre":
                            r.move_top_left_to((r.left, r.top - rh/2 + r.height/2))
                        new_rects.append(r)
        else:
            for col in range(cols):
                cw = 0
                for row in range(rows):
                    if dict_idx(row, col) in rd:
                        cw = max(cw, rd[dict_idx(row, col)].width)
                for row in range(rows):
                    if dict_idx(row, col) in rd:
                        r = copy.copy(rd[dict_idx(row, col)])
                        if horz_align == "right":
                            r.move_top_left_to((r.left + cw - r.width, r.top))
                        elif horz_align == "centre":
                            r.move_top_left_to((r.left + cw/2 - r.width/2, r.top))
                        new_rects.append(r)
          
        return new_rects


def GetBoundingRect(length, width, angle):
    thr = radians(angle)
    rlength = abs(length * cos(thr)) + abs(width * sin(thr))
    rwidth = abs(length * sin(thr)) + abs(width * cos(thr))
    return rlength, rwidth


def GetFinalPoint(startPoint, segments, startAngle=0.0):

    currentPoint = startPoint
    currentAngle = startAngle

    for seg in segments:
        if seg[0] == "S":
            currentPoint.x += seg[1] * cos(radians(currentAngle))
            currentPoint.y += seg[1] * sin(radians(currentAngle))
        elif seg[0] == "R":
            offy = seg[1] * (1.0 - cos(radians(seg[2])))
            offx = seg[1] * sin(radians(seg[2]))
            op = Point(offx, offy)
            opn = op.rotate(radians(currentAngle))
            if seg[2] > 0:
                currentPoint.x += opn.x
                currentPoint.y += opn.y
            else:
                currentPoint.x -= opn.x
                currentPoint.y -= opn.y
            currentAngle += seg[2]
        elif seg[0] == "T":
            d = sqrt(seg[1] * seg[1] + seg[2] * seg[2])
            ang = atan(seg[2] / seg[1])
            currentPoint.x += d * cos(ang + radians(currentAngle))
            currentPoint.y += d * sin(ang + radians(currentAngle))
            currentAngle += seg[3]

    return currentPoint, currentAngle


def GetBestRectMetrics(fromWidth, fromHeight, inWidth, inHeight):

    if fromWidth > fromHeight:
        bestWidth = inWidth
        bestHeight = inWidth * fromHeight / fromWidth
    else:
        bestHeight = inHeight
        bestWidth = inHeight * fromWidth / fromHeight
    return bestWidth, bestHeight



class RadialPoint:
    """ Symmetric Radial Points

    A specialized class for computing symmetrically offset points
    on a circle at a specified angluar offset.  The point on the circle
    is called 'mid', the point inside the circle is 'inner' and the point
    outside the cirucle is 'outer' as referred to by the methods 'mid_xy', etc.

    The points are returned re-centred to the origin.  That is, a 'mid' point
    at angle=0 deg and radius=R is returned at (0, 0).  At angle 45, it would
    return as (-R + R*cos(45), R*sin(45)).  Positive angles are in the positive
    'Y' axis and negative angles are in negative 'Y'.

    An optional linear offset can be specified which offsets the point by a
    tangential amount in either the positive or negative direction.
    """
    def __init__(self, radius=0, offset=0, angle=0, origin=(0,0,0)):
        self.radius = radius
        self.offset = offset
        self.angleDeg = angle
        self.angleRad = radians(angle)
        self.origin = origin
        self.r_inner = 0
        self.r_outer = 0
        self.lin_offset = 0.0
        self.lin_x = 0.0
        self.lin_y = 0.0
        self._compute_points()

    def _compute_points(self):
        ri = self.radius - self.offset / 2.0 + self.origin[0]
        ro = self.radius + self.offset / 2.0 + self.origin[0]
        rir = self.radius - self.offset / 2.0
        ror = self.radius + self.offset / 2.0
        if (ri < 0) and (ro < 0):
            self.r_outer = rir
            self.r_inner = ror
        elif (ri < 0) and (abs(ri) > abs(ro)):
            self.r_outer = rir
            self.r_inner = ror
        else:
            self.r_outer = ror
            self.r_inner = rir
        self.lin_x = self.lin_offset * sin(self.angleRad)
        self.lin_y = self.lin_offset * cos(self.angleRad)
        self.angleRad = radians(self.angleDeg)

    def _radial_x(self, r):
        x = (r * cos(self.angleRad)) - self.radius - self.lin_x
        return self.origin[0] + x

    def _radial_y(self, r):
        y = r * sin(self.angleRad) + self.lin_y
        return self.origin[1] + y

    def _radial_xoffs(self, r):
        return r * sin(self.angleRad)
    def _radial_yoffs(self, r):
        return r * cos(self.angleRad)

    def distance_to(self, other):
        xx = (self.origin[0] - other.origin[0])
        yy = (self.origin[1] - other.origin[1])
        zz = (self.origin[2] - other.origin[2])
        return sqrt(xx*xx + yy*yy + zz * zz)

    def slide_xy_copy(self, x, y):
        rp = copy.copy(self)
        o = (self.origin[0] + x, self.origin[1] + y, 0)
        rp.origin = o
        return rp

    def slide_polar_copy(self, r, theta):
        x = r * cos(radians(theta))
        y = r * sin(radians(theta))
        return self.slide_xy_copy(x, y)

    def slide_xy(self, x, y):
        o = (self.origin[0] + x, self.origin[1] + y, 0)
        self.origin = o

    def _swapped(x, y):
        return y, x

    def slide_polar(self, r, theta):
        x = r * cos(radians(theta))
        y = r * sin(radians(theta))
        self.slide_xy(x, y)

    def inner_xy(self, radial_offset=0.0):
        self._compute_points()
        return (self._radial_x(self.r_inner) - self._radial_xoffs(radial_offset),
                self._radial_y(self.r_inner) + self._radial_yoffs(radial_offset))

    def inner_yx(self, radial_offset=0.0):
        return (_swapped(self.inner_xy(radial_offset)))

    def inner_3d(self, radial_offset=0.0):
        p = self.inner_xy(radial_offset)
        return (p[0], p[1], self.origin[2])

    def outer_xy(self, radial_offset=0.0):
        self._compute_points()
        return (self._radial_x(self.r_outer) - self._radial_xoffs(radial_offset),
                self._radial_y(self.r_outer) + self._radial_yoffs(radial_offset))

    def outer_yx(self, radial_offset=0.0):
        return (_swapped(self.outer_xy(radial_offset)))

    def outer_3d(self, radial_offset=0.0):
        p = self.outer_xy(radial_offset)
        return (p[0], p[1], self.origin[2])

    def mid_xy(self, radial_offset=0.0):
        self._compute_points()
        return (self._radial_x(self.radius) - self._radial_xoffs(radial_offset),
                self._radial_y(self.radius) + self._radial_yoffs(radial_offset))
    def mid_yx(self, radial_offset=0.0):
        return (_swapped(self.mid_xy(radial_offset)))

    def mid_3d(self, radial_offset=0.0):
        p = self.mid_xy(radial_offset)
        return (p[0], p[1], self.origin[2])

    def angle(self):
        return -self.angleDeg

    def __str__(self):
        pi = self.inner_xy()
        po = self.outer_xy()
        pm = self.mid_xy()
        return "(%7.2f, %7.2f) -- (%7.2f, %7.2f) -- (%7.2f, %7.2f) / %7.2f deg R=%.2f " % (
          pi[0], pi[1], pm[0], pm[1], po[0], po[1], self.angleDeg, self.radius
        )

    def __repr__(self):
        return "%s(%s, %s, %s)" % (self.__class__.__name__, self.radius, self.offset, self.angleDeg)

def ShiftToOrigin(pts):
    opts = []
    xo = pts[0][0]
    yo = pts[0][1]
    for pt in pts:
        opts.append((pt[0] - xo, pt[1] - yo))
    return opts

class SplinePoints():

    def __init__(self, pts):
        self.pts = pts
        self.pts_origin = []
        self.xo = pts[0][0]
        self.yo = pts[0][1]
        for pt in pts[1:]:
            self.pts_origin.append((pt[0] - self.xo, pt[1] - self.yo))

    def origin_offset(self):
        return (self.xo, self.yo)


def PrintPointList(pts):
    nPts = len(pts)
    for i, pt in enumerate(pts):
        print("%d/%d: %s" % (i+1, nPts, pt))

def PrintPointsInDict(dict):
    for key, value in dict.items():
        if isinstance(value, list):
            s = []
            s.append("%17s: [" %(key))
            if isinstance(value, (tuple, list)):
                for v in value:
                    if isinstance(v, (tuple, list)):
                        s.append("(%6.2f, %6.2f), " % (v[0], v[1]))
            else:
                s.append("%s" % (value))
            s.append("]")
            rs = ''.join(s)
            rs = rs.replace("), ]", ")]")
            print(rs)
        elif isinstance(value, tuple):
            if isinstance(value[0], tuple):
                print("%17s: (%6.2f, %6.2f), (%6.2f, %6.2f)" % (key, value[0][0], value[0][1], value[1][0], value[1][1]))
            else:
                print("%17s: (%6.2f, %6.2f)" % (key, value[0], value[1]))
        else:
            print("%17s: %s" % (key, value))

def points2d_at_height(pts, height):
    """ Returns a list of 2D point tuples as 3D tuples at height"""
    if isinstance(pts, tuple):
        if len(pts) == 2:
            return [(*pts, height)]
        return [(pts[0], pts[1], height)]
    pts3d = []
    for pt in pts:
        if len(pt) == 3:
            pts3d.append((pt[0], pt[1], height))
        else:
            pts3d.append((*pt, height))
    return pts3d

def grid_points_2d(length, width, div, width_div=None):
    """ Returns a regularly spaced grid of points occupying a rectangular
    region of length x width partitioned into div intervals.  If different
    spacing is desired in width, then width_div can be specified, otherwise
    it will default to div. If div < 2 in either x or y, then the corresponding
    coordinate will be set to length or width respectively."""
    if div > 1:
        px = [-length/2.0 + (x/(div-1))*length for x in range(div)]
    else:
        px = [length]
    if width_div is not None:
        wd = width_div
    else:
        wd = div
    if wd > 1:
        py = [-width/2.0 + (y/(wd-1))*width for y in range(wd)]
    else:
        py = [width]
    pts = []
    for x in px:
        for y in py:
            pts.append((x, y))
    return pts

def grid_points_at_height(length, width, height, div, width_div=None):
    """ A convenience method to return 2D grid points as 3D points at
    a specified height"""
    pts = grid_points_2d(length, width, div, width_div)
    return points2d_at_height(pts, height)

def euler_to_rot_matrix(euler):
    """ converts a 3D tuple of euler rotation angles into a rotation matrix """
    ax = Identity().rotate(euler[0], XAxis)
    ay = Identity().rotate(euler[1], YAxis)
    az = Identity().rotate(euler[2], ZAxis)
    rm = az * ay * ax
    rm = rm.transpose()
    return rm

def safe_vector(v):
    """ returns a Vector object by automatically inferring the input argument v """
    if isinstance(v, Vector):
        return v
    elif isinstance(v, (tuple, list)):
        return Vector(v[0], v[1], v[2])
    elif isinstance(v, (float, int)):
        return Vector(v, v, v)
    return Vector(0, 0, 0)
    