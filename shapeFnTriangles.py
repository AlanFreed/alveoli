#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from shapeFunctions import ShapeFunction as ShapeFn

"""
Module shapeFnTriangles.py implements base class ShapeFunction for triangles.

Copyright (c) 2020 Alan D. Freed

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Module metadata
__version__ = "1.0.0"
__date__ = "03-10-2020"
__update__ = "10-19-2020"
__author__ = "Shahla Zamani and Alan D. Freed"
__author_email__ = "Zamani.Shahla@tamu.edu, afreed@tamu.edu"

r"""
Changes made with respect to this version release can be found at end of file.


Overview of module shapeFnTriangles.py:


Module shapeFnTriangles.py provides the class ShapeFunction whose objects
provide the various shape-function functions at some location (xi, eta)
residing within a triangle in its natural co-ordinate system, where
0 <= xi <= 1 and 0 <= eta <= 1-xi.  The vertices of a triangle in its natural
co-ordinate system are located at
    vertex1:  xi = 0, eta = 0
    vertex2:  xi = 1, eta = 0
    vertex3:  xi = 0, eta = 1
so that the area of a triangle in its natural co-ordiante system is 1/2.

Also provided are the spatial derivatives for these shape functions, taken
with respect co-ordinates 'xi' and 'eta', from which one can construct
approximations for the Jacobian J, plus the displacement G and deformation F
gradients.


class

    shapeFunction
        Implements class ShapeFunction exported by module shapeFunctions.py.

constructor

    sf = ShapeFunction(coordinates)
        coordinates     A tuple of natural co-ordinate for interpolating at.
                        For a triangle, coordinates = (xi, eta) where xi is
                        the 'x' natural co-ordinate in range [0, 1] and eta is
                        the 'y' natural co-ordinate in range [0, 1-xi].
    returns
        sf              a new instance of class ShapeFunction for 2D triangles

variables

    # the shape functions are

    sf.N1        the 1st shape function
    sf.N2        the 2nd shape function
    sf.N3        the 3rd shape function

    sf.Nmtx      a 2x6 matrix of shape functions for representing a triangle

    # partial derivatives of the shape functions

    # partial derivative: d N_i / dXi, i = 1..3
    sf.dN1dXi    gradient of the 1st shape function wrt the xi co-ordinate
    sf.dN2dXi    gradient of the 2nd shape function wrt the xi co-ordinate
    sf.dN3dXi    gradient of the 3rd shape function wrt the xi co-ordinate

    # partial derivative: d N_i / dEta, i = 1..3
    sf.dN1dEta   gradient of the 1st shape function wrt the eta co-ordinate
    sf.dN2dEta   gradient of the 2nd shape function wrt the eta co-ordinate
    sf.dN3dEta   gradient of the 3rd shape function wrt the eta co-ordinate

inherited methods

    y = sf.interpolate(y1, y2, y3)
        y1   is a physical field of arbitrary type located at vertex 1
        y2   is a physical field of arbitrary type located at vertex 2
        y3   is a physical field of arbitrary type located at vertex 3
    returns
        y    is the interpolated value for this field at location (xi, eta)
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    Jmtx = sf.jacobianMatrix(x1, x2, x3)
        x1   is a tuple of physical co-ordinates (x) locating vertex 1
        x2   is a tuple of physical co-ordinates (x) locating vertex 2
        x3   is a tuple of physical co-ordinates (x) locating vertex 3
    returns
        Jmtx is the Jacobian matrix (a 2x2 matrix) at location (xi, eta)
                    /  dx/dXi  dy/dXi  \
             Jmtx = |                  |
                    \ dx/dEta  dy/dEta /
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Jdet = sf.jacobianDeterminant(x1, x2, x3)
        x1   is a tuple of physical co-ordinates (x) locating vertex 1
        x2   is a tuple of physical co-ordinates (x) locating vertex 2
        x3   is a tuple of physical co-ordinates (x) locating vertex 3
    returns
        Jdet is the determinant of the Jacobian matrix
    Inputs are co-ordinates evaluated in a global co-ordinate system.

    Gmtx = sf.G(x1, x2, x3, x01, x02, x03)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
    returns
        Gmtx is the displacement gradient (a 2x2 matrix) at location (xi, eta)
                    / du/dx  du/dy \             u = x - X  or  x - x0
             Gmtx = |              |    where
                    \ dv/dx  dv/dy /             v = y - Y  or  y - y0
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Fmtx = sf.F(x1, x2, x3, x01, x02, x03)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
    returns
        Fmtx is the deformation gradient (a 2x2 matrix) at location (xi, eta)
                    / dx/dx0  dx/dy0 \
             Fmtx = |                |
                    \ dy/dx0  dy/dy0 /
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.
"""


class ShapeFunction(ShapeFn):

    # constructor

    def __init__(self, coordinates):
        super(ShapeFunction, self).__init__(coordinates)
        if len(coordinates) == 2:
            xi = coordinates[0]
            eta = coordinates[1]
        else:
            raise RuntimeError("The co-ordinates for a triangle are in 2D.")
        if (xi < -0.0) or (xi > 1.0) or (eta < -0.0) or (eta > (1.0-xi)):
            raise RuntimeError("Co-ordinate 'xi' must be in [0, 1], and "
                               + "co-ordinate 'eta' must be in [0, 1-xi].\n"
                               + "You sent xi = {:06.4f} and ".format(xi)
                               + "eta = {:06.4f}.".format(eta))

        # create the four exported shape functions
        self.N1 = 1.0 - xi - eta
        self.N2 = xi
        self.N3 = eta

        # construct the 2x6 matrix of shape functions for a triangle
        self.Nmtx = np.array([[self.N1, 0.0, self.N2, 0.0, self.N3, 0.0, 0.0, 
                               0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, self.N1, 0.0, self.N2, 0.0, self.N3, 0.0, 
                               0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, self.N1, 0.0, self.N2, 0.0, self.N3, 
                               0.0, 0.0, 0.0, 0.0, 0.0]])

        # create the six, exported, derivatives of these shape functions
        self.dN1dXi = -1.0
        self.dN2dXi = 1.0
        self.dN3dXi = 0.0

        self.dN1dEta = -1.0
        self.dN2dEta = 0.0
        self.dN3dEta = 1.0
        return  # the object

    # methods

    def interpolate(self, y1, y2, y3):
        if type(y1) == type(y2) and type(y2) == type(y3):
            y = self.N1 * y1 + self.N2 * y2 + self.N3 * y3
        else:
            raise RuntimeError("interpolate arguments must be the same type.")
        return y

    def jacobianMatrix(self, x1, x2, x3):
        if (isinstance(x1, tuple) and len(x1) == 3
           and isinstance(x2, tuple) and len(x2) == 3
           and isinstance(x3, tuple) and len(x3) == 3):
            Jmtx = np.zeros((2, 2), dtype=float)
            Jmtx[0, 0] = (self.dN1dXi * x1[0] + self.dN2dXi * x2[0]
                          + self.dN3dXi * x3[0])
            Jmtx[0, 1] = (self.dN1dXi * x1[1] + self.dN2dXi * x2[1]
                          + self.dN3dXi * x3[1])
            Jmtx[1, 0] = (self.dN1dEta * x1[0] + self.dN2dEta * x2[0]
                          + self.dN3dEta * x3[0])
            Jmtx[1, 1] = (self.dN1dEta * x1[1] + self.dN2dEta * x2[1]
                          + self.dN3dEta * x3[1])
        else:
            raise RuntimeError("Each argument of shapeFunction.jacobianMatrix "
                               + "must be a tuple of co-ordinates. ")
        return Jmtx

    def jacobianDeterminant(self, x1, x2, x3):
        Jmtx = self.jacobianMatrix(x1, x2, x3)
        return np.linalg.det(Jmtx)

    def G(self, x1, x2, x3, x01, x02, x03):
        if (isinstance(x1, tuple) and len(x1) == 3
           and isinstance(x2, tuple) and len(x2) == 3
           and isinstance(x3, tuple) and len(x3) == 3
           and isinstance(x01, tuple) and len(x01) == 3
           and isinstance(x02, tuple) and len(x02) == 3
           and isinstance(x03, tuple) and len(x03) == 3):
            u1 = x1[0] - x01[0]
            u2 = x2[0] - x02[0]
            u3 = x3[0] - x03[0]
            v1 = x1[1] - x01[1]
            v2 = x2[1] - x02[1]
            v3 = x3[1] - x03[1]

            # determine the displacement gradient
            disGrad = np.zeros((2, 2), dtype=float)
            disGrad[0, 0] = (self.dN1dXi * u1 + self.dN2dXi * u2
                             + self.dN3dXi * u3)
            disGrad[0, 1] = (self.dN1dEta * u1 + self.dN2dEta * u2
                             + self.dN3dEta * u3)
            disGrad[1, 0] = (self.dN1dXi * v1 + self.dN2dXi * v2
                             + self.dN3dXi * v3)
            disGrad[1, 1] = (self.dN1dEta * v1 + self.dN2dEta * v2
                             + self.dN3dEta * v3)

            # determine the current gradient of position
            curGrad = np.transpose(self.jacobianMatrix(x1, x2, x3))
        else:
            raise RuntimeError("Each argument of shapeFunction.G must be a "
                               + "tuple of co-ordinates, e.g., (x, y).")
        Gmtx = np.matmul(disGrad, np.linalg.inv(curGrad))
        return Gmtx

    def F(self, x1, x2, x3, x01, x02, x03):
        if (isinstance(x1, tuple) and len(x1) == 3
           and isinstance(x2, tuple) and len(x2) == 3
           and isinstance(x3, tuple) and len(x3) == 3
           and isinstance(x01, tuple) and len(x01) == 3
           and isinstance(x02, tuple) and len(x02) == 3
           and isinstance(x03, tuple) and len(x03) == 3):
            u1 = x1[0] - x01[0]
            u2 = x2[0] - x02[0]
            u3 = x3[0] - x03[0]
            v1 = x1[1] - x01[1]
            v2 = x2[1] - x02[1]
            v3 = x3[1] - x03[1]

            # determine the displacement gradient
            disGrad = np.zeros((2, 2), dtype=float)
            disGrad[0, 0] = (self.dN1dXi * u1 + self.dN2dXi * u2
                             + self.dN3dXi * u3)
            disGrad[0, 1] = (self.dN1dEta * u1 + self.dN2dEta * u2
                             + self.dN3dEta * u3)
            disGrad[1, 0] = (self.dN1dXi * v1 + self.dN2dXi * v2
                             + self.dN3dXi * v3)
            disGrad[1, 1] = (self.dN1dEta * v1 + self.dN2dEta * v2
                             + self.dN3dEta * v3)

            # determine the reference gradient of position
            refGrad = np.zeros((2, 2), dtype=float)
            refGrad[0, 0] = (self.dN1dXi * x01[0] + self.dN2dXi * x02[0]
                             + self.dN3dXi * x03[0])
            refGrad[0, 1] = (self.dN1dEta * x01[0] + self.dN2dEta * x02[0]
                             + self.dN3dEta * x03[0])
            refGrad[1, 0] = (self.dN1dXi * x01[1] + self.dN2dXi * x02[1]
                             + self.dN3dXi * x03[1])
            refGrad[1, 1] = (self.dN1dEta * x01[1] + self.dN2dEta * x02[1]
                             + self.dN3dEta * x03[1])
        else:
            raise RuntimeError("Each argument of shapeFunction.F must be a "
                               + "tuple of co-ordinates, e.g., (x, y).")
        Fmtx = (np.eye(2, dtype=float)
                + np.matmul(disGrad, np.linalg.inv(refGrad)))
        return Fmtx


"""
Changes made in version "1.0.0":

Original version
"""
