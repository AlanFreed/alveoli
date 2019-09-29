#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numbers

"""
Module shapeFnTetrahedra.py provides shape functions for interpolating a
tetrahedron.

Copyright (c) 2019 Alan D. Freed

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Module metadata
__version__ = "1.3.0"
__date__ = "09-18-2019"
__update__ = "09-27-2019"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

r"""
Change in version "1.3.0":

Created

Overview of module shapeFnTetrahedra.py:

Module shapeFnTetrahedra.py provides the four shape functions for a point
(xi, eta, zeta) residing within a tetrahedron, or along its boundary, where
'xi', 'eta' and 'zeta' are the x, y and z coordinates, respectively.  Its
vertices in the tetrahedon's natural coordinate system are located at
    vertex1: (xi, eta, zeta) = (0, 0, 0)
    vertex2: (xi, eta, zeta) = (1, 0, 0)
    vertex3: (xi, eta, zeta) = (0, 1, 0)
    vertex4: (xi, eta, zeta) = (0, 0, 1)
The volume of this tetrahedron is 1/6.

Also provided are the spatial derivatives of these shape functions, taken
with respect to coordinates 'xi', 'eta' and 'zeta'.  From these one can
construct approxiamtions for the displacement G and deformation F gradients.


class

    shapeFunction

constructor

    sf = shapeFunction(xi, eta, zeta)
        xi    is the x coordinate in the natural coordinate system
        eta   is the y coordiante in the natural coordiante system
        zeta  is the z coordinate in the natural coordinate system

methods

    y = sf.interpolate(y1, y2, y3, y4)
        y1   is the value of field  y  located at vertex 1
        y2   is the value of field  y  located at vertex 2
        y3   is the value of field  y  located at vertex 3
        y4   is the value of field  y  located at vertex 4
    returns
        y    is its interpolated value for field y at location (xi, eta, zeta)

    det = sf.detJacobian(x1, x2, x3, x4)
        x1    is a tuple of physical coordinates (x, y, z) located at vertex 1
        x2    is a tuple of physical coordinates (x, y, z) located at vertex 2
        x3    is a tuple of physical coordinates (x, y, z) located at vertex 3
        x4    is a tuple of physical coordinates (x, y, z) located at vertex 4
    returns
        det   is the determinant of the Jacobian matrix
    inputs are tuples of coordinates evaluated in a global coordinate system

    Gmtx = sf.G(x1, x2, x3, x4, x01, x02, x03, x04)
        x1   is a tuple of physical  coordinates (x, y, z) located at vertex 1
        x2   is a tuple of physical  coordinates (x, y, z) located at vertex 2
        x3   is a tuple of physical  coordinates (x, y, z) located at vertex 3
        x4   is a tuple of physical  coordinates (x, y, z) located at vertex 4
        x01  is a tuple of reference coordinates (x, y, z) located at vertex 1
        x02  is a tuple of reference coordinates (x, y, z) located at vertex 2
        x03  is a tuple of reference coordinates (x, y, z) located at vertex 3
        x04  is a tuple of reference coordinates (x, y, z) located at vertex 4
    returns
        Gmtx is the displacement gradient (a 3x3 matrix) at (xi, eta, zeta)
                    / du/dx  du/dy  du/dz \         u = x - X  or  x - x0
             Gmtx = | dv/dx  dv/dy  dv/dz |  where  v = y - Y  or  y - y0
                    \ dw/dx  dw/dy  dw/dz /         w = z - Z  or  z - z0
    inputs are tuples of coordinates evaluated in a global coordinate system

    Fmtx = sf.F(x1, x2, x3, x4, x01, x02, x03, x04)
        x1   is a tuple of physical  coordinates (x, y, z) located at vertex 1
        x2   is a tuple of physical  coordinates (x, y, z) located at vertex 2
        x3   is a tuple of physical  coordinates (x, y, z) located at vertex 3
        x4   is a tuple of physical  coordinates (x, y, z) located at vertex 4
        x01  is a tuple of reference coordinates (x, y, z) located at vertex 1
        x02  is a tuple of reference coordinates (x, y, z) located at vertex 2
        x03  is a tuple of reference coordinates (x, y, z) located at vertex 3
        x04  is a tuple of reference coordinates (x, y, z) located at vertex 4
    returns
        Fmtx is the deformation gradient (a 3x3 matrix) at (xi, eta, zeta)
                    / dx/dX  dx/dY  dx/dZ \         X = x0
             Fmtx = | dy/dX  dy/dY  dy/dZ |  where  Y = y0
                    \ dz/dX  dz/dY  dz/dZ /         Z = z0
    inputs are tuples of coordinates evaluated in a global coordinate system

variables

    # the shape functions

    sf.N1        the 1st shape function
    sf.N2        the 2nd shape function
    sf.N3        the 3rd shape function
    sf.N4        the 4th shape function

    sf.Nmatx     the 3x12 matrix of shape functions for the tetrahedron

    # partial derivatives of the shape functions

    # partial derivative: d N_i / dXi, i = 1..4
    sf.dN1dXi    gradient of the 1st shape function wrt the xi coordinate
    sf.dN2dXi    gradient of the 2nd shape function wrt the xi coordinate
    sf.dN3dXi    gradient of the 3rd shape function wrt the xi coordinate
    sf.dN4dXi    gradient of the 4th shape function wrt the xi coordinate

    # partial derivative: d N_i / dEta, i = 1..4
    sf.dN1dEta   gradient of the 1st shape function wrt the eta coordinate
    sf.dN2dEta   gradient of the 2nd shape function wrt the eta coordinate
    sf.dN3dEta   gradient of the 3rd shape function wrt the eta coordinate
    sf.dN4dEta   gradient of the 4th shape function wrt the eta coordinate

    # partial derivative: d N_i / dZeta, i = 1..4
    sf.dN1dZeta  gradient of the 1st shape function wrt the zeta coordinate
    sf.dN2dZeta  gradient of the 2nd shape function wrt the zeta coordinate
    sf.dN3dZeta  gradient of the 3rd shape function wrt the zeta coordinate
    sf.dN4dZeta  gradient of the 4th shape function wrt the zeta coordinate

Reference
    1) Guido Dhondt, "The Finite Element Method for Three-dimensional
       Thermomechanical Applications", John Wiley & Sons Ltd, 2004.
"""


class shapeFunction(object):

    def __init__(self, xi, eta, zeta):
        # create the four exported shape functions
        self.N1 = 1 - xi - eta - zeta
        self.N2 = xi
        self.N3 = eta
        self.N4 = zeta

        # the 3x12 matrix of shape functions for a tetrahedron
        self.Nmatx = np.array([[self.N1, 0.0, 0.0, self.N2, 0.0, 0.0, self.N3,
                                0.0, 0.0, self.N4, 0.0, 0.0],
                               [0.0, self.N1, 0.0, 0.0, self.N2, 0.0, 0.0,
                                self.N3, 0.0, 0.0, self.N4, 0.0],
                               [0.0, 0.0, self.N1, 0.0, 0.0, self.N2, 0.0,
                                0.0, self.N3, 0.0, 0.0, self.N4]])

        # create the ten, eported, first derivatives of the shape functions
        self.dN1dXi = -1
        self.dN2dXi = 1
        self.dN3dXi = 0
        self.dN4dXi = 0

        self.dN1dEta = -1
        self.dN2dEta = 0
        self.dN3dEta = 1
        self.dN4dEta = 0

        self.dN1dZeta = -1
        self.dN2dZeta = 0
        self.dN3dZeta = 0
        self.dN4dZeta = 1

        return  # the object

    def interpolate(self, y1, y2, y3, y4):
        y = self.N1 * y1 + self.N2 * y2 + self.N3 * y3 + self.N4 * y4
        return y

    def detJacobian(self, x1, x2, x3, x4):
        jacob = np.zeros((3, 3), dtype=float)
        if isinstance(x1, tuple):
            jacob[0, 0] = (self.dN1dXi * x1[0] + self.dN2dXi * x2[0] +
                           self.dN3dXi * x3[0] + self.dN4dXi * x4[0])
            jacob[0, 1] = (self.dN1dXi * x1[1] + self.dN2dXi * x2[1] +
                           self.dN3dXi * x3[1] + self.dN4dXi * x4[1])
            jacob[0, 2] = (self.dN1dXi * x1[2] + self.dN2dXi * x2[2] +
                           self.dN3dXi * x3[2] + self.dN4dXi * x4[2])
            jacob[1, 0] = (self.dN1dEta * x1[0] + self.dN2dEta * x2[0] +
                           self.dN3dEta * x3[0] + self.dN4dEta * x4[0])
            jacob[1, 1] = (self.dN1dEta * x1[1] + self.dN2dEta * x2[1] +
                           self.dN3dEta * x3[1] + self.dN4dEta * x4[1])
            jacob[1, 2] = (self.dN1dEta * x1[2] + self.dN2dEta * x2[2] +
                           self.dN3dEta * x3[2] + self.dN4dEta * x4[2])
            jacob[2, 0] = (self.dN1dZeta * x1[0] + self.dN2dZeta * x2[0] +
                           self.dN3dZeta * x3[0] + self.dN4dZeta * x4[0])
            jacob[2, 1] = (self.dN1dZeta * x1[1] + self.dN2dZeta * x2[1] +
                           self.dN3dZeta * x3[1] + self.dN4dZeta * x4[1])
            jacob[2, 2] = (self.dN1dZeta * x1[2] + self.dN2dZeta * x2[2] +
                           self.dN3dZeta * x3[2] + self.dN4dZeta * x4[2])
            # determine the determinant of the Jacobian of a tetrahedron
            det = np.linalg.det(jacob)
        else:
            raise RuntimeError(
                    "Error: each argument of shapeFunction.detJacobian must " +
                    "be a tuple of coordinates, e.g., (x, y, z).")
        return det

    def G(self, x1, x2, x3, x4, x01, x02, x03, x04):
        if isinstance(x1, tuple):
            u1 = x1[0] - x01[0]
            u2 = x2[0] - x02[0]
            u3 = x3[0] - x03[0]
            u4 = x4[0] - x04[0]
            v1 = x1[1] - x01[1]
            v2 = x2[1] - x02[1]
            v3 = x3[1] - x03[1]
            v4 = x4[1] - x04[1]
            w1 = x1[2] - x01[2]
            w2 = x2[2] - x02[2]
            w3 = x3[2] - x03[2]
            w4 = x4[2] - x04[2]

            # determine the displacement gradient
            disGrad = np.zeros((3, 3), dtype=float)
            disGrad[0, 0] = (self.dN1dXi * u1 + self.dN2dXi * u2 +
                             self.dN3dXi * u3 + self.dN4dXi * u4)
            disGrad[0, 1] = (self.dN1dEta * u1 + self.dN2dEta * u2 +
                             self.dN3dEta * u3 + self.dN4dEta * u4)
            disGrad[0, 2] = (self.dN1dZeta * u1 + self.dN2dZeta * u2 +
                             self.dN3dZeta * u3 + self.dN4dZeta * u4)
            disGrad[1, 0] = (self.dN1dXi * v1 + self.dN2dXi * v2 +
                             self.dN3dXi * v3 + self.dN4dXi * v4)
            disGrad[1, 1] = (self.dN1dEta * v1 + self.dN2dEta * v2 +
                             self.dN3dEta * v3 + self.dN4dEta * v4)
            disGrad[1, 2] = (self.dN1dZeta * v1 + self.dN2dZeta * v2 +
                             self.dN3dZeta * v3 + self.dN4dZeta * v4)
            disGrad[2, 0] = (self.dN1dXi * w1 + self.dN2dXi * w2 +
                             self.dN3dXi * w3 + self.dN4dXi * w4)
            disGrad[2, 1] = (self.dN1dEta * w1 + self.dN2dEta * w2 +
                             self.dN3dEta * w3 + self.dN4dEta * w4)
            disGrad[2, 2] = (self.dN1dZeta * w1 + self.dN2dZeta * w2 +
                             self.dN3dZeta * w3 + self.dN4dZeta * w4)

            # determine the current position gradient
            curGrad = np.zeros((3, 3), dtype=float)
            curGrad[0, 0] = (self.dN1dXi * x1[0] + self.dN2dXi * x2[0] +
                             self.dN3dXi * x3[0] + self.dN4dXi * x4[0])
            curGrad[0, 1] = (self.dN1dEta * x1[0] + self.dN2dEta * x2[0] +
                             self.dN3dEta * x3[0] + self.dN4dEta * x4[0])
            curGrad[0, 2] = (self.dN1dZeta * x1[0] + self.dN2dZeta * x2[0] +
                             self.dN3dZeta * x3[0] + self.dN4dZeta * x4[0])
            curGrad[1, 0] = (self.dN1dXi * x1[1] + self.dN2dXi * x2[1] +
                             self.dN3dXi * x3[1] + self.dN4dXi * x4[1])
            curGrad[1, 1] = (self.dN1dEta * x1[1] + self.dN2dEta * x2[1] +
                             self.dN3dEta * x3[1] + self.dN4dEta * x4[1])
            curGrad[1, 2] = (self.dN1dZeta * x1[1] + self.dN2dZeta * x2[1] +
                             self.dN3dZeta * x3[1] + self.dN4dZeta * x4[1])
            curGrad[2, 0] = (self.dN1dXi * x1[2] + self.dN2dXi * x2[2] +
                             self.dN3dXi * x3[2] + self.dN4dXi * x4[2])
            curGrad[2, 1] = (self.dN1dEta * x1[2] + self.dN2dEta * x2[2] +
                             self.dN3dEta * x3[2] + self.dN4dEta * x4[2])
            curGrad[2, 2] = (self.dN1dZeta * x1[2] + self.dN2dZeta * x2[2] +
                             self.dN3dZeta * x3[2] + self.dN4dZeta * x4[2])

            # determine the inverse of the current position gradient
            curGradInv = np.linalg.inv(curGrad)

            # calculate the displacement gradient
            Gmtx = np.dot(disGrad, curGradInv)
        else:
            raise RuntimeError(
                         "Error: each argument of shapeFunction.G must be a " +
                         "tuple of coordinates, e.g., (x, y, z).")
        return Gmtx

    def F(self, x1, x2, x3, x4, x01, x02, x03, x04):
        if isinstance(x1, tuple):
            u1 = x1[0] - x01[0]
            u2 = x2[0] - x02[0]
            u3 = x3[0] - x03[0]
            u4 = x4[0] - x04[0]
            v1 = x1[1] - x01[1]
            v2 = x2[1] - x02[1]
            v3 = x3[1] - x03[1]
            v4 = x4[1] - x04[1]
            w1 = x1[2] - x01[2]
            w2 = x2[2] - x02[2]
            w3 = x3[2] - x03[2]
            w4 = x4[2] - x04[2]
            # determine the displacement gradient
            disGrad = np.zeros((3, 3), dtype=float)
            disGrad[0, 0] = (self.dN1dXi * u1 + self.dN2dXi * u2 +
                             self.dN3dXi * u3 + self.dN4dXi * u4)
            disGrad[0, 1] = (self.dN1dEta * u1 + self.dN2dEta * u2 +
                             self.dN3dEta * u3 + self.dN4dEta * u4)
            disGrad[0, 2] = (self.dN1dZeta * u1 + self.dN2dZeta * u2 +
                             self.dN3dZeta * u3 + self.dN4dZeta * u4)
            disGrad[1, 0] = (self.dN1dXi * v1 + self.dN2dXi * v2 +
                             self.dN3dXi * v3 + self.dN4dXi * v4)
            disGrad[1, 1] = (self.dN1dEta * v1 + self.dN2dEta * v2 +
                             self.dN3dEta * v3 + self.dN4dEta * v4)
            disGrad[1, 2] = (self.dN1dZeta * v1 + self.dN2dZeta * v2 +
                             self.dN3dZeta * v3 + self.dN4dZeta * v4)
            disGrad[2, 0] = (self.dN1dXi * w1 + self.dN2dXi * w2 +
                             self.dN3dXi * w3 + self.dN4dXi * w4)
            disGrad[2, 1] = (self.dN1dEta * w1 + self.dN2dEta * w2 +
                             self.dN3dEta * w3 + self.dN4dEta * w4)
            disGrad[2, 2] = (self.dN1dZeta * w1 + self.dN2dZeta * w2 +
                             self.dN3dZeta * w3 + self.dN4dZeta * w4)

            # determine the reference position gradient
            refGrad = np.zeros((3, 3), dtype=float)
            refGrad[0, 0] = (self.dN1dXi * x01[0] + self.dN2dXi * x02[0] +
                             self.dN3dXi * x03[0] + self.dN4dXi * x04[0])
            refGrad[0, 1] = (self.dN1dEta * x01[0] + self.dN2dEta * x02[0] +
                             self.dN3dEta * x03[0] + self.dN4dEta * x04[0])
            refGrad[0, 2] = (self.dN1dZeta * x01[0] + self.dN2dZeta * x02[0] +
                             self.dN3dZeta * x03[0] + self.dN4dZeta * x04[0])
            refGrad[1, 0] = (self.dN1dXi * x01[1] + self.dN2dXi * x02[1] +
                             self.dN3dXi * x03[1] + self.dN4dXi * x04[1])
            refGrad[1, 1] = (self.dN1dEta * x01[1] + self.dN2dEta * x02[1] +
                             self.dN3dEta * x03[1] + self.dN4dEta * x04[1])
            refGrad[1, 2] = (self.dN1dZeta * x01[1] + self.dN2dZeta * x02[1] +
                             self.dN3dZeta * x03[1] + self.dN4dZeta * x04[1])
            refGrad[2, 0] = (self.dN1dXi * x01[2] + self.dN2dXi * x02[2] +
                             self.dN3dXi * x03[2] + self.dN4dXi * x04[2])
            refGrad[2, 1] = (self.dN1dEta * x01[2] + self.dN2dEta * x02[2] +
                             self.dN3dEta * x03[2] + self.dN4dEta * x04[2])
            refGrad[2, 2] = (self.dN1dZeta * x01[2] + self.dN2dZeta * x02[2] +
                             self.dN3dZeta * x03[2] + self.dN4dZeta * x04[2])

            # determine the inverse of the reference position gradient
            refGradInv = np.linalg.inv(refGrad)

            # calculate the deformation gradient
            Fmtx = np.identity(3, dtype=float) + np.dot(disGrad, refGradInv)
        else:
            raise RuntimeError(
                         "Error: each argument of shapeFunction.F must be a " +
                         "tuple of coordinates, e.g., (x, y, z).")
        return Fmtx
