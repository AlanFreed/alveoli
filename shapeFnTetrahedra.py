#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from shapeFunctions import ShapeFunction as ShapeFn

"""
Module shapeFnTetrahedra.py implements base class ShapeFunction for tetrahedra.

Copyright (c) 2019-2020 Alan D. Freed

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
__date__ = "09-18-2019"
__update__ = "10-19-2020"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

r"""
Changes made with respect to this version release can be found at end of file.


Overview of module shapeFnTetrahedra.py:


Module shapeFnTetrahedra.py provides the class ShapeFunction whose objects
provide the various shape-function functions at some location (xi, eta, zeta)
residing within a triangle in its natural co-ordinate system, where
0 <= xi <= 1, 0 <= eta <= 1-xi and 0 <= zeta <= 1-xi-eta.  The vertices of a
tetrahedron located in its natural co-ordinate system are positioned at
    vertex1: (xi, eta, zeta) = (0, 0, 0)
    vertex2: (xi, eta, zeta) = (1, 0, 0)
    vertex3: (xi, eta, zeta) = (0, 1, 0)
    vertex4: (xi, eta, zeta) = (0, 0, 1)
The volume of this tetrahedron is 1/6.

Also provided are the spatial derivatives of these shape functions, taken with
respect to co-ordinates 'xi', 'eta' and 'zeta', from which one can construct
approximations for the Jacobian J, plus the displacement G and deformation F
gradients, and also the B matrix, both linear and nonlinear.

class

    ShapeFunction
        Implements class ShapeFunction exported by module shapeFunctions.py.

constructor

    sf = ShapeFunction(coordinates)
        coordinates     A tuple of natural co-ordinate for interpolating at.
                        For a tetrahedron, coordinates = (xi, eta, zeta) where
                        xi is the 'x' natural co-ordinate in range [0, 1],
                        eta is the 'y' natural co-ordinate in range [0, 1-xi],
                        zeta is the 'z' natural co-ordinate with [0, 1-xi-eta].
    returns
        sf              a new instance of class ShapeFunction for 3D tetrahedra

variables

    # the shape functions

    sf.N1        the 1st shape function
    sf.N2        the 2nd shape function
    sf.N3        the 3rd shape function
    sf.N4        the 4th shape function

    sf.Nmtx      a 3x12 matrix of shape functions for the tetrahedron located
                 at (xi, eta, zeta)

    # partial derivatives of the shape functions

    # partial derivative: d N_i / dXi, i = 1..4
    sf.dN1dXi    gradient of the 1st shape function wrt the xi co-ordinate
    sf.dN2dXi    gradient of the 2nd shape function wrt the xi co-ordinate
    sf.dN3dXi    gradient of the 3rd shape function wrt the xi co-ordinate
    sf.dN4dXi    gradient of the 4th shape function wrt the xi co-ordinate

    # partial derivative: d N_i / dEta, i = 1..4
    sf.dN1dEta   gradient of the 1st shape function wrt the eta co-ordinate
    sf.dN2dEta   gradient of the 2nd shape function wrt the eta co-ordinate
    sf.dN3dEta   gradient of the 3rd shape function wrt the eta co-ordinate
    sf.dN4dEta   gradient of the 4th shape function wrt the eta co-ordinate

    # partial derivative: d N_i / dZeta, i = 1..4
    sf.dN1dZeta  gradient of the 1st shape function wrt the zeta co-ordinate
    sf.dN2dZeta  gradient of the 2nd shape function wrt the zeta co-ordinate
    sf.dN3dZeta  gradient of the 3rd shape function wrt the zeta co-ordinate
    sf.dN4dZeta  gradient of the 4th shape function wrt the zeta co-ordinate

inherited methods

    y = sf.interpolate(y1, y2, y3, y4)
        y1   is a physical field of arbitrary type located at vertex 1
        y2   is a physical field of arbitrary type located at vertex 2
        y3   is a physical field of arbitrary type located at vertex 3
        y4   is a physical field of arbitrary type located at vertex 4
    returns
        y    is the interpolated value for field y at location (xi, eta, zeta)
    inputs must allow for: i) scalar multiplication and ii) the '+' operator

    Jmtx = sf.jacobianMatrix(x1, x2, x3, x4)
        x1   is a tuple of physical co-ordinates (x, y, z) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y, z) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y, z) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y, z) locating vertex 4
    returns
        Jmtx is the Jacobian matrix (a 2x2 matrix) at location (xi, eta)
                    /  dx/dXi   dy/dXi   dz/dXi  \
             Jmtx = | dx/dEta  dy/dEta  dz/dEta  |
                    \ dx/dZeta dy/dZeta dz/dZeta /
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Jdet = sf.jacobianDeterminant(x1, x2, x3, x4)
        x1   is a tuple of physical co-ordinates (x, y, z) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y, z) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y, z) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y, z) locating vertex 4
    returns
        Jdet is the determinant of the Jacobian matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Gmtx = sf.G(x1, x2, x3, x4, x01, x02, x03, x04)
        x1   is a tuple of physical  co-ordinates (x, y, z) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y, z) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y, z) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y, z) locating vertex 4
        x01  is a tuple of reference co-ordinates (x, y, z) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y, z) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y, z) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y, z) locating vertex 4
    returns
        Gmtx is the displacement gradient (a 3x3 matrix) at (xi, eta, zeta)
                    / du/dx  du/dy  du/dz \         u = x - X  or  u = x - x0
             Gmtx = | dv/dx  dv/dy  dv/dz |  where  v = y - Y  or  v = y - y0
                    \ dw/dx  dw/dy  dw/dz /         w = z - Z  or  w = z - z0
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Fmtx = sf.F(x1, x2, x3, x4, x01, x02, x03, x04)
        x1   is a tuple of physical  co-ordinates (x, y, z) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y, z) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y, z) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y, z) locating vertex 4
        x01  is a tuple of reference co-ordinates (x, y, z) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y, z) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y, z) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y, z) locating vertex 4
    returns
        Fmtx is the deformation gradient (a 3x3 matrix) at (xi, eta, zeta)
                    / dx/dX  dx/dY  dx/dZ \         X = x0
             Fmtx = | dy/dX  dy/dY  dy/dZ |  where  Y = y0
                    \ dz/dX  dz/dY  dz/dZ /         Z = z0
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    BL = sf.BLinear(x1, x2, x3, x4)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y) locating vertex 4
    returns
        BL   is the linear strain displacement matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Hmtx = sf.HmatrixF(x1, x2, x3, x4)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y) locating vertex 4
    returns
        Hmtx  is the first H matrix, which is a derivative of shape functions
              from theta = H * D in the first contribution to nonlinear strain
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Hmtx = sf.HmatrixS(x1, x2, x3, x4)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y) locating vertex 4
    returns
        Hmtx is the second H matrix, which is a derivative of shape functions
             from theta = H * D in second contribution to nonlinear strain
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Hmtx = sf.HmatrixT(x1, x2, x3, x4)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y) locating vertex 4
    returns
        Hmtx is the third H matrix, which is a derivative of shape functions
             from theta = H * D in second contribution to nonlinear strain
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    BN = sf.firstBNonLinear(x1, x2, x3, x4, x01, x02, x03, x04)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y) locating vertex 4
    returns
        BN is first nonlinear contribution to the strain displacement matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    BN = sf.secondBNonLinear(x1, x2, x3, x4, x01, x02, x03, x04)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y) locating vertex 4
    returns
        BN is second nonlinear contribution to the strain displacement matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    BN = sf.thirdBNonLinear(x1, x2, x3, x4, x01, x02, x03, x04)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y) locating vertex 4
    returns
        BN is third nonlinear contribution to the strain displacement matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

Reference
    1) Guido Dhondt, "The Finite Element Method for Three-dimensional
       Thermomechanical Applications", John Wiley & Sons Ltd, 2004.
"""


class ShapeFunction(ShapeFn):

    # constructor

    def __init__(self, coordinates):
        super(ShapeFunction, self).__init__(coordinates)
        if len(coordinates) == 3:
            xi = coordinates[0]
            eta = coordinates[1]
            zeta = coordinates[2]
        else:
            raise RuntimeError("The co-ordinates for a tetrahedron are in 3D.")
        if ((xi < -0.0) or (xi > 1.0) or (eta < -0.0) or (eta > 1.0-xi)
           or (zeta < -0.0) or (zeta > 1.0-xi-eta)):
            raise RuntimeError("Co-ordinate 'xi' must be in [0, 1], "
                               + "co-ordinate 'eta' must be in [0, 1-xi],\n"
                               + "and co-ordinate 'zeta' must be in "
                               + "[0, 1-xi-eta].\nYou sent "
                               + "xi = {:06.4f}, ".format(xi)
                               + "eta = {:06.4f}, and ".format(eta)
                               + "zeta = {:0.6.4f}.".format(zeta))

        # create the four exported shape functions
        self.N1 = 1 - xi - eta - zeta
        self.N2 = xi
        self.N3 = eta
        self.N4 = zeta

        # construct the 3x12 matrix of shape functions for a tetrahedron
        self.Nmtx = np.array([[self.N1, 0.0, 0.0, self.N2, 0.0, 0.0,
                               self.N3, 0.0, 0.0, self.N4, 0.0, 0.0],
                              [0.0, self.N1, 0.0, 0.0, self.N2, 0.0,
                               0.0, self.N3, 0.0, 0.0, self.N4, 0.0],
                              [0.0, 0.0, self.N1, 0.0, 0.0, self.N2,
                               0.0, 0.0, self.N3, 0.0, 0.0, self.N4]])

        # create the twelve, eported, derivatives of these shape functions
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

    # methods

    def interpolate(self, y1, y2, y3, y4):
        if (type(y1) == type(y2) and type(y2) == type(y3)
           and type(y3) == type(y4)):
            y = self.N1 * y1 + self.N2 * y2 + self.N3 * y3 + self.N4 * y4
        else:
            raise RuntimeError("interpolate arguments must be the same type.")
        return y

    def jacobianMatrix(self, x1, x2, x3, x4):
        if (isinstance(x1, tuple) and len(x1) == 3
           and isinstance(x2, tuple) and len(x2) == 3
           and isinstance(x3, tuple) and len(x3) == 3
           and isinstance(x4, tuple) and len(x4) == 3):
            Jmtx = np.zeros((3, 3), dtype=float)
            Jmtx[0, 0] = (self.dN1dXi * x1[0] + self.dN2dXi * x2[0]
                          + self.dN3dXi * x3[0] + self.dN4dXi * x4[0])
            Jmtx[0, 1] = (self.dN1dXi * x1[1] + self.dN2dXi * x2[1]
                          + self.dN3dXi * x3[1] + self.dN4dXi * x4[1])
            Jmtx[0, 2] = (self.dN1dXi * x1[2] + self.dN2dXi * x2[2]
                          + self.dN3dXi * x3[2] + self.dN4dXi * x4[2])
            Jmtx[1, 0] = (self.dN1dEta * x1[0] + self.dN2dEta * x2[0]
                          + self.dN3dEta * x3[0] + self.dN4dEta * x4[0])
            Jmtx[1, 1] = (self.dN1dEta * x1[1] + self.dN2dEta * x2[1]
                          + self.dN3dEta * x3[1] + self.dN4dEta * x4[1])
            Jmtx[1, 2] = (self.dN1dEta * x1[2] + self.dN2dEta * x2[2]
                          + self.dN3dEta * x3[2] + self.dN4dEta * x4[2])
            Jmtx[2, 0] = (self.dN1dZeta * x1[0] + self.dN2dZeta * x2[0]
                          + self.dN3dZeta * x3[0] + self.dN4dZeta * x4[0])
            Jmtx[2, 1] = (self.dN1dZeta * x1[1] + self.dN2dZeta * x2[1]
                          + self.dN3dZeta * x3[1] + self.dN4dZeta * x4[1])
            Jmtx[2, 2] = (self.dN1dZeta * x1[2] + self.dN2dZeta * x2[2]
                          + self.dN3dZeta * x3[2] + self.dN4dZeta * x4[2])
        else:
            raise RuntimeError("Each argument of shapeFunction.jacobianMatrix "
                               + "must be a tuple of co-ordinates, "
                               + "e.g., (x, y, z).")
        return Jmtx

    def jacobianDeterminant(self, x1, x2, x3, x4):
        Jmtx = self.jacobianMatrix(x1, x2, x3, x4)
        return np.linalg.det(Jmtx)

    def G(self, x1, x2, x3, x4, x01, x02, x03, x04):
        if (isinstance(x1, tuple) and len(x1) == 3
           and isinstance(x2, tuple) and len(x2) == 3
           and isinstance(x3, tuple) and len(x3) == 3
           and isinstance(x4, tuple) and len(x4) == 3
           and isinstance(x01, tuple) and len(x01) == 3
           and isinstance(x02, tuple) and len(x02) == 3
           and isinstance(x03, tuple) and len(x03) == 3
           and isinstance(x04, tuple) and len(x04) == 3):
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
            disGrad[0, 0] = (self.dN1dXi * u1 + self.dN2dXi * u2
                             + self.dN3dXi * u3 + self.dN4dXi * u4)
            disGrad[0, 1] = (self.dN1dEta * u1 + self.dN2dEta * u2
                             + self.dN3dEta * u3 + self.dN4dEta * u4)
            disGrad[0, 2] = (self.dN1dZeta * u1 + self.dN2dZeta * u2
                             + self.dN3dZeta * u3 + self.dN4dZeta * u4)
            disGrad[1, 0] = (self.dN1dXi * v1 + self.dN2dXi * v2
                             + self.dN3dXi * v3 + self.dN4dXi * v4)
            disGrad[1, 1] = (self.dN1dEta * v1 + self.dN2dEta * v2
                             + self.dN3dEta * v3 + self.dN4dEta * v4)
            disGrad[1, 2] = (self.dN1dZeta * v1 + self.dN2dZeta * v2
                             + self.dN3dZeta * v3 + self.dN4dZeta * v4)
            disGrad[2, 0] = (self.dN1dXi * w1 + self.dN2dXi * w2
                             + self.dN3dXi * w3 + self.dN4dXi * w4)
            disGrad[2, 1] = (self.dN1dEta * w1 + self.dN2dEta * w2
                             + self.dN3dEta * w3 + self.dN4dEta * w4)
            disGrad[2, 2] = (self.dN1dZeta * w1 + self.dN2dZeta * w2
                             + self.dN3dZeta * w3 + self.dN4dZeta * w4)

            # determine the current gradient of position
            curGrad = np.transpose(self.jacobianMatrix(x1, x2, x3, x4))
        else:
            raise RuntimeError("Each argument of shapeFunction.G must be "
                               + "a tuple of co-ordinates, e.g., (x, y, z).")
        Gmtx = np.matmul(disGrad, np.linalg.inv(curGrad))
        return Gmtx

    def F(self, x1, x2, x3, x4, x01, x02, x03, x04):
        if (isinstance(x1, tuple) and len(x1) == 3
           and isinstance(x2, tuple) and len(x2) == 3
           and isinstance(x3, tuple) and len(x3) == 3
           and isinstance(x4, tuple) and len(x4) == 3
           and isinstance(x01, tuple) and len(x01) == 3
           and isinstance(x02, tuple) and len(x02) == 3
           and isinstance(x03, tuple) and len(x03) == 3
           and isinstance(x04, tuple) and len(x04) == 3):
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
            disGrad[0, 0] = (self.dN1dXi * u1 + self.dN2dXi * u2
                             + self.dN3dXi * u3 + self.dN4dXi * u4)
            disGrad[0, 1] = (self.dN1dEta * u1 + self.dN2dEta * u2
                             + self.dN3dEta * u3 + self.dN4dEta * u4)
            disGrad[0, 2] = (self.dN1dZeta * u1 + self.dN2dZeta * u2
                             + self.dN3dZeta * u3 + self.dN4dZeta * u4)
            disGrad[1, 0] = (self.dN1dXi * v1 + self.dN2dXi * v2
                             + self.dN3dXi * v3 + self.dN4dXi * v4)
            disGrad[1, 1] = (self.dN1dEta * v1 + self.dN2dEta * v2
                             + self.dN3dEta * v3 + self.dN4dEta * v4)
            disGrad[1, 2] = (self.dN1dZeta * v1 + self.dN2dZeta * v2
                             + self.dN3dZeta * v3 + self.dN4dZeta * v4)
            disGrad[2, 0] = (self.dN1dXi * w1 + self.dN2dXi * w2
                             + self.dN3dXi * w3 + self.dN4dXi * w4)
            disGrad[2, 1] = (self.dN1dEta * w1 + self.dN2dEta * w2
                             + self.dN3dEta * w3 + self.dN4dEta * w4)
            disGrad[2, 2] = (self.dN1dZeta * w1 + self.dN2dZeta * w2
                             + self.dN3dZeta * w3 + self.dN4dZeta * w4)

            # determine the reference gradient of position
            refGrad = np.zeros((3, 3), dtype=float)
            refGrad[0, 0] = (self.dN1dXi * x01[0] + self.dN2dXi * x02[0]
                             + self.dN3dXi * x03[0] + self.dN4dXi * x04[0])
            refGrad[0, 1] = (self.dN1dEta * x01[0] + self.dN2dEta * x02[0]
                             + self.dN3dEta * x03[0] + self.dN4dEta * x04[0])
            refGrad[0, 2] = (self.dN1dZeta * x01[0] + self.dN2dZeta * x02[0]
                             + self.dN3dZeta * x03[0] + self.dN4dZeta * x04[0])
            refGrad[1, 0] = (self.dN1dXi * x01[1] + self.dN2dXi * x02[1]
                             + self.dN3dXi * x03[1] + self.dN4dXi * x04[1])
            refGrad[1, 1] = (self.dN1dEta * x01[1] + self.dN2dEta * x02[1]
                             + self.dN3dEta * x03[1] + self.dN4dEta * x04[1])
            refGrad[1, 2] = (self.dN1dZeta * x01[1] + self.dN2dZeta * x02[1]
                             + self.dN3dZeta * x03[1] + self.dN4dZeta * x04[1])
            refGrad[2, 0] = (self.dN1dXi * x01[2] + self.dN2dXi * x02[2]
                             + self.dN3dXi * x03[2] + self.dN4dXi * x04[2])
            refGrad[2, 1] = (self.dN1dEta * x01[2] + self.dN2dEta * x02[2]
                             + self.dN3dEta * x03[2] + self.dN4dEta * x04[2])
            refGrad[2, 2] = (self.dN1dZeta * x01[2] + self.dN2dZeta * x02[2]
                             + self.dN3dZeta * x03[2] + self.dN4dZeta * x04[2])
        else:
            raise RuntimeError("Each argument of shapeFunction.F must be a "
                               + "tuple of co-ordinates, e.g., (x, y, z).")
        Fmtx = (np.eye(3, dtype=float)
                + np.matmul(disGrad, np.linalg.inv(refGrad)))
        return Fmtx

    def BL(self, x1, x2, x3, x4):
        Jmtx = self.jacobianMatrix(x1, x2, x3, x4)

        BL = np.zeros((7, 12), dtype=float)

        BL[0, 0] = ((self.dN1dXi * Jmtx[0, 0] + self.dN1dEta * Jmtx[0, 1] +
                     self.dN1dZeta * Jmtx[0, 2]) / 3)
        BL[0, 1] = ((self.dN1dXi * Jmtx[1, 0] + self.dN1dEta * Jmtx[1, 1] +
                     self.dN1dZeta * Jmtx[1, 2]) / 3)
        BL[0, 2] = ((self.dN1dXi * Jmtx[2, 0] + self.dN1dEta * Jmtx[2, 1] +
                     self.dN1dZeta * Jmtx[2, 2]) / 3)
        BL[0, 3] = ((self.dN2dXi * Jmtx[0, 0] + self.dN2dEta * Jmtx[0, 1] +
                     self.dN2dZeta * Jmtx[0, 2]) / 3)
        BL[0, 4] = ((self.dN2dXi * Jmtx[1, 0] + self.dN2dEta * Jmtx[1, 1] +
                     self.dN2dZeta * Jmtx[1, 2]) / 3)
        BL[0, 5] = ((self.dN2dXi * Jmtx[2, 0] + self.dN2dEta * Jmtx[2, 1] +
                     self.dN2dZeta * Jmtx[2, 2]) / 3)
        BL[0, 6] = ((self.dN3dXi * Jmtx[0, 0] + self.dN3dEta * Jmtx[0, 1] +
                     self.dN3dZeta * Jmtx[0, 2]) / 3)
        BL[0, 7] = ((self.dN3dXi * Jmtx[1, 0] + self.dN3dEta * Jmtx[1, 1] +
                     self.dN3dZeta * Jmtx[1, 2]) / 3)
        BL[0, 8] = ((self.dN3dXi * Jmtx[2, 0] + self.dN3dEta * Jmtx[2, 1] +
                     self.dN3dZeta * Jmtx[2, 2]) / 3)
        BL[0, 9] = ((self.dN4dXi * Jmtx[0, 0] + self.dN4dEta * Jmtx[0, 1] +
                     self.dN4dZeta * Jmtx[0, 2]) / 3)
        BL[0, 10] = ((self.dN4dXi * Jmtx[1, 0] + self.dN4dEta * Jmtx[1, 1] +
                     self.dN4dZeta * Jmtx[1, 2]) / 3)
        BL[0, 11] = ((self.dN4dXi * Jmtx[2, 0] + self.dN4dEta * Jmtx[2, 1] +
                     self.dN4dZeta * Jmtx[2, 2]) / 3)

        BL[1, 0] = ((self.dN1dXi * Jmtx[0, 0] + self.dN1dEta * Jmtx[0, 1] +
                     self.dN1dZeta * Jmtx[0, 2]) / 3)
        BL[1, 1] = (-(self.dN1dXi * Jmtx[1, 0] + self.dN1dEta * Jmtx[1, 1] +
                    self.dN1dZeta * Jmtx[1, 2]) / 3)
        BL[1, 3] = ((self.dN2dXi * Jmtx[0, 0] + self.dN2dEta * Jmtx[0, 1] +
                     self.dN2dZeta * Jmtx[0, 2]) / 3)
        BL[1, 4] = (-(self.dN2dXi * Jmtx[1, 0] + self.dN2dEta * Jmtx[1, 1] +
                    self.dN2dZeta * Jmtx[1, 2]) / 3)
        BL[1, 6] = ((self.dN3dXi * Jmtx[0, 0] + self.dN3dEta * Jmtx[0, 1] +
                     self.dN3dZeta * Jmtx[0, 2]) / 3)
        BL[1, 7] = (-(self.dN3dXi * Jmtx[1, 0] + self.dN3dEta * Jmtx[1, 1] +
                    self.dN3dZeta * Jmtx[1, 2]) / 3)
        BL[1, 9] = ((self.dN4dXi * Jmtx[0, 0] + self.dN4dEta * Jmtx[0, 1] +
                     self.dN4dZeta * Jmtx[0, 2]) / 3)
        BL[1, 10] = (-(self.dN4dXi * Jmtx[1, 0] + self.dN4dEta * Jmtx[1, 1] +
                     self.dN4dZeta * Jmtx[1, 2]) / 3)

        BL[2, 1] = ((self.dN1dXi * Jmtx[1, 0] + self.dN1dEta * Jmtx[1, 1] +
                     self.dN1dZeta * Jmtx[1, 2]) / 3)
        BL[2, 2] = (-(self.dN1dXi * Jmtx[2, 0] + self.dN1dEta * Jmtx[2, 1] +
                      self.dN1dZeta * Jmtx[2, 2]) / 3)
        BL[2, 4] = ((self.dN2dXi * Jmtx[1, 0] + self.dN2dEta * Jmtx[1, 1] +
                     self.dN2dZeta * Jmtx[1, 2]) / 3)
        BL[2, 5] = (-(self.dN2dXi * Jmtx[2, 0] + self.dN2dEta * Jmtx[2, 1] +
                      self.dN2dZeta * Jmtx[2, 2]) / 3)
        BL[2, 7] = ((self.dN3dXi * Jmtx[1, 0] + self.dN3dEta * Jmtx[1, 1] +
                     self.dN3dZeta * Jmtx[1, 2]) / 3)
        BL[2, 8] = (-(self.dN3dXi * Jmtx[2, 0] + self.dN3dEta * Jmtx[2, 1] +
                      self.dN3dZeta * Jmtx[2, 2]) / 3)
        BL[2, 10] = ((self.dN4dXi * Jmtx[1, 0] + self.dN4dEta * Jmtx[1, 1] +
                     self.dN4dZeta * Jmtx[1, 2]) / 3)
        BL[2, 11] = (-(self.dN4dXi * Jmtx[2, 0] + self.dN4dEta * Jmtx[2, 1] +
                       self.dN4dZeta * Jmtx[2, 2]) / 3)

        BL[3, 0] = (-(self.dN1dXi * Jmtx[0, 0] + self.dN1dEta * Jmtx[0, 1] +
                      self.dN1dZeta * Jmtx[0, 2]) / 3)
        BL[3, 2] = ((self.dN1dXi * Jmtx[2, 0] + self.dN1dEta * Jmtx[2, 1] +
                     self.dN1dZeta * Jmtx[2, 2]) / 3)
        BL[3, 3] = (-(self.dN2dXi * Jmtx[0, 0] + self.dN2dEta * Jmtx[0, 1] +
                      self.dN2dZeta * Jmtx[0, 2]) / 3)
        BL[3, 5] = ((self.dN2dXi * Jmtx[2, 0] + self.dN2dEta * Jmtx[2, 1] +
                     self.dN2dZeta * Jmtx[2, 2]) / 3)
        BL[3, 6] = (-(self.dN3dXi * Jmtx[0, 0] + self.dN3dEta * Jmtx[0, 1] +
                      self.dN3dZeta * Jmtx[0, 2]) / 3)
        BL[3, 8] = ((self.dN3dXi * Jmtx[2, 0] + self.dN3dEta * Jmtx[2, 1] +
                     self.dN3dZeta * Jmtx[2, 2]) / 3)
        BL[3, 9] = (-(self.dN4dXi * Jmtx[0, 0] + self.dN4dEta * Jmtx[0, 1] +
                      self.dN4dZeta * Jmtx[0, 2]) / 3)
        BL[3, 11] = ((self.dN4dXi * Jmtx[2, 0] + self.dN4dEta * Jmtx[2, 1] +
                     self.dN4dZeta * Jmtx[2, 2]) / 3)

        BL[4, 1] = (self.dN1dXi * Jmtx[2, 0] + self.dN1dEta * Jmtx[2, 1] +
                    self.dN1dZeta * Jmtx[2, 2])
        BL[4, 2] = (self.dN1dXi * Jmtx[1, 0] + self.dN1dEta * Jmtx[1, 1] +
                    self.dN1dZeta * Jmtx[1, 2])
        BL[4, 4] = (self.dN2dXi * Jmtx[2, 0] + self.dN2dEta * Jmtx[2, 1] +
                    self.dN2dZeta * Jmtx[2, 2])
        BL[4, 5] = (self.dN2dXi * Jmtx[1, 0] + self.dN2dEta * Jmtx[1, 1] +
                    self.dN2dZeta * Jmtx[1, 2])
        BL[4, 7] = (self.dN3dXi * Jmtx[2, 0] + self.dN3dEta * Jmtx[2, 1] +
                    self.dN3dZeta * Jmtx[2, 2])
        BL[4, 8] = (self.dN3dXi * Jmtx[1, 0] + self.dN3dEta * Jmtx[1, 1] +
                    self.dN3dZeta * Jmtx[1, 2])
        BL[4, 10] = (self.dN4dXi * Jmtx[2, 0] + self.dN4dEta * Jmtx[2, 1] +
                     self.dN4dZeta * Jmtx[2, 2])
        BL[4, 11] = (self.dN4dXi * Jmtx[1, 0] + self.dN4dEta * Jmtx[1, 1] +
                     self.dN4dZeta * Jmtx[1, 2])

        BL[5, 1] = (self.dN1dXi * Jmtx[2, 0] + self.dN1dEta * Jmtx[2, 1] +
                    self.dN1dZeta * Jmtx[2, 2])
        BL[5, 2] = (self.dN1dXi * Jmtx[1, 0] + self.dN1dEta * Jmtx[1, 1] +
                    self.dN1dZeta * Jmtx[1, 2])
        BL[5, 4] = (self.dN2dXi * Jmtx[2, 0] + self.dN2dEta * Jmtx[2, 1] +
                    self.dN2dZeta * Jmtx[2, 2])
        BL[5, 5] = (self.dN2dXi * Jmtx[1, 0] + self.dN2dEta * Jmtx[1, 1] +
                    self.dN2dZeta * Jmtx[1, 2])
        BL[5, 7] = (self.dN3dXi * Jmtx[2, 0] + self.dN3dEta * Jmtx[2, 1] +
                    self.dN3dZeta * Jmtx[2, 2])
        BL[5, 8] = (self.dN3dXi * Jmtx[1, 0] + self.dN3dEta * Jmtx[1, 1] +
                    self.dN3dZeta * Jmtx[1, 2])
        BL[5, 10] = (self.dN4dXi * Jmtx[2, 0] + self.dN4dEta * Jmtx[2, 1] +
                     self.dN4dZeta * Jmtx[2, 2])
        BL[5, 11] = (self.dN4dXi * Jmtx[1, 0] + self.dN4dEta * Jmtx[1, 1] +
                     self.dN4dZeta * Jmtx[1, 2])

        BL[6, 0] = (self.dN1dXi * Jmtx[1, 0] + self.dN1dEta * Jmtx[1, 1] +
                    self.dN1dZeta * Jmtx[1, 2])
        BL[6, 1] = (self.dN1dXi * Jmtx[0, 0] + self.dN1dEta * Jmtx[0, 1] +
                    self.dN1dZeta * Jmtx[0, 2])
        BL[6, 3] = (self.dN2dXi * Jmtx[1, 0] + self.dN2dEta * Jmtx[1, 1] +
                    self.dN2dZeta * Jmtx[1, 2])
        BL[6, 4] = (self.dN2dXi * Jmtx[0, 0] + self.dN2dEta * Jmtx[0, 1] +
                    self.dN2dZeta * Jmtx[0, 2])
        BL[6, 6] = (self.dN3dXi * Jmtx[1, 0] + self.dN3dEta * Jmtx[1, 1] +
                    self.dN3dZeta * Jmtx[1, 2])
        BL[6, 7] = (self.dN3dXi * Jmtx[0, 0] + self.dN3dEta * Jmtx[0, 1] +
                    self.dN3dZeta * Jmtx[0, 2])
        BL[6, 9] = (self.dN4dXi * Jmtx[1, 0] + self.dN4dEta * Jmtx[1, 1] +
                    self.dN4dZeta * Jmtx[1, 2])
        BL[6, 10] = (self.dN4dXi * Jmtx[0, 0] + self.dN4dEta * Jmtx[0, 1] +
                     self.dN4dZeta * Jmtx[0, 2])

        detJ = np.linalg.det(Jmtx)
        BLmtx = BL / detJ
        
        return BLmtx

    def H1(self, x1, x2, x3, x4):
        H1mtx = np.zeros((3, 12), dtype=float)
        BLmtx = self.BL(x1, x2, x3, x4)

        # create the H1 matrix by differentiation of shape functions.
        H1mtx[0, 0] = 3 * BLmtx[0, 0]
        H1mtx[0, 3] = 3 * BLmtx[0, 3]
        H1mtx[0, 6] = 3 * BLmtx[0, 6]
        H1mtx[0, 9] = 3 * BLmtx[0, 9]

        H1mtx[1, 1] = 3 * BLmtx[0, 1]
        H1mtx[1, 4] = 3 * BLmtx[0, 4]
        H1mtx[1, 7] = 3 * BLmtx[0, 7]
        H1mtx[1, 10] = 3 * BLmtx[0, 10]

        H1mtx[2, 2] = 3 * BLmtx[0, 2]
        H1mtx[2, 5] = 3 * BLmtx[0, 5]
        H1mtx[2, 8] = 3 * BLmtx[0, 8]
        H1mtx[2, 11] = 3 * BLmtx[0, 11]

        return H1mtx

    def H2(self, x1, x2, x3, x4):
        H2mtx = np.zeros((3, 12), dtype=float)
        BLmtx = self.BL(x1, x2, x3, x4)

        # create the H2 matrix by differentiation of shape functions.
        H2mtx[0, 0] = 3 * BLmtx[0, 2]
        H2mtx[0, 3] = 3 * BLmtx[0, 5]
        H2mtx[0, 6] = 3 * BLmtx[0, 8]
        H2mtx[0, 9] = 3 * BLmtx[0, 11]

        H2mtx[1, 1] = 3 * BLmtx[0, 2]
        H2mtx[1, 4] = 3 * BLmtx[0, 5]
        H2mtx[1, 7] = 3 * BLmtx[0, 8]
        H2mtx[1, 10] = 3 * BLmtx[0, 11]

        H2mtx[2, 2] = 3 * BLmtx[0, 1]
        H2mtx[2, 5] = 3 * BLmtx[0, 4]
        H2mtx[2, 8] = 3 * BLmtx[0, 7]
        H2mtx[2, 11] = 3 * BLmtx[0, 10]

        return H2mtx

    def H3(self, x1, x2, x3, x4):
        H3mtx = np.zeros((3, 12), dtype=float)
        BLmtx = self.BL(x1, x2, x3, x4)

        # create the H3 matrix by differentiation of shape functions.
        H3mtx[0, 0] = 3 * BLmtx[0, 1]
        H3mtx[0, 3] = 3 * BLmtx[0, 4]
        H3mtx[0, 6] = 3 * BLmtx[0, 7]
        H3mtx[0, 9] = 3 * BLmtx[0, 10]

        H3mtx[1, 1] = 3 * BLmtx[0, 2]
        H3mtx[1, 4] = 3 * BLmtx[0, 5]
        H3mtx[1, 7] = 3 * BLmtx[0, 8]
        H3mtx[1, 10] = 3 * BLmtx[0, 11]

        H3mtx[2, 2] = 3 * BLmtx[0, 0]
        H3mtx[2, 5] = 3 * BLmtx[0, 3]
        H3mtx[2, 8] = 3 * BLmtx[0, 6]
        H3mtx[2, 11] = 3 * BLmtx[0, 9]

        return H3mtx

    def H4(self, x1, x2, x3, x4):
        H4mtx = np.zeros((3, 12), dtype=float)
        BLmtx = self.BL(x1, x2, x3, x4)

        # create the H3 matrix by differentiation of shape functions.
        H4mtx[0, 0] = 3 * BLmtx[0, 0]
        H4mtx[0, 3] = 3 * BLmtx[0, 3]
        H4mtx[0, 6] = 3 * BLmtx[0, 6]
        H4mtx[0, 9] = 3 * BLmtx[0, 9]

        H4mtx[1, 1] = 3 * BLmtx[0, 0]
        H4mtx[1, 4] = 3 * BLmtx[0, 3]
        H4mtx[1, 7] = 3 * BLmtx[0, 6]
        H4mtx[1, 10] = 3 * BLmtx[0, 9]

        H4mtx[2, 2] = 3 * BLmtx[0, 1]
        H4mtx[2, 5] = 3 * BLmtx[0, 4]
        H4mtx[2, 8] = 3 * BLmtx[0, 7]
        H4mtx[2, 11] = 3 * BLmtx[0, 10]

        return H4mtx

    def H5(self, x1, x2, x3, x4):
        H5mtx = np.zeros((3, 12), dtype=float)
        BLmtx = self.BL(x1, x2, x3, x4)

        # create the H3 matrix by differentiation of shape functions.
        H5mtx[0, 0] = 3 * BLmtx[0, 2]
        H5mtx[0, 3] = 3 * BLmtx[0, 5]
        H5mtx[0, 6] = 3 * BLmtx[0, 8]
        H5mtx[0, 9] = 3 * BLmtx[0, 11]

        H5mtx[1, 1] = 3 * BLmtx[0, 1]
        H5mtx[1, 4] = 3 * BLmtx[0, 4]
        H5mtx[1, 7] = 3 * BLmtx[0, 7]
        H5mtx[1, 10] = 3 * BLmtx[0, 10]

        H5mtx[2, 2] = 3 * BLmtx[0, 0]
        H5mtx[2, 5] = 3 * BLmtx[0, 3]
        H5mtx[2, 8] = 3 * BLmtx[0, 6]
        H5mtx[2, 11] = 3 * BLmtx[0, 9]
        
        return H5mtx


    def A1(self, x1, x2, x3, x4, x01, x02, x03, x04):
        A1mtx = np.zeros((7, 3), dtype=float)
        Gmtx = self.G(x1, x2, x3, x4, x01, x02, x03, x04)

        # create the A1 matrix from nonlinear part of strain
        A1mtx[0, 0] = - Gmtx[0, 0] / 3
        A1mtx[0, 1] = - Gmtx[1, 1] / 3
        A1mtx[0, 2] = - Gmtx[2, 2] / 3
        A1mtx[1, 0] = - Gmtx[0, 0] / 3
        A1mtx[1, 1] = Gmtx[1, 1] / 3
        A1mtx[2, 1] = - Gmtx[1, 1] / 3
        A1mtx[2, 2] = Gmtx[2, 2] / 3
        A1mtx[3, 0] = Gmtx[0, 0] / 3
        A1mtx[3, 2] = - Gmtx[2, 2] / 3
        A1mtx[4, 1] = - 2 * Gmtx[1, 2] 
        A1mtx[4, 2] = 2 * Gmtx[2, 1]
        A1mtx[5, 1] = 2 * Gmtx[1, 2] 
        A1mtx[5, 2] = 2 * Gmtx[2, 1]
        A1mtx[6, 0] = - 2 * Gmtx[0, 1]
        A1mtx[6, 1] = 2 * Gmtx[1, 0]
        
        return A1mtx

    def A2(self, x1, x2, x3, x4, x01, x02, x03, x04):
        A2mtx = np.zeros((7, 3), dtype=float)
        Gmtx = self.G(x1, x2, x3, x4, x01, x02, x03, x04)

        # create the A2 matrix from nonlinear part of strain
        A2mtx[0, 0] = Gmtx[0, 2] / 3
        A2mtx[0, 1] = - Gmtx[1, 2] / 3
        A2mtx[0, 2] = - Gmtx[2, 1] / 3
        A2mtx[1, 2] = - Gmtx[2, 1] / 3
        A2mtx[2, 0] = - Gmtx[0, 2] / 3
        A2mtx[2, 1] = Gmtx[1, 2] / 3
        A2mtx[2, 2] = Gmtx[2, 1] 
        A2mtx[3, 0] = Gmtx[0, 2] / 3
        A2mtx[3, 1] = - Gmtx[1, 2] / 3
        A2mtx[3, 2] = - 2 * Gmtx[2, 1] / 3
        A2mtx[5, 0] = 2 * Gmtx[0, 1] 
        A2mtx[6, 2] = 2 * Gmtx[2, 0]
        
        return A2mtx

    def A3(self, x1, x2, x3, x4, x01, x02, x03, x04):
        A3mtx = np.zeros((7, 3), dtype=float)
        Gmtx = self.G(x1, x2, x3, x4, x01, x02, x03, x04)

        # create the A3 matrix from nonlinear part of strain
        A3mtx[0, 0] = - 2 * Gmtx[1, 0] / 3
        A3mtx[0, 1] = - 4 * Gmtx[2, 1] / 3
        A3mtx[0, 2] = Gmtx[2, 0] / 3
        A3mtx[1, 0] = 2 * Gmtx[1, 0] / 3
        A3mtx[1, 2] = Gmtx[2, 0] / 3
        A3mtx[2, 0] = - 2 * Gmtx[1, 0] / 3
        A3mtx[2, 1] = 4 * Gmtx[2, 1] / 3
        A3mtx[3, 1] = - 4 * Gmtx[2, 1] / 3
        A3mtx[3, 2] = - Gmtx[2, 0] / 3
        
        return A3mtx

    def A4(self, x1, x2, x3, x4, x01, x02, x03, x04):
        A4mtx = np.zeros((7, 3), dtype=float)
        Gmtx = self.G(x1, x2, x3, x4, x01, x02, x03, x04)

        # create the A4 matrix from nonlinear part of strain
        A4mtx[1, 1] = 2 * Gmtx[1, 0] / 3
        A4mtx[2, 1] = - Gmtx[1, 0] / 3
        A4mtx[3, 1] = - Gmtx[1, 0] / 3
        A4mtx[4, 0] = 4 * Gmtx[1, 2] 
        A4mtx[4, 2] = 4 * Gmtx[0, 0] 
        A4mtx[5, 0] = - 4 * Gmtx[1, 2] 
        A4mtx[5, 2] = - 4 * Gmtx[0, 0] 
        A4mtx[6, 0] = - 4 * Gmtx[1, 0] 
        
        return A4mtx

    def A5(self, x1, x2, x3, x4, x01, x02, x03, x04):
        A5mtx = np.zeros((7, 3), dtype=float)
        Gmtx = self.G(x1, x2, x3, x4, x01, x02, x03, x04)

        # create the A5 matrix from nonlinear part of strain
        A5mtx[4, 0] = - 2 * Gmtx[1, 0] 
        A5mtx[4, 1] = - 4 * Gmtx[2, 1] 
        A5mtx[4, 2] = - 2 * Gmtx[1, 0] 
        
        return A5mtx

    # derivative of shape functions in contribution to the first nonlinear 
    # strain
    def L1(self, x1, x2, x3, x4):
        B = self.BL(x1, x2, x3, x4)
        L1 = np.zeros((7, 12), dtype=float)

        L1[0, 0] = - B[0, 0]
        L1[0, 1] = - B[0, 1]
        L1[0, 2] = - B[0, 2]
        L1[0, 3] = - B[0, 3]
        L1[0, 4] = - B[0, 4]
        L1[0, 5] = - B[0, 5]
        L1[0, 6] = - B[0, 6]
        L1[0, 7] = - B[0, 7]
        L1[0, 8] = - B[0, 8]
        L1[0, 9] = - B[0, 9]
        L1[0, 10] = - B[0, 10]
        L1[0, 11] = - B[0, 11]
        
        L1[1, 0] = - B[1, 0]
        L1[1, 1] = - B[1, 1]
        L1[1, 3] = - B[1, 3]
        L1[1, 4] = - B[1, 4]
        L1[1, 6] = - B[1, 6]
        L1[1, 7] = - B[1, 7]
        L1[1, 9] = - B[1, 9]
        L1[1, 10] = - B[1, 10]

        L1[2, 1] = - B[2, 1]
        L1[2, 2] = - B[2, 2]
        L1[2, 4] = - B[2, 4]
        L1[2, 5] = - B[2, 5]
        L1[2, 7] = - B[2, 7]
        L1[2, 8] = - B[2, 8]
        L1[2, 10] = - B[2, 10]
        L1[2, 11] = - B[2, 11]

        L1[3, 0] = - B[3, 0]
        L1[3, 2] = - B[3, 2]
        L1[3, 3] = - B[3, 3]
        L1[3, 5] = - B[3, 5]
        L1[3, 6] = - B[3, 6]
        L1[3, 8] = - B[3, 8]
        L1[3, 9] = - B[3, 9]
        L1[3, 11] = - B[3, 11]
        
        L1[4, 1] = - 2 * B[4, 1]
        L1[4, 2] = 2 * B[4, 2]
        L1[4, 4] = - 2 * B[4, 4]
        L1[4, 5] = 2 * B[4, 5]
        L1[4, 7] = - 2 * B[4, 7]
        L1[4, 8] = 2 * B[4, 8]
        L1[4, 10] = - 2 * B[4, 10]
        L1[4, 11] = 2 * B[4, 11]

        L1[5, 1] = 2 * B[5, 1]
        L1[5, 2] = 2 * B[5, 2]
        L1[5, 4] = 2 * B[5, 4]
        L1[5, 5] = 2 * B[5, 5]
        L1[5, 7] = 2 * B[5, 7]
        L1[5, 8] = 2 * B[5, 8]
        L1[5, 10] = 2 * B[5, 10]
        L1[5, 11] = 2 * B[5, 11]

        L1[6, 0] = - 2 * B[6, 0]
        L1[6, 1] = 2 * B[6, 1]
        L1[6, 3] = - 2 * B[6, 3]
        L1[6, 4] = 2 * B[6, 4]
        L1[6, 6] = - 2 * B[6, 6]
        L1[6, 7] = 2 * B[6, 7]
        L1[6, 9] = - 2 * B[6, 9]
        L1[6, 10] = 2 * B[6, 10]
        
        return L1

    # derivative of shape functions in contribution to the second nonlinear 
    # strain
    def L2(self, x1, x2, x3, x4):
        B = self.BL(x1, x2, x3, x4)
        L2 = np.zeros((7, 12), dtype=float)

        L2[0, 0] = B[0, 2]
        L2[0, 1] = - B[0, 2]
        L2[0, 2] = - B[0, 1]
        L2[0, 3] = B[0, 5]
        L2[0, 4] = - B[0, 5]
        L2[0, 5] = - B[0, 4]
        L2[0, 6] = B[0, 8]
        L2[0, 7] = - B[0, 8]
        L2[0, 8] = - B[0, 7]
        L2[0, 9] = B[0, 11]
        L2[0, 10] = - B[0, 11]
        L2[0, 11] = - B[0, 10]
        
        L2[1, 2] = B[1, 1]
        L2[1, 5] = B[1, 4]
        L2[1, 8] = B[1, 7]
        L2[1, 11] = B[1, 10]

        L2[2, 0] = B[2, 2]
        L2[2, 1] = - B[2, 2]
        L2[2, 2] = 3 * B[2, 1]
        L2[2, 3] = B[2, 5]
        L2[2, 4] = - B[2, 5]
        L2[2, 5] = 3 * B[2, 4]
        L2[2, 6] = B[2, 8]
        L2[2, 7] = - B[2, 8]
        L2[2, 8] = 3 * B[2, 7]
        L2[2, 9] = B[2, 11]
        L2[2, 10] = - B[2, 11]
        L2[2, 11] = 3 * B[2, 10]

        L2[3, 0] = B[3, 2]
        L2[3, 1] = - B[3, 2]
        L2[3, 2] = - 2 * B[2, 2] / 3
        L2[3, 3] = B[3, 5]
        L2[3, 4] = - B[3, 5]
        L2[3, 5] = - 2 * B[2, 5] / 3
        L2[3, 6] = B[3, 8]
        L2[3, 7] = - B[3, 8]
        L2[3, 8] = - 2 * B[2, 8] / 3
        L2[3, 9] = B[3, 11]
        L2[3, 10] = - B[3, 11]
        L2[3, 11] = - 2 * B[2, 11] 

        L2[5, 0] = 2 * B[5, 2]
        L2[5, 3] = 2 * B[5, 5]
        L2[5, 6] = 2 * B[5, 8]
        L2[5, 9] = 2 * B[5, 11]

        L2[6, 2] = 2 * B[6, 1]
        L2[6, 5] = 2 * B[6, 4]
        L2[6, 8] = 2 * B[6, 7]
        L2[6, 11] = 2 * B[6, 10]
       
        return L2
   
    # derivative of shape functions in contribution to the third nonlinear 
    # strain
    def L3(self, x1, x2, x3, x4):
        B = self.BL(x1, x2, x3, x4)
        L3 = np.zeros((7, 12), dtype=float)

        L3[0, 0] = -2 * B[0, 0]
        L3[0, 1] = - 4 * B[0,1]
        L3[0, 2] = B[0, 0]
        L3[0, 3] = -2 * B[0, 3]
        L3[0, 4] = - 4 * B[0, 4]
        L3[0, 5] = B[0, 3]
        L3[0, 6] = -2 * B[0, 6]
        L3[0, 7] = - 4 * B[0, 7]
        L3[0, 8] = B[0, 6]
        L3[0, 9] = -2 * B[0, 9]
        L3[0, 10] = - 4 * B[0, 10]
        L3[0, 11] = B[0, 9]
        
        L3[1, 0] = 2 * B[1, 0]
        L3[1, 2] = B[1, 0]
        L3[1, 3] = 2 * B[1, 3]
        L3[1, 5] = B[1, 3]
        L3[1, 6] = 2 * B[1, 6]
        L3[1, 8] = B[1, 6]
        L3[1, 9] = 2 * B[1, 9]
        L3[1, 11] = B[1, 9]

        L3[2, 0] = - B[1, 0]
        L3[2, 1] = 4 * B[2, 1]
        L3[2, 3] = - B[1, 3]
        L3[2, 4] = 4 * B[2, 4]
        L3[2, 6] = - B[1, 6]
        L3[2, 7] = 4 * B[2, 7]
        L3[2, 9] = - B[1, 9]
        L3[2, 10] = 4 * B[2, 10]

        L3[3, 1] = - B[2, 1]
        L3[3, 2] = B[3, 0]
        L3[3, 4] = - B[2, 4]
        L3[3, 5] = B[3, 3]
        L3[3, 7] = - B[2, 7]
        L3[3, 8] = B[3, 6]
        L3[3, 10] = - B[2, 10]
        L3[3, 11] = B[3, 9] 
       
        return L3
 
    # derivative of shape functions in contribution to the forth nonlinear 
    # strain
    def L4(self, x1, x2, x3, x4):
        B = self.BL(x1, x2, x3, x4)
        L4 = np.zeros((7, 12), dtype=float)
        
        L4[1, 1] = 2 * B[1, 0]
        L4[1, 4] = 2 * B[1, 3]
        L4[1, 7] = 2 * B[1, 6]
        L4[1, 10] = 2 * B[1, 9]

        L4[2, 1] = - B[1, 0]
        L4[2, 4] = - B[1, 3]
        L4[2, 7] = - B[1, 6]
        L4[2, 10] = - B[1, 9]

        L4[3, 1] = B[3, 0]
        L4[3, 4] = B[3, 3]
        L4[3, 7] = B[3, 6]
        L4[3, 10] = B[3, 9]

        L4[4, 0] = 4 * B[4, 1]
        L4[4, 2] = 4 * B[6, 1]
        L4[4, 3] = 4 * B[4, 4]
        L4[4, 5] = 4 * B[6, 4]
        L4[4, 6] = 4 * B[4, 7]
        L4[4, 8] = 4 * B[4, 7]
        L4[4, 9] = 4 * B[4, 10]
        L4[4, 11] = 4 * B[4, 10]

        L4[5, 0] = - 4 * B[5, 1]
        L4[5, 2] = - 4 * B[6, 1]
        L4[5, 3] = - 4 * B[5, 4]
        L4[5, 5] = - 4 * B[6, 4]
        L4[5, 6] = - 4 * B[5, 7]
        L4[5, 8] = - 4 * B[5, 7]
        L4[5, 9] = - 4 * B[5, 10]
        L4[5, 11] = - 4 * B[5, 10]
        
        L4[6, 0] = - 4 * B[6, 1]
        L4[6, 3] = - 4 * B[6, 4]
        L4[6, 6] = - 4 * B[6, 7]
        L4[6, 9] = - 4 * B[6, 10]
       
        return L4

    # derivative of shape functions in contribution to the fifth nonlinear 
    # strain
    def L5(self, x1, x2, x3, x4):
        B = self.BL(x1, x2, x3, x4)
        L5 = np.zeros((7, 12), dtype=float)

        L5[4, 0] = - 2 * B[6, 1]
        L5[4, 1] = - 4 * B[4, 2]
        L5[4, 2] = - 2 * B[6, 1]
        L5[4, 3] = - 2 * B[6, 4]
        L5[4, 4] = - 4 * B[4, 5]
        L5[4, 5] = - 2 * B[6, 4]
        L5[4, 6] = - 2 * B[6, 7]
        L5[4, 7] = - 4 * B[4, 8]
        L5[4, 8] = - 2 * B[4, 7]
        L5[4, 9] = - 2 * B[4, 10]
        L5[4, 10] = - 4 * B[4, 11]
        L5[4, 11] = - 2 * B[4, 10]
       
        return L5
    
    # first nonlinear contribution to strain displacement matrix    
    def BN1(self, x1, x2, x3, x4, x01, x02, x03, x04):
        A1mtx = self.A1(x1, x2, x3, x4, x01, x02, x03, x04)
        H1mtx = self.H1(x1, x2, x3, x4)

        B1Nmtx = np.zeros((7, 12), dtype=float)
        B1Nmtx = np.matmul(A1mtx, H1mtx)

        return B1Nmtx

    # second nonlinear contribution to strain displacement matrix    
    def BN2(self, x1, x2, x3, x4, x01, x02, x03, x04):
        A2mtx = self.A2(x1, x2, x3, x4, x01, x02, x03, x04)
        H2mtx = self.H2(x1, x2, x3, x4)

        B2Nmtx = np.zeros((7, 12), dtype=float)
        B2Nmtx = np.matmul(A2mtx, H2mtx)

        return B2Nmtx

    # third nonlinear contribution to strain displacement matrix    
    def BN3(self, x1, x2, x3, x4, x01, x02, x03, x04):
        A3mtx = self.A3(x1, x2, x3, x4, x01, x02, x03, x04)
        H3mtx = self.H3(x1, x2, x3, x4)

        B3Nmtx = np.zeros((7, 12), dtype=float)
        B3Nmtx = np.matmul(A3mtx, H3mtx)

        return B3Nmtx

    # forth nonlinear contribution to strain displacement matrix    
    def BN4(self, x1, x2, x3, x4, x01, x02, x03, x04):
        A4mtx = self.A4(x1, x2, x3, x4, x01, x02, x03, x04)
        H4mtx = self.H4(x1, x2, x3, x4)

        B4Nmtx = np.zeros((7, 12), dtype=float)
        B4Nmtx = np.matmul(A4mtx, H4mtx)

        return B4Nmtx
    
    # fifth nonlinear contribution to strain displacement matrix    
    def BN5(self, x1, x2, x3, x4, x01, x02, x03, x04):
        A5mtx = self.A5(x1, x2, x3, x4, x01, x02, x03, x04)
        H5mtx = self.H5(x1, x2, x3, x4)

        B5Nmtx = np.zeros((7, 12), dtype=float)
        B5Nmtx = np.matmul(A5mtx, H5mtx)

        return B5Nmtx    
    
"""
Changes made in version "1.0.0":

Original version
"""
