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
__update__ = "07-17-2020"
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
        self.Nmatx = np.array([[self.N1, 0.0, 0.0, self.N2, 0.0, 0.0,
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
            curGrad = np.transpose(self.jacobianMtx(x1, x2, x3, x4))
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

    def BLinear(self, x1, x2, x3, x4):
        Jmtx = self.jacobianMatrix(x1, x2, x3, x4)
        IJmtx = np.zeros((3, 3), dtype=float)
        IJmtx = np.linalg.inv(Jmtx)

        BL = np.zeros((7, 12), dtype=float)

        BL[0, 0] = ((self.dN1dXi * IJmtx[0, 0] + self.dN1dEta * IJmtx[0, 1] +
                     self.dN1dZeta * IJmtx[0, 2]) / 3)
        BL[0, 1] = ((self.dN1dXi * IJmtx[1, 0] + self.dN1dEta * IJmtx[1, 1] +
                     self.dN1dZeta * IJmtx[1, 2]) / 3)
        BL[0, 2] = ((self.dN1dXi * IJmtx[2, 0] + self.dN1dEta * IJmtx[2, 1] +
                     self.dN1dZeta * IJmtx[2, 2]) / 3)
        BL[0, 3] = ((self.dN2dXi * IJmtx[0, 0] + self.dN2dEta * IJmtx[0, 1] +
                     self.dN2dZeta * IJmtx[0, 2]) / 3)
        BL[0, 4] = ((self.dN2dXi * IJmtx[1, 0] + self.dN2dEta * IJmtx[1, 1] +
                     self.dN2dZeta * IJmtx[1, 2]) / 3)
        BL[0, 5] = ((self.dN2dXi * IJmtx[2, 0] + self.dN2dEta * IJmtx[2, 1] +
                     self.dN2dZeta * IJmtx[2, 2]) / 3)
        BL[0, 6] = ((self.dN3dXi * IJmtx[0, 0] + self.dN3dEta * IJmtx[0, 1] +
                     self.dN3dZeta * IJmtx[0, 2]) / 3)
        BL[0, 7] = ((self.dN3dXi * IJmtx[1, 0] + self.dN3dEta * IJmtx[1, 1] +
                     self.dN3dZeta * IJmtx[1, 2]) / 3)
        BL[0, 8] = ((self.dN3dXi * IJmtx[2, 0] + self.dN3dEta * IJmtx[2, 1] +
                     self.dN3dZeta * IJmtx[2, 2]) / 3)
        BL[0, 9] = ((self.dN4dXi * IJmtx[0, 0] + self.dN4dEta * IJmtx[0, 1] +
                     self.dN4dZeta * IJmtx[0, 2]) / 3)
        BL[0, 10] = ((self.dN4dXi * IJmtx[1, 0] + self.dN4dEta * IJmtx[1, 1] +
                     self.dN4dZeta * IJmtx[1, 2]) / 3)
        BL[0, 11] = ((self.dN4dXi * IJmtx[2, 0] + self.dN4dEta * IJmtx[2, 1] +
                     self.dN4dZeta * IJmtx[2, 2]) / 3)

        BL[1, 0] = ((self.dN1dXi * IJmtx[0, 0] + self.dN1dEta * IJmtx[0, 1] +
                     self.dN1dZeta * IJmtx[0, 2]) / 3)
        BL[1, 1] = (-(self.dN1dXi * IJmtx[1, 0] + self.dN1dEta * IJmtx[1, 1] +
                    self.dN1dZeta * IJmtx[1, 2]) / 3)
        BL[1, 3] = ((self.dN2dXi * IJmtx[0, 0] + self.dN2dEta * IJmtx[0, 1] +
                     self.dN2dZeta * IJmtx[0, 2]) / 3)
        BL[1, 4] = (-(self.dN2dXi * IJmtx[1, 0] + self.dN2dEta * IJmtx[1, 1] +
                    self.dN2dZeta * IJmtx[1, 2]) / 3)
        BL[1, 6] = ((self.dN3dXi * IJmtx[0, 0] + self.dN3dEta * IJmtx[0, 1] +
                     self.dN3dZeta * IJmtx[0, 2]) / 3)
        BL[1, 7] = (-(self.dN3dXi * IJmtx[1, 0] + self.dN3dEta * IJmtx[1, 1] +
                    self.dN3dZeta * IJmtx[1, 2]) / 3)
        BL[1, 9] = ((self.dN4dXi * IJmtx[0, 0] + self.dN4dEta * IJmtx[0, 1] +
                     self.dN4dZeta * IJmtx[0, 2]) / 3)
        BL[1, 10] = (-(self.dN4dXi * IJmtx[1, 0] + self.dN4dEta * IJmtx[1, 1] +
                     self.dN4dZeta * IJmtx[1, 2]) / 3)

        BL[2, 1] = ((self.dN1dXi * IJmtx[1, 0] + self.dN1dEta * IJmtx[1, 1] +
                     self.dN1dZeta * IJmtx[1, 2]) / 3)
        BL[2, 2] = (-(self.dN1dXi * IJmtx[2, 0] + self.dN1dEta * IJmtx[2, 1] +
                      self.dN1dZeta * IJmtx[2, 2]) / 3)
        BL[2, 4] = ((self.dN2dXi * IJmtx[1, 0] + self.dN2dEta * IJmtx[1, 1] +
                     self.dN2dZeta * IJmtx[1, 2]) / 3)
        BL[2, 5] = (-(self.dN2dXi * IJmtx[2, 0] + self.dN2dEta * IJmtx[2, 1] +
                      self.dN2dZeta * IJmtx[2, 2]) / 3)
        BL[2, 7] = ((self.dN3dXi * IJmtx[1, 0] + self.dN3dEta * IJmtx[1, 1] +
                     self.dN3dZeta * IJmtx[1, 2]) / 3)
        BL[2, 8] = (-(self.dN3dXi * IJmtx[2, 0] + self.dN3dEta * IJmtx[2, 1] +
                      self.dN3dZeta * IJmtx[2, 2]) / 3)
        BL[2, 10] = ((self.dN4dXi * IJmtx[1, 0] + self.dN4dEta * IJmtx[1, 1] +
                     self.dN4dZeta * IJmtx[1, 2]) / 3)
        BL[2, 11] = (-(self.dN4dXi * IJmtx[2, 0] + self.dN4dEta * IJmtx[2, 1] +
                       self.dN4dZeta * IJmtx[2, 2]) / 3)

        BL[3, 0] = (-(self.dN1dXi * IJmtx[0, 0] + self.dN1dEta * IJmtx[0, 1] +
                      self.dN1dZeta * IJmtx[0, 2]) / 3)
        BL[3, 2] = ((self.dN1dXi * IJmtx[2, 0] + self.dN1dEta * IJmtx[2, 1] +
                     self.dN1dZeta * IJmtx[2, 2]) / 3)
        BL[3, 3] = (-(self.dN2dXi * IJmtx[0, 0] + self.dN2dEta * IJmtx[0, 1] +
                      self.dN2dZeta * IJmtx[0, 2]) / 3)
        BL[3, 5] = ((self.dN2dXi * IJmtx[2, 0] + self.dN2dEta * IJmtx[2, 1] +
                     self.dN2dZeta * IJmtx[2, 2]) / 3)
        BL[3, 6] = (-(self.dN3dXi * IJmtx[0, 0] + self.dN3dEta * IJmtx[0, 1] +
                      self.dN3dZeta * IJmtx[0, 2]) / 3)
        BL[3, 8] = ((self.dN3dXi * IJmtx[2, 0] + self.dN3dEta * IJmtx[2, 1] +
                     self.dN3dZeta * IJmtx[2, 2]) / 3)
        BL[3, 9] = (-(self.dN4dXi * IJmtx[0, 0] + self.dN4dEta * IJmtx[0, 1] +
                      self.dN4dZeta * IJmtx[0, 2]) / 3)
        BL[3, 11] = ((self.dN4dXi * IJmtx[2, 0] + self.dN4dEta * IJmtx[2, 1] +
                     self.dN4dZeta * IJmtx[2, 2]) / 3)

        BL[4, 1] = (self.dN1dXi * IJmtx[2, 0] + self.dN1dEta * IJmtx[2, 1] +
                    self.dN1dZeta * IJmtx[2, 2])
        BL[4, 2] = (self.dN1dXi * IJmtx[1, 0] + self.dN1dEta * IJmtx[1, 1] +
                    self.dN1dZeta * IJmtx[1, 2])
        BL[4, 4] = (self.dN2dXi * IJmtx[2, 0] + self.dN2dEta * IJmtx[2, 1] +
                    self.dN2dZeta * IJmtx[2, 2])
        BL[4, 5] = (self.dN2dXi * IJmtx[1, 0] + self.dN2dEta * IJmtx[1, 1] +
                    self.dN2dZeta * IJmtx[1, 2])
        BL[4, 7] = (self.dN3dXi * IJmtx[2, 0] + self.dN3dEta * IJmtx[2, 1] +
                    self.dN3dZeta * IJmtx[2, 2])
        BL[4, 8] = (self.dN3dXi * IJmtx[1, 0] + self.dN3dEta * IJmtx[1, 1] +
                    self.dN3dZeta * IJmtx[1, 2])
        BL[4, 10] = (self.dN4dXi * IJmtx[2, 0] + self.dN4dEta * IJmtx[2, 1] +
                     self.dN4dZeta * IJmtx[2, 2])
        BL[4, 11] = (self.dN4dXi * IJmtx[1, 0] + self.dN4dEta * IJmtx[1, 1] +
                     self.dN4dZeta * IJmtx[1, 2])

        BL[5, 1] = (self.dN1dXi * IJmtx[2, 0] + self.dN1dEta * IJmtx[2, 1] +
                    self.dN1dZeta * IJmtx[2, 2])
        BL[5, 2] = (self.dN1dXi * IJmtx[1, 0] + self.dN1dEta * IJmtx[1, 1] +
                    self.dN1dZeta * IJmtx[1, 2])
        BL[5, 4] = (self.dN2dXi * IJmtx[2, 0] + self.dN2dEta * IJmtx[2, 1] +
                    self.dN2dZeta * IJmtx[2, 2])
        BL[5, 5] = (self.dN2dXi * IJmtx[1, 0] + self.dN2dEta * IJmtx[1, 1] +
                    self.dN2dZeta * IJmtx[1, 2])
        BL[5, 7] = (self.dN3dXi * IJmtx[2, 0] + self.dN3dEta * IJmtx[2, 1] +
                    self.dN3dZeta * IJmtx[2, 2])
        BL[5, 8] = (self.dN3dXi * IJmtx[1, 0] + self.dN3dEta * IJmtx[1, 1] +
                    self.dN3dZeta * IJmtx[1, 2])
        BL[5, 10] = (self.dN4dXi * IJmtx[2, 0] + self.dN4dEta * IJmtx[2, 1] +
                     self.dN4dZeta * IJmtx[2, 2])
        BL[5, 11] = (self.dN4dXi * IJmtx[1, 0] + self.dN4dEta * IJmtx[1, 1] +
                     self.dN4dZeta * IJmtx[1, 2])

        BL[6, 0] = (self.dN1dXi * IJmtx[1, 0] + self.dN1dEta * IJmtx[1, 1] +
                    self.dN1dZeta * IJmtx[1, 2])
        BL[6, 1] = (self.dN1dXi * IJmtx[0, 0] + self.dN1dEta * IJmtx[0, 1] +
                    self.dN1dZeta * IJmtx[0, 2])
        BL[6, 3] = (self.dN2dXi * IJmtx[1, 0] + self.dN2dEta * IJmtx[1, 1] +
                    self.dN2dZeta * IJmtx[1, 2])
        BL[6, 4] = (self.dN2dXi * IJmtx[0, 0] + self.dN2dEta * IJmtx[0, 1] +
                    self.dN2dZeta * IJmtx[0, 2])
        BL[6, 6] = (self.dN3dXi * IJmtx[1, 0] + self.dN3dEta * IJmtx[1, 1] +
                    self.dN3dZeta * IJmtx[1, 2])
        BL[6, 7] = (self.dN3dXi * IJmtx[0, 0] + self.dN3dEta * IJmtx[0, 1] +
                    self.dN3dZeta * IJmtx[0, 2])
        BL[6, 9] = (self.dN4dXi * IJmtx[1, 0] + self.dN4dEta * IJmtx[1, 1] +
                    self.dN4dZeta * IJmtx[1, 2])
        BL[6, 10] = (self.dN4dXi * IJmtx[0, 0] + self.dN4dEta * IJmtx[0, 1] +
                     self.dN4dZeta * IJmtx[0, 2])

        return BL

    def HMatrixF(self, x1, x2, x3, x4):
        HF = np.zeros((3, 12), dtype=float)
        BL = self.BLinear(x1, x2, x3, x4)

        # create the H1 matrix by differentiation of shape functions.
        HF[0, 0] = 3 * BL[0, 0]
        HF[0, 3] = 3 * BL[0, 3]
        HF[0, 6] = 3 * BL[0, 6]
        HF[0, 9] = 3 * BL[0, 9]

        HF[1, 1] = 3 * BL[0, 0]
        HF[1, 4] = 3 * BL[0, 3]
        HF[1, 7] = 3 * BL[0, 6]
        HF[1, 10] = 3 * BL[0, 9]

        HF[2, 2] = 3 * BL[0, 0]
        HF[2, 5] = 3 * BL[0, 3]
        HF[2, 8] = 3 * BL[0, 6]
        HF[2, 11] = 3 * BL[0, 9]

        return HF

    def HmatrixS(self, x1, x2, x3, x4):
        HS = np.zeros((3, 12), dtype=float)
        BL = self.BLinear(x1, x2, x3, x4)

        # create the H2 matrix by differentiation of shape functions.
        HS[0, 0] = 3 * BL[0, 1]
        HS[0, 3] = 3 * BL[0, 4]
        HS[0, 6] = 3 * BL[0, 7]
        HS[0, 9] = 3 * BL[0, 10]

        HS[1, 1] = 3 * BL[0, 1]
        HS[1, 4] = 3 * BL[0, 4]
        HS[1, 7] = 3 * BL[0, 7]
        HS[1, 10] = 3 * BL[0, 10]

        HS[2, 2] = 3 * BL[0, 1]
        HS[2, 5] = 3 * BL[0, 4]
        HS[2, 8] = 3 * BL[0, 7]
        HS[2, 11] = 3 * BL[0, 10]

        return HS

    def HmatrixT(self, x1, x2, x3, x4):
        HT = np.zeros((3, 12), dtype=float)
        BL = self.BLinear(x1, x2, x3, x4)

        # create the H3 matrix by differentiation of shape functions.
        HT[0, 0] = 3 * BL[0, 2]
        HT[0, 3] = 3 * BL[0, 5]
        HT[0, 6] = 3 * BL[0, 8]
        HT[0, 9] = 3 * BL[0, 11]

        HT[1, 1] = 3 * BL[0, 2]
        HT[1, 4] = 3 * BL[0, 5]
        HT[1, 7] = 3 * BL[0, 8]
        HT[1, 10] = 3 * BL[0, 11]

        HT[2, 2] = 3 * BL[0, 2]
        HT[2, 5] = 3 * BL[0, 5]
        HT[2, 8] = 3 * BL[0, 8]
        HT[2, 11] = 3 * BL[0, 11]

        return HT

    def firstBNonLinear(self, x1, x2, x3, x4, x01, x02, x03, x04):
        AF = np.zeros((7, 3), dtype=float)
        G = self.G(x1, x2, x3, x4, x01, x02, x03, x04)
        HF = self.HmatrixF(x1, x2, x3, x4)

        # create the A1 matrix from nonlinear part of strain
        AF[0, 0] = - G[0, 0] / 3
        AF[0, 1] = - 2 * G[0, 1] / 3
        AF[0, 2] = G[2, 0] / 3

        AF[1, 0] = - G[0, 0] / 3
        AF[1, 1] = 2 * G[1, 0] / 3
        AF[1, 2] = G[2, 0] / 3

        AF[2, 1] = - G[1, 0] / 3

        AF[3, 0] = G[0, 0] / 3
        AF[3, 1] = - G[1, 0] / 3
        AF[3, 2] = - G[2, 0] / 3

        AF[4, 0] = 4 * G[1, 2]
        AF[4, 1] = - 2 * G[0, 2]
        AF[4, 2] = - 2 * G[1, 0]

        AF[5, 0] = - 4 * G[1, 2]

        AF[6, 0] = - 2 * G[0, 1]
        AF[6, 1] = - 4 * G[0, 0]

        BNF = np.zeros((7, 12), dtype=float)
        BNF = np.matmul(AF, HF)

        return BNF

    def secondBNonLinear(self, x1, x2, x3, x4, x01, x02, x03, x04):
        AS = np.zeros((7, 3), dtype=float)
        G = self.G(x1, x2, x3, x4, x01, x02, x03, x04)
        HS = self.HmatrixS(x1, x2, x3, x4)

        # create the A2 matrix from nonlinear part of strain
        AS[0, 1] = - G[1, 1] / 3
        AS[0, 2] = G[2, 1] / 3

        AS[1, 0] = 2 * G[1, 0] / 3
        AS[1, 1] = 2 * G[1, 1] / 3
        AS[1, 2] = - G[2, 1] / 3

        AS[2, 0] = - 2 * G[1, 0] / 3
        AS[2, 1] = - G[1, 1] / 3
        AS[2, 2] = G[2, 1]

        AS[3, 2] = - 2 * G[2, 1] / 3

        AS[4, 0] = - 2 * G[2, 0]
        AS[4, 1] = - 4 * G[2, 1]
        AS[4, 2] = 2 * G[2, 2]

        AS[5, 0] = 2 * G[0, 2]
        AS[5, 2] = - 4 * G[0, 0]

        AS[6, 1] = 2 * G[1, 0]
        AS[6, 2] = 2 * G[2, 0]

        BNS = np.zeros((7, 12), dtype=float)
        BNS = np.matmul(AS, HS)

        return BNS

    def thirdBNonLinear(self, x1, x2, x3, x4, x01, x02, x03, x04):
        AT = np.zeros((7, 3), dtype=float)
        G = self.G(x1, x2, x3, x4, x01, x02, x03, x04)
        HT = self.HmatrixS(x1, x2, x3, x4)

        # create the A3 matrix from nonlinear part of strain
        AT[0, 0] = G[0, 2] / 3
        AT[0, 1] = (- G[1, 2] - 4 * G[2, 1]) / 3
        AT[0, 2] = - G[2, 2] / 3

        AT[2, 0] = - G[0, 2] / 3
        AT[2, 1] = (G[1, 2] + 4 * G[2, 1]) / 3
        AT[2, 2] = G[2, 2] / 3

        AT[3, 0] = G[0, 2] / 3
        AT[3, 1] = (- G[1, 2] - 4 * G[2, 1]) / 3
        AT[3, 2] = - G[2, 2] / 3

        AT[4, 1] = - 2 * G[1, 1]

        AT[5, 1] = 2 * G[1, 1]
        AT[5, 2] = 2 * G[2, 1]

        BNT = np.zeros((7, 12), dtype=float)
        BNT = np.matmul(AT, HT)

        return BNT


"""
Changes made in version "1.0.0":

Original version
"""
