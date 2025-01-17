#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import cos, pi, sin
import numpy as np
from shapeFunctions import ShapeFunction as ShapeFn

"""
File shapeFnPentagons.py implements base class ShapeFunction for pentagons.

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
__date__ = "04-30-2019"
__update__ = "12-06-2020"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"


r"""

Changes made with respect to this version release can be found at end of file.


Overview of module shapeFnPentagons.py:


Module shapeFnPentagons.py provides five shape functions for a point (xi, eta)
residing within a pentagon, or along its boundary, where 'xi' and 'eta' are the
x and y coordinates, respectively, in the pentagon's natural co-ordinate frame
of reference, which inscribes a pentagon within the unit circle.  The centroid
of this regular pentagon is at the origin of the natural co-ordinate system.
These shape functions are used for interpolating within this region where
values at the five vertices for this interpolated field are supplied.

Also provided are the spatial derivatives for these shape functions, taken
with respect to coordinates 'xi' and 'eta', from which one can construct
approxiamtions for the Jacobian, plus the displacement G and deformation F
gradients, plus the B matrix, both linear and nonlinear.

The five vertices and the five chords of this pentagon are numbered according
to the following graphic when looking outward in:

                            v1
                          /   \
                       c2       c1
                     /             \
                  v2                 v5
                   \                 /
                    c3              c5
                     \              /
                      v3 -- c4 -- v4

where
    v1 = (xi_1, eta_1)
    v2 = (xi_2, eta_2)
    v3 = (xi_3, eta_3)
    v4 = (xi_4, eta_4)
    v5 = (xi_5, eta_5)
with
    xi_i  = cos(2*(i-1)*pi/5 + pi/2),     i = 1,2,..,5
    eta_i = sin(2*(i-1)*pi/5 + pi/2),     i = 1,2,..,5

By numbering the vertices and chords in a counterclockwise direction, the
algorithm used to compute its area will be positive; otherwise, if they had
been numbered clockwise, then the derived area would become negative.


class

    ShapeFunction
        Implements class ShapeFunction exported by module shapeFunctions.py.

constructor

    sf = ShapeFunction(coordinates)
        coordinates     A tuple of natural co-ordinate for interpolating at.
                        For a pentagon, coordinates = (xi, eta) where xi is
                        the 'x' natural co-ordinate and eta is the 'y' natural
                        co-ordinate.
    returns
        sf              a new instance of class ShapeFunction for 2D pentagons

variables

    # the shape functions

    sf.N1        the 1st shape function, it associates with vertex 1
    sf.N2        the 2nd shape function, it associates with vertex 2
    sf.N3        the 3rd shape function, it associates with vertex 3
    sf.N4        the 4th shape function, it associates with vertex 4
    sf.N5        the 5th shape function, it associates with vertex 5

    sf.Nmtx      a 2x10 matrix of shape functions for the pentagon

    # partial derivatives of the shape functions

    # first partial derivative: d N_i / dXi, i = 1..5
    sf.dN1dXi    gradient of the 1st shape function wrt the xi co-ordinate
    sf.dN2dXi    gradient of the 2nd shape function wrt the xi co-ordinate
    sf.dN3dXi    gradient of the 3rd shape function wrt the xi co-ordinate
    sf.dN4dXi    gradient of the 4th shape function wrt the xi co-ordinate
    sf.dN5dXi    gradient of the 5th shape function wrt the xi co-ordinate

    # first partial derivative: d N_i / dEta, i = 1..5
    sf.dN1dEta   gradient of the 1st shape function wrt the eta co-ordinate
    sf.dN2dEta   gradient of the 2nd shape function wrt the eta co-ordinate
    sf.dN3dEta   gradient of the 3rd shape function wrt the eta co-ordinate
    sf.dN4dEta   gradient of the 4th shape function wrt the eta co-ordinate
    sf.dN5dEta   gradient of the 5th shape function wrt the eta co-ordinate

inherited methods

    y = sf.interpolate(y1, y2, y3, y4, y5)
        y1   is a physical field of arbitrary type located at vertex 1
        y2   is a physical field of arbitrary type located at vertex 2
        y3   is a physical field of arbitrary type located at vertex 3
        y4   is a physical field of arbitrary type located at vertex 4
        y5   is a physical field of arbitrary type located at vertex 5
    returns
        y    is its interpolated value for this field at location (xi, eta)
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    Jmtx = sf.jacobianMatrix(x1, x2, x3, x4, x5)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical co-ordinates (x, y) locating vertex 5
    returns
        Jmtx is the Jacobian matrix (a 2x2 matrix) at location (xi, eta)
                    /  dx/dXi  dy/dXi  \
             Jmtx = |                  |
                    \ dx/dEta  dy/dEta /
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Jdet = sf.jacobianDeterminant(x1, x2, x3, x4, x5)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical co-ordinates (x, y) locating vertex 5
    returns
        Jdet is the determinant of the Jacobian matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Gmtx = sf.G(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical  co-ordinates (x, y) locating vertex 5
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y) locating vertex 4
        x05  is a tuple of reference co-ordinates (x, y) locating vertex 5
    returns
        Gmtx is the displacement gradient (a 2x2 matrix) at location (xi, eta)
                    / du/dx  du/dy \             u = x - X  or  x - x0
             Gmtx = |              |    where
                    \ dv/dx  dv/dy /             v = y - Y  or  y - y0
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Fmtx = sf.F(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical  co-ordinates (x, y) locating vertex 5
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y) locating vertex 4
        x05  is a tuple of reference co-ordinates (x, y) locating vertex 5
    returns
        Fmtx is the deformation gradient (a 2x2 matrix) at location (xi, eta)
                    / dx/dX  dx/dY \
             Fmtx = |              |
                    \ dy/dX  dy/dY /
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    BLmtx = sf.BL(x1, x2, x3, x4, x5)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical co-ordinates (x, y) locating vertex 5
    returns
        BLmtx   is the linear strain displacement matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    H1mtx = sf.H1(x1, x2, x3, x4, x5)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical co-ordinates (x, y) locating vertex 5
    returns
        H1mtx  is the derivative of shape functions from theta1 = H1 * D1 in the 
        first contribution to nonlinear strain
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    H2mtx = sf.H2(x1, x2, x3, x4, x5)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical co-ordinates (x, y) locating vertex 5
    returns
        H2mtx is the derivative of shape functions from theta2 = H2 * D2 in 
        second contribution to nonlinear strain
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    H3mtx = sf.H3(x1, x2, x3, x4, x5)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical co-ordinates (x, y) locating vertex 5
    returns
        H3mtx is the derivative of shape functions from theta3 = H3 * D3 in 
        third contribution to nonlinear strain
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.
    
    A1mtx = sf.A1(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical  co-ordinates (x, y) locating vertex 5
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y) locating vertex 4
        x05  is a tuple of reference co-ordinates (x, y) locating vertex 5
    returns
        A1mtx is the displacement gradient matrix EN1 = 1/2 * A1 * Theta1
    Inputs are co-ordinates evaluated in a global chordal co-ordinate system.

    A2mtx = sf.A2(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical  co-ordinates (x, y) locating vertex 5
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y) locating vertex 4
        x05  is a tuple of reference co-ordinates (x, y) locating vertex 5
    returns
        A2mtx is the displacement gradient matrix EN2 = 1/2 * A2 * Theta2
    Inputs are co-ordinates evaluated in a global chordal co-ordinate system.

    A3mtx = sf.A3(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical  co-ordinates (x, y) locating vertex 5
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y) locating vertex 4
        x05  is a tuple of reference co-ordinates (x, y) locating vertex 5
    returns
        A3mtx is the displacement gradient matrix EN3 = 1/2 * A3 * Theta3
    Inputs are co-ordinates evaluated in a global chordal co-ordinate system.
    
    L1mtx = sf.L1(x1, x2, x3, x4, x5)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical  co-ordinates (x, y) locating vertex 5
    returns
    L1mtx is the derivative of shape functions from dA1 = L1 * D1 in 
          contribution to nonlinear strain
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    L2mtx = sf.L2(x1, x2, x3, x4, x5)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical  co-ordinates (x, y) locating vertex 5
    returns
    L2mtx is the derivative of shape functions from dA2 = L2 * D2 in 
          contribution to nonlinear strain
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    L3mtx = sf.L3(x1, x2, x3, x4, x5)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical  co-ordinates (x, y) locating vertex 5
    returns
    L3mtx is the derivative of shape functions from dA3 = L3 * D3 in 
          contribution to nonlinear strain
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.
    
    BN1mtx = sf.BN1(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical  co-ordinates (x, y) locating vertex 5
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y) locating vertex 4
        x05  is a tuple of reference co-ordinates (x, y) locating vertex 5
    returns
        BN1mtx is first nonlinear contribution to strain displacement matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    BN2mtx = sf.BN2(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical  co-ordinates (x, y) locating vertex 5
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y) locating vertex 4
        x05  is a tuple of reference co-ordinates (x, y) locating vertex 5
    returns
        BN2mtx is second nonlinear contribution to the strain displacement matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    BN3mtx = sf.BN3(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x3   is a tuple of physical  co-ordinates (x, y) locating vertex 3
        x4   is a tuple of physical  co-ordinates (x, y) locating vertex 4
        x5   is a tuple of physical  co-ordinates (x, y) locating vertex 5
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
        x03  is a tuple of reference co-ordinates (x, y) locating vertex 3
        x04  is a tuple of reference co-ordinates (x, y) locating vertex 4
        x05  is a tuple of reference co-ordinates (x, y) locating vertex 5
    returns
        BN3mtx is third nonlinear contribution to the strain displacement matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.
    
Reference
    1) Dasgupta, G., "Interpolants within convex polygons: Wachspress shape
       functions", Journal of Aerospace Engineering, vol. 16, 2003, pp. 1-8.
"""


class ShapeFunction(ShapeFn):

    # constructor

    def __init__(self, coordinates):
        # construct the five, exported, shape functions for location (xi, eta)
        super(ShapeFunction, self).__init__(coordinates)
        if len(coordinates) == 2:
            xi = coordinates[0]
            eta = coordinates[1]
        else:
            raise RuntimeError("The co-ordinates for a pentagon are in 2D.")

        # create the x and y vertex vectors in their natural coordinate frame
        self.det = 0
        xiVec = np.zeros(6, dtype=float)
        etaVec = np.zeros(6, dtype=float)
        for i in range(1, 6):
            xiVec[i] = cos(2.0*(i-1)*pi/5.0 + pi/2.0)
            etaVec[i] = sin(2.0*(i-1)*pi/5.0 + pi/2.0)
        xiVec[0] = xiVec[5]
        etaVec[0] = etaVec[5]

        # create the a and b vectors that describe a chord as a straight line
        # between points xi_{i-1} and xi_i for chord c_i = 1 - a_i xi - b_i eta
        # such that c_i = 0
        aVec = np.zeros(9, dtype=float)
        bVec = np.zeros(9, dtype=float)
        for i in range(1, 6):
            denom = xiVec[i-1] * etaVec[i] - xiVec[i] * etaVec[i-1]
            aVec[i] = (etaVec[i] - etaVec[i-1]) / denom
            bVec[i] = (xiVec[i-1] - xiVec[i]) / denom
        aVec[0] = aVec[5]
        bVec[0] = bVec[5]
        aVec[6] = aVec[1]
        bVec[6] = bVec[1]
        aVec[7] = aVec[2]
        bVec[7] = bVec[2]
        aVec[8] = aVec[3]
        bVec[8] = bVec[3]

        # create the kappa vector for scaling this rational polynomial
        kappa = np.zeros(6, dtype=float)
        kappa[1] = 1.0
        for i in range(2, 6):
            kappa[i] = kappa[i-1] * ((aVec[i+1] * (xiVec[i-1] - xiVec[i])
                                      + bVec[i+1] * (etaVec[i-1] - etaVec[i]))
                                     / (aVec[i-1] * (xiVec[i] - xiVec[i-1])
                                        + bVec[i-1] * (etaVec[i]
                                                       - etaVec[i-1])))

        # create the alpha coefficients for the numerator's polynomial
        alpha = np.zeros((10, 6), dtype=float)
        for i in range(1, 6):
            alpha[0, i] = 1.0
            alpha[1, i] = -(aVec[i+1] + aVec[i+2] + aVec[i+3])
            alpha[2, i] = -(bVec[i+1] + bVec[i+2] + bVec[i+3])
            alpha[3, i] = (aVec[i+1] * aVec[i+2] + aVec[i+2] * aVec[i+3]
                           + aVec[i+3] * aVec[i+1])
            alpha[4, i] = (aVec[i+1] * (bVec[i+2] + bVec[i+3])
                           + aVec[i+2] * (bVec[i+1] + bVec[i+3])
                           + aVec[i+3] * (bVec[i+1] + bVec[i+2]))
            alpha[5, i] = (bVec[i+1] * bVec[i+2] + bVec[i+2] * bVec[i+3]
                           + bVec[i+3] * bVec[i+1])
            alpha[6, i] = -aVec[i+1] * aVec[i+2] * aVec[i+3]
            alpha[7, i] = -(aVec[i+1] * aVec[i+2] * bVec[i+3]
                            + aVec[i+1] * bVec[i+2] * aVec[i+3]
                            + bVec[i+1] * aVec[i+2] * aVec[i+3])
            alpha[8, i] = -(aVec[i+1] * bVec[i+2] * bVec[i+3]
                            + bVec[i+1] * aVec[i+2] * bVec[i+3]
                            + bVec[i+1] * bVec[i+2] * aVec[i+3])
            alpha[9, i] = -bVec[i+1] * bVec[i+2] * bVec[i+3]

        # create the beta coefficients for the denominator's polynomial
        beta = np.zeros(6, dtype=float)
        for j in range(6):
            for i in range(1, 6):
                beta[j] = beta[j] + alpha[j, i] * kappa[i]

        # create the polynomials for the numerators of these shape functions
        aPoly = np.zeros(6, dtype=float)
        for i in range(1, 6):
            aPoly[i] = (alpha[0, i] + alpha[1, i] * xi + alpha[2, i] * eta
                        + alpha[3, i] * xi**2 + alpha[4, i] * xi * eta
                        + alpha[5, i] * eta**2 + alpha[6, i] * xi**3
                        + alpha[7, i] * xi**2 * eta + alpha[8, i] * xi * eta**2
                        + alpha[9, i] * eta**3)

        # create the polynomial for the denominator of these shape functions
        bPoly = (beta[0] + beta[1] * xi * beta[2] * eta + beta[3] * xi**2
                 + beta[4] * xi * eta + beta[5] * eta**2)

        # create the five exported shape functions
        self.N5 = kappa[1] * aPoly[1] / bPoly
        self.N1 = kappa[2] * aPoly[2] / bPoly
        self.N2 = kappa[3] * aPoly[3] / bPoly
        self.N3 = kappa[4] * aPoly[4] / bPoly
        self.N4 = kappa[5] * aPoly[5] / bPoly

        # construct the 2x10 shape function matrix for a pentagon
        self.Nmtx = np.array([[self.N1, 0, self.N2, 0, self.N3, 0,
                                self.N4, 0, self.N5, 0],
                               [0, self.N1, 0, self.N2, 0, self.N3,
                                0, self.N4, 0, self.N5]])

        # determine the gradients of the numerator polynomials
        dAdXi = np.zeros(6, dtype=float)
        dAdEta = np.zeros(6, dtype=float)
        for i in range(1, 6):
            dAdXi[i] = (alpha[1, i] + 2 * alpha[3, i] * xi
                        + alpha[4, i] * eta + 3 * alpha[6, i] * xi**2
                        + 2 * alpha[7, i] * xi * eta + alpha[8, i] * eta**2)
            dAdEta[i] = (alpha[2, i] + alpha[4, i] * xi
                         + 2 * alpha[5, i] * eta + alpha[7, i] * xi**2
                         + 2 * alpha[8, i] * xi * eta
                         + 3 * alpha[9, i] * eta**2)

        # determine the gradients of the denominator polynomial
        dBdXi = beta[1] + 2 * beta[3] * xi + beta[4] * eta
        dBdEta = beta[2] + beta[4] * xi + 2 * beta[5] * eta

        # determine the numerators for the spatial derivatives
        dNumdXi = np.zeros(6, dtype=float)
        dNumdEta = np.zeros(6, dtype=float)
        for i in range(1, 6):
            dNumdXi[i] = bPoly * dAdXi[i] - dBdXi * aPoly[i]
            dNumdEta[i] = bPoly * dAdEta[i] - dBdEta * aPoly[i]

        # create the ten, exported derivatives of the shape functions
        self.dN5dXi = kappa[1] * dNumdXi[1] / bPoly**2
        self.dN1dXi = kappa[2] * dNumdXi[2] / bPoly**2
        self.dN2dXi = kappa[3] * dNumdXi[3] / bPoly**2
        self.dN3dXi = kappa[4] * dNumdXi[4] / bPoly**2
        self.dN4dXi = kappa[5] * dNumdXi[5] / bPoly**2

        self.dN5dEta = kappa[1] * dNumdEta[1] / bPoly**2
        self.dN1dEta = kappa[2] * dNumdEta[2] / bPoly**2
        self.dN2dEta = kappa[3] * dNumdEta[3] / bPoly**2
        self.dN3dEta = kappa[4] * dNumdEta[4] / bPoly**2
        self.dN4dEta = kappa[5] * dNumdEta[5] / bPoly**2
        return  # the object

    # methods

    def interpolate(self, y1, y2, y3, y4, y5):
        if (type(y1) == type(y2) and type(y2) == type(y3)
           and type(y3) == type(y4)) and type(y4) == type(y5):
            y = (self.N1 * y1 + self.N2 * y2 + self.N3 * y3 + self.N4 * y4
                 + self.N5 * y5)
        else:
            raise RuntimeError("interpolate arguments must be the same type.")
        return y
   
    # determine the jacobian matrix
    def jacobianMatrix(self, x1, x2, x3, x4, x5):
        if (isinstance(x1, tuple) and len(x1) == 2
           and isinstance(x2, tuple) and len(x2) == 2
           and isinstance(x3, tuple) and len(x3) == 2
           and isinstance(x4, tuple) and len(x4) == 2
           and isinstance(x5, tuple) and len(x5) == 2):
            Jmtx = np.zeros((2, 2), dtype=float)
            Jmtx[0, 0] = (self.dN1dXi * x1[0] + self.dN2dXi * x2[0]
                          + self.dN3dXi * x3[0] + self.dN4dXi * x4[0]
                          + self.dN5dXi * x5[0])
            Jmtx[0, 1] = (self.dN1dXi * x1[1] + self.dN2dXi * x2[1]
                          + self.dN3dXi * x3[1] + self.dN4dXi * x4[1]
                          + self.dN5dXi * x5[1])
            Jmtx[1, 0] = (self.dN1dEta * x1[0] + self.dN2dEta * x2[0]
                          + self.dN3dEta * x3[0] + self.dN4dEta * x4[0]
                          + self.dN5dEta * x5[0])
            Jmtx[1, 1] = (self.dN1dEta * x1[1] + self.dN2dEta * x2[1]
                          + self.dN3dEta * x3[1] + self.dN4dEta * x4[1]
                          + self.dN5dEta * x5[1])
        else:
            raise RuntimeError("Each argument of shapeFunction.jacobianMatrix "
                               + "must be a tuple of co-ordinates, "
                               + "e.g., (x, y).")
        return Jmtx

    # determine the determinant of jacobian matrix
    def jacobianDeterminant(self, x1, x2, x3, x4, x5):
        Jmtx = self.jacobianMatrix(x1, x2, x3, x4, x5)
        return np.linalg.det(Jmtx)
 
    def G(self, x1, x2, x3, x4, x5, x01, x02, x03, x04, x05):
        if (isinstance(x1, tuple) and len(x1) == 2
           and isinstance(x2, tuple) and len(x2) == 2
           and isinstance(x3, tuple) and len(x3) == 2
           and isinstance(x4, tuple) and len(x4) == 2
           and isinstance(x5, tuple) and len(x5) == 2
           and isinstance(x01, tuple) and len(x01) == 2
           and isinstance(x02, tuple) and len(x02) == 2
           and isinstance(x03, tuple) and len(x03) == 2
           and isinstance(x04, tuple) and len(x04) == 2
           and isinstance(x05, tuple) and len(x05) == 2):
            u1 = x1[0] - x01[0]
            u2 = x2[0] - x02[0]
            u3 = x3[0] - x03[0]
            u4 = x4[0] - x04[0]
            u5 = x5[0] - x05[0]
            v1 = x1[1] - x01[1]
            v2 = x2[1] - x02[1]
            v3 = x3[1] - x03[1]
            v4 = x4[1] - x04[1]
            v5 = x5[1] - x05[1]

            # determine the displacement gradient
            disGrad = np.zeros((2, 2), dtype=float)
            disGrad[0, 0] = (self.dN1dXi * u1 + self.dN2dXi * u2
                             + self.dN3dXi * u3 + self.dN4dXi * u4
                             + self.dN5dXi * u5)
            disGrad[0, 1] = (self.dN1dEta * u1 + self.dN2dEta * u2
                             + self.dN3dEta * u3 + self.dN4dEta * u4
                             + self.dN5dEta * u5)
            disGrad[1, 0] = (self.dN1dXi * v1 + self.dN2dXi * v2
                             + self.dN3dXi * v3 + self.dN4dXi * v4
                             + self.dN5dXi * v5)
            disGrad[1, 1] = (self.dN1dEta * v1 + self.dN2dEta * v2
                             + self.dN3dEta * v3 + self.dN4dEta * v4
                             + self.dN5dEta * v5)

            # determine the current gradient of position
            curGrad = np.transpose(self.jacobianMatrix(x1, x2, x3, x4, x5))
        else:
            raise RuntimeError("Each argument of shapeFunction.G must be a "
                               + "tuple of co-ordinates, e.g., (x, y).")
        Gmtx = np.matmul(disGrad, np.linalg.inv(curGrad))
        return Gmtx

    def F(self, x1, x2, x3, x4, x5, x01, x02, x03, x04, x05):
        if (isinstance(x1, tuple) and len(x1) == 2
           and isinstance(x2, tuple) and len(x2) == 2
           and isinstance(x3, tuple) and len(x3) == 2
           and isinstance(x4, tuple) and len(x4) == 2
           and isinstance(x5, tuple) and len(x5) == 2
           and isinstance(x01, tuple) and len(x01) == 2
           and isinstance(x02, tuple) and len(x02) == 2
           and isinstance(x03, tuple) and len(x03) == 2
           and isinstance(x04, tuple) and len(x04) == 2
           and isinstance(x05, tuple) and len(x05) == 2):
            u1 = x1[0] - x01[0]
            u2 = x2[0] - x02[0]
            u3 = x3[0] - x03[0]
            u4 = x4[0] - x04[0]
            u5 = x5[0] - x05[0]
            v1 = x1[1] - x01[1]
            v2 = x2[1] - x02[1]
            v3 = x3[1] - x03[1]
            v4 = x4[1] - x04[1]
            v5 = x5[1] - x05[1]

            # determine the displacement gradient
            disGrad = np.zeros((2, 2), dtype=float)
            disGrad[0, 0] = (self.dN1dXi * u1 + self.dN2dXi * u2
                             + self.dN3dXi * u3 + self.dN4dXi * u4
                             + self.dN5dXi * u5)
            disGrad[0, 1] = (self.dN1dEta * u1 + self.dN2dEta * u2
                             + self.dN3dEta * u3 + self.dN4dEta * u4
                             + self.dN5dEta * u5)
            disGrad[1, 0] = (self.dN1dXi * v1 + self.dN2dXi * v2
                             + self.dN3dXi * v3 + self.dN4dXi * v4
                             + self.dN5dXi * v5)
            disGrad[1, 1] = (self.dN1dEta * v1 + self.dN2dEta * v2
                             + self.dN3dEta * v3 + self.dN4dEta * v4
                             + self.dN5dEta * v5)

            # determine the reference gradient of position
            refGrad = np.zeros((2, 2), dtype=float)
            refGrad[0, 0] = (self.dN1dXi * x01[0] + self.dN2dXi * x02[0]
                             + self.dN3dXi * x03[0] + self.dN4dXi * x04[0]
                             + self.dN5dXi * x05[0])
            refGrad[0, 1] = (self.dN1dEta * x01[0] + self.dN2dEta * x02[0]
                             + self.dN3dEta * x03[0] + self.dN4dEta * x04[0]
                             + self.dN5dEta * x05[0])
            refGrad[1, 0] = (self.dN1dXi * x01[1] + self.dN2dXi * x02[1]
                             + self.dN3dXi * x03[1] + self.dN4dXi * x04[1]
                             + self.dN5dXi * x05[1])
            refGrad[1, 1] = (self.dN1dEta * x01[1] + self.dN2dEta * x02[1]
                             + self.dN3dEta * x03[1] + self.dN4dEta * x04[1]
                             + self.dN5dEta * x05[1])
        else:
            raise RuntimeError("Each argument of shapeFunction.F must be a "
                               + "tuple of co-ordinates, e.g., (x, y).")
        Fmtx = (np.eye(2, dtype=float)
                + np.matmul(disGrad, np.linalg.inv(refGrad)))
        return Fmtx
    
    # determine the linear strain displacement matrix
    def BL(self, x1, x2, x3, x4, x5):
        Jmtx = self.jacobianMatrix(x1, x2, x3, x4, x5)
        BL = np.zeros((3, 10), dtype=float)

        BL[0, 0] = (self.dN1dXi * Jmtx[1, 1] - self.dN1dEta * Jmtx[0, 1]) / 2
        BL[0, 1] = (-self.dN1dXi * Jmtx[1, 0] + self.dN1dEta * Jmtx[0, 0]) / 2
        BL[0, 2] = (self.dN2dXi * Jmtx[1, 1] - self.dN2dEta * Jmtx[0, 1]) / 2
        BL[0, 3] = (-self.dN2dXi * Jmtx[1, 0] + self.dN2dEta * Jmtx[0, 0]) / 2
        BL[0, 4] = (self.dN3dXi * Jmtx[1, 1] - self.dN3dEta * Jmtx[0, 1]) / 2
        BL[0, 5] = (-self.dN3dXi * Jmtx[1, 0] + self.dN3dEta * Jmtx[0, 0]) / 2
        BL[0, 6] = (self.dN4dXi * Jmtx[1, 1] - self.dN4dEta * Jmtx[0, 1]) / 2
        BL[0, 7] = (-self.dN4dXi * Jmtx[1, 0] + self.dN4dEta * Jmtx[0, 0]) / 2
        BL[0, 8] = (self.dN5dXi * Jmtx[1, 1] - self.dN5dEta * Jmtx[0, 1]) / 2
        BL[0, 9] = (-self.dN5dXi * Jmtx[1, 0] + self.dN5dEta * Jmtx[0, 0]) / 2

        BL[1, 0] = (self.dN1dXi * Jmtx[1, 1] - self.dN1dEta * Jmtx[0, 1]) / 2
        BL[1, 1] = (self.dN1dXi * Jmtx[1, 0] - self.dN1dEta * Jmtx[0, 0]) / 2
        BL[1, 2] = (self.dN2dXi * Jmtx[1, 1] - self.dN2dEta * Jmtx[0, 1]) / 2
        BL[1, 3] = (self.dN2dXi * Jmtx[1, 0] - self.dN2dEta * Jmtx[0, 0]) / 2
        BL[1, 4] = (self.dN3dXi * Jmtx[1, 1] - self.dN3dEta * Jmtx[0, 1]) / 2
        BL[1, 5] = (self.dN3dXi * Jmtx[1, 0] - self.dN3dEta * Jmtx[0, 0]) / 2
        BL[1, 6] = (self.dN4dXi * Jmtx[1, 1] - self.dN4dEta * Jmtx[0, 1]) / 2
        BL[1, 7] = (self.dN4dXi * Jmtx[1, 0] - self.dN4dEta * Jmtx[0, 0]) / 2
        BL[1, 8] = (self.dN5dXi * Jmtx[1, 1] - self.dN5dEta * Jmtx[0, 1]) / 2
        BL[1, 9] = (self.dN5dXi * Jmtx[1, 0] - self.dN5dEta * Jmtx[0, 0]) / 2

        BL[2, 0] = -self.dN1dXi * Jmtx[1, 0] + self.dN1dEta * Jmtx[0, 0]
        BL[2, 1] = self.dN1dXi * Jmtx[1, 1] - self.dN1dEta * Jmtx[0, 1]
        BL[2, 2] = -self.dN2dXi * Jmtx[1, 0] + self.dN2dEta * Jmtx[0, 0]
        BL[2, 3] = self.dN2dXi * Jmtx[1, 1] - self.dN2dEta * Jmtx[0, 1]
        BL[2, 4] = -self.dN3dXi * Jmtx[1, 0] + self.dN3dEta * Jmtx[0, 0]
        BL[2, 5] = self.dN3dXi * Jmtx[1, 1] - self.dN3dEta * Jmtx[0, 1]
        BL[2, 6] = -self.dN4dXi * Jmtx[1, 0] + self.dN4dEta * Jmtx[0, 0]
        BL[2, 7] = self.dN4dXi * Jmtx[1, 1] - self.dN4dEta * Jmtx[0, 1]
        BL[2, 8] = -self.dN5dXi * Jmtx[1, 0] + self.dN5dEta * Jmtx[0, 0]
        BL[2, 9] = self.dN5dXi * Jmtx[1, 1] - self.dN5dEta * Jmtx[0, 1]

        detJ = np.linalg.det(Jmtx)
        BLmtx = BL / detJ

        return BLmtx

    def H1(self, x1, x2, x3, x4, x5):
        H1mtx = np.zeros((2, 10), dtype=float)
        BLmtx = self.BL(x1, x2, x3, x4, x5)

        # create the H1 matrix by differentiation of shape functions.
        H1mtx[0, 0] = 2 * BLmtx[0, 0]
        H1mtx[0, 2] = 2 * BLmtx[0, 2]
        H1mtx[0, 4] = 2 * BLmtx[0, 4]
        H1mtx[0, 6] = 2 * BLmtx[0, 6]
        H1mtx[0, 8] = 2 * BLmtx[0, 8]

        H1mtx[1, 1] = 2 * BLmtx[0, 1]
        H1mtx[1, 3] = 2 * BLmtx[0, 3]
        H1mtx[1, 5] = 2 * BLmtx[0, 5]
        H1mtx[1, 7] = 2 * BLmtx[0, 7]
        H1mtx[1, 9] = 2 * BLmtx[0, 9]

        return H1mtx

    def H2(self, x1, x2, x3, x4, x5):
        H2mtx = np.zeros((2, 10), dtype=float)
        BLmtx = self.BL(x1, x2, x3, x4, x5)

        # create the H2 matrix by differentiation of shape functions.
        H2mtx[0, 0] = 2 * BLmtx[0, 1]
        H2mtx[0, 2] = 2 * BLmtx[0, 3]
        H2mtx[0, 4] = 2 * BLmtx[0, 5]
        H2mtx[0, 6] = 2 * BLmtx[0, 7]
        H2mtx[0, 8] = 2 * BLmtx[0, 9]

        H2mtx[1, 1] = 2 * BLmtx[0, 0]
        H2mtx[1, 3] = 2 * BLmtx[0, 2]
        H2mtx[1, 5] = 2 * BLmtx[0, 4]
        H2mtx[1, 7] = 2 * BLmtx[0, 6]
        H2mtx[1, 9] = 2 * BLmtx[0, 8]

        return H2mtx

    def H3(self, x1, x2, x3, x4, x5):
        H3mtx = np.zeros((2, 10), dtype=float)
        BLmtx = self.BL(x1, x2, x3, x4, x5)

        # create the H3 matrix by differentiation of shape functions.
        H3mtx[0, 0] = 2 * BLmtx[0, 0]
        H3mtx[0, 2] = 2 * BLmtx[0, 2]
        H3mtx[0, 4] = 2 * BLmtx[0, 4]
        H3mtx[0, 6] = 2 * BLmtx[0, 6]
        H3mtx[0, 8] = 2 * BLmtx[0, 8]

        H3mtx[1, 1] = 2 * BLmtx[0, 0]
        H3mtx[1, 3] = 2 * BLmtx[0, 2]
        H3mtx[1, 5] = 2 * BLmtx[0, 4]
        H3mtx[1, 7] = 2 * BLmtx[0, 6]
        H3mtx[1, 9] = 2 * BLmtx[0, 8]

        return H3mtx
    
    def A1(self, x1, x2, x3, x4, x5, x01, x02, x03, x04, x05):
        A1mtx = np.zeros((3, 2), dtype=float)
        Gmtx = self.G(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)

        # create the A1 matrix from nonlinear part of strain
        A1mtx[0, 1] = - Gmtx[1, 1] / 2
        A1mtx[1, 0] = - Gmtx[0, 0] / 2
        A1mtx[1, 1] = Gmtx[1, 1] / 2
        A1mtx[2, 1] = 2 * Gmtx[1, 0]

        return A1mtx
    
    def A2(self, x1, x2, x3, x4, x5, x01, x02, x03, x04, x05):
        A2mtx = np.zeros((3, 2), dtype=float)
        Gmtx = self.G(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)

        # create the A2 matrix from nonlinear part of strain
        A2mtx[0, 0] = - Gmtx[1, 0]
        A2mtx[1, 0] = Gmtx[1, 0]
        A2mtx[2, 1] = - 4 * Gmtx[0, 0]

        return A2mtx

    def A3(self, x1, x2, x3, x4, x5, x01, x02, x03, x04, x05):
        A3mtx = np.zeros((3, 2), dtype=float)
        Gmtx = self.G(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)

        # create the A1 matrix from nonlinear part of strain
        A3mtx[0, 0] = - Gmtx[0, 0] / 2
        A3mtx[1, 1] = Gmtx[1, 0] 
        A3mtx[2, 0] = - 2 * Gmtx[0, 1] 

        return A3mtx
    
    # derivative of shape functions in contribution to the first nonlinear 
    # strain
    def L1(self, x1, x2, x3, x4, x5):
        B = self.BL(x1, x2, x3, x4, x5)
        L1 = np.zeros((3, 10), dtype=float)

        L1[0, 1] = - B[0, 1]
        L1[0, 3] = - B[0, 3]
        L1[0, 5] = - B[0, 5]
        L1[0, 7] = - B[0, 7]
        L1[0, 9] = - B[0, 9]

        L1[1, 0] = - B[1, 0]
        L1[1, 1] = - B[1, 1]
        L1[1, 2] = - B[1, 2]
        L1[1, 3] = - B[1, 3]
        L1[1, 4] = - B[1, 4]
        L1[1, 5] = - B[1, 5]
        L1[1, 6] = - B[1, 6]
        L1[1, 7] = - B[1, 7]
        L1[1, 8] = - B[1, 8]
        L1[1, 9] = - B[1, 9]
        
        L1[2, 1] = 2 * B[2, 1]
        L1[2, 3] = 2 * B[2, 3]
        L1[2, 5] = 2 * B[2, 5]
        L1[2, 7] = 2 * B[2, 7]
        L1[2, 9] = 2 * B[2, 9]

        return L1

    # derivative of shape functions in contribution to the second nonlinear 
    # strain
    def L2(self, x1, x2, x3, x4, x5):
        B = self.BL(x1, x2, x3, x4, x5)
        L2 = np.zeros((3, 10), dtype=float)

        L2[0, 0] = - 2 * B[0, 0]
        L2[0, 2] = - 2 * B[0, 2]
        L2[0, 4] = - 2 * B[0, 4]
        L2[0, 6] = - 2 * B[0, 6]
        L2[0, 8] = - 2 * B[0, 8]

        L2[1, 0] = 2 * B[1, 0]
        L2[1, 2] = 2 * B[1, 2]
        L2[1, 4] = 2 * B[1, 4]
        L2[1, 6] = 2 * B[1, 6]
        L2[1, 8] = 2 * B[1, 8]
        
        L2[2, 1] = - 4 * B[2, 1]
        L2[2, 3] = - 4 * B[2, 3]
        L2[2, 5] = - 4 * B[2, 5]
        L2[2, 7] = - 4 * B[2, 7]
        L2[2, 9] = - 4 * B[2, 9]

        return L2   

    def L3(self, x1, x2, x3, x4, x5):
        B = self.BL(x1, x2, x3, x4, x5)
        L3 = np.zeros((3, 10), dtype=float)

        L3[0, 0] = - B[0, 0]
        L3[0, 2] = - B[0, 2]
        L3[0, 4] = - B[0, 4]
        L3[0, 6] = - B[0, 6]
        L3[0, 8] = - B[0, 8]

        L3[1, 1] = - B[1, 1]
        L3[1, 3] = - B[1, 3]
        L3[1, 5] = - B[1, 5]
        L3[1, 7] = - B[1, 7]
        L3[1, 9] = - B[1, 9]
        
        L3[2, 0] = - 2 * B[2, 0]
        L3[2, 2] = - 2 * B[2, 2]
        L3[2, 4] = - 2 * B[2, 4]
        L3[2, 6] = - 2 * B[2, 6]
        L3[2, 8] = - 2 * B[2, 8]

        return L3
        
    # first nonlinear contribution to strain displacement matrix
    def BN1(self, x1, x2, x3, x4, x5, x01, x02, x03, x04, x05):
        H1mtx = self.H1(x1, x2, x3, x4, x5)
        A1mtx = self.A1(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)

        B1Nmtx = np.zeros((3, 10), dtype=float)
        B1Nmtx = np.matmul(A1mtx, H1mtx)

        return B1Nmtx
    
    # second nonlinear contribution to strain displacement matrix
    def BN2(self, x1, x2, x3, x4, x5, x01, x02, x03, x04, x05):
        H2mtx = self.H2(x1, x2, x3, x4, x5)
        A2mtx = self.A2(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)

        B2Nmtx = np.zeros((3, 10), dtype=float)
        B2Nmtx = np.matmul(A2mtx, H2mtx)

        return B2Nmtx

    # third nonlinear contribution to strain displacement matrix
    def BN3(self, x1, x2, x3, x4, x5, x01, x02, x03, x04, x05):
        H3mtx = self.H3(x1, x2, x3, x4, x5)
        A3mtx = self.A3(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)

        B3Nmtx = np.zeros((3, 10), dtype=float)
        B3Nmtx = np.matmul(A3mtx, H3mtx)

        return B3Nmtx
    
"""
Changes made in version "1.0.0":

Original version
"""
