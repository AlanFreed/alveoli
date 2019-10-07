#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import cos, pi, sin
import numpy as np

"""
File shapeFnPentagons.py provides shape functions for interpolating pentagons.

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
__version__ = "1.3.2"
__date__ = "04-30-2019"
__update__ = "10-05-2019"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"


r"""

Change in version "1.3.0":

method added

    det = sf.detJacobian(x1, x2, x3, x4, x5)
        x1    is a tuple of current coordinates (x, y) located at vertex 1
        x2    is a tuple of current coordinates (x, y) located at vertex 2
        x3    is a tuple of current coordinates (x, y) located at vertex 3
        x4    is a tuple of current coordinates (x, y) located at vertex 4
        x5    is a tuple of current coordinates (x, y) located at vertex 5
    returns
        det   is the determinant of the Jacobian matrix

variables

    sf.Nmatx  a 2x10 shape function matrix for the pentagon


Overview of module shapeFnPentagons.py:


Module shapeFnPentagons.py provides five shape functions for a point (xi, eta)
residing within a pentagon, or along its boundary, where 'xi' and 'eta' are the
x and y coordinates, respectively, in the pentagon's natural coordinate frame
of reference, which inscribes a pentagon within the unit circle.  The centroid
of this regular pentagon is at the origin of the natural coordinate system.
These shape functions are used for interpolating within this region where
values at the five vertices for this interpolated field are supplied.

Also provided are the spatial derivatives for these shape functions, taken
with respect to coordinates 'xi' and 'eta'.  From these one can construct
approxiamtions for the displacement G and deformation F gradients.

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

By numbering the vertices and chords in a counterclockwise direction, the
algorithm used to compute its area will be positive; otherwise, if they had
been numbered clockwise, then the derived area would have been negative.


class

    shapeFunction

constructor

    sf = shapeFunction(xi, eta)
        xi    is the x coordinate in the natural coordinate system
        eta   is the y coordiante in the natural coordiante system

methods

    y = sf.interpolate(y1, y2, y3, y4, y5)
        y1   is the value of field y located at vertex 1
        y2   is the value of field y located at vertex 2
        y3   is the value of field y located at vertex 3
        y4   is the value of field y located at vertex 4
        y5   is the value of field y located at vertex 5
    returns
        y    is its interpolated value for field y at location (xi, eta)

    det = sf.detJacobian(x1, x2, x3, x4, x5)
        x1   is a tuple of physical coordinates (x, y) located at vertex 1
        x2   is a tuple of physical coordinates (x, y) located at vertex 2
        x3   is a tuple of physical coordinates (x, y) located at vertex 3
        x4   is a tuple of physical coordinates (x, y) located at vertex 4
        x5   is a tuple of physical coordinates (x, y) located at vertex 5
    returns
        det  is the determinant of the Jacobian matrix
    inputs are tuples of coordinates evaluated in a global coordinate system

    Gmtx = sf.G(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)
        x1   is a tuple of physical  coordinates (x, y) located at vertex 1
        x2   is a tuple of physical  coordinates (x, y) located at vertex 2
        x3   is a tuple of physical  coordinates (x, y) located at vertex 3
        x4   is a tuple of physical  coordinates (x, y) located at vertex 4
        x5   is a tuple of physical  coordinates (x, y) located at vertex 5
        x01  is a tuple of reference coordinates (x, y) located at vertex 1
        x02  is a tuple of reference coordinates (x, y) located at vertex 2
        x03  is a tuple of reference coordinates (x, y) located at vertex 3
        x04  is a tuple of reference coordinates (x, y) located at vertex 4
        x05  is a tuple of reference coordinates (x, y) located at vertex 5
    returns
        Gmtx is the displacement gradient (a 2x2 matrix) at location (xi, eta)
                    / du/dx  du/dy \             u = x - X  or  x - x0
             Gmtx = |              |    where
                    \ dv/dx  dv/dy /             v = y - Y  or  y - y0
    inputs are tuples of coordinates evaluated in a global coordinate system

    Fmtx = sf.F(x1, x2, x3, x4, x5, x01, x02, x03, x04, x05)
        x1   is a tuple of physical  coordinates (x, y) located at vertex 1
        x2   is a tuple of physical  coordinates (x, y) located at vertex 2
        x3   is a tuple of physical  coordinates (x, y) located at vertex 3
        x4   is a tuple of physical  coordinates (x, y) located at vertex 4
        x5   is a tuple of physical  coordinates (x, y) located at vertex 5
        x01  is a tuple of reference coordinates (x, y) located at vertex 1
        x02  is a tuple of reference coordinates (x, y) located at vertex 2
        x03  is a tuple of reference coordinates (x, y) located at vertex 3
        x04  is a tuple of reference coordinates (x, y) located at vertex 4
        x05  is a tuple of reference coordinates (x, y) located at vertex 5
    returns
        Fmtx is the deformation gradient (a 2x2 matrix) at location (xi, eta)
                    / dx/dX  dx/dY \             X = x0
             Fmtx = |              |    where
                    \ dy/dX  dy/dY /             Y = y0
    inputs are tuples of coordinates evaluated in a global coordinate system

variables

    # the shape functions

    sf.N1        the 1st shape function, it associates with vertex 1
    sf.N2        the 2nd shape function, it associates with vertex 2
    sf.N3        the 3rd shape function, it associates with vertex 3
    sf.N4        the 4th shape function, it associates with vertex 4
    sf.N5        the 5th shape function, it associates with vertex 5

    sf.Nmatx     a 2x10 matrix of shape functions for the pentagon

    # first derivatives of the shape functions

    # first partial derivative: d N_i / dXi, i = 1..5
    sf.dN1dXi    gradient of the 1st shape function wrt the xi coordinate
    sf.dN2dXi    gradient of the 2nd shape function wrt the xi coordinate
    sf.dN3dXi    gradient of the 3rd shape function wrt the xi coordinate
    sf.dN4dXi    gradient of the 4th shape function wrt the xi coordinate
    sf.dN5dXi    gradient of the 5th shape function wrt the xi coordinate

    # first partial derivative: d N_i / dEta, i = 1..5
    sf.dN1dEta   gradient of the 1st shape function wrt the eta coordinate
    sf.dN2dEta   gradient of the 2nd shape function wrt the eta coordinate
    sf.dN3dEta   gradient of the 3rd shape function wrt the eta coordinate
    sf.dN4dEta   gradient of the 4th shape function wrt the eta coordinate
    sf.dN5dEta   gradient of the 5th shape function wrt the eta coordinate

Reference
    1) Dasgupta, G., "Interpolants within convex polygons: Wachspress shape
       funtions", Journal of Aerospace Engineering, vol. 16, 2003, pp. 1-8.
"""


class shapeFunction(object):

    def __init__(self, xi, eta):
        # construct the five, exported, shape functions for location (xi, eta)

        # create the x and y vertex vectors in their natural coordinate frame
        self.det = 0
        xiVec = np.zeros(6, dtype=float)
        etaVec = np.zeros(6, dtype=float)
        for i in range(1, 6):
            xiVec[i] = cos(2*(i-1)*pi/5.0 + pi/2.0)
            etaVec[i] = sin(2*(i-1)*pi/5.0 + pi/2.0)
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
            kappa[i] = kappa[i-1] * ((aVec[i+1] * (xiVec[i-1] - xiVec[i]) +
                                      bVec[i+1] * (etaVec[i-1] - etaVec[i])) /
                                     (aVec[i-1] * (xiVec[i] - xiVec[i-1]) +
                                      bVec[i-1] * (etaVec[i] - etaVec[i-1])))

        # create the alpha coefficients for the numerator's polynomial
        alpha = np.zeros((10, 6), dtype=float)
        for i in range(1, 6):
            alpha[0, i] = 1.0
            alpha[1, i] = -(aVec[i+1] + aVec[i+2] + aVec[i+3])
            alpha[2, i] = -(bVec[i+1] + bVec[i+2] + bVec[i+3])
            alpha[3, i] = (aVec[i+1] * aVec[i+2] + aVec[i+2] * aVec[i+3] +
                           aVec[i+3] * aVec[i+1])
            alpha[4, i] = (aVec[i+1] * (bVec[i+2] + bVec[i+3]) +
                           aVec[i+2] * (bVec[i+1] + bVec[i+3]) +
                           aVec[i+3] * (bVec[i+1] + bVec[i+2]))
            alpha[5, i] = (bVec[i+1] * bVec[i+2] + bVec[i+2] * bVec[i+3] +
                           bVec[i+3] * bVec[i+1])
            alpha[6, i] = -aVec[i+1] * aVec[i+2] * aVec[i+3]
            alpha[7, i] = -(aVec[i+1] * aVec[i+2] * bVec[i+3] +
                            aVec[i+1] * bVec[i+2] * aVec[i+3] +
                            bVec[i+1] * aVec[i+2] * aVec[i+3])
            alpha[8, i] = -(aVec[i+1] * bVec[i+2] * bVec[i+3] +
                            bVec[i+1] * aVec[i+2] * bVec[i+3] +
                            bVec[i+1] * bVec[i+2] * aVec[i+3])
            alpha[9, i] = -bVec[i+1] * bVec[i+2] * bVec[i+3]

        # create the beta coefficients for the denominator's polynomial
        beta = np.zeros(6, dtype=float)
        for j in range(6):
            for i in range(1, 6):
                beta[j] = beta[j] + alpha[j, i] * kappa[i]

        # create the polynomials for the numerators of these shape functions
        aPoly = np.zeros(6, dtype=float)
        for i in range(1, 6):
            aPoly[i] = (alpha[0, i] + alpha[1, i] * xi + alpha[2, i] * eta +
                        alpha[3, i] * xi**2 + alpha[4, i] * xi * eta +
                        alpha[5, i] * eta**2 + alpha[6, i] * xi**3 +
                        alpha[7, i] * xi**2 * eta + alpha[8, i] * xi * eta**2 +
                        alpha[9, i] * eta**3)

        # create the polynomial for the denominator of these shape functions
        bPoly = (beta[0] + beta[1] * xi * beta[2] * eta + beta[3] * xi**2 +
                 beta[4] * xi * eta + beta[5] * eta**2)

        # create the five exported shape functions
        self.N5 = kappa[1] * aPoly[1] / bPoly
        self.N1 = kappa[2] * aPoly[2] / bPoly
        self.N2 = kappa[3] * aPoly[3] / bPoly
        self.N3 = kappa[4] * aPoly[4] / bPoly
        self.N4 = kappa[5] * aPoly[5] / bPoly

        # create the 2x10 shape function matrix for a pentagon
        self.Nmatx = np.array([[self.N1, 0, self.N2, 0, self.N3, 0,
                                self.N4, 0, self.N5, 0],
                               [0, self.N1, 0, self.N2, 0, self.N3,
                                0, self.N4, 0, self.N5]])

        # determine the gradients of the numerator polynomials
        dAdXi = np.zeros(6, dtype=float)
        dAdEta = np.zeros(6, dtype=float)
        for i in range(1, 6):
            dAdXi[i] = (alpha[1, i] + 2 * alpha[3, i] * xi +
                        alpha[4, i] * eta + 3 * alpha[6, i] * xi**2 +
                        2 * alpha[7, i] * xi * eta + alpha[8, i] * eta**2)
            dAdEta[i] = (alpha[2, i] + alpha[4, i] * xi +
                         2 * alpha[5, i] * eta + alpha[7, i] * xi**2 +
                         2 * alpha[8, i] * xi * eta + 3 * alpha[9, i] * eta**2)

        # determine the gradients of the denominator polynomial
        dBdXi = beta[1] + 2 * beta[3] * xi + beta[4] * eta
        dBdEta = beta[2] + beta[4] * xi + 2 * beta[5] * eta

        # determine the numerators for the spatial derivatives
        dNumdXi = np.zeros(6, dtype=float)
        dNumdEta = np.zeros(6, dtype=float)
        for i in range(1, 6):
            dNumdXi[i] = bPoly * dAdXi[i] - dBdXi * aPoly[i]
            dNumdEta[i] = bPoly * dAdEta[i] - dBdEta * aPoly[i]

        # create the ten, exported, first derivatives of the shape functions
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

    def interpolate(self, y1, y2, y3, y4, y5):
        y = (self.N1 * y1 + self.N2 * y2 + self.N3 * y3 + self.N4 * y4 +
             self.N5 * y5)
        return y

    def detJacobian(self, x1, x2, x3, x4, x5):
        jacob = np.zeros((2, 2), dtype=float)
        if isinstance(x1, tuple):
            jacob[0, 0] = (self.dN1dXi * x1[0] + self.dN2dXi * x2[0] +
                           self.dN3dXi * x3[0] + self.dN4dXi * x4[0] +
                           self.dN5dXi * x5[0])
            jacob[0, 1] = (self.dN1dXi * x1[1] + self.dN2dXi * x2[1] +
                           self.dN3dXi * x3[1] + self.dN4dXi * x4[1] +
                           self.dN5dXi * x5[1])
            jacob[1, 0] = (self.dN1dEta * x1[0] + self.dN2dEta * x2[0] +
                           self.dN3dEta * x3[0] + self.dN4dEta * x4[0] +
                           self.dN5dEta * x5[0])
            jacob[1, 1] = (self.dN1dEta * x1[1] + self.dN2dEta * x2[1] +
                           self.dN3dEta * x3[1] + self.dN4dEta * x4[1] +
                           self.dN5dEta * x5[1])

            # determine the determinant of the Jacobian
            det = jacob[0, 0] * jacob[1, 1] - jacob[1, 0] * jacob[0, 1]
        else:
            raise RuntimeError("Each argument of shapeFunction.detJacobian " +
                               "must be a tuple of coordinates, e.g., (x, y).")
        return det

    def G(self, x1, x2, x3, x4, x5, x01, x02, x03, x04, x05):
        if isinstance(x1, tuple):
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
            disGrad[0, 0] = (self.dN1dXi * u1 + self.dN2dXi * u2 +
                             self.dN3dXi * u3 + self.dN4dXi * u4 +
                             self.dN5dXi * u5)
            disGrad[0, 1] = (self.dN1dEta * u1 + self.dN2dEta * u2 +
                             self.dN3dEta * u3 + self.dN4dEta * u4 +
                             self.dN5dEta * u5)
            disGrad[1, 0] = (self.dN1dXi * v1 + self.dN2dXi * v2 +
                             self.dN3dXi * v3 + self.dN4dXi * v4 +
                             self.dN5dXi * v5)
            disGrad[1, 1] = (self.dN1dEta * v1 + self.dN2dEta * v2 +
                             self.dN3dEta * v3 + self.dN4dEta * v4 +
                             self.dN5dEta * v5)

            # determine the current position gradient
            curGrad = np.zeros((2, 2), dtype=float)
            curGrad[0, 0] = (self.dN1dXi * x1[0] + self.dN2dXi * x2[0] +
                             self.dN3dXi * x3[0] + self.dN4dXi * x4[0] +
                             self.dN5dXi * x5[0])
            curGrad[0, 1] = (self.dN1dEta * x1[0] + self.dN2dEta * x2[0] +
                             self.dN3dEta * x3[0] + self.dN4dEta * x4[0] +
                             self.dN5dEta * x5[0])
            curGrad[1, 0] = (self.dN1dXi * x1[1] + self.dN2dXi * x2[1] +
                             self.dN3dXi * x3[1] + self.dN4dXi * x4[1] +
                             self.dN5dXi * x5[1])
            curGrad[1, 1] = (self.dN1dEta * x1[1] + self.dN2dEta * x2[1] +
                             self.dN3dEta * x3[1] + self.dN4dEta * x4[1] +
                             self.dN5dEta * x5[1])

            # determine the inverse of the current position gradient
            curGradInv = np.zeros((2, 2), dtype=float)
            det = curGrad[0, 0] * curGrad[1, 1] - curGrad[1, 0] * curGrad[0, 1]
            curGradInv[0, 0] = curGrad[1, 1] / det
            curGradInv[0, 1] = -curGrad[0, 1] / det
            curGradInv[1, 0] = -curGrad[1, 0] / det
            curGradInv[1, 1] = curGrad[0, 0] / det

            # calculate the displacement gradient
            Gmtx = np.dot(disGrad, curGradInv)
        else:
            raise RuntimeError("Each argument of shapeFunction.G must be a " +
                               "tuple of coordinates, e.g., (x, y).")
        return Gmtx

    def F(self, x1, x2, x3, x4, x5, x01, x02, x03, x04, x05):
        if isinstance(x1, tuple):
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
            disGrad[0, 0] = (self.dN1dXi * u1 + self.dN2dXi * u2 +
                             self.dN3dXi * u3 + self.dN4dXi * u4 +
                             self.dN5dXi * u5)
            disGrad[0, 1] = (self.dN1dEta * u1 + self.dN2dEta * u2 +
                             self.dN3dEta * u3 + self.dN4dEta * u4 +
                             self.dN5dEta * u5)
            disGrad[1, 0] = (self.dN1dXi * v1 + self.dN2dXi * v2 +
                             self.dN3dXi * v3 + self.dN4dXi * v4 +
                             self.dN5dXi * v5)
            disGrad[1, 1] = (self.dN1dEta * v1 + self.dN2dEta * v2 +
                             self.dN3dEta * v3 + self.dN4dEta * v4 +
                             self.dN5dEta * v5)

            # determine the reference position gradient
            refGrad = np.zeros((2, 2), dtype=float)
            refGrad[0, 0] = (self.dN1dXi * x01[0] + self.dN2dXi * x02[0] +
                             self.dN3dXi * x03[0] + self.dN4dXi * x04[0] +
                             self.dN5dXi * x05[0])
            refGrad[0, 1] = (self.dN1dEta * x01[0] + self.dN2dEta * x02[0] +
                             self.dN3dEta * x03[0] + self.dN4dEta * x04[0] +
                             self.dN5dEta * x05[0])
            refGrad[1, 0] = (self.dN1dXi * x01[1] + self.dN2dXi * x02[1] +
                             self.dN3dXi * x03[1] + self.dN4dXi * x04[1] +
                             self.dN5dXi * x05[1])
            refGrad[1, 1] = (self.dN1dEta * x01[1] + self.dN2dEta * x02[1] +
                             self.dN3dEta * x03[1] + self.dN4dEta * x04[1] +
                             self.dN5dEta * x05[1])

            # determine the inverse of the reference position gradient
            refGradInv = np.zeros((2, 2), dtype=float)
            det = refGrad[0, 0] * refGrad[1, 1] - refGrad[1, 0] * refGrad[0, 1]
            refGradInv[0, 0] = refGrad[1, 1] / det
            refGradInv[0, 1] = -refGrad[0, 1] / det
            refGradInv[1, 0] = -refGrad[1, 0] / det
            refGradInv[1, 1] = refGrad[0, 0] / det

            # calculate the deformation gradient
            Fmtx = np.identity(2) + np.dot(disGrad, refGradInv)
        else:
            raise RuntimeError("Each argument of shapeFunction.F must be a " +
                               "tuple of coordinates, e.g., (x, y).")
        return Fmtx


"""
Examples of shape functions and how to create them are listed below:

objects

    sfV1       shape functions at vertex 1
    sfV2       shape functions at vertex 2
    sfV3       shape functions at vertex 3
    sfV4       shape functions at vertex 4
    sfV5       shape functions at vertex 5

    sfCG       shape functions for the center of gravity, i.e., at the centroid

    sfGPt1O1   shape fns at Gauss point 1 for 1st-order accurate integration

    sfGPt1O3   shape fns at Gauss point 1 for 3rd-order accurate integration
    sfGPt2O3   shape fns at Gauss point 2 for 3rd-order accurate integration
    sfGPt3O3   shape fns at Gauss point 3 for 3rd-order accurate integration
    sfGPt4O3   shape fns at Gauss point 4 for 3rd-order accurate integration

    sfGPt3O5   shape fns at Gauss point 1 for 5th-order accurate integration
    sfGPt2O5   shape fns at Gauss point 2 for 5th-order accurate integration
    sfGPt3O5   shape fns at Gauss point 3 for 5th-order accurate integration
    sfGPt4O5   shape fns at Gauss point 4 for 5th-order accurate integration
    sfGPt5O5   shape fns at Gauss point 5 for 5th-order accurate integration
    sfGPt6O5   shape fns at Gauss point 6 for 5th-order accurate integration
    sfGPt7O5   shape fns at Gauss point 7 for 5th-order accurate integration

The natural coordinates used to create these objects are:

    v1Xi, v1Eta       natural coordinates at vertex 1
    v2Xi, v2Eta       natural coordinates at vertex 2
    v3Xi, v3Eta       natural coordinates at vertex 3
    v4Xi, v4Eta       natural coordinates at vertex 4
    v5Xi, v5Eta       natural coordinates at vertex 5

    gPt1O1Xi, gPt1O1Eta   natural coords at Gauss point 1 for 1st-order method

    gPt1O3Xi, gPt1O3Eta   natural coords at Gauss point 1 for 3rd-order method
    gPt2O3Xi, gPt2O3Eta   natural coords at Gauss point 2 for 3rd-order method
    gPt3O3Xi, gPt3O3Eta   natural coords at Gauss point 3 for 3rd-order method
    gPt4O3Xi, gPt4O3Eta   natural coords at Gauss point 4 for 3rd-order method

    gPt1O5Xi, gPt1O5Eta   natural coords at Gauss point 1 for 5th-order method
    gPt2O5Xi, gPt2O5Eta   natural coords at Gauss point 2 for 5th-order method
    gPt3O5Xi, gPt3O5Eta   natural coords at Gauss point 3 for 5th-order method
    gPt4O5Xi, gPt4O5Eta   natural coords at Gauss point 4 for 5th-order method
    gPt5O5Xi, gPt5O5Eta   natural coords at Gauss point 5 for 5th-order method
    gPt6O5Xi, gPt6O5Eta   natural coords at Gauss point 6 for 5th-order method
    gPt7O5Xi, gPt7O5Eta   natural coords at Gauss point 7 for 5th-order method

# shape functions at the five vertices of a pentagon

v1Xi = cos(pi / 2.0)
v1Eta = sin(pi / 2.0)
sfV1 = shapeFn(v1Xi, v1Eta)

v2Xi = cos(9.0 * pi / 10.0)
v2Eta = sin(9.0 * pi / 10.0)
sfV2 = shapeFn(v2Xi, v2Eta)

v3Xi = cos(13.0 * pi / 10.0)
v3Eta = sin(13.0 * pi / 10.0)
sfV3 = shapeFn(v3Xi, v3Eta)

v4Xi = cos(17.0 * pi / 10.0)
v4Eta = cos(17.0 * pi / 10.0)
sfV4 = shapeFn(v4Xi, v4Eta)

v5Xi = cos(21.0 * pi / 10.0)
v5Eta = sin(21.0 * pi / 10.0)
sfV5 = shapeFn(v5Xi, v5Eta)

# shape functions at the centroid of the pentagon

zero = 0.0000000000000000
sfCG = shapeFn(zero, zero)

# shape functions at the Gauss point for 1st-order accurate integrations

gPt1O1Xi = 0.0000000000000000
gPt1O1Eta = 0.0000000000000000
sfGPt1O1 = shapeFn(gPt1O1Xi, gPt1O1Eta)

# shape functions at the Gauss points for 3rd-order accurate integrations

gPt1O3Xi = -0.0349156305831802
gPt1O3Eta = 0.6469731019095136
sfGPt1O3 = shapeFn(gPt1O3Xi, gPt1O3Eta)

gPt2O3Xi = -0.5951653065516678
gPt2O3Eta = -0.0321196846022659
sfGPt2O3 = shapeFn(gPt2O3Xi, gPt2O3Eta)

gPt3O3Xi = 0.0349156305831798
gPt3O3Eta = -0.6469731019095134
sfGPt3O3 = shapeFn(gPt3O3Xi, gPt3O3Eta)

gPt4O3Xi = 0.5951653065516677
gPt4O3Eta = 0.0321196846022661
sfGPt4O3 = shapeFn(gPt4O3Xi, gPt4O3Eta)

# shape functions at the Gauss points for 5th-order accurate integrations

gPt1O5Xi = -0.0000000000000000
gPt1O5Eta = -0.0000000000000002
sfGPt3O5 = shapeFn(gPt1O5Xi, gPt1O5Eta)

gPt2O5Xi = -0.1351253857178451
gPt2O5Eta = 0.7099621260052327
sfGPt2O5 = shapeFn(gPt2O5Xi, gPt2O5Eta)

gPt3O5Xi = -0.6970858746672087
gPt3O5Eta = 0.1907259121533272
sfGPt3O5 = shapeFn(gPt3O5Xi, gPt3O5Eta)

gPt4O5Xi = -0.4651171392611024
gPt4O5Eta = -0.5531465782166917
sfGPt4O5 = shapeFn(gPt4O5Xi, gPt4O5Eta)

gPt5O5Xi = 0.2842948078559476
gPt5O5Eta = -0.6644407817506509
sfGPt5O5 = shapeFn(gPt5O5Xi, gPt5O5Eta)

gPt6O5Xi = 0.7117958231685716
gPt6O5Eta = -0.1251071394727008
sfGPt6O5 = shapeFn(gPt6O5Xi, gPt6O5Eta)

gPt7O5Xi = 0.5337947578638855
gPt7O5Eta = 0.4872045224587945
sfGPt7O5 = shapeFn(gPt7O5Xi, gPt7O5Eta)
"""
