#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import cos, pi, sin
import numpy as np
from gaussQuadratures import GaussQuadrature as GaussQuad

"""
Module gaussQuadPentagons.py implements Gauss quadrature for pentagons.

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
__date__ = "07-07-2020"
__update__ = "07-13-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

r"""
A listing of changes made wrt version release can be found at the end of file.

The five nodal points of a pentagon are numbered according to the graphic

                            N1
                          /   \
                       /         \
                     /             \
                  N2                 N5
                   \                 /
                    \               /
                     \             /
                     N3 --------- N4
where
    N1 = (xi_1, eta_1)
    N2 = (xi_2, eta_2)
    N3 = (xi_3, eta_3)
    N4 = (xi_4, eta_4)
    N5 = (xi_5, eta_5)
with
    xi_i  = cos(2*(i-1)*pi/5 + pi/2),     i = 1,2,..,5
    eta_i = sin(2*(i-1)*pi/5 + pi/2),     i = 1,2,..,5
whose associated Gauss points index as: G1 is the closest Gauss point to N1,
G2 is the closest Gauss point to N2, etc.

Overview of module gaussQuadPentagons.py:

Module gaussQuadPentagons.py exports class GaussQuadrature that allows for the
interpolation and extrapolation of fields between nodal and Gauss points.  It
also provides the weights and nodes (co-ordinates) belonging a Gauss quadrature
rule that is suitable for integrating fields over the area of a pentagon.

constructor

    gq = GaussQuadrature()
    output
        gq  is a new instance of the class GaussQuadrature for a pentagon

methods

    yG1, yG2, yG3, yG4, yG5 = gq.interpolate(yN1, yN2, yN3, yN4, yN5)
    inputs
        yN1 is physical field y of arbitrary type located at nodal point 1
        yN2 is physical field y of arbitrary type located at nodal point 2
        yN3 is physical field y of arbitrary type located at nodal point 3
        yN4 is physical field y of arbitrary type located at nodal point 4
        yN5 is physical field y of arbitrary type located at nodal point 5
    outputs
        yG1 is physical field y of arbitrary type located at Gauss point 1
        yG2 is physical field y of arbitrary type located at Gauss point 2
        yG3 is physical field y of arbitrary type located at Gauss point 3
        yG4 is physical field y of arbitrary type located at Gauss point 4
        yG5 is physical field y of arbitrary type located at Gauss point 5
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    yN1, yN2, yN3, yN4, yN5 = gq.extrapolate(yG1, yG2, yG3, yG4, yG5)
    inputs
        yG1 is physical field y of arbitrary type located at Gauss point 1
        yG2 is physical field y of arbitrary type located at Gauss point 2
        yG3 is physical field y of arbitrary type located at Gauss point 3
        yG4 is physical field y of arbitrary type located at Gauss point 4
        yG5 is physical field y of arbitrary type located at Gauss point 5
    outputs
        yN1 is physical field y of arbitrary type located at nodal point 1
        yN2 is physical field y of arbitrary type located at nodal point 2
        yN3 is physical field y of arbitrary type located at nodal point 3
        yN4 is physical field y of arbitrary type located at nodal point 4
        yN5 is physical field y of arbitrary type located at nodal point 5
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    gPts = gq.gaussPoints()
    output
        gPts    the number of Gauss points (and nodes, in our implementation)

    coord = gq.coordinates(atGaussPt)
    input
        atGaussPt   is the Gauss point at which the co-ordinates are sought
    output
        coord       are the natural co-ordinates at the specified Gauss point

    wgt = gq.weight(atGaussPt)
    input
        atGaussPt   is the Gauss point at which the co-ordinates are sought
    output
        wgt         is the weight of quadrature at the specified Gauss point
"""


class GaussQuadrature(GaussQuad):

    def __init__(self):
        super(GaussQuadrature, self).__init__()
        self._gaussPts = 5
        # the sines and cosines that appear in the geometry of a pentagon
        sin1 = sin(pi/10.0)
        cos1 = cos(pi/10.0)
        sin3 = sin(3.0*pi/10.0)
        cos3 = cos(3.0*pi/10.0)
        # Distance along radial lines to nodes that locates the centroids for
        # the four-sided polygons that make up a pentagon.
        ell = (1.0 + sin3) / (3.0 * sin3)
        # Assign the quadrature points
        self._coordinates = {
            1: (0.0, ell),
            2: (-ell*cos1, ell*sin1),
            3: (-ell*cos3, -ell*sin3),
            4: (ell*cos3, -ell*sin3),
            5: (ell*cos1, ell*sin1)
            }
        # Assign the quadrature weights
        w = sin3 * cos3
        self._weights = {
            1: w,
            2: w,
            3: w,
            4: w,
            5: w
            }
        # Assign the interpolation coefficients
        # These come from the Wachspress shape functions
        a = 0.6901471673508344
        b = 0.1367959452017669
        c = 0.0181304711228159
        self._interpCoef = np.zeros((5, 5), dtype=float)
        self._interpCoef[0, 0] = a
        self._interpCoef[0, 1] = b
        self._interpCoef[0, 2] = c
        self._interpCoef[0, 3] = c
        self._interpCoef[0, 4] = b
        self._interpCoef[1, 0] = b
        self._interpCoef[1, 1] = a
        self._interpCoef[1, 2] = b
        self._interpCoef[1, 3] = c
        self._interpCoef[1, 4] = c
        self._interpCoef[2, 0] = c
        self._interpCoef[2, 1] = b
        self._interpCoef[2, 2] = a
        self._interpCoef[2, 3] = b
        self._interpCoef[2, 4] = c
        self._interpCoef[3, 0] = c
        self._interpCoef[3, 1] = c
        self._interpCoef[3, 2] = b
        self._interpCoef[3, 3] = a
        self._interpCoef[3, 4] = b
        self._interpCoef[4, 0] = b
        self._interpCoef[4, 1] = c
        self._interpCoef[4, 2] = c
        self._interpCoef[4, 3] = b
        self._interpCoef[4, 4] = a
        # Assign the extrapolation coefficients
        det = a**2 - (a + b) * b - (a - 3*b) * c - c**2
        x = (a**2 + (a - b) * (b + c) - c**2) / det
        y = ((b + c) * c - (a + b) * b) / det
        z = (b**2 - (a - b) * c - c**2) / det
        self._extrapCoef = np.zeros((5, 5), dtype=float)
        self._extrapCoef[0, 0] = x
        self._extrapCoef[0, 1] = y
        self._extrapCoef[0, 2] = z
        self._extrapCoef[0, 3] = z
        self._extrapCoef[0, 4] = y
        self._extrapCoef[1, 0] = y
        self._extrapCoef[1, 1] = x
        self._extrapCoef[1, 2] = y
        self._extrapCoef[1, 3] = z
        self._extrapCoef[1, 4] = z
        self._extrapCoef[2, 0] = z
        self._extrapCoef[2, 1] = y
        self._extrapCoef[2, 2] = x
        self._extrapCoef[2, 3] = y
        self._extrapCoef[2, 4] = z
        self._extrapCoef[3, 0] = z
        self._extrapCoef[3, 1] = z
        self._extrapCoef[3, 2] = y
        self._extrapCoef[3, 3] = x
        self._extrapCoef[3, 4] = y
        self._extrapCoef[4, 0] = y
        self._extrapCoef[4, 1] = z
        self._extrapCoef[4, 2] = z
        self._extrapCoef[4, 3] = y
        self._extrapCoef[4, 4] = x
        return  # a new instance of a Gauss quadrature rule for pentagons

    def interpolate(self, yN1, yN2, yN3, yN4, yN5):
        if (type(yN1) == type(yN2) and type(yN2) == type(yN3)
           and type(yN3) == type(yN4) and type(yN4) == type(yN5)):
            yG1 = (self._interpCoef[0, 0] * yN1
                   + self._interpCoef[0, 1] * yN2
                   + self._interpCoef[0, 2] * yN3
                   + self._interpCoef[0, 3] * yN4
                   + self._interpCoef[0, 4] * yN5)
            yG2 = (self._interpCoef[1, 0] * yN1
                   + self._interpCoef[1, 1] * yN2
                   + self._interpCoef[1, 2] * yN3
                   + self._interpCoef[1, 3] * yN4
                   + self._interpCoef[1, 4] * yN5)
            yG3 = (self._interpCoef[2, 0] * yN1
                   + self._interpCoef[2, 1] * yN2
                   + self._interpCoef[2, 2] * yN3
                   + self._interpCoef[2, 3] * yN4
                   + self._interpCoef[2, 4] * yN5)
            yG4 = (self._interpCoef[3, 0] * yN1
                   + self._interpCoef[3, 1] * yN2
                   + self._interpCoef[3, 2] * yN3
                   + self._interpCoef[3, 3] * yN4
                   + self._interpCoef[3, 4] * yN5)
            yG5 = (self._interpCoef[4, 0] * yN1
                   + self._interpCoef[4, 1] * yN2
                   + self._interpCoef[4, 2] * yN3
                   + self._interpCoef[4, 3] * yN4
                   + self._interpCoef[4, 4] * yN5)
        else:
            raise RuntimeError("Arguments for interpolation are not of the "
                               + "same type.")
        return yG1, yG2, yG3, yG4, yG5

    def extrapolate(self, yG1, yG2, yG3, yG4, yG5):
        if (type(yG1) == type(yG2) and type(yG2) == type(yG3)
           and type(yG3) == type(yG4) and type(yG4) == type(yG5)):
            yN1 = (self._extrapCoef[0, 0] * yG1
                   + self._extrapCoef[0, 1] * yG2
                   + self._extrapCoef[0, 2] * yG3
                   + self._extrapCoef[0, 3] * yG4
                   + self._extrapCoef[0, 4] * yG5)
            yN2 = (self._extrapCoef[1, 0] * yG1
                   + self._extrapCoef[1, 1] * yG2
                   + self._extrapCoef[1, 2] * yG3
                   + self._extrapCoef[1, 3] * yG4
                   + self._extrapCoef[1, 4] * yG5)
            yN3 = (self._extrapCoef[2, 0] * yG1
                   + self._extrapCoef[2, 1] * yG2
                   + self._extrapCoef[2, 2] * yG3
                   + self._extrapCoef[2, 3] * yG4
                   + self._extrapCoef[2, 4] * yG5)
            yN4 = (self._extrapCoef[3, 0] * yG1
                   + self._extrapCoef[3, 1] * yG2
                   + self._extrapCoef[3, 2] * yG3
                   + self._extrapCoef[3, 3] * yG4
                   + self._extrapCoef[3, 4] * yG5)
            yN5 = (self._extrapCoef[4, 0] * yG1
                   + self._extrapCoef[4, 1] * yG2
                   + self._extrapCoef[4, 2] * yG3
                   + self._extrapCoef[4, 3] * yG4
                   + self._extrapCoef[4, 4] * yG5)
        else:
            raise RuntimeError("Arguments for extrapolation are not of the "
                               + "same type.")
        return yN1, yN2, yN3, yN4, yN5

    def gaussPoints(self):
        return self._gaussPts

    def coordinates(self, atGaussPt):
        if atGaussPt > 0 and atGaussPt < 6:
            return self._coordinates[atGaussPt]
        else:
            raise RuntimeError("atGaussPt can take on values of: 1, .., 5.")

    def weight(self, atGaussPt):
        if atGaussPt > 0 and atGaussPt < 6:
            return self._weights[atGaussPt]
        else:
            raise RuntimeError("atGaussPt can take on values of: 1, .., 5.")


"""
Changes made in version "1.0.0":

Original version
"""
