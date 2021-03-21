#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gaussQuadratures import GaussQuadrature as GaussQuad

"""
Module gaussQuadTriangles.py implements Gauss quadrature for triangles.

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
__update__ = "07-17-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

r"""
A listing of changes made wrt version release can be found at the end of file.

Triangles have 3 nodes and 3 Gauss points that in their natural co-ordinates

            N3
    (0, 1)  |\
            |  \
            | G3 \
            |      \
            |        \
            | G1    G2 \
    (0, 0)  -------------  (1, 0)
            N1         N2

Overview of module gaussQuadTriangles.py:

Module gaussQuadTriangles.py exports class GaussQuadrature that allows for the
interpolation and extrapolation of fields between nodal and Gauss points.  It
also provides the weights and nodes (co-ordinates) belonging a Gauss quadrature
rule that is suitable for integrating fields over the area of a triangle.

constructor

    gq = GaussQuadrature()
    output
        gq  is a new instance of the class GaussQuadrature for a triangle

property

    gPts = gq.gaussPoints()
    output
        gPts is the number of Gauss points (and nodes, in our implementation)

inherited methods

    yG1, yG2, yG3 = gq.interpolate(yN1, yN2, yN3)
    inputs
        yN1 is physical field y of arbitrary type located at nodal point 1
        yN2 is physical field y of arbitrary type located at nodal point 2
        yN3 is physical field y of arbitrary type located at nodal point 3
    outputs
        yG1 is physical field y of arbitrary type located at Gauss point 1
        yG2 is physical field y of arbitrary type located at Gauss point 2
        yG3 is physical field y of arbitrary type located at Gauss point 3
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    yN1, yN2, yN3 = gq.extrapolate(yG1, yG2, yG3)
    inputs
        yG1 is physical field y of arbitrary type located at Gauss point 1
        yG2 is physical field y of arbitrary type located at Gauss point 2
        yG3 is physical field y of arbitrary type located at Gauss point 3
    outputs
        yN1 is physical field y of arbitrary type located at nodal point 1
        yN2 is physical field y of arbitrary type located at nodal point 2
        yN3 is physical field y of arbitrary type located at nodal point 3
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

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

    # constructor

    def __init__(self):
        super(GaussQuadrature, self).__init__()
        self._coordinates = {
            1: (1.0/6.0, 1.0/6.0),
            2: (2.0/3.0, 1.0/6.0),
            3: (1.0/6.0, 2.0/3.0)
            }
        self._weights = {
            1: 1.0 / 6.0,
            2: 1.0 / 6.0,
            3: 1.0 / 6.0
            }
        self._interpCoef = np.zeros((3, 3), dtype=float)
        self._interpCoef[0, 0] = 2.0 / 3.0
        self._interpCoef[0, 1] = 1.0 / 6.0
        self._interpCoef[0, 2] = 1.0 / 6.0
        self._interpCoef[1, 0] = 1.0 / 6.0
        self._interpCoef[1, 1] = 2.0 / 3.0
        self._interpCoef[1, 2] = 1.0 / 6.0
        self._interpCoef[2, 0] = 1.0 / 6.0
        self._interpCoef[2, 1] = 1.0 / 6.0
        self._interpCoef[2, 2] = 2.0 / 3.0
        self._extrapCoef = np.zeros((3, 3), dtype=float)
        self._extrapCoef[0, 0] = 5.0 / 3.0
        self._extrapCoef[0, 1] = -1.0 / 3.0
        self._extrapCoef[0, 2] = -1.0 / 3.0
        self._extrapCoef[1, 0] = -1.0 / 3.0
        self._extrapCoef[1, 1] = 5.0 / 3.0
        self._extrapCoef[1, 2] = -1.0 / 3.0
        self._extrapCoef[2, 0] = -1.0 / 3.0
        self._extrapCoef[2, 1] = -1.0 / 3.0
        self._extrapCoef[2, 2] = 5.0 / 3.0
        return  # a new instance of a Gauss quadrature rule for triangles

    # property

    def gaussPoints(self):
        gPts = 3
        return gPts

    # methods

    def interpolate(self, yN1, yN2, yN3):
        if type(yN1) == type(yN2) and type(yN2) == type(yN3):
            yG1 = (self._interpCoef[0, 0] * yN1 + self._interpCoef[0, 1] * yN2
                   + self._interpCoef[0, 2] * yN3)
            yG2 = (self._interpCoef[1, 0] * yN1 + self._interpCoef[1, 1] * yN2
                   + self._interpCoef[1, 2] * yN3)
            yG3 = (self._interpCoef[2, 0] * yN1 + self._interpCoef[2, 1] * yN2
                   + self._interpCoef[2, 2] * yN3)
        else:
            raise RuntimeError("Arguments for interpolation are not of the "
                               + "same type.")
        return yG1, yG2, yG3

    def extrapolate(self, yG1, yG2, yG3):
        if type(yG1) == type(yG2) and type(yG2) == type(yG3):
            yN1 = (self._extrapCoef[0, 0] * yG1 + self._extrapCoef[0, 1] * yG2
                   + self._extrapCoef[0, 2] * yG3)
            yN2 = (self._extrapCoef[1, 0] * yG1 + self._extrapCoef[1, 1] * yG2
                   + self._extrapCoef[1, 2] * yG3)
            yN3 = (self._extrapCoef[2, 0] * yG1 + self._extrapCoef[2, 1] * yG2
                   + self._extrapCoef[2, 2] * yG3)
        else:
            raise RuntimeError("Arguments for extrapolation are not of the "
                               + "same type.")
        return yN1, yN2, yN3

    def coordinates(self, atGaussPt):
        if atGaussPt > 0 and atGaussPt < 4:
            return self._coordinates[atGaussPt]
        else:
            raise RuntimeError("atGaussPt can take on values of: 1, 2, 3.")

    def weight(self, atGaussPt):
        if atGaussPt > 0 and atGaussPt < 4:
            return self._weights[atGaussPt]
        else:
            raise RuntimeError("atGaussPt can take on values of: 1, 2, 3.")


"""
Changes made in version "1.0.0":

Original version
"""
