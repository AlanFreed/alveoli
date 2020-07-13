#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gaussQuadratures import GaussQuadrature as GaussQuad

"""
Module gaussQuadTetrahedra.py implements Gauss quadrature for tetrahedra.

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
__update__ = "07-08-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

r"""
A listing of changes made wrt version release can be found at the end of file.

A tetrahedron, in its natrual co-ordinates, has nodal points located at:
    N1 = (0, 0, 0)
    N2 = (1, 0, 0)
    N3 = (0, 1, 0)
    N4 = (0, 0, 1)
whose associated Gauss points index as: G1 is the closest Gauss point to N1,
G2 is the closest Gauss point to N2, etc.

Overview of module gaussQuadTetrahedra.py:

Module gaussQuadTetrahedra.py exports class GaussQuadrature that allows for the
interpolation and extrapolation of fields between nodal and Gauss points.  It
also provides the weights and nodes (co-ordinates) belonging a Gauss quadrature
rule that is suitable for integrating fields over the volume of a tetrahedron.

constructor

    gq = GaussQuadrature()
    output
        gq  is a new instance of class GaussQuadrature for a tetrahedron

methods

    yG1, yG2, yG3, yG4 = gq.interpolate(yN1, yN2, yN3, yN4)
    inputs
        yN1 is physical field y of arbitrary type located at nodal point 1
        yN2 is physical field y of arbitrary type located at nodal point 2
        yN3 is physical field y of arbitrary type located at nodal point 3
        yN4 is physical field y of arbitrary type located at nodal point 4
    outputs
        yG1 is physical field y of arbitrary type located at Gauss point 1
        yG2 is physical field y of arbitrary type located at Gauss point 2
        yG3 is physical field y of arbitrary type located at Gauss point 3
        yG4 is physical field y of arbitrary type located at Gauss point 4
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    yN1, yN2, yN3, yN4 = gq.extrapolate(yG1, yG2, yG3, yG4)
    inputs
        yG1 is physical field y of arbitrary type located at Gauss point 1
        yG2 is physical field y of arbitrary type located at Gauss point 2
        yG3 is physical field y of arbitrary type located at Gauss point 3
        yG4 is physical field y of arbitrary type located at Gauss point 4
    outputs
        yN1 is physical field y of arbitrary type located at nodal point 1
        yN2 is physical field y of arbitrary type located at nodal point 2
        yN3 is physical field y of arbitrary type located at nodal point 3
        yN4 is physical field y of arbitrary type located at nodal point 4
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    gPts = gq.gaussPoints()
    output
        gPts is the number of Gauss points (and nodes, in our implementation)

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
        self._gaussPts = 4
        a = 0.1381966011250105
        b = 0.5854101966249685
        self._coordinates = {
            1: (a, a, a),
            2: (b, a, a),
            3: (a, b, a),
            4: (a, a, b)
            }
        self._weights = {
            1: 1.0 / 24.0,
            2: 1.0 / 24.0,
            3: 1.0 / 24.0,
            4: 1.0 / 24.0
            }
        self._interpCoef = np.zeros((4, 4), dtype=float)
        self._interpCoef[0, 0] = 1.0 - 3.0 * a
        self._interpCoef[0, 1] = a
        self._interpCoef[0, 2] = a
        self._interpCoef[0, 3] = a
        self._interpCoef[1, 0] = 1.0 - 2.0 * a - b
        self._interpCoef[1, 1] = b
        self._interpCoef[1, 2] = a
        self._interpCoef[1, 3] = a
        self._interpCoef[2, 0] = 1.0 - 2.0 * a - b
        self._interpCoef[2, 1] = a
        self._interpCoef[2, 2] = b
        self._interpCoef[2, 3] = a
        self._interpCoef[3, 0] = 1.0 - 2.0 * a - b
        self._interpCoef[3, 1] = a
        self._interpCoef[3, 2] = a
        self._interpCoef[3, 3] = b
        self._extrapCoef = np.zeros((4, 4), dtype=float)
        self._extrapCoef[0, 0] = (2.0 * a + b) / (b - a)
        self._extrapCoef[0, 1] = -a / (b - a)
        self._extrapCoef[0, 2] = -a / (b - a)
        self._extrapCoef[0, 3] = -a / (b - a)
        self._extrapCoef[1, 0] = (2.0 * a + b - 1.0) / (b - a)
        self._extrapCoef[1, 1] = (1.0 - a) / (b - a)
        self._extrapCoef[1, 2] = -a / (b - a)
        self._extrapCoef[1, 3] = -a / (b - a)
        self._extrapCoef[2, 0] = (2.0 * a + b - 1.0) / (b - a)
        self._extrapCoef[2, 1] = -a / (b - a)
        self._extrapCoef[2, 2] = (1.0 - a) / (b - a)
        self._extrapCoef[2, 3] = -a / (b - a)
        self._extrapCoef[3, 0] = (2.0 * a + b - 1.0) / (b - a)
        self._extrapCoef[3, 1] = -a / (b - a)
        self._extrapCoef[3, 2] = -a / (b - a)
        self._extrapCoef[3, 3] = (1.0 - a) / (b - a)
        return  # a new instance of a Gauss quadrature rule for tetrahedra

    def interpolate(self, yN1, yN2, yN3, yN4):
        if type(yN1) == type(yN2) and type(yN2) == type(yN3):
            yG1 = (self._interpCoef[0, 0] * yN1
                   + self._interpCoef[0, 1] * yN2
                   + self._interpCoef[0, 2] * yN3
                   + self._interpCoef[0, 3] * yN4)
            yG2 = (self._interpCoef[1, 0] * yN1
                   + self._interpCoef[1, 1] * yN2
                   + self._interpCoef[1, 2] * yN3
                   + self._interpCoef[1, 3] * yN4)
            yG3 = (self._interpCoef[2, 0] * yN1
                   + self._interpCoef[2, 1] * yN2
                   + self._interpCoef[2, 2] * yN3
                   + self._interpCoef[2, 3] * yN4)
            yG4 = (self._interpCoef[3, 0] * yN1
                   + self._interpCoef[3, 1] * yN2
                   + self._interpCoef[3, 2] * yN3
                   + self._interpCoef[3, 3] * yN4)
        else:
            raise RuntimeError("Arguments for interpolation are not of the "
                               + "same type.")
        return yG1, yG2, yG3, yG4

    def extrapolate(self, yG1, yG2, yG3, yG4):
        if (type(yG1) == type(yG2) and type(yG2) == type(yG3)
           and type(yG3) == type(yG4)):
            yN1 = (self._extrapCoef[0, 0] * yG1
                   + self._extrapCoef[0, 1] * yG2
                   + self._extrapCoef[0, 2] * yG3
                   + self._extrapCoef[0, 3] * yG4)
            yN2 = (self._extrapCoef[1, 0] * yG1
                   + self._extrapCoef[1, 1] * yG2
                   + self._extrapCoef[1, 2] * yG3
                   + self._extrapCoef[1, 3] * yG4)
            yN3 = (self._extrapCoef[2, 0] * yG1
                   + self._extrapCoef[2, 1] * yG2
                   + self._extrapCoef[2, 2] * yG3
                   + self._extrapCoef[2, 3] * yG4)
            yN4 = (self._extrapCoef[3, 0] * yG1
                   + self._extrapCoef[3, 1] * yG2
                   + self._extrapCoef[3, 2] * yG3
                   + self._extrapCoef[3, 3] * yG4)
        else:
            raise RuntimeError("Arguments for extrapolation are not of the "
                               + "same type.")
        return yN1, yN2, yN3, yN4

    def gaussPoints(self):
        return self._gaussPts

    def coordinates(self, atGaussPt):
        if atGaussPt > 0 and atGaussPt < 5:
            return self._coordinates[atGaussPt]
        else:
            raise RuntimeError("atGaussPt can take on values of: 1, 2, 3, 4.")

    def weight(self, atGaussPt):
        if atGaussPt > 0 and atGaussPt < 5:
            return self._weights[atGaussPt]
        else:
            raise RuntimeError("atGaussPt can take on values of: 1, 2, 3, 4.")


"""
Changes made in version "1.0.0":

Original version
"""
