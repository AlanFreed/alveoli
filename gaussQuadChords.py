#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import sqrt
import numpy as np
from gaussQuadratures import GaussQuadrature as GaussQuad

"""
Module gaussQuadChords.py implements a Gauss quadrature rule for chords.

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
__date__ = "07-10-2020"
__update__ = "07-17-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

r"""
A listing of changes made wrt version release can be found at the end of file.

Chords have 2 nodes and 2 Gauss points that in their natural co-ordinates

    N1    G1              G2    N2
    X-----X----------------X-----X
   -1              0             1

Overview of module gaussQuadChords.py:

Module gaussQuadChords.py exports class GaussQuadrature that allows for the
interpolation and extrapolation of fields between nodal and Gauss points.  It
also provides the weights and nodes (co-ordinates) belonging a Gauss quadrature
rule that is suitable for integrating fields over the length of a chord.

constructor

    gq = GaussQuadrature()
    output
        gq  is a new instance of the class GaussQuadrature for a chord

property

    gPts = gq.gaussPoints()
    output
        gPts is the number of Gauss points (and nodes, in our implementation)

inherited methods

    yG1, yG2 = gq.interpolate(yN1, yN2)
    inputs
        yN1 is physical field y of arbitrary type located at nodal point 1
        yN2 is physical field y of arbitrary type located at nodal point 2
    outputs
        yG1 is physical field y of arbitrary type located at Gauss point 1
        yG2 is physical field y of arbitrary type located at Gauss point 2
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    yN1, yN2 = gq.extrapolate(yG1, yG2)
    inputs
        yG1 is physical field y of arbitrary type located at Gauss point 1
        yG2 is physical field y of arbitrary type located at Gauss point 2
    outputs
        yN1 is physical field y of arbitrary type located at nodal point 1
        yN2 is physical field y of arbitrary type located at nodal point 2
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    coord = gq.coordinates(atGaussPt)
    input
        atGaussPt   is the Gauss point at which the co-ordinates are sought
    output
        coord       are the natural co-ordinates at the specified Gauss point
    Input atGaussPt must be either 1 or 2 for a chord.

    wgt = gq.weight(atGaussPt)
    input
        atGaussPt   is the Gauss point at which the weight is sought
    output
        wgt         is the weight of quadrature at the specified Gauss point
    Input atGaussPt must be either 1 or 2 for a chord.
"""


class GaussQuadrature(GaussQuad):

    # constructor

    def __init__(self):
        super(GaussQuadrature, self).__init__()
        sqrt3 = sqrt(3.0)
        self._coordinates = {
            1: (-1.0 / sqrt3,),
            2: (1.0 / sqrt3,)
            }
        self._weights = {
            1: 1.0,
            2: 1.0
            }
        self._interpCoef = np.zeros((2, 2), dtype=float)
        self._interpCoef[0, 0] = (3.0 + sqrt3) / 6.0
        self._interpCoef[0, 1] = (3.0 - sqrt3) / 6.0
        self._interpCoef[1, 0] = (3.0 + sqrt3) / 6.0
        self._interpCoef[1, 1] = (3.0 - sqrt3) / 6.0
        self._extrapCoef = np.zeros((2, 2), dtype=float)
        self._extrapCoef[0, 0] = (sqrt3 + 3.0) / (2.0 * sqrt3)
        self._extrapCoef[0, 1] = (sqrt3 - 3.0) / (2.0 * sqrt3)
        self._extrapCoef[1, 0] = (sqrt3 - 3.0) / (2.0 * sqrt3)
        self._extrapCoef[1, 1] = (sqrt3 + 3.0) / (2.0 * sqrt3)
        return  # a new instance of a Gauss quadrature rule for chords

    # property

    def gaussPoints(self):
        gPts = 2
        return gPts

    # methods

    def interpolate(self, yN1, yN2):
        if type(yN1) == type(yN2):
            yG1 = self._interpCoef[0, 0] * yN1 + self._interpCoef[0, 1] * yN2
            yG2 = self._interpCoef[1, 0] * yN1 + self._interpCoef[1, 1] * yN2
        else:
            raise RuntimeError("Arguments for interpolation are not of the "
                               + "same type.")
        return yG1, yG2

    def extrapolate(self, yG1, yG2):
        if type(yG1) == type(yG2):
            yN1 = self._extrapCoef[0, 0] * yG1 + self._extrapCoef[0, 1] * yG2
            yN2 = self._extrapCoef[1, 0] * yG1 + self._extrapCoef[1, 1] * yG2
        else:
            raise RuntimeError("Arguments for extrapolation are not of the "
                               + "same type.")
        return yN1, yN2

    def coordinates(self, atGaussPt):
        if atGaussPt > 0 and atGaussPt < 3:
            return self._coordinates[atGaussPt]
        else:
            raise RuntimeError("atGaussPt can take on values of: 1, 2.")

    def weight(self, atGaussPt):
        if atGaussPt > 0 and atGaussPt < 3:
            return self._weights[atGaussPt]
        else:
            raise RuntimeError("atGaussPt can take on values of: 1, 2.")


"""
Changes made in version "1.0.0":

Original version
"""
