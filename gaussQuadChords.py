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
__date__ = "07-07-2020"
__update__ = "07-08-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

r"""
A listing of changes made wrt version release can be found at the end of file.


Overview of module gaussQuadChords.py:

Module gaussQuadChords.py exports class GaussQuadrature which allows for the
extrapolation of fields from their Gauss points out to their nodal points,
along with providing for the weights and nodes (co-ordinates) of a Gaussian
quadrature rule suitable for integrating over the length of a chord.

constructor

    gq = GaussQuadrature()
    returns
        gq  is a new instance of the class GaussQuadrature for a chord

methods

    dY = gq.extrapolate(y1, y2)
    inputs
        y1  is a physical field of arbitrary type located at Gauss point 1
        y2  is a physical field of arbitrary type located at Gauss point 2
    returns
        dY  is a dictionary holding the extrapolated values for this field,
            pushed out to the element nodes.  dY[1] holds the extrapolated
            value at node 1, and dY[2] holds the extrapolated value at node 2.
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    gPts = gq.gaussPoints()
    returns
        gPts is the number of Gauss points (and nodes, in our implementation)

    coord = gq.coordinates(atGaussPt)
    inputs
        atGaussPt   is the Gauss point at which the co-ordinates are sought
    returns
        chord       are the natural co-ordinates at the specified Gauss point

    wgt = gq.weight(atGaussPt)
    inputs
        atGaussPt   is the Gauss point at which the weight is sought
    returns
        wgt         is the weight of quadrature at the specified Gauss point
"""


class GaussQuadrature(GaussQuad):

    def __init__(self):
        super(GaussQuadrature, self).__init__()
        self._gaussPts = 2
        sqrt3 = sqrt(3.0)
        self._coordinates = {
            1: (-1.0 / sqrt3,),
            2: (1.0 / sqrt3,)
            }
        self._weights = {
            1: 1.0,
            2: 1.0
            }
        self._extrapCoef = np.zeros((2, 2), dtype=float)
        self._extrapCoef[0, 0] = (sqrt3 + 3.0) / (2.0 * sqrt3)
        self._extrapCoef[0, 1] = (sqrt3 - 3.0) / (2.0 * sqrt3)
        self._extrapCoef[1, 0] = (sqrt3 - 3.0) / (2.0 * sqrt3)
        self._extrapCoef[1, 1] = (sqrt3 + 3.0) / (2.0 * sqrt3)
        return  # a new instance of a Gauss quadrature rule for chords

    def extrapolate(self, y1, y2):
        if type(y1) == type(y2):
            yAtNode1 = (self._extrapCoef[0, 0] * y1
                        + self._extrapCoef[0, 1] * y2)
            yAtNode2 = (self._extrapCoef[1, 0] * y1
                        + self._extrapCoef[1, 1] * y2)
        else:
            raise RuntimeError("Arguments for extrapolation are not of the "
                               + "same type.")
        extrapolation = {
            1: yAtNode1,
            2: yAtNode2
            }
        return extrapolation

    def gaussPoints(self):
        return self._gaussPts

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
