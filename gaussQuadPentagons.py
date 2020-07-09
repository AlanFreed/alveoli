#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
__update__ = "07-08-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

r"""
A listing of changes made wrt version release can be found at the end of file.


Overview of module gaussQuadPentagons.py:

Module gaussQuadPentagons.py exports class GaussQuadrature which allows for
the extrapolation of fields from their Gauss points out to their nodal points,
along with providing for the weights and nodes (co-ordinates) of a Gaussian
quadrature rule suitable for integrating over the area of a pentagon.

constructor

    gq = GaussQuadrature()
    returns
        gq  is a new instance of the class GaussQuadrature for a pentagon

methods

    dY = gq.extrapolate(y1, y2, y3, y4, y5)
    inputs
        y1  is a physical field of arbitrary type located at Gauss point 1
        y2  is a physical field of arbitrary type located at Gauss point 2
        y3  is a physical field of arbitrary type located at Gauss point 3
        y4  is a physical field of arbitrary type located at Gauss point 4
        y5  is a physical field of arbitrary type located at Gauss point 5
    returns
        dY  is a dictionary holding the extrapolated values for this field,
            pushed out to the element nodes.  dY[1] holds the extrapolated
            value at node 1, dY[2] holds the extrapolated value at node 2,
            dY[3] holds the extrapolated value at node 3, dY[4] holds the
            extrapolated value at node 4, and dY[5] holds the extrapolated
            value at node 5.
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    gPts = gq.gaussPoints()
    returns
        gPts    the number of Gauss points (and nodes, in our implementation)

    coord = gq.coordinates(atGaussPt)
    inputs
        atGaussPt   is the Gauss point at which the co-ordinates are sought
    returns
        chord       are the natural co-ordinates at the specified Gauss point

    wgt = gq.weight(atGaussPt)
    inputs
        atGaussPt   is the Gauss point at which the co-ordinates are sought
    returns
        wgt         is the weight of quadrature at the specified Gauss point
"""


class GaussQuadrature(GaussQuad):

    def __init__(self):
        super(GaussQuadrature, self).__init__()
        self._gaussPts = 5
        self._coordinates = {
            1: (-0.0904672856259437, 0.6261163672367185),
            2: (-0.5988664907434159, -0.0372772228317812),
            3: (0.0151090564021255, -0.6475950318414561),
            4: (0.5913912812273181, -0.0149548962936793),
            5: (0.3651143172435236, 0.5872395873378157)
            }
        self._weights = {
            1: 0.4934869125017041,
            2: 0.6190315158026863,
            3: 0.5321153805577828,
            4: 0.6173424062184376,
            5: 0.1156650756572724
            }
        self._extrapCoef = np.zeros((5, 5), dtype=float)
        """
        self._extrapCoef[0, 0] =
        self._extrapCoef[0, 1] =
        self._extrapCoef[0, 2] =
        self._extrapCoef[0, 3] =
        self._extrapCoef[0, 4] =
        self._extrapCoef[1, 0] =
        self._extrapCoef[1, 1] =
        self._extrapCoef[1, 2] =
        self._extrapCoef[1, 3] =
        self._extrapCoef[1, 4] =
        self._extrapCoef[2, 0] =
        self._extrapCoef[2, 1] =
        self._extrapCoef[2, 2] =
        self._extrapCoef[2, 3] =
        self._extrapCoef[2, 4] =
        self._extrapCoef[3, 0] =
        self._extrapCoef[3, 1] =
        self._extrapCoef[3, 2] =
        self._extrapCoef[3, 3] =
        self._extrapCoef[3, 4] =
        self._extrapCoef[4, 0] =
        self._extrapCoef[4, 1] =
        self._extrapCoef[4, 2] =
        self._extrapCoef[4, 3] =
        self._extrapCoef[4, 4] =
        """
        return  # a new instance of a Gauss quadrature rule for pentagons

    def extrapolate(self, y1, y2, y3, y4, y5):
        if (type(y1) == type(y2) and type(y2) == type(y3)
           and type(y3) == type(y4) and type(y4) == type(y5)):
            yAtNode1 = (self._extrapCoef[0, 0] * y1
                        + self._extrapCoef[0, 1] * y2
                        + self._extrapCoef[0, 2] * y3
                        + self._extrapCoef[0, 3] * y4
                        + self._extrapCoef[0, 4] * y5)
            yAtNode2 = (self._extrapCoef[1, 0] * y1
                        + self._extrapCoef[1, 1] * y2
                        + self._extrapCoef[1, 2] * y3
                        + self._extrapCoef[1, 3] * y4
                        + self._extrapCoef[1, 4] * y5)
            yAtNode3 = (self._extrapCoef[2, 0] * y1
                        + self._extrapCoef[2, 1] * y2
                        + self._extrapCoef[2, 2] * y3
                        + self._extrapCoef[2, 3] * y4
                        + self._extrapCoef[2, 4] * y5)
            yAtNode4 = (self._extrapCoef[3, 0] * y1
                        + self._extrapCoef[3, 1] * y2
                        + self._extrapCoef[3, 2] * y3
                        + self._extrapCoef[3, 3] * y4
                        + self._extrapCoef[3, 4] * y5)
            yAtNode5 = (self._extrapCoef[4, 0] * y1
                        + self._extrapCoef[4, 1] * y2
                        + self._extrapCoef[4, 2] * y3
                        + self._extrapCoef[4, 3] * y4
                        + self._extrapCoef[4, 4] * y5)
        else:
            raise RuntimeError("Arguments for extrapolation are not of the "
                               + "same type.")
        extrapolation = {
            1: yAtNode1,
            2: yAtNode2,
            3: yAtNode3,
            4: yAtNode4,
            5: yAtNode5
            }
        return extrapolation

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
