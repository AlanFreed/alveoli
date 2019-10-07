#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
Module shapeFnChords.py provides shape functions for interpolating a chord.

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
__version__ = "1.3.0"
__date__ = "09-23-2019"
__update__ = "09-30-2019"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

r"""
Change in version "1.3.0":

Created


Overview of module shapeFnChords.py:


Module shapeFnChords.py provides the various shape functions for a point
xi residing along a chord, where 'xi' associates with an x coordinate.
Its vertices in the chord's natural coordinate system are located at
    vertex1: xi = -1
    vertex2: xi = +1
The length of such a chord is 2.

Also provided are the spatial derivatives of these shape functions, taken
with respect to coordinate 'xi'.  From this one can construct approxiamtions
for the displacement G and deformation F gradients.


class

    shapeFunction

constructor

    sf = shapeFunction(xi)
        xi   is the coordinate, positioned in the natural coordinate system,
             whereat the shape function is evaluated

methods

    yXi = sf.interpolate(y1, y2)
        y1   is the value of a field y located at vertex 1
        y2   is the value of a field y located at vertex 2
    returns
        yXi  is its interpolated value for y at location xi

    det = sf.detJacobian(x1, x2)
        x1   is a physical coordinate, a float, located at vertex 1
        x2   is a physical coordinate, a float, located at vertex 2
    returns
        det  is the determinant of the Jacobian matrix
    inputs are coordinates evaluated in a global chordal coordinate system

    Gmtx = sf.G(x1, x2, x01, x02)
        x1   is a physical  coordinate, a float, located at vertex 1
        x2   is a physical  coordinate, a float, located at vertex 2
        x01  is a reference coordinate, a float, located at vertex 1
        x02  is a reference coordinate, a float, located at vertex 2
    returns
        Gmtx is the displacement gradient at location xi
             Gmtx = du/dx    where    u = x - X    or    u = x - x0
    inputs are coordinates evaluated in a global chordal coordinate system

    Fmtx = sf.F(x1, x2, x01, x02)
        x1   is a physical  coordinate, a float, located at vertex 1
        x2   is a physical  coordinate, a float, located at vertex 2
        x01  is a reference coordinate, a float, located at vertex 1
        x02  is a reference coordinate, a float, located at vertex 2
    returns
        Fmtx is the deformation gradient at location (xi)
            Fmtx = dx/dX    where    X = x0
    inputs are coordinates evaluated in a global chordal coordinate system

variables

    # the shape functions

    sf.N1        the 1st shape function, it associates with vertex 1
    sf.N2        the 2nd shape function, it associates with vertex 2

    sf.Nmatx     the 1x2 matrix of shape functions for the chord

    # partial derivatives: d N_i / dXi, i = 1, 2

    sf.dN1dXi    gradient of the 1st shape function wrt the xi coordinate
    sf.dN2dXi    gradient of the 2nd shape function wrt the xi coordinate

Reference
    1) Guido Dhondt, "The Finite Element Method for Three-dimensional
       Thermomechanical Applications", John Wiley & Sons Ltd, 2004.
"""


class shapeFunction(object):

    def __init__(self, xi):
        # create the two exported shape functions
        self.N1 = (1.0 - xi) / 2.0
        self.N2 = (1.0 + xi) / 2.0

        # create the exported 1x2 shape function matrix for a chord
        self.Nmatx = np.array((1, 2), dtype=float)
        self.Nmatx[0, 0] = self.N1
        self.Nmatx[0, 1] = self.N2

        # create the two, exported, derivatives of the shape functions
        self.dN1dXi = -0.5
        self.dN2dXi = 0.5

        return  # the object

    # interpolate a field y known at nodes 1 and 2 to the Gauss point xi
    def interpolate(self, y1, y2):
        y = self.N1 * y1 + self.N2 * y2
        return y

    # calculate the determinant of the Jacobian at Gauss point xi
    def detJacobian(self, x1, x2):
        jacobian = self.dN1dXi * x1 + self.dN2dXi * x2
        return jacobian

    # calculate the displacement gradient at Gauss point xi
    def G(self, x1, x2, x01, x02):
        u1 = x1 - x01
        u2 = x2 - x02
        # determine the displacement gradient
        disGrad = self.dN1dXi * u1 + self.dN2dXi * u2
        # determine the current position gradient
        curGrad = self.dN1dXi * x1 + self.dN2dXi * x2

        Gmtx = disGrad / curGrad
        return Gmtx

    # calculate the deformation gradient at Gauss point xi
    def F(self, x1, x2, x01, x02):
        u1 = x1 - x01
        u2 = x2 - x02
        # determine the displacement gradient
        disGrad = self.dN1dXi * u1 + self.dN2dXi * u2
        # determine the reference position gradient
        refGrad = self.dN1dXi * x01 + self.dN2dXi * x02

        Fmtx = 1.0 + disGrad / refGrad
        return Fmtx
