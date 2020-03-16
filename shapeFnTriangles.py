#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
Module shapeFnTriangles.py provides shape functions for interpolating a
triangle.

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
__date__ = "03-10-2020"
__update__ = "03-16-2020"
__author__ = "Shahla Zamani"
__author_email__ = "Zamani.Shahla@tamu.edu"

r"""
Change in version "1.3.0":

Created

Overview of module shapeFnTetrahedra.py:

Module triaShapeFunction.py provides the three shape functions for a point
(xi, eta) residing within a triangle, where 'xi', 'eta' are the x, y 
coordinates, respectively.  

class

    triaShapeFunction

constructor

    sf = triaShapeFunction(xi, eta)
        xi    is the x coordinate in the natural coordinate system
        eta   is the y coordiante in the natural coordiante system

methods

    y = sf.interpolate(y1, y2, y3)
        y1   is the value of field y located at vertex 1
        y2   is the value of field y located at vertex 2
        y3   is the value of field y located at vertex 3
    returns
        y    is its interpolated value for field y at location (xi, eta)

    det = sf.jacobian(x1, x2, x3)
        x1    is a tuple of physical coordinates (x, y) located at vertex 1
        x2    is a tuple of physical coordinates (x, y) located at vertex 2
        x3    is a tuple of physical coordinates (x, y) located at vertex 3
    returns
        jacob   is the Jacobian matrix
    inputs are tuples of coordinates evaluated in a global coordinate system

variables

    # the shape functions

    sf.N1        the 1st shape function
    sf.N2        the 2nd shape function
    sf.N3        the 3rd shape function

    sf.Nmatx     a 3x12 matrix of shape functions for the tetrahedron with 
                 triangular interpolation function located at (xi, eta)

    # partial derivatives of the shape functions

    # partial derivative: d N_i / dXi, i = 1..3
    sf.dN1dXi    gradient of the 1st shape function wrt the xi coordinate
    sf.dN2dXi    gradient of the 2nd shape function wrt the xi coordinate
    sf.dN3dXi    gradient of the 3rd shape function wrt the xi coordinate

    # partial derivative: d N_i / dEta, i = 1..3
    sf.dN1dEta   gradient of the 1st shape function wrt the eta coordinate
    sf.dN2dEta   gradient of the 2nd shape function wrt the eta coordinate
    sf.dN3dEta   gradient of the 3rd shape function wrt the eta coordinate

"""

class triaShapeFunction(object):

    def __init__(self, xi, eta):
        # create the four exported shape functions
        self.N1 = xi
        self.N2 = eta
        self.N3 = 1 - eta - xi

        # the 3x12 matrix of shape functions for a tetrahedron with triangular 
        # interpolation function
        self.Nmatx = np.array([[self.N1, 0.0, 0.0, self.N2, 0.0, 0.0, self.N3,
                                0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, self.N1, 0.0, 0.0, self.N2, 0.0, 0.0,
                                self.N3, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, self.N1, 0.0, 0.0, self.N2, 0.0,
                                0.0, self.N3, 0.0, 0.0, 0.0]])

        # create the ten, eported, derivatives of these shape functions
        self.dN1dXi = 1
        self.dN2dXi = 0
        self.dN3dXi = -1

        self.dN1dEta = 0
        self.dN2dEta = 1
        self.dN3dEta = -1

        return  # the object

    def interpolate(self, y1, y2, y3):
        y = self.N1 * y1 + self.N2 * y2 + self.N3 * y3
        return y

    def jacobian(self, x1, x2, x3):
        jacob = np.zeros((2, 2), dtype=float)
        if isinstance(x1, tuple):
            jacob[0, 0] = (self.dN1dXi * x1[0] + self.dN2dXi * x2[0] +
                           self.dN3dXi * x3[0])
            jacob[0, 1] = (self.dN1dXi * x1[1] + self.dN2dXi * x2[1] +
                           self.dN3dXi * x3[1])
            jacob[1, 0] = (self.dN1dEta * x1[0] + self.dN2dEta * x2[0] +
                           self.dN3dEta * x3[0])
            jacob[1, 1] = (self.dN1dEta * x1[1] + self.dN2dEta * x2[1] +
                           self.dN3dEta * x3[1])
            # determine the determinant of the Jacobian 
        else:
            raise RuntimeError("Each argument of shapeFunction.jacobian " +
                               "must be a tuple of coordinates, " +
                               "e.g., (x, y).")
        return jacob

