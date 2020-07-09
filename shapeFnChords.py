#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from shapeFunctions import ShapeFunction as ShapeFn

"""
Module shapeFnChords.py implements class ShapeFunction for chords.

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
__date__ = "09-23-2019"
__update__ = "07-07-2020"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"


"""
A listing of changes made wrt version release can be found at the end of file.


Overview of module shapeFnChords.py:


Module shapeFnChords.py provides the class ShapeFunction whose objects provide
the various interpolation functions at some point xi residing along a chord.
The vertices of a chord in its natural co-ordinate system are located at
    vertex1: xi = -1
    vertex2: xi = +1
so that the length of a chord in this natural co-ordiante system is 2.

Also provided are the spatial derivatives for these shape functions, taken
with respect to co-ordinate 'xi', from which one can construct approximations
for the the Jacobian J, plus the displacement G and deformation F gradients.


class

    ShapeFunction
        Implements class ShapeFunction exported by module shapeFunctions.py.

constructor

    sf = ShapeFunction(coordinates)
        coordinates     A tuple of natural co-ordinates for interpolating at.
                        For a chord, coordinates = (xi,) where xi is the 'x'
                        co-ordinate location of interest residing in [-1, 1]
    returns
        sf              a new instance of class ShapeFunction for 1D chords

inherited methods

    yXi = sf.interpolate(y1, y2)
        y1   is a physical field of arbitrary type located at vertex 1
        y2   is a physical field of arbitrary type located at vertex 2
    returns
        yXi  is the interpolated value for this field located at xi
    Inputs must allow for: i) scalar multiplication and ii) the '+' operator.

    Jmtx = sf.jacobianMatrix(x1, x2)
        x1   is a tuple of physical co-ordinates (x,) locating vertex 1
        x2   is a tuple of physical co-ordinates (x,) locating vertex 2
    returns
        Jmtx is the Jacobian matrix
    Inputs are co-ordinates evaluated in a global chordal co-ordinate system.

    Jdet = sf.jacobianDeterminant(x1, x2)
        x1   is a tuple of physical co-ordinates (x,) locating vertex 1
        x2   is a tuple of physical co-ordinates (x,) locating vertex 2
    returns
        Jdet is the determinant of the Jacobian matrix
    Inputs are co-ordinates evaluated in a global chordal co-ordinate system.

    Gmtx = sf.G(x1, x2, x01, x02)
        x1   is a tuple of physical  co-ordinates (x,) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x,) locating vertex 2
        x01  is a tuple of reference co-ordinates (x0,) locating vertex 1
        x02  is a tuple of reference co-ordinates (x0,) locating vertex 2
    returns
        Gmtx is the displacement gradient matrix at location xi
             Gmtx = du/dx    where    u = x - X    or    u = x - x0
    Inputs are co-ordinates evaluated in a global chordal co-ordinate system.

    Fmtx = sf.F(x1, x2, x01, x02)
        x1   is a tuple of physical  co-ordinates (x,) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x,) locating vertex 2
        x01  is a tuple of reference co-ordinates (x0,) locating vertex 1
        x02  is a tuple of reference co-ordinates (x0,) locating vertex 2
    returns
        Fmtx is the deformation gradient matrix at location (xi)
            Fmtx = dx/dX    where    X = x0
    Inputs are co-ordinates evaluated in a global chordal co-ordinate system.

variables

    # the shape functions are

    sf.N1        the 1st shape function
    sf.N2        the 2nd shape function

    sf.Nmtx      the 1x2 matrix of shape functions for representing a chord

    # partial derivatives: d N_i / dXi, i = 1, 2

    sf.dN1dXi    gradient of the 1st shape function wrt the xi co-ordinate
    sf.dN2dXi    gradient of the 2nd shape function wrt the xi co-ordinate

Reference
    1) Guido Dhondt, "The Finite Element Method for Three-dimensional
       Thermomechanical Applications", John Wiley & Sons Ltd, 2004.
"""


class ShapeFunction(ShapeFn):

    def __init__(self, coordinates):
        super(ShapeFunction, self).__init(coordinates)
        if len(coordinates) == 1:
            xi = coordinates[0]
        else:
            raise RuntimeError("The co-ordinates for a chord are in 1D.")
        if (xi < -1.0) or (xi > 1.0):
            raise RuntimeError("Co-ordinate 'xi' must be in [-1, 1], and "
                               + "you sent {:06.4f}.".format(xi))

        # create the two exported shape functions
        self.N1 = (1.0 - xi) / 2.0
        self.N2 = (1.0 + xi) / 2.0

        # construct the exported 1x2 shape function matrix for a chord
        self.Nmtx = np.zeros((1, 2), dtype=float)
        self.Nmtx[0, 0] = self.N1
        self.Nmtx[0, 1] = self.N2

        # create the two, exported, derivatives of the shape functions
        self.dN1dXi = -0.5
        self.dN2dXi = 0.5
        return  # the object

    def interpolate(self, y1, y2):
        if type(y1) == type(y2):
            y = self.N1 * y1 + self.N2 * y2
        else:
            raise RuntimeError("interpolate arguments must be the same type.")
        return y

    def jacobianMatrix(self, x1, x2):
        if (isinstance(x1, tuple) and len(x1) == 1
           and isinstance(x2, tuple) and len(x2) == 1):
            Jmtx = np.zeros((1, 1), dtype=float)
            Jmtx[0, 0] = self.dN1dXi * x1[0] + self.dN2dXi * x2[0]
        else:
            raise RuntimeError("Each argument of shapeFunction.jacobianMatrix "
                               + "must be a tuple of co-ordinates, eg., (x,).")
        return Jmtx

    def jacobianDeterminant(self, x1, x2):
        Jmtx = self.jacobianMartrix(x1, x2)
        return Jmtx[0, 0]

    def G(self, x1, x2, x01, x02):
        if (isinstance(x1, tuple) and len(x1) == 1
           and isinstance(x2, tuple) and len(x2) == 1
           and isinstance(x01, tuple) and len(x01) == 1
           and isinstance(x02, tuple) and len(x02) == 1):
            # determine the displacement gradient
            disGrad = (self.dN1dXi * (x1[0] - x01[0])
                       + self.dN2dXi * (x2[0] - x02[0]))
            # determine the current position gradient
            curGrad = self.dN1dXi * x1[0] + self.dN2dXi * x2[0]
        else:
            raise RuntimeError("Each argument of shapeFunction.G must be "
                               + "a tuple of co-ordinates, e.g., (x,).")
        Gmtx = np.zeros((1, 1), dtype=float)
        Gmtx[0, 0] = disGrad / curGrad
        return Gmtx

    def F(self, x1, x2, x01, x02):
        if (isinstance(x1, tuple) and len(x1) == 1
           and isinstance(x2, tuple) and len(x2) == 1
           and isinstance(x01, tuple) and len(x01) == 1
           and isinstance(x02, tuple) and len(x02) == 1):
            # determine the displacement gradient
            disGrad = (self.dN1dXi * (x1[0] - x01[0])
                       + self.dN2dXi * (x2[0] - x02[0]))
            # determine the current gradient of position
            refGrad = self.dN1dXi * x01[0] + self.dN2dXi * x02[0]
        else:
            raise RuntimeError("Each argument of shapeFunction.F must be "
                               + "a tuple of co-ordinates, e.g., (x,).")
        Fmtx = np.zeros((1, 1), dtype=float)
        Fmtx[0, 0] = 1.0 + disGrad / refGrad
        return Fmtx


"""
Changes made in version "1.0.0":

Original version
"""
