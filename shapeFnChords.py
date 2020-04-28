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
__version__ = "1.4.0"
__date__ = "09-23-2019"
__update__ = "04-16-2019"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"


"""
A listing of changes made wrt version release can be found at the end of file.


Overview of module shapeFnChords.py:


Module shapeFnChords.py provides the class shapeFunction whose objects provide
the various shape-function functions at some point xi residing along a chord.
The vertices of a chord in its natural co-ordinate system are located at
    vertex1: xi = -1
    vertex2: xi = +1
so that the length of a chord in this natural co-ordiante system is 2.

Also provided are the spatial derivatives for these shape functions, taken
with respect to co-ordinate 'xi', from which one can construct approximations
for the the Jacobian J, and the displacement G and deformation F gradients.


class

    shapeFunction

constructor

    sf = shapeFunction(xi)
        xi   is the co-ordinate, positioned in the natural co-ordinate system,
             whereat the shape function is evaluated

methods

    yXi = sf.interpolate(y1, y2)
        y1   is a physical field of arbitrary type located at vertex 1
        y2   is a physical field of arbitrary type located at vertex 2
    returns
        yXi  is the interpolated value for this field located at xi
    inputs must allow for: i) scalar multiplication and ii) the '+' operator

    Jmtx = sf.jacobianMtx(x1, x2)
        x1   is a tuple of physical co-ordinates (x) located at vertex 1
        x2   is a tuple of physical co-ordinates (x) located at vertex 2
    returns
        Jmtx is the Jacobian matrix
    inputs are co-ordinates evaluated in a global chordal co-ordinate system

    Jdet = sf.jacobianDet(x1, x2)
        x1   is a tuple of physical co-ordinates (x) located at vertex 1
        x2   is a tuple of physical co-ordinates (x) located at vertex 2
    returns
        Jdet is the determinant of the Jacobian matrix
    inputs are co-ordinates evaluated in a global chordal coordinate system

    Gmtx = sf.G(x1, x2, x01, x02)
        x1   is a tuple of physical  co-ordinates (x) located at vertex 1
        x2   is a tuple of physical  co-ordinates (x) located at vertex 2
        x01  is a tuple of reference co-ordinates (x0) located at vertex 1
        x02  is a tuple of reference co-ordinates (x0) located at vertex 2
    returns
        Gmtx is the displacement gradient matrix at location xi
             Gmtx = du/dx    where    u = x - X    or    u = x - x0
    inputs are co-ordinates evaluated in a global chordal co-ordinate system

    Fmtx = sf.F(x1, x2, x01, x02)
        x1   is a tuple of physical  co-ordinates (x) located at vertex 1
        x2   is a tuple of physical  co-ordinates (x) located at vertex 2
        x01  is a tuple of reference co-ordinates (x0) located at vertex 1
        x02  is a tuple of reference co-ordinates (x0) located at vertex 2
    returns
        Fmtx is the deformation gradient matrix at location (xi)
            Fmtx = dx/dX    where    X = x0
    inputs are co-ordinates evaluated in a global chordal co-ordinate system

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


class shapeFunction(object):

    def __init__(self, xi):
        if (xi < -1.0) or (xi > 1.0):
            raise RuntimeError("Co-ordinate 'xi' must be in [-1, 1], and " +
                               "you sent {:06.4f}.".format(xi))

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
        y = self.N1 * y1 + self.N2 * y2
        return y

    def jacobianMtx(self, x1, x2):
        Jmtx = np.zeros((1, 1), dtype=float)
        if isinstance(x1, tuple):
            Jmtx[0, 0] = self.dN1dXi * x1[0] + self.dN2dXi * x2[0]
        else:
            raise RuntimeError("Each argument of shapeFunction.jacobianMtx " +
                               "must be a tuple of co-ordinates, e.g., (x).")
        return Jmtx

    def jacobianDet(self, x1, x2):
        if isinstance(x1, tuple):
            Jdet = self.dN1dXi * x1[0] + self.dN2dXi * x2[0]
        else:
            raise RuntimeError("Each argument of shapeFunction.jacobianDet " +
                               "must be a tuple of co-ordinates, e.g., (x).")
        return Jdet

    def G(self, x1, x2, x01, x02):
        Gmtx = np.zeros((1, 1), dtype=float)
        if isinstance(x1, tuple):
            # determine the displacement gradient
            disGrad = (self.dN1dXi * (x1[0] - x01[0]) +
                       self.dN2dXi * (x2[0] - x02[0]))
            # determine the current position gradient
            curGrad = self.dN1dXi * x1[0] + self.dN2dXi * x2[0]
        else:
            raise RuntimeError("Each argument of shapeFunction.G " +
                               "must be a tuple of coordinates, e.g., (x).")
        Gmtx[0, 0] = disGrad / curGrad
        return Gmtx

    def F(self, x1, x2, x01, x02):
        Fmtx = np.zeros((1, 1), dtype=float)
        if isinstance(x1, tuple):
            # determine the displacement gradient
            disGrad = (self.dN1dXi * (x1[0] - x01[0]) +
                       self.dN2dXi * (x2[0] - x02[0]))
            # determine the current gradient of position
            refGrad = self.dN1dXi * x01[0] + self.dN2dXi * x02[0]
        else:
            raise RuntimeError("Each argument of shapeFunction.F " +
                               "must be a tuple of co-ordinates, e.g., (x).")
        Fmtx[0, 0] = 1.0 + disGrad / refGrad
        return Fmtx


"""
Changes made in version "1.4.0":


Tuples are now passed as arguments instead of floats to make its interface
consistent with those for the other geometries considered.

Entities that return matrices for multi-dimensional fields now return matrices
here, too, allbeit they are 1x1 matrices.  This was done for consistency.

Other changes made:

variable sf.Nmat renamed to

    sf.Nmtx

method

    jacob = sf.jacobian(x1, x2)

was replaced with two methods

    Jmtx = sf.jacobianMtx(x1, x2)

and

    Jdet = sf.jacobianDet(x1, x2)

Removed method

    dNmat = sf.dNdximat()


Changes made in version "1.3.0":

Original version
"""
