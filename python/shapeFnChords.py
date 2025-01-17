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
__update__ = "10-13-2020"
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

variables

    # the shape functions are

    sf.N1        the 1st shape function
    sf.N2        the 2nd shape function

    sf.Nmtx      the 1x2 matrix of shape functions for representing a chord

    # partial derivatives: d N_i / dXi, i = 1, 2

    sf.dN1dXi    gradient of the 1st shape function wrt the xi co-ordinate
    sf.dN2dXi    gradient of the 2nd shape function wrt the xi co-ordinate

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

    Hmtx = sf.H(x1, x2)
        x1   is a tuple of physical  co-ordinates (x,) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x,) locating vertex 2
    returns
         Hmtx is the derivative of shape functions 
         theta = H * D in the contribution to nonlinear strain
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    Amtx = sf.A(x1, x2, x01, x02)
        x1   is a tuple of physical  co-ordinates (x,) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x,) locating vertex 2
        x01  is a tuple of reference co-ordinates (x0,) locating vertex 1
        x02  is a tuple of reference co-ordinates (x0,) locating vertex 2
    returns
        Amtx is the displacement gradient matrix at location xi 
        EN = 1/2 * A * Theta
    Inputs are co-ordinates evaluated in a global chordal co-ordinate system.

    Lmtx = sf.L(x1, x2)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
    returns
    Lmtx is the derivative of shape functions ( dA = L * D ) in the 
    contribution to nonlinear strain
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    BLmtx = sf.BL(x1, x2)
        x1   is a tuple of physical co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical co-ordinates (x, y) locating vertex 2
    returns
        BLmtx   is the linear strain displacement matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.

    BNmtx = sf.BN(x1, x2, x01, x02)
        x1   is a tuple of physical  co-ordinates (x, y) locating vertex 1
        x2   is a tuple of physical  co-ordinates (x, y) locating vertex 2
        x01  is a tuple of reference co-ordinates (x, y) locating vertex 1
        x02  is a tuple of reference co-ordinates (x, y) locating vertex 2
    returns
        BNmtx is nonlinear contribution to the strain displacement matrix
    Inputs are tuples of co-ordinates evaluated in a global co-ordinate system.


Reference
    1) Guido Dhondt, "The Finite Element Method for Three-dimensional
       Thermomechanical Applications", John Wiley & Sons Ltd, 2004.
"""


class ShapeFunction(ShapeFn):

    # constructor

    def __init__(self, coordinates):
        super(ShapeFunction, self).__init__(coordinates)
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

    # methods
    # determine the interpolated value located at xi
    def interpolate(self, y1, y2):
        if type(y1) == type(y2):
            y = self.N1 * y1[0] + self.N2 * y2[0]
        else:
            raise RuntimeError("interpolate arguments must be the same type.")
        return y
    
    # determine the Jacobian Matrix
    def jacobianMatrix(self, x1, x2):
        Jmtx = np.zeros((1, 1), dtype=float)
        Jmtx[0, 0] = self.dN1dXi * x1[0] + self.dN2dXi * x2[0]
        return Jmtx
    
    # determine the determinant of the Jacobian matrix
    def jacobianDeterminant(self, x1, x2):
        Jmtx = self.jacobianMatrix(x1, x2)
        return Jmtx[0, 0]
    
    # determine the displacement gradient matrix at location xi
    def G(self, x1, x2, x01, x02):
        # determine the displacement gradient
        disGrad = (self.dN1dXi * (x1[0] - x01[0]) + self.dN2dXi * (x2[0] - x02[0]))
        # determine the current position gradient
        curGrad = self.dN1dXi * x1[0] + self.dN2dXi * x2[0]

        Gmtx = np.zeros((1, 1), dtype=float)
        Gmtx[0, 0] = disGrad / curGrad
        return Gmtx
    
    # determine  the deformation gradient matrix at location (xi)
    def F(self, x1, x2, x01, x02):
        # determine the displacement gradient
        disGrad = (self.dN1dXi * (x1[0] - x01[0]) 
                   + self.dN2dXi * (x2[0] - x02[0]))
        # determine the current gradient of position
        refGrad = self.dN1dXi * x01[0] + self.dN2dXi * x02[0]
        Fmtx = np.zeros((1, 1), dtype=float)
        Fmtx[0, 0] = 1.0 + disGrad / refGrad
        return Fmtx
    
    # determine the derivative of shape functions ( theta = H * D )
    def H(self, x1, x2):
        Jmtx = self.jacobianMatrix(x1, x2)
        H = np.zeros((1, 2), dtype=float)

        H[0, 0] = self.dN1dXi * Jmtx[0, 0]
        H[0, 1] = self.dN2dXi * Jmtx[0, 0]

        detJ = np.linalg.det(Jmtx)
        Hmtx = H / detJ

        return Hmtx
    
    # determine the displacement gradient matrix at location xi 
    # EN = 1/2 * A * Theta
    def A(self, x1, x2, x01, x02):
        # determine the displacement gradient
        disGrad = (self.dN1dXi * (x1[0] - x01[0]) 
                   + self.dN2dXi * (x2[0] - x02[0]))
        # determine the current position gradient
        curGrad = self.dN1dXi * x1[0] + self.dN2dXi * x2[0]
        A = np.zeros((1, 1), dtype=float)
        A[0, 0] = - disGrad / curGrad
        return A
    
    # determine the derivative of shape functions ( dA = L * D )
    def L(self, x1, x2):
        Jmtx = self.jacobianMatrix(x1, x2)
        L = np.zeros((1, 2), dtype=float)

        L[0, 0] = - self.dN1dXi * Jmtx[0, 0]
        L[0, 1] = - self.dN2dXi * Jmtx[0, 0]

        detJ = np.linalg.det(Jmtx)
        Lmtx = L / detJ

        return Lmtx
    
    # determne the linear strain displacement matrix
    def BL(self, x1, x2):
        Jmtx = self.jacobianMatrix(x1, x2)
        BL = np.zeros((1, 2), dtype=float)

        BL[0, 0] = self.dN1dXi * Jmtx[0, 0]
        BL[0, 1] = self.dN2dXi * Jmtx[0, 0]
        
        detJ = np.linalg.det(Jmtx)
        BLmtx = BL / detJ

        return BLmtx
    
    # determine the nonlinear strain displacement matrix ( EN = A * H )
    def BN(self, x1, x2, x01, x02):
        A = self.A(x1, x2, x01, x02)
        H = self.H(x1, x2)
        BNmtx = np.zeros((1, 2), dtype=float)

        BNmtx = np.dot(A, H)
        return BNmtx


"""
Changes made in version "1.0.0":

Original version
"""
