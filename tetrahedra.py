#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math as m
import numpy as np
import numpy.linalg as linalg
from shapeFnTetrahedra import shapeFunction
from vertices import vertex

"""
Module tetrahedra.py provides geometric information about irregular tetrahedra.

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
__date__ = "09-25-2019"
__update__ = "09-27-2019"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

r"""

Change in version "1.3.0":

Created

Overview of module tetrahedra.py:

Class tetrahedron in file tetrahedra.py allows for the creation of objects that
are to be used to represent irregular tetrahedra.  Its vertices are located at
    vertex1: (xi, eta, zeta) = (0, 0, 0)
    vertex2: (xi, eta, zeta) = (1, 0, 0)
    vertex3: (xi, eta, zeta) = (0, 1, 0)
    vertex4: (xi, eta, zeta) = (0, 0, 1)
wherein xi, eta, and zeta denote the element's natural coordinates.  The volume
of such a tetrahedron is 1/6.

Numerous methods have a string argument that is denoted as  state  which can
take on any of the following values:
    'c', 'curr', 'current'       gets the value for a current configuration
    'n', 'next'                  gets the value for a next configuration
    'p', 'prev', 'previous'      gets the value for a previous configuration
    'r', 'ref', 'reference'      gets the value for the reference configuration

class tetrahedron

constructor

    t = tetrahedron(number, vertex1, vertex2, vertex3, vertex4, h, gaussPts)
        number    immutable value that is unique to this tetrahedron
        vertex1   unique node of the tetrahedron, an instance of class vertex
        vertex2   unique node of the tetrahedron, an instance of class vertex
        vertex3   unique node of the tetrahedron, an instance of class vertex
        vertex4   unique node of the tetrahedron, an instance of class vertex
        h         timestep size between two successive calls to 'advance'
        gaussPts  number of Gauss points to be used: must be 1, 4 or 5

methods

    s = t.toString()
        returns string representation for tetrahedron in configuration 'state'

    n = t.number()
        returns the unique number affiated with this tetrahedron

    n1, n2, n3, n4 = t.vertexNumbers()
        returns unique numbers associated with the vertices of a tetrahedron

    truth = t.hasVertex(number)
        returns True if one of the four vertices has this vertex number

    v = t.getVertex(number)
        returns a vertex; typically called from within a t.hasVertex if clause

    n = t.gaussPoints()
        returns the number of Gauss points assigned to the tetrahedron

    t.update()
        assigns new coordinate values to the tetrahedorn for its next location
        and updates all effected fields.  To be called after all vertices have
        had their coordinates updated.  This may be called multiple times
        before freezing it with advance

    t.advance()
        assigns the current location into the previous location, and then it
        assigns the next location into the current location, thereby freezing
        the location of the present next-location in preparation to advance to
        the next step along a solution path

    Geometric fields associated with a tetrahedral volume in 3 space

    a = t.volume(state)
        returns the volume of the tetrahedron in configuration 'state'

    vLambda = t.volumetricStretch(state)
        returns the cube root of: volume(state) divided by its reference volume

    vStrain = t.volumetricStrain(state)
        returns the logarithm of volumetric stretch evaluated at 'state'

    dvStrain = t.dVolumetricStrain(state)
        returns the time rate of change in volumetric strain at 'state'

    Kinematic fields associated with the centroid of a tetrahedron in 3 space

    [cx, cy, cz] = t.centroid(state)
        returns centroid of this tetrahedron in configuration 'state'

    [ux, uy, uz] = t.displacement(state)
        returns the displacement at the centroid in configuration 'state'

    [vx, vy, vz] = t.velocity(state)
        returns the velocity at the centroid in configuration 'state'

    [ax, ay, az] = t.acceleration(state)
        returns the acceleration at the centroid in configuration 'state'

    pMtx = t.rotation(state)
        returns a 3x3 orthogonal matrix that rotates the reference base vectors
        of the dodecahedron into a set of local base vectors pertaining to an
        irregular tetrahedron.  The returned matrix associates with
        configuration 'state'

    omegaMtx = p.spin(state)
        returns a 3x3 skew symmetric matrix that describes the time rate of
        rotation, i.e., spin, of the local tetrahedral coordinate system about
        the fixed dodecahedral coordinate system with reference base vectors.
        The returned matrix associates with configuration 'state'

    Fields needed to construct finite element representations

    sf = p.shapeFunction(self, gaussPt):
        returns the shape function associated with the specified Gauss point

    massM = t.massMatrix(rho)
        rho      the mass density with units of mass per unit volume
    returns
        massM    a 12x12 mass matrix for the tetrahedron

    The fundamental fields of kinematics

    gMtx = p.G(gaussPt, state)
        returns 3x3 matrix describing the displacement gradient for the
        tetrahedron at 'gaussPt' in configuration 'state'

    fMtx = p.F(gaussPt, state)
        returns 3x3 matrix describing the deformation gradient for the
        tetrahedron at 'gaussPt' in configuration 'state'

Reference
    1) Guido Dhondt, "The Finite Element Method for Three-dimensional
       Thermomechanical Applications", John Wiley & Sons Ltd, 2004.
    2) Colins, K. D. "Cayley-Menger Determinant." From MathWorld--A Wolfram Web
       Resource, created by Eric W. Weisstein. http://mathworld.wolfram.com/
       Cayley-MengerDeterminant.html
"""


class tetrahedron(object):

    def __init__(self, number, vertex1, vertex2, vertex3, vertex4, h,
                 gaussPts):
        # verify the input
        self._number = int(number)
        # place the vertices into their data structure
        if not isinstance(vertex1, vertex):
            raise RuntimeError("vertex1 must be an instance of type vertex.")
        if not isinstance(vertex2, vertex):
            raise RuntimeError("vertex2 must be an instance of type vertex.")
        if not isinstance(vertex3, vertex):
            raise RuntimeError("vertex3 must be an instance of type vertex.")
        if not isinstance(vertex4, vertex):
            raise RuntimeError("vertex4 must be an instance of type vertex.")
        self._vertex = {
            1: vertex1,
            2: vertex2,
            3: vertex3,
            4: vertex4
        }
        self._setOfVertices = {
            vertex1.number(),
            vertex2.number(),
            vertex3.number(),
            vertex4.number()
        }
        # check the stepsize
        if h > np.finfo(float).eps:
            self._h = float(h)
        else:
            raise RuntimeError("Error: stepsize in the tetrahedron " +
                               "constructor isn't positive.")
        # check the number of Gauss points to use
        if gaussPts == 1 or gaussPts == 4 or gaussPts == 5:
            self._gaussPts = gaussPts
        else:
            raise RuntimeError('Error: {} Gauss points were specified in ' +
                               'tetrahedra constructor; must be 1, 4 or 5.'
                               .format(gaussPts))

        # create a shape function for the centroid of the tetrahedron
        self._centroidSF = shapeFunction(0.25, 0.25, 0.25)

        # establish shape functions located at the Gauss points (xi, eta, zeta)
        if gaussPts == 1:
            # this single Gauss point has a weight of 1/6
            xi = 0.25
            eta = 0.25
            zeta = 0.25
            sf1 = shapeFunction(xi, eta, zeta)

            self._shapeFns = {
                1: sf1
            }
        elif gaussPts == 4:
            # each of these four Gauss points has a weight of 1/24
            a = (5.0 - m.sqrt(5.0)) / 20.0
            b = (5.0 + 3.0 * m.sqrt(5.0)) / 20.0

            xi1 = a
            eta1 = a
            zeta1 = a
            sf1 = shapeFunction(xi1, eta1, zeta1)

            xi2 = b
            eta2 = a
            zeta2 = a
            sf2 = shapeFunction(xi2, eta2, zeta2)

            xi3 = a
            eta3 = b
            zeta3 = a
            sf3 = shapeFunction(xi3, eta3, zeta3)

            xi4 = a
            eta4 = a
            zeta4 = b
            sf4 = shapeFunction(xi4, eta4, zeta4)

            self._shapeFns = {
                1: sf1,
                2: sf2,
                3: sf3,
                4: sf4
            }
        else:  # gaussPts = 5
            # Gauss point 1 is at the centroid and has a weight of  -2/15
            xi1 = 0.25
            eta1 = 0.25
            zeta1 = 0.25
            sf1 = shapeFunction(xi1, eta1, zeta1)

            # the remaining four Gauss points each has a weight of  3/40
            xi2 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf2 = shapeFunction(xi2, eta2, zeta2)

            xi3 = 0.5
            eta3 = 1.0 / 6.0
            zeta4 = 1.0 / 6.0
            sf3 = shapeFunction(xi3, eta3, zeta3)

            xi4 = 1.0 / 6.0
            eta4 = 0.5
            zeta4 = 1.0 / 6.0
            sf4 = shapeFunction(xi4, eta4, zeta4)

            xi5 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta5 = 0.5
            sf5 = shapeFunction(xi5, eta5, zeta5)

            self._shapeFns = {
                1: sf1,
                2: sf2,
                3: sf3,
                4: sf4,
                5: sf5
            }

        # create matrices for tetrahedron at its Gauss points via dictionaries
        # p implies previous, c implies current, n implies next
        if gaussPts == 1:
            # displacement gradients located at the Gauss points of tetrahedron
            self._G0 = {
                1: np.zeros((3, 3), dtype=float)
            }
            self._Gp = {
                1: np.zeros((3, 3), dtype=float)
            }
            self._Gc = {
                1: np.zeros((3, 3), dtype=float)
            }
            self._Gn = {
                1: np.zeros((3, 3), dtype=float)
            }
            # deformation gradients located at the Gauss points of tetrahedron
            self._F0 = {
                1: np.identity(3, dtype=float)
            }
            self._Fp = {
                1: np.identity(3, dtype=float)
            }
            self._Fc = {
                1: np.identity(3, dtype=float)
            }
            self._Fn = {
                1: np.identity(3, dtype=float)
            }
        elif gaussPts == 4:
            # displacement gradients located at the Gauss points of tetrahedron
            self._G0 = {
                1: np.zeros((3, 3), dtype=float),
                2: np.zeros((3, 3), dtype=float),
                3: np.zeros((3, 3), dtype=float),
                4: np.zeros((3, 3), dtype=float)
            }
            self._Gp = {
                1: np.zeros((3, 3), dtype=float),
                2: np.zeros((3, 3), dtype=float),
                3: np.zeros((3, 3), dtype=float),
                4: np.zeros((3, 3), dtype=float)
            }
            self._Gc = {
                1: np.zeros((3, 3), dtype=float),
                2: np.zeros((3, 3), dtype=float),
                3: np.zeros((3, 3), dtype=float),
                4: np.zeros((3, 3), dtype=float)
            }
            self._Gn = {
                1: np.zeros((3, 3), dtype=float),
                2: np.zeros((3, 3), dtype=float),
                3: np.zeros((3, 3), dtype=float),
                4: np.zeros((3, 3), dtype=float)
            }
            # deformation gradients located at the Gauss points of tetrahedron
            self._F0 = {
                1: np.identity(3, dtype=float),
                2: np.identity(3, dtype=float),
                3: np.identity(3, dtype=float),
                4: np.identity(3, dtype=float)
            }
            self._Fp = {
                1: np.identity(3, dtype=float),
                2: np.identity(3, dtype=float),
                3: np.identity(3, dtype=float),
                4: np.identity(3, dtype=float)
            }
            self._Fc = {
                1: np.identity(3, dtype=float),
                2: np.identity(3, dtype=float),
                3: np.identity(3, dtype=float),
                4: np.identity(3, dtype=float)
            }
            self._Fn = {
                1: np.identity(3, dtype=float),
                2: np.identity(3, dtype=float),
                3: np.identity(3, dtype=float),
                4: np.identity(3, dtype=float)
            }
        else:  # gaussPts = 5
            # displacement gradients located at the Gauss points of tetrahedron
            self._G0 = {
                1: np.zeros((3, 3), dtype=float),
                2: np.zeros((3, 3), dtype=float),
                3: np.zeros((3, 3), dtype=float),
                4: np.zeros((3, 3), dtype=float),
                5: np.zeros((3, 3), dtype=float)
            }
            self._Gp = {
                1: np.zeros((3, 3), dtype=float),
                2: np.zeros((3, 3), dtype=float),
                3: np.zeros((3, 3), dtype=float),
                4: np.zeros((3, 3), dtype=float),
                5: np.zeros((3, 3), dtype=float)
            }
            self._Gc = {
                1: np.zeros((3, 3), dtype=float),
                2: np.zeros((3, 3), dtype=float),
                3: np.zeros((3, 3), dtype=float),
                4: np.zeros((3, 3), dtype=float),
                5: np.zeros((3, 3), dtype=float)
            }
            self._Gn = {
                1: np.zeros((3, 3), dtype=float),
                2: np.zeros((3, 3), dtype=float),
                3: np.zeros((3, 3), dtype=float),
                4: np.zeros((3, 3), dtype=float),
                5: np.zeros((3, 3), dtype=float)
            }

            # deformation gradients located at the Gauss points of pentagon
            self._F0 = {
                1: np.identity(3, dtype=float),
                2: np.identity(3, dtype=float),
                3: np.identity(3, dtype=float),
                4: np.identity(3, dtype=float),
                5: np.identity(3, dtype=float)
            }
            self._Fp = {
                1: np.identity(3, dtype=float),
                2: np.identity(3, dtype=float),
                3: np.identity(3, dtype=float),
                4: np.identity(3, dtype=float),
                5: np.identity(3, dtype=float)
            }
            self._Fc = {
                1: np.identity(3, dtype=float),
                2: np.identity(3, dtype=float),
                3: np.identity(3, dtype=float),
                4: np.identity(3, dtype=float),
                5: np.identity(3, dtype=float)
            }
            self._Fn = {
                1: np.identity(3, dtype=float),
                2: np.identity(3, dtype=float),
                3: np.identity(3, dtype=float),
                4: np.identity(3, dtype=float),
                5: np.identity(3, dtype=float)
            }

        # get the reference coordinates for the vetices of the tetrahedron
        self._x10, self._y10, self._z10 = self._vertex[1].coordinates('ref')
        self._x20, self._y20, self._z20 = self._vertex[2].coordinates('ref')
        self._x30, self._y30, self._z30 = self._vertex[3].coordinates('ref')
        self._x40, self._y40, self._z40 = self._vertex[4].coordinates('ref')

        # determine the volume of this tetrahedron in its reference state
        self._V0 = self.volTet(self._x10, self._y10, self._z10,
                               self._x20, self._y20, self._z20,
                               self._x30, self._y30, self._z30,
                               self._x40, self._y40, self._z40)
        self._Vp = self._V0
        self._Vc = self._V0
        self._Vn = self._V0

        # establish the centroidal location of this tetrahedron
        self._centroidX0 = self._centroidSF.interpolate(self._x10, self._x20,
                                                        self._x30, self._x40)
        self._centroidY0 = self._centroidSF.interpolate(self._y10, self._y20,
                                                        self._y30, self._y40)
        self._centroidZ0 = self._centroidSF.interpolate(self._z10, self._z20,
                                                        self._z30, self._z40)
        self._centroidXp = self._centroidX0
        self._centroidXc = self._centroidX0
        self._centroidXn = self._centroidX0
        self._centroidYp = self._centroidY0
        self._centroidYc = self._centroidY0
        self._centroidYn = self._centroidY0
        self._centroidZp = self._centroidZ0
        self._centroidZc = self._centroidZ0
        self._centroidZn = self._centroidZ0

        return  # a new instance of type tetrahedron

    # volume of an irregular tetrahedron
    def _volTet(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
        # compute the square of the lengths of its six edges
        l12 = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2
        l13 = (x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2
        l14 = (x4 - x1)**2 + (y4 - y1)**2 + (z4 - z1)**2
        l23 = (x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2
        l24 = (x4 - x2)**2 + (y4 - y2)**2 + (z4 - z2)**2
        l34 = (x4 - x3)**2 + (y4 - y3)**2 + (z4 - z3)**2

        # prepare matrix from which the volume is computed (Ref. 2 above)
        A = np.array([[0.0, 1.0, 1.0, 1.0, 1.0],
                      [1.0, 0.0, l12, l13, l14],
                      [1.0, l12, 0.0, l23, l24],
                      [1.0, l13, l23, 0.0, l34],
                      [1.0, l14, l24, l34, 0.0]])
        volT = m.sqrt(linalg.det(A) / 288.0)
        return volT

    def toString(self, state):
        if self._number < 10:
            s = 'tetrahedron[0'
        else:
            s = 'tetrahedron['
        s = s + str(self._number)
        s = s + '] has vertices: \n'
        if isinstance(state, str):
            s = s + '   1: ' + self._vertex[1].toString(state) + '\n'
            s = s + '   2: ' + self._vertex[2].toString(state) + '\n'
            s = s + '   3: ' + self._vertex[3].toString(state) + '\n'
            s = s + '   4: ' + self._vertex[4].toString(state)
        else:
            raise RuntimeError(
                     "Error: unknown state {} in call to tetrahedron.toString."
                     .format(str(state)))
        return s

    def number(self):
        return self._number

    def vertexNumbers(self):
        numbers = sorted(self._setOfVertices)
        return numbers[0], numbers[1], numbers[2], numbers[3]

    def hasVertex(self, number):
        return number in self._setOfVertices

    def getVertex(self, number):
        if self._vertex[1].number() == number:
            return self._vertex[1]
        elif self._vertex[2].number() == number:
            return self._vertex[2]
        elif self._vertex[3].number() == number:
            return self._vertex[3]
        elif self._vertex[4].number() == number:
            return self._vertex[4]
        else:
            raise RuntimeError(
                     'Error: the requested vertex {} is not in tetrhaderon {}.'
                     .format(number, self._number))

    def gaussPoints(self):
        return self._gaussPts

    def update(self):
        # computes the fields positioned at the next time step

        # get the updated coordinates for the vetices of the tetrahedron
        x1, y1, z1 = self._vertex[1].coordinates('next')
        x2, y2, z2 = self._vertex[2].coordinates('next')
        x3, y3, z3 = self._vertex[3].coordinates('next')
        x4, y4, z4 = self._vertex[4].coordinates('next')

        # determine the volume of this tetrahedron in this next state
        self._Vn = self.volTet(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        # locate the centroid of this tetrahedrond in this next state
        self._centroidXn = self._centroidSF.interpolate(x1, x2, x3, x4)
        self._centroidYn = self._centroidSF.interpolate(y1, y2, y3, y4)
        self._centroidZn = self._centroidSF.interpolate(z1, z2, z3, z4)

        # coordiantes for the next updated vertices as tuples
        v1n = (x1, y1, z1)
        v2n = (x2, y2, z2)
        v3n = (x3, y3, z3)
        v4n = (x4, y4, z4)

        # coordinates for the reference vertices as tuples
        v1r = (self._x10, self._y10, self._z10)
        v2r = (self._x20, self._y20, self._z20)
        v3r = (self._x30, self._y30, self._z30)
        v4r = (self._x40, self._y40, self._y40)

        # establish the deformation and displacement gradients as dictionaries
        if self._gaussPts == 1:
            # displacement gradients located at the Gauss points of tetrahedron
            self._Gn[1] = self._shapeFns[1].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            # deformation gradients located at the Gauss points of tetrahedron
            self._Fn[1] = self._shapeFns[1].F(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
        elif self._gaussPts == 4:
            # displacement gradients located at the Gauss points of tetrahedron
            self._Gn[1] = self._shapeFns[1].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Gn[2] = self._shapeFns[2].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Gn[3] = self._shapeFns[3].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Gn[4] = self._shapeFns[4].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            # deformation gradients located at the Gauss points of tetrahedron
            self._Fn[1] = self._shapeFns[1].F(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Fn[2] = self._shapeFns[2].F(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Fn[3] = self._shapeFns[3].F(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Fn[4] = self._shapeFns[4].F(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
        else:  # gaussPts = 5
            # displacement gradients located at the Gauss points of tetrahedron
            self._Gn[1] = self._shapeFns[1].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Gn[2] = self._shapeFns[2].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Gn[3] = self._shapeFns[3].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Gn[4] = self._shapeFns[4].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Gn[5] = self._shapeFns[5].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            # deformation gradients located at the Gauss points of tetrahedron
            self._Fn[1] = self._shapeFns[1].F(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Fn[2] = self._shapeFns[2].F(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Fn[3] = self._shapeFns[3].F(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Fn[4] = self._shapeFns[4].F(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Fn[5] = self._shapeFns[5].F(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)

        return  # nothing

    def advance(self):
        # advance the geometric properties of the pentagon
        self._Vp = self._Vc
        self._Vc = self._Vn
        self._centroidXp = self._centroidXc
        self._centroidYp = self._centroidYc
        self._centroidZp = self._centroidZc
        self._centroidXc = self._centroidXn
        self._centroidYc = self._centroidYn
        self._centroidZc = self._centroidZn

        # advance the matrix fields associated with each Gauss point
        for i in range(1, self._gaussPts+1):
            self._Fp[i][:, :] = self._Fc[i][:, :]
            self._Fc[i][:, :] = self._Fn[i][:, :]
            self._Gp[i][:, :] = self._Gc[i][:, :]
            self._Gc[i][:, :] = self._Gn[i][:, :]

        return  # nothing

    def volume(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._Vc
            elif state == 'n' or state == 'next':
                return self._Vn
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._Vp
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._V0
            else:
                raise RuntimeError(
                       "Error: unknown state {} in call to tetrahedron.volume."
                       .format(state))
        else:
            raise RuntimeError(
                       "Error: unknown state {} in call to tetrahedron.volume."
                       .format(str(state)))

    def volumetricStretch(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return (self._Vc / self._V0)**(1.0 / 3.0)
            elif state == 'n' or state == 'next':
                return (self._Vn / self._V0)**(1.0 / 3.0)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return (self._Vp / self._V0)**(1.0 / 3.0)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 1.0
            else:
                raise RuntimeError("Error: unknown state {} in a call to " +
                                   "tetrahedron.volumetricStretch."
                                   .format(state))
        else:
            raise RuntimeError("Error: unknown state {} in a call to " +
                               "tetrahedron.volumetricStretch."
                               .format(str(state)))

    def volumetricStrain(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return m.log(self._Vc / self._V0) / 3.0
            elif state == 'n' or state == 'next':
                return m.log(self._Vn / self._V0) / 3.0
            elif state == 'p' or state == 'prev' or state == 'previous':
                return m.log(self._Vp / self._V0) / 3.0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("Error: unknown state {} in a call to " +
                                   "tetrahedron.volumetricStrain."
                                   .format(state))
        else:
            raise RuntimeError("Error: unknown state {} in a call to " +
                               "tetrahedron.volumetricStrain."
                               .format(str(state)))

    def dVolumetricStrain(self, state):
        if isinstance(state, str):
            h = 2.0 * self._h
            if state == 'c' or state == 'curr' or state == 'current':
                # use second-order central difference formula
                dVol = (self._Vn - self._Vp) / h
                return (dVol / self._Vc) / 3.0
            elif state == 'n' or state == 'next':
                # use second-order backward difference formula
                dVol = (3.0 * self._Vn - 4.0 * self._Vc + self._Vp) / h
                return (dVol / self._Vn) / 3.0
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use second-order forward difference formula
                dVol = (-self._Vn + 4.0 * self._Vc - 3.0 * self._Vp) / h
                return (dVol / self._Vp) / 3.0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("Error: unknown state {} in a call to " +
                                   "tetrahedron.dVolumetricStrain."
                                   .format(state))
        else:
            raise RuntimeError("Error: unknown state {} in a call to " +
                               "tetrahedron.dVolumetricStrain."
                               .format(str(state)))

    def centroid(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                cx = self._centroidXc
                cy = self._centroidYc
                cz = self._centroidZc
            elif state == 'n' or state == 'next':
                cx = self._centroidXn
                cy = self._centroidYn
                cz = self._centroidZn
            elif state == 'p' or state == 'prev' or state == 'previous':
                cx = self._centroidXp
                cy = self._centroidYp
                cz = self._centroidZp
            elif state == 'r' or state == 'ref' or state == 'reference':
                cx = self._centroidX0
                cy = self._centroidY0
                cz = self._centroidZ0
            else:
                raise RuntimeError("Error: unknown state {} in a call to " +
                                   "tetrahedron.centroid.".format(state))
        else:
            raise RuntimeError("Error: unknown state {} in a call to " +
                               "tetrahedron.centroid.".format(str(state)))
        return np.array([cx, cy, cz])

    def displacement(self, state):
        x0, y0, z0 = self.centroid('reference')
        x, y, z = self.centroid(state)
        ux = x - x0
        uy = y - y0
        uz = z - z0
        return np.array([ux, uy, uz])

    def velocity(self, state):
        if isinstance(state, str):
            h = 2.0 * self._h
            xp, yp, zp = self.centroid('prev')
            xc, yc, zc = self.centroid('curr')
            xn, yn, zn = self.centroid('next')
            if state == 'c' or state == 'curr' or state == 'current':
                # use second-order central difference formula
                vx = (xn - xp) / h
                vy = (yn - yp) / h
                vz = (zn - zp) / h
            elif state == 'n' or state == 'next':
                # use second-order backward difference formula
                vx = (3.0 * xn - 4.0 * xc + xp) / h
                vy = (3.0 * yn - 4.0 * yc + yp) / h
                vz = (3.0 * zn - 4.0 * zc + zp) / h
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use second-order forward difference formula
                vx = (-xn + 4.0 * xc - 3.0 * xp) / h
                vy = (-yn + 4.0 * yc - 3.0 * yp) / h
                vz = (-zn + 4.0 * zc - 3.0 * zp) / h
            elif state == 'r' or state == 'ref' or state == 'reference':
                vx = 0.0
                vy = 0.0
                vz = 0.0
            else:
                raise RuntimeError("Error: unknown state {} in a call to " +
                                   "tetrahedron.velocity.".format(state))
        else:
            raise RuntimeError("Error: unknown state {} in a call to " +
                               "tetrahedron.velocity.".format(str(state)))
        return np.array([vx, vy, vz])

    def acceleration(self, state):
        if isinstance(state, str):
            if state == 'r' or state == 'ref' or state == 'reference':
                ax = 0.0
                ay = 0.0
                az = 0.0
            else:
                h2 = self._h**2
                xp, yp, zp = self.prevCentroid()
                xc, yc, zc = self.currCentroid()
                xn, yn, zn = self.nextCentroid()
                # use second-order central difference formula
                ax = (xn - 2.0 * xc + xp) / h2
                ay = (yn - 2.0 * yc + yp) / h2
                az = (zn - 2.0 * zc + zp) / h2
        else:
            raise RuntimeError("Error: unknown state {} in call a to " +
                               "tetrahedron.acceleration.".format(str(state)))
        return np.array([ax, ay, az])

    def shapeFunction(self, gaussPt):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "tetrahedron.shapeFunction and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) + "to " +
                                   "tetrahedron.shapeFunction and you sent {}."
                                   .format(gaussPt))
            sf = self._shapeFns[gaussPt]
        return sf

    def massMatrix(self, rho):
        if rho <= 0.0:
            raise RuntimeError("Mass density rho must be positive, you sent " +
                               "{} to tetrahedron.massMatrix.".format(rho))

        # assign coordinates at the vertices in the reference configuration
        x10 = (self._x10, self._y10, self._z10)
        x20 = (self._x20, self._y20, self._z20)
        x30 = (self._x30, self._y30, self._z30)
        x40 = (self._x40, self._y40, self._z40)

        # determine the mass matrix
        if self._gaussPts == 1:
            # 'natural' weight of the element
            wgt = 1.0 / 6.0
            wel = np.array([wgt])

            detJ = self._shapeFns[1].detJacobian(x10, x20, x30, x40)

            nn1 = np.dot(np.transpose(self._shapeFns[1].Nmatx),
                         self._shapeFns[1].Nmatx)

            # Integration to get the mass matrix for 1 Gauss point
            mass = rho * (detJ * wel[0]) * nn1
        elif self._gaussPts == 4:
            # 'natural' weights of the element
            wgt = 1.0 / 24.0
            wel = np.array([wgt, wgt, wgt, wgt])

            detJ1 = self._shapeFns[1].detJacobian(x10, x20, x30, x40)
            detJ2 = self._shapeFns[2].detJacobian(x10, x20, x30, x40)
            detJ3 = self._shapeFns[3].detJacobian(x10, x20, x30, x40)
            detJ4 = self._shapeFns[4].detJacobian(x10, x20, x30, x40)

            nn1 = np.dot(np.transpose(self._shapeFns[1].Nmatx),
                         self._shapeFns[1].Nmatx)
            nn2 = np.dot(np.transpose(self._shapeFns[2].Nmatx),
                         self._shapeFns[2].Nmatx)
            nn3 = np.dot(np.transpose(self._shapeFns[3].Nmatx),
                         self._shapeFns[3].Nmatx)
            nn4 = np.dot(np.transpose(self._shapeFns[4].Nmatx),
                         self._shapeFns[4].Nmatx)

            # Integration to get the mass matrix for 4 Gauss points
            mass = (rho * (detJ1 * wel[0] * nn1 + detJ2 * wel[1] * nn2 +
                           detJ3 * wel[2] * nn3 + detJ4 * wel[3] * nn4))
        else:  # gaussPts = 5
            # 'natural' weights of the element
            wgt1 = -2.0 / 15.0
            wgt2 = 3.0 / 40.0
            wel = np.array([wgt1, wgt2, wgt2, wgt2, wgt2])

            detJ1 = self._shapeFns[1].detJacobian(x10, x20, x30, x40)
            detJ2 = self._shapeFns[2].detJacobian(x10, x20, x30, x40)
            detJ3 = self._shapeFns[3].detJacobian(x10, x20, x30, x40)
            detJ4 = self._shapeFns[4].detJacobian(x10, x20, x30, x40)
            detJ5 = self._shapeFns[5].detJacobian(x10, x20, x30, x40)

            nn1 = np.dot(np.transpose(self._shapeFns[1].Nmatx),
                         self._shapeFns[1].Nmatx)
            nn2 = np.dot(np.transpose(self._shapeFns[2].Nmatx),
                         self._shapeFns[2].Nmatx)
            nn3 = np.dot(np.transpose(self._shapeFns[3].Nmatx),
                         self._shapeFns[3].Nmatx)
            nn4 = np.dot(np.transpose(self._shapeFns[4].Nmatx),
                         self._shapeFns[4].Nmatx)
            nn5 = np.dot(np.transpose(self._shapeFns[5].Nmatx),
                         self._shapeFns[5].Nmatx)

            # Integration to get the mass Matrix for 5 Gauss points
            mass = (rho * (detJ1 * wel[0] * nn1 + detJ2 * wel[1] * nn2 +
                           detJ3 * wel[2] * nn3 + detJ4 * wel[3] * nn4 +
                           detJ5 * wel[4] * nn5))
        return mass

    # displacement gradient at a Gauss point
    def G(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "tetrahedron.G and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to tetrahedron.G and you sent {}."
                                   .format(gaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Gc[gaussPt])
            elif state == 'n' or state == 'next':
                return np.copy(self._Gn[gaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Gp[gaussPt])
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._G0[gaussPt])
            else:
                raise RuntimeError("Error: unknown state {} in call a to " +
                                   "tetrahedron.G.".format(state))
        else:
            raise RuntimeError("Error: unknown state {} in a call to " +
                               "tetrahedron.G.".format(str(state)))

    # deformation gradient at a Gauss point
    def F(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in a call " +
                                   "to tetrahedron.F and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in a "
                                   .format(self._gaussPts) +
                                   "call to tetrahedron.F and you sent {}."
                                   .format(gaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Fc[gaussPt])
            elif state == 'n' or state == 'next':
                return np.copy(self._Fn[gaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Fp[gaussPt])
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._F0[gaussPt])
            else:
                raise RuntimeError("Error: unknown state {} in a call to " +
                                   "tetrahedron.F.".format(state))
        else:
            raise RuntimeError("Error: unknown state {} in a call to " +
                               "tetrahedron.F.".format(str(state)))
