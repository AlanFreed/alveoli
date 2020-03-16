#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import materialProperties as mp
import math as m
import numpy as np
from numpy.linalg import det
from numpy.linalg import inv

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
__update__ = "10-11-2019"
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
        and updates all affected fields.  To be called after all vertices have
        had their coordinates updated.  This may be called multiple times
        before freezing it with a call to advance

    t.advance()
        assigns fields belonging to the current location into their cournter-
        parts in the previous location, and then it assigns their next values
        into the current location, thereby freezing the location of the present
        next-location in preparation to advance to the next step along a
        solution path

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

    omegaMtx = t.spin(state)
        returns a 3x3 skew symmetric matrix that describes the time rate of
        rotation, i.e., spin, of the local tetrahedral coordinate system about
        the fixed dodecahedral coordinate system with reference base vectors.
        The returned matrix associates with configuration 'state'

    The fundamental fields of kinematics

    gMtx = t.G(gaussPt, state)
        returns 3x3 matrix describing the displacement gradient for the
        tetrahedron at 'gaussPt' in configuration 'state'

    fMtx = t.F(gaussPt, state)
        returns 3x3 matrix describing the deformation gradient for the
        tetrahedron at 'gaussPt' in configuration 'state'

    lMtx = t.L(gaussPt, state)
        returns the velocity gradient at the specified Gauss point for the
        specified configuration

    Fields needed to construct finite element representations.  The mass and
    stiffness matrices are 12x12

    sf = t.shapeFunction(gaussPt):
        returns the shape function associated with the specified Gauss point

    massM = t.massMatrix(rho)
        returns an average of the lumped and consistent mass matrices (ensures
        the mass matrix is not singular) of dimension 12x12 for the chosen
        number of Gauss points for a tetrahedron whose mass density, rho,
        is specified.

    kMtx = c.stiffnessMatrix()
        returns a tangent stiffness matrix for the chosen number of Gauss
        points.

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
            raise RuntimeError("The stepsize sent to the tetrahedron " +
                               "constructor wasn't positive.")
        # check the number of Gauss points to use
        if gaussPts == 1 or gaussPts == 4 or gaussPts == 5:
            self._gaussPts = gaussPts
        else:
            raise RuntimeError('{} Gauss points were '.format(gaussPts) +
                               'specified in the tetrahedron constructor; ' +
                               'it must be 1, 4 or 5.')

        # create a shape function for the centroid of the tetrahedron
        self._centroidSF = shapeFunction(0.25, 0.25, 0.25)

        # establish shape functions located at the Gauss points (xi, eta, zeta)
        if gaussPts == 1:
            # this single Gauss point has a weight of 1/6
            xi = 0.25
            eta = 0.25
            zeta = 0.25
            sf111 = shapeFunction(xi, eta, zeta)

            self._shapeFns = {
                111: sf111
                }
        elif gaussPts == 4:
            # each of these four Gauss points has a weight of 1/24
            a = (5.0 - m.sqrt(5.0)) / 20.0
            b = (5.0 + 3.0 * m.sqrt(5.0)) / 20.0

            xi1 = a
            eta1 = a
            zeta1 = a
            sf111 = shapeFunction(xi1, eta1, zeta1)
            xi1 = a
            eta1 = a
            zeta2 = a
            sf112 = shapeFunction(xi1, eta1, zeta2)
            xi1 = a
            eta1 = a
            zeta3 = a
            sf113 = shapeFunction(xi1, eta1, zeta3)
            xi1 = a
            eta1 = a
            zeta4 = b
            sf114 = shapeFunction(xi1, eta1, zeta4)
            
            xi1 = a
            eta2 = a
            zeta1 = a
            sf121 = shapeFunction(xi1, eta2, zeta1)
            xi1 = a
            eta2 = a
            zeta2 = a
            sf122 = shapeFunction(xi1, eta2, zeta2)
            xi1 = a
            eta2 = a
            zeta3 = a
            sf123 = shapeFunction(xi1, eta2, zeta3)
            xi1 = a
            eta2 = a
            zeta4 = b
            sf124 = shapeFunction(xi1, eta2, zeta4)

            xi1 = a
            eta3 = b
            zeta1 = a
            sf131 = shapeFunction(xi1, eta3, zeta1)
            xi1 = a
            eta3 = b
            zeta2 = a
            sf132 = shapeFunction(xi1, eta3, zeta2)
            xi1 = a
            eta3 = b
            zeta3 = a
            sf133 = shapeFunction(xi1, eta3, zeta3)
            xi1 = a
            eta3 = b
            zeta4 = b
            sf134 = shapeFunction(xi1, eta3, zeta4)

            xi1 = a
            eta4 = a
            zeta1 = a
            sf141 = shapeFunction(xi1, eta4, zeta1)
            xi1 = a
            eta4 = a
            zeta2 = a
            sf142 = shapeFunction(xi1, eta4, zeta2)
            xi1 = a
            eta4 = a
            zeta3 = a
            sf143 = shapeFunction(xi1, eta4, zeta3)
            xi1 = a
            eta4 = a
            zeta4 = b
            sf144 = shapeFunction(xi1, eta4, zeta4)
            
            xi2 = b
            eta1 = a
            zeta1 = a
            sf211 = shapeFunction(xi2, eta1, zeta1)
            xi2 = b
            eta1 = a
            zeta2 = a
            sf212 = shapeFunction(xi2, eta1, zeta2)
            xi2 = b
            eta1 = a
            zeta3 = a
            sf213 = shapeFunction(xi2, eta1, zeta3)
            xi2 = b
            eta1 = a
            zeta4 = b
            sf214 = shapeFunction(xi2, eta1, zeta4)
            
            xi2 = b
            eta2 = a
            zeta1 = a
            sf221 = shapeFunction(xi2, eta2, zeta1)
            xi2 = b
            eta2 = a
            zeta2 = a
            sf222 = shapeFunction(xi2, eta2, zeta2)
            xi2 = b
            eta2 = a
            zeta3 = a
            sf223 = shapeFunction(xi2, eta2, zeta3)
            xi2 = b
            eta2 = a
            zeta4 = b
            sf224 = shapeFunction(xi2, eta2, zeta4)

            xi2 = b
            eta3 = b
            zeta1 = a
            sf231 = shapeFunction(xi2, eta3, zeta1)
            xi2 = b
            eta3 = b
            zeta2 = a
            sf232 = shapeFunction(xi2, eta3, zeta2)
            xi2 = b
            eta3 = b
            zeta3 = a
            sf233 = shapeFunction(xi2, eta3, zeta3)
            xi2 = b
            eta3 = b
            zeta4 = b
            sf234 = shapeFunction(xi2, eta3, zeta4)

            xi2 = b
            eta4 = a
            zeta1 = a
            sf241 = shapeFunction(xi2, eta4, zeta1)
            xi2 = b
            eta4 = a
            zeta2 = a
            sf242 = shapeFunction(xi2, eta4, zeta2)
            xi2 = b
            eta4 = a
            zeta3 = a
            sf243 = shapeFunction(xi2, eta4, zeta3)
            xi2 = b
            eta4 = a
            zeta4 = b
            sf244 = shapeFunction(xi2, eta4, zeta4)            

            xi3 = a
            eta1 = a
            zeta1 = a
            sf311 = shapeFunction(xi3, eta1, zeta1)
            xi3 = a
            eta1 = a
            zeta2 = a
            sf312 = shapeFunction(xi3, eta1, zeta2)
            xi3 = a
            eta1 = a
            zeta3 = a
            sf313 = shapeFunction(xi3, eta1, zeta3)
            xi3 = a
            eta1 = a
            zeta4 = b
            sf314 = shapeFunction(xi3, eta1, zeta4)
            
            xi3 = a
            eta2 = a
            zeta1 = a
            sf321 = shapeFunction(xi3, eta2, zeta1)
            xi3 = a
            eta2 = a
            zeta2 = a
            sf322 = shapeFunction(xi3, eta2, zeta2)
            xi3 = a
            eta2 = a
            zeta3 = a
            sf323 = shapeFunction(xi3, eta2, zeta3)
            xi3 = a
            eta2 = a
            zeta4 = b
            sf324 = shapeFunction(xi3, eta2, zeta4)

            xi3 = a
            eta3 = b
            zeta1 = a
            sf331 = shapeFunction(xi3, eta3, zeta1)
            xi3 = a
            eta3 = b
            zeta2 = a
            sf332 = shapeFunction(xi3, eta3, zeta2)
            xi3 = a
            eta3 = b
            zeta3 = a
            sf333 = shapeFunction(xi3, eta3, zeta3)
            xi3 = a
            eta3 = b
            zeta4 = b
            sf334 = shapeFunction(xi3, eta3, zeta4)

            xi3 = a
            eta4 = a
            zeta1 = a
            sf341 = shapeFunction(xi3, eta4, zeta1)
            xi3 = a
            eta4 = a
            zeta2 = a
            sf342 = shapeFunction(xi3, eta4, zeta2)
            xi3 = a
            eta4 = a
            zeta3 = a
            sf343 = shapeFunction(xi3, eta4, zeta3)
            xi3 = a
            eta4 = a
            zeta4 = b
            sf344 = shapeFunction(xi3, eta4, zeta4)   

            xi4 = a
            eta1 = a
            zeta1 = a
            sf411 = shapeFunction(xi4, eta1, zeta1)
            xi4 = a
            eta1 = a
            zeta2 = a
            sf412 = shapeFunction(xi4, eta1, zeta2)
            xi4 = a
            eta1 = a
            zeta3 = a
            sf413 = shapeFunction(xi4, eta1, zeta3)
            xi4 = a
            eta1 = a
            zeta4 = b
            sf414 = shapeFunction(xi4, eta1, zeta4)
            
            xi4 = a
            eta2 = a
            zeta1 = a
            sf421 = shapeFunction(xi4, eta2, zeta1)
            xi4 = a
            eta2 = a
            zeta2 = a
            sf422 = shapeFunction(xi4, eta2, zeta2)
            xi4 = a
            eta2 = a
            zeta3 = a
            sf423 = shapeFunction(xi4, eta2, zeta3)
            xi4 = a
            eta2 = a
            zeta4 = b
            sf424 = shapeFunction(xi4, eta2, zeta4)

            xi4 = a
            eta3 = b
            zeta1 = a
            sf431 = shapeFunction(xi4, eta3, zeta1)
            xi4 = a
            eta3 = b
            zeta2 = a
            sf432 = shapeFunction(xi4, eta3, zeta2)
            xi4 = a
            eta3 = b
            zeta3 = a
            sf433 = shapeFunction(xi4, eta3, zeta3)
            xi4 = a
            eta3 = b
            zeta4 = b
            sf434 = shapeFunction(xi4, eta3, zeta4)

            xi4 = a
            eta4 = a
            zeta1 = a
            sf441 = shapeFunction(xi4, eta4, zeta1)
            xi4 = a
            eta4 = a
            zeta2 = a
            sf442 = shapeFunction(xi4, eta4, zeta2)
            xi4 = a
            eta4 = a
            zeta3 = a
            sf443 = shapeFunction(xi4, eta4, zeta3)
            xi4 = a
            eta4 = a
            zeta4 = b
            sf444 = shapeFunction(xi4, eta4, zeta4)

            self._shapeFns = {
                111: sf111,
                112: sf112,
                113: sf113,
                114: sf114,
                121: sf121,
                122: sf122,
                123: sf123,
                124: sf124,
                131: sf131,
                132: sf132,
                133: sf133,
                134: sf134,
                141: sf141,
                142: sf142,
                143: sf143,
                144: sf144,
                211: sf211,
                212: sf212,
                213: sf213,
                214: sf214,
                221: sf221,
                222: sf222,
                223: sf223,
                224: sf224,
                231: sf231,
                232: sf232,
                233: sf233,
                234: sf234,
                241: sf241,
                242: sf242,
                243: sf243,
                244: sf244,
                311: sf311,
                312: sf312,
                313: sf313,
                314: sf314,
                321: sf321,
                322: sf322,
                323: sf323,
                324: sf324,
                331: sf331,
                332: sf332,
                333: sf333,
                334: sf334,
                341: sf341,
                342: sf342,
                343: sf343,
                344: sf344,
                411: sf411,
                412: sf412,
                413: sf413,
                414: sf414,
                421: sf421,
                422: sf422,
                423: sf423,
                424: sf424,
                431: sf431,
                432: sf432,
                433: sf433,
                434: sf434,
                441: sf441,
                442: sf442,
                443: sf443,
                444: sf444
            }
        else:  # gaussPts = 5
            xi1 = 0.25
            eta1 = 0.25
            zeta1 = 0.25
            sf111 = shapeFunction(xi1, eta1, zeta1)
            xi1 = 0.25
            eta1 = 0.25
            zeta2 = 1.0 / 6.0
            sf112 = shapeFunction(xi1, eta1, zeta2)
            xi1 = 0.25
            eta1 = 0.25
            zeta3 = 1.0 / 6.0
            sf113 = shapeFunction(xi1, eta1, zeta3)
            xi1 = 0.25
            eta1 = 0.25
            zeta4 = 0.5
            sf114 = shapeFunction(xi1, eta1, zeta4)
            xi1 = 0.25
            eta1 = 0.25
            zeta5 = 1.0 / 6.0
            sf115 = shapeFunction(xi1, eta1, zeta5)
            
            xi1 = 0.25
            eta2 = 1.0 / 6.0
            zeta1 = 0.25
            sf121 = shapeFunction(xi1, eta2, zeta1)
            xi1 = 0.25
            eta2 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf122 = shapeFunction(xi1, eta2, zeta2)
            xi1 = 0.25
            eta2 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf123 = shapeFunction(xi1, eta2, zeta3)
            xi1 = 0.25
            eta2 = 1.0 / 6.0
            zeta4 = 0.5
            sf124 = shapeFunction(xi1, eta2, zeta4)
            xi1 = 0.25
            eta2 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf125 = shapeFunction(xi1, eta2, zeta5)
            
            xi1 = 0.25
            eta3 = 0.5
            zeta1 = 0.25
            sf131 = shapeFunction(xi1, eta3, zeta1)
            xi1 = 0.25
            eta3 = 0.5
            zeta2 = 1.0 / 6.0
            sf132 = shapeFunction(xi1, eta3, zeta2)
            xi1 = 0.25
            eta3 = 0.5
            zeta3 = 1.0 / 6.0
            sf133 = shapeFunction(xi1, eta3, zeta3)
            xi1 = 0.25
            eta3 = 0.5
            zeta4 = 0.5
            sf134 = shapeFunction(xi1, eta3, zeta4)
            xi1 = 0.25
            eta3 = 0.5
            zeta5 = 1.0 / 6.0
            sf135 = shapeFunction(xi1, eta3, zeta5)
            
            xi1 = 0.25
            eta4 = 1.0 / 6.0
            zeta1 = 0.25
            sf141 = shapeFunction(xi1, eta4, zeta1)
            xi1 = 0.25
            eta4 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf142 = shapeFunction(xi1, eta4, zeta2)
            xi1 = 0.25
            eta4 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf143 = shapeFunction(xi1, eta4, zeta3)
            xi1 = 0.25
            eta4 = 1.0 / 6.0
            zeta4 = 0.5
            sf144 = shapeFunction(xi1, eta4, zeta4)
            xi1 = 0.25
            eta4 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf145 = shapeFunction(xi1, eta4, zeta5)
            
            xi1 = 0.25
            eta5 = 1.0 / 6.0
            zeta1 = 0.25
            sf151 = shapeFunction(xi1, eta5, zeta1)
            xi1 = 0.25
            eta5 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf152 = shapeFunction(xi1, eta5, zeta2)
            xi1 = 0.25
            eta5 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf153 = shapeFunction(xi1, eta5, zeta3)
            xi1 = 0.25
            eta5 = 1.0 / 6.0
            zeta4 = 0.5
            sf154 = shapeFunction(xi1, eta5, zeta4)
            xi1 = 0.25
            eta5 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf155 = shapeFunction(xi1, eta5, zeta5)           
            
            xi2 = 0.5
            eta1 = 0.25
            zeta1 = 0.25
            sf211 = shapeFunction(xi2, eta1, zeta1)
            xi2 = 0.5
            eta1 = 0.25
            zeta2 = 1.0 / 6.0
            sf212 = shapeFunction(xi2, eta1, zeta2)
            xi2 = 0.5
            eta1 = 0.25
            zeta3 = 1.0 / 6.0
            sf213 = shapeFunction(xi2, eta1, zeta3)
            xi2 = 0.5
            eta1 = 0.25
            zeta4 = 0.5
            sf214 = shapeFunction(xi2, eta1, zeta4)
            xi2 = 0.5
            eta1 = 0.25
            zeta5 = 1.0 / 6.0
            sf215 = shapeFunction(xi2, eta1, zeta5)
            
            xi2 = 0.5
            eta2 = 1.0 / 6.0
            zeta1 = 0.25
            sf221 = shapeFunction(xi2, eta2, zeta1)
            xi2 = 0.5
            eta2 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf222 = shapeFunction(xi2, eta2, zeta2)
            xi2 = 0.5
            eta2 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf223 = shapeFunction(xi2, eta2, zeta3)
            xi2 = 0.5
            eta2 = 1.0 / 6.0
            zeta4 = 0.5
            sf224 = shapeFunction(xi2, eta2, zeta4)
            xi2 = 0.5
            eta2 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf225 = shapeFunction(xi2, eta2, zeta5)
            
            xi2 = 0.5
            eta3 = 0.5
            zeta1 = 0.25
            sf231 = shapeFunction(xi2, eta3, zeta1)
            xi2 = 0.5
            eta3 = 0.5
            zeta2 = 1.0 / 6.0
            sf232 = shapeFunction(xi2, eta3, zeta2)
            xi2 = 0.5
            eta3 = 0.5
            zeta3 = 1.0 / 6.0
            sf233 = shapeFunction(xi2, eta3, zeta3)
            xi2 = 0.5
            eta3 = 0.5
            zeta4 = 0.5
            sf234 = shapeFunction(xi2, eta3, zeta4)
            xi2 = 0.5
            eta3 = 0.5
            zeta5 =1.0 / 6.0
            sf235 = shapeFunction(xi2, eta3, zeta5)
            
            xi2 = 0.5
            eta4 = 1.0 / 6.0
            zeta1 = 0.25
            sf241 = shapeFunction(xi2, eta4, zeta1)
            xi2 = 0.5
            eta4 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf242 = shapeFunction(xi2, eta4, zeta2)
            xi2 = 0.5
            eta4 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf243 = shapeFunction(xi2, eta4, zeta3)
            xi2 = 0.5
            eta4 = 1.0 / 6.0
            zeta4 = 0.5
            sf244 = shapeFunction(xi2, eta4, zeta4)
            xi2 = 0.5
            eta4 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf245 = shapeFunction(xi2, eta4, zeta5)
            
            xi2 = 0.5
            eta5 = 1.0 / 6.0
            zeta1 = 0.25
            sf251 = shapeFunction(xi2, eta5, zeta1)
            xi2 = 0.5
            eta5 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf252 = shapeFunction(xi2, eta5, zeta2)
            xi2 = 0.5
            eta5 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf253 = shapeFunction(xi2, eta5, zeta3)
            xi2 = 0.5
            eta5 = 1.0 / 6.0
            zeta4 = 0.5
            sf254 = shapeFunction(xi2, eta5, zeta4)
            xi2 = 0.5
            eta5 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf255 = shapeFunction(xi2, eta5, zeta5)
            
            xi3 = 1.0 / 6.0
            eta1 = 0.25
            zeta1 = 0.25
            sf311 = shapeFunction(xi3, eta1, zeta1)
            xi3 = 1.0 / 6.0
            eta1 = 0.25
            zeta2 = 1.0 / 6.0
            sf312 = shapeFunction(xi3, eta1, zeta2)
            xi3 = 1.0 / 6.0
            eta1 = 0.25
            zeta3 = 1.0 / 6.0
            sf313 = shapeFunction(xi3, eta1, zeta3)
            xi3 = 1.0 / 6.0
            eta1 = 0.25
            zeta4 = 0.5
            sf314 = shapeFunction(xi3, eta1, zeta4)
            xi3 = 1.0 / 6.0
            eta1 = 0.25
            zeta5 = 1.0 / 6.0
            sf315 = shapeFunction(xi3, eta1, zeta5)
            
            xi3 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta1 = 0.25
            sf321 = shapeFunction(xi3, eta2, zeta1)
            xi3 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf322 = shapeFunction(xi3, eta2, zeta2)
            xi3 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf323 = shapeFunction(xi3, eta2, zeta3)
            xi3 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta4 = 0.5
            sf324 = shapeFunction(xi3, eta2, zeta4)
            xi3 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf325 = shapeFunction(xi3, eta2, zeta5)
            
            xi3 = 1.0 / 6.0
            eta3 = 0.5
            zeta1 = 0.25
            sf331 = shapeFunction(xi3, eta3, zeta1)
            xi3 = 1.0 / 6.0
            eta3 = 0.5
            zeta2 = 1.0 / 6.0
            sf332 = shapeFunction(xi3, eta3, zeta2)
            xi3 = 1.0 / 6.0
            eta3 = 0.5
            zeta3 = 1.0 / 6.0
            sf333 = shapeFunction(xi3, eta3, zeta3)
            xi3 = 1.0 / 6.0
            eta3 = 0.5
            zeta4 = 0.5
            sf334 = shapeFunction(xi3, eta3, zeta4)
            xi3 = 1.0 / 6.0
            eta3 = 0.5
            zeta5 = 1.0 / 6.0
            sf335 = shapeFunction(xi3, eta3, zeta5)
            
            xi3 = 1.0 / 6.0
            eta4 = 1.0 / 6.0
            zeta1 = 0.25
            sf341 = shapeFunction(xi3, eta4, zeta1)
            xi3 = 1.0 / 6.0
            eta4 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf342 = shapeFunction(xi3, eta4, zeta2)
            xi3 = 1.0 / 6.0
            eta4 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf343 = shapeFunction(xi3, eta4, zeta3)
            xi3 = 1.0 / 6.0
            eta4 = 1.0 / 6.0
            zeta4 = 0.5
            sf344 = shapeFunction(xi3, eta4, zeta4)
            xi3 = 1.0 / 6.0
            eta4 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf345 = shapeFunction(xi3, eta4, zeta5)
            
            xi3 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta1 = 0.25
            sf351 = shapeFunction(xi3, eta5, zeta1)
            xi3 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf352 = shapeFunction(xi3, eta5, zeta2)
            xi3 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf353 = shapeFunction(xi3, eta5, zeta3)
            xi3 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta4 = 0.5
            sf354 = shapeFunction(xi3, eta5, zeta4)
            xi3 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf355 = shapeFunction(xi3, eta5, zeta5)
                        
            xi4 = 1.0 / 6.0
            eta1 = 0.25
            zeta1 = 0.25
            sf411 = shapeFunction(xi4, eta1, zeta1)
            xi4 = 1.0 / 6.0
            eta1 = 0.25
            zeta2 = 1.0 / 6.0
            sf412 = shapeFunction(xi4, eta1, zeta2)
            xi4 = 1.0 / 6.0
            eta1 = 0.25
            zeta3 = 1.0 / 6.0
            sf413 = shapeFunction(xi4, eta1, zeta3)
            xi4 = 1.0 / 6.0
            eta1 = 0.25
            zeta4 = 0.5
            sf414 = shapeFunction(xi4, eta1, zeta4)
            xi4 = 1.0 / 6.0
            eta1 = 0.25
            zeta5 = 1.0 / 6.0
            sf415 = shapeFunction(xi4, eta1, zeta5)
            
            xi4 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta1 = 0.25
            sf421 = shapeFunction(xi4, eta2, zeta1)
            xi4 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf422 = shapeFunction(xi4, eta2, zeta2)
            xi4 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf423 = shapeFunction(xi4, eta2, zeta3)
            xi4 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta4 = 0.5
            sf424 = shapeFunction(xi4, eta2, zeta4)
            xi4 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf425 = shapeFunction(xi4, eta2, zeta5)
            
            xi4 = 1.0 / 6.0
            eta3 = 0.5
            zeta1 = 0.25
            sf431 = shapeFunction(xi4, eta3, zeta1)
            xi4 = 1.0 / 6.0
            eta3 = 0.5
            zeta2 = 1.0 / 6.0
            sf432 = shapeFunction(xi4, eta3, zeta2)
            xi4 = 1.0 / 6.0
            eta3 = 0.5
            zeta3 = 1.0 / 6.0
            sf433 = shapeFunction(xi4, eta3, zeta3)
            xi4 = 1.0 / 6.0
            eta3 = 0.5
            zeta4 = 0.5
            sf434 = shapeFunction(xi4, eta3, zeta4)
            xi4 = 1.0 / 6.0
            eta3 = 0.5
            zeta5 =1.0 / 6.0
            sf435 = shapeFunction(xi4, eta3, zeta5)
            
            xi4 = 1.0 / 6.0
            eta4 = 1.0 / 6.0
            zeta1 = 0.25
            sf441 = shapeFunction(xi4, eta4, zeta1)
            xi4 = 1.0 / 6.0
            eta4 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf442 = shapeFunction(xi4, eta4, zeta2)
            xi4 = 1.0 / 6.0
            eta4 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf443 = shapeFunction(xi4, eta4, zeta3)
            xi4 = 1.0 / 6.0
            eta4 = 1.0 / 6.0
            zeta4 = 0.5
            sf444 = shapeFunction(xi4, eta4, zeta4)
            xi4 = 1.0 / 6.0
            eta4 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf445 = shapeFunction(xi4, eta4, zeta5)
            
            xi4 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta1 = 0.25
            sf451 = shapeFunction(xi4, eta5, zeta1)
            xi4 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf452 = shapeFunction(xi4, eta5, zeta2)
            xi4 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf453 = shapeFunction(xi4, eta5, zeta3)
            xi4 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta4 = 0.5
            sf454 = shapeFunction(xi4, eta5, zeta4)
            xi4 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf455 = shapeFunction(xi4, eta5, zeta5)
            
            
            
            
            xi5 = 1.0 / 6.0
            eta1 = 0.25
            zeta1 = 0.25
            sf511 = shapeFunction(xi5, eta1, zeta1)
            xi5 = 1.0 / 6.0
            eta1 = 0.25
            zeta2 = 1.0 / 6.0
            sf512 = shapeFunction(xi5, eta1, zeta2)
            xi5 = 1.0 / 6.0
            eta1 = 0.25
            zeta3 = 1.0 / 6.0
            sf513 = shapeFunction(xi5, eta1, zeta3)
            xi5 = 1.0 / 6.0
            eta1 = 0.25
            zeta4 = 0.5
            sf514 = shapeFunction(xi5, eta1, zeta4)
            xi5 = 1.0 / 6.0
            eta1 = 0.25
            zeta5 = 1.0 / 6.0
            sf515 = shapeFunction(xi5, eta1, zeta5)
            
            xi5 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta1 = 0.25
            sf521 = shapeFunction(xi5, eta2, zeta1)
            xi5 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf522 = shapeFunction(xi5, eta2, zeta2)
            xi5 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf523 = shapeFunction(xi5, eta2, zeta3)
            xi5 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta4 = 0.5
            sf524 = shapeFunction(xi5, eta2, zeta4)
            xi5 = 1.0 / 6.0
            eta2 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf525 = shapeFunction(xi5, eta2, zeta5)
            
            xi5 = 1.0 / 6.0
            eta3 = 0.5
            zeta1 = 0.25
            sf531 = shapeFunction(xi5, eta3, zeta1)
            xi5 = 1.0 / 6.0
            eta3 = 0.5
            zeta2 = 1.0 / 6.0
            sf532 = shapeFunction(xi5, eta3, zeta2)
            xi5 = 1.0 / 6.0
            eta3 = 0.5
            zeta3 = 1.0 / 6.0
            sf533 = shapeFunction(xi5, eta3, zeta3)
            xi5 = 1.0 / 6.0
            eta3 = 0.5
            zeta4 = 0.5
            sf534 = shapeFunction(xi5, eta3, zeta4)
            xi5 = 1.0 / 6.0
            eta3 = 0.5
            zeta5 =1.0 / 6.0
            sf535 = shapeFunction(xi5, eta3, zeta5)
            
            xi5 = 1.0 / 6.0
            eta4 = 0.5
            zeta1 = 0.25
            sf541 = shapeFunction(xi5, eta4, zeta1)
            xi5 = 1.0 / 6.0
            eta4 = 0.5
            zeta2 = 1.0 / 6.0
            sf542 = shapeFunction(xi5, eta4, zeta2)
            xi5 = 1.0 / 6.0
            eta4 = 0.5
            zeta3 = 1.0 / 6.0
            sf543 = shapeFunction(xi5, eta4, zeta3)
            xi5 = 1.0 / 6.0
            eta4 = 0.5
            zeta4 = 0.5
            sf544 = shapeFunction(xi5, eta4, zeta4)
            xi5 = 1.0 / 6.0
            eta4 = 0.5
            zeta5 = 1.0 / 6.0
            sf545 = shapeFunction(xi5, eta4, zeta5)
            
            xi5 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta1 = 0.25
            sf551 = shapeFunction(xi5, eta5, zeta1)
            xi5 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta2 = 1.0 / 6.0
            sf552 = shapeFunction(xi5, eta5, zeta2)
            xi5 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta3 = 1.0 / 6.0
            sf553 = shapeFunction(xi5, eta5, zeta3)
            xi5 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta4 = 0.5
            sf554 = shapeFunction(xi5, eta5, zeta4)
            xi5 = 1.0 / 6.0
            eta5 = 1.0 / 6.0
            zeta5 = 1.0 / 6.0
            sf555 = shapeFunction(xi5, eta5, zeta5)
            

            self._shapeFns = {
                111: sf111,
                112: sf112,
                113: sf113,
                114: sf114,
                115: sf115,
                121: sf121,
                122: sf122,
                123: sf123,
                124: sf124,
                125: sf125,
                131: sf131,
                132: sf132,
                133: sf133,
                134: sf134,
                135: sf135,
                141: sf141,
                142: sf142,
                143: sf143,
                144: sf144,
                145: sf145,
                151: sf151,
                152: sf152,
                153: sf153,
                154: sf154,
                155: sf155,
                211: sf211,
                212: sf212,
                213: sf213,
                214: sf214,
                215: sf215,
                221: sf221,
                222: sf222,
                223: sf223,
                224: sf224,
                225: sf225,
                231: sf231,
                232: sf232,
                233: sf233,
                234: sf234,
                235: sf235,
                241: sf241,
                242: sf242,
                243: sf243,
                244: sf244,
                245: sf245,
                251: sf251,
                252: sf252,
                253: sf253,
                254: sf254,
                255: sf255,
                311: sf311,
                312: sf312,
                313: sf313,
                314: sf314,
                315: sf315,
                321: sf321,
                322: sf322,
                323: sf323,
                324: sf324,
                325: sf325,
                331: sf331,
                332: sf332,
                333: sf333,
                334: sf334,
                335: sf335,
                341: sf341,
                342: sf342,
                343: sf343,
                344: sf344,
                345: sf345,
                351: sf351,
                352: sf352,
                353: sf353,
                354: sf354,
                355: sf355,
                411: sf411,
                412: sf412,
                413: sf413,
                414: sf414,
                415: sf415,
                421: sf421,
                422: sf422,
                423: sf423,
                424: sf424,
                425: sf425,
                431: sf431,
                432: sf432,
                433: sf433,
                434: sf434,
                435: sf435,
                441: sf441,
                442: sf442,
                443: sf443,
                444: sf444,
                445: sf445,
                451: sf451,
                452: sf452,
                453: sf453,
                454: sf454,
                455: sf455,
                511: sf511,
                512: sf512,
                513: sf513,
                514: sf514,
                515: sf515,
                521: sf521,
                522: sf522,
                523: sf523,
                524: sf524,
                525: sf525,
                531: sf531,
                532: sf532,
                533: sf533,
                534: sf534,
                535: sf535,
                541: sf541,
                542: sf542,
                543: sf543,
                544: sf544,
                545: sf545,
                551: sf551,
                552: sf552,
                553: sf553,
                554: sf554,
                555: sf555                
            }
                        
        # create matrices for tetrahedron at its Gauss points via dictionaries
        # p implies previous, c implies current, n implies next
        if gaussPts == 1:
            # displacement gradients located at the Gauss points of tetrahedron
            self._G0 = {
                111: np.identity(3, dtype=float)
            }
            self._Gp = {
                111: np.zeros((3, 3), dtype=float)
            }
            self._Gc = {
                111: np.zeros((3, 3), dtype=float)
            }
            self._Gn = {
                111: np.zeros((3, 3), dtype=float)
            }
            # deformation gradients located at the Gauss points of tetrahedron
            self._F0 = {
                111: np.identity(3, dtype=float)
            }
            self._Fp = {
                111: np.identity(3, dtype=float)
            }
            self._Fc = {
                111: np.identity(3, dtype=float)
            }
            self._Fn = {
                111: np.identity(3, dtype=float)
            }
        elif gaussPts == 4:
            # displacement gradients located at the Gauss points of tetrahedron
            self._G0 = {
                111: np.zeros((3, 3), dtype=float),
                112: np.zeros((3, 3), dtype=float),
                113: np.zeros((3, 3), dtype=float),
                114: np.zeros((3, 3), dtype=float),
                121: np.zeros((3, 3), dtype=float),
                122: np.zeros((3, 3), dtype=float),
                123: np.zeros((3, 3), dtype=float),
                124: np.zeros((3, 3), dtype=float),
                131: np.zeros((3, 3), dtype=float),
                132: np.zeros((3, 3), dtype=float),
                133: np.zeros((3, 3), dtype=float),
                134: np.zeros((3, 3), dtype=float),
                141: np.zeros((3, 3), dtype=float),
                142: np.zeros((3, 3), dtype=float),
                143: np.zeros((3, 3), dtype=float),
                144: np.zeros((3, 3), dtype=float),
                211: np.zeros((3, 3), dtype=float),
                212: np.zeros((3, 3), dtype=float),
                213: np.zeros((3, 3), dtype=float),
                214: np.zeros((3, 3), dtype=float),
                221: np.zeros((3, 3), dtype=float),
                222: np.zeros((3, 3), dtype=float),
                223: np.zeros((3, 3), dtype=float),
                224: np.zeros((3, 3), dtype=float),
                231: np.zeros((3, 3), dtype=float),
                232: np.zeros((3, 3), dtype=float),
                233: np.zeros((3, 3), dtype=float),
                234: np.zeros((3, 3), dtype=float),
                241: np.zeros((3, 3), dtype=float),
                242: np.zeros((3, 3), dtype=float),
                243: np.zeros((3, 3), dtype=float),
                244: np.zeros((3, 3), dtype=float),
                311: np.zeros((3, 3), dtype=float),
                312: np.zeros((3, 3), dtype=float),
                313: np.zeros((3, 3), dtype=float),
                314: np.zeros((3, 3), dtype=float),
                321: np.zeros((3, 3), dtype=float),
                322: np.zeros((3, 3), dtype=float),
                323: np.zeros((3, 3), dtype=float),
                324: np.zeros((3, 3), dtype=float),
                331: np.zeros((3, 3), dtype=float),
                332: np.zeros((3, 3), dtype=float),
                333: np.zeros((3, 3), dtype=float),
                334: np.zeros((3, 3), dtype=float),
                341: np.zeros((3, 3), dtype=float),
                342: np.zeros((3, 3), dtype=float),
                343: np.zeros((3, 3), dtype=float),
                344: np.zeros((3, 3), dtype=float),
                411: np.zeros((3, 3), dtype=float),
                412: np.zeros((3, 3), dtype=float),
                413: np.zeros((3, 3), dtype=float),
                414: np.zeros((3, 3), dtype=float),
                421: np.zeros((3, 3), dtype=float),
                422: np.zeros((3, 3), dtype=float),
                423: np.zeros((3, 3), dtype=float),
                424: np.zeros((3, 3), dtype=float),
                431: np.zeros((3, 3), dtype=float),
                432: np.zeros((3, 3), dtype=float),
                433: np.zeros((3, 3), dtype=float),
                434: np.zeros((3, 3), dtype=float),
                441: np.zeros((3, 3), dtype=float),
                442: np.zeros((3, 3), dtype=float),
                443: np.zeros((3, 3), dtype=float),
                444: np.zeros((3, 3), dtype=float)
            }
            self._Gp = {
                111: np.zeros((3, 3), dtype=float),
                112: np.zeros((3, 3), dtype=float),
                113: np.zeros((3, 3), dtype=float),
                114: np.zeros((3, 3), dtype=float),
                121: np.zeros((3, 3), dtype=float),
                122: np.zeros((3, 3), dtype=float),
                123: np.zeros((3, 3), dtype=float),
                124: np.zeros((3, 3), dtype=float),
                131: np.zeros((3, 3), dtype=float),
                132: np.zeros((3, 3), dtype=float),
                133: np.zeros((3, 3), dtype=float),
                134: np.zeros((3, 3), dtype=float),
                141: np.zeros((3, 3), dtype=float),
                142: np.zeros((3, 3), dtype=float),
                143: np.zeros((3, 3), dtype=float),
                144: np.zeros((3, 3), dtype=float),
                211: np.zeros((3, 3), dtype=float),
                212: np.zeros((3, 3), dtype=float),
                213: np.zeros((3, 3), dtype=float),
                214: np.zeros((3, 3), dtype=float),
                221: np.zeros((3, 3), dtype=float),
                222: np.zeros((3, 3), dtype=float),
                223: np.zeros((3, 3), dtype=float),
                224: np.zeros((3, 3), dtype=float),
                231: np.zeros((3, 3), dtype=float),
                232: np.zeros((3, 3), dtype=float),
                233: np.zeros((3, 3), dtype=float),
                234: np.zeros((3, 3), dtype=float),
                241: np.zeros((3, 3), dtype=float),
                242: np.zeros((3, 3), dtype=float),
                243: np.zeros((3, 3), dtype=float),
                244: np.zeros((3, 3), dtype=float),
                311: np.zeros((3, 3), dtype=float),
                312: np.zeros((3, 3), dtype=float),
                313: np.zeros((3, 3), dtype=float),
                314: np.zeros((3, 3), dtype=float),
                321: np.zeros((3, 3), dtype=float),
                322: np.zeros((3, 3), dtype=float),
                323: np.zeros((3, 3), dtype=float),
                324: np.zeros((3, 3), dtype=float),
                331: np.zeros((3, 3), dtype=float),
                332: np.zeros((3, 3), dtype=float),
                333: np.zeros((3, 3), dtype=float),
                334: np.zeros((3, 3), dtype=float),
                341: np.zeros((3, 3), dtype=float),
                342: np.zeros((3, 3), dtype=float),
                343: np.zeros((3, 3), dtype=float),
                344: np.zeros((3, 3), dtype=float),
                411: np.zeros((3, 3), dtype=float),
                412: np.zeros((3, 3), dtype=float),
                413: np.zeros((3, 3), dtype=float),
                414: np.zeros((3, 3), dtype=float),
                421: np.zeros((3, 3), dtype=float),
                422: np.zeros((3, 3), dtype=float),
                423: np.zeros((3, 3), dtype=float),
                424: np.zeros((3, 3), dtype=float),
                431: np.zeros((3, 3), dtype=float),
                432: np.zeros((3, 3), dtype=float),
                433: np.zeros((3, 3), dtype=float),
                434: np.zeros((3, 3), dtype=float),
                441: np.zeros((3, 3), dtype=float),
                442: np.zeros((3, 3), dtype=float),
                443: np.zeros((3, 3), dtype=float),
                444: np.zeros((3, 3), dtype=float)
            }
            self._Gc = {
                111: np.zeros((3, 3), dtype=float),
                112: np.zeros((3, 3), dtype=float),
                113: np.zeros((3, 3), dtype=float),
                114: np.zeros((3, 3), dtype=float),
                121: np.zeros((3, 3), dtype=float),
                122: np.zeros((3, 3), dtype=float),
                123: np.zeros((3, 3), dtype=float),
                124: np.zeros((3, 3), dtype=float),
                131: np.zeros((3, 3), dtype=float),
                132: np.zeros((3, 3), dtype=float),
                133: np.zeros((3, 3), dtype=float),
                134: np.zeros((3, 3), dtype=float),
                141: np.zeros((3, 3), dtype=float),
                142: np.zeros((3, 3), dtype=float),
                143: np.zeros((3, 3), dtype=float),
                144: np.zeros((3, 3), dtype=float),
                211: np.zeros((3, 3), dtype=float),
                212: np.zeros((3, 3), dtype=float),
                213: np.zeros((3, 3), dtype=float),
                214: np.zeros((3, 3), dtype=float),
                221: np.zeros((3, 3), dtype=float),
                222: np.zeros((3, 3), dtype=float),
                223: np.zeros((3, 3), dtype=float),
                224: np.zeros((3, 3), dtype=float),
                231: np.zeros((3, 3), dtype=float),
                232: np.zeros((3, 3), dtype=float),
                233: np.zeros((3, 3), dtype=float),
                234: np.zeros((3, 3), dtype=float),
                241: np.zeros((3, 3), dtype=float),
                242: np.zeros((3, 3), dtype=float),
                243: np.zeros((3, 3), dtype=float),
                244: np.zeros((3, 3), dtype=float),
                311: np.zeros((3, 3), dtype=float),
                312: np.zeros((3, 3), dtype=float),
                313: np.zeros((3, 3), dtype=float),
                314: np.zeros((3, 3), dtype=float),
                321: np.zeros((3, 3), dtype=float),
                322: np.zeros((3, 3), dtype=float),
                323: np.zeros((3, 3), dtype=float),
                324: np.zeros((3, 3), dtype=float),
                331: np.zeros((3, 3), dtype=float),
                332: np.zeros((3, 3), dtype=float),
                333: np.zeros((3, 3), dtype=float),
                334: np.zeros((3, 3), dtype=float),
                341: np.zeros((3, 3), dtype=float),
                342: np.zeros((3, 3), dtype=float),
                343: np.zeros((3, 3), dtype=float),
                344: np.zeros((3, 3), dtype=float),
                411: np.zeros((3, 3), dtype=float),
                412: np.zeros((3, 3), dtype=float),
                413: np.zeros((3, 3), dtype=float),
                414: np.zeros((3, 3), dtype=float),
                421: np.zeros((3, 3), dtype=float),
                422: np.zeros((3, 3), dtype=float),
                423: np.zeros((3, 3), dtype=float),
                424: np.zeros((3, 3), dtype=float),
                431: np.zeros((3, 3), dtype=float),
                432: np.zeros((3, 3), dtype=float),
                433: np.zeros((3, 3), dtype=float),
                434: np.zeros((3, 3), dtype=float),
                441: np.zeros((3, 3), dtype=float),
                442: np.zeros((3, 3), dtype=float),
                443: np.zeros((3, 3), dtype=float),
                444: np.zeros((3, 3), dtype=float)
            }
            self._Gn = {
                111: np.zeros((3, 3), dtype=float),
                112: np.zeros((3, 3), dtype=float),
                113: np.zeros((3, 3), dtype=float),
                114: np.zeros((3, 3), dtype=float),
                121: np.zeros((3, 3), dtype=float),
                122: np.zeros((3, 3), dtype=float),
                123: np.zeros((3, 3), dtype=float),
                124: np.zeros((3, 3), dtype=float),
                131: np.zeros((3, 3), dtype=float),
                132: np.zeros((3, 3), dtype=float),
                133: np.zeros((3, 3), dtype=float),
                134: np.zeros((3, 3), dtype=float),
                141: np.zeros((3, 3), dtype=float),
                142: np.zeros((3, 3), dtype=float),
                143: np.zeros((3, 3), dtype=float),
                144: np.zeros((3, 3), dtype=float),
                211: np.zeros((3, 3), dtype=float),
                212: np.zeros((3, 3), dtype=float),
                213: np.zeros((3, 3), dtype=float),
                214: np.zeros((3, 3), dtype=float),
                221: np.zeros((3, 3), dtype=float),
                222: np.zeros((3, 3), dtype=float),
                223: np.zeros((3, 3), dtype=float),
                224: np.zeros((3, 3), dtype=float),
                231: np.zeros((3, 3), dtype=float),
                232: np.zeros((3, 3), dtype=float),
                233: np.zeros((3, 3), dtype=float),
                234: np.zeros((3, 3), dtype=float),
                241: np.zeros((3, 3), dtype=float),
                242: np.zeros((3, 3), dtype=float),
                243: np.zeros((3, 3), dtype=float),
                244: np.zeros((3, 3), dtype=float),
                311: np.zeros((3, 3), dtype=float),
                312: np.zeros((3, 3), dtype=float),
                313: np.zeros((3, 3), dtype=float),
                314: np.zeros((3, 3), dtype=float),
                321: np.zeros((3, 3), dtype=float),
                322: np.zeros((3, 3), dtype=float),
                323: np.zeros((3, 3), dtype=float),
                324: np.zeros((3, 3), dtype=float),
                331: np.zeros((3, 3), dtype=float),
                332: np.zeros((3, 3), dtype=float),
                333: np.zeros((3, 3), dtype=float),
                334: np.zeros((3, 3), dtype=float),
                341: np.zeros((3, 3), dtype=float),
                342: np.zeros((3, 3), dtype=float),
                343: np.zeros((3, 3), dtype=float),
                344: np.zeros((3, 3), dtype=float),
                411: np.zeros((3, 3), dtype=float),
                412: np.zeros((3, 3), dtype=float),
                413: np.zeros((3, 3), dtype=float),
                414: np.zeros((3, 3), dtype=float),
                421: np.zeros((3, 3), dtype=float),
                422: np.zeros((3, 3), dtype=float),
                423: np.zeros((3, 3), dtype=float),
                424: np.zeros((3, 3), dtype=float),
                431: np.zeros((3, 3), dtype=float),
                432: np.zeros((3, 3), dtype=float),
                433: np.zeros((3, 3), dtype=float),
                434: np.zeros((3, 3), dtype=float),
                441: np.zeros((3, 3), dtype=float),
                442: np.zeros((3, 3), dtype=float),
                443: np.zeros((3, 3), dtype=float),
                444: np.zeros((3, 3), dtype=float)
            }
            # deformation gradients located at the Gauss points of tetrahedron
            self._F0 = {
                111: np.identity(3, dtype=float),
                112: np.identity(3, dtype=float),
                113: np.identity(3, dtype=float),
                114: np.identity(3, dtype=float),
                121: np.identity(3, dtype=float),
                122: np.identity(3, dtype=float),
                123: np.identity(3, dtype=float),
                124: np.identity(3, dtype=float),
                131: np.identity(3, dtype=float),
                132: np.identity(3, dtype=float),
                133: np.identity(3, dtype=float),
                134: np.identity(3, dtype=float),
                141: np.identity(3, dtype=float),
                142: np.identity(3, dtype=float),
                143: np.identity(3, dtype=float),
                144: np.identity(3, dtype=float),
                211: np.identity(3, dtype=float),
                212: np.identity(3, dtype=float),
                213: np.identity(3, dtype=float),
                214: np.identity(3, dtype=float),
                221: np.identity(3, dtype=float),
                222: np.identity(3, dtype=float),
                223: np.identity(3, dtype=float),
                224: np.identity(3, dtype=float),
                231: np.identity(3, dtype=float),
                232: np.identity(3, dtype=float),
                233: np.identity(3, dtype=float),
                234: np.identity(3, dtype=float),
                241: np.identity(3, dtype=float),
                242: np.identity(3, dtype=float),
                243: np.identity(3, dtype=float),
                244: np.identity(3, dtype=float),
                311: np.identity(3, dtype=float),
                312: np.identity(3, dtype=float),
                313: np.identity(3, dtype=float),
                314: np.identity(3, dtype=float),
                321: np.identity(3, dtype=float),
                322: np.identity(3, dtype=float),
                323: np.identity(3, dtype=float),
                324: np.identity(3, dtype=float),
                331: np.identity(3, dtype=float),
                332: np.identity(3, dtype=float),
                333: np.identity(3, dtype=float),
                334: np.identity(3, dtype=float),
                341: np.identity(3, dtype=float),
                342: np.identity(3, dtype=float),
                343: np.identity(3, dtype=float),
                344: np.identity(3, dtype=float),
                411: np.identity(3, dtype=float),
                412: np.identity(3, dtype=float),
                413: np.identity(3, dtype=float),
                414: np.identity(3, dtype=float),
                421: np.identity(3, dtype=float),
                422: np.identity(3, dtype=float),
                423: np.identity(3, dtype=float),
                424: np.identity(3, dtype=float),
                431: np.identity(3, dtype=float),
                432: np.identity(3, dtype=float),
                433: np.identity(3, dtype=float),
                434: np.identity(3, dtype=float),
                441: np.identity(3, dtype=float),
                442: np.identity(3, dtype=float),
                443: np.identity(3, dtype=float),
                444: np.identity(3, dtype=float)
            }
            self._Fp = {
                111: np.identity(3, dtype=float),
                112: np.identity(3, dtype=float),
                113: np.identity(3, dtype=float),
                114: np.identity(3, dtype=float),
                121: np.identity(3, dtype=float),
                122: np.identity(3, dtype=float),
                123: np.identity(3, dtype=float),
                124: np.identity(3, dtype=float),
                131: np.identity(3, dtype=float),
                132: np.identity(3, dtype=float),
                133: np.identity(3, dtype=float),
                134: np.identity(3, dtype=float),
                141: np.identity(3, dtype=float),
                142: np.identity(3, dtype=float),
                143: np.identity(3, dtype=float),
                144: np.identity(3, dtype=float),
                211: np.identity(3, dtype=float),
                212: np.identity(3, dtype=float),
                213: np.identity(3, dtype=float),
                214: np.identity(3, dtype=float),
                221: np.identity(3, dtype=float),
                222: np.identity(3, dtype=float),
                223: np.identity(3, dtype=float),
                224: np.identity(3, dtype=float),
                231: np.identity(3, dtype=float),
                232: np.identity(3, dtype=float),
                233: np.identity(3, dtype=float),
                234: np.identity(3, dtype=float),
                241: np.identity(3, dtype=float),
                242: np.identity(3, dtype=float),
                243: np.identity(3, dtype=float),
                244: np.identity(3, dtype=float),
                311: np.identity(3, dtype=float),
                312: np.identity(3, dtype=float),
                313: np.identity(3, dtype=float),
                314: np.identity(3, dtype=float),
                321: np.identity(3, dtype=float),
                322: np.identity(3, dtype=float),
                323: np.identity(3, dtype=float),
                324: np.identity(3, dtype=float),
                331: np.identity(3, dtype=float),
                332: np.identity(3, dtype=float),
                333: np.identity(3, dtype=float),
                334: np.identity(3, dtype=float),
                341: np.identity(3, dtype=float),
                342: np.identity(3, dtype=float),
                343: np.identity(3, dtype=float),
                344: np.identity(3, dtype=float),
                411: np.identity(3, dtype=float),
                412: np.identity(3, dtype=float),
                413: np.identity(3, dtype=float),
                414: np.identity(3, dtype=float),
                421: np.identity(3, dtype=float),
                422: np.identity(3, dtype=float),
                423: np.identity(3, dtype=float),
                424: np.identity(3, dtype=float),
                431: np.identity(3, dtype=float),
                432: np.identity(3, dtype=float),
                433: np.identity(3, dtype=float),
                434: np.identity(3, dtype=float),
                441: np.identity(3, dtype=float),
                442: np.identity(3, dtype=float),
                443: np.identity(3, dtype=float),
                444: np.identity(3, dtype=float)
            }
            self._Fc = {
                111: np.identity(3, dtype=float),
                112: np.identity(3, dtype=float),
                113: np.identity(3, dtype=float),
                114: np.identity(3, dtype=float),
                121: np.identity(3, dtype=float),
                122: np.identity(3, dtype=float),
                123: np.identity(3, dtype=float),
                124: np.identity(3, dtype=float),
                131: np.identity(3, dtype=float),
                132: np.identity(3, dtype=float),
                133: np.identity(3, dtype=float),
                134: np.identity(3, dtype=float),
                141: np.identity(3, dtype=float),
                142: np.identity(3, dtype=float),
                143: np.identity(3, dtype=float),
                144: np.identity(3, dtype=float),
                211: np.identity(3, dtype=float),
                212: np.identity(3, dtype=float),
                213: np.identity(3, dtype=float),
                214: np.identity(3, dtype=float),
                221: np.identity(3, dtype=float),
                222: np.identity(3, dtype=float),
                223: np.identity(3, dtype=float),
                224: np.identity(3, dtype=float),
                231: np.identity(3, dtype=float),
                232: np.identity(3, dtype=float),
                233: np.identity(3, dtype=float),
                234: np.identity(3, dtype=float),
                241: np.identity(3, dtype=float),
                242: np.identity(3, dtype=float),
                243: np.identity(3, dtype=float),
                244: np.identity(3, dtype=float),
                311: np.identity(3, dtype=float),
                312: np.identity(3, dtype=float),
                313: np.identity(3, dtype=float),
                314: np.identity(3, dtype=float),
                321: np.identity(3, dtype=float),
                322: np.identity(3, dtype=float),
                323: np.identity(3, dtype=float),
                324: np.identity(3, dtype=float),
                331: np.identity(3, dtype=float),
                332: np.identity(3, dtype=float),
                333: np.identity(3, dtype=float),
                334: np.identity(3, dtype=float),
                341: np.identity(3, dtype=float),
                342: np.identity(3, dtype=float),
                343: np.identity(3, dtype=float),
                344: np.identity(3, dtype=float),
                411: np.identity(3, dtype=float),
                412: np.identity(3, dtype=float),
                413: np.identity(3, dtype=float),
                414: np.identity(3, dtype=float),
                421: np.identity(3, dtype=float),
                422: np.identity(3, dtype=float),
                423: np.identity(3, dtype=float),
                424: np.identity(3, dtype=float),
                431: np.identity(3, dtype=float),
                432: np.identity(3, dtype=float),
                433: np.identity(3, dtype=float),
                434: np.identity(3, dtype=float),
                441: np.identity(3, dtype=float),
                442: np.identity(3, dtype=float),
                443: np.identity(3, dtype=float),
                444: np.identity(3, dtype=float)
            }
            self._Fn = {
                111: np.identity(3, dtype=float),
                112: np.identity(3, dtype=float),
                113: np.identity(3, dtype=float),
                114: np.identity(3, dtype=float),
                121: np.identity(3, dtype=float),
                122: np.identity(3, dtype=float),
                123: np.identity(3, dtype=float),
                124: np.identity(3, dtype=float),
                131: np.identity(3, dtype=float),
                132: np.identity(3, dtype=float),
                133: np.identity(3, dtype=float),
                134: np.identity(3, dtype=float),
                141: np.identity(3, dtype=float),
                142: np.identity(3, dtype=float),
                143: np.identity(3, dtype=float),
                144: np.identity(3, dtype=float),
                211: np.identity(3, dtype=float),
                212: np.identity(3, dtype=float),
                213: np.identity(3, dtype=float),
                214: np.identity(3, dtype=float),
                221: np.identity(3, dtype=float),
                222: np.identity(3, dtype=float),
                223: np.identity(3, dtype=float),
                224: np.identity(3, dtype=float),
                231: np.identity(3, dtype=float),
                232: np.identity(3, dtype=float),
                233: np.identity(3, dtype=float),
                234: np.identity(3, dtype=float),
                241: np.identity(3, dtype=float),
                242: np.identity(3, dtype=float),
                243: np.identity(3, dtype=float),
                244: np.identity(3, dtype=float),
                311: np.identity(3, dtype=float),
                312: np.identity(3, dtype=float),
                313: np.identity(3, dtype=float),
                314: np.identity(3, dtype=float),
                321: np.identity(3, dtype=float),
                322: np.identity(3, dtype=float),
                323: np.identity(3, dtype=float),
                324: np.identity(3, dtype=float),
                331: np.identity(3, dtype=float),
                332: np.identity(3, dtype=float),
                333: np.identity(3, dtype=float),
                334: np.identity(3, dtype=float),
                341: np.identity(3, dtype=float),
                342: np.identity(3, dtype=float),
                343: np.identity(3, dtype=float),
                344: np.identity(3, dtype=float),
                411: np.identity(3, dtype=float),
                412: np.identity(3, dtype=float),
                413: np.identity(3, dtype=float),
                414: np.identity(3, dtype=float),
                421: np.identity(3, dtype=float),
                422: np.identity(3, dtype=float),
                423: np.identity(3, dtype=float),
                424: np.identity(3, dtype=float),
                431: np.identity(3, dtype=float),
                432: np.identity(3, dtype=float),
                433: np.identity(3, dtype=float),
                434: np.identity(3, dtype=float),
                441: np.identity(3, dtype=float),
                442: np.identity(3, dtype=float),
                443: np.identity(3, dtype=float),
                444: np.identity(3, dtype=float)
            }
        else:  # gaussPts = 5
            # displacement gradients located at the Gauss points of tetrahedron
            self._G0 = {
                111: np.zeros((3, 3), dtype=float),
                112: np.zeros((3, 3), dtype=float),
                113: np.zeros((3, 3), dtype=float),
                114: np.zeros((3, 3), dtype=float),
                115: np.zeros((3, 3), dtype=float),
                121: np.zeros((3, 3), dtype=float),
                122: np.zeros((3, 3), dtype=float),
                123: np.zeros((3, 3), dtype=float),
                124: np.zeros((3, 3), dtype=float),
                125: np.zeros((3, 3), dtype=float),
                131: np.zeros((3, 3), dtype=float),
                132: np.zeros((3, 3), dtype=float),
                133: np.zeros((3, 3), dtype=float),
                134: np.zeros((3, 3), dtype=float),
                135: np.zeros((3, 3), dtype=float),
                141: np.zeros((3, 3), dtype=float),
                142: np.zeros((3, 3), dtype=float),
                143: np.zeros((3, 3), dtype=float),
                144: np.zeros((3, 3), dtype=float),
                145: np.zeros((3, 3), dtype=float),
                151: np.zeros((3, 3), dtype=float),
                152: np.zeros((3, 3), dtype=float),
                153: np.zeros((3, 3), dtype=float),
                154: np.zeros((3, 3), dtype=float),
                155: np.zeros((3, 3), dtype=float),
                211: np.zeros((3, 3), dtype=float),
                212: np.zeros((3, 3), dtype=float),
                213: np.zeros((3, 3), dtype=float),
                214: np.zeros((3, 3), dtype=float),
                215: np.zeros((3, 3), dtype=float),
                221: np.zeros((3, 3), dtype=float),
                222: np.zeros((3, 3), dtype=float),
                223: np.zeros((3, 3), dtype=float),
                224: np.zeros((3, 3), dtype=float),
                225: np.zeros((3, 3), dtype=float),
                231: np.zeros((3, 3), dtype=float),
                232: np.zeros((3, 3), dtype=float),
                233: np.zeros((3, 3), dtype=float),
                234: np.zeros((3, 3), dtype=float),
                235: np.zeros((3, 3), dtype=float),
                241: np.zeros((3, 3), dtype=float),
                242: np.zeros((3, 3), dtype=float),
                243: np.zeros((3, 3), dtype=float),
                244: np.zeros((3, 3), dtype=float),
                245: np.zeros((3, 3), dtype=float),
                251: np.zeros((3, 3), dtype=float),
                252: np.zeros((3, 3), dtype=float),
                253: np.zeros((3, 3), dtype=float),
                254: np.zeros((3, 3), dtype=float),
                255: np.zeros((3, 3), dtype=float),
                311: np.zeros((3, 3), dtype=float),
                312: np.zeros((3, 3), dtype=float),
                313: np.zeros((3, 3), dtype=float),
                314: np.zeros((3, 3), dtype=float),
                315: np.zeros((3, 3), dtype=float),
                321: np.zeros((3, 3), dtype=float),
                322: np.zeros((3, 3), dtype=float),
                323: np.zeros((3, 3), dtype=float),
                324: np.zeros((3, 3), dtype=float),
                325: np.zeros((3, 3), dtype=float),
                331: np.zeros((3, 3), dtype=float),
                332: np.zeros((3, 3), dtype=float),
                333: np.zeros((3, 3), dtype=float),
                334: np.zeros((3, 3), dtype=float),
                335: np.zeros((3, 3), dtype=float),
                341: np.zeros((3, 3), dtype=float),
                342: np.zeros((3, 3), dtype=float),
                343: np.zeros((3, 3), dtype=float),
                344: np.zeros((3, 3), dtype=float),
                345: np.zeros((3, 3), dtype=float),
                351: np.zeros((3, 3), dtype=float),
                352: np.zeros((3, 3), dtype=float),
                353: np.zeros((3, 3), dtype=float),
                354: np.zeros((3, 3), dtype=float),
                355: np.zeros((3, 3), dtype=float),
                411: np.zeros((3, 3), dtype=float),
                412: np.zeros((3, 3), dtype=float),
                413: np.zeros((3, 3), dtype=float),
                414: np.zeros((3, 3), dtype=float),
                415: np.zeros((3, 3), dtype=float),
                421: np.zeros((3, 3), dtype=float),
                422: np.zeros((3, 3), dtype=float),
                423: np.zeros((3, 3), dtype=float),
                424: np.zeros((3, 3), dtype=float),
                425: np.zeros((3, 3), dtype=float),
                431: np.zeros((3, 3), dtype=float),
                432: np.zeros((3, 3), dtype=float),
                433: np.zeros((3, 3), dtype=float),
                434: np.zeros((3, 3), dtype=float),
                435: np.zeros((3, 3), dtype=float),
                441: np.zeros((3, 3), dtype=float),
                442: np.zeros((3, 3), dtype=float),
                443: np.zeros((3, 3), dtype=float),
                444: np.zeros((3, 3), dtype=float),
                445: np.zeros((3, 3), dtype=float),
                451: np.zeros((3, 3), dtype=float),
                452: np.zeros((3, 3), dtype=float),
                453: np.zeros((3, 3), dtype=float),
                454: np.zeros((3, 3), dtype=float),
                455: np.zeros((3, 3), dtype=float),
                511: np.zeros((3, 3), dtype=float),
                512: np.zeros((3, 3), dtype=float),
                513: np.zeros((3, 3), dtype=float),
                514: np.zeros((3, 3), dtype=float),
                515: np.zeros((3, 3), dtype=float),
                521: np.zeros((3, 3), dtype=float),
                522: np.zeros((3, 3), dtype=float),
                523: np.zeros((3, 3), dtype=float),
                524: np.zeros((3, 3), dtype=float),
                525: np.zeros((3, 3), dtype=float),
                531: np.zeros((3, 3), dtype=float),
                532: np.zeros((3, 3), dtype=float),
                533: np.zeros((3, 3), dtype=float),
                534: np.zeros((3, 3), dtype=float),
                535: np.zeros((3, 3), dtype=float),
                541: np.zeros((3, 3), dtype=float),
                542: np.zeros((3, 3), dtype=float),
                543: np.zeros((3, 3), dtype=float),
                544: np.zeros((3, 3), dtype=float),
                545: np.zeros((3, 3), dtype=float),
                551: np.zeros((3, 3), dtype=float),
                552: np.zeros((3, 3), dtype=float),
                553: np.zeros((3, 3), dtype=float),
                554: np.zeros((3, 3), dtype=float),
                555: np.zeros((3, 3), dtype=float)
            }
            self._Gp = {
                111: np.zeros((3, 3), dtype=float),
                112: np.zeros((3, 3), dtype=float),
                113: np.zeros((3, 3), dtype=float),
                114: np.zeros((3, 3), dtype=float),
                115: np.zeros((3, 3), dtype=float),
                121: np.zeros((3, 3), dtype=float),
                122: np.zeros((3, 3), dtype=float),
                123: np.zeros((3, 3), dtype=float),
                124: np.zeros((3, 3), dtype=float),
                125: np.zeros((3, 3), dtype=float),
                131: np.zeros((3, 3), dtype=float),
                132: np.zeros((3, 3), dtype=float),
                133: np.zeros((3, 3), dtype=float),
                134: np.zeros((3, 3), dtype=float),
                135: np.zeros((3, 3), dtype=float),
                141: np.zeros((3, 3), dtype=float),
                142: np.zeros((3, 3), dtype=float),
                143: np.zeros((3, 3), dtype=float),
                144: np.zeros((3, 3), dtype=float),
                145: np.zeros((3, 3), dtype=float),
                151: np.zeros((3, 3), dtype=float),
                152: np.zeros((3, 3), dtype=float),
                153: np.zeros((3, 3), dtype=float),
                154: np.zeros((3, 3), dtype=float),
                155: np.zeros((3, 3), dtype=float),
                211: np.zeros((3, 3), dtype=float),
                212: np.zeros((3, 3), dtype=float),
                213: np.zeros((3, 3), dtype=float),
                214: np.zeros((3, 3), dtype=float),
                215: np.zeros((3, 3), dtype=float),
                221: np.zeros((3, 3), dtype=float),
                222: np.zeros((3, 3), dtype=float),
                223: np.zeros((3, 3), dtype=float),
                224: np.zeros((3, 3), dtype=float),
                225: np.zeros((3, 3), dtype=float),
                231: np.zeros((3, 3), dtype=float),
                232: np.zeros((3, 3), dtype=float),
                233: np.zeros((3, 3), dtype=float),
                234: np.zeros((3, 3), dtype=float),
                235: np.zeros((3, 3), dtype=float),
                241: np.zeros((3, 3), dtype=float),
                242: np.zeros((3, 3), dtype=float),
                243: np.zeros((3, 3), dtype=float),
                244: np.zeros((3, 3), dtype=float),
                245: np.zeros((3, 3), dtype=float),
                251: np.zeros((3, 3), dtype=float),
                252: np.zeros((3, 3), dtype=float),
                253: np.zeros((3, 3), dtype=float),
                254: np.zeros((3, 3), dtype=float),
                255: np.zeros((3, 3), dtype=float),
                311: np.zeros((3, 3), dtype=float),
                312: np.zeros((3, 3), dtype=float),
                313: np.zeros((3, 3), dtype=float),
                314: np.zeros((3, 3), dtype=float),
                315: np.zeros((3, 3), dtype=float),
                321: np.zeros((3, 3), dtype=float),
                322: np.zeros((3, 3), dtype=float),
                323: np.zeros((3, 3), dtype=float),
                324: np.zeros((3, 3), dtype=float),
                325: np.zeros((3, 3), dtype=float),
                331: np.zeros((3, 3), dtype=float),
                332: np.zeros((3, 3), dtype=float),
                333: np.zeros((3, 3), dtype=float),
                334: np.zeros((3, 3), dtype=float),
                335: np.zeros((3, 3), dtype=float),
                341: np.zeros((3, 3), dtype=float),
                342: np.zeros((3, 3), dtype=float),
                343: np.zeros((3, 3), dtype=float),
                344: np.zeros((3, 3), dtype=float),
                345: np.zeros((3, 3), dtype=float),
                351: np.zeros((3, 3), dtype=float),
                352: np.zeros((3, 3), dtype=float),
                353: np.zeros((3, 3), dtype=float),
                354: np.zeros((3, 3), dtype=float),
                355: np.zeros((3, 3), dtype=float),
                411: np.zeros((3, 3), dtype=float),
                412: np.zeros((3, 3), dtype=float),
                413: np.zeros((3, 3), dtype=float),
                414: np.zeros((3, 3), dtype=float),
                415: np.zeros((3, 3), dtype=float),
                421: np.zeros((3, 3), dtype=float),
                422: np.zeros((3, 3), dtype=float),
                423: np.zeros((3, 3), dtype=float),
                424: np.zeros((3, 3), dtype=float),
                425: np.zeros((3, 3), dtype=float),
                431: np.zeros((3, 3), dtype=float),
                432: np.zeros((3, 3), dtype=float),
                433: np.zeros((3, 3), dtype=float),
                434: np.zeros((3, 3), dtype=float),
                435: np.zeros((3, 3), dtype=float),
                441: np.zeros((3, 3), dtype=float),
                442: np.zeros((3, 3), dtype=float),
                443: np.zeros((3, 3), dtype=float),
                444: np.zeros((3, 3), dtype=float),
                445: np.zeros((3, 3), dtype=float),
                451: np.zeros((3, 3), dtype=float),
                452: np.zeros((3, 3), dtype=float),
                453: np.zeros((3, 3), dtype=float),
                454: np.zeros((3, 3), dtype=float),
                455: np.zeros((3, 3), dtype=float),
                511: np.zeros((3, 3), dtype=float),
                512: np.zeros((3, 3), dtype=float),
                513: np.zeros((3, 3), dtype=float),
                514: np.zeros((3, 3), dtype=float),
                515: np.zeros((3, 3), dtype=float),
                521: np.zeros((3, 3), dtype=float),
                522: np.zeros((3, 3), dtype=float),
                523: np.zeros((3, 3), dtype=float),
                524: np.zeros((3, 3), dtype=float),
                525: np.zeros((3, 3), dtype=float),
                531: np.zeros((3, 3), dtype=float),
                532: np.zeros((3, 3), dtype=float),
                533: np.zeros((3, 3), dtype=float),
                534: np.zeros((3, 3), dtype=float),
                535: np.zeros((3, 3), dtype=float),
                541: np.zeros((3, 3), dtype=float),
                542: np.zeros((3, 3), dtype=float),
                543: np.zeros((3, 3), dtype=float),
                544: np.zeros((3, 3), dtype=float),
                545: np.zeros((3, 3), dtype=float),
                551: np.zeros((3, 3), dtype=float),
                552: np.zeros((3, 3), dtype=float),
                553: np.zeros((3, 3), dtype=float),
                554: np.zeros((3, 3), dtype=float),
                555: np.zeros((3, 3), dtype=float)
            }
            self._Gc = {
                111: np.zeros((3, 3), dtype=float),
                112: np.zeros((3, 3), dtype=float),
                113: np.zeros((3, 3), dtype=float),
                114: np.zeros((3, 3), dtype=float),
                115: np.zeros((3, 3), dtype=float),
                121: np.zeros((3, 3), dtype=float),
                122: np.zeros((3, 3), dtype=float),
                123: np.zeros((3, 3), dtype=float),
                124: np.zeros((3, 3), dtype=float),
                125: np.zeros((3, 3), dtype=float),
                131: np.zeros((3, 3), dtype=float),
                132: np.zeros((3, 3), dtype=float),
                133: np.zeros((3, 3), dtype=float),
                134: np.zeros((3, 3), dtype=float),
                135: np.zeros((3, 3), dtype=float),
                141: np.zeros((3, 3), dtype=float),
                142: np.zeros((3, 3), dtype=float),
                143: np.zeros((3, 3), dtype=float),
                144: np.zeros((3, 3), dtype=float),
                145: np.zeros((3, 3), dtype=float),
                151: np.zeros((3, 3), dtype=float),
                152: np.zeros((3, 3), dtype=float),
                153: np.zeros((3, 3), dtype=float),
                154: np.zeros((3, 3), dtype=float),
                155: np.zeros((3, 3), dtype=float),
                211: np.zeros((3, 3), dtype=float),
                212: np.zeros((3, 3), dtype=float),
                213: np.zeros((3, 3), dtype=float),
                214: np.zeros((3, 3), dtype=float),
                215: np.zeros((3, 3), dtype=float),
                221: np.zeros((3, 3), dtype=float),
                222: np.zeros((3, 3), dtype=float),
                223: np.zeros((3, 3), dtype=float),
                224: np.zeros((3, 3), dtype=float),
                225: np.zeros((3, 3), dtype=float),
                231: np.zeros((3, 3), dtype=float),
                232: np.zeros((3, 3), dtype=float),
                233: np.zeros((3, 3), dtype=float),
                234: np.zeros((3, 3), dtype=float),
                235: np.zeros((3, 3), dtype=float),
                241: np.zeros((3, 3), dtype=float),
                242: np.zeros((3, 3), dtype=float),
                243: np.zeros((3, 3), dtype=float),
                244: np.zeros((3, 3), dtype=float),
                245: np.zeros((3, 3), dtype=float),
                251: np.zeros((3, 3), dtype=float),
                252: np.zeros((3, 3), dtype=float),
                253: np.zeros((3, 3), dtype=float),
                254: np.zeros((3, 3), dtype=float),
                255: np.zeros((3, 3), dtype=float),
                311: np.zeros((3, 3), dtype=float),
                312: np.zeros((3, 3), dtype=float),
                313: np.zeros((3, 3), dtype=float),
                314: np.zeros((3, 3), dtype=float),
                315: np.zeros((3, 3), dtype=float),
                321: np.zeros((3, 3), dtype=float),
                322: np.zeros((3, 3), dtype=float),
                323: np.zeros((3, 3), dtype=float),
                324: np.zeros((3, 3), dtype=float),
                325: np.zeros((3, 3), dtype=float),
                331: np.zeros((3, 3), dtype=float),
                332: np.zeros((3, 3), dtype=float),
                333: np.zeros((3, 3), dtype=float),
                334: np.zeros((3, 3), dtype=float),
                335: np.zeros((3, 3), dtype=float),
                341: np.zeros((3, 3), dtype=float),
                342: np.zeros((3, 3), dtype=float),
                343: np.zeros((3, 3), dtype=float),
                344: np.zeros((3, 3), dtype=float),
                345: np.zeros((3, 3), dtype=float),
                351: np.zeros((3, 3), dtype=float),
                352: np.zeros((3, 3), dtype=float),
                353: np.zeros((3, 3), dtype=float),
                354: np.zeros((3, 3), dtype=float),
                355: np.zeros((3, 3), dtype=float),
                411: np.zeros((3, 3), dtype=float),
                412: np.zeros((3, 3), dtype=float),
                413: np.zeros((3, 3), dtype=float),
                414: np.zeros((3, 3), dtype=float),
                415: np.zeros((3, 3), dtype=float),
                421: np.zeros((3, 3), dtype=float),
                422: np.zeros((3, 3), dtype=float),
                423: np.zeros((3, 3), dtype=float),
                424: np.zeros((3, 3), dtype=float),
                425: np.zeros((3, 3), dtype=float),
                431: np.zeros((3, 3), dtype=float),
                432: np.zeros((3, 3), dtype=float),
                433: np.zeros((3, 3), dtype=float),
                434: np.zeros((3, 3), dtype=float),
                435: np.zeros((3, 3), dtype=float),
                441: np.zeros((3, 3), dtype=float),
                442: np.zeros((3, 3), dtype=float),
                443: np.zeros((3, 3), dtype=float),
                444: np.zeros((3, 3), dtype=float),
                445: np.zeros((3, 3), dtype=float),
                451: np.zeros((3, 3), dtype=float),
                452: np.zeros((3, 3), dtype=float),
                453: np.zeros((3, 3), dtype=float),
                454: np.zeros((3, 3), dtype=float),
                455: np.zeros((3, 3), dtype=float),
                511: np.zeros((3, 3), dtype=float),
                512: np.zeros((3, 3), dtype=float),
                513: np.zeros((3, 3), dtype=float),
                514: np.zeros((3, 3), dtype=float),
                515: np.zeros((3, 3), dtype=float),
                521: np.zeros((3, 3), dtype=float),
                522: np.zeros((3, 3), dtype=float),
                523: np.zeros((3, 3), dtype=float),
                524: np.zeros((3, 3), dtype=float),
                525: np.zeros((3, 3), dtype=float),
                531: np.zeros((3, 3), dtype=float),
                532: np.zeros((3, 3), dtype=float),
                533: np.zeros((3, 3), dtype=float),
                534: np.zeros((3, 3), dtype=float),
                535: np.zeros((3, 3), dtype=float),
                541: np.zeros((3, 3), dtype=float),
                542: np.zeros((3, 3), dtype=float),
                543: np.zeros((3, 3), dtype=float),
                544: np.zeros((3, 3), dtype=float),
                545: np.zeros((3, 3), dtype=float),
                551: np.zeros((3, 3), dtype=float),
                552: np.zeros((3, 3), dtype=float),
                553: np.zeros((3, 3), dtype=float),
                554: np.zeros((3, 3), dtype=float),
                555: np.zeros((3, 3), dtype=float)
            }
            self._Gn = {
                111: np.zeros((3, 3), dtype=float),
                112: np.zeros((3, 3), dtype=float),
                113: np.zeros((3, 3), dtype=float),
                114: np.zeros((3, 3), dtype=float),
                115: np.zeros((3, 3), dtype=float),
                121: np.zeros((3, 3), dtype=float),
                122: np.zeros((3, 3), dtype=float),
                123: np.zeros((3, 3), dtype=float),
                124: np.zeros((3, 3), dtype=float),
                125: np.zeros((3, 3), dtype=float),
                131: np.zeros((3, 3), dtype=float),
                132: np.zeros((3, 3), dtype=float),
                133: np.zeros((3, 3), dtype=float),
                134: np.zeros((3, 3), dtype=float),
                135: np.zeros((3, 3), dtype=float),
                141: np.zeros((3, 3), dtype=float),
                142: np.zeros((3, 3), dtype=float),
                143: np.zeros((3, 3), dtype=float),
                144: np.zeros((3, 3), dtype=float),
                145: np.zeros((3, 3), dtype=float),
                151: np.zeros((3, 3), dtype=float),
                152: np.zeros((3, 3), dtype=float),
                153: np.zeros((3, 3), dtype=float),
                154: np.zeros((3, 3), dtype=float),
                155: np.zeros((3, 3), dtype=float),
                211: np.zeros((3, 3), dtype=float),
                212: np.zeros((3, 3), dtype=float),
                213: np.zeros((3, 3), dtype=float),
                214: np.zeros((3, 3), dtype=float),
                215: np.zeros((3, 3), dtype=float),
                221: np.zeros((3, 3), dtype=float),
                222: np.zeros((3, 3), dtype=float),
                223: np.zeros((3, 3), dtype=float),
                224: np.zeros((3, 3), dtype=float),
                225: np.zeros((3, 3), dtype=float),
                231: np.zeros((3, 3), dtype=float),
                232: np.zeros((3, 3), dtype=float),
                233: np.zeros((3, 3), dtype=float),
                234: np.zeros((3, 3), dtype=float),
                235: np.zeros((3, 3), dtype=float),
                241: np.zeros((3, 3), dtype=float),
                242: np.zeros((3, 3), dtype=float),
                243: np.zeros((3, 3), dtype=float),
                244: np.zeros((3, 3), dtype=float),
                245: np.zeros((3, 3), dtype=float),
                251: np.zeros((3, 3), dtype=float),
                252: np.zeros((3, 3), dtype=float),
                253: np.zeros((3, 3), dtype=float),
                254: np.zeros((3, 3), dtype=float),
                255: np.zeros((3, 3), dtype=float),
                311: np.zeros((3, 3), dtype=float),
                312: np.zeros((3, 3), dtype=float),
                313: np.zeros((3, 3), dtype=float),
                314: np.zeros((3, 3), dtype=float),
                315: np.zeros((3, 3), dtype=float),
                321: np.zeros((3, 3), dtype=float),
                322: np.zeros((3, 3), dtype=float),
                323: np.zeros((3, 3), dtype=float),
                324: np.zeros((3, 3), dtype=float),
                325: np.zeros((3, 3), dtype=float),
                331: np.zeros((3, 3), dtype=float),
                332: np.zeros((3, 3), dtype=float),
                333: np.zeros((3, 3), dtype=float),
                334: np.zeros((3, 3), dtype=float),
                335: np.zeros((3, 3), dtype=float),
                341: np.zeros((3, 3), dtype=float),
                342: np.zeros((3, 3), dtype=float),
                343: np.zeros((3, 3), dtype=float),
                344: np.zeros((3, 3), dtype=float),
                345: np.zeros((3, 3), dtype=float),
                351: np.zeros((3, 3), dtype=float),
                352: np.zeros((3, 3), dtype=float),
                353: np.zeros((3, 3), dtype=float),
                354: np.zeros((3, 3), dtype=float),
                355: np.zeros((3, 3), dtype=float),
                411: np.zeros((3, 3), dtype=float),
                412: np.zeros((3, 3), dtype=float),
                413: np.zeros((3, 3), dtype=float),
                414: np.zeros((3, 3), dtype=float),
                415: np.zeros((3, 3), dtype=float),
                421: np.zeros((3, 3), dtype=float),
                422: np.zeros((3, 3), dtype=float),
                423: np.zeros((3, 3), dtype=float),
                424: np.zeros((3, 3), dtype=float),
                425: np.zeros((3, 3), dtype=float),
                431: np.zeros((3, 3), dtype=float),
                432: np.zeros((3, 3), dtype=float),
                433: np.zeros((3, 3), dtype=float),
                434: np.zeros((3, 3), dtype=float),
                435: np.zeros((3, 3), dtype=float),
                441: np.zeros((3, 3), dtype=float),
                442: np.zeros((3, 3), dtype=float),
                443: np.zeros((3, 3), dtype=float),
                444: np.zeros((3, 3), dtype=float),
                445: np.zeros((3, 3), dtype=float),
                451: np.zeros((3, 3), dtype=float),
                452: np.zeros((3, 3), dtype=float),
                453: np.zeros((3, 3), dtype=float),
                454: np.zeros((3, 3), dtype=float),
                455: np.zeros((3, 3), dtype=float),
                511: np.zeros((3, 3), dtype=float),
                512: np.zeros((3, 3), dtype=float),
                513: np.zeros((3, 3), dtype=float),
                514: np.zeros((3, 3), dtype=float),
                515: np.zeros((3, 3), dtype=float),
                521: np.zeros((3, 3), dtype=float),
                522: np.zeros((3, 3), dtype=float),
                523: np.zeros((3, 3), dtype=float),
                524: np.zeros((3, 3), dtype=float),
                525: np.zeros((3, 3), dtype=float),
                531: np.zeros((3, 3), dtype=float),
                532: np.zeros((3, 3), dtype=float),
                533: np.zeros((3, 3), dtype=float),
                534: np.zeros((3, 3), dtype=float),
                535: np.zeros((3, 3), dtype=float),
                541: np.zeros((3, 3), dtype=float),
                542: np.zeros((3, 3), dtype=float),
                543: np.zeros((3, 3), dtype=float),
                544: np.zeros((3, 3), dtype=float),
                545: np.zeros((3, 3), dtype=float),
                551: np.zeros((3, 3), dtype=float),
                552: np.zeros((3, 3), dtype=float),
                553: np.zeros((3, 3), dtype=float),
                554: np.zeros((3, 3), dtype=float),
                555: np.zeros((3, 3), dtype=float)
            }

            # deformation gradients located at the Gauss points of pentagon
            self._F0 = {
                111: np.identity(3, dtype=float),
                112: np.identity(3, dtype=float),
                113: np.identity(3, dtype=float),
                114: np.identity(3, dtype=float),
                115: np.identity(3, dtype=float),
                121: np.identity(3, dtype=float),
                122: np.identity(3, dtype=float),
                123: np.identity(3, dtype=float),
                124: np.identity(3, dtype=float),
                125: np.identity(3, dtype=float),
                131: np.identity(3, dtype=float),
                132: np.identity(3, dtype=float),
                133: np.identity(3, dtype=float),
                134: np.identity(3, dtype=float),
                135: np.identity(3, dtype=float),
                141: np.identity(3, dtype=float),
                142: np.identity(3, dtype=float),
                143: np.identity(3, dtype=float),
                144: np.identity(3, dtype=float),
                145: np.identity(3, dtype=float),
                151: np.identity(3, dtype=float),
                152: np.identity(3, dtype=float),
                153: np.identity(3, dtype=float),
                154: np.identity(3, dtype=float),
                155: np.identity(3, dtype=float),
                211: np.identity(3, dtype=float),
                212: np.identity(3, dtype=float),
                213: np.identity(3, dtype=float),
                214: np.identity(3, dtype=float),
                215: np.identity(3, dtype=float),
                221: np.identity(3, dtype=float),
                222: np.identity(3, dtype=float),
                223: np.identity(3, dtype=float),
                224: np.identity(3, dtype=float),
                225: np.identity(3, dtype=float),
                231: np.identity(3, dtype=float),
                232: np.identity(3, dtype=float),
                233: np.identity(3, dtype=float),
                234: np.identity(3, dtype=float),
                235: np.identity(3, dtype=float),
                241: np.identity(3, dtype=float),
                242: np.identity(3, dtype=float),
                243: np.identity(3, dtype=float),
                244: np.identity(3, dtype=float),
                245: np.identity(3, dtype=float),
                251: np.identity(3, dtype=float),
                252: np.identity(3, dtype=float),
                253: np.identity(3, dtype=float),
                254: np.identity(3, dtype=float),
                255: np.identity(3, dtype=float),
                311: np.identity(3, dtype=float),
                312: np.identity(3, dtype=float),
                313: np.identity(3, dtype=float),
                314: np.identity(3, dtype=float),
                315: np.identity(3, dtype=float),
                321: np.identity(3, dtype=float),
                322: np.identity(3, dtype=float),
                323: np.identity(3, dtype=float),
                324: np.identity(3, dtype=float),
                325: np.identity(3, dtype=float),
                331: np.identity(3, dtype=float),
                332: np.identity(3, dtype=float),
                333: np.identity(3, dtype=float),
                334: np.identity(3, dtype=float),
                335: np.identity(3, dtype=float),
                341: np.identity(3, dtype=float),
                342: np.identity(3, dtype=float),
                343: np.identity(3, dtype=float),
                344: np.identity(3, dtype=float),
                345: np.identity(3, dtype=float),
                351: np.identity(3, dtype=float),
                352: np.identity(3, dtype=float),
                353: np.identity(3, dtype=float),
                354: np.identity(3, dtype=float),
                355: np.identity(3, dtype=float),
                411: np.identity(3, dtype=float),
                412: np.identity(3, dtype=float),
                413: np.identity(3, dtype=float),
                414: np.identity(3, dtype=float),
                415: np.identity(3, dtype=float),
                421: np.identity(3, dtype=float),
                422: np.identity(3, dtype=float),
                423: np.identity(3, dtype=float),
                424: np.identity(3, dtype=float),
                425: np.identity(3, dtype=float),
                431: np.identity(3, dtype=float),
                432: np.identity(3, dtype=float),
                433: np.identity(3, dtype=float),
                434: np.identity(3, dtype=float),
                435: np.identity(3, dtype=float),
                441: np.identity(3, dtype=float),
                442: np.identity(3, dtype=float),
                443: np.identity(3, dtype=float),
                444: np.identity(3, dtype=float),
                445: np.identity(3, dtype=float),
                451: np.identity(3, dtype=float),
                452: np.identity(3, dtype=float),
                453: np.identity(3, dtype=float),
                454: np.identity(3, dtype=float),
                455: np.identity(3, dtype=float),
                511: np.identity(3, dtype=float),
                512: np.identity(3, dtype=float),
                513: np.identity(3, dtype=float),
                514: np.identity(3, dtype=float),
                515: np.identity(3, dtype=float),
                521: np.identity(3, dtype=float),
                522: np.identity(3, dtype=float),
                523: np.identity(3, dtype=float),
                524: np.identity(3, dtype=float),
                525: np.identity(3, dtype=float),
                531: np.identity(3, dtype=float),
                532: np.identity(3, dtype=float),
                533: np.identity(3, dtype=float),
                534: np.identity(3, dtype=float),
                535: np.identity(3, dtype=float),
                541: np.identity(3, dtype=float),
                542: np.identity(3, dtype=float),
                543: np.identity(3, dtype=float),
                544: np.identity(3, dtype=float),
                545: np.identity(3, dtype=float),
                551: np.identity(3, dtype=float),
                552: np.identity(3, dtype=float),
                553: np.identity(3, dtype=float),
                554: np.identity(3, dtype=float),
                555: np.identity(3, dtype=float)
            }
            self._Fp = {
                111: np.identity(3, dtype=float),
                112: np.identity(3, dtype=float),
                113: np.identity(3, dtype=float),
                114: np.identity(3, dtype=float),
                115: np.identity(3, dtype=float),
                121: np.identity(3, dtype=float),
                122: np.identity(3, dtype=float),
                123: np.identity(3, dtype=float),
                124: np.identity(3, dtype=float),
                125: np.identity(3, dtype=float),
                131: np.identity(3, dtype=float),
                132: np.identity(3, dtype=float),
                133: np.identity(3, dtype=float),
                134: np.identity(3, dtype=float),
                135: np.identity(3, dtype=float),
                141: np.identity(3, dtype=float),
                142: np.identity(3, dtype=float),
                143: np.identity(3, dtype=float),
                144: np.identity(3, dtype=float),
                145: np.identity(3, dtype=float),
                151: np.identity(3, dtype=float),
                152: np.identity(3, dtype=float),
                153: np.identity(3, dtype=float),
                154: np.identity(3, dtype=float),
                155: np.identity(3, dtype=float),
                211: np.identity(3, dtype=float),
                212: np.identity(3, dtype=float),
                213: np.identity(3, dtype=float),
                214: np.identity(3, dtype=float),
                215: np.identity(3, dtype=float),
                221: np.identity(3, dtype=float),
                222: np.identity(3, dtype=float),
                223: np.identity(3, dtype=float),
                224: np.identity(3, dtype=float),
                225: np.identity(3, dtype=float),
                231: np.identity(3, dtype=float),
                232: np.identity(3, dtype=float),
                233: np.identity(3, dtype=float),
                234: np.identity(3, dtype=float),
                235: np.identity(3, dtype=float),
                241: np.identity(3, dtype=float),
                242: np.identity(3, dtype=float),
                243: np.identity(3, dtype=float),
                244: np.identity(3, dtype=float),
                245: np.identity(3, dtype=float),
                251: np.identity(3, dtype=float),
                252: np.identity(3, dtype=float),
                253: np.identity(3, dtype=float),
                254: np.identity(3, dtype=float),
                255: np.identity(3, dtype=float),
                311: np.identity(3, dtype=float),
                312: np.identity(3, dtype=float),
                313: np.identity(3, dtype=float),
                314: np.identity(3, dtype=float),
                315: np.identity(3, dtype=float),
                321: np.identity(3, dtype=float),
                322: np.identity(3, dtype=float),
                323: np.identity(3, dtype=float),
                324: np.identity(3, dtype=float),
                325: np.identity(3, dtype=float),
                331: np.identity(3, dtype=float),
                332: np.identity(3, dtype=float),
                333: np.identity(3, dtype=float),
                334: np.identity(3, dtype=float),
                335: np.identity(3, dtype=float),
                341: np.identity(3, dtype=float),
                342: np.identity(3, dtype=float),
                343: np.identity(3, dtype=float),
                344: np.identity(3, dtype=float),
                345: np.identity(3, dtype=float),
                351: np.identity(3, dtype=float),
                352: np.identity(3, dtype=float),
                353: np.identity(3, dtype=float),
                354: np.identity(3, dtype=float),
                355: np.identity(3, dtype=float),
                411: np.identity(3, dtype=float),
                412: np.identity(3, dtype=float),
                413: np.identity(3, dtype=float),
                414: np.identity(3, dtype=float),
                415: np.identity(3, dtype=float),
                421: np.identity(3, dtype=float),
                422: np.identity(3, dtype=float),
                423: np.identity(3, dtype=float),
                424: np.identity(3, dtype=float),
                425: np.identity(3, dtype=float),
                431: np.identity(3, dtype=float),
                432: np.identity(3, dtype=float),
                433: np.identity(3, dtype=float),
                434: np.identity(3, dtype=float),
                435: np.identity(3, dtype=float),
                441: np.identity(3, dtype=float),
                442: np.identity(3, dtype=float),
                443: np.identity(3, dtype=float),
                444: np.identity(3, dtype=float),
                445: np.identity(3, dtype=float),
                451: np.identity(3, dtype=float),
                452: np.identity(3, dtype=float),
                453: np.identity(3, dtype=float),
                454: np.identity(3, dtype=float),
                455: np.identity(3, dtype=float),
                511: np.identity(3, dtype=float),
                512: np.identity(3, dtype=float),
                513: np.identity(3, dtype=float),
                514: np.identity(3, dtype=float),
                515: np.identity(3, dtype=float),
                521: np.identity(3, dtype=float),
                522: np.identity(3, dtype=float),
                523: np.identity(3, dtype=float),
                524: np.identity(3, dtype=float),
                525: np.identity(3, dtype=float),
                531: np.identity(3, dtype=float),
                532: np.identity(3, dtype=float),
                533: np.identity(3, dtype=float),
                534: np.identity(3, dtype=float),
                535: np.identity(3, dtype=float),
                541: np.identity(3, dtype=float),
                542: np.identity(3, dtype=float),
                543: np.identity(3, dtype=float),
                544: np.identity(3, dtype=float),
                545: np.identity(3, dtype=float),
                551: np.identity(3, dtype=float),
                552: np.identity(3, dtype=float),
                553: np.identity(3, dtype=float),
                554: np.identity(3, dtype=float),
                555: np.identity(3, dtype=float)
            }
            self._Fc = {
                111: np.identity(3, dtype=float),
                112: np.identity(3, dtype=float),
                113: np.identity(3, dtype=float),
                114: np.identity(3, dtype=float),
                115: np.identity(3, dtype=float),
                121: np.identity(3, dtype=float),
                122: np.identity(3, dtype=float),
                123: np.identity(3, dtype=float),
                124: np.identity(3, dtype=float),
                125: np.identity(3, dtype=float),
                131: np.identity(3, dtype=float),
                132: np.identity(3, dtype=float),
                133: np.identity(3, dtype=float),
                134: np.identity(3, dtype=float),
                135: np.identity(3, dtype=float),
                141: np.identity(3, dtype=float),
                142: np.identity(3, dtype=float),
                143: np.identity(3, dtype=float),
                144: np.identity(3, dtype=float),
                145: np.identity(3, dtype=float),
                151: np.identity(3, dtype=float),
                152: np.identity(3, dtype=float),
                153: np.identity(3, dtype=float),
                154: np.identity(3, dtype=float),
                155: np.identity(3, dtype=float),
                211: np.identity(3, dtype=float),
                212: np.identity(3, dtype=float),
                213: np.identity(3, dtype=float),
                214: np.identity(3, dtype=float),
                215: np.identity(3, dtype=float),
                221: np.identity(3, dtype=float),
                222: np.identity(3, dtype=float),
                223: np.identity(3, dtype=float),
                224: np.identity(3, dtype=float),
                225: np.identity(3, dtype=float),
                231: np.identity(3, dtype=float),
                232: np.identity(3, dtype=float),
                233: np.identity(3, dtype=float),
                234: np.identity(3, dtype=float),
                235: np.identity(3, dtype=float),
                241: np.identity(3, dtype=float),
                242: np.identity(3, dtype=float),
                243: np.identity(3, dtype=float),
                244: np.identity(3, dtype=float),
                245: np.identity(3, dtype=float),
                251: np.identity(3, dtype=float),
                252: np.identity(3, dtype=float),
                253: np.identity(3, dtype=float),
                254: np.identity(3, dtype=float),
                255: np.identity(3, dtype=float),
                311: np.identity(3, dtype=float),
                312: np.identity(3, dtype=float),
                313: np.identity(3, dtype=float),
                314: np.identity(3, dtype=float),
                315: np.identity(3, dtype=float),
                321: np.identity(3, dtype=float),
                322: np.identity(3, dtype=float),
                323: np.identity(3, dtype=float),
                324: np.identity(3, dtype=float),
                325: np.identity(3, dtype=float),
                331: np.identity(3, dtype=float),
                332: np.identity(3, dtype=float),
                333: np.identity(3, dtype=float),
                334: np.identity(3, dtype=float),
                335: np.identity(3, dtype=float),
                341: np.identity(3, dtype=float),
                342: np.identity(3, dtype=float),
                343: np.identity(3, dtype=float),
                344: np.identity(3, dtype=float),
                345: np.identity(3, dtype=float),
                351: np.identity(3, dtype=float),
                352: np.identity(3, dtype=float),
                353: np.identity(3, dtype=float),
                354: np.identity(3, dtype=float),
                355: np.identity(3, dtype=float),
                411: np.identity(3, dtype=float),
                412: np.identity(3, dtype=float),
                413: np.identity(3, dtype=float),
                414: np.identity(3, dtype=float),
                415: np.identity(3, dtype=float),
                421: np.identity(3, dtype=float),
                422: np.identity(3, dtype=float),
                423: np.identity(3, dtype=float),
                424: np.identity(3, dtype=float),
                425: np.identity(3, dtype=float),
                431: np.identity(3, dtype=float),
                432: np.identity(3, dtype=float),
                433: np.identity(3, dtype=float),
                434: np.identity(3, dtype=float),
                435: np.identity(3, dtype=float),
                441: np.identity(3, dtype=float),
                442: np.identity(3, dtype=float),
                443: np.identity(3, dtype=float),
                444: np.identity(3, dtype=float),
                445: np.identity(3, dtype=float),
                451: np.identity(3, dtype=float),
                452: np.identity(3, dtype=float),
                453: np.identity(3, dtype=float),
                454: np.identity(3, dtype=float),
                455: np.identity(3, dtype=float),
                511: np.identity(3, dtype=float),
                512: np.identity(3, dtype=float),
                513: np.identity(3, dtype=float),
                514: np.identity(3, dtype=float),
                515: np.identity(3, dtype=float),
                521: np.identity(3, dtype=float),
                522: np.identity(3, dtype=float),
                523: np.identity(3, dtype=float),
                524: np.identity(3, dtype=float),
                525: np.identity(3, dtype=float),
                531: np.identity(3, dtype=float),
                532: np.identity(3, dtype=float),
                533: np.identity(3, dtype=float),
                534: np.identity(3, dtype=float),
                535: np.identity(3, dtype=float),
                541: np.identity(3, dtype=float),
                542: np.identity(3, dtype=float),
                543: np.identity(3, dtype=float),
                544: np.identity(3, dtype=float),
                545: np.identity(3, dtype=float),
                551: np.identity(3, dtype=float),
                552: np.identity(3, dtype=float),
                553: np.identity(3, dtype=float),
                554: np.identity(3, dtype=float),
                555: np.identity(3, dtype=float)
            }
            self._Fn = {
                111: np.identity(3, dtype=float),
                112: np.identity(3, dtype=float),
                113: np.identity(3, dtype=float),
                114: np.identity(3, dtype=float),
                115: np.identity(3, dtype=float),
                121: np.identity(3, dtype=float),
                122: np.identity(3, dtype=float),
                123: np.identity(3, dtype=float),
                124: np.identity(3, dtype=float),
                125: np.identity(3, dtype=float),
                131: np.identity(3, dtype=float),
                132: np.identity(3, dtype=float),
                133: np.identity(3, dtype=float),
                134: np.identity(3, dtype=float),
                135: np.identity(3, dtype=float),
                141: np.identity(3, dtype=float),
                142: np.identity(3, dtype=float),
                143: np.identity(3, dtype=float),
                144: np.identity(3, dtype=float),
                145: np.identity(3, dtype=float),
                151: np.identity(3, dtype=float),
                152: np.identity(3, dtype=float),
                153: np.identity(3, dtype=float),
                154: np.identity(3, dtype=float),
                155: np.identity(3, dtype=float),
                211: np.identity(3, dtype=float),
                212: np.identity(3, dtype=float),
                213: np.identity(3, dtype=float),
                214: np.identity(3, dtype=float),
                215: np.identity(3, dtype=float),
                221: np.identity(3, dtype=float),
                222: np.identity(3, dtype=float),
                223: np.identity(3, dtype=float),
                224: np.identity(3, dtype=float),
                225: np.identity(3, dtype=float),
                231: np.identity(3, dtype=float),
                232: np.identity(3, dtype=float),
                233: np.identity(3, dtype=float),
                234: np.identity(3, dtype=float),
                235: np.identity(3, dtype=float),
                241: np.identity(3, dtype=float),
                242: np.identity(3, dtype=float),
                243: np.identity(3, dtype=float),
                244: np.identity(3, dtype=float),
                245: np.identity(3, dtype=float),
                251: np.identity(3, dtype=float),
                252: np.identity(3, dtype=float),
                253: np.identity(3, dtype=float),
                254: np.identity(3, dtype=float),
                255: np.identity(3, dtype=float),
                311: np.identity(3, dtype=float),
                312: np.identity(3, dtype=float),
                313: np.identity(3, dtype=float),
                314: np.identity(3, dtype=float),
                315: np.identity(3, dtype=float),
                321: np.identity(3, dtype=float),
                322: np.identity(3, dtype=float),
                323: np.identity(3, dtype=float),
                324: np.identity(3, dtype=float),
                325: np.identity(3, dtype=float),
                331: np.identity(3, dtype=float),
                332: np.identity(3, dtype=float),
                333: np.identity(3, dtype=float),
                334: np.identity(3, dtype=float),
                335: np.identity(3, dtype=float),
                341: np.identity(3, dtype=float),
                342: np.identity(3, dtype=float),
                343: np.identity(3, dtype=float),
                344: np.identity(3, dtype=float),
                345: np.identity(3, dtype=float),
                351: np.identity(3, dtype=float),
                352: np.identity(3, dtype=float),
                353: np.identity(3, dtype=float),
                354: np.identity(3, dtype=float),
                355: np.identity(3, dtype=float),
                411: np.identity(3, dtype=float),
                412: np.identity(3, dtype=float),
                413: np.identity(3, dtype=float),
                414: np.identity(3, dtype=float),
                415: np.identity(3, dtype=float),
                421: np.identity(3, dtype=float),
                422: np.identity(3, dtype=float),
                423: np.identity(3, dtype=float),
                424: np.identity(3, dtype=float),
                425: np.identity(3, dtype=float),
                431: np.identity(3, dtype=float),
                432: np.identity(3, dtype=float),
                433: np.identity(3, dtype=float),
                434: np.identity(3, dtype=float),
                435: np.identity(3, dtype=float),
                441: np.identity(3, dtype=float),
                442: np.identity(3, dtype=float),
                443: np.identity(3, dtype=float),
                444: np.identity(3, dtype=float),
                445: np.identity(3, dtype=float),
                451: np.identity(3, dtype=float),
                452: np.identity(3, dtype=float),
                453: np.identity(3, dtype=float),
                454: np.identity(3, dtype=float),
                455: np.identity(3, dtype=float),
                511: np.identity(3, dtype=float),
                512: np.identity(3, dtype=float),
                513: np.identity(3, dtype=float),
                514: np.identity(3, dtype=float),
                515: np.identity(3, dtype=float),
                521: np.identity(3, dtype=float),
                522: np.identity(3, dtype=float),
                523: np.identity(3, dtype=float),
                524: np.identity(3, dtype=float),
                525: np.identity(3, dtype=float),
                531: np.identity(3, dtype=float),
                532: np.identity(3, dtype=float),
                533: np.identity(3, dtype=float),
                534: np.identity(3, dtype=float),
                535: np.identity(3, dtype=float),
                541: np.identity(3, dtype=float),
                542: np.identity(3, dtype=float),
                543: np.identity(3, dtype=float),
                544: np.identity(3, dtype=float),
                545: np.identity(3, dtype=float),
                551: np.identity(3, dtype=float),
                552: np.identity(3, dtype=float),
                553: np.identity(3, dtype=float),
                554: np.identity(3, dtype=float),
                555: np.identity(3, dtype=float)
            }

        # get the reference coordinates for the vetices of the tetrahedron
        self._x10, self._y10, self._z10 = self._vertex[1].coordinates('ref')
        self._x20, self._y20, self._z20 = self._vertex[2].coordinates('ref')
        self._x30, self._y30, self._z30 = self._vertex[3].coordinates('ref')
        self._x40, self._y40, self._z40 = self._vertex[4].coordinates('ref')
        
        # initialize current vertice coordinates 
        self._x1 = self._x10
        self._y1 = self._y10
        self._z1 = self._z10
        self._x2 = self._x20
        self._y2 = self._y20
        self._z2 = self._z20
        self._x3 = self._x30
        self._y3 = self._y30
        self._z3 = self._z30
        self._x4 = self._x40
        self._y4 = self._y40
        self._z4 = self._z40
                
    
        # determine the volume of this tetrahedron in its reference state
        self._V0 = self._volTet(self._x10, self._y10, self._z10,
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

        self._rho = mp.rhoAir()

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
        volT = m.sqrt(det(A) / 288.0)
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
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.toString.")
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
            raise RuntimeError('The requested vertex {} is '.format(number) +
                               'not in tetrhaderon {}.'.format(self._number))

    def gaussPoints(self):
        return self._gaussPts

    def update(self):
        # computes the fields positioned at the next time step

        # get the updated coordinates for the vetices of the tetrahedron
        self._x1, self._y1, self._z1 = self._vertex[1].coordinates('next')
        self._x2, self._y2, self._z2 = self._vertex[2].coordinates('next')
        self._x3, self._y3, self._z3 = self._vertex[3].coordinates('next')
        self._x4, self._y4, self._z4 = self._vertex[4].coordinates('next')

        # determine the volume of this tetrahedron in this next state
        self._Vn = self._volTet(self._x1, self._y1, self._z1, self._x2, 
                                self._y2, self._z2, self._x3, self._y3, 
                                self._z3, self._x4, self._y4, self._z4)

        # locate the centroid of this tetrahedrond in this next state
        self._centroidXn = self._centroidSF.interpolate(self._x1, self._x2, 
                                                        self._x3, self._x4)
        self._centroidYn = self._centroidSF.interpolate(self._y1, self._y2, 
                                                        self._y3, self._y4)
        self._centroidZn = self._centroidSF.interpolate(self._z1, self._z2, 
                                                        self._z3, self._z4)

        # coordiantes for the next updated vertices as tuples
        v1n = (self._x1, self._y1, self._z1)
        v2n = (self._x2, self._y2, self._z2)
        v3n = (self._x3, self._y3, self._z3)
        v4n = (self._x4, self._y4, self._z4)

        # coordinates for the reference vertices as tuples
        v1r = (self._x10, self._y10, self._z10)
        v2r = (self._x20, self._y20, self._z20)
        v3r = (self._x30, self._y30, self._z30)
        v4r = (self._x40, self._y40, self._z40)

        # establish the deformation and displacement gradients as dictionaries
        if self._gaussPts == 1:
            # displacement gradients located at the Gauss points of tetrahedron
            self._Gn[111] = self._shapeFns[111].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            # deformation gradients located at the Gauss points of tetrahedron
            self._Fn[111] = self._shapeFns[111].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
        elif self._gaussPts == 4:
            # displacement gradients located at the Gauss points of tetrahedron
            self._Gn[111] = self._shapeFns[111].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[112] = self._shapeFns[112].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[113] = self._shapeFns[113].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[114] = self._shapeFns[114].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[121] = self._shapeFns[121].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[122] = self._shapeFns[122].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[123] = self._shapeFns[123].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[124] = self._shapeFns[124].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[131] = self._shapeFns[131].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[132] = self._shapeFns[132].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[133] = self._shapeFns[133].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[134] = self._shapeFns[134].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[141] = self._shapeFns[141].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[142] = self._shapeFns[142].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[143] = self._shapeFns[143].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[144] = self._shapeFns[144].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Gn[211] = self._shapeFns[211].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[212] = self._shapeFns[212].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[213] = self._shapeFns[213].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[214] = self._shapeFns[214].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[221] = self._shapeFns[221].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[222] = self._shapeFns[222].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[223] = self._shapeFns[223].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[224] = self._shapeFns[224].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[231] = self._shapeFns[231].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[232] = self._shapeFns[232].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[233] = self._shapeFns[233].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[234] = self._shapeFns[234].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[241] = self._shapeFns[241].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[242] = self._shapeFns[242].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[243] = self._shapeFns[243].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[244] = self._shapeFns[244].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Gn[311] = self._shapeFns[311].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[312] = self._shapeFns[312].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[313] = self._shapeFns[313].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[314] = self._shapeFns[314].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[321] = self._shapeFns[321].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[322] = self._shapeFns[322].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[323] = self._shapeFns[323].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[324] = self._shapeFns[324].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[331] = self._shapeFns[331].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Gn[332] = self._shapeFns[332].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Gn[333] = self._shapeFns[333].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            self._Gn[334] = self._shapeFns[334].G(v1n, v2n, v3n, v4n,
                                              v1r, v2r, v3r, v4r)
            
            self._Gn[341] = self._shapeFns[341].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[342] = self._shapeFns[342].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[343] = self._shapeFns[343].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[344] = self._shapeFns[344].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Gn[411] = self._shapeFns[411].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[412] = self._shapeFns[412].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[413] = self._shapeFns[413].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[414] = self._shapeFns[414].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[421] = self._shapeFns[421].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[422] = self._shapeFns[422].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[423] = self._shapeFns[423].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[424] = self._shapeFns[424].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[431] = self._shapeFns[431].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[432] = self._shapeFns[432].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[433] = self._shapeFns[433].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[434] = self._shapeFns[434].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[441] = self._shapeFns[441].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[442] = self._shapeFns[442].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[443] = self._shapeFns[443].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[444] = self._shapeFns[444].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            # deformation gradients located at the Gauss points of tetrahedron
            self._Fn[111] = self._shapeFns[111].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[112] = self._shapeFns[112].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[113] = self._shapeFns[113].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[114] = self._shapeFns[114].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[121] = self._shapeFns[121].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[122] = self._shapeFns[122].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[123] = self._shapeFns[123].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[124] = self._shapeFns[124].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[131] = self._shapeFns[131].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[132] = self._shapeFns[132].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[133] = self._shapeFns[133].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[134] = self._shapeFns[134].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[141] = self._shapeFns[141].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[142] = self._shapeFns[142].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[143] = self._shapeFns[143].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[144] = self._shapeFns[144].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Fn[211] = self._shapeFns[211].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[212] = self._shapeFns[212].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[213] = self._shapeFns[213].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[214] = self._shapeFns[214].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[221] = self._shapeFns[221].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[222] = self._shapeFns[222].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[223] = self._shapeFns[223].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[224] = self._shapeFns[224].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[231] = self._shapeFns[231].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[232] = self._shapeFns[232].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[233] = self._shapeFns[233].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[234] = self._shapeFns[234].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[241] = self._shapeFns[241].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[242] = self._shapeFns[242].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[243] = self._shapeFns[243].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[244] = self._shapeFns[244].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Fn[311] = self._shapeFns[311].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[312] = self._shapeFns[312].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[313] = self._shapeFns[313].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[314] = self._shapeFns[314].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[321] = self._shapeFns[321].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[322] = self._shapeFns[322].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[323] = self._shapeFns[323].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[324] = self._shapeFns[324].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[331] = self._shapeFns[331].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[332] = self._shapeFns[332].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[333] = self._shapeFns[333].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[334] = self._shapeFns[334].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[341] = self._shapeFns[341].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[342] = self._shapeFns[342].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[343] = self._shapeFns[343].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[344] = self._shapeFns[344].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Fn[411] = self._shapeFns[411].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[412] = self._shapeFns[412].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[413] = self._shapeFns[413].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[414] = self._shapeFns[414].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[421] = self._shapeFns[421].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[422] = self._shapeFns[422].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[423] = self._shapeFns[423].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[424] = self._shapeFns[424].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[431] = self._shapeFns[431].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[432] = self._shapeFns[432].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[433] = self._shapeFns[433].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[434] = self._shapeFns[434].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[441] = self._shapeFns[441].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[442] = self._shapeFns[442].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[443] = self._shapeFns[443].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[444] = self._shapeFns[444].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
        else:  # gaussPts = 5
            # displacement gradients located at the Gauss points of tetrahedron
            self._Gn[111] = self._shapeFns[111].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[112] = self._shapeFns[112].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[113] = self._shapeFns[113].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[114] = self._shapeFns[114].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[115] = self._shapeFns[115].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[121] = self._shapeFns[121].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[122] = self._shapeFns[122].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[123] = self._shapeFns[123].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[124] = self._shapeFns[124].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[125] = self._shapeFns[125].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[131] = self._shapeFns[131].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[132] = self._shapeFns[132].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[133] = self._shapeFns[133].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[134] = self._shapeFns[134].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[135] = self._shapeFns[135].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[141] = self._shapeFns[141].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[142] = self._shapeFns[142].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[143] = self._shapeFns[143].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[144] = self._shapeFns[144].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[145] = self._shapeFns[145].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[151] = self._shapeFns[151].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[152] = self._shapeFns[152].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[153] = self._shapeFns[153].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[154] = self._shapeFns[154].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[155] = self._shapeFns[155].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Gn[211] = self._shapeFns[211].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[212] = self._shapeFns[212].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[213] = self._shapeFns[213].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[214] = self._shapeFns[214].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[215] = self._shapeFns[215].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[221] = self._shapeFns[221].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[222] = self._shapeFns[222].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[223] = self._shapeFns[223].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[224] = self._shapeFns[224].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[225] = self._shapeFns[225].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[231] = self._shapeFns[231].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[232] = self._shapeFns[232].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[233] = self._shapeFns[233].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[234] = self._shapeFns[234].G(v1n, v2n, v3n, v4n,
                                                 v1r, v2r, v3r, v4r)
            self._Gn[235] = self._shapeFns[235].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[241] = self._shapeFns[241].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[242] = self._shapeFns[242].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[243] = self._shapeFns[243].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[244] = self._shapeFns[244].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[245] = self._shapeFns[245].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[251] = self._shapeFns[251].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[252] = self._shapeFns[252].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[253] = self._shapeFns[253].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[254] = self._shapeFns[254].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[255] = self._shapeFns[255].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            
            self._Gn[311] = self._shapeFns[311].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[312] = self._shapeFns[312].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[313] = self._shapeFns[313].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[314] = self._shapeFns[314].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[315] = self._shapeFns[315].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[321] = self._shapeFns[321].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[322] = self._shapeFns[322].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[323] = self._shapeFns[323].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[324] = self._shapeFns[324].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[325] = self._shapeFns[325].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[331] = self._shapeFns[331].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[332] = self._shapeFns[332].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[333] = self._shapeFns[333].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[334] = self._shapeFns[334].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[335] = self._shapeFns[335].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[341] = self._shapeFns[341].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[342] = self._shapeFns[342].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[343] = self._shapeFns[343].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[344] = self._shapeFns[344].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[345] = self._shapeFns[345].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[351] = self._shapeFns[351].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[352] = self._shapeFns[352].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[353] = self._shapeFns[353].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[354] = self._shapeFns[354].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[355] = self._shapeFns[355].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Gn[411] = self._shapeFns[411].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[412] = self._shapeFns[412].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[413] = self._shapeFns[413].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[414] = self._shapeFns[414].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[415] = self._shapeFns[415].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
        
            self._Gn[421] = self._shapeFns[421].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[422] = self._shapeFns[422].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[423] = self._shapeFns[423].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[424] = self._shapeFns[424].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[425] = self._shapeFns[425].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[431] = self._shapeFns[431].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[432] = self._shapeFns[432].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[433] = self._shapeFns[433].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[434] = self._shapeFns[434].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[435] = self._shapeFns[435].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
        
            self._Gn[441] = self._shapeFns[441].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[442] = self._shapeFns[442].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[443] = self._shapeFns[443].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[444] = self._shapeFns[444].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[445] = self._shapeFns[445].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[451] = self._shapeFns[451].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[452] = self._shapeFns[452].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[453] = self._shapeFns[453].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[454] = self._shapeFns[454].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[455] = self._shapeFns[455].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Gn[511] = self._shapeFns[511].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[512] = self._shapeFns[512].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[513] = self._shapeFns[513].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[514] = self._shapeFns[514].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[515] = self._shapeFns[515].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[521] = self._shapeFns[521].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[522] = self._shapeFns[522].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[523] = self._shapeFns[523].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[524] = self._shapeFns[524].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[525] = self._shapeFns[525].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[531] = self._shapeFns[531].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[532] = self._shapeFns[532].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[533] = self._shapeFns[533].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[534] = self._shapeFns[534].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[535] = self._shapeFns[535].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[541] = self._shapeFns[541].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[542] = self._shapeFns[542].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[543] = self._shapeFns[543].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[544] = self._shapeFns[544].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[545] = self._shapeFns[545].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Gn[551] = self._shapeFns[551].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[552] = self._shapeFns[552].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[553] = self._shapeFns[553].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[554] = self._shapeFns[554].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Gn[555] = self._shapeFns[555].G(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
        
            
            
            # deformation gradients located at the Gauss points of tetrahedron
            self._Fn[111] = self._shapeFns[111].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[112] = self._shapeFns[112].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[113] = self._shapeFns[113].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[114] = self._shapeFns[114].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[115] = self._shapeFns[115].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[121] = self._shapeFns[121].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[122] = self._shapeFns[122].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[123] = self._shapeFns[123].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[124] = self._shapeFns[124].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[125] = self._shapeFns[125].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[131] = self._shapeFns[131].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[132] = self._shapeFns[132].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[133] = self._shapeFns[133].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[134] = self._shapeFns[134].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[135] = self._shapeFns[135].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[141] = self._shapeFns[141].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[142] = self._shapeFns[142].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[143] = self._shapeFns[143].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[144] = self._shapeFns[144].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[145] = self._shapeFns[145].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[151] = self._shapeFns[151].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[152] = self._shapeFns[152].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[153] = self._shapeFns[153].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[154] = self._shapeFns[154].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[155] = self._shapeFns[155].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Fn[211] = self._shapeFns[211].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[212] = self._shapeFns[212].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[213] = self._shapeFns[213].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[214] = self._shapeFns[214].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[215] = self._shapeFns[215].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[221] = self._shapeFns[221].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[222] = self._shapeFns[222].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[223] = self._shapeFns[223].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[224] = self._shapeFns[224].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[225] = self._shapeFns[225].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[231] = self._shapeFns[231].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[232] = self._shapeFns[232].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[233] = self._shapeFns[233].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[234] = self._shapeFns[234].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[235] = self._shapeFns[235].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[241] = self._shapeFns[241].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[242] = self._shapeFns[242].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[243] = self._shapeFns[243].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[244] = self._shapeFns[244].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[245] = self._shapeFns[245].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[251] = self._shapeFns[251].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[252] = self._shapeFns[252].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[253] = self._shapeFns[253].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[254] = self._shapeFns[254].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[255] = self._shapeFns[255].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Fn[311] = self._shapeFns[311].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[312] = self._shapeFns[312].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[313] = self._shapeFns[313].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[314] = self._shapeFns[314].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[315] = self._shapeFns[315].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
        
            self._Fn[321] = self._shapeFns[321].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[322] = self._shapeFns[322].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[323] = self._shapeFns[323].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[324] = self._shapeFns[324].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[325] = self._shapeFns[325].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[331] = self._shapeFns[331].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[332] = self._shapeFns[332].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[333] = self._shapeFns[333].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[334] = self._shapeFns[334].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[335] = self._shapeFns[335].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[341] = self._shapeFns[341].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[342] = self._shapeFns[342].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[343] = self._shapeFns[343].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[344] = self._shapeFns[344].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[345] = self._shapeFns[345].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[351] = self._shapeFns[351].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[352] = self._shapeFns[352].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[353] = self._shapeFns[353].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[354] = self._shapeFns[354].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[355] = self._shapeFns[355].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Fn[411] = self._shapeFns[411].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[412] = self._shapeFns[412].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[413] = self._shapeFns[413].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[414] = self._shapeFns[414].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[415] = self._shapeFns[415].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[421] = self._shapeFns[421].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[422] = self._shapeFns[422].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[423] = self._shapeFns[423].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[424] = self._shapeFns[424].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[425] = self._shapeFns[425].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[431] = self._shapeFns[431].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[432] = self._shapeFns[432].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[433] = self._shapeFns[433].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[434] = self._shapeFns[434].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[435] = self._shapeFns[435].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[441] = self._shapeFns[441].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[442] = self._shapeFns[442].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[443] = self._shapeFns[443].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[444] = self._shapeFns[444].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[445] = self._shapeFns[445].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[451] = self._shapeFns[451].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[452] = self._shapeFns[452].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[453] = self._shapeFns[453].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[454] = self._shapeFns[454].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[455] = self._shapeFns[455].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            
            self._Fn[511] = self._shapeFns[511].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[512] = self._shapeFns[512].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[513] = self._shapeFns[513].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[514] = self._shapeFns[514].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[515] = self._shapeFns[515].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[521] = self._shapeFns[521].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[522] = self._shapeFns[522].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[523] = self._shapeFns[523].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[524] = self._shapeFns[524].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[525] = self._shapeFns[525].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[531] = self._shapeFns[531].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[532] = self._shapeFns[532].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[533] = self._shapeFns[533].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[534] = self._shapeFns[534].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[535] = self._shapeFns[535].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[541] = self._shapeFns[541].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[542] = self._shapeFns[542].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[543] = self._shapeFns[543].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[544] = self._shapeFns[544].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[545] = self._shapeFns[545].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            
            self._Fn[551] = self._shapeFns[551].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[552] = self._shapeFns[552].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[553] = self._shapeFns[553].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[554] = self._shapeFns[554].F(v1n, v2n, v3n, v4n,
                                                  v1r, v2r, v3r, v4r)
            self._Fn[555] = self._shapeFns[555].F(v1n, v2n, v3n, v4n,
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
        if self._gaussPts == 1:
            for i in range(111, self._gaussPts+111):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
                
        elif self._gaussPts == 4:   
            for i in range(111, self._gaussPts+111):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]  
            for i in range(121, self._gaussPts+121):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(131, self._gaussPts+131):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(141, self._gaussPts+141):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(211, self._gaussPts+211):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(221, self._gaussPts+221):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(231, self._gaussPts+231):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(241, self._gaussPts+241):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(311, self._gaussPts+311):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]   
            for i in range(321, self._gaussPts+321):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(331, self._gaussPts+331):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(341, self._gaussPts+341):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]  
            for i in range(411, self._gaussPts+411):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
            for i in range(421, self._gaussPts+421):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
            for i in range(431, self._gaussPts+431):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
            for i in range(441, self._gaussPts+441):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
                
        else:  # gaussPts = 5
            for i in range(111, self._gaussPts+111):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]  
            for i in range(121, self._gaussPts+121):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(131, self._gaussPts+131):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(141, self._gaussPts+141):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(151, self._gaussPts+151):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(211, self._gaussPts+211):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(221, self._gaussPts+221):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(231, self._gaussPts+231):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(241, self._gaussPts+241):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(251, self._gaussPts+251):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(311, self._gaussPts+311):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]   
            for i in range(321, self._gaussPts+321):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(331, self._gaussPts+331):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(341, self._gaussPts+341):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(351, self._gaussPts+351):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(411, self._gaussPts+411):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
            for i in range(421, self._gaussPts+421):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
            for i in range(431, self._gaussPts+431):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
            for i in range(441, self._gaussPts+441):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(451, self._gaussPts+451):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(511, self._gaussPts+511):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
            for i in range(521, self._gaussPts+521):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
            for i in range(531, self._gaussPts+531):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
            for i in range(541, self._gaussPts+541):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
            for i in range(551, self._gaussPts+551):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :] 
                 

        return  # nothing

    # Material properties that associate with this tetrahedron.

    def massDensity(self):
        # returns the mass density of the chord (collagen and elastin fibers)
        return self._rho

    # Geometric properties of this tetrahedron

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
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to tetrahedron.volume.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.volume.")

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
                raise RuntimeError("An unknown state {} in ".format(state) +
                                   "a call to tetrahedron.volumetricStretch.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to tetrahedron.volumetricStretch.")

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
                raise RuntimeError("An unknown state {} in ".format(state) +
                                   "a call to tetrahedron.volumetricStrain.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to tetrahedron.volumetricStrain.")

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
                raise RuntimeError("An unknown state {} in ".format(state) +
                                   "a call to tetrahedron.dVolumetricStrain.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to tetrahedron.dVolumetricStrain.")

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
                raise RuntimeError("An unknown state {} in ".format(state) +
                                   "a call to tetrahedron.centroid.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to tetrahedron.centroid.")
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
                raise RuntimeError("An unknown state {} in ".format(state) +
                                   "a call to tetrahedron.velocity.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to tetrahedron.velocity.")
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
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to tetrahedron.acceleration.")
        return np.array([ax, ay, az])

    # displacement gradient at a Gauss point
    def G(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "tetrahedron.G, you sent {}.".format(gaussPt))
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
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to tetrahedron.G.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.G.")

    # deformation gradient at a Gauss point
    def F(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "tetrahedron.F, you sent {}.".format(gaussPt))
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
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to tetrahedron.F.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.F.")

    def L(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "tetrahedron.L, you sent {}.".format(gaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                # use central difference scheme
                dF = ((self._Fn[gaussPt] - self._Fp[gaussPt])
                      / (2.0 * self._h))
                fInv = inv(self._Fc[gaussPt])
            elif state == 'n' or state == 'next':
                # use backward difference scheme
                dF = ((3.0 * self._Fn[gaussPt] - 4.0 * self._Fc[gaussPt] +
                       self._Fp[gaussPt]) / (2.0 * self._h))
                fInv = inv(self._Fn[gaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use forward difference scheme
                dF = ((-self._Fn[gaussPt] + 4.0 * self._Fc[gaussPt] -
                       3.0 * self._Fp[gaussPt]) / (2.0 * self._h))
                fInv = inv(self._Fp[gaussPt])
            elif state == 'r' or state == 'ref' or state == 'reference':
                dF = np.zeros(3, dtype=float)
                fInv = np.identity(3, dtype=float)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to tetrahedron.L.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.L.")
        return np.dot(dF, fInv)

    def shapeFunction(self, gaussPt):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "tetrahedron.shapeFunction and " +
                               " you sent {}.".format(gaussPt))
            sf = self._shapeFns[gaussPt]
        return sf

    def massMatrix(self):
        # assign coordinates at the vertices in the reference configuration
        x1 = (self._x1, self._y1, self._z1)
        x2 = (self._x2, self._y2, self._z2)
        x3 = (self._x3, self._y3, self._z3)
        x4 = (self._x4, self._y4, self._z4)

        # determine the mass matrix
        if self._gaussPts == 1:
            # 'natural' weight of the element
            wgt = 1.0 / 6.0
            w = np.array([wgt])

            jacob111 = self._shapeFns[111].jacobian(x1, x2, x3, x4)
            detJ = det(jacob111)

            nn1 = np.dot(np.transpose(self._shapeFns[111].Nmatx),
                         self._shapeFns[111].Nmatx)

            # the consistent mass matrix for 1 Gauss point
            massC = self._rho * (detJ * w[0] * w[0] * w[0] * nn1)

            # the lumped mass matrix for 1 Gauss point
            massL = np.zeros((12, 12), dtype=float)
            row, col = np.diag_indices_from(massC)
            massL[row, col] = massC.sum(axis=1)

            # the mass matrix is the average of the above two mass matrices
            mass = 0.5 * (massC + massL)
        elif self._gaussPts == 4:
            # 'natural' weights of the element
            wgt = 1.0 / 24.0
            wel = np.array([wgt, wgt, wgt, wgt])

            jacob111 = self._shapeFns[111].jacobian(x1, x2, x3, x4)
            jacob112 = self._shapeFns[112].jacobian(x1, x2, x3, x4)
            jacob113 = self._shapeFns[113].jacobian(x1, x2, x3, x4)
            jacob114 = self._shapeFns[114].jacobian(x1, x2, x3, x4)
            
            jacob121 = self._shapeFns[121].jacobian(x1, x2, x3, x4)
            jacob122 = self._shapeFns[122].jacobian(x1, x2, x3, x4)
            jacob123 = self._shapeFns[123].jacobian(x1, x2, x3, x4)
            jacob124 = self._shapeFns[124].jacobian(x1, x2, x3, x4)
            
            jacob131 = self._shapeFns[131].jacobian(x1, x2, x3, x4)
            jacob132 = self._shapeFns[132].jacobian(x1, x2, x3, x4)
            jacob133 = self._shapeFns[133].jacobian(x1, x2, x3, x4)
            jacob134 = self._shapeFns[134].jacobian(x1, x2, x3, x4)
            
            jacob141 = self._shapeFns[141].jacobian(x1, x2, x3, x4)
            jacob142 = self._shapeFns[142].jacobian(x1, x2, x3, x4)
            jacob143 = self._shapeFns[143].jacobian(x1, x2, x3, x4)
            jacob144 = self._shapeFns[144].jacobian(x1, x2, x3, x4)
            
            
            jacob211 = self._shapeFns[211].jacobian(x1, x2, x3, x4)
            jacob212 = self._shapeFns[212].jacobian(x1, x2, x3, x4)
            jacob213 = self._shapeFns[213].jacobian(x1, x2, x3, x4)
            jacob214 = self._shapeFns[214].jacobian(x1, x2, x3, x4)
            
            jacob221 = self._shapeFns[221].jacobian(x1, x2, x3, x4)
            jacob222 = self._shapeFns[222].jacobian(x1, x2, x3, x4)
            jacob223 = self._shapeFns[223].jacobian(x1, x2, x3, x4)
            jacob224 = self._shapeFns[224].jacobian(x1, x2, x3, x4)
            
            jacob231 = self._shapeFns[231].jacobian(x1, x2, x3, x4)
            jacob232 = self._shapeFns[232].jacobian(x1, x2, x3, x4)
            jacob233 = self._shapeFns[233].jacobian(x1, x2, x3, x4)
            jacob234 = self._shapeFns[234].jacobian(x1, x2, x3, x4)
            
            jacob241 = self._shapeFns[241].jacobian(x1, x2, x3, x4)
            jacob242 = self._shapeFns[242].jacobian(x1, x2, x3, x4)
            jacob243 = self._shapeFns[243].jacobian(x1, x2, x3, x4)
            jacob244 = self._shapeFns[244].jacobian(x1, x2, x3, x4)
            
            
            jacob311 = self._shapeFns[311].jacobian(x1, x2, x3, x4)
            jacob312 = self._shapeFns[312].jacobian(x1, x2, x3, x4)
            jacob313 = self._shapeFns[313].jacobian(x1, x2, x3, x4)
            jacob314 = self._shapeFns[314].jacobian(x1, x2, x3, x4)
            
            jacob321 = self._shapeFns[321].jacobian(x1, x2, x3, x4)
            jacob322 = self._shapeFns[322].jacobian(x1, x2, x3, x4)
            jacob323 = self._shapeFns[323].jacobian(x1, x2, x3, x4)
            jacob324 = self._shapeFns[324].jacobian(x1, x2, x3, x4)
            
            jacob331 = self._shapeFns[331].jacobian(x1, x2, x3, x4)
            jacob332 = self._shapeFns[332].jacobian(x1, x2, x3, x4)
            jacob333 = self._shapeFns[333].jacobian(x1, x2, x3, x4)
            jacob334 = self._shapeFns[334].jacobian(x1, x2, x3, x4)
            
            jacob341 = self._shapeFns[341].jacobian(x1, x2, x3, x4)
            jacob342 = self._shapeFns[342].jacobian(x1, x2, x3, x4)
            jacob343 = self._shapeFns[343].jacobian(x1, x2, x3, x4)
            jacob344 = self._shapeFns[344].jacobian(x1, x2, x3, x4)
            
            
            jacob411 = self._shapeFns[411].jacobian(x1, x2, x3, x4)
            jacob412 = self._shapeFns[412].jacobian(x1, x2, x3, x4)
            jacob413 = self._shapeFns[413].jacobian(x1, x2, x3, x4)
            jacob414 = self._shapeFns[414].jacobian(x1, x2, x3, x4)
            
            jacob421 = self._shapeFns[421].jacobian(x1, x2, x3, x4)
            jacob422 = self._shapeFns[422].jacobian(x1, x2, x3, x4)
            jacob423 = self._shapeFns[423].jacobian(x1, x2, x3, x4)
            jacob424 = self._shapeFns[424].jacobian(x1, x2, x3, x4)
            
            jacob431 = self._shapeFns[431].jacobian(x1, x2, x3, x4)
            jacob432 = self._shapeFns[432].jacobian(x1, x2, x3, x4)
            jacob433 = self._shapeFns[433].jacobian(x1, x2, x3, x4)
            jacob434 = self._shapeFns[434].jacobian(x1, x2, x3, x4)
            
            jacob441 = self._shapeFns[441].jacobian(x1, x2, x3, x4)
            jacob442 = self._shapeFns[442].jacobian(x1, x2, x3, x4)
            jacob443 = self._shapeFns[443].jacobian(x1, x2, x3, x4)
            jacob444 = self._shapeFns[444].jacobian(x1, x2, x3, x4)
            

            # determinant of the Jacobian matrix
            detJ111 = det(jacob111)
            detJ112 = det(jacob112)
            detJ113 = det(jacob113)
            detJ114 = det(jacob114)
            
            detJ121 = det(jacob121)
            detJ122 = det(jacob122)
            detJ123 = det(jacob123)
            detJ124 = det(jacob124)
            
            detJ131 = det(jacob131)
            detJ132 = det(jacob132)
            detJ133 = det(jacob133)
            detJ134 = det(jacob134)
            
            detJ141 = det(jacob141)
            detJ142 = det(jacob142)
            detJ143 = det(jacob143)
            detJ144 = det(jacob144)
            
            
            detJ211 = det(jacob211)
            detJ212 = det(jacob212)
            detJ213 = det(jacob213)
            detJ214 = det(jacob214)
            
            detJ221 = det(jacob221)
            detJ222 = det(jacob222)
            detJ223 = det(jacob223)
            detJ224 = det(jacob224)
            
            detJ231 = det(jacob231)
            detJ232 = det(jacob232)
            detJ233 = det(jacob233)
            detJ234 = det(jacob234)
            
            detJ241 = det(jacob241)
            detJ242 = det(jacob242)
            detJ243 = det(jacob243)
            detJ244 = det(jacob244)
            
            
            detJ311 = det(jacob311)
            detJ312 = det(jacob312)
            detJ313 = det(jacob313)
            detJ314 = det(jacob314)
            
            detJ321 = det(jacob321)
            detJ322 = det(jacob322)
            detJ323 = det(jacob323)
            detJ324 = det(jacob324)
            
            detJ331 = det(jacob331)
            detJ332 = det(jacob332)
            detJ333 = det(jacob333)
            detJ334 = det(jacob334)
            
            detJ341 = det(jacob341)
            detJ342 = det(jacob342)
            detJ343 = det(jacob343)
            detJ344 = det(jacob344)
            
            
            detJ411 = det(jacob411)
            detJ412 = det(jacob412)
            detJ413 = det(jacob413)
            detJ414 = det(jacob414)
            
            detJ421 = det(jacob421)
            detJ422 = det(jacob422)
            detJ423 = det(jacob423)
            detJ424 = det(jacob424)
            
            detJ431 = det(jacob431)
            detJ432 = det(jacob432)
            detJ433 = det(jacob433)
            detJ434 = det(jacob434)
            
            detJ441 = det(jacob441)
            detJ442 = det(jacob442)
            detJ443 = det(jacob443)
            detJ444 = det(jacob444)




            nn111 = np.dot(np.transpose(self._shapeFns[111].Nmatx),
                         self._shapeFns[111].Nmatx)
            nn112 = np.dot(np.transpose(self._shapeFns[112].Nmatx),
                         self._shapeFns[112].Nmatx)
            nn113 = np.dot(np.transpose(self._shapeFns[113].Nmatx),
                         self._shapeFns[113].Nmatx)
            nn114 = np.dot(np.transpose(self._shapeFns[114].Nmatx),
                         self._shapeFns[114].Nmatx)
            
            nn121 = np.dot(np.transpose(self._shapeFns[121].Nmatx),
                         self._shapeFns[121].Nmatx)
            nn122 = np.dot(np.transpose(self._shapeFns[122].Nmatx),
                         self._shapeFns[122].Nmatx)
            nn123 = np.dot(np.transpose(self._shapeFns[123].Nmatx),
                         self._shapeFns[123].Nmatx)
            nn124 = np.dot(np.transpose(self._shapeFns[124].Nmatx),
                         self._shapeFns[124].Nmatx)
            
            nn131 = np.dot(np.transpose(self._shapeFns[131].Nmatx),
                         self._shapeFns[131].Nmatx)
            nn132 = np.dot(np.transpose(self._shapeFns[132].Nmatx),
                         self._shapeFns[132].Nmatx)
            nn133 = np.dot(np.transpose(self._shapeFns[133].Nmatx),
                         self._shapeFns[133].Nmatx)
            nn134 = np.dot(np.transpose(self._shapeFns[134].Nmatx),
                         self._shapeFns[134].Nmatx)
            
            nn141 = np.dot(np.transpose(self._shapeFns[141].Nmatx),
                         self._shapeFns[141].Nmatx)
            nn142 = np.dot(np.transpose(self._shapeFns[142].Nmatx),
                         self._shapeFns[142].Nmatx)
            nn143 = np.dot(np.transpose(self._shapeFns[143].Nmatx),
                         self._shapeFns[143].Nmatx)
            nn144 = np.dot(np.transpose(self._shapeFns[144].Nmatx),
                         self._shapeFns[144].Nmatx)
            
            
            
            nn211 = np.dot(np.transpose(self._shapeFns[211].Nmatx),
                         self._shapeFns[211].Nmatx)
            nn212 = np.dot(np.transpose(self._shapeFns[212].Nmatx),
                         self._shapeFns[212].Nmatx)
            nn213 = np.dot(np.transpose(self._shapeFns[213].Nmatx),
                         self._shapeFns[213].Nmatx)
            nn214 = np.dot(np.transpose(self._shapeFns[214].Nmatx),
                         self._shapeFns[214].Nmatx)
            
            nn221 = np.dot(np.transpose(self._shapeFns[221].Nmatx),
                         self._shapeFns[221].Nmatx)
            nn222 = np.dot(np.transpose(self._shapeFns[222].Nmatx),
                         self._shapeFns[222].Nmatx)
            nn223 = np.dot(np.transpose(self._shapeFns[223].Nmatx),
                         self._shapeFns[223].Nmatx)
            nn224 = np.dot(np.transpose(self._shapeFns[224].Nmatx),
                         self._shapeFns[224].Nmatx)
            
            nn231 = np.dot(np.transpose(self._shapeFns[231].Nmatx),
                         self._shapeFns[231].Nmatx)
            nn232 = np.dot(np.transpose(self._shapeFns[232].Nmatx),
                         self._shapeFns[232].Nmatx)
            nn233 = np.dot(np.transpose(self._shapeFns[233].Nmatx),
                         self._shapeFns[233].Nmatx)
            nn234 = np.dot(np.transpose(self._shapeFns[234].Nmatx),
                         self._shapeFns[234].Nmatx)
            
            nn241 = np.dot(np.transpose(self._shapeFns[241].Nmatx),
                         self._shapeFns[241].Nmatx)
            nn242 = np.dot(np.transpose(self._shapeFns[242].Nmatx),
                         self._shapeFns[242].Nmatx)
            nn243 = np.dot(np.transpose(self._shapeFns[243].Nmatx),
                         self._shapeFns[243].Nmatx)
            nn244 = np.dot(np.transpose(self._shapeFns[244].Nmatx),
                         self._shapeFns[244].Nmatx)
            
            
            nn311 = np.dot(np.transpose(self._shapeFns[311].Nmatx),
                         self._shapeFns[311].Nmatx)
            nn312 = np.dot(np.transpose(self._shapeFns[312].Nmatx),
                         self._shapeFns[312].Nmatx)
            nn313 = np.dot(np.transpose(self._shapeFns[313].Nmatx),
                         self._shapeFns[313].Nmatx)
            nn314 = np.dot(np.transpose(self._shapeFns[314].Nmatx),
                         self._shapeFns[314].Nmatx)
            
            nn321 = np.dot(np.transpose(self._shapeFns[321].Nmatx),
                         self._shapeFns[321].Nmatx)
            nn322 = np.dot(np.transpose(self._shapeFns[322].Nmatx),
                         self._shapeFns[322].Nmatx)
            nn323 = np.dot(np.transpose(self._shapeFns[323].Nmatx),
                         self._shapeFns[323].Nmatx)
            nn324 = np.dot(np.transpose(self._shapeFns[324].Nmatx),
                         self._shapeFns[324].Nmatx)
            
            nn331 = np.dot(np.transpose(self._shapeFns[331].Nmatx),
                         self._shapeFns[331].Nmatx)
            nn332 = np.dot(np.transpose(self._shapeFns[332].Nmatx),
                         self._shapeFns[332].Nmatx)
            nn333 = np.dot(np.transpose(self._shapeFns[333].Nmatx),
                         self._shapeFns[333].Nmatx)
            nn334 = np.dot(np.transpose(self._shapeFns[334].Nmatx),
                         self._shapeFns[334].Nmatx)
            
            nn341 = np.dot(np.transpose(self._shapeFns[341].Nmatx),
                         self._shapeFns[341].Nmatx)
            nn342 = np.dot(np.transpose(self._shapeFns[342].Nmatx),
                         self._shapeFns[342].Nmatx)
            nn343 = np.dot(np.transpose(self._shapeFns[343].Nmatx),
                         self._shapeFns[343].Nmatx)
            nn344 = np.dot(np.transpose(self._shapeFns[344].Nmatx),
                         self._shapeFns[344].Nmatx)
            
            
            nn411 = np.dot(np.transpose(self._shapeFns[411].Nmatx),
                         self._shapeFns[411].Nmatx)
            nn412 = np.dot(np.transpose(self._shapeFns[412].Nmatx),
                         self._shapeFns[412].Nmatx)
            nn413 = np.dot(np.transpose(self._shapeFns[413].Nmatx),
                         self._shapeFns[413].Nmatx)
            nn414 = np.dot(np.transpose(self._shapeFns[414].Nmatx),
                         self._shapeFns[414].Nmatx)
            
            nn421 = np.dot(np.transpose(self._shapeFns[421].Nmatx),
                         self._shapeFns[421].Nmatx)
            nn422 = np.dot(np.transpose(self._shapeFns[422].Nmatx),
                         self._shapeFns[422].Nmatx)
            nn423 = np.dot(np.transpose(self._shapeFns[423].Nmatx),
                         self._shapeFns[423].Nmatx)
            nn424 = np.dot(np.transpose(self._shapeFns[424].Nmatx),
                         self._shapeFns[424].Nmatx)
            
            nn431 = np.dot(np.transpose(self._shapeFns[431].Nmatx),
                         self._shapeFns[431].Nmatx)
            nn432 = np.dot(np.transpose(self._shapeFns[432].Nmatx),
                         self._shapeFns[432].Nmatx)
            nn433 = np.dot(np.transpose(self._shapeFns[433].Nmatx),
                         self._shapeFns[433].Nmatx)
            nn434 = np.dot(np.transpose(self._shapeFns[434].Nmatx),
                         self._shapeFns[434].Nmatx)
            
            nn441 = np.dot(np.transpose(self._shapeFns[441].Nmatx),
                         self._shapeFns[441].Nmatx)
            nn442 = np.dot(np.transpose(self._shapeFns[442].Nmatx),
                         self._shapeFns[442].Nmatx)
            nn443 = np.dot(np.transpose(self._shapeFns[443].Nmatx),
                         self._shapeFns[443].Nmatx)
            nn444 = np.dot(np.transpose(self._shapeFns[444].Nmatx),
                         self._shapeFns[444].Nmatx)
            
            

            # the consistent mass matrix for 4 Gauss points
            massC = (self._rho * (detJ111 * wel[0] * wel[0] * wel[0] * nn111 +
                                  detJ112 * wel[0] * wel[0] * wel[1] * nn112 +
                                  detJ113 * wel[0] * wel[0] * wel[2] * nn113 + 
                                  detJ114 * wel[0] * wel[0] * wel[3] * nn114 +
                                  detJ121 * wel[0] * wel[1] * wel[0] * nn121 +
                                  detJ122 * wel[0] * wel[1] * wel[1] * nn122 +
                                  detJ123 * wel[0] * wel[1] * wel[2] * nn123 + 
                                  detJ124 * wel[0] * wel[1] * wel[3] * nn124 +
                                  detJ131 * wel[0] * wel[2] * wel[0] * nn131 +
                                  detJ132 * wel[0] * wel[2] * wel[1] * nn132 +
                                  detJ133 * wel[0] * wel[2] * wel[2] * nn133 + 
                                  detJ134 * wel[0] * wel[2] * wel[3] * nn134 +
                                  detJ141 * wel[0] * wel[3] * wel[0] * nn141 +
                                  detJ142 * wel[0] * wel[3] * wel[1] * nn142 +
                                  detJ143 * wel[0] * wel[3] * wel[2] * nn143 + 
                                  detJ144 * wel[0] * wel[3] * wel[3] * nn144 +
                                  detJ211 * wel[1] * wel[0] * wel[0] * nn211 +
                                  detJ212 * wel[1] * wel[0] * wel[1] * nn212 +
                                  detJ213 * wel[1] * wel[0] * wel[2] * nn213 + 
                                  detJ214 * wel[1] * wel[0] * wel[3] * nn214 +
                                  detJ221 * wel[1] * wel[1] * wel[0] * nn221 +
                                  detJ222 * wel[1] * wel[1] * wel[1] * nn222 +
                                  detJ223 * wel[1] * wel[1] * wel[2] * nn223 + 
                                  detJ224 * wel[1] * wel[1] * wel[3] * nn224 +
                                  detJ231 * wel[1] * wel[2] * wel[0] * nn231 +
                                  detJ232 * wel[1] * wel[2] * wel[1] * nn232 +
                                  detJ233 * wel[1] * wel[2] * wel[2] * nn233 + 
                                  detJ234 * wel[1] * wel[2] * wel[3] * nn234 +
                                  detJ241 * wel[1] * wel[3] * wel[0] * nn241 +
                                  detJ242 * wel[1] * wel[3] * wel[1] * nn242 +
                                  detJ243 * wel[1] * wel[3] * wel[2] * nn243 + 
                                  detJ244 * wel[1] * wel[3] * wel[3] * nn244 +
                                  detJ311 * wel[2] * wel[0] * wel[0] * nn311 +
                                  detJ312 * wel[2] * wel[0] * wel[1] * nn312 +
                                  detJ313 * wel[2] * wel[0] * wel[2] * nn313 + 
                                  detJ314 * wel[2] * wel[0] * wel[3] * nn314 +
                                  detJ321 * wel[2] * wel[1] * wel[0] * nn321 +
                                  detJ322 * wel[2] * wel[1] * wel[1] * nn322 +
                                  detJ323 * wel[2] * wel[1] * wel[2] * nn323 + 
                                  detJ324 * wel[2] * wel[1] * wel[3] * nn324 +
                                  detJ331 * wel[2] * wel[2] * wel[0] * nn331 +
                                  detJ332 * wel[2] * wel[2] * wel[1] * nn332 +
                                  detJ333 * wel[2] * wel[2] * wel[2] * nn333 + 
                                  detJ334 * wel[2] * wel[2] * wel[3] * nn334 +
                                  detJ341 * wel[2] * wel[3] * wel[0] * nn341 +
                                  detJ342 * wel[2] * wel[3] * wel[1] * nn342 +
                                  detJ343 * wel[2] * wel[3] * wel[2] * nn343 + 
                                  detJ344 * wel[2] * wel[3] * wel[3] * nn344 +
                                  detJ411 * wel[3] * wel[0] * wel[0] * nn411 +
                                  detJ412 * wel[3] * wel[0] * wel[1] * nn412 +
                                  detJ413 * wel[3] * wel[0] * wel[2] * nn413 + 
                                  detJ414 * wel[3] * wel[0] * wel[3] * nn414 +
                                  detJ421 * wel[3] * wel[1] * wel[0] * nn421 +
                                  detJ422 * wel[3] * wel[1] * wel[1] * nn422 +
                                  detJ423 * wel[3] * wel[1] * wel[2] * nn423 + 
                                  detJ424 * wel[3] * wel[1] * wel[3] * nn424 +
                                  detJ431 * wel[3] * wel[2] * wel[0] * nn431 +
                                  detJ432 * wel[3] * wel[2] * wel[1] * nn432 +
                                  detJ433 * wel[3] * wel[2] * wel[2] * nn433 + 
                                  detJ434 * wel[3] * wel[2] * wel[3] * nn434 +
                                  detJ441 * wel[3] * wel[3] * wel[0] * nn441 +
                                  detJ442 * wel[3] * wel[3] * wel[1] * nn442 +
                                  detJ443 * wel[3] * wel[3] * wel[2] * nn443 + 
                                  detJ444 * wel[3] * wel[3] * wel[3] * nn444))

            # the lumped mass matrix for 4 Gauss points
            massL = np.zeros((12, 12), dtype=float)
            row, col = np.diag_indices_from(massC)
            massL[row, col] = massC.sum(axis=1)

            # the mass matrix is the average of the above two mass matrices
            mass = 0.5 * (massC + massL)
        else:  # gaussPts = 5
            # 'natural' weights of the element
            wgt1 = -2.0 / 15.0
            wgt2 = 3.0 / 40.0
            wel = np.array([wgt1, wgt2, wgt2, wgt2, wgt2])


            jacob111 = self._shapeFns[111].jacobian(x1, x2, x3, x4)
            jacob112 = self._shapeFns[112].jacobian(x1, x2, x3, x4)
            jacob113 = self._shapeFns[113].jacobian(x1, x2, x3, x4)
            jacob114 = self._shapeFns[114].jacobian(x1, x2, x3, x4)
            jacob115 = self._shapeFns[115].jacobian(x1, x2, x3, x4)
            
            jacob121 = self._shapeFns[121].jacobian(x1, x2, x3, x4)
            jacob122 = self._shapeFns[122].jacobian(x1, x2, x3, x4)
            jacob123 = self._shapeFns[123].jacobian(x1, x2, x3, x4)
            jacob124 = self._shapeFns[124].jacobian(x1, x2, x3, x4)
            jacob125 = self._shapeFns[125].jacobian(x1, x2, x3, x4)
            
            jacob131 = self._shapeFns[131].jacobian(x1, x2, x3, x4)
            jacob132 = self._shapeFns[132].jacobian(x1, x2, x3, x4)
            jacob133 = self._shapeFns[133].jacobian(x1, x2, x3, x4)
            jacob134 = self._shapeFns[134].jacobian(x1, x2, x3, x4)
            jacob135 = self._shapeFns[135].jacobian(x1, x2, x3, x4)
            
            jacob141 = self._shapeFns[141].jacobian(x1, x2, x3, x4)
            jacob142 = self._shapeFns[142].jacobian(x1, x2, x3, x4)
            jacob143 = self._shapeFns[143].jacobian(x1, x2, x3, x4)
            jacob144 = self._shapeFns[144].jacobian(x1, x2, x3, x4)
            jacob145 = self._shapeFns[145].jacobian(x1, x2, x3, x4)
            
            jacob151 = self._shapeFns[151].jacobian(x1, x2, x3, x4)
            jacob152 = self._shapeFns[152].jacobian(x1, x2, x3, x4)
            jacob153 = self._shapeFns[153].jacobian(x1, x2, x3, x4)
            jacob154 = self._shapeFns[154].jacobian(x1, x2, x3, x4)
            jacob155 = self._shapeFns[155].jacobian(x1, x2, x3, x4)
            
            
            jacob211 = self._shapeFns[211].jacobian(x1, x2, x3, x4)
            jacob212 = self._shapeFns[212].jacobian(x1, x2, x3, x4)
            jacob213 = self._shapeFns[213].jacobian(x1, x2, x3, x4)
            jacob214 = self._shapeFns[214].jacobian(x1, x2, x3, x4)
            jacob215 = self._shapeFns[215].jacobian(x1, x2, x3, x4)
                      
            jacob221 = self._shapeFns[221].jacobian(x1, x2, x3, x4)
            jacob222 = self._shapeFns[222].jacobian(x1, x2, x3, x4)
            jacob223 = self._shapeFns[223].jacobian(x1, x2, x3, x4)
            jacob224 = self._shapeFns[224].jacobian(x1, x2, x3, x4)
            jacob225 = self._shapeFns[225].jacobian(x1, x2, x3, x4)
            
            jacob231 = self._shapeFns[231].jacobian(x1, x2, x3, x4)
            jacob232 = self._shapeFns[232].jacobian(x1, x2, x3, x4)
            jacob233 = self._shapeFns[233].jacobian(x1, x2, x3, x4)
            jacob234 = self._shapeFns[234].jacobian(x1, x2, x3, x4)
            jacob235 = self._shapeFns[235].jacobian(x1, x2, x3, x4)
            
            jacob241 = self._shapeFns[241].jacobian(x1, x2, x3, x4)
            jacob242 = self._shapeFns[242].jacobian(x1, x2, x3, x4)
            jacob243 = self._shapeFns[243].jacobian(x1, x2, x3, x4)
            jacob244 = self._shapeFns[244].jacobian(x1, x2, x3, x4)
            jacob245 = self._shapeFns[245].jacobian(x1, x2, x3, x4)
            
            jacob251 = self._shapeFns[251].jacobian(x1, x2, x3, x4)
            jacob252 = self._shapeFns[252].jacobian(x1, x2, x3, x4)
            jacob253 = self._shapeFns[253].jacobian(x1, x2, x3, x4)
            jacob254 = self._shapeFns[254].jacobian(x1, x2, x3, x4)
            jacob255 = self._shapeFns[255].jacobian(x1, x2, x3, x4)
            
            
            jacob311 = self._shapeFns[311].jacobian(x1, x2, x3, x4)
            jacob312 = self._shapeFns[312].jacobian(x1, x2, x3, x4)
            jacob313 = self._shapeFns[313].jacobian(x1, x2, x3, x4)
            jacob314 = self._shapeFns[314].jacobian(x1, x2, x3, x4)
            jacob315 = self._shapeFns[315].jacobian(x1, x2, x3, x4)
            
            jacob321 = self._shapeFns[321].jacobian(x1, x2, x3, x4)
            jacob322 = self._shapeFns[322].jacobian(x1, x2, x3, x4)
            jacob323 = self._shapeFns[323].jacobian(x1, x2, x3, x4)
            jacob324 = self._shapeFns[324].jacobian(x1, x2, x3, x4)
            jacob325 = self._shapeFns[325].jacobian(x1, x2, x3, x4)
            
            jacob331 = self._shapeFns[331].jacobian(x1, x2, x3, x4)
            jacob332 = self._shapeFns[332].jacobian(x1, x2, x3, x4)
            jacob333 = self._shapeFns[333].jacobian(x1, x2, x3, x4)
            jacob334 = self._shapeFns[334].jacobian(x1, x2, x3, x4)
            jacob335 = self._shapeFns[335].jacobian(x1, x2, x3, x4)
            
            jacob341 = self._shapeFns[341].jacobian(x1, x2, x3, x4)
            jacob342 = self._shapeFns[342].jacobian(x1, x2, x3, x4)
            jacob343 = self._shapeFns[343].jacobian(x1, x2, x3, x4)
            jacob344 = self._shapeFns[344].jacobian(x1, x2, x3, x4)
            jacob345 = self._shapeFns[345].jacobian(x1, x2, x3, x4)
            
            jacob351 = self._shapeFns[351].jacobian(x1, x2, x3, x4)
            jacob352 = self._shapeFns[352].jacobian(x1, x2, x3, x4)
            jacob353 = self._shapeFns[353].jacobian(x1, x2, x3, x4)
            jacob354 = self._shapeFns[354].jacobian(x1, x2, x3, x4)
            jacob355 = self._shapeFns[355].jacobian(x1, x2, x3, x4)
            
            
            jacob411 = self._shapeFns[411].jacobian(x1, x2, x3, x4)
            jacob412 = self._shapeFns[412].jacobian(x1, x2, x3, x4)
            jacob413 = self._shapeFns[413].jacobian(x1, x2, x3, x4)
            jacob414 = self._shapeFns[414].jacobian(x1, x2, x3, x4)
            jacob415 = self._shapeFns[415].jacobian(x1, x2, x3, x4)
            
            jacob421 = self._shapeFns[421].jacobian(x1, x2, x3, x4)
            jacob422 = self._shapeFns[422].jacobian(x1, x2, x3, x4)
            jacob423 = self._shapeFns[423].jacobian(x1, x2, x3, x4)
            jacob424 = self._shapeFns[424].jacobian(x1, x2, x3, x4)
            jacob425 = self._shapeFns[425].jacobian(x1, x2, x3, x4)
            
            jacob431 = self._shapeFns[431].jacobian(x1, x2, x3, x4)
            jacob432 = self._shapeFns[432].jacobian(x1, x2, x3, x4)
            jacob433 = self._shapeFns[433].jacobian(x1, x2, x3, x4)
            jacob434 = self._shapeFns[434].jacobian(x1, x2, x3, x4)
            jacob435 = self._shapeFns[435].jacobian(x1, x2, x3, x4)
            
            jacob441 = self._shapeFns[441].jacobian(x1, x2, x3, x4)
            jacob442 = self._shapeFns[442].jacobian(x1, x2, x3, x4)
            jacob443 = self._shapeFns[443].jacobian(x1, x2, x3, x4)
            jacob444 = self._shapeFns[444].jacobian(x1, x2, x3, x4)
            jacob445 = self._shapeFns[445].jacobian(x1, x2, x3, x4)
            
            jacob451 = self._shapeFns[451].jacobian(x1, x2, x3, x4)
            jacob452 = self._shapeFns[452].jacobian(x1, x2, x3, x4)
            jacob453 = self._shapeFns[453].jacobian(x1, x2, x3, x4)
            jacob454 = self._shapeFns[454].jacobian(x1, x2, x3, x4)
            jacob455 = self._shapeFns[455].jacobian(x1, x2, x3, x4)
            
            
            jacob511 = self._shapeFns[511].jacobian(x1, x2, x3, x4)
            jacob512 = self._shapeFns[512].jacobian(x1, x2, x3, x4)
            jacob513 = self._shapeFns[513].jacobian(x1, x2, x3, x4)
            jacob514 = self._shapeFns[514].jacobian(x1, x2, x3, x4)
            jacob515 = self._shapeFns[515].jacobian(x1, x2, x3, x4)
            
            jacob521 = self._shapeFns[521].jacobian(x1, x2, x3, x4)
            jacob522 = self._shapeFns[522].jacobian(x1, x2, x3, x4)
            jacob523 = self._shapeFns[523].jacobian(x1, x2, x3, x4)
            jacob524 = self._shapeFns[524].jacobian(x1, x2, x3, x4)
            jacob525 = self._shapeFns[525].jacobian(x1, x2, x3, x4)
            
            jacob531 = self._shapeFns[531].jacobian(x1, x2, x3, x4)
            jacob532 = self._shapeFns[532].jacobian(x1, x2, x3, x4)
            jacob533 = self._shapeFns[533].jacobian(x1, x2, x3, x4)
            jacob534 = self._shapeFns[534].jacobian(x1, x2, x3, x4)
            jacob535 = self._shapeFns[535].jacobian(x1, x2, x3, x4)
            
            jacob541 = self._shapeFns[541].jacobian(x1, x2, x3, x4)
            jacob542 = self._shapeFns[542].jacobian(x1, x2, x3, x4)
            jacob543 = self._shapeFns[543].jacobian(x1, x2, x3, x4)
            jacob544 = self._shapeFns[544].jacobian(x1, x2, x3, x4)
            jacob545 = self._shapeFns[545].jacobian(x1, x2, x3, x4)
            
            jacob551 = self._shapeFns[551].jacobian(x1, x2, x3, x4)
            jacob552 = self._shapeFns[552].jacobian(x1, x2, x3, x4)
            jacob553 = self._shapeFns[553].jacobian(x1, x2, x3, x4)
            jacob554 = self._shapeFns[554].jacobian(x1, x2, x3, x4)
            jacob555 = self._shapeFns[555].jacobian(x1, x2, x3, x4)
            
            


            # determinant of the Jacobian matrix
            detJ111 = det(jacob111)
            detJ112 = det(jacob112)
            detJ113 = det(jacob113)
            detJ114 = det(jacob114)
            detJ115 = det(jacob115)
            
            detJ121 = det(jacob121)
            detJ122 = det(jacob122)
            detJ123 = det(jacob123)
            detJ124 = det(jacob124)
            detJ125 = det(jacob125)
            
            detJ131 = det(jacob131)
            detJ132 = det(jacob132)
            detJ133 = det(jacob133)
            detJ134 = det(jacob134)
            detJ135 = det(jacob135)
            
            detJ141 = det(jacob141)
            detJ142 = det(jacob142)
            detJ143 = det(jacob143)
            detJ144 = det(jacob144)
            detJ145 = det(jacob145)
            
            detJ151 = det(jacob151)
            detJ152 = det(jacob152)
            detJ153 = det(jacob153)
            detJ154 = det(jacob154)
            detJ155 = det(jacob155)
            
            
            detJ211 = det(jacob211)
            detJ212 = det(jacob212)
            detJ213 = det(jacob213)
            detJ214 = det(jacob214)
            detJ215 = det(jacob215)
            
            detJ221 = det(jacob221)
            detJ222 = det(jacob222)
            detJ223 = det(jacob223)
            detJ224 = det(jacob224)
            detJ225 = det(jacob225)
            
            detJ231 = det(jacob231)
            detJ232 = det(jacob232)
            detJ233 = det(jacob233)
            detJ234 = det(jacob234)
            detJ235 = det(jacob235)
            
            detJ241 = det(jacob241)
            detJ242 = det(jacob242)
            detJ243 = det(jacob243)
            detJ244 = det(jacob244)
            detJ245 = det(jacob245)
            
            detJ251 = det(jacob251)
            detJ252 = det(jacob252)
            detJ253 = det(jacob253)
            detJ254 = det(jacob254)
            detJ255 = det(jacob255)
            
            
            detJ311 = det(jacob311)
            detJ312 = det(jacob312)
            detJ313 = det(jacob313)
            detJ314 = det(jacob314)
            detJ315 = det(jacob315)
            
            detJ321 = det(jacob321)
            detJ322 = det(jacob322)
            detJ323 = det(jacob323)
            detJ324 = det(jacob324)
            detJ325 = det(jacob325)
            
            detJ331 = det(jacob331)
            detJ332 = det(jacob332)
            detJ333 = det(jacob333)
            detJ334 = det(jacob334)
            detJ335 = det(jacob335)
            
            detJ341 = det(jacob341)
            detJ342 = det(jacob342)
            detJ343 = det(jacob343)
            detJ344 = det(jacob344)
            detJ345 = det(jacob345)
            
            detJ351 = det(jacob351)
            detJ352 = det(jacob352)
            detJ353 = det(jacob353)
            detJ354 = det(jacob354)
            detJ355 = det(jacob355)
            
            
            detJ411 = det(jacob411)
            detJ412 = det(jacob412)
            detJ413 = det(jacob413)
            detJ414 = det(jacob414)
            detJ415 = det(jacob415)
            
            detJ421 = det(jacob421)
            detJ422 = det(jacob422)
            detJ423 = det(jacob423)
            detJ424 = det(jacob424)
            detJ425 = det(jacob425)
            
            detJ431 = det(jacob431)
            detJ432 = det(jacob432)
            detJ433 = det(jacob433)
            detJ434 = det(jacob434)
            detJ435 = det(jacob435)
            
            detJ441 = det(jacob441)
            detJ442 = det(jacob442)
            detJ443 = det(jacob443)
            detJ444 = det(jacob444)
            detJ445 = det(jacob445)
            
            detJ451 = det(jacob451)
            detJ452 = det(jacob452)
            detJ453 = det(jacob453)
            detJ454 = det(jacob454)
            detJ455 = det(jacob455)
            
        
            detJ511 = det(jacob511)
            detJ512 = det(jacob512)
            detJ513 = det(jacob513)
            detJ514 = det(jacob514)
            detJ515 = det(jacob515)
            
            detJ521 = det(jacob521)
            detJ522 = det(jacob522)
            detJ523 = det(jacob523)
            detJ524 = det(jacob524)
            detJ525 = det(jacob525)
            
            detJ531 = det(jacob531)
            detJ532 = det(jacob532)
            detJ533 = det(jacob533)
            detJ534 = det(jacob534)
            detJ535 = det(jacob535)
            
            detJ541 = det(jacob541)
            detJ542 = det(jacob542)
            detJ543 = det(jacob543)
            detJ544 = det(jacob544)
            detJ545 = det(jacob545)
            
            detJ551 = det(jacob551)
            detJ552 = det(jacob552)
            detJ553 = det(jacob553)
            detJ554 = det(jacob554)
            detJ555 = det(jacob555)



            nn111 = np.dot(np.transpose(self._shapeFns[111].Nmatx),
                         self._shapeFns[111].Nmatx)
            nn112 = np.dot(np.transpose(self._shapeFns[112].Nmatx),
                         self._shapeFns[112].Nmatx)
            nn113 = np.dot(np.transpose(self._shapeFns[113].Nmatx),
                         self._shapeFns[113].Nmatx)
            nn114 = np.dot(np.transpose(self._shapeFns[114].Nmatx),
                         self._shapeFns[114].Nmatx)
            nn115 = np.dot(np.transpose(self._shapeFns[115].Nmatx),
                         self._shapeFns[115].Nmatx)
            
            nn121 = np.dot(np.transpose(self._shapeFns[121].Nmatx),
                         self._shapeFns[121].Nmatx)
            nn122 = np.dot(np.transpose(self._shapeFns[122].Nmatx),
                         self._shapeFns[122].Nmatx)
            nn123 = np.dot(np.transpose(self._shapeFns[123].Nmatx),
                         self._shapeFns[123].Nmatx)
            nn124 = np.dot(np.transpose(self._shapeFns[124].Nmatx),
                         self._shapeFns[124].Nmatx)
            nn125 = np.dot(np.transpose(self._shapeFns[125].Nmatx),
                         self._shapeFns[125].Nmatx)
            
            nn131 = np.dot(np.transpose(self._shapeFns[131].Nmatx),
                         self._shapeFns[131].Nmatx)
            nn132 = np.dot(np.transpose(self._shapeFns[132].Nmatx),
                         self._shapeFns[132].Nmatx)
            nn133 = np.dot(np.transpose(self._shapeFns[133].Nmatx),
                         self._shapeFns[133].Nmatx)
            nn134 = np.dot(np.transpose(self._shapeFns[134].Nmatx),
                         self._shapeFns[134].Nmatx)
            nn135 = np.dot(np.transpose(self._shapeFns[135].Nmatx),
                         self._shapeFns[135].Nmatx)
            
            nn141 = np.dot(np.transpose(self._shapeFns[141].Nmatx),
                         self._shapeFns[141].Nmatx)
            nn142 = np.dot(np.transpose(self._shapeFns[142].Nmatx),
                         self._shapeFns[142].Nmatx)
            nn143 = np.dot(np.transpose(self._shapeFns[143].Nmatx),
                         self._shapeFns[143].Nmatx)
            nn144 = np.dot(np.transpose(self._shapeFns[144].Nmatx),
                         self._shapeFns[144].Nmatx)
            nn145 = np.dot(np.transpose(self._shapeFns[145].Nmatx),
                         self._shapeFns[145].Nmatx)
            
            nn151 = np.dot(np.transpose(self._shapeFns[151].Nmatx),
                         self._shapeFns[151].Nmatx)
            nn152 = np.dot(np.transpose(self._shapeFns[152].Nmatx),
                         self._shapeFns[152].Nmatx)
            nn153 = np.dot(np.transpose(self._shapeFns[153].Nmatx),
                         self._shapeFns[153].Nmatx)
            nn154 = np.dot(np.transpose(self._shapeFns[154].Nmatx),
                         self._shapeFns[154].Nmatx)
            nn155 = np.dot(np.transpose(self._shapeFns[155].Nmatx),
                         self._shapeFns[155].Nmatx)
            
            
            
            nn211 = np.dot(np.transpose(self._shapeFns[211].Nmatx),
                         self._shapeFns[211].Nmatx)
            nn212 = np.dot(np.transpose(self._shapeFns[212].Nmatx),
                         self._shapeFns[212].Nmatx)
            nn213 = np.dot(np.transpose(self._shapeFns[213].Nmatx),
                         self._shapeFns[213].Nmatx)
            nn214 = np.dot(np.transpose(self._shapeFns[214].Nmatx),
                         self._shapeFns[214].Nmatx)
            nn215 = np.dot(np.transpose(self._shapeFns[215].Nmatx),
                         self._shapeFns[215].Nmatx)
            
            nn221 = np.dot(np.transpose(self._shapeFns[221].Nmatx),
                         self._shapeFns[221].Nmatx)
            nn222 = np.dot(np.transpose(self._shapeFns[222].Nmatx),
                         self._shapeFns[222].Nmatx)
            nn223 = np.dot(np.transpose(self._shapeFns[223].Nmatx),
                         self._shapeFns[223].Nmatx)
            nn224 = np.dot(np.transpose(self._shapeFns[224].Nmatx),
                         self._shapeFns[224].Nmatx)
            nn225 = np.dot(np.transpose(self._shapeFns[225].Nmatx),
                         self._shapeFns[225].Nmatx)
            
            nn231 = np.dot(np.transpose(self._shapeFns[231].Nmatx),
                         self._shapeFns[231].Nmatx)
            nn232 = np.dot(np.transpose(self._shapeFns[232].Nmatx),
                         self._shapeFns[232].Nmatx)
            nn233 = np.dot(np.transpose(self._shapeFns[233].Nmatx),
                         self._shapeFns[233].Nmatx)
            nn234 = np.dot(np.transpose(self._shapeFns[234].Nmatx),
                         self._shapeFns[234].Nmatx)
            nn235 = np.dot(np.transpose(self._shapeFns[235].Nmatx),
                         self._shapeFns[235].Nmatx)
            
            nn241 = np.dot(np.transpose(self._shapeFns[241].Nmatx),
                         self._shapeFns[241].Nmatx)
            nn242 = np.dot(np.transpose(self._shapeFns[242].Nmatx),
                         self._shapeFns[242].Nmatx)
            nn243 = np.dot(np.transpose(self._shapeFns[243].Nmatx),
                         self._shapeFns[243].Nmatx)
            nn244 = np.dot(np.transpose(self._shapeFns[244].Nmatx),
                         self._shapeFns[244].Nmatx)
            nn245 = np.dot(np.transpose(self._shapeFns[245].Nmatx),
                         self._shapeFns[245].Nmatx)
            
            nn251 = np.dot(np.transpose(self._shapeFns[251].Nmatx),
                         self._shapeFns[251].Nmatx)
            nn252 = np.dot(np.transpose(self._shapeFns[252].Nmatx),
                         self._shapeFns[252].Nmatx)
            nn253 = np.dot(np.transpose(self._shapeFns[253].Nmatx),
                         self._shapeFns[253].Nmatx)
            nn254 = np.dot(np.transpose(self._shapeFns[254].Nmatx),
                         self._shapeFns[254].Nmatx)
            nn255 = np.dot(np.transpose(self._shapeFns[255].Nmatx),
                         self._shapeFns[255].Nmatx)
            
            
            nn311 = np.dot(np.transpose(self._shapeFns[311].Nmatx),
                         self._shapeFns[311].Nmatx)
            nn312 = np.dot(np.transpose(self._shapeFns[312].Nmatx),
                         self._shapeFns[312].Nmatx)
            nn313 = np.dot(np.transpose(self._shapeFns[313].Nmatx),
                         self._shapeFns[313].Nmatx)
            nn314 = np.dot(np.transpose(self._shapeFns[314].Nmatx),
                         self._shapeFns[314].Nmatx)
            nn315 = np.dot(np.transpose(self._shapeFns[315].Nmatx),
                         self._shapeFns[315].Nmatx)
            
            nn321 = np.dot(np.transpose(self._shapeFns[321].Nmatx),
                         self._shapeFns[321].Nmatx)
            nn322 = np.dot(np.transpose(self._shapeFns[322].Nmatx),
                         self._shapeFns[322].Nmatx)
            nn323 = np.dot(np.transpose(self._shapeFns[323].Nmatx),
                         self._shapeFns[323].Nmatx)
            nn324 = np.dot(np.transpose(self._shapeFns[324].Nmatx),
                         self._shapeFns[324].Nmatx)
            nn325 = np.dot(np.transpose(self._shapeFns[325].Nmatx),
                         self._shapeFns[325].Nmatx)
            
            nn331 = np.dot(np.transpose(self._shapeFns[331].Nmatx),
                         self._shapeFns[331].Nmatx)
            nn332 = np.dot(np.transpose(self._shapeFns[332].Nmatx),
                         self._shapeFns[332].Nmatx)
            nn333 = np.dot(np.transpose(self._shapeFns[333].Nmatx),
                         self._shapeFns[333].Nmatx)
            nn334 = np.dot(np.transpose(self._shapeFns[334].Nmatx),
                         self._shapeFns[334].Nmatx)
            nn335 = np.dot(np.transpose(self._shapeFns[335].Nmatx),
                         self._shapeFns[335].Nmatx)
            
            nn341 = np.dot(np.transpose(self._shapeFns[341].Nmatx),
                         self._shapeFns[341].Nmatx)
            nn342 = np.dot(np.transpose(self._shapeFns[342].Nmatx),
                         self._shapeFns[342].Nmatx)
            nn343 = np.dot(np.transpose(self._shapeFns[343].Nmatx),
                         self._shapeFns[343].Nmatx)
            nn344 = np.dot(np.transpose(self._shapeFns[344].Nmatx),
                         self._shapeFns[344].Nmatx)
            nn345 = np.dot(np.transpose(self._shapeFns[345].Nmatx),
                         self._shapeFns[345].Nmatx)
            
            nn351 = np.dot(np.transpose(self._shapeFns[351].Nmatx),
                         self._shapeFns[351].Nmatx)
            nn352 = np.dot(np.transpose(self._shapeFns[352].Nmatx),
                         self._shapeFns[352].Nmatx)
            nn353 = np.dot(np.transpose(self._shapeFns[353].Nmatx),
                         self._shapeFns[353].Nmatx)
            nn354 = np.dot(np.transpose(self._shapeFns[354].Nmatx),
                         self._shapeFns[354].Nmatx)
            nn355 = np.dot(np.transpose(self._shapeFns[355].Nmatx),
                         self._shapeFns[355].Nmatx)
            
            
            nn411 = np.dot(np.transpose(self._shapeFns[411].Nmatx),
                         self._shapeFns[411].Nmatx)
            nn412 = np.dot(np.transpose(self._shapeFns[412].Nmatx),
                         self._shapeFns[412].Nmatx)
            nn413 = np.dot(np.transpose(self._shapeFns[413].Nmatx),
                         self._shapeFns[413].Nmatx)
            nn414 = np.dot(np.transpose(self._shapeFns[414].Nmatx),
                         self._shapeFns[414].Nmatx)
            nn415 = np.dot(np.transpose(self._shapeFns[415].Nmatx),
                         self._shapeFns[415].Nmatx)
            
            nn421 = np.dot(np.transpose(self._shapeFns[421].Nmatx),
                         self._shapeFns[421].Nmatx)
            nn422 = np.dot(np.transpose(self._shapeFns[422].Nmatx),
                         self._shapeFns[422].Nmatx)
            nn423 = np.dot(np.transpose(self._shapeFns[423].Nmatx),
                         self._shapeFns[423].Nmatx)
            nn424 = np.dot(np.transpose(self._shapeFns[424].Nmatx),
                         self._shapeFns[424].Nmatx)
            nn425 = np.dot(np.transpose(self._shapeFns[425].Nmatx),
                         self._shapeFns[425].Nmatx)
            
            nn431 = np.dot(np.transpose(self._shapeFns[431].Nmatx),
                         self._shapeFns[431].Nmatx)
            nn432 = np.dot(np.transpose(self._shapeFns[432].Nmatx),
                         self._shapeFns[432].Nmatx)
            nn433 = np.dot(np.transpose(self._shapeFns[433].Nmatx),
                         self._shapeFns[433].Nmatx)
            nn434 = np.dot(np.transpose(self._shapeFns[434].Nmatx),
                         self._shapeFns[434].Nmatx)
            nn435 = np.dot(np.transpose(self._shapeFns[435].Nmatx),
                         self._shapeFns[435].Nmatx)
            
            nn441 = np.dot(np.transpose(self._shapeFns[441].Nmatx),
                         self._shapeFns[441].Nmatx)
            nn442 = np.dot(np.transpose(self._shapeFns[442].Nmatx),
                         self._shapeFns[442].Nmatx)
            nn443 = np.dot(np.transpose(self._shapeFns[443].Nmatx),
                         self._shapeFns[443].Nmatx)
            nn444 = np.dot(np.transpose(self._shapeFns[444].Nmatx),
                         self._shapeFns[444].Nmatx)
            nn445 = np.dot(np.transpose(self._shapeFns[445].Nmatx),
                         self._shapeFns[445].Nmatx)
            
            nn451 = np.dot(np.transpose(self._shapeFns[451].Nmatx),
                         self._shapeFns[451].Nmatx)
            nn452 = np.dot(np.transpose(self._shapeFns[452].Nmatx),
                         self._shapeFns[452].Nmatx)
            nn453 = np.dot(np.transpose(self._shapeFns[453].Nmatx),
                         self._shapeFns[453].Nmatx)
            nn454 = np.dot(np.transpose(self._shapeFns[454].Nmatx),
                         self._shapeFns[454].Nmatx)
            nn455 = np.dot(np.transpose(self._shapeFns[455].Nmatx),
                         self._shapeFns[455].Nmatx)
            
            
            nn511 = np.dot(np.transpose(self._shapeFns[511].Nmatx),
                         self._shapeFns[511].Nmatx)
            nn512 = np.dot(np.transpose(self._shapeFns[512].Nmatx),
                         self._shapeFns[512].Nmatx)
            nn513 = np.dot(np.transpose(self._shapeFns[513].Nmatx),
                         self._shapeFns[513].Nmatx)
            nn514 = np.dot(np.transpose(self._shapeFns[514].Nmatx),
                         self._shapeFns[514].Nmatx)
            nn515 = np.dot(np.transpose(self._shapeFns[515].Nmatx),
                         self._shapeFns[515].Nmatx)
            
            nn521 = np.dot(np.transpose(self._shapeFns[521].Nmatx),
                         self._shapeFns[521].Nmatx)
            nn522 = np.dot(np.transpose(self._shapeFns[522].Nmatx),
                         self._shapeFns[522].Nmatx)
            nn523 = np.dot(np.transpose(self._shapeFns[523].Nmatx),
                         self._shapeFns[523].Nmatx)
            nn524 = np.dot(np.transpose(self._shapeFns[524].Nmatx),
                         self._shapeFns[524].Nmatx)
            nn525 = np.dot(np.transpose(self._shapeFns[525].Nmatx),
                         self._shapeFns[525].Nmatx)
            
            nn531 = np.dot(np.transpose(self._shapeFns[531].Nmatx),
                         self._shapeFns[531].Nmatx)
            nn532 = np.dot(np.transpose(self._shapeFns[532].Nmatx),
                         self._shapeFns[532].Nmatx)
            nn533 = np.dot(np.transpose(self._shapeFns[533].Nmatx),
                         self._shapeFns[533].Nmatx)
            nn534 = np.dot(np.transpose(self._shapeFns[534].Nmatx),
                         self._shapeFns[534].Nmatx)
            nn535 = np.dot(np.transpose(self._shapeFns[535].Nmatx),
                         self._shapeFns[535].Nmatx)
            
            nn541 = np.dot(np.transpose(self._shapeFns[541].Nmatx),
                         self._shapeFns[541].Nmatx)
            nn542 = np.dot(np.transpose(self._shapeFns[542].Nmatx),
                         self._shapeFns[542].Nmatx)
            nn543 = np.dot(np.transpose(self._shapeFns[543].Nmatx),
                         self._shapeFns[543].Nmatx)
            nn544 = np.dot(np.transpose(self._shapeFns[544].Nmatx),
                         self._shapeFns[544].Nmatx)
            nn545 = np.dot(np.transpose(self._shapeFns[545].Nmatx),
                         self._shapeFns[545].Nmatx)
            
            nn551 = np.dot(np.transpose(self._shapeFns[551].Nmatx),
                         self._shapeFns[551].Nmatx)
            nn552 = np.dot(np.transpose(self._shapeFns[552].Nmatx),
                         self._shapeFns[552].Nmatx)
            nn553 = np.dot(np.transpose(self._shapeFns[553].Nmatx),
                         self._shapeFns[553].Nmatx)
            nn554 = np.dot(np.transpose(self._shapeFns[554].Nmatx),
                         self._shapeFns[554].Nmatx)
            nn555 = np.dot(np.transpose(self._shapeFns[555].Nmatx),
                         self._shapeFns[555].Nmatx)



            # the consistent mass Matrix for 5 Gauss points
            massC = (self._rho * (detJ111 * wel[0] * wel[0] * wel[0] * nn111 +
                                  detJ112 * wel[0] * wel[0] * wel[1] * nn112 +
                                  detJ113 * wel[0] * wel[0] * wel[2] * nn113 + 
                                  detJ114 * wel[0] * wel[0] * wel[3] * nn114 +
                                  detJ115 * wel[0] * wel[0] * wel[4] * nn115 +
                                  detJ121 * wel[0] * wel[1] * wel[0] * nn121 +
                                  detJ122 * wel[0] * wel[1] * wel[1] * nn122 +
                                  detJ123 * wel[0] * wel[1] * wel[2] * nn123 + 
                                  detJ124 * wel[0] * wel[1] * wel[3] * nn124 +
                                  detJ125 * wel[0] * wel[1] * wel[4] * nn125 +
                                  detJ131 * wel[0] * wel[2] * wel[0] * nn131 +
                                  detJ132 * wel[0] * wel[2] * wel[1] * nn132 +
                                  detJ133 * wel[0] * wel[2] * wel[2] * nn133 + 
                                  detJ134 * wel[0] * wel[2] * wel[3] * nn134 +
                                  detJ135 * wel[0] * wel[2] * wel[4] * nn135 +
                                  detJ141 * wel[0] * wel[3] * wel[0] * nn141 +
                                  detJ142 * wel[0] * wel[3] * wel[1] * nn142 +
                                  detJ143 * wel[0] * wel[3] * wel[2] * nn143 + 
                                  detJ144 * wel[0] * wel[3] * wel[3] * nn144 +
                                  detJ145 * wel[0] * wel[3] * wel[4] * nn145 +
                                  detJ151 * wel[0] * wel[4] * wel[0] * nn151 +
                                  detJ152 * wel[0] * wel[4] * wel[1] * nn152 +
                                  detJ153 * wel[0] * wel[4] * wel[2] * nn153 + 
                                  detJ154 * wel[0] * wel[4] * wel[3] * nn154 +
                                  detJ155 * wel[0] * wel[4] * wel[4] * nn155 +                                  
                                  detJ211 * wel[1] * wel[0] * wel[0] * nn211 +
                                  detJ212 * wel[1] * wel[0] * wel[1] * nn212 +
                                  detJ213 * wel[1] * wel[0] * wel[2] * nn213 + 
                                  detJ214 * wel[1] * wel[0] * wel[3] * nn214 +
                                  detJ215 * wel[1] * wel[0] * wel[4] * nn215 +
                                  detJ221 * wel[1] * wel[1] * wel[0] * nn221 +
                                  detJ222 * wel[1] * wel[1] * wel[1] * nn222 +
                                  detJ223 * wel[1] * wel[1] * wel[2] * nn223 + 
                                  detJ224 * wel[1] * wel[1] * wel[3] * nn224 +
                                  detJ225 * wel[1] * wel[1] * wel[4] * nn225 +
                                  detJ231 * wel[1] * wel[2] * wel[0] * nn231 +
                                  detJ232 * wel[1] * wel[2] * wel[1] * nn232 +
                                  detJ233 * wel[1] * wel[2] * wel[2] * nn233 + 
                                  detJ234 * wel[1] * wel[2] * wel[3] * nn234 +
                                  detJ235 * wel[1] * wel[2] * wel[4] * nn235 +
                                  detJ241 * wel[1] * wel[3] * wel[0] * nn241 +
                                  detJ242 * wel[1] * wel[3] * wel[1] * nn242 +
                                  detJ243 * wel[1] * wel[3] * wel[2] * nn243 + 
                                  detJ244 * wel[1] * wel[3] * wel[3] * nn244 +
                                  detJ245 * wel[1] * wel[3] * wel[4] * nn245 +
                                  detJ251 * wel[1] * wel[4] * wel[0] * nn251 +
                                  detJ252 * wel[1] * wel[4] * wel[1] * nn252 +
                                  detJ253 * wel[1] * wel[4] * wel[2] * nn253 + 
                                  detJ254 * wel[1] * wel[4] * wel[3] * nn254 +
                                  detJ255 * wel[1] * wel[4] * wel[4] * nn255 +                                  
                                  detJ311 * wel[2] * wel[0] * wel[0] * nn311 +
                                  detJ312 * wel[2] * wel[0] * wel[1] * nn312 +
                                  detJ313 * wel[2] * wel[0] * wel[2] * nn313 + 
                                  detJ314 * wel[2] * wel[0] * wel[3] * nn314 +
                                  detJ315 * wel[2] * wel[0] * wel[4] * nn315 +
                                  detJ321 * wel[2] * wel[1] * wel[0] * nn321 +
                                  detJ322 * wel[2] * wel[1] * wel[1] * nn322 +
                                  detJ323 * wel[2] * wel[1] * wel[2] * nn323 + 
                                  detJ324 * wel[2] * wel[1] * wel[3] * nn324 +
                                  detJ325 * wel[2] * wel[1] * wel[4] * nn325 +
                                  detJ331 * wel[2] * wel[2] * wel[0] * nn331 +
                                  detJ332 * wel[2] * wel[2] * wel[1] * nn332 +
                                  detJ333 * wel[2] * wel[2] * wel[2] * nn333 + 
                                  detJ334 * wel[2] * wel[2] * wel[3] * nn334 +
                                  detJ335 * wel[2] * wel[2] * wel[4] * nn335 +
                                  detJ341 * wel[2] * wel[3] * wel[0] * nn341 +
                                  detJ342 * wel[2] * wel[3] * wel[1] * nn342 +
                                  detJ343 * wel[2] * wel[3] * wel[2] * nn343 + 
                                  detJ344 * wel[2] * wel[3] * wel[3] * nn344 +
                                  detJ345 * wel[2] * wel[3] * wel[4] * nn345 +
                                  detJ351 * wel[2] * wel[4] * wel[0] * nn351 +
                                  detJ352 * wel[2] * wel[4] * wel[1] * nn352 +
                                  detJ353 * wel[2] * wel[4] * wel[2] * nn353 + 
                                  detJ354 * wel[2] * wel[4] * wel[3] * nn354 +
                                  detJ355 * wel[2] * wel[4] * wel[4] * nn355 +                                  
                                  detJ411 * wel[3] * wel[0] * wel[0] * nn411 +
                                  detJ412 * wel[3] * wel[0] * wel[1] * nn412 +
                                  detJ413 * wel[3] * wel[0] * wel[2] * nn413 + 
                                  detJ414 * wel[3] * wel[0] * wel[3] * nn414 +
                                  detJ415 * wel[3] * wel[0] * wel[4] * nn415 +
                                  detJ421 * wel[3] * wel[1] * wel[0] * nn421 +
                                  detJ422 * wel[3] * wel[1] * wel[1] * nn422 +
                                  detJ423 * wel[3] * wel[1] * wel[2] * nn423 + 
                                  detJ424 * wel[3] * wel[1] * wel[3] * nn424 +
                                  detJ425 * wel[3] * wel[1] * wel[4] * nn425 +
                                  detJ431 * wel[3] * wel[2] * wel[0] * nn431 +
                                  detJ432 * wel[3] * wel[2] * wel[1] * nn432 +
                                  detJ433 * wel[3] * wel[2] * wel[2] * nn433 + 
                                  detJ434 * wel[3] * wel[2] * wel[3] * nn434 +
                                  detJ435 * wel[3] * wel[2] * wel[4] * nn435 +
                                  detJ441 * wel[3] * wel[3] * wel[0] * nn441 +
                                  detJ442 * wel[3] * wel[3] * wel[1] * nn442 +
                                  detJ443 * wel[3] * wel[3] * wel[2] * nn443 + 
                                  detJ444 * wel[3] * wel[3] * wel[3] * nn444 +
                                  detJ445 * wel[3] * wel[3] * wel[4] * nn445 +
                                  detJ451 * wel[3] * wel[4] * wel[0] * nn451 +
                                  detJ452 * wel[3] * wel[4] * wel[1] * nn452 +
                                  detJ453 * wel[3] * wel[4] * wel[2] * nn453 + 
                                  detJ454 * wel[3] * wel[4] * wel[3] * nn454 +
                                  detJ455 * wel[3] * wel[4] * wel[4] * nn455 +                                
                                  detJ511 * wel[4] * wel[0] * wel[0] * nn511 +
                                  detJ512 * wel[4] * wel[0] * wel[1] * nn512 +
                                  detJ513 * wel[4] * wel[0] * wel[2] * nn513 + 
                                  detJ514 * wel[4] * wel[0] * wel[3] * nn514 +
                                  detJ515 * wel[4] * wel[0] * wel[4] * nn515 +
                                  detJ521 * wel[4] * wel[1] * wel[0] * nn521 +
                                  detJ522 * wel[4] * wel[1] * wel[1] * nn522 +
                                  detJ523 * wel[4] * wel[1] * wel[2] * nn523 + 
                                  detJ524 * wel[4] * wel[1] * wel[3] * nn524 +
                                  detJ525 * wel[4] * wel[1] * wel[4] * nn525 +
                                  detJ531 * wel[4] * wel[2] * wel[0] * nn531 +
                                  detJ532 * wel[4] * wel[2] * wel[1] * nn532 +
                                  detJ533 * wel[4] * wel[2] * wel[2] * nn533 + 
                                  detJ534 * wel[4] * wel[2] * wel[3] * nn534 +
                                  detJ535 * wel[4] * wel[2] * wel[4] * nn535 +
                                  detJ541 * wel[4] * wel[3] * wel[0] * nn541 +
                                  detJ542 * wel[4] * wel[3] * wel[1] * nn542 +
                                  detJ543 * wel[4] * wel[3] * wel[2] * nn543 + 
                                  detJ544 * wel[4] * wel[3] * wel[3] * nn544 +
                                  detJ545 * wel[4] * wel[3] * wel[4] * nn545 +
                                  detJ551 * wel[4] * wel[4] * wel[0] * nn551 +
                                  detJ552 * wel[4] * wel[4] * wel[1] * nn552 +
                                  detJ553 * wel[4] * wel[4] * wel[2] * nn553 + 
                                  detJ554 * wel[4] * wel[4] * wel[3] * nn554 +
                                  detJ555 * wel[4] * wel[4] * wel[4] * nn555 ))

            # the lumped mass matrix for 5 Gauss points
            massL = np.zeros((12, 12), dtype=float)
            row, col = np.diag_indices_from(massC)
            massL[row, col] = massC.sum(axis=1)

            # the mass matrix is the average of the above two mass matrices
            mass = 0.5 * (massC + massL)
        return mass

    def stiffnessMatrix(self, M, p, s1, s2, t1, t2, t3):
        # assign coordinates at the vertices in the reference configuration
        x1 = (self._x1, self._y1, self._z1)
        x2 = (self._x2, self._y2, self._z2)
        x3 = (self._x3, self._y3, self._z3)
        x4 = (self._x4, self._y4, self._z4)

        # coordinates for the reference vertices as tuples
        x01 = (self._x10, self._y10, self._z10)
        x02 = (self._x20, self._y20, self._z20)
        x03 = (self._x30, self._y30, self._z30)
        x04 = (self._x40, self._y40, self._z40)
        
        # create the stress matrix
        T = np.zeros((3, 3), dtype=float)
        T[0, 0] = p
        T[0, 1] = t1
        T[0, 2] = t3
        T[1, 0] = t1
        T[1, 1] = s1
        T[1, 2] = t2
        T[2, 0] = t3
        T[2, 1] = t2
        T[2, 2] = s2
                    
        # determine the stiffness matrix
        if self._gaussPts == 1:
            # 'natural' weight of the element
            wgt = 1.0 / 6.0
            w = np.array([wgt])
            
            jacob111 = self._shapeFns[111].jacobian(x1, x2, x3, x4)
            
            # determinant of the Jacobian matrix
            detJ = det(jacob111)
            
            # create the linear Bmatrix
            BL = self._shapeFns[111].BLinear(x1, x2, x3, x4)
            # the linear stiffness matrix for 1 Gauss point
            KL = detJ * w[0] * w[0] * w[0] * BL.T.dot(M).dot(BL)
            
            # create the nonlinear Bmatrix
            BNF = self._shapeFns[111].FirstBNonLinear(x1, x2, x3, x4, 
                                                      x01, x02, x03, x04)
            BNS = self._shapeFns[111].SecondBNonLinear(x1, x2, x3, x4, 
                                                      x01, x02, x03, x04)
            BNT = self._shapeFns[111].ThirdBNonLinear(x1, x2, x3, x4, 
                                                      x01, x02, x03, x04)
            # total nonlinear Bmatrix
            BN = BNF + BNS + BNT
            
            # creat the H1 matrix
            HF = self._shapeFns[111].HmatrixF(x1, x2, x3, x4)
            # creat the H2 matrix
            HS = self._shapeFns[111].HmatrixS(x1, x2, x3, x4)
            # creat the H3 matrix
            HT = self._shapeFns[111].HmatrixT(x1, x2, x3, x4)
            
            # the nonlinear stiffness matrix for 1 Gauss point
            KN = (detJ * w[0] * w[0] * w[0] * (BL.T.dot(M).dot(BN) +
                  BN.T.dot(M).dot(BL) + BN.T.dot(M).dot(BN)))
            
            # create the stress stiffness matrix            
            KS = (detJ * w[0] * w[0] * w[0] * HF.T.dot(T).dot(HF) +
                  detJ * w[0] * w[0] * w[0] * HS.T.dot(T).dot(HS) +
                  detJ * w[0] * w[0] * w[0] * HT.T.dot(T).dot(HT))
            
            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS
            
        elif self._gaussPts == 4:
            # 'natural' weights of the element
            wgt = 1.0 / 24.0
            w = np.array([wgt, wgt, wgt, wgt])

            jacob111 = self._shapeFns[111].jacobian(x1, x2, x3, x4)
            jacob112 = self._shapeFns[112].jacobian(x1, x2, x3, x4)
            jacob113 = self._shapeFns[113].jacobian(x1, x2, x3, x4)
            jacob114 = self._shapeFns[114].jacobian(x1, x2, x3, x4)
            
            jacob121 = self._shapeFns[121].jacobian(x1, x2, x3, x4)
            jacob122 = self._shapeFns[122].jacobian(x1, x2, x3, x4)
            jacob123 = self._shapeFns[123].jacobian(x1, x2, x3, x4)
            jacob124 = self._shapeFns[124].jacobian(x1, x2, x3, x4)
            
            jacob131 = self._shapeFns[131].jacobian(x1, x2, x3, x4)
            jacob132 = self._shapeFns[132].jacobian(x1, x2, x3, x4)
            jacob133 = self._shapeFns[133].jacobian(x1, x2, x3, x4)
            jacob134 = self._shapeFns[134].jacobian(x1, x2, x3, x4)
            
            jacob141 = self._shapeFns[141].jacobian(x1, x2, x3, x4)
            jacob142 = self._shapeFns[142].jacobian(x1, x2, x3, x4)
            jacob143 = self._shapeFns[143].jacobian(x1, x2, x3, x4)
            jacob144 = self._shapeFns[144].jacobian(x1, x2, x3, x4)
            
            
            jacob211 = self._shapeFns[211].jacobian(x1, x2, x3, x4)
            jacob212 = self._shapeFns[212].jacobian(x1, x2, x3, x4)
            jacob213 = self._shapeFns[213].jacobian(x1, x2, x3, x4)
            jacob214 = self._shapeFns[214].jacobian(x1, x2, x3, x4)
            
            jacob221 = self._shapeFns[221].jacobian(x1, x2, x3, x4)
            jacob222 = self._shapeFns[222].jacobian(x1, x2, x3, x4)
            jacob223 = self._shapeFns[223].jacobian(x1, x2, x3, x4)
            jacob224 = self._shapeFns[224].jacobian(x1, x2, x3, x4)
            
            jacob231 = self._shapeFns[231].jacobian(x1, x2, x3, x4)
            jacob232 = self._shapeFns[232].jacobian(x1, x2, x3, x4)
            jacob233 = self._shapeFns[233].jacobian(x1, x2, x3, x4)
            jacob234 = self._shapeFns[234].jacobian(x1, x2, x3, x4)
            
            jacob241 = self._shapeFns[241].jacobian(x1, x2, x3, x4)
            jacob242 = self._shapeFns[242].jacobian(x1, x2, x3, x4)
            jacob243 = self._shapeFns[243].jacobian(x1, x2, x3, x4)
            jacob244 = self._shapeFns[244].jacobian(x1, x2, x3, x4)
            
            
            jacob311 = self._shapeFns[311].jacobian(x1, x2, x3, x4)
            jacob312 = self._shapeFns[312].jacobian(x1, x2, x3, x4)
            jacob313 = self._shapeFns[313].jacobian(x1, x2, x3, x4)
            jacob314 = self._shapeFns[314].jacobian(x1, x2, x3, x4)
            
            jacob321 = self._shapeFns[321].jacobian(x1, x2, x3, x4)
            jacob322 = self._shapeFns[322].jacobian(x1, x2, x3, x4)
            jacob323 = self._shapeFns[323].jacobian(x1, x2, x3, x4)
            jacob324 = self._shapeFns[324].jacobian(x1, x2, x3, x4)
            
            jacob331 = self._shapeFns[331].jacobian(x1, x2, x3, x4)
            jacob332 = self._shapeFns[332].jacobian(x1, x2, x3, x4)
            jacob333 = self._shapeFns[333].jacobian(x1, x2, x3, x4)
            jacob334 = self._shapeFns[334].jacobian(x1, x2, x3, x4)
            
            jacob341 = self._shapeFns[341].jacobian(x1, x2, x3, x4)
            jacob342 = self._shapeFns[342].jacobian(x1, x2, x3, x4)
            jacob343 = self._shapeFns[343].jacobian(x1, x2, x3, x4)
            jacob344 = self._shapeFns[344].jacobian(x1, x2, x3, x4)
            
            
            jacob411 = self._shapeFns[411].jacobian(x1, x2, x3, x4)
            jacob412 = self._shapeFns[412].jacobian(x1, x2, x3, x4)
            jacob413 = self._shapeFns[413].jacobian(x1, x2, x3, x4)
            jacob414 = self._shapeFns[414].jacobian(x1, x2, x3, x4)
            
            jacob421 = self._shapeFns[421].jacobian(x1, x2, x3, x4)
            jacob422 = self._shapeFns[422].jacobian(x1, x2, x3, x4)
            jacob423 = self._shapeFns[423].jacobian(x1, x2, x3, x4)
            jacob424 = self._shapeFns[424].jacobian(x1, x2, x3, x4)
            
            jacob431 = self._shapeFns[431].jacobian(x1, x2, x3, x4)
            jacob432 = self._shapeFns[432].jacobian(x1, x2, x3, x4)
            jacob433 = self._shapeFns[433].jacobian(x1, x2, x3, x4)
            jacob434 = self._shapeFns[434].jacobian(x1, x2, x3, x4)
            
            jacob441 = self._shapeFns[441].jacobian(x1, x2, x3, x4)
            jacob442 = self._shapeFns[442].jacobian(x1, x2, x3, x4)
            jacob443 = self._shapeFns[443].jacobian(x1, x2, x3, x4)
            jacob444 = self._shapeFns[444].jacobian(x1, x2, x3, x4)

            # determinant of the Jacobian matrix
            detJ111 = det(jacob111)
            detJ112 = det(jacob112)
            detJ113 = det(jacob113)
            detJ114 = det(jacob114)
            
            detJ121 = det(jacob121)
            detJ122 = det(jacob122)
            detJ123 = det(jacob123)
            detJ124 = det(jacob124)
            
            detJ131 = det(jacob131)
            detJ132 = det(jacob132)
            detJ133 = det(jacob133)
            detJ134 = det(jacob134)
            
            detJ141 = det(jacob141)
            detJ142 = det(jacob142)
            detJ143 = det(jacob143)
            detJ144 = det(jacob144)
            
            
            detJ211 = det(jacob211)
            detJ212 = det(jacob212)
            detJ213 = det(jacob213)
            detJ214 = det(jacob214)
            
            detJ221 = det(jacob221)
            detJ222 = det(jacob222)
            detJ223 = det(jacob223)
            detJ224 = det(jacob224)
            
            detJ231 = det(jacob231)
            detJ232 = det(jacob232)
            detJ233 = det(jacob233)
            detJ234 = det(jacob234)
            
            detJ241 = det(jacob241)
            detJ242 = det(jacob242)
            detJ243 = det(jacob243)
            detJ244 = det(jacob244)
            
            
            detJ311 = det(jacob311)
            detJ312 = det(jacob312)
            detJ313 = det(jacob313)
            detJ314 = det(jacob314)
            
            detJ321 = det(jacob321)
            detJ322 = det(jacob322)
            detJ323 = det(jacob323)
            detJ324 = det(jacob324)
            
            detJ331 = det(jacob331)
            detJ332 = det(jacob332)
            detJ333 = det(jacob333)
            detJ334 = det(jacob334)
            
            detJ341 = det(jacob341)
            detJ342 = det(jacob342)
            detJ343 = det(jacob343)
            detJ344 = det(jacob344)
            
            
            detJ411 = det(jacob411)
            detJ412 = det(jacob412)
            detJ413 = det(jacob413)
            detJ414 = det(jacob414)
            
            detJ421 = det(jacob421)
            detJ422 = det(jacob422)
            detJ423 = det(jacob423)
            detJ424 = det(jacob424)
            
            detJ431 = det(jacob431)
            detJ432 = det(jacob432)
            detJ433 = det(jacob433)
            detJ434 = det(jacob434)
            
            detJ441 = det(jacob441)
            detJ442 = det(jacob442)
            detJ443 = det(jacob443)
            detJ444 = det(jacob444)

            # create the linear Bmatrix
            BL111 = self._shapeFns[111].BLinear(x1, x2, x3, x4) 
            BL112 = self._shapeFns[112].BLinear(x1, x2, x3, x4) 
            BL113 = self._shapeFns[113].BLinear(x1, x2, x3, x4)
            BL114 = self._shapeFns[114].BLinear(x1, x2, x3, x4) 
            
            BL121 = self._shapeFns[121].BLinear(x1, x2, x3, x4)
            BL122 = self._shapeFns[122].BLinear(x1, x2, x3, x4) 
            BL123 = self._shapeFns[123].BLinear(x1, x2, x3, x4) 
            BL124 = self._shapeFns[124].BLinear(x1, x2, x3, x4) 
            
            BL131 = self._shapeFns[131].BLinear(x1, x2, x3, x4) 
            BL132 = self._shapeFns[132].BLinear(x1, x2, x3, x4) 
            BL133 = self._shapeFns[133].BLinear(x1, x2, x3, x4) 
            BL134 = self._shapeFns[134].BLinear(x1, x2, x3, x4) 
            
            BL141 = self._shapeFns[141].BLinear(x1, x2, x3, x4)
            BL142 = self._shapeFns[142].BLinear(x1, x2, x3, x4) 
            BL143 = self._shapeFns[143].BLinear(x1, x2, x3, x4) 
            BL144 = self._shapeFns[144].BLinear(x1, x2, x3, x4) 
            
            
            BL211 = self._shapeFns[211].BLinear(x1, x2, x3, x4) 
            BL212 = self._shapeFns[212].BLinear(x1, x2, x3, x4) 
            BL213 = self._shapeFns[213].BLinear(x1, x2, x3, x4)
            BL214 = self._shapeFns[214].BLinear(x1, x2, x3, x4) 
    
            BL221 = self._shapeFns[221].BLinear(x1, x2, x3, x4)
            BL222 = self._shapeFns[222].BLinear(x1, x2, x3, x4) 
            BL223 = self._shapeFns[223].BLinear(x1, x2, x3, x4) 
            BL224 = self._shapeFns[224].BLinear(x1, x2, x3, x4) 
        
            BL231 = self._shapeFns[231].BLinear(x1, x2, x3, x4) 
            BL232 = self._shapeFns[232].BLinear(x1, x2, x3, x4) 
            BL233 = self._shapeFns[233].BLinear(x1, x2, x3, x4) 
            BL234 = self._shapeFns[234].BLinear(x1, x2, x3, x4) 
            
            BL241 = self._shapeFns[241].BLinear(x1, x2, x3, x4)
            BL242 = self._shapeFns[242].BLinear(x1, x2, x3, x4) 
            BL243 = self._shapeFns[243].BLinear(x1, x2, x3, x4) 
            BL244 = self._shapeFns[244].BLinear(x1, x2, x3, x4)
            
            
            BL311 = self._shapeFns[311].BLinear(x1, x2, x3, x4) 
            BL312 = self._shapeFns[312].BLinear(x1, x2, x3, x4) 
            BL313 = self._shapeFns[313].BLinear(x1, x2, x3, x4)
            BL314 = self._shapeFns[314].BLinear(x1, x2, x3, x4) 
            
            BL321 = self._shapeFns[321].BLinear(x1, x2, x3, x4)
            BL322 = self._shapeFns[322].BLinear(x1, x2, x3, x4) 
            BL323 = self._shapeFns[323].BLinear(x1, x2, x3, x4) 
            BL324 = self._shapeFns[324].BLinear(x1, x2, x3, x4) 
            
            BL331 = self._shapeFns[331].BLinear(x1, x2, x3, x4) 
            BL332 = self._shapeFns[332].BLinear(x1, x2, x3, x4) 
            BL333 = self._shapeFns[333].BLinear(x1, x2, x3, x4) 
            BL334 = self._shapeFns[334].BLinear(x1, x2, x3, x4) 
            
            BL341 = self._shapeFns[341].BLinear(x1, x2, x3, x4)
            BL342 = self._shapeFns[342].BLinear(x1, x2, x3, x4) 
            BL343 = self._shapeFns[343].BLinear(x1, x2, x3, x4) 
            BL344 = self._shapeFns[344].BLinear(x1, x2, x3, x4)
            
            
            BL411 = self._shapeFns[411].BLinear(x1, x2, x3, x4) 
            BL412 = self._shapeFns[412].BLinear(x1, x2, x3, x4) 
            BL413 = self._shapeFns[413].BLinear(x1, x2, x3, x4)
            BL414 = self._shapeFns[414].BLinear(x1, x2, x3, x4) 
            
            BL421 = self._shapeFns[421].BLinear(x1, x2, x3, x4)
            BL422 = self._shapeFns[422].BLinear(x1, x2, x3, x4) 
            BL423 = self._shapeFns[423].BLinear(x1, x2, x3, x4) 
            BL424 = self._shapeFns[424].BLinear(x1, x2, x3, x4) 
            
            BL431 = self._shapeFns[431].BLinear(x1, x2, x3, x4) 
            BL432 = self._shapeFns[432].BLinear(x1, x2, x3, x4) 
            BL433 = self._shapeFns[433].BLinear(x1, x2, x3, x4) 
            BL434 = self._shapeFns[434].BLinear(x1, x2, x3, x4) 
            
            BL441 = self._shapeFns[441].BLinear(x1, x2, x3, x4)
            BL442 = self._shapeFns[442].BLinear(x1, x2, x3, x4) 
            BL443 = self._shapeFns[443].BLinear(x1, x2, x3, x4) 
            BL444 = self._shapeFns[444].BLinear(x1, x2, x3, x4)

            
            # the linear stiffness matrix for 4 Gauss points
            KL = (detJ111 * w[0] * w[0] * w[0] * BL111.T.dot(M).dot(BL111) +
                  detJ112 * w[0] * w[0] * w[1] * BL112.T.dot(M).dot(BL112) +
                  detJ113 * w[0] * w[0] * w[2] * BL113.T.dot(M).dot(BL113) +
                  detJ114 * w[0] * w[0] * w[3] * BL114.T.dot(M).dot(BL114) +
                  detJ121 * w[0] * w[1] * w[0] * BL121.T.dot(M).dot(BL121) +
                  detJ122 * w[0] * w[1] * w[1] * BL122.T.dot(M).dot(BL122) +
                  detJ123 * w[0] * w[1] * w[2] * BL123.T.dot(M).dot(BL123) +
                  detJ124 * w[0] * w[1] * w[3] * BL124.T.dot(M).dot(BL124) +
                  detJ131 * w[0] * w[2] * w[0] * BL131.T.dot(M).dot(BL131) +
                  detJ132 * w[0] * w[2] * w[1] * BL132.T.dot(M).dot(BL132) +
                  detJ133 * w[0] * w[2] * w[2] * BL133.T.dot(M).dot(BL133) +
                  detJ134 * w[0] * w[2] * w[3] * BL134.T.dot(M).dot(BL134) +
                  detJ141 * w[0] * w[3] * w[0] * BL141.T.dot(M).dot(BL141) +
                  detJ142 * w[0] * w[3] * w[1] * BL142.T.dot(M).dot(BL142) +
                  detJ143 * w[0] * w[3] * w[2] * BL143.T.dot(M).dot(BL143) +
                  detJ144 * w[0] * w[3] * w[3] * BL144.T.dot(M).dot(BL144) +                  
                  detJ211 * w[1] * w[0] * w[0] * BL211.T.dot(M).dot(BL211) +
                  detJ212 * w[1] * w[0] * w[1] * BL212.T.dot(M).dot(BL212) +
                  detJ213 * w[1] * w[0] * w[2] * BL213.T.dot(M).dot(BL213) +
                  detJ214 * w[1] * w[0] * w[3] * BL214.T.dot(M).dot(BL214) +
                  detJ221 * w[1] * w[1] * w[0] * BL221.T.dot(M).dot(BL221) +
                  detJ222 * w[1] * w[1] * w[1] * BL222.T.dot(M).dot(BL222) +
                  detJ223 * w[1] * w[1] * w[2] * BL223.T.dot(M).dot(BL223) +
                  detJ224 * w[1] * w[1] * w[3] * BL224.T.dot(M).dot(BL224) +
                  detJ231 * w[1] * w[2] * w[0] * BL231.T.dot(M).dot(BL231) +
                  detJ232 * w[1] * w[2] * w[1] * BL232.T.dot(M).dot(BL232) +
                  detJ233 * w[1] * w[2] * w[2] * BL233.T.dot(M).dot(BL233) +
                  detJ234 * w[1] * w[2] * w[3] * BL234.T.dot(M).dot(BL234) +
                  detJ241 * w[1] * w[3] * w[0] * BL241.T.dot(M).dot(BL241) +
                  detJ242 * w[1] * w[3] * w[1] * BL242.T.dot(M).dot(BL242) +
                  detJ243 * w[1] * w[3] * w[2] * BL243.T.dot(M).dot(BL243) +
                  detJ244 * w[1] * w[3] * w[3] * BL244.T.dot(M).dot(BL244) +                  
                  detJ311 * w[2] * w[0] * w[0] * BL311.T.dot(M).dot(BL311) +
                  detJ312 * w[2] * w[0] * w[1] * BL312.T.dot(M).dot(BL312) +
                  detJ313 * w[2] * w[0] * w[2] * BL313.T.dot(M).dot(BL313) +
                  detJ314 * w[2] * w[0] * w[3] * BL314.T.dot(M).dot(BL314) +
                  detJ321 * w[2] * w[1] * w[0] * BL321.T.dot(M).dot(BL321) +
                  detJ322 * w[2] * w[1] * w[1] * BL322.T.dot(M).dot(BL322) +
                  detJ323 * w[2] * w[1] * w[2] * BL323.T.dot(M).dot(BL323) +
                  detJ324 * w[2] * w[1] * w[3] * BL324.T.dot(M).dot(BL324) +
                  detJ331 * w[2] * w[2] * w[0] * BL331.T.dot(M).dot(BL331) +
                  detJ332 * w[2] * w[2] * w[1] * BL332.T.dot(M).dot(BL332) +
                  detJ333 * w[2] * w[2] * w[2] * BL333.T.dot(M).dot(BL333) +
                  detJ334 * w[2] * w[2] * w[3] * BL334.T.dot(M).dot(BL334) +
                  detJ341 * w[2] * w[3] * w[0] * BL341.T.dot(M).dot(BL341) +
                  detJ342 * w[2] * w[3] * w[1] * BL342.T.dot(M).dot(BL342) +
                  detJ343 * w[2] * w[3] * w[2] * BL343.T.dot(M).dot(BL343) +
                  detJ344 * w[2] * w[3] * w[3] * BL344.T.dot(M).dot(BL344) +                  
                  detJ411 * w[3] * w[0] * w[0] * BL411.T.dot(M).dot(BL411) +
                  detJ412 * w[3] * w[0] * w[1] * BL412.T.dot(M).dot(BL412) +
                  detJ413 * w[3] * w[0] * w[2] * BL413.T.dot(M).dot(BL413) +
                  detJ414 * w[3] * w[0] * w[3] * BL414.T.dot(M).dot(BL414) +
                  detJ421 * w[3] * w[1] * w[0] * BL421.T.dot(M).dot(BL421) +
                  detJ422 * w[3] * w[1] * w[1] * BL422.T.dot(M).dot(BL422) +
                  detJ423 * w[3] * w[1] * w[2] * BL423.T.dot(M).dot(BL423) +
                  detJ424 * w[3] * w[1] * w[3] * BL424.T.dot(M).dot(BL424) +
                  detJ431 * w[3] * w[2] * w[0] * BL431.T.dot(M).dot(BL431) +
                  detJ432 * w[3] * w[2] * w[1] * BL432.T.dot(M).dot(BL432) +
                  detJ433 * w[3] * w[2] * w[2] * BL433.T.dot(M).dot(BL433) +
                  detJ434 * w[3] * w[2] * w[3] * BL434.T.dot(M).dot(BL434) +
                  detJ441 * w[3] * w[3] * w[0] * BL441.T.dot(M).dot(BL441) +
                  detJ442 * w[3] * w[3] * w[1] * BL442.T.dot(M).dot(BL442) +
                  detJ443 * w[3] * w[3] * w[2] * BL443.T.dot(M).dot(BL443) +
                  detJ444 * w[3] * w[3] * w[3] * BL444.T.dot(M).dot(BL444))
            
            # create the first nonlinear Bmatrix
            BNF111 = self._shapeFns[111].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF112 = self._shapeFns[112].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF113 = self._shapeFns[113].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF114 = self._shapeFns[114].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF121 = self._shapeFns[121].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF122 = self._shapeFns[122].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF123 = self._shapeFns[123].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF124 = self._shapeFns[124].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF131 = self._shapeFns[131].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF132 = self._shapeFns[132].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF133 = self._shapeFns[133].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF134 = self._shapeFns[134].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF141 = self._shapeFns[141].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF142 = self._shapeFns[142].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF143 = self._shapeFns[143].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF144 = self._shapeFns[144].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNF211 = self._shapeFns[211].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF212 = self._shapeFns[212].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF213 = self._shapeFns[213].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF214 = self._shapeFns[214].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF221 = self._shapeFns[221].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF222 = self._shapeFns[222].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF223 = self._shapeFns[223].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF224 = self._shapeFns[224].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF231 = self._shapeFns[231].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF232 = self._shapeFns[232].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF233 = self._shapeFns[233].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF234 = self._shapeFns[234].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF241 = self._shapeFns[241].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF242 = self._shapeFns[242].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF243 = self._shapeFns[243].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF244 = self._shapeFns[244].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNF311 = self._shapeFns[311].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF312 = self._shapeFns[312].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF313 = self._shapeFns[313].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF314 = self._shapeFns[314].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF321 = self._shapeFns[321].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF322 = self._shapeFns[322].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF323 = self._shapeFns[323].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF324 = self._shapeFns[324].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF331 = self._shapeFns[331].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF332 = self._shapeFns[332].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF333 = self._shapeFns[333].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF334 = self._shapeFns[334].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF341 = self._shapeFns[341].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF342 = self._shapeFns[342].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF343 = self._shapeFns[343].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF344 = self._shapeFns[344].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
                        
            BNF411 = self._shapeFns[411].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF412 = self._shapeFns[412].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF413 = self._shapeFns[413].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF414 = self._shapeFns[414].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF421 = self._shapeFns[421].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF422 = self._shapeFns[422].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF423 = self._shapeFns[423].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF424 = self._shapeFns[424].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF431 = self._shapeFns[431].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF432 = self._shapeFns[432].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF433 = self._shapeFns[433].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF434 = self._shapeFns[434].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNF441 = self._shapeFns[441].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF442 = self._shapeFns[442].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF443 = self._shapeFns[443].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF444 = self._shapeFns[444].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            # create the second nonlinear Bmatrix
            BNS111 = self._shapeFns[111].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS112 = self._shapeFns[112].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS113 = self._shapeFns[113].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS114 = self._shapeFns[114].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS121 = self._shapeFns[121].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS122 = self._shapeFns[122].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS123 = self._shapeFns[123].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS124 = self._shapeFns[124].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS131 = self._shapeFns[131].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS132 = self._shapeFns[132].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS133 = self._shapeFns[133].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS134 = self._shapeFns[134].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS141 = self._shapeFns[141].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS142 = self._shapeFns[142].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS143 = self._shapeFns[143].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS144 = self._shapeFns[144].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNS211 = self._shapeFns[211].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS212 = self._shapeFns[212].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS213 = self._shapeFns[213].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS214 = self._shapeFns[214].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS221 = self._shapeFns[221].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS222 = self._shapeFns[222].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS223 = self._shapeFns[223].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS224 = self._shapeFns[224].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS231 = self._shapeFns[231].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS232 = self._shapeFns[232].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS233 = self._shapeFns[233].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS234 = self._shapeFns[234].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS241 = self._shapeFns[241].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS242 = self._shapeFns[242].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS243 = self._shapeFns[243].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS244 = self._shapeFns[244].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNS311 = self._shapeFns[311].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS312 = self._shapeFns[312].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS313 = self._shapeFns[313].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS314 = self._shapeFns[314].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS321 = self._shapeFns[321].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS322 = self._shapeFns[322].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS323 = self._shapeFns[323].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS324 = self._shapeFns[324].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS331 = self._shapeFns[331].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS332 = self._shapeFns[332].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS333 = self._shapeFns[333].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS334 = self._shapeFns[334].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS341 = self._shapeFns[341].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS342 = self._shapeFns[342].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS343 = self._shapeFns[343].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS344 = self._shapeFns[344].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
                        
            BNS411 = self._shapeFns[411].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS412 = self._shapeFns[412].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS413 = self._shapeFns[413].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS414 = self._shapeFns[414].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS421 = self._shapeFns[421].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS422 = self._shapeFns[422].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS423 = self._shapeFns[423].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS424 = self._shapeFns[424].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS431 = self._shapeFns[431].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS432 = self._shapeFns[432].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS433 = self._shapeFns[433].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS434 = self._shapeFns[434].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNS441 = self._shapeFns[441].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS442 = self._shapeFns[442].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS443 = self._shapeFns[443].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS444 = self._shapeFns[444].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
                       
            
            # create the third nonlinear Bmatrix
            BNT111 = self._shapeFns[111].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT112 = self._shapeFns[112].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT113 = self._shapeFns[113].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT114 = self._shapeFns[114].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT121 = self._shapeFns[121].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT122 = self._shapeFns[122].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT123 = self._shapeFns[123].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT124 = self._shapeFns[124].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT131 = self._shapeFns[131].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT132 = self._shapeFns[132].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT133 = self._shapeFns[133].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT134 = self._shapeFns[134].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT141 = self._shapeFns[141].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT142 = self._shapeFns[142].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT143 = self._shapeFns[143].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT144 = self._shapeFns[144].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNT211 = self._shapeFns[211].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT212 = self._shapeFns[212].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT213 = self._shapeFns[213].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT214 = self._shapeFns[214].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT221 = self._shapeFns[221].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT222 = self._shapeFns[222].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT223 = self._shapeFns[223].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT224 = self._shapeFns[224].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT231 = self._shapeFns[231].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT232 = self._shapeFns[232].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT233 = self._shapeFns[233].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT234 = self._shapeFns[234].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT241 = self._shapeFns[241].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT242 = self._shapeFns[242].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT243 = self._shapeFns[243].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT244 = self._shapeFns[244].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNT311 = self._shapeFns[311].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT312 = self._shapeFns[312].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT313 = self._shapeFns[313].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT314 = self._shapeFns[314].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT321 = self._shapeFns[321].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT322 = self._shapeFns[322].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT323 = self._shapeFns[323].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT324 = self._shapeFns[324].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT331 = self._shapeFns[331].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT332 = self._shapeFns[332].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT333 = self._shapeFns[333].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT334 = self._shapeFns[334].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT341 = self._shapeFns[341].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT342 = self._shapeFns[342].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT343 = self._shapeFns[343].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT344 = self._shapeFns[344].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
                        
            BNT411 = self._shapeFns[411].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT412 = self._shapeFns[412].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT413 = self._shapeFns[413].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT414 = self._shapeFns[414].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT421 = self._shapeFns[421].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT422 = self._shapeFns[422].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT423 = self._shapeFns[423].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT424 = self._shapeFns[424].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT431 = self._shapeFns[431].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT432 = self._shapeFns[432].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT433 = self._shapeFns[433].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT434 = self._shapeFns[434].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)            
            BNT441 = self._shapeFns[441].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT442 = self._shapeFns[442].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT443 = self._shapeFns[443].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT444 = self._shapeFns[444].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            # total nonlinear Bmatrix
            BN111 = BNF111 + BNS111 + BNT111
            BN112 = BNF112 + BNS112 + BNT112
            BN113 = BNF113 + BNS113 + BNT113
            BN114 = BNF114 + BNS114 + BNT114            
            BN121 = BNF121 + BNS121 + BNT121
            BN122 = BNF122 + BNS122 + BNT122
            BN123 = BNF123 + BNS123 + BNT123
            BN124 = BNF124 + BNS124 + BNT124            
            BN131 = BNF131 + BNS131 + BNT131
            BN132 = BNF132 + BNS132 + BNT132
            BN133 = BNF133 + BNS133 + BNT133
            BN134 = BNF134 + BNS134 + BNT134            
            BN141 = BNF141 + BNS141 + BNT141
            BN142 = BNF142 + BNS142 + BNT142
            BN143 = BNF143 + BNS143 + BNT143
            BN144 = BNF144 + BNS144 + BNT144
            
            BN211 = BNF211 + BNS211 + BNT211
            BN212 = BNF212 + BNS212 + BNT212
            BN213 = BNF213 + BNS213 + BNT213
            BN214 = BNF214 + BNS214 + BNT214            
            BN221 = BNF221 + BNS221 + BNT221
            BN222 = BNF222 + BNS222 + BNT222
            BN223 = BNF223 + BNS223 + BNT223
            BN224 = BNF224 + BNS224 + BNT224            
            BN231 = BNF231 + BNS231 + BNT231
            BN232 = BNF232 + BNS232 + BNT232
            BN233 = BNF233 + BNS233 + BNT233
            BN234 = BNF234 + BNS234 + BNT234            
            BN241 = BNF241 + BNS241 + BNT241
            BN242 = BNF242 + BNS242 + BNT242
            BN243 = BNF243 + BNS243 + BNT243
            BN244 = BNF244 + BNS244 + BNT244
            
            BN311 = BNF311 + BNS311 + BNT311
            BN312 = BNF312 + BNS312 + BNT312
            BN313 = BNF313 + BNS313 + BNT313
            BN314 = BNF314 + BNS314 + BNT314            
            BN321 = BNF321 + BNS321 + BNT321
            BN322 = BNF322 + BNS322 + BNT322
            BN323 = BNF323 + BNS323 + BNT323
            BN324 = BNF324 + BNS324 + BNT324            
            BN331 = BNF331 + BNS331 + BNT331
            BN332 = BNF332 + BNS332 + BNT332
            BN333 = BNF333 + BNS333 + BNT333
            BN334 = BNF334 + BNS334 + BNT334            
            BN341 = BNF341 + BNS341 + BNT341
            BN342 = BNF342 + BNS342 + BNT342
            BN343 = BNF343 + BNS343 + BNT343
            BN344 = BNF344 + BNS344 + BNT344
            
            BN411 = BNF411 + BNS411 + BNT411
            BN412 = BNF412 + BNS412 + BNT412
            BN413 = BNF413 + BNS413 + BNT413
            BN414 = BNF414 + BNS414 + BNT414            
            BN421 = BNF421 + BNS421 + BNT421
            BN422 = BNF422 + BNS422 + BNT422
            BN423 = BNF423 + BNS423 + BNT423
            BN424 = BNF424 + BNS424 + BNT424            
            BN431 = BNF431 + BNS431 + BNT431
            BN432 = BNF432 + BNS432 + BNT432
            BN433 = BNF433 + BNS433 + BNT433
            BN434 = BNF434 + BNS434 + BNT434            
            BN441 = BNF441 + BNS441 + BNT441
            BN442 = BNF442 + BNS442 + BNT442
            BN443 = BNF443 + BNS443 + BNT443
            BN444 = BNF444 + BNS444 + BNT444
            
            
            # create the first H matrix
            HF111 = self._shapeFns[111].HmatrixF(x1, x2, x3, x4)
            HF112 = self._shapeFns[112].HmatrixF(x1, x2, x3, x4)
            HF113 = self._shapeFns[113].HmatrixF(x1, x2, x3, x4)
            HF114 = self._shapeFns[114].HmatrixF(x1, x2, x3, x4)
            HF121 = self._shapeFns[121].HmatrixF(x1, x2, x3, x4)
            HF122 = self._shapeFns[122].HmatrixF(x1, x2, x3, x4)
            HF123 = self._shapeFns[123].HmatrixF(x1, x2, x3, x4)
            HF124 = self._shapeFns[124].HmatrixF(x1, x2, x3, x4)            
            HF131 = self._shapeFns[131].HmatrixF(x1, x2, x3, x4)
            HF132 = self._shapeFns[132].HmatrixF(x1, x2, x3, x4)
            HF133 = self._shapeFns[133].HmatrixF(x1, x2, x3, x4)
            HF134 = self._shapeFns[134].HmatrixF(x1, x2, x3, x4)            
            HF141 = self._shapeFns[141].HmatrixF(x1, x2, x3, x4)
            HF142 = self._shapeFns[142].HmatrixF(x1, x2, x3, x4)
            HF143 = self._shapeFns[143].HmatrixF(x1, x2, x3, x4)
            HF144 = self._shapeFns[144].HmatrixF(x1, x2, x3, x4)
            
            HF211 = self._shapeFns[211].HmatrixF(x1, x2, x3, x4)
            HF212 = self._shapeFns[212].HmatrixF(x1, x2, x3, x4)
            HF213 = self._shapeFns[213].HmatrixF(x1, x2, x3, x4)
            HF214 = self._shapeFns[214].HmatrixF(x1, x2, x3, x4)
            HF221 = self._shapeFns[221].HmatrixF(x1, x2, x3, x4)
            HF222 = self._shapeFns[222].HmatrixF(x1, x2, x3, x4)
            HF223 = self._shapeFns[223].HmatrixF(x1, x2, x3, x4)
            HF224 = self._shapeFns[224].HmatrixF(x1, x2, x3, x4)            
            HF231 = self._shapeFns[231].HmatrixF(x1, x2, x3, x4)
            HF232 = self._shapeFns[232].HmatrixF(x1, x2, x3, x4)
            HF233 = self._shapeFns[233].HmatrixF(x1, x2, x3, x4)
            HF234 = self._shapeFns[234].HmatrixF(x1, x2, x3, x4)            
            HF241 = self._shapeFns[241].HmatrixF(x1, x2, x3, x4)
            HF242 = self._shapeFns[242].HmatrixF(x1, x2, x3, x4)
            HF243 = self._shapeFns[243].HmatrixF(x1, x2, x3, x4)
            HF244 = self._shapeFns[244].HmatrixF(x1, x2, x3, x4)
            
            HF311 = self._shapeFns[311].HmatrixF(x1, x2, x3, x4)
            HF312 = self._shapeFns[312].HmatrixF(x1, x2, x3, x4)
            HF313 = self._shapeFns[313].HmatrixF(x1, x2, x3, x4)
            HF314 = self._shapeFns[314].HmatrixF(x1, x2, x3, x4)
            HF321 = self._shapeFns[321].HmatrixF(x1, x2, x3, x4)
            HF322 = self._shapeFns[322].HmatrixF(x1, x2, x3, x4)
            HF323 = self._shapeFns[323].HmatrixF(x1, x2, x3, x4)
            HF324 = self._shapeFns[324].HmatrixF(x1, x2, x3, x4)            
            HF331 = self._shapeFns[331].HmatrixF(x1, x2, x3, x4)
            HF332 = self._shapeFns[332].HmatrixF(x1, x2, x3, x4)
            HF333 = self._shapeFns[333].HmatrixF(x1, x2, x3, x4)
            HF334 = self._shapeFns[334].HmatrixF(x1, x2, x3, x4)            
            HF341 = self._shapeFns[341].HmatrixF(x1, x2, x3, x4)
            HF342 = self._shapeFns[342].HmatrixF(x1, x2, x3, x4)
            HF343 = self._shapeFns[343].HmatrixF(x1, x2, x3, x4)
            HF344 = self._shapeFns[344].HmatrixF(x1, x2, x3, x4)
            
            HF411 = self._shapeFns[411].HmatrixF(x1, x2, x3, x4)
            HF412 = self._shapeFns[412].HmatrixF(x1, x2, x3, x4)
            HF413 = self._shapeFns[413].HmatrixF(x1, x2, x3, x4)
            HF414 = self._shapeFns[414].HmatrixF(x1, x2, x3, x4)
            HF421 = self._shapeFns[421].HmatrixF(x1, x2, x3, x4)
            HF422 = self._shapeFns[422].HmatrixF(x1, x2, x3, x4)
            HF423 = self._shapeFns[423].HmatrixF(x1, x2, x3, x4)
            HF424 = self._shapeFns[424].HmatrixF(x1, x2, x3, x4)            
            HF431 = self._shapeFns[431].HmatrixF(x1, x2, x3, x4)
            HF432 = self._shapeFns[432].HmatrixF(x1, x2, x3, x4)
            HF433 = self._shapeFns[433].HmatrixF(x1, x2, x3, x4)
            HF434 = self._shapeFns[434].HmatrixF(x1, x2, x3, x4)            
            HF441 = self._shapeFns[441].HmatrixF(x1, x2, x3, x4)
            HF442 = self._shapeFns[442].HmatrixF(x1, x2, x3, x4)
            HF443 = self._shapeFns[443].HmatrixF(x1, x2, x3, x4)
            HF444 = self._shapeFns[444].HmatrixF(x1, x2, x3, x4)


            # create the second H matrix
            HS111 = self._shapeFns[111].HmatrixS(x1, x2, x3, x4)
            HS112 = self._shapeFns[112].HmatrixS(x1, x2, x3, x4)
            HS113 = self._shapeFns[113].HmatrixS(x1, x2, x3, x4)
            HS114 = self._shapeFns[114].HmatrixS(x1, x2, x3, x4)
            HS121 = self._shapeFns[121].HmatrixS(x1, x2, x3, x4)
            HS122 = self._shapeFns[122].HmatrixS(x1, x2, x3, x4)
            HS123 = self._shapeFns[123].HmatrixS(x1, x2, x3, x4)
            HS124 = self._shapeFns[124].HmatrixS(x1, x2, x3, x4)            
            HS131 = self._shapeFns[131].HmatrixS(x1, x2, x3, x4)
            HS132 = self._shapeFns[132].HmatrixS(x1, x2, x3, x4)
            HS133 = self._shapeFns[133].HmatrixS(x1, x2, x3, x4)
            HS134 = self._shapeFns[134].HmatrixS(x1, x2, x3, x4)            
            HS141 = self._shapeFns[141].HmatrixS(x1, x2, x3, x4)
            HS142 = self._shapeFns[142].HmatrixS(x1, x2, x3, x4)
            HS143 = self._shapeFns[143].HmatrixS(x1, x2, x3, x4)
            HS144 = self._shapeFns[144].HmatrixS(x1, x2, x3, x4)
            
            HS211 = self._shapeFns[211].HmatrixS(x1, x2, x3, x4)
            HS212 = self._shapeFns[212].HmatrixS(x1, x2, x3, x4)
            HS213 = self._shapeFns[213].HmatrixS(x1, x2, x3, x4)
            HS214 = self._shapeFns[214].HmatrixS(x1, x2, x3, x4)
            HS221 = self._shapeFns[221].HmatrixS(x1, x2, x3, x4)
            HS222 = self._shapeFns[222].HmatrixS(x1, x2, x3, x4)
            HS223 = self._shapeFns[223].HmatrixS(x1, x2, x3, x4)
            HS224 = self._shapeFns[224].HmatrixS(x1, x2, x3, x4)            
            HS231 = self._shapeFns[231].HmatrixS(x1, x2, x3, x4)
            HS232 = self._shapeFns[232].HmatrixS(x1, x2, x3, x4)
            HS233 = self._shapeFns[233].HmatrixS(x1, x2, x3, x4)
            HS234 = self._shapeFns[234].HmatrixS(x1, x2, x3, x4)            
            HS241 = self._shapeFns[241].HmatrixS(x1, x2, x3, x4)
            HS242 = self._shapeFns[242].HmatrixS(x1, x2, x3, x4)
            HS243 = self._shapeFns[243].HmatrixS(x1, x2, x3, x4)
            HS244 = self._shapeFns[244].HmatrixS(x1, x2, x3, x4)
            
            HS311 = self._shapeFns[311].HmatrixS(x1, x2, x3, x4)
            HS312 = self._shapeFns[312].HmatrixS(x1, x2, x3, x4)
            HS313 = self._shapeFns[313].HmatrixS(x1, x2, x3, x4)
            HS314 = self._shapeFns[314].HmatrixS(x1, x2, x3, x4)
            HS321 = self._shapeFns[321].HmatrixS(x1, x2, x3, x4)
            HS322 = self._shapeFns[322].HmatrixS(x1, x2, x3, x4)
            HS323 = self._shapeFns[323].HmatrixS(x1, x2, x3, x4)
            HS324 = self._shapeFns[324].HmatrixS(x1, x2, x3, x4)            
            HS331 = self._shapeFns[331].HmatrixS(x1, x2, x3, x4)
            HS332 = self._shapeFns[332].HmatrixS(x1, x2, x3, x4)
            HS333 = self._shapeFns[333].HmatrixS(x1, x2, x3, x4)
            HS334 = self._shapeFns[334].HmatrixS(x1, x2, x3, x4)            
            HS341 = self._shapeFns[341].HmatrixS(x1, x2, x3, x4)
            HS342 = self._shapeFns[342].HmatrixS(x1, x2, x3, x4)
            HS343 = self._shapeFns[343].HmatrixS(x1, x2, x3, x4)
            HS344 = self._shapeFns[344].HmatrixS(x1, x2, x3, x4)
            
            HS411 = self._shapeFns[411].HmatrixS(x1, x2, x3, x4)
            HS412 = self._shapeFns[412].HmatrixS(x1, x2, x3, x4)
            HS413 = self._shapeFns[413].HmatrixS(x1, x2, x3, x4)
            HS414 = self._shapeFns[414].HmatrixS(x1, x2, x3, x4)
            HS421 = self._shapeFns[421].HmatrixS(x1, x2, x3, x4)
            HS422 = self._shapeFns[422].HmatrixS(x1, x2, x3, x4)
            HS423 = self._shapeFns[423].HmatrixS(x1, x2, x3, x4)
            HS424 = self._shapeFns[424].HmatrixS(x1, x2, x3, x4)            
            HS431 = self._shapeFns[431].HmatrixS(x1, x2, x3, x4)
            HS432 = self._shapeFns[432].HmatrixS(x1, x2, x3, x4)
            HS433 = self._shapeFns[433].HmatrixS(x1, x2, x3, x4)
            HS434 = self._shapeFns[434].HmatrixS(x1, x2, x3, x4)            
            HS441 = self._shapeFns[441].HmatrixS(x1, x2, x3, x4)
            HS442 = self._shapeFns[442].HmatrixS(x1, x2, x3, x4)
            HS443 = self._shapeFns[443].HmatrixS(x1, x2, x3, x4)
            HS444 = self._shapeFns[444].HmatrixS(x1, x2, x3, x4)
            
            
            
            # create the third H matrix
            HT111 = self._shapeFns[111].HmatrixT(x1, x2, x3, x4)
            HT112 = self._shapeFns[112].HmatrixT(x1, x2, x3, x4)
            HT113 = self._shapeFns[113].HmatrixT(x1, x2, x3, x4)
            HT114 = self._shapeFns[114].HmatrixT(x1, x2, x3, x4)
            HT121 = self._shapeFns[121].HmatrixT(x1, x2, x3, x4)
            HT122 = self._shapeFns[122].HmatrixT(x1, x2, x3, x4)
            HT123 = self._shapeFns[123].HmatrixT(x1, x2, x3, x4)
            HT124 = self._shapeFns[124].HmatrixT(x1, x2, x3, x4)            
            HT131 = self._shapeFns[131].HmatrixT(x1, x2, x3, x4)
            HT132 = self._shapeFns[132].HmatrixT(x1, x2, x3, x4)
            HT133 = self._shapeFns[133].HmatrixT(x1, x2, x3, x4)
            HT134 = self._shapeFns[134].HmatrixT(x1, x2, x3, x4)            
            HT141 = self._shapeFns[141].HmatrixT(x1, x2, x3, x4)
            HT142 = self._shapeFns[142].HmatrixT(x1, x2, x3, x4)
            HT143 = self._shapeFns[143].HmatrixT(x1, x2, x3, x4)
            HT144 = self._shapeFns[144].HmatrixT(x1, x2, x3, x4)
            
            HT211 = self._shapeFns[211].HmatrixT(x1, x2, x3, x4)
            HT212 = self._shapeFns[212].HmatrixT(x1, x2, x3, x4)
            HT213 = self._shapeFns[213].HmatrixT(x1, x2, x3, x4)
            HT214 = self._shapeFns[214].HmatrixT(x1, x2, x3, x4)
            HT221 = self._shapeFns[221].HmatrixT(x1, x2, x3, x4)
            HT222 = self._shapeFns[222].HmatrixT(x1, x2, x3, x4)
            HT223 = self._shapeFns[223].HmatrixT(x1, x2, x3, x4)
            HT224 = self._shapeFns[224].HmatrixT(x1, x2, x3, x4)            
            HT231 = self._shapeFns[231].HmatrixT(x1, x2, x3, x4)
            HT232 = self._shapeFns[232].HmatrixT(x1, x2, x3, x4)
            HT233 = self._shapeFns[233].HmatrixT(x1, x2, x3, x4)
            HT234 = self._shapeFns[234].HmatrixT(x1, x2, x3, x4)            
            HT241 = self._shapeFns[241].HmatrixT(x1, x2, x3, x4)
            HT242 = self._shapeFns[242].HmatrixT(x1, x2, x3, x4)
            HT243 = self._shapeFns[243].HmatrixT(x1, x2, x3, x4)
            HT244 = self._shapeFns[244].HmatrixT(x1, x2, x3, x4)
            
            HT311 = self._shapeFns[311].HmatrixT(x1, x2, x3, x4)
            HT312 = self._shapeFns[312].HmatrixT(x1, x2, x3, x4)
            HT313 = self._shapeFns[313].HmatrixT(x1, x2, x3, x4)
            HT314 = self._shapeFns[314].HmatrixT(x1, x2, x3, x4)
            HT321 = self._shapeFns[321].HmatrixT(x1, x2, x3, x4)
            HT322 = self._shapeFns[322].HmatrixT(x1, x2, x3, x4)
            HT323 = self._shapeFns[323].HmatrixT(x1, x2, x3, x4)
            HT324 = self._shapeFns[324].HmatrixT(x1, x2, x3, x4)            
            HT331 = self._shapeFns[331].HmatrixT(x1, x2, x3, x4)
            HT332 = self._shapeFns[332].HmatrixT(x1, x2, x3, x4)
            HT333 = self._shapeFns[333].HmatrixT(x1, x2, x3, x4)
            HT334 = self._shapeFns[334].HmatrixT(x1, x2, x3, x4)            
            HT341 = self._shapeFns[341].HmatrixT(x1, x2, x3, x4)
            HT342 = self._shapeFns[342].HmatrixT(x1, x2, x3, x4)
            HT343 = self._shapeFns[343].HmatrixT(x1, x2, x3, x4)
            HT344 = self._shapeFns[344].HmatrixT(x1, x2, x3, x4)
            
            HT411 = self._shapeFns[411].HmatrixT(x1, x2, x3, x4)
            HT412 = self._shapeFns[412].HmatrixT(x1, x2, x3, x4)
            HT413 = self._shapeFns[413].HmatrixT(x1, x2, x3, x4)
            HT414 = self._shapeFns[414].HmatrixT(x1, x2, x3, x4)
            HT421 = self._shapeFns[421].HmatrixT(x1, x2, x3, x4)
            HT422 = self._shapeFns[422].HmatrixT(x1, x2, x3, x4)
            HT423 = self._shapeFns[423].HmatrixT(x1, x2, x3, x4)
            HT424 = self._shapeFns[424].HmatrixT(x1, x2, x3, x4)            
            HT431 = self._shapeFns[431].HmatrixT(x1, x2, x3, x4)
            HT432 = self._shapeFns[432].HmatrixT(x1, x2, x3, x4)
            HT433 = self._shapeFns[433].HmatrixT(x1, x2, x3, x4)
            HT434 = self._shapeFns[434].HmatrixT(x1, x2, x3, x4)            
            HT441 = self._shapeFns[441].HmatrixT(x1, x2, x3, x4)
            HT442 = self._shapeFns[442].HmatrixT(x1, x2, x3, x4)
            HT443 = self._shapeFns[443].HmatrixT(x1, x2, x3, x4)
            HT444 = self._shapeFns[444].HmatrixT(x1, x2, x3, x4)
         
            
            # the nonlinear stiffness matrix for 4 Gauss point
            f111 = (BL111.T.dot(M).dot(BN111) + BN111.T.dot(M).dot(BL111) + 
                   BN111.T.dot(M).dot(BN111))
            f112 = (BL112.T.dot(M).dot(BN112) + BN112.T.dot(M).dot(BL112) + 
                   BN112.T.dot(M).dot(BN112))
            f113 = (BL113.T.dot(M).dot(BN113) + BN113.T.dot(M).dot(BL113) + 
                   BN113.T.dot(M).dot(BN113))
            f114 = (BL114.T.dot(M).dot(BN114) + BN114.T.dot(M).dot(BL114) + 
                   BN114.T.dot(M).dot(BN114))            
            f121 = (BL121.T.dot(M).dot(BN121) + BN121.T.dot(M).dot(BL121) + 
                   BN121.T.dot(M).dot(BN121))
            f122 = (BL122.T.dot(M).dot(BN122) + BN122.T.dot(M).dot(BL122) + 
                   BN122.T.dot(M).dot(BN122))
            f123 = (BL123.T.dot(M).dot(BN123) + BN123.T.dot(M).dot(BL123) + 
                   BN123.T.dot(M).dot(BN123))
            f124 = (BL124.T.dot(M).dot(BN124) + BN124.T.dot(M).dot(BL124) + 
                   BN124.T.dot(M).dot(BN124))            
            f131 = (BL131.T.dot(M).dot(BN131) + BN131.T.dot(M).dot(BL131) + 
                   BN131.T.dot(M).dot(BN131))
            f132 = (BL132.T.dot(M).dot(BN132) + BN132.T.dot(M).dot(BL132) + 
                   BN132.T.dot(M).dot(BN132))
            f133 = (BL133.T.dot(M).dot(BN133) + BN133.T.dot(M).dot(BL133) + 
                   BN133.T.dot(M).dot(BN133))
            f134 = (BL134.T.dot(M).dot(BN134) + BN134.T.dot(M).dot(BL134) + 
                   BN134.T.dot(M).dot(BN134))            
            f141 = (BL141.T.dot(M).dot(BN141) + BN141.T.dot(M).dot(BL141) + 
                   BN141.T.dot(M).dot(BN141))
            f142 = (BL142.T.dot(M).dot(BN142) + BN142.T.dot(M).dot(BL142) + 
                   BN142.T.dot(M).dot(BN142))
            f143 = (BL143.T.dot(M).dot(BN143) + BN143.T.dot(M).dot(BL143) + 
                   BN143.T.dot(M).dot(BN143))
            f144 = (BL144.T.dot(M).dot(BN144) + BN144.T.dot(M).dot(BL144) + 
                   BN144.T.dot(M).dot(BN144))
            
            f211 = (BL211.T.dot(M).dot(BN211) + BN211.T.dot(M).dot(BL211) + 
                   BN211.T.dot(M).dot(BN211))
            f212 = (BL212.T.dot(M).dot(BN212) + BN212.T.dot(M).dot(BL212) + 
                   BN212.T.dot(M).dot(BN212))
            f213 = (BL213.T.dot(M).dot(BN213) + BN213.T.dot(M).dot(BL213) + 
                   BN213.T.dot(M).dot(BN213))
            f214 = (BL214.T.dot(M).dot(BN214) + BN214.T.dot(M).dot(BL214) + 
                   BN214.T.dot(M).dot(BN214))            
            f221 = (BL221.T.dot(M).dot(BN221) + BN221.T.dot(M).dot(BL221) + 
                   BN221.T.dot(M).dot(BN221))
            f222 = (BL222.T.dot(M).dot(BN222) + BN222.T.dot(M).dot(BL222) + 
                   BN222.T.dot(M).dot(BN222))
            f223 = (BL223.T.dot(M).dot(BN223) + BN223.T.dot(M).dot(BL223) + 
                   BN223.T.dot(M).dot(BN223))
            f224 = (BL224.T.dot(M).dot(BN224) + BN224.T.dot(M).dot(BL224) + 
                   BN224.T.dot(M).dot(BN224))            
            f231 = (BL231.T.dot(M).dot(BN231) + BN231.T.dot(M).dot(BL231) + 
                   BN231.T.dot(M).dot(BN231))
            f232 = (BL232.T.dot(M).dot(BN232) + BN232.T.dot(M).dot(BL232) + 
                   BN232.T.dot(M).dot(BN232))
            f233 = (BL233.T.dot(M).dot(BN233) + BN233.T.dot(M).dot(BL233) + 
                   BN233.T.dot(M).dot(BN233))
            f234 = (BL234.T.dot(M).dot(BN234) + BN234.T.dot(M).dot(BL234) + 
                   BN234.T.dot(M).dot(BN234))            
            f241 = (BL241.T.dot(M).dot(BN241) + BN241.T.dot(M).dot(BL241) + 
                   BN241.T.dot(M).dot(BN241))
            f242 = (BL242.T.dot(M).dot(BN242) + BN242.T.dot(M).dot(BL242) + 
                   BN242.T.dot(M).dot(BN242))
            f243 = (BL243.T.dot(M).dot(BN243) + BN243.T.dot(M).dot(BL243) + 
                   BN243.T.dot(M).dot(BN243))
            f244 = (BL244.T.dot(M).dot(BN244) + BN244.T.dot(M).dot(BL244) + 
                   BN244.T.dot(M).dot(BN244))
            
            f311 = (BL311.T.dot(M).dot(BN311) + BN311.T.dot(M).dot(BL311) + 
                   BN311.T.dot(M).dot(BN311))
            f312 = (BL312.T.dot(M).dot(BN312) + BN312.T.dot(M).dot(BL312) + 
                   BN312.T.dot(M).dot(BN312))
            f313 = (BL313.T.dot(M).dot(BN313) + BN313.T.dot(M).dot(BL313) + 
                   BN313.T.dot(M).dot(BN313))
            f314 = (BL314.T.dot(M).dot(BN314) + BN314.T.dot(M).dot(BL314) + 
                   BN314.T.dot(M).dot(BN314))            
            f321 = (BL321.T.dot(M).dot(BN321) + BN321.T.dot(M).dot(BL321) + 
                   BN321.T.dot(M).dot(BN321))
            f322 = (BL322.T.dot(M).dot(BN322) + BN322.T.dot(M).dot(BL322) + 
                   BN322.T.dot(M).dot(BN322))
            f323 = (BL323.T.dot(M).dot(BN323) + BN323.T.dot(M).dot(BL323) + 
                   BN323.T.dot(M).dot(BN323))
            f324 = (BL324.T.dot(M).dot(BN324) + BN324.T.dot(M).dot(BL324) + 
                   BN324.T.dot(M).dot(BN324))            
            f331 = (BL331.T.dot(M).dot(BN331) + BN331.T.dot(M).dot(BL331) + 
                   BN331.T.dot(M).dot(BN331))
            f332 = (BL332.T.dot(M).dot(BN332) + BN332.T.dot(M).dot(BL332) + 
                   BN332.T.dot(M).dot(BN332))
            f333 = (BL333.T.dot(M).dot(BN333) + BN333.T.dot(M).dot(BL333) + 
                   BN333.T.dot(M).dot(BN333))
            f334 = (BL334.T.dot(M).dot(BN334) + BN334.T.dot(M).dot(BL334) + 
                   BN334.T.dot(M).dot(BN334))            
            f341 = (BL341.T.dot(M).dot(BN341) + BN341.T.dot(M).dot(BL341) + 
                   BN341.T.dot(M).dot(BN341))
            f342 = (BL342.T.dot(M).dot(BN342) + BN342.T.dot(M).dot(BL342) + 
                   BN342.T.dot(M).dot(BN342))
            f343 = (BL343.T.dot(M).dot(BN343) + BN343.T.dot(M).dot(BL343) + 
                   BN343.T.dot(M).dot(BN343))
            f344 = (BL344.T.dot(M).dot(BN344) + BN344.T.dot(M).dot(BL344) + 
                   BN344.T.dot(M).dot(BN344))
            
            f411 = (BL411.T.dot(M).dot(BN411) + BN411.T.dot(M).dot(BL411) + 
                   BN411.T.dot(M).dot(BN411))
            f412 = (BL412.T.dot(M).dot(BN412) + BN412.T.dot(M).dot(BL412) + 
                   BN412.T.dot(M).dot(BN412))
            f413 = (BL413.T.dot(M).dot(BN413) + BN413.T.dot(M).dot(BL413) + 
                   BN413.T.dot(M).dot(BN413))
            f414 = (BL414.T.dot(M).dot(BN414) + BN414.T.dot(M).dot(BL414) + 
                   BN414.T.dot(M).dot(BN414))            
            f421 = (BL421.T.dot(M).dot(BN421) + BN421.T.dot(M).dot(BL421) + 
                   BN421.T.dot(M).dot(BN421))
            f422 = (BL422.T.dot(M).dot(BN422) + BN422.T.dot(M).dot(BL422) + 
                   BN422.T.dot(M).dot(BN422))
            f423 = (BL423.T.dot(M).dot(BN423) + BN423.T.dot(M).dot(BL423) + 
                   BN423.T.dot(M).dot(BN423))
            f424 = (BL424.T.dot(M).dot(BN424) + BN424.T.dot(M).dot(BL424) + 
                   BN424.T.dot(M).dot(BN424))            
            f431 = (BL431.T.dot(M).dot(BN431) + BN431.T.dot(M).dot(BL431) + 
                   BN431.T.dot(M).dot(BN431))
            f432 = (BL432.T.dot(M).dot(BN432) + BN432.T.dot(M).dot(BL432) + 
                   BN432.T.dot(M).dot(BN432))
            f433 = (BL433.T.dot(M).dot(BN433) + BN433.T.dot(M).dot(BL433) + 
                   BN433.T.dot(M).dot(BN433))
            f434 = (BL434.T.dot(M).dot(BN434) + BN434.T.dot(M).dot(BL434) + 
                   BN434.T.dot(M).dot(BN434))            
            f441 = (BL441.T.dot(M).dot(BN441) + BN441.T.dot(M).dot(BL441) + 
                   BN441.T.dot(M).dot(BN441))
            f442 = (BL442.T.dot(M).dot(BN442) + BN442.T.dot(M).dot(BL442) + 
                   BN442.T.dot(M).dot(BN442))
            f443 = (BL443.T.dot(M).dot(BN443) + BN443.T.dot(M).dot(BL443) + 
                   BN443.T.dot(M).dot(BN443))
            f444 = (BL444.T.dot(M).dot(BN444) + BN444.T.dot(M).dot(BL444) + 
                   BN444.T.dot(M).dot(BN444))
                                
            
            KN = (detJ111 * w[0] * w[0] * w[0] * f111 +
                  detJ112 * w[0] * w[0] * w[1] * f112 +
                  detJ113 * w[0] * w[0] * w[2] * f113 +
                  detJ114 * w[0] * w[0] * w[3] * f114 +
                  detJ121 * w[0] * w[1] * w[0] * f121 +
                  detJ122 * w[0] * w[1] * w[1] * f122 +
                  detJ123 * w[0] * w[1] * w[2] * f123 +
                  detJ124 * w[0] * w[1] * w[3] * f124 +
                  detJ131 * w[0] * w[2] * w[0] * f131 +
                  detJ132 * w[0] * w[2] * w[1] * f132 +
                  detJ133 * w[0] * w[2] * w[2] * f133 +
                  detJ134 * w[0] * w[2] * w[3] * f134 +
                  detJ141 * w[0] * w[3] * w[0] * f141 +
                  detJ142 * w[0] * w[3] * w[1] * f142 +
                  detJ143 * w[0] * w[3] * w[2] * f143 +
                  detJ144 * w[0] * w[3] * w[3] * f144 +                  
                  detJ211 * w[1] * w[0] * w[0] * f211 +
                  detJ212 * w[1] * w[0] * w[1] * f212 +
                  detJ213 * w[1] * w[0] * w[2] * f213 +
                  detJ214 * w[1] * w[0] * w[3] * f214 +
                  detJ221 * w[1] * w[1] * w[0] * f221 +
                  detJ222 * w[1] * w[1] * w[1] * f222 +
                  detJ223 * w[1] * w[1] * w[2] * f223 +
                  detJ224 * w[1] * w[1] * w[3] * f224 +
                  detJ231 * w[1] * w[2] * w[0] * f231 +
                  detJ232 * w[1] * w[2] * w[1] * f232 +
                  detJ233 * w[1] * w[2] * w[2] * f233 +
                  detJ234 * w[1] * w[2] * w[3] * f234 +
                  detJ241 * w[1] * w[3] * w[0] * f241 +
                  detJ242 * w[1] * w[3] * w[1] * f242 +
                  detJ243 * w[1] * w[3] * w[2] * f243 +
                  detJ244 * w[1] * w[3] * w[3] * f244 +                  
                  detJ311 * w[2] * w[0] * w[0] * f311 +
                  detJ312 * w[2] * w[0] * w[1] * f312 +
                  detJ313 * w[2] * w[0] * w[2] * f313 +
                  detJ314 * w[2] * w[0] * w[3] * f314 +
                  detJ321 * w[2] * w[1] * w[0] * f321 +
                  detJ322 * w[2] * w[1] * w[1] * f322 +
                  detJ323 * w[2] * w[1] * w[2] * f323 +
                  detJ324 * w[2] * w[1] * w[3] * f324 +
                  detJ331 * w[2] * w[2] * w[0] * f331 +
                  detJ332 * w[2] * w[2] * w[1] * f332 +
                  detJ333 * w[2] * w[2] * w[2] * f333 +
                  detJ334 * w[2] * w[2] * w[3] * f334 +
                  detJ341 * w[2] * w[3] * w[0] * f341 +
                  detJ342 * w[2] * w[3] * w[1] * f342 +
                  detJ343 * w[2] * w[3] * w[2] * f343 +
                  detJ344 * w[2] * w[3] * w[3] * f344 +
                  detJ411 * w[3] * w[0] * w[0] * f411 +
                  detJ412 * w[3] * w[0] * w[1] * f412 +
                  detJ413 * w[3] * w[0] * w[2] * f413 +
                  detJ414 * w[3] * w[0] * w[3] * f414 +
                  detJ421 * w[3] * w[1] * w[0] * f421 +
                  detJ422 * w[3] * w[1] * w[1] * f422 +
                  detJ423 * w[3] * w[1] * w[2] * f423 +
                  detJ424 * w[3] * w[1] * w[3] * f424 +
                  detJ431 * w[3] * w[2] * w[0] * f431 +
                  detJ432 * w[3] * w[2] * w[1] * f432 +
                  detJ433 * w[3] * w[2] * w[2] * f433 +
                  detJ434 * w[3] * w[2] * w[3] * f434 +
                  detJ441 * w[3] * w[3] * w[0] * f441 +
                  detJ442 * w[3] * w[3] * w[1] * f442 +
                  detJ443 * w[3] * w[3] * w[2] * f443 +
                  detJ444 * w[3] * w[3] * w[3] * f444)

            # create the stress stiffness matrix
            KS = (detJ111 * w[0] * w[0] * w[0] * HF111.T.dot(T).dot(HF111) +
                  detJ112 * w[0] * w[0] * w[1] * HF112.T.dot(T).dot(HF112) +
                  detJ113 * w[0] * w[0] * w[2] * HF113.T.dot(T).dot(HF113) +
                  detJ114 * w[0] * w[0] * w[3] * HF114.T.dot(T).dot(HF114) +
                  detJ121 * w[0] * w[1] * w[0] * HF121.T.dot(T).dot(HF121) +
                  detJ122 * w[0] * w[1] * w[1] * HF122.T.dot(T).dot(HF122) +
                  detJ123 * w[0] * w[1] * w[2] * HF123.T.dot(T).dot(HF123) +
                  detJ124 * w[0] * w[1] * w[3] * HF124.T.dot(T).dot(HF124) +
                  detJ131 * w[0] * w[2] * w[0] * HF131.T.dot(T).dot(HF131) +
                  detJ132 * w[0] * w[2] * w[1] * HF132.T.dot(T).dot(HF132) +
                  detJ133 * w[0] * w[2] * w[2] * HF133.T.dot(T).dot(HF133) +
                  detJ134 * w[0] * w[2] * w[3] * HF134.T.dot(T).dot(HF134) +
                  detJ141 * w[0] * w[3] * w[0] * HF141.T.dot(T).dot(HF141) +
                  detJ142 * w[0] * w[3] * w[1] * HF142.T.dot(T).dot(HF142) +
                  detJ143 * w[0] * w[3] * w[2] * HF143.T.dot(T).dot(HF143) +
                  detJ144 * w[0] * w[3] * w[3] * HF144.T.dot(T).dot(HF144) +                  
                  detJ211 * w[1] * w[0] * w[0] * HF211.T.dot(T).dot(HF211) +
                  detJ212 * w[1] * w[0] * w[1] * HF212.T.dot(T).dot(HF212) +
                  detJ213 * w[1] * w[0] * w[2] * HF213.T.dot(T).dot(HF213) +
                  detJ214 * w[1] * w[0] * w[3] * HF214.T.dot(T).dot(HF214) +
                  detJ221 * w[1] * w[1] * w[0] * HF221.T.dot(T).dot(HF221) +
                  detJ222 * w[1] * w[1] * w[1] * HF222.T.dot(T).dot(HF222) +
                  detJ223 * w[1] * w[1] * w[2] * HF223.T.dot(T).dot(HF223) +
                  detJ224 * w[1] * w[1] * w[3] * HF224.T.dot(T).dot(HF224) +
                  detJ231 * w[1] * w[2] * w[0] * HF231.T.dot(T).dot(HF231) +
                  detJ232 * w[1] * w[2] * w[1] * HF232.T.dot(T).dot(HF232) +
                  detJ233 * w[1] * w[2] * w[2] * HF233.T.dot(T).dot(HF233) +
                  detJ234 * w[1] * w[2] * w[3] * HF234.T.dot(T).dot(HF234) +
                  detJ241 * w[1] * w[3] * w[0] * HF241.T.dot(T).dot(HF241) +
                  detJ242 * w[1] * w[3] * w[1] * HF242.T.dot(T).dot(HF242) +
                  detJ243 * w[1] * w[3] * w[2] * HF243.T.dot(T).dot(HF243) +
                  detJ244 * w[1] * w[3] * w[3] * HF244.T.dot(T).dot(HF244) +                  
                  detJ311 * w[2] * w[0] * w[0] * HF311.T.dot(T).dot(HF311) +
                  detJ312 * w[2] * w[0] * w[1] * HF312.T.dot(T).dot(HF312) +
                  detJ313 * w[2] * w[0] * w[2] * HF313.T.dot(T).dot(HF313) +
                  detJ314 * w[2] * w[0] * w[3] * HF314.T.dot(T).dot(HF314) +
                  detJ321 * w[2] * w[1] * w[0] * HF321.T.dot(T).dot(HF321) +
                  detJ322 * w[2] * w[1] * w[1] * HF322.T.dot(T).dot(HF322) +
                  detJ323 * w[2] * w[1] * w[2] * HF323.T.dot(T).dot(HF323) +
                  detJ324 * w[2] * w[1] * w[3] * HF324.T.dot(T).dot(HF324) +
                  detJ331 * w[2] * w[2] * w[0] * HF331.T.dot(T).dot(HF331) +
                  detJ332 * w[2] * w[2] * w[1] * HF332.T.dot(T).dot(HF332) +
                  detJ333 * w[2] * w[2] * w[2] * HF333.T.dot(T).dot(HF333) +
                  detJ334 * w[2] * w[2] * w[3] * HF334.T.dot(T).dot(HF334) +
                  detJ341 * w[2] * w[3] * w[0] * HF341.T.dot(T).dot(HF341) +
                  detJ342 * w[2] * w[3] * w[1] * HF342.T.dot(T).dot(HF342) +
                  detJ343 * w[2] * w[3] * w[2] * HF343.T.dot(T).dot(HF343) +
                  detJ344 * w[2] * w[3] * w[3] * HF344.T.dot(T).dot(HF344) +                  
                  detJ411 * w[3] * w[0] * w[0] * HF411.T.dot(T).dot(HF411) +
                  detJ412 * w[3] * w[0] * w[1] * HF412.T.dot(T).dot(HF412) +
                  detJ413 * w[3] * w[0] * w[2] * HF413.T.dot(T).dot(HF413) +
                  detJ414 * w[3] * w[0] * w[3] * HF414.T.dot(T).dot(HF414) +
                  detJ421 * w[3] * w[1] * w[0] * HF421.T.dot(T).dot(HF421) +
                  detJ422 * w[3] * w[1] * w[1] * HF422.T.dot(T).dot(HF422) +
                  detJ423 * w[3] * w[1] * w[2] * HF423.T.dot(T).dot(HF423) +
                  detJ424 * w[3] * w[1] * w[3] * HF424.T.dot(T).dot(HF424) +
                  detJ431 * w[3] * w[2] * w[0] * HF431.T.dot(T).dot(HF431) +
                  detJ432 * w[3] * w[2] * w[1] * HF432.T.dot(T).dot(HF432) +
                  detJ433 * w[3] * w[2] * w[2] * HF433.T.dot(T).dot(HF433) +
                  detJ434 * w[3] * w[2] * w[3] * HF434.T.dot(T).dot(HF434) +
                  detJ441 * w[3] * w[3] * w[0] * HF441.T.dot(T).dot(HF441) +
                  detJ442 * w[3] * w[3] * w[1] * HF442.T.dot(T).dot(HF442) +
                  detJ443 * w[3] * w[3] * w[2] * HF443.T.dot(T).dot(HF443) +
                  detJ444 * w[3] * w[3] * w[3] * HF444.T.dot(T).dot(HF444) +                  
                  detJ111 * w[0] * w[0] * w[0] * HS111.T.dot(T).dot(HS111) +
                  detJ112 * w[0] * w[0] * w[1] * HS112.T.dot(T).dot(HS112) +
                  detJ113 * w[0] * w[0] * w[2] * HS113.T.dot(T).dot(HS113) +
                  detJ114 * w[0] * w[0] * w[3] * HS114.T.dot(T).dot(HS114) +
                  detJ121 * w[0] * w[1] * w[0] * HS121.T.dot(T).dot(HS121) +
                  detJ122 * w[0] * w[1] * w[1] * HS122.T.dot(T).dot(HS122) +
                  detJ123 * w[0] * w[1] * w[2] * HS123.T.dot(T).dot(HS123) +
                  detJ124 * w[0] * w[1] * w[3] * HS124.T.dot(T).dot(HS124) +
                  detJ131 * w[0] * w[2] * w[0] * HS131.T.dot(T).dot(HS131) +
                  detJ132 * w[0] * w[2] * w[1] * HS132.T.dot(T).dot(HS132) +
                  detJ133 * w[0] * w[2] * w[2] * HS133.T.dot(T).dot(HS133) +
                  detJ134 * w[0] * w[2] * w[3] * HS134.T.dot(T).dot(HS134) +
                  detJ141 * w[0] * w[3] * w[0] * HS141.T.dot(T).dot(HS141) +
                  detJ142 * w[0] * w[3] * w[1] * HS142.T.dot(T).dot(HS142) +
                  detJ143 * w[0] * w[3] * w[2] * HS143.T.dot(T).dot(HS143) +
                  detJ144 * w[0] * w[3] * w[3] * HS144.T.dot(T).dot(HS144) +                  
                  detJ211 * w[1] * w[0] * w[0] * HS211.T.dot(T).dot(HS211) +
                  detJ212 * w[1] * w[0] * w[1] * HS212.T.dot(T).dot(HS212) +
                  detJ213 * w[1] * w[0] * w[2] * HS213.T.dot(T).dot(HS213) +
                  detJ214 * w[1] * w[0] * w[3] * HS214.T.dot(T).dot(HS214) +
                  detJ221 * w[1] * w[1] * w[0] * HS221.T.dot(T).dot(HS221) +
                  detJ222 * w[1] * w[1] * w[1] * HS222.T.dot(T).dot(HS222) +
                  detJ223 * w[1] * w[1] * w[2] * HS223.T.dot(T).dot(HS223) +
                  detJ224 * w[1] * w[1] * w[3] * HS224.T.dot(T).dot(HS224) +
                  detJ231 * w[1] * w[2] * w[0] * HS231.T.dot(T).dot(HS231) +
                  detJ232 * w[1] * w[2] * w[1] * HS232.T.dot(T).dot(HS232) +
                  detJ233 * w[1] * w[2] * w[2] * HS233.T.dot(T).dot(HS233) +
                  detJ234 * w[1] * w[2] * w[3] * HS234.T.dot(T).dot(HS234) +
                  detJ241 * w[1] * w[3] * w[0] * HS241.T.dot(T).dot(HS241) +
                  detJ242 * w[1] * w[3] * w[1] * HS242.T.dot(T).dot(HS242) +
                  detJ243 * w[1] * w[3] * w[2] * HS243.T.dot(T).dot(HS243) +
                  detJ244 * w[1] * w[3] * w[3] * HS244.T.dot(T).dot(HS244) +                  
                  detJ311 * w[2] * w[0] * w[0] * HS311.T.dot(T).dot(HS311) +
                  detJ312 * w[2] * w[0] * w[1] * HS312.T.dot(T).dot(HS312) +
                  detJ313 * w[2] * w[0] * w[2] * HS313.T.dot(T).dot(HS313) +
                  detJ314 * w[2] * w[0] * w[3] * HS314.T.dot(T).dot(HS314) +
                  detJ321 * w[2] * w[1] * w[0] * HS321.T.dot(T).dot(HS321) +
                  detJ322 * w[2] * w[1] * w[1] * HS322.T.dot(T).dot(HS322) +
                  detJ323 * w[2] * w[1] * w[2] * HS323.T.dot(T).dot(HS323) +
                  detJ324 * w[2] * w[1] * w[3] * HS324.T.dot(T).dot(HS324) +
                  detJ331 * w[2] * w[2] * w[0] * HS331.T.dot(T).dot(HS331) +
                  detJ332 * w[2] * w[2] * w[1] * HS332.T.dot(T).dot(HS332) +
                  detJ333 * w[2] * w[2] * w[2] * HS333.T.dot(T).dot(HS333) +
                  detJ334 * w[2] * w[2] * w[3] * HS334.T.dot(T).dot(HS334) +
                  detJ341 * w[2] * w[3] * w[0] * HS341.T.dot(T).dot(HS341) +
                  detJ342 * w[2] * w[3] * w[1] * HS342.T.dot(T).dot(HS342) +
                  detJ343 * w[2] * w[3] * w[2] * HS343.T.dot(T).dot(HS343) +
                  detJ344 * w[2] * w[3] * w[3] * HS344.T.dot(T).dot(HS344) +                  
                  detJ411 * w[3] * w[0] * w[0] * HS411.T.dot(T).dot(HS411) +
                  detJ412 * w[3] * w[0] * w[1] * HS412.T.dot(T).dot(HS412) +
                  detJ413 * w[3] * w[0] * w[2] * HS413.T.dot(T).dot(HS413) +
                  detJ414 * w[3] * w[0] * w[3] * HS414.T.dot(T).dot(HS414) +
                  detJ421 * w[3] * w[1] * w[0] * HS421.T.dot(T).dot(HS421) +
                  detJ422 * w[3] * w[1] * w[1] * HS422.T.dot(T).dot(HS422) +
                  detJ423 * w[3] * w[1] * w[2] * HS423.T.dot(T).dot(HS423) +
                  detJ424 * w[3] * w[1] * w[3] * HS424.T.dot(T).dot(HS424) +
                  detJ431 * w[3] * w[2] * w[0] * HS431.T.dot(T).dot(HS431) +
                  detJ432 * w[3] * w[2] * w[1] * HS432.T.dot(T).dot(HS432) +
                  detJ433 * w[3] * w[2] * w[2] * HS433.T.dot(T).dot(HS433) +
                  detJ434 * w[3] * w[2] * w[3] * HS434.T.dot(T).dot(HS434) +
                  detJ441 * w[3] * w[3] * w[0] * HS441.T.dot(T).dot(HS441) +
                  detJ442 * w[3] * w[3] * w[1] * HS442.T.dot(T).dot(HS442) +
                  detJ443 * w[3] * w[3] * w[2] * HS443.T.dot(T).dot(HS443) +
                  detJ444 * w[3] * w[3] * w[3] * HS444.T.dot(T).dot(HS444) +                  
                  detJ111 * w[0] * w[0] * w[0] * HT111.T.dot(T).dot(HT111) +
                  detJ112 * w[0] * w[0] * w[1] * HT112.T.dot(T).dot(HT112) +
                  detJ113 * w[0] * w[0] * w[2] * HT113.T.dot(T).dot(HT113) +
                  detJ114 * w[0] * w[0] * w[3] * HT114.T.dot(T).dot(HT114) +
                  detJ121 * w[0] * w[1] * w[0] * HT121.T.dot(T).dot(HT121) +
                  detJ122 * w[0] * w[1] * w[1] * HT122.T.dot(T).dot(HT122) +
                  detJ123 * w[0] * w[1] * w[2] * HT123.T.dot(T).dot(HT123) +
                  detJ124 * w[0] * w[1] * w[3] * HT124.T.dot(T).dot(HT124) +
                  detJ131 * w[0] * w[2] * w[0] * HT131.T.dot(T).dot(HT131) +
                  detJ132 * w[0] * w[2] * w[1] * HT132.T.dot(T).dot(HT132) +
                  detJ133 * w[0] * w[2] * w[2] * HT133.T.dot(T).dot(HT133) +
                  detJ134 * w[0] * w[2] * w[3] * HT134.T.dot(T).dot(HT134) +
                  detJ141 * w[0] * w[3] * w[0] * HT141.T.dot(T).dot(HT141) +
                  detJ142 * w[0] * w[3] * w[1] * HT142.T.dot(T).dot(HT142) +
                  detJ143 * w[0] * w[3] * w[2] * HT143.T.dot(T).dot(HT143) +
                  detJ144 * w[0] * w[3] * w[3] * HT144.T.dot(T).dot(HT144) +                  
                  detJ211 * w[1] * w[0] * w[0] * HT211.T.dot(T).dot(HT211) +
                  detJ212 * w[1] * w[0] * w[1] * HT212.T.dot(T).dot(HT212) +
                  detJ213 * w[1] * w[0] * w[2] * HT213.T.dot(T).dot(HT213) +
                  detJ214 * w[1] * w[0] * w[3] * HT214.T.dot(T).dot(HT214) +
                  detJ221 * w[1] * w[1] * w[0] * HT221.T.dot(T).dot(HT221) +
                  detJ222 * w[1] * w[1] * w[1] * HT222.T.dot(T).dot(HT222) +
                  detJ223 * w[1] * w[1] * w[2] * HT223.T.dot(T).dot(HT223) +
                  detJ224 * w[1] * w[1] * w[3] * HT224.T.dot(T).dot(HT224) +
                  detJ231 * w[1] * w[2] * w[0] * HT231.T.dot(T).dot(HT231) +
                  detJ232 * w[1] * w[2] * w[1] * HT232.T.dot(T).dot(HT232) +
                  detJ233 * w[1] * w[2] * w[2] * HT233.T.dot(T).dot(HT233) +
                  detJ234 * w[1] * w[2] * w[3] * HT234.T.dot(T).dot(HT234) +
                  detJ241 * w[1] * w[3] * w[0] * HT241.T.dot(T).dot(HT241) +
                  detJ242 * w[1] * w[3] * w[1] * HT242.T.dot(T).dot(HT242) +
                  detJ243 * w[1] * w[3] * w[2] * HT243.T.dot(T).dot(HT243) +
                  detJ244 * w[1] * w[3] * w[3] * HT244.T.dot(T).dot(HT244) +                  
                  detJ311 * w[2] * w[0] * w[0] * HT311.T.dot(T).dot(HT311) +
                  detJ312 * w[2] * w[0] * w[1] * HT312.T.dot(T).dot(HT312) +
                  detJ313 * w[2] * w[0] * w[2] * HT313.T.dot(T).dot(HT313) +
                  detJ314 * w[2] * w[0] * w[3] * HT314.T.dot(T).dot(HT314) +
                  detJ321 * w[2] * w[1] * w[0] * HT321.T.dot(T).dot(HT321) +
                  detJ322 * w[2] * w[1] * w[1] * HT322.T.dot(T).dot(HT322) +
                  detJ323 * w[2] * w[1] * w[2] * HT323.T.dot(T).dot(HT323) +
                  detJ324 * w[2] * w[1] * w[3] * HT324.T.dot(T).dot(HT324) +
                  detJ331 * w[2] * w[2] * w[0] * HT331.T.dot(T).dot(HT331) +
                  detJ332 * w[2] * w[2] * w[1] * HT332.T.dot(T).dot(HT332) +
                  detJ333 * w[2] * w[2] * w[2] * HT333.T.dot(T).dot(HT333) +
                  detJ334 * w[2] * w[2] * w[3] * HT334.T.dot(T).dot(HT334) +
                  detJ341 * w[2] * w[3] * w[0] * HT341.T.dot(T).dot(HT341) +
                  detJ342 * w[2] * w[3] * w[1] * HT342.T.dot(T).dot(HT342) +
                  detJ343 * w[2] * w[3] * w[2] * HT343.T.dot(T).dot(HT343) +
                  detJ344 * w[2] * w[3] * w[3] * HT344.T.dot(T).dot(HT344) +                  
                  detJ411 * w[3] * w[0] * w[0] * HT411.T.dot(T).dot(HT411) +
                  detJ412 * w[3] * w[0] * w[1] * HT412.T.dot(T).dot(HT412) +
                  detJ413 * w[3] * w[0] * w[2] * HT413.T.dot(T).dot(HT413) +
                  detJ414 * w[3] * w[0] * w[3] * HT414.T.dot(T).dot(HT414) +
                  detJ421 * w[3] * w[1] * w[0] * HT421.T.dot(T).dot(HT421) +
                  detJ422 * w[3] * w[1] * w[1] * HT422.T.dot(T).dot(HT422) +
                  detJ423 * w[3] * w[1] * w[2] * HT423.T.dot(T).dot(HT423) +
                  detJ424 * w[3] * w[1] * w[3] * HT424.T.dot(T).dot(HT424) +
                  detJ431 * w[3] * w[2] * w[0] * HT431.T.dot(T).dot(HT431) +
                  detJ432 * w[3] * w[2] * w[1] * HT432.T.dot(T).dot(HT432) +
                  detJ433 * w[3] * w[2] * w[2] * HT433.T.dot(T).dot(HT433) +
                  detJ434 * w[3] * w[2] * w[3] * HT434.T.dot(T).dot(HT434) +
                  detJ441 * w[3] * w[3] * w[0] * HT441.T.dot(T).dot(HT441) +
                  detJ442 * w[3] * w[3] * w[1] * HT442.T.dot(T).dot(HT442) +
                  detJ443 * w[3] * w[3] * w[2] * HT443.T.dot(T).dot(HT443) +
                  detJ444 * w[3] * w[3] * w[3] * HT444.T.dot(T).dot(HT444))
            
            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS

        else:  # gaussPts = 5
            # 'natural' weights of the element
            wgt1 = -2.0 / 15.0
            wgt2 = 3.0 / 40.0
            w = np.array([wgt1, wgt2, wgt2, wgt2, wgt2])

            jacob111 = self._shapeFns[111].jacobian(x1, x2, x3, x4)
            jacob112 = self._shapeFns[112].jacobian(x1, x2, x3, x4)
            jacob113 = self._shapeFns[113].jacobian(x1, x2, x3, x4)
            jacob114 = self._shapeFns[114].jacobian(x1, x2, x3, x4)
            jacob115 = self._shapeFns[115].jacobian(x1, x2, x3, x4)
            
            jacob121 = self._shapeFns[121].jacobian(x1, x2, x3, x4)
            jacob122 = self._shapeFns[122].jacobian(x1, x2, x3, x4)
            jacob123 = self._shapeFns[123].jacobian(x1, x2, x3, x4)
            jacob124 = self._shapeFns[124].jacobian(x1, x2, x3, x4)
            jacob125 = self._shapeFns[125].jacobian(x1, x2, x3, x4)
            
            jacob131 = self._shapeFns[131].jacobian(x1, x2, x3, x4)
            jacob132 = self._shapeFns[132].jacobian(x1, x2, x3, x4)
            jacob133 = self._shapeFns[133].jacobian(x1, x2, x3, x4)
            jacob134 = self._shapeFns[134].jacobian(x1, x2, x3, x4)
            jacob135 = self._shapeFns[135].jacobian(x1, x2, x3, x4)
            
            jacob141 = self._shapeFns[141].jacobian(x1, x2, x3, x4)
            jacob142 = self._shapeFns[142].jacobian(x1, x2, x3, x4)
            jacob143 = self._shapeFns[143].jacobian(x1, x2, x3, x4)
            jacob144 = self._shapeFns[144].jacobian(x1, x2, x3, x4)
            jacob145 = self._shapeFns[145].jacobian(x1, x2, x3, x4)
            
            jacob151 = self._shapeFns[151].jacobian(x1, x2, x3, x4)
            jacob152 = self._shapeFns[152].jacobian(x1, x2, x3, x4)
            jacob153 = self._shapeFns[153].jacobian(x1, x2, x3, x4)
            jacob154 = self._shapeFns[154].jacobian(x1, x2, x3, x4)
            jacob155 = self._shapeFns[155].jacobian(x1, x2, x3, x4)
            
            
            jacob211 = self._shapeFns[211].jacobian(x1, x2, x3, x4)
            jacob212 = self._shapeFns[212].jacobian(x1, x2, x3, x4)
            jacob213 = self._shapeFns[213].jacobian(x1, x2, x3, x4)
            jacob214 = self._shapeFns[214].jacobian(x1, x2, x3, x4)
            jacob215 = self._shapeFns[215].jacobian(x1, x2, x3, x4)
                      
            jacob221 = self._shapeFns[221].jacobian(x1, x2, x3, x4)
            jacob222 = self._shapeFns[222].jacobian(x1, x2, x3, x4)
            jacob223 = self._shapeFns[223].jacobian(x1, x2, x3, x4)
            jacob224 = self._shapeFns[224].jacobian(x1, x2, x3, x4)
            jacob225 = self._shapeFns[225].jacobian(x1, x2, x3, x4)
            
            jacob231 = self._shapeFns[231].jacobian(x1, x2, x3, x4)
            jacob232 = self._shapeFns[232].jacobian(x1, x2, x3, x4)
            jacob233 = self._shapeFns[233].jacobian(x1, x2, x3, x4)
            jacob234 = self._shapeFns[234].jacobian(x1, x2, x3, x4)
            jacob235 = self._shapeFns[235].jacobian(x1, x2, x3, x4)
            
            jacob241 = self._shapeFns[241].jacobian(x1, x2, x3, x4)
            jacob242 = self._shapeFns[242].jacobian(x1, x2, x3, x4)
            jacob243 = self._shapeFns[243].jacobian(x1, x2, x3, x4)
            jacob244 = self._shapeFns[244].jacobian(x1, x2, x3, x4)
            jacob245 = self._shapeFns[245].jacobian(x1, x2, x3, x4)
            
            jacob251 = self._shapeFns[251].jacobian(x1, x2, x3, x4)
            jacob252 = self._shapeFns[252].jacobian(x1, x2, x3, x4)
            jacob253 = self._shapeFns[253].jacobian(x1, x2, x3, x4)
            jacob254 = self._shapeFns[254].jacobian(x1, x2, x3, x4)
            jacob255 = self._shapeFns[255].jacobian(x1, x2, x3, x4)
            
            
            jacob311 = self._shapeFns[311].jacobian(x1, x2, x3, x4)
            jacob312 = self._shapeFns[312].jacobian(x1, x2, x3, x4)
            jacob313 = self._shapeFns[313].jacobian(x1, x2, x3, x4)
            jacob314 = self._shapeFns[314].jacobian(x1, x2, x3, x4)
            jacob315 = self._shapeFns[315].jacobian(x1, x2, x3, x4)
            
            jacob321 = self._shapeFns[321].jacobian(x1, x2, x3, x4)
            jacob322 = self._shapeFns[322].jacobian(x1, x2, x3, x4)
            jacob323 = self._shapeFns[323].jacobian(x1, x2, x3, x4)
            jacob324 = self._shapeFns[324].jacobian(x1, x2, x3, x4)
            jacob325 = self._shapeFns[325].jacobian(x1, x2, x3, x4)
            
            jacob331 = self._shapeFns[331].jacobian(x1, x2, x3, x4)
            jacob332 = self._shapeFns[332].jacobian(x1, x2, x3, x4)
            jacob333 = self._shapeFns[333].jacobian(x1, x2, x3, x4)
            jacob334 = self._shapeFns[334].jacobian(x1, x2, x3, x4)
            jacob335 = self._shapeFns[335].jacobian(x1, x2, x3, x4)
            
            jacob341 = self._shapeFns[341].jacobian(x1, x2, x3, x4)
            jacob342 = self._shapeFns[342].jacobian(x1, x2, x3, x4)
            jacob343 = self._shapeFns[343].jacobian(x1, x2, x3, x4)
            jacob344 = self._shapeFns[344].jacobian(x1, x2, x3, x4)
            jacob345 = self._shapeFns[345].jacobian(x1, x2, x3, x4)
            
            jacob351 = self._shapeFns[351].jacobian(x1, x2, x3, x4)
            jacob352 = self._shapeFns[352].jacobian(x1, x2, x3, x4)
            jacob353 = self._shapeFns[353].jacobian(x1, x2, x3, x4)
            jacob354 = self._shapeFns[354].jacobian(x1, x2, x3, x4)
            jacob355 = self._shapeFns[355].jacobian(x1, x2, x3, x4)
            
            
            jacob411 = self._shapeFns[411].jacobian(x1, x2, x3, x4)
            jacob412 = self._shapeFns[412].jacobian(x1, x2, x3, x4)
            jacob413 = self._shapeFns[413].jacobian(x1, x2, x3, x4)
            jacob414 = self._shapeFns[414].jacobian(x1, x2, x3, x4)
            jacob415 = self._shapeFns[415].jacobian(x1, x2, x3, x4)
            
            jacob421 = self._shapeFns[421].jacobian(x1, x2, x3, x4)
            jacob422 = self._shapeFns[422].jacobian(x1, x2, x3, x4)
            jacob423 = self._shapeFns[423].jacobian(x1, x2, x3, x4)
            jacob424 = self._shapeFns[424].jacobian(x1, x2, x3, x4)
            jacob425 = self._shapeFns[425].jacobian(x1, x2, x3, x4)
            
            jacob431 = self._shapeFns[431].jacobian(x1, x2, x3, x4)
            jacob432 = self._shapeFns[432].jacobian(x1, x2, x3, x4)
            jacob433 = self._shapeFns[433].jacobian(x1, x2, x3, x4)
            jacob434 = self._shapeFns[434].jacobian(x1, x2, x3, x4)
            jacob435 = self._shapeFns[435].jacobian(x1, x2, x3, x4)
            
            jacob441 = self._shapeFns[441].jacobian(x1, x2, x3, x4)
            jacob442 = self._shapeFns[442].jacobian(x1, x2, x3, x4)
            jacob443 = self._shapeFns[443].jacobian(x1, x2, x3, x4)
            jacob444 = self._shapeFns[444].jacobian(x1, x2, x3, x4)
            jacob445 = self._shapeFns[445].jacobian(x1, x2, x3, x4)
            
            jacob451 = self._shapeFns[451].jacobian(x1, x2, x3, x4)
            jacob452 = self._shapeFns[452].jacobian(x1, x2, x3, x4)
            jacob453 = self._shapeFns[453].jacobian(x1, x2, x3, x4)
            jacob454 = self._shapeFns[454].jacobian(x1, x2, x3, x4)
            jacob455 = self._shapeFns[455].jacobian(x1, x2, x3, x4)
            
            
            jacob511 = self._shapeFns[511].jacobian(x1, x2, x3, x4)
            jacob512 = self._shapeFns[512].jacobian(x1, x2, x3, x4)
            jacob513 = self._shapeFns[513].jacobian(x1, x2, x3, x4)
            jacob514 = self._shapeFns[514].jacobian(x1, x2, x3, x4)
            jacob515 = self._shapeFns[515].jacobian(x1, x2, x3, x4)
            
            jacob521 = self._shapeFns[521].jacobian(x1, x2, x3, x4)
            jacob522 = self._shapeFns[522].jacobian(x1, x2, x3, x4)
            jacob523 = self._shapeFns[523].jacobian(x1, x2, x3, x4)
            jacob524 = self._shapeFns[524].jacobian(x1, x2, x3, x4)
            jacob525 = self._shapeFns[525].jacobian(x1, x2, x3, x4)
            
            jacob531 = self._shapeFns[531].jacobian(x1, x2, x3, x4)
            jacob532 = self._shapeFns[532].jacobian(x1, x2, x3, x4)
            jacob533 = self._shapeFns[533].jacobian(x1, x2, x3, x4)
            jacob534 = self._shapeFns[534].jacobian(x1, x2, x3, x4)
            jacob535 = self._shapeFns[535].jacobian(x1, x2, x3, x4)
            
            jacob541 = self._shapeFns[541].jacobian(x1, x2, x3, x4)
            jacob542 = self._shapeFns[542].jacobian(x1, x2, x3, x4)
            jacob543 = self._shapeFns[543].jacobian(x1, x2, x3, x4)
            jacob544 = self._shapeFns[544].jacobian(x1, x2, x3, x4)
            jacob545 = self._shapeFns[545].jacobian(x1, x2, x3, x4)
            
            jacob551 = self._shapeFns[551].jacobian(x1, x2, x3, x4)
            jacob552 = self._shapeFns[552].jacobian(x1, x2, x3, x4)
            jacob553 = self._shapeFns[553].jacobian(x1, x2, x3, x4)
            jacob554 = self._shapeFns[554].jacobian(x1, x2, x3, x4)
            jacob555 = self._shapeFns[555].jacobian(x1, x2, x3, x4)
            
            


            # determinant of the Jacobian matrix
            detJ111 = det(jacob111)
            detJ112 = det(jacob112)
            detJ113 = det(jacob113)
            detJ114 = det(jacob114)
            detJ115 = det(jacob115)
            
            detJ121 = det(jacob121)
            detJ122 = det(jacob122)
            detJ123 = det(jacob123)
            detJ124 = det(jacob124)
            detJ125 = det(jacob125)
            
            detJ131 = det(jacob131)
            detJ132 = det(jacob132)
            detJ133 = det(jacob133)
            detJ134 = det(jacob134)
            detJ135 = det(jacob135)
            
            detJ141 = det(jacob141)
            detJ142 = det(jacob142)
            detJ143 = det(jacob143)
            detJ144 = det(jacob144)
            detJ145 = det(jacob145)
            
            detJ151 = det(jacob151)
            detJ152 = det(jacob152)
            detJ153 = det(jacob153)
            detJ154 = det(jacob154)
            detJ155 = det(jacob155)
            
            
            detJ211 = det(jacob211)
            detJ212 = det(jacob212)
            detJ213 = det(jacob213)
            detJ214 = det(jacob214)
            detJ215 = det(jacob215)
            
            detJ221 = det(jacob221)
            detJ222 = det(jacob222)
            detJ223 = det(jacob223)
            detJ224 = det(jacob224)
            detJ225 = det(jacob225)
            
            detJ231 = det(jacob231)
            detJ232 = det(jacob232)
            detJ233 = det(jacob233)
            detJ234 = det(jacob234)
            detJ235 = det(jacob235)
            
            detJ241 = det(jacob241)
            detJ242 = det(jacob242)
            detJ243 = det(jacob243)
            detJ244 = det(jacob244)
            detJ245 = det(jacob245)
            
            detJ251 = det(jacob251)
            detJ252 = det(jacob252)
            detJ253 = det(jacob253)
            detJ254 = det(jacob254)
            detJ255 = det(jacob255)
            
            
            detJ311 = det(jacob311)
            detJ312 = det(jacob312)
            detJ313 = det(jacob313)
            detJ314 = det(jacob314)
            detJ315 = det(jacob315)
            
            detJ321 = det(jacob321)
            detJ322 = det(jacob322)
            detJ323 = det(jacob323)
            detJ324 = det(jacob324)
            detJ325 = det(jacob325)
            
            detJ331 = det(jacob331)
            detJ332 = det(jacob332)
            detJ333 = det(jacob333)
            detJ334 = det(jacob334)
            detJ335 = det(jacob335)
            
            detJ341 = det(jacob341)
            detJ342 = det(jacob342)
            detJ343 = det(jacob343)
            detJ344 = det(jacob344)
            detJ345 = det(jacob345)
            
            detJ351 = det(jacob351)
            detJ352 = det(jacob352)
            detJ353 = det(jacob353)
            detJ354 = det(jacob354)
            detJ355 = det(jacob355)
            
            
            detJ411 = det(jacob411)
            detJ412 = det(jacob412)
            detJ413 = det(jacob413)
            detJ414 = det(jacob414)
            detJ415 = det(jacob415)
            
            detJ421 = det(jacob421)
            detJ422 = det(jacob422)
            detJ423 = det(jacob423)
            detJ424 = det(jacob424)
            detJ425 = det(jacob425)
            
            detJ431 = det(jacob431)
            detJ432 = det(jacob432)
            detJ433 = det(jacob433)
            detJ434 = det(jacob434)
            detJ435 = det(jacob435)
            
            detJ441 = det(jacob441)
            detJ442 = det(jacob442)
            detJ443 = det(jacob443)
            detJ444 = det(jacob444)
            detJ445 = det(jacob445)
            
            detJ451 = det(jacob451)
            detJ452 = det(jacob452)
            detJ453 = det(jacob453)
            detJ454 = det(jacob454)
            detJ455 = det(jacob455)
            
        
            detJ511 = det(jacob511)
            detJ512 = det(jacob512)
            detJ513 = det(jacob513)
            detJ514 = det(jacob514)
            detJ515 = det(jacob515)
            
            detJ521 = det(jacob521)
            detJ522 = det(jacob522)
            detJ523 = det(jacob523)
            detJ524 = det(jacob524)
            detJ525 = det(jacob525)
            
            detJ531 = det(jacob531)
            detJ532 = det(jacob532)
            detJ533 = det(jacob533)
            detJ534 = det(jacob534)
            detJ535 = det(jacob535)
            
            detJ541 = det(jacob541)
            detJ542 = det(jacob542)
            detJ543 = det(jacob543)
            detJ544 = det(jacob544)
            detJ545 = det(jacob545)
            
            detJ551 = det(jacob551)
            detJ552 = det(jacob552)
            detJ553 = det(jacob553)
            detJ554 = det(jacob554)
            detJ555 = det(jacob555)
            
            # create the linear Bmatrix
            BL111 = self._shapeFns[111].BLinear(x1, x2, x3, x4) 
            BL112 = self._shapeFns[112].BLinear(x1, x2, x3, x4) 
            BL113 = self._shapeFns[113].BLinear(x1, x2, x3, x4)
            BL114 = self._shapeFns[114].BLinear(x1, x2, x3, x4)
            BL115 = self._shapeFns[115].BLinear(x1, x2, x3, x4)
            
            BL121 = self._shapeFns[121].BLinear(x1, x2, x3, x4)
            BL122 = self._shapeFns[122].BLinear(x1, x2, x3, x4) 
            BL123 = self._shapeFns[123].BLinear(x1, x2, x3, x4) 
            BL124 = self._shapeFns[124].BLinear(x1, x2, x3, x4)
            BL125 = self._shapeFns[125].BLinear(x1, x2, x3, x4)
            
            BL131 = self._shapeFns[131].BLinear(x1, x2, x3, x4) 
            BL132 = self._shapeFns[132].BLinear(x1, x2, x3, x4) 
            BL133 = self._shapeFns[133].BLinear(x1, x2, x3, x4) 
            BL134 = self._shapeFns[134].BLinear(x1, x2, x3, x4) 
            BL135 = self._shapeFns[135].BLinear(x1, x2, x3, x4)
            
            BL141 = self._shapeFns[141].BLinear(x1, x2, x3, x4)
            BL142 = self._shapeFns[142].BLinear(x1, x2, x3, x4) 
            BL143 = self._shapeFns[143].BLinear(x1, x2, x3, x4) 
            BL144 = self._shapeFns[144].BLinear(x1, x2, x3, x4) 
            BL145 = self._shapeFns[145].BLinear(x1, x2, x3, x4)
            
            BL151 = self._shapeFns[151].BLinear(x1, x2, x3, x4)
            BL152 = self._shapeFns[152].BLinear(x1, x2, x3, x4) 
            BL153 = self._shapeFns[153].BLinear(x1, x2, x3, x4) 
            BL154 = self._shapeFns[154].BLinear(x1, x2, x3, x4) 
            BL155 = self._shapeFns[155].BLinear(x1, x2, x3, x4)
            
            
            BL211 = self._shapeFns[211].BLinear(x1, x2, x3, x4) 
            BL212 = self._shapeFns[212].BLinear(x1, x2, x3, x4) 
            BL213 = self._shapeFns[213].BLinear(x1, x2, x3, x4)
            BL214 = self._shapeFns[214].BLinear(x1, x2, x3, x4) 
            BL215 = self._shapeFns[215].BLinear(x1, x2, x3, x4)
            
    
            BL221 = self._shapeFns[221].BLinear(x1, x2, x3, x4)
            BL222 = self._shapeFns[222].BLinear(x1, x2, x3, x4) 
            BL223 = self._shapeFns[223].BLinear(x1, x2, x3, x4) 
            BL224 = self._shapeFns[224].BLinear(x1, x2, x3, x4) 
            BL225 = self._shapeFns[225].BLinear(x1, x2, x3, x4)
        
            BL231 = self._shapeFns[231].BLinear(x1, x2, x3, x4) 
            BL232 = self._shapeFns[232].BLinear(x1, x2, x3, x4) 
            BL233 = self._shapeFns[233].BLinear(x1, x2, x3, x4) 
            BL234 = self._shapeFns[234].BLinear(x1, x2, x3, x4) 
            BL235 = self._shapeFns[235].BLinear(x1, x2, x3, x4)
            
            BL241 = self._shapeFns[241].BLinear(x1, x2, x3, x4)
            BL242 = self._shapeFns[242].BLinear(x1, x2, x3, x4) 
            BL243 = self._shapeFns[243].BLinear(x1, x2, x3, x4) 
            BL244 = self._shapeFns[244].BLinear(x1, x2, x3, x4)
            BL245 = self._shapeFns[245].BLinear(x1, x2, x3, x4)
            
            BL251 = self._shapeFns[251].BLinear(x1, x2, x3, x4)
            BL252 = self._shapeFns[252].BLinear(x1, x2, x3, x4) 
            BL253 = self._shapeFns[253].BLinear(x1, x2, x3, x4) 
            BL254 = self._shapeFns[254].BLinear(x1, x2, x3, x4)
            BL255 = self._shapeFns[255].BLinear(x1, x2, x3, x4)
            
            
            BL311 = self._shapeFns[311].BLinear(x1, x2, x3, x4) 
            BL312 = self._shapeFns[312].BLinear(x1, x2, x3, x4) 
            BL313 = self._shapeFns[313].BLinear(x1, x2, x3, x4)
            BL314 = self._shapeFns[314].BLinear(x1, x2, x3, x4) 
            BL315 = self._shapeFns[315].BLinear(x1, x2, x3, x4)
            
            BL321 = self._shapeFns[321].BLinear(x1, x2, x3, x4)
            BL322 = self._shapeFns[322].BLinear(x1, x2, x3, x4) 
            BL323 = self._shapeFns[323].BLinear(x1, x2, x3, x4) 
            BL324 = self._shapeFns[324].BLinear(x1, x2, x3, x4) 
            BL325 = self._shapeFns[325].BLinear(x1, x2, x3, x4)
            
            BL331 = self._shapeFns[331].BLinear(x1, x2, x3, x4) 
            BL332 = self._shapeFns[332].BLinear(x1, x2, x3, x4) 
            BL333 = self._shapeFns[333].BLinear(x1, x2, x3, x4) 
            BL334 = self._shapeFns[334].BLinear(x1, x2, x3, x4)
            BL335 = self._shapeFns[335].BLinear(x1, x2, x3, x4)
            
            BL341 = self._shapeFns[341].BLinear(x1, x2, x3, x4)
            BL342 = self._shapeFns[342].BLinear(x1, x2, x3, x4) 
            BL343 = self._shapeFns[343].BLinear(x1, x2, x3, x4) 
            BL344 = self._shapeFns[344].BLinear(x1, x2, x3, x4)
            BL345 = self._shapeFns[345].BLinear(x1, x2, x3, x4)
            
            BL351 = self._shapeFns[351].BLinear(x1, x2, x3, x4)
            BL352 = self._shapeFns[352].BLinear(x1, x2, x3, x4) 
            BL353 = self._shapeFns[353].BLinear(x1, x2, x3, x4) 
            BL354 = self._shapeFns[354].BLinear(x1, x2, x3, x4)
            BL355 = self._shapeFns[355].BLinear(x1, x2, x3, x4)
            
            
            BL411 = self._shapeFns[411].BLinear(x1, x2, x3, x4) 
            BL412 = self._shapeFns[412].BLinear(x1, x2, x3, x4) 
            BL413 = self._shapeFns[413].BLinear(x1, x2, x3, x4)
            BL414 = self._shapeFns[414].BLinear(x1, x2, x3, x4)
            BL415 = self._shapeFns[415].BLinear(x1, x2, x3, x4)
            
            BL421 = self._shapeFns[421].BLinear(x1, x2, x3, x4)
            BL422 = self._shapeFns[422].BLinear(x1, x2, x3, x4) 
            BL423 = self._shapeFns[423].BLinear(x1, x2, x3, x4) 
            BL424 = self._shapeFns[424].BLinear(x1, x2, x3, x4) 
            BL425 = self._shapeFns[425].BLinear(x1, x2, x3, x4)
            
            BL431 = self._shapeFns[431].BLinear(x1, x2, x3, x4) 
            BL432 = self._shapeFns[432].BLinear(x1, x2, x3, x4) 
            BL433 = self._shapeFns[433].BLinear(x1, x2, x3, x4) 
            BL434 = self._shapeFns[434].BLinear(x1, x2, x3, x4) 
            BL435 = self._shapeFns[435].BLinear(x1, x2, x3, x4)
            
            BL441 = self._shapeFns[441].BLinear(x1, x2, x3, x4)
            BL442 = self._shapeFns[442].BLinear(x1, x2, x3, x4) 
            BL443 = self._shapeFns[443].BLinear(x1, x2, x3, x4) 
            BL444 = self._shapeFns[444].BLinear(x1, x2, x3, x4) 
            BL445 = self._shapeFns[445].BLinear(x1, x2, x3, x4)
            
            BL451 = self._shapeFns[451].BLinear(x1, x2, x3, x4)
            BL452 = self._shapeFns[452].BLinear(x1, x2, x3, x4) 
            BL453 = self._shapeFns[453].BLinear(x1, x2, x3, x4) 
            BL454 = self._shapeFns[454].BLinear(x1, x2, x3, x4) 
            BL455 = self._shapeFns[455].BLinear(x1, x2, x3, x4)
            
            
            BL511 = self._shapeFns[411].BLinear(x1, x2, x3, x4) 
            BL512 = self._shapeFns[412].BLinear(x1, x2, x3, x4) 
            BL513 = self._shapeFns[413].BLinear(x1, x2, x3, x4)
            BL514 = self._shapeFns[414].BLinear(x1, x2, x3, x4)
            BL515 = self._shapeFns[415].BLinear(x1, x2, x3, x4)
        
            BL521 = self._shapeFns[421].BLinear(x1, x2, x3, x4)
            BL522 = self._shapeFns[422].BLinear(x1, x2, x3, x4) 
            BL523 = self._shapeFns[423].BLinear(x1, x2, x3, x4) 
            BL524 = self._shapeFns[424].BLinear(x1, x2, x3, x4) 
            BL525 = self._shapeFns[425].BLinear(x1, x2, x3, x4)
            
            BL531 = self._shapeFns[431].BLinear(x1, x2, x3, x4) 
            BL532 = self._shapeFns[432].BLinear(x1, x2, x3, x4) 
            BL533 = self._shapeFns[433].BLinear(x1, x2, x3, x4) 
            BL534 = self._shapeFns[434].BLinear(x1, x2, x3, x4) 
            BL535 = self._shapeFns[435].BLinear(x1, x2, x3, x4)
            
            BL541 = self._shapeFns[441].BLinear(x1, x2, x3, x4)
            BL542 = self._shapeFns[442].BLinear(x1, x2, x3, x4) 
            BL543 = self._shapeFns[443].BLinear(x1, x2, x3, x4) 
            BL544 = self._shapeFns[444].BLinear(x1, x2, x3, x4) 
            BL545 = self._shapeFns[445].BLinear(x1, x2, x3, x4)
            
            BL551 = self._shapeFns[551].BLinear(x1, x2, x3, x4)
            BL552 = self._shapeFns[552].BLinear(x1, x2, x3, x4) 
            BL553 = self._shapeFns[553].BLinear(x1, x2, x3, x4) 
            BL554 = self._shapeFns[554].BLinear(x1, x2, x3, x4) 
            BL555 = self._shapeFns[555].BLinear(x1, x2, x3, x4)

            # the consistent mass matrix for 5 Gauss points
            KL = (detJ111 * w[0] * w[0] * w[0] * BL111.T.dot(M).dot(BL111) +
                  detJ112 * w[0] * w[0] * w[1] * BL112.T.dot(M).dot(BL112) +
                  detJ113 * w[0] * w[0] * w[2] * BL113.T.dot(M).dot(BL113) +
                  detJ114 * w[0] * w[0] * w[3] * BL114.T.dot(M).dot(BL114) +
                  detJ115 * w[0] * w[0] * w[4] * BL115.T.dot(M).dot(BL115) +
                  detJ121 * w[0] * w[1] * w[0] * BL121.T.dot(M).dot(BL121) +
                  detJ122 * w[0] * w[1] * w[1] * BL122.T.dot(M).dot(BL122) +
                  detJ123 * w[0] * w[1] * w[2] * BL123.T.dot(M).dot(BL123) +
                  detJ124 * w[0] * w[1] * w[3] * BL124.T.dot(M).dot(BL124) +
                  detJ125 * w[0] * w[1] * w[4] * BL125.T.dot(M).dot(BL125) +
                  detJ131 * w[0] * w[2] * w[0] * BL131.T.dot(M).dot(BL131) +
                  detJ132 * w[0] * w[2] * w[1] * BL132.T.dot(M).dot(BL132) +
                  detJ133 * w[0] * w[2] * w[2] * BL133.T.dot(M).dot(BL133) +
                  detJ134 * w[0] * w[2] * w[3] * BL134.T.dot(M).dot(BL134) +
                  detJ135 * w[0] * w[2] * w[4] * BL135.T.dot(M).dot(BL135) +
                  detJ141 * w[0] * w[3] * w[0] * BL141.T.dot(M).dot(BL141) +
                  detJ142 * w[0] * w[3] * w[1] * BL142.T.dot(M).dot(BL142) +
                  detJ143 * w[0] * w[3] * w[2] * BL143.T.dot(M).dot(BL143) +
                  detJ144 * w[0] * w[3] * w[3] * BL144.T.dot(M).dot(BL144) + 
                  detJ145 * w[0] * w[3] * w[4] * BL145.T.dot(M).dot(BL145) +
                  detJ151 * w[0] * w[4] * w[0] * BL151.T.dot(M).dot(BL151) +
                  detJ152 * w[0] * w[4] * w[1] * BL152.T.dot(M).dot(BL152) +
                  detJ153 * w[0] * w[4] * w[2] * BL153.T.dot(M).dot(BL153) +
                  detJ154 * w[0] * w[4] * w[3] * BL154.T.dot(M).dot(BL154) + 
                  detJ155 * w[0] * w[4] * w[4] * BL155.T.dot(M).dot(BL155) +                  
                  detJ211 * w[1] * w[0] * w[0] * BL211.T.dot(M).dot(BL211) +
                  detJ212 * w[1] * w[0] * w[1] * BL212.T.dot(M).dot(BL212) +
                  detJ213 * w[1] * w[0] * w[2] * BL213.T.dot(M).dot(BL213) +
                  detJ214 * w[1] * w[0] * w[3] * BL214.T.dot(M).dot(BL214) +
                  detJ215 * w[1] * w[0] * w[4] * BL215.T.dot(M).dot(BL215) +
                  detJ221 * w[1] * w[1] * w[0] * BL221.T.dot(M).dot(BL221) +
                  detJ222 * w[1] * w[1] * w[1] * BL222.T.dot(M).dot(BL222) +
                  detJ223 * w[1] * w[1] * w[2] * BL223.T.dot(M).dot(BL223) +
                  detJ224 * w[1] * w[1] * w[3] * BL224.T.dot(M).dot(BL224) +
                  detJ225 * w[1] * w[1] * w[4] * BL225.T.dot(M).dot(BL225) +
                  detJ231 * w[1] * w[2] * w[0] * BL231.T.dot(M).dot(BL231) +
                  detJ232 * w[1] * w[2] * w[1] * BL232.T.dot(M).dot(BL232) +
                  detJ233 * w[1] * w[2] * w[2] * BL233.T.dot(M).dot(BL233) +
                  detJ234 * w[1] * w[2] * w[3] * BL234.T.dot(M).dot(BL234) +
                  detJ235 * w[1] * w[2] * w[4] * BL235.T.dot(M).dot(BL235) +
                  detJ241 * w[1] * w[3] * w[0] * BL241.T.dot(M).dot(BL241) +
                  detJ242 * w[1] * w[3] * w[1] * BL242.T.dot(M).dot(BL242) +
                  detJ243 * w[1] * w[3] * w[2] * BL243.T.dot(M).dot(BL243) +
                  detJ244 * w[1] * w[3] * w[3] * BL244.T.dot(M).dot(BL244) + 
                  detJ245 * w[1] * w[3] * w[4] * BL245.T.dot(M).dot(BL245) +
                  detJ251 * w[1] * w[4] * w[0] * BL251.T.dot(M).dot(BL251) +
                  detJ252 * w[1] * w[4] * w[1] * BL252.T.dot(M).dot(BL252) +
                  detJ253 * w[1] * w[4] * w[2] * BL253.T.dot(M).dot(BL253) +
                  detJ254 * w[1] * w[4] * w[3] * BL254.T.dot(M).dot(BL254) + 
                  detJ255 * w[1] * w[4] * w[4] * BL255.T.dot(M).dot(BL255) +                  
                  detJ311 * w[2] * w[0] * w[0] * BL311.T.dot(M).dot(BL311) +
                  detJ312 * w[2] * w[0] * w[1] * BL312.T.dot(M).dot(BL312) +
                  detJ313 * w[2] * w[0] * w[2] * BL313.T.dot(M).dot(BL313) +
                  detJ314 * w[2] * w[0] * w[3] * BL314.T.dot(M).dot(BL314) +
                  detJ315 * w[2] * w[0] * w[4] * BL315.T.dot(M).dot(BL315) +
                  detJ321 * w[2] * w[1] * w[0] * BL321.T.dot(M).dot(BL321) +
                  detJ322 * w[2] * w[1] * w[1] * BL322.T.dot(M).dot(BL322) +
                  detJ323 * w[2] * w[1] * w[2] * BL323.T.dot(M).dot(BL323) +
                  detJ324 * w[2] * w[1] * w[3] * BL324.T.dot(M).dot(BL324) +
                  detJ325 * w[2] * w[1] * w[4] * BL325.T.dot(M).dot(BL325) +
                  detJ331 * w[2] * w[2] * w[0] * BL331.T.dot(M).dot(BL331) +
                  detJ332 * w[2] * w[2] * w[1] * BL332.T.dot(M).dot(BL332) +
                  detJ333 * w[2] * w[2] * w[2] * BL333.T.dot(M).dot(BL333) +
                  detJ334 * w[2] * w[2] * w[3] * BL334.T.dot(M).dot(BL334) +
                  detJ335 * w[2] * w[2] * w[4] * BL335.T.dot(M).dot(BL335) +
                  detJ341 * w[2] * w[3] * w[0] * BL341.T.dot(M).dot(BL341) +
                  detJ342 * w[2] * w[3] * w[1] * BL342.T.dot(M).dot(BL342) +
                  detJ343 * w[2] * w[3] * w[2] * BL343.T.dot(M).dot(BL343) +
                  detJ344 * w[2] * w[3] * w[3] * BL344.T.dot(M).dot(BL344) + 
                  detJ345 * w[2] * w[3] * w[4] * BL345.T.dot(M).dot(BL345) +
                  detJ351 * w[2] * w[4] * w[0] * BL351.T.dot(M).dot(BL351) +
                  detJ352 * w[2] * w[4] * w[1] * BL352.T.dot(M).dot(BL352) +
                  detJ353 * w[2] * w[4] * w[2] * BL353.T.dot(M).dot(BL353) +
                  detJ354 * w[2] * w[4] * w[3] * BL354.T.dot(M).dot(BL354) + 
                  detJ355 * w[2] * w[4] * w[4] * BL355.T.dot(M).dot(BL355) +                  
                  detJ411 * w[3] * w[0] * w[0] * BL411.T.dot(M).dot(BL411) +
                  detJ412 * w[3] * w[0] * w[1] * BL412.T.dot(M).dot(BL412) +
                  detJ413 * w[3] * w[0] * w[2] * BL413.T.dot(M).dot(BL413) +
                  detJ414 * w[3] * w[0] * w[3] * BL414.T.dot(M).dot(BL414) +
                  detJ415 * w[3] * w[0] * w[4] * BL415.T.dot(M).dot(BL415) +
                  detJ421 * w[3] * w[1] * w[0] * BL421.T.dot(M).dot(BL421) +
                  detJ422 * w[3] * w[1] * w[1] * BL422.T.dot(M).dot(BL422) +
                  detJ423 * w[3] * w[1] * w[2] * BL423.T.dot(M).dot(BL423) +
                  detJ424 * w[3] * w[1] * w[3] * BL424.T.dot(M).dot(BL424) +
                  detJ425 * w[3] * w[1] * w[4] * BL425.T.dot(M).dot(BL425) +
                  detJ431 * w[3] * w[2] * w[0] * BL431.T.dot(M).dot(BL431) +
                  detJ432 * w[3] * w[2] * w[1] * BL432.T.dot(M).dot(BL432) +
                  detJ433 * w[3] * w[2] * w[2] * BL433.T.dot(M).dot(BL433) +
                  detJ434 * w[3] * w[2] * w[3] * BL434.T.dot(M).dot(BL434) +
                  detJ435 * w[3] * w[2] * w[4] * BL435.T.dot(M).dot(BL435) +
                  detJ441 * w[3] * w[3] * w[0] * BL441.T.dot(M).dot(BL441) +
                  detJ442 * w[3] * w[3] * w[1] * BL442.T.dot(M).dot(BL442) +
                  detJ443 * w[3] * w[3] * w[2] * BL443.T.dot(M).dot(BL443) +
                  detJ444 * w[3] * w[3] * w[3] * BL444.T.dot(M).dot(BL444) + 
                  detJ445 * w[3] * w[3] * w[4] * BL445.T.dot(M).dot(BL445) +
                  detJ451 * w[3] * w[4] * w[0] * BL451.T.dot(M).dot(BL451) +
                  detJ452 * w[3] * w[4] * w[1] * BL452.T.dot(M).dot(BL452) +
                  detJ453 * w[3] * w[4] * w[2] * BL453.T.dot(M).dot(BL453) +
                  detJ454 * w[3] * w[4] * w[3] * BL454.T.dot(M).dot(BL454) + 
                  detJ455 * w[3] * w[4] * w[4] * BL455.T.dot(M).dot(BL455) +                  
                  detJ511 * w[4] * w[0] * w[0] * BL511.T.dot(M).dot(BL511) +
                  detJ512 * w[4] * w[0] * w[1] * BL512.T.dot(M).dot(BL512) +
                  detJ513 * w[4] * w[0] * w[2] * BL513.T.dot(M).dot(BL513) +
                  detJ514 * w[4] * w[0] * w[3] * BL514.T.dot(M).dot(BL514) +
                  detJ515 * w[4] * w[0] * w[4] * BL515.T.dot(M).dot(BL515) +
                  detJ521 * w[4] * w[1] * w[0] * BL521.T.dot(M).dot(BL521) +
                  detJ522 * w[4] * w[1] * w[1] * BL522.T.dot(M).dot(BL522) +
                  detJ523 * w[4] * w[1] * w[2] * BL523.T.dot(M).dot(BL523) +
                  detJ524 * w[4] * w[1] * w[3] * BL524.T.dot(M).dot(BL524) +
                  detJ525 * w[4] * w[1] * w[4] * BL525.T.dot(M).dot(BL525) +
                  detJ531 * w[4] * w[2] * w[0] * BL531.T.dot(M).dot(BL531) +
                  detJ532 * w[4] * w[2] * w[1] * BL532.T.dot(M).dot(BL532) +
                  detJ533 * w[4] * w[2] * w[2] * BL533.T.dot(M).dot(BL533) +
                  detJ534 * w[4] * w[2] * w[3] * BL534.T.dot(M).dot(BL534) +
                  detJ535 * w[4] * w[2] * w[4] * BL535.T.dot(M).dot(BL535) +
                  detJ541 * w[4] * w[3] * w[0] * BL541.T.dot(M).dot(BL541) +
                  detJ542 * w[4] * w[3] * w[1] * BL542.T.dot(M).dot(BL542) +
                  detJ543 * w[4] * w[3] * w[2] * BL543.T.dot(M).dot(BL543) +
                  detJ544 * w[4] * w[3] * w[3] * BL544.T.dot(M).dot(BL544) + 
                  detJ545 * w[4] * w[3] * w[4] * BL545.T.dot(M).dot(BL545) +
                  detJ551 * w[4] * w[4] * w[0] * BL551.T.dot(M).dot(BL551) +
                  detJ552 * w[4] * w[4] * w[1] * BL552.T.dot(M).dot(BL552) +
                  detJ553 * w[4] * w[4] * w[2] * BL553.T.dot(M).dot(BL553) +
                  detJ554 * w[4] * w[4] * w[3] * BL554.T.dot(M).dot(BL554) + 
                  detJ555 * w[4] * w[4] * w[4] * BL555.T.dot(M).dot(BL555))

            # create the first nonlinear Bmatrix
            BNF111 = self._shapeFns[111].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF112 = self._shapeFns[112].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF113 = self._shapeFns[113].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF114 = self._shapeFns[114].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF115 = self._shapeFns[115].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF121 = self._shapeFns[121].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF122 = self._shapeFns[122].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF123 = self._shapeFns[123].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF124 = self._shapeFns[124].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF125 = self._shapeFns[125].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNF131 = self._shapeFns[131].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF132 = self._shapeFns[132].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF133 = self._shapeFns[133].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF134 = self._shapeFns[134].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF135 = self._shapeFns[135].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF141 = self._shapeFns[141].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF142 = self._shapeFns[142].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF143 = self._shapeFns[143].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF144 = self._shapeFns[144].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF145 = self._shapeFns[145].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF151 = self._shapeFns[151].FirstBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNF152 = self._shapeFns[152].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF153 = self._shapeFns[153].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF154 = self._shapeFns[154].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF155 = self._shapeFns[155].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNF211 = self._shapeFns[211].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF212 = self._shapeFns[212].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF213 = self._shapeFns[213].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF214 = self._shapeFns[214].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF215 = self._shapeFns[215].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF221 = self._shapeFns[221].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF222 = self._shapeFns[222].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF223 = self._shapeFns[223].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF224 = self._shapeFns[224].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF225 = self._shapeFns[225].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNF231 = self._shapeFns[231].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF232 = self._shapeFns[232].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF233 = self._shapeFns[233].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF234 = self._shapeFns[234].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF235 = self._shapeFns[235].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF241 = self._shapeFns[241].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF242 = self._shapeFns[242].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF243 = self._shapeFns[243].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF244 = self._shapeFns[244].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF245 = self._shapeFns[245].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF251 = self._shapeFns[251].FirstBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNF252 = self._shapeFns[252].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF253 = self._shapeFns[253].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF254 = self._shapeFns[254].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF255 = self._shapeFns[255].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNF311 = self._shapeFns[311].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF312 = self._shapeFns[312].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF313 = self._shapeFns[313].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF314 = self._shapeFns[314].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF315 = self._shapeFns[315].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF321 = self._shapeFns[321].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF322 = self._shapeFns[322].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF323 = self._shapeFns[323].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF324 = self._shapeFns[324].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF325 = self._shapeFns[325].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNF331 = self._shapeFns[331].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF332 = self._shapeFns[332].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF333 = self._shapeFns[333].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF334 = self._shapeFns[334].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF335 = self._shapeFns[335].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF341 = self._shapeFns[341].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF342 = self._shapeFns[342].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF343 = self._shapeFns[343].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF344 = self._shapeFns[344].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF345 = self._shapeFns[345].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF351 = self._shapeFns[351].FirstBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNF352 = self._shapeFns[352].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF353 = self._shapeFns[353].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF354 = self._shapeFns[354].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF355 = self._shapeFns[355].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNF411 = self._shapeFns[411].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF412 = self._shapeFns[412].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF413 = self._shapeFns[413].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF414 = self._shapeFns[414].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF415 = self._shapeFns[415].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF421 = self._shapeFns[421].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF422 = self._shapeFns[422].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF423 = self._shapeFns[423].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF424 = self._shapeFns[424].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF425 = self._shapeFns[425].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNF431 = self._shapeFns[431].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF432 = self._shapeFns[432].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF433 = self._shapeFns[433].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF434 = self._shapeFns[434].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF435 = self._shapeFns[435].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF441 = self._shapeFns[441].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF442 = self._shapeFns[442].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF443 = self._shapeFns[443].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF444 = self._shapeFns[444].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF445 = self._shapeFns[445].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF451 = self._shapeFns[451].FirstBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNF452 = self._shapeFns[452].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF453 = self._shapeFns[453].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF454 = self._shapeFns[454].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF455 = self._shapeFns[455].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNF511 = self._shapeFns[511].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF512 = self._shapeFns[512].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF513 = self._shapeFns[513].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF514 = self._shapeFns[514].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF515 = self._shapeFns[515].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF521 = self._shapeFns[521].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF522 = self._shapeFns[522].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF523 = self._shapeFns[523].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF524 = self._shapeFns[524].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF525 = self._shapeFns[525].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNF531 = self._shapeFns[531].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF532 = self._shapeFns[532].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF533 = self._shapeFns[533].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF534 = self._shapeFns[534].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF535 = self._shapeFns[535].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF541 = self._shapeFns[541].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF542 = self._shapeFns[542].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF543 = self._shapeFns[543].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF544 = self._shapeFns[544].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF545 = self._shapeFns[545].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNF551 = self._shapeFns[551].FirstBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNF552 = self._shapeFns[552].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF553 = self._shapeFns[553].FirstBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNF554 = self._shapeFns[554].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNF555 = self._shapeFns[555].FirstBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            

            
            # create the second nonlinear Bmatrix
            BNS111 = self._shapeFns[111].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS112 = self._shapeFns[112].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS113 = self._shapeFns[113].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS114 = self._shapeFns[114].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS115 = self._shapeFns[115].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS121 = self._shapeFns[121].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS122 = self._shapeFns[122].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS123 = self._shapeFns[123].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS124 = self._shapeFns[124].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS125 = self._shapeFns[125].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNS131 = self._shapeFns[131].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS132 = self._shapeFns[132].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS133 = self._shapeFns[133].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS134 = self._shapeFns[134].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS135 = self._shapeFns[135].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS141 = self._shapeFns[141].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS142 = self._shapeFns[142].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS143 = self._shapeFns[143].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS144 = self._shapeFns[144].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS145 = self._shapeFns[145].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS151 = self._shapeFns[151].SecondBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNS152 = self._shapeFns[152].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS153 = self._shapeFns[153].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS154 = self._shapeFns[154].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS155 = self._shapeFns[155].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNS211 = self._shapeFns[211].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS212 = self._shapeFns[212].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS213 = self._shapeFns[213].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS214 = self._shapeFns[214].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS215 = self._shapeFns[215].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS221 = self._shapeFns[221].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS222 = self._shapeFns[222].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS223 = self._shapeFns[223].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS224 = self._shapeFns[224].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS225 = self._shapeFns[225].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNS231 = self._shapeFns[231].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS232 = self._shapeFns[232].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS233 = self._shapeFns[233].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS234 = self._shapeFns[234].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS235 = self._shapeFns[235].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS241 = self._shapeFns[241].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS242 = self._shapeFns[242].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS243 = self._shapeFns[243].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS244 = self._shapeFns[244].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS245 = self._shapeFns[245].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS251 = self._shapeFns[251].SecondBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNS252 = self._shapeFns[252].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS253 = self._shapeFns[253].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS254 = self._shapeFns[254].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS255 = self._shapeFns[255].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNS311 = self._shapeFns[311].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS312 = self._shapeFns[312].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS313 = self._shapeFns[313].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS314 = self._shapeFns[314].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS315 = self._shapeFns[315].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS321 = self._shapeFns[321].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS322 = self._shapeFns[322].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS323 = self._shapeFns[323].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS324 = self._shapeFns[324].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS325 = self._shapeFns[325].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNS331 = self._shapeFns[331].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS332 = self._shapeFns[332].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS333 = self._shapeFns[333].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS334 = self._shapeFns[334].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS335 = self._shapeFns[335].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS341 = self._shapeFns[341].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS342 = self._shapeFns[342].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS343 = self._shapeFns[343].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS344 = self._shapeFns[344].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS345 = self._shapeFns[345].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS351 = self._shapeFns[351].SecondBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNS352 = self._shapeFns[352].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS353 = self._shapeFns[353].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS354 = self._shapeFns[354].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS355 = self._shapeFns[355].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNS411 = self._shapeFns[411].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS412 = self._shapeFns[412].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS413 = self._shapeFns[413].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS414 = self._shapeFns[414].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS415 = self._shapeFns[415].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS421 = self._shapeFns[421].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS422 = self._shapeFns[422].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS423 = self._shapeFns[423].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS424 = self._shapeFns[424].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS425 = self._shapeFns[425].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNS431 = self._shapeFns[431].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS432 = self._shapeFns[432].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS433 = self._shapeFns[433].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS434 = self._shapeFns[434].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS435 = self._shapeFns[435].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS441 = self._shapeFns[441].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS442 = self._shapeFns[442].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS443 = self._shapeFns[443].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS444 = self._shapeFns[444].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS445 = self._shapeFns[445].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS451 = self._shapeFns[451].SecondBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNS452 = self._shapeFns[452].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS453 = self._shapeFns[453].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS454 = self._shapeFns[454].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS455 = self._shapeFns[455].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNS511 = self._shapeFns[511].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS512 = self._shapeFns[512].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS513 = self._shapeFns[513].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS514 = self._shapeFns[514].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS515 = self._shapeFns[515].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS521 = self._shapeFns[521].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS522 = self._shapeFns[522].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS523 = self._shapeFns[523].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS524 = self._shapeFns[524].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS525 = self._shapeFns[525].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNS531 = self._shapeFns[531].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS532 = self._shapeFns[532].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS533 = self._shapeFns[533].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS534 = self._shapeFns[534].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS535 = self._shapeFns[535].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS541 = self._shapeFns[541].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS542 = self._shapeFns[542].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS543 = self._shapeFns[543].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS544 = self._shapeFns[544].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS545 = self._shapeFns[545].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNS551 = self._shapeFns[551].SecondBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNS552 = self._shapeFns[552].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS553 = self._shapeFns[553].SecondBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNS554 = self._shapeFns[554].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNS555 = self._shapeFns[555].SecondBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
                       
            
            # create the third nonlinear Bmatrix
            BNT111 = self._shapeFns[111].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT112 = self._shapeFns[112].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT113 = self._shapeFns[113].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT114 = self._shapeFns[114].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT115 = self._shapeFns[115].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT121 = self._shapeFns[121].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT122 = self._shapeFns[122].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT123 = self._shapeFns[123].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT124 = self._shapeFns[124].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT125 = self._shapeFns[125].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNT131 = self._shapeFns[131].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT132 = self._shapeFns[132].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT133 = self._shapeFns[133].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT134 = self._shapeFns[134].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT135 = self._shapeFns[135].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT141 = self._shapeFns[141].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT142 = self._shapeFns[142].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT143 = self._shapeFns[143].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT144 = self._shapeFns[144].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT145 = self._shapeFns[145].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT151 = self._shapeFns[151].ThirdBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNT152 = self._shapeFns[152].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT153 = self._shapeFns[153].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT154 = self._shapeFns[154].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT155 = self._shapeFns[155].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNT211 = self._shapeFns[211].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT212 = self._shapeFns[212].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT213 = self._shapeFns[213].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT214 = self._shapeFns[214].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT215 = self._shapeFns[215].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT221 = self._shapeFns[221].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT222 = self._shapeFns[222].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT223 = self._shapeFns[223].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT224 = self._shapeFns[224].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT225 = self._shapeFns[225].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNT231 = self._shapeFns[231].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT232 = self._shapeFns[232].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT233 = self._shapeFns[233].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT234 = self._shapeFns[234].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT235 = self._shapeFns[235].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT241 = self._shapeFns[241].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT242 = self._shapeFns[242].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT243 = self._shapeFns[243].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT244 = self._shapeFns[244].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT245 = self._shapeFns[245].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT251 = self._shapeFns[251].ThirdBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNT252 = self._shapeFns[252].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT253 = self._shapeFns[253].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT254 = self._shapeFns[254].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT255 = self._shapeFns[255].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNT311 = self._shapeFns[311].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT312 = self._shapeFns[312].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT313 = self._shapeFns[313].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT314 = self._shapeFns[314].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT315 = self._shapeFns[315].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT321 = self._shapeFns[321].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT322 = self._shapeFns[322].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT323 = self._shapeFns[323].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT324 = self._shapeFns[324].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT325 = self._shapeFns[325].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNT331 = self._shapeFns[331].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT332 = self._shapeFns[332].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT333 = self._shapeFns[333].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT334 = self._shapeFns[334].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT335 = self._shapeFns[335].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT341 = self._shapeFns[341].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT342 = self._shapeFns[342].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT343 = self._shapeFns[343].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT344 = self._shapeFns[344].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT345 = self._shapeFns[345].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT351 = self._shapeFns[351].ThirdBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNT352 = self._shapeFns[352].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT353 = self._shapeFns[353].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT354 = self._shapeFns[354].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT355 = self._shapeFns[355].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNT411 = self._shapeFns[411].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT412 = self._shapeFns[412].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT413 = self._shapeFns[413].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT414 = self._shapeFns[414].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT415 = self._shapeFns[415].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT421 = self._shapeFns[421].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT422 = self._shapeFns[422].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT423 = self._shapeFns[423].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT424 = self._shapeFns[424].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT425 = self._shapeFns[425].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNT431 = self._shapeFns[431].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT432 = self._shapeFns[432].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT433 = self._shapeFns[433].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT434 = self._shapeFns[434].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT435 = self._shapeFns[435].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT441 = self._shapeFns[441].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT442 = self._shapeFns[442].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT443 = self._shapeFns[443].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT444 = self._shapeFns[444].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT445 = self._shapeFns[445].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT451 = self._shapeFns[451].ThirdBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNT452 = self._shapeFns[452].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT453 = self._shapeFns[453].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT454 = self._shapeFns[454].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT455 = self._shapeFns[455].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            
            BNT511 = self._shapeFns[511].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT512 = self._shapeFns[512].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT513 = self._shapeFns[513].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT514 = self._shapeFns[514].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT515 = self._shapeFns[515].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT521 = self._shapeFns[521].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT522 = self._shapeFns[522].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT523 = self._shapeFns[523].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT524 = self._shapeFns[524].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT525 = self._shapeFns[525].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)             
            BNT531 = self._shapeFns[531].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT532 = self._shapeFns[532].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT533 = self._shapeFns[533].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT534 = self._shapeFns[534].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT535 = self._shapeFns[535].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT541 = self._shapeFns[541].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT542 = self._shapeFns[542].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT543 = self._shapeFns[543].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT544 = self._shapeFns[544].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT545 = self._shapeFns[545].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04) 
            BNT551 = self._shapeFns[551].ThirdBNonLinear(x1, x2, x3, x4,
                                                        x01, x02, x03, x04)
            BNT552 = self._shapeFns[552].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT553 = self._shapeFns[553].ThirdBNonLinear(x1, x2, x3, x4,
                                                         x01, x02, x03, x04)
            BNT554 = self._shapeFns[554].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)
            BNT555 = self._shapeFns[555].ThirdBNonLinear(x1, x2, x3, x4, 
                                                         x01, x02, x03, x04)

            # total nonlinear Bmatrix
            BN111 = BNF111 + BNS111 + BNT111
            BN112 = BNF112 + BNS112 + BNT112
            BN113 = BNF113 + BNS113 + BNT113
            BN114 = BNF114 + BNS114 + BNT114 
            BN115 = BNF115 + BNS115 + BNT115
            BN121 = BNF121 + BNS121 + BNT121
            BN122 = BNF122 + BNS122 + BNT122
            BN123 = BNF123 + BNS123 + BNT123
            BN124 = BNF124 + BNS124 + BNT124
            BN125 = BNF125 + BNS125 + BNT125            
            BN131 = BNF131 + BNS131 + BNT131
            BN132 = BNF132 + BNS132 + BNT132
            BN133 = BNF133 + BNS133 + BNT133
            BN134 = BNF134 + BNS134 + BNT134 
            BN135 = BNF135 + BNS135 + BNT135
            BN141 = BNF141 + BNS141 + BNT141
            BN142 = BNF142 + BNS142 + BNT142
            BN143 = BNF143 + BNS143 + BNT143
            BN144 = BNF144 + BNS144 + BNT144
            BN145 = BNF145 + BNS145 + BNT145
            BN151 = BNF151 + BNS151 + BNT151
            BN152 = BNF152 + BNS152 + BNT152
            BN153 = BNF153 + BNS153 + BNT153
            BN154 = BNF154 + BNS154 + BNT154
            BN155 = BNF155 + BNS155 + BNT155
            
            BN211 = BNF211 + BNS211 + BNT211
            BN212 = BNF212 + BNS212 + BNT212
            BN213 = BNF213 + BNS213 + BNT213
            BN214 = BNF214 + BNS214 + BNT214 
            BN215 = BNF215 + BNS215 + BNT215
            BN221 = BNF221 + BNS221 + BNT221
            BN222 = BNF222 + BNS222 + BNT222
            BN223 = BNF223 + BNS223 + BNT223
            BN224 = BNF224 + BNS224 + BNT224
            BN225 = BNF225 + BNS225 + BNT225            
            BN231 = BNF231 + BNS231 + BNT231
            BN232 = BNF232 + BNS232 + BNT232
            BN233 = BNF233 + BNS233 + BNT233
            BN234 = BNF234 + BNS234 + BNT234 
            BN235 = BNF235 + BNS235 + BNT235
            BN241 = BNF241 + BNS241 + BNT241
            BN242 = BNF242 + BNS242 + BNT242
            BN243 = BNF243 + BNS243 + BNT243
            BN244 = BNF244 + BNS244 + BNT244
            BN245 = BNF245 + BNS245 + BNT245
            BN251 = BNF251 + BNS251 + BNT251
            BN252 = BNF252 + BNS252 + BNT252
            BN253 = BNF253 + BNS253 + BNT253
            BN254 = BNF254 + BNS254 + BNT254
            BN255 = BNF255 + BNS255 + BNT255
            
            BN311 = BNF311 + BNS311 + BNT311
            BN312 = BNF312 + BNS312 + BNT312
            BN313 = BNF313 + BNS313 + BNT313
            BN314 = BNF314 + BNS314 + BNT314 
            BN315 = BNF315 + BNS315 + BNT315
            BN321 = BNF321 + BNS321 + BNT321
            BN322 = BNF322 + BNS322 + BNT322
            BN323 = BNF323 + BNS323 + BNT323
            BN324 = BNF324 + BNS324 + BNT324
            BN325 = BNF325 + BNS325 + BNT325            
            BN331 = BNF331 + BNS331 + BNT331
            BN332 = BNF332 + BNS332 + BNT332
            BN333 = BNF333 + BNS333 + BNT333
            BN334 = BNF334 + BNS334 + BNT334 
            BN335 = BNF335 + BNS335 + BNT335
            BN341 = BNF341 + BNS341 + BNT341
            BN342 = BNF342 + BNS342 + BNT342
            BN343 = BNF343 + BNS343 + BNT343
            BN344 = BNF344 + BNS344 + BNT344
            BN345 = BNF345 + BNS345 + BNT345
            BN351 = BNF351 + BNS351 + BNT351
            BN352 = BNF352 + BNS352 + BNT352
            BN353 = BNF353 + BNS353 + BNT353
            BN354 = BNF354 + BNS354 + BNT354
            BN355 = BNF355 + BNS355 + BNT355
            
            BN411 = BNF411 + BNS411 + BNT411
            BN412 = BNF412 + BNS412 + BNT412
            BN413 = BNF413 + BNS413 + BNT413
            BN414 = BNF414 + BNS414 + BNT414 
            BN415 = BNF415 + BNS415 + BNT415
            BN421 = BNF421 + BNS421 + BNT421
            BN422 = BNF422 + BNS422 + BNT422
            BN423 = BNF423 + BNS423 + BNT423
            BN424 = BNF424 + BNS424 + BNT424
            BN425 = BNF425 + BNS425 + BNT425            
            BN431 = BNF431 + BNS431 + BNT431
            BN432 = BNF432 + BNS432 + BNT432
            BN433 = BNF433 + BNS433 + BNT433
            BN434 = BNF434 + BNS434 + BNT434 
            BN435 = BNF435 + BNS435 + BNT435
            BN441 = BNF441 + BNS441 + BNT441
            BN442 = BNF442 + BNS442 + BNT442
            BN443 = BNF443 + BNS443 + BNT443
            BN444 = BNF444 + BNS444 + BNT444
            BN445 = BNF445 + BNS445 + BNT445
            BN451 = BNF451 + BNS451 + BNT451
            BN452 = BNF452 + BNS452 + BNT452
            BN453 = BNF453 + BNS453 + BNT453
            BN454 = BNF454 + BNS454 + BNT454
            BN455 = BNF455 + BNS455 + BNT455
            
            BN511 = BNF511 + BNS511 + BNT511
            BN512 = BNF512 + BNS512 + BNT512
            BN513 = BNF513 + BNS513 + BNT513
            BN514 = BNF514 + BNS514 + BNT514 
            BN515 = BNF515 + BNS515 + BNT515
            BN521 = BNF521 + BNS521 + BNT521
            BN522 = BNF522 + BNS522 + BNT522
            BN523 = BNF523 + BNS523 + BNT523
            BN524 = BNF524 + BNS524 + BNT524
            BN525 = BNF525 + BNS525 + BNT525            
            BN531 = BNF531 + BNS531 + BNT531
            BN532 = BNF532 + BNS532 + BNT532
            BN533 = BNF533 + BNS533 + BNT533
            BN534 = BNF534 + BNS534 + BNT534 
            BN535 = BNF535 + BNS535 + BNT535
            BN541 = BNF541 + BNS541 + BNT541
            BN542 = BNF542 + BNS542 + BNT542
            BN543 = BNF543 + BNS543 + BNT543
            BN544 = BNF544 + BNS544 + BNT544
            BN545 = BNF545 + BNS545 + BNT545
            BN551 = BNF551 + BNS551 + BNT551
            BN552 = BNF552 + BNS552 + BNT552
            BN553 = BNF553 + BNS553 + BNT553
            BN554 = BNF554 + BNS554 + BNT554
            BN555 = BNF555 + BNS555 + BNT555
                                    

            # create the first H matrix
            HF111 = self._shapeFns[111].HmatrixF(x1, x2, x3, x4)
            HF112 = self._shapeFns[112].HmatrixF(x1, x2, x3, x4)
            HF113 = self._shapeFns[113].HmatrixF(x1, x2, x3, x4)
            HF114 = self._shapeFns[114].HmatrixF(x1, x2, x3, x4)
            HF115 = self._shapeFns[115].HmatrixF(x1, x2, x3, x4)
            HF121 = self._shapeFns[121].HmatrixF(x1, x2, x3, x4)
            HF122 = self._shapeFns[122].HmatrixF(x1, x2, x3, x4)
            HF123 = self._shapeFns[123].HmatrixF(x1, x2, x3, x4)
            HF124 = self._shapeFns[124].HmatrixF(x1, x2, x3, x4)
            HF125 = self._shapeFns[125].HmatrixF(x1, x2, x3, x4)            
            HF131 = self._shapeFns[131].HmatrixF(x1, x2, x3, x4)
            HF132 = self._shapeFns[132].HmatrixF(x1, x2, x3, x4)
            HF133 = self._shapeFns[133].HmatrixF(x1, x2, x3, x4)
            HF134 = self._shapeFns[134].HmatrixF(x1, x2, x3, x4) 
            HF135 = self._shapeFns[135].HmatrixF(x1, x2, x3, x4)
            HF141 = self._shapeFns[141].HmatrixF(x1, x2, x3, x4)
            HF142 = self._shapeFns[142].HmatrixF(x1, x2, x3, x4)
            HF143 = self._shapeFns[143].HmatrixF(x1, x2, x3, x4)
            HF144 = self._shapeFns[144].HmatrixF(x1, x2, x3, x4)
            HF145 = self._shapeFns[145].HmatrixF(x1, x2, x3, x4)
            HF151 = self._shapeFns[151].HmatrixF(x1, x2, x3, x4)
            HF152 = self._shapeFns[152].HmatrixF(x1, x2, x3, x4)
            HF153 = self._shapeFns[153].HmatrixF(x1, x2, x3, x4)
            HF154 = self._shapeFns[154].HmatrixF(x1, x2, x3, x4)
            HF155 = self._shapeFns[155].HmatrixF(x1, x2, x3, x4)
            
            HF211 = self._shapeFns[211].HmatrixF(x1, x2, x3, x4)
            HF212 = self._shapeFns[212].HmatrixF(x1, x2, x3, x4)
            HF213 = self._shapeFns[213].HmatrixF(x1, x2, x3, x4)
            HF214 = self._shapeFns[214].HmatrixF(x1, x2, x3, x4)
            HF215 = self._shapeFns[215].HmatrixF(x1, x2, x3, x4)
            HF221 = self._shapeFns[221].HmatrixF(x1, x2, x3, x4)
            HF222 = self._shapeFns[222].HmatrixF(x1, x2, x3, x4)
            HF223 = self._shapeFns[223].HmatrixF(x1, x2, x3, x4)
            HF224 = self._shapeFns[224].HmatrixF(x1, x2, x3, x4)
            HF225 = self._shapeFns[225].HmatrixF(x1, x2, x3, x4)            
            HF231 = self._shapeFns[231].HmatrixF(x1, x2, x3, x4)
            HF232 = self._shapeFns[232].HmatrixF(x1, x2, x3, x4)
            HF233 = self._shapeFns[233].HmatrixF(x1, x2, x3, x4)
            HF234 = self._shapeFns[234].HmatrixF(x1, x2, x3, x4) 
            HF235 = self._shapeFns[235].HmatrixF(x1, x2, x3, x4)
            HF241 = self._shapeFns[241].HmatrixF(x1, x2, x3, x4)
            HF242 = self._shapeFns[242].HmatrixF(x1, x2, x3, x4)
            HF243 = self._shapeFns[243].HmatrixF(x1, x2, x3, x4)
            HF244 = self._shapeFns[244].HmatrixF(x1, x2, x3, x4)
            HF245 = self._shapeFns[245].HmatrixF(x1, x2, x3, x4)
            HF251 = self._shapeFns[251].HmatrixF(x1, x2, x3, x4)
            HF252 = self._shapeFns[252].HmatrixF(x1, x2, x3, x4)
            HF253 = self._shapeFns[253].HmatrixF(x1, x2, x3, x4)
            HF254 = self._shapeFns[254].HmatrixF(x1, x2, x3, x4)
            HF255 = self._shapeFns[255].HmatrixF(x1, x2, x3, x4)
            
            HF311 = self._shapeFns[311].HmatrixF(x1, x2, x3, x4)
            HF312 = self._shapeFns[312].HmatrixF(x1, x2, x3, x4)
            HF313 = self._shapeFns[313].HmatrixF(x1, x2, x3, x4)
            HF314 = self._shapeFns[314].HmatrixF(x1, x2, x3, x4)
            HF315 = self._shapeFns[315].HmatrixF(x1, x2, x3, x4)
            HF321 = self._shapeFns[321].HmatrixF(x1, x2, x3, x4)
            HF322 = self._shapeFns[322].HmatrixF(x1, x2, x3, x4)
            HF323 = self._shapeFns[323].HmatrixF(x1, x2, x3, x4)
            HF324 = self._shapeFns[324].HmatrixF(x1, x2, x3, x4)
            HF325 = self._shapeFns[325].HmatrixF(x1, x2, x3, x4)            
            HF331 = self._shapeFns[331].HmatrixF(x1, x2, x3, x4)
            HF332 = self._shapeFns[332].HmatrixF(x1, x2, x3, x4)
            HF333 = self._shapeFns[333].HmatrixF(x1, x2, x3, x4)
            HF334 = self._shapeFns[334].HmatrixF(x1, x2, x3, x4) 
            HF335 = self._shapeFns[335].HmatrixF(x1, x2, x3, x4)
            HF341 = self._shapeFns[341].HmatrixF(x1, x2, x3, x4)
            HF342 = self._shapeFns[342].HmatrixF(x1, x2, x3, x4)
            HF343 = self._shapeFns[343].HmatrixF(x1, x2, x3, x4)
            HF344 = self._shapeFns[344].HmatrixF(x1, x2, x3, x4)
            HF345 = self._shapeFns[345].HmatrixF(x1, x2, x3, x4)
            HF351 = self._shapeFns[351].HmatrixF(x1, x2, x3, x4)
            HF352 = self._shapeFns[352].HmatrixF(x1, x2, x3, x4)
            HF353 = self._shapeFns[353].HmatrixF(x1, x2, x3, x4)
            HF354 = self._shapeFns[354].HmatrixF(x1, x2, x3, x4)
            HF355 = self._shapeFns[355].HmatrixF(x1, x2, x3, x4)
            
            HF411 = self._shapeFns[411].HmatrixF(x1, x2, x3, x4)
            HF412 = self._shapeFns[412].HmatrixF(x1, x2, x3, x4)
            HF413 = self._shapeFns[413].HmatrixF(x1, x2, x3, x4)
            HF414 = self._shapeFns[414].HmatrixF(x1, x2, x3, x4)
            HF415 = self._shapeFns[415].HmatrixF(x1, x2, x3, x4)
            HF421 = self._shapeFns[421].HmatrixF(x1, x2, x3, x4)
            HF422 = self._shapeFns[422].HmatrixF(x1, x2, x3, x4)
            HF423 = self._shapeFns[423].HmatrixF(x1, x2, x3, x4)
            HF424 = self._shapeFns[424].HmatrixF(x1, x2, x3, x4)
            HF425 = self._shapeFns[425].HmatrixF(x1, x2, x3, x4)            
            HF431 = self._shapeFns[431].HmatrixF(x1, x2, x3, x4)
            HF432 = self._shapeFns[432].HmatrixF(x1, x2, x3, x4)
            HF433 = self._shapeFns[433].HmatrixF(x1, x2, x3, x4)
            HF434 = self._shapeFns[434].HmatrixF(x1, x2, x3, x4) 
            HF435 = self._shapeFns[435].HmatrixF(x1, x2, x3, x4)
            HF441 = self._shapeFns[441].HmatrixF(x1, x2, x3, x4)
            HF442 = self._shapeFns[442].HmatrixF(x1, x2, x3, x4)
            HF443 = self._shapeFns[443].HmatrixF(x1, x2, x3, x4)
            HF444 = self._shapeFns[444].HmatrixF(x1, x2, x3, x4)
            HF445 = self._shapeFns[445].HmatrixF(x1, x2, x3, x4)
            HF451 = self._shapeFns[451].HmatrixF(x1, x2, x3, x4)
            HF452 = self._shapeFns[452].HmatrixF(x1, x2, x3, x4)
            HF453 = self._shapeFns[453].HmatrixF(x1, x2, x3, x4)
            HF454 = self._shapeFns[454].HmatrixF(x1, x2, x3, x4)
            HF455 = self._shapeFns[455].HmatrixF(x1, x2, x3, x4)
            
            HF511 = self._shapeFns[511].HmatrixF(x1, x2, x3, x4)
            HF512 = self._shapeFns[512].HmatrixF(x1, x2, x3, x4)
            HF513 = self._shapeFns[513].HmatrixF(x1, x2, x3, x4)
            HF514 = self._shapeFns[514].HmatrixF(x1, x2, x3, x4)
            HF515 = self._shapeFns[515].HmatrixF(x1, x2, x3, x4)
            HF521 = self._shapeFns[521].HmatrixF(x1, x2, x3, x4)
            HF522 = self._shapeFns[522].HmatrixF(x1, x2, x3, x4)
            HF523 = self._shapeFns[523].HmatrixF(x1, x2, x3, x4)
            HF524 = self._shapeFns[524].HmatrixF(x1, x2, x3, x4)
            HF525 = self._shapeFns[525].HmatrixF(x1, x2, x3, x4)            
            HF531 = self._shapeFns[531].HmatrixF(x1, x2, x3, x4)
            HF532 = self._shapeFns[532].HmatrixF(x1, x2, x3, x4)
            HF533 = self._shapeFns[533].HmatrixF(x1, x2, x3, x4)
            HF534 = self._shapeFns[534].HmatrixF(x1, x2, x3, x4) 
            HF535 = self._shapeFns[535].HmatrixF(x1, x2, x3, x4)
            HF541 = self._shapeFns[541].HmatrixF(x1, x2, x3, x4)
            HF542 = self._shapeFns[542].HmatrixF(x1, x2, x3, x4)
            HF543 = self._shapeFns[543].HmatrixF(x1, x2, x3, x4)
            HF544 = self._shapeFns[544].HmatrixF(x1, x2, x3, x4)
            HF545 = self._shapeFns[545].HmatrixF(x1, x2, x3, x4)
            HF551 = self._shapeFns[551].HmatrixF(x1, x2, x3, x4)
            HF552 = self._shapeFns[552].HmatrixF(x1, x2, x3, x4)
            HF553 = self._shapeFns[553].HmatrixF(x1, x2, x3, x4)
            HF554 = self._shapeFns[554].HmatrixF(x1, x2, x3, x4)
            HF555 = self._shapeFns[555].HmatrixF(x1, x2, x3, x4)
            
            
            # create the second H matrix
            HS111 = self._shapeFns[111].HmatrixS(x1, x2, x3, x4)
            HS112 = self._shapeFns[112].HmatrixS(x1, x2, x3, x4)
            HS113 = self._shapeFns[113].HmatrixS(x1, x2, x3, x4)
            HS114 = self._shapeFns[114].HmatrixS(x1, x2, x3, x4)
            HS115 = self._shapeFns[115].HmatrixS(x1, x2, x3, x4)
            HS121 = self._shapeFns[121].HmatrixS(x1, x2, x3, x4)
            HS122 = self._shapeFns[122].HmatrixS(x1, x2, x3, x4)
            HS123 = self._shapeFns[123].HmatrixS(x1, x2, x3, x4)
            HS124 = self._shapeFns[124].HmatrixS(x1, x2, x3, x4)
            HS125 = self._shapeFns[125].HmatrixS(x1, x2, x3, x4)            
            HS131 = self._shapeFns[131].HmatrixS(x1, x2, x3, x4)
            HS132 = self._shapeFns[132].HmatrixS(x1, x2, x3, x4)
            HS133 = self._shapeFns[133].HmatrixS(x1, x2, x3, x4)
            HS134 = self._shapeFns[134].HmatrixS(x1, x2, x3, x4) 
            HS135 = self._shapeFns[135].HmatrixS(x1, x2, x3, x4)
            HS141 = self._shapeFns[141].HmatrixS(x1, x2, x3, x4)
            HS142 = self._shapeFns[142].HmatrixS(x1, x2, x3, x4)
            HS143 = self._shapeFns[143].HmatrixS(x1, x2, x3, x4)
            HS144 = self._shapeFns[144].HmatrixS(x1, x2, x3, x4)
            HS145 = self._shapeFns[145].HmatrixS(x1, x2, x3, x4)
            HS151 = self._shapeFns[151].HmatrixS(x1, x2, x3, x4)
            HS152 = self._shapeFns[152].HmatrixS(x1, x2, x3, x4)
            HS153 = self._shapeFns[153].HmatrixS(x1, x2, x3, x4)
            HS154 = self._shapeFns[154].HmatrixS(x1, x2, x3, x4)
            HS155 = self._shapeFns[155].HmatrixS(x1, x2, x3, x4)
            
            HS211 = self._shapeFns[211].HmatrixS(x1, x2, x3, x4)
            HS212 = self._shapeFns[212].HmatrixS(x1, x2, x3, x4)
            HS213 = self._shapeFns[213].HmatrixS(x1, x2, x3, x4)
            HS214 = self._shapeFns[214].HmatrixS(x1, x2, x3, x4)
            HS215 = self._shapeFns[215].HmatrixS(x1, x2, x3, x4)
            HS221 = self._shapeFns[221].HmatrixS(x1, x2, x3, x4)
            HS222 = self._shapeFns[222].HmatrixS(x1, x2, x3, x4)
            HS223 = self._shapeFns[223].HmatrixS(x1, x2, x3, x4)
            HS224 = self._shapeFns[224].HmatrixS(x1, x2, x3, x4)
            HS225 = self._shapeFns[225].HmatrixS(x1, x2, x3, x4)            
            HS231 = self._shapeFns[231].HmatrixS(x1, x2, x3, x4)
            HS232 = self._shapeFns[232].HmatrixS(x1, x2, x3, x4)
            HS233 = self._shapeFns[233].HmatrixS(x1, x2, x3, x4)
            HS234 = self._shapeFns[234].HmatrixS(x1, x2, x3, x4) 
            HS235 = self._shapeFns[235].HmatrixS(x1, x2, x3, x4)
            HS241 = self._shapeFns[241].HmatrixS(x1, x2, x3, x4)
            HS242 = self._shapeFns[242].HmatrixS(x1, x2, x3, x4)
            HS243 = self._shapeFns[243].HmatrixS(x1, x2, x3, x4)
            HS244 = self._shapeFns[244].HmatrixS(x1, x2, x3, x4)
            HS245 = self._shapeFns[245].HmatrixS(x1, x2, x3, x4)
            HS251 = self._shapeFns[251].HmatrixS(x1, x2, x3, x4)
            HS252 = self._shapeFns[252].HmatrixS(x1, x2, x3, x4)
            HS253 = self._shapeFns[253].HmatrixS(x1, x2, x3, x4)
            HS254 = self._shapeFns[254].HmatrixS(x1, x2, x3, x4)
            HS255 = self._shapeFns[255].HmatrixS(x1, x2, x3, x4)
            
            HS311 = self._shapeFns[311].HmatrixS(x1, x2, x3, x4)
            HS312 = self._shapeFns[312].HmatrixS(x1, x2, x3, x4)
            HS313 = self._shapeFns[313].HmatrixS(x1, x2, x3, x4)
            HS314 = self._shapeFns[314].HmatrixS(x1, x2, x3, x4)
            HS315 = self._shapeFns[315].HmatrixS(x1, x2, x3, x4)
            HS321 = self._shapeFns[321].HmatrixS(x1, x2, x3, x4)
            HS322 = self._shapeFns[322].HmatrixS(x1, x2, x3, x4)
            HS323 = self._shapeFns[323].HmatrixS(x1, x2, x3, x4)
            HS324 = self._shapeFns[324].HmatrixS(x1, x2, x3, x4)
            HS325 = self._shapeFns[325].HmatrixS(x1, x2, x3, x4)            
            HS331 = self._shapeFns[331].HmatrixS(x1, x2, x3, x4)
            HS332 = self._shapeFns[332].HmatrixS(x1, x2, x3, x4)
            HS333 = self._shapeFns[333].HmatrixS(x1, x2, x3, x4)
            HS334 = self._shapeFns[334].HmatrixS(x1, x2, x3, x4) 
            HS335 = self._shapeFns[335].HmatrixS(x1, x2, x3, x4)
            HS341 = self._shapeFns[341].HmatrixS(x1, x2, x3, x4)
            HS342 = self._shapeFns[342].HmatrixS(x1, x2, x3, x4)
            HS343 = self._shapeFns[343].HmatrixS(x1, x2, x3, x4)
            HS344 = self._shapeFns[344].HmatrixS(x1, x2, x3, x4)
            HS345 = self._shapeFns[345].HmatrixS(x1, x2, x3, x4)
            HS351 = self._shapeFns[351].HmatrixS(x1, x2, x3, x4)
            HS352 = self._shapeFns[352].HmatrixS(x1, x2, x3, x4)
            HS353 = self._shapeFns[353].HmatrixS(x1, x2, x3, x4)
            HS354 = self._shapeFns[354].HmatrixS(x1, x2, x3, x4)
            HS355 = self._shapeFns[355].HmatrixS(x1, x2, x3, x4)
            
            HS411 = self._shapeFns[411].HmatrixS(x1, x2, x3, x4)
            HS412 = self._shapeFns[412].HmatrixS(x1, x2, x3, x4)
            HS413 = self._shapeFns[413].HmatrixS(x1, x2, x3, x4)
            HS414 = self._shapeFns[414].HmatrixS(x1, x2, x3, x4)
            HS415 = self._shapeFns[415].HmatrixS(x1, x2, x3, x4)
            HS421 = self._shapeFns[421].HmatrixS(x1, x2, x3, x4)
            HS422 = self._shapeFns[422].HmatrixS(x1, x2, x3, x4)
            HS423 = self._shapeFns[423].HmatrixS(x1, x2, x3, x4)
            HS424 = self._shapeFns[424].HmatrixS(x1, x2, x3, x4)
            HS425 = self._shapeFns[425].HmatrixS(x1, x2, x3, x4)            
            HS431 = self._shapeFns[431].HmatrixS(x1, x2, x3, x4)
            HS432 = self._shapeFns[432].HmatrixS(x1, x2, x3, x4)
            HS433 = self._shapeFns[433].HmatrixS(x1, x2, x3, x4)
            HS434 = self._shapeFns[434].HmatrixS(x1, x2, x3, x4) 
            HS435 = self._shapeFns[435].HmatrixS(x1, x2, x3, x4)
            HS441 = self._shapeFns[441].HmatrixS(x1, x2, x3, x4)
            HS442 = self._shapeFns[442].HmatrixS(x1, x2, x3, x4)
            HS443 = self._shapeFns[443].HmatrixS(x1, x2, x3, x4)
            HS444 = self._shapeFns[444].HmatrixS(x1, x2, x3, x4)
            HS445 = self._shapeFns[445].HmatrixS(x1, x2, x3, x4)
            HS451 = self._shapeFns[451].HmatrixS(x1, x2, x3, x4)
            HS452 = self._shapeFns[452].HmatrixS(x1, x2, x3, x4)
            HS453 = self._shapeFns[453].HmatrixS(x1, x2, x3, x4)
            HS454 = self._shapeFns[454].HmatrixS(x1, x2, x3, x4)
            HS455 = self._shapeFns[455].HmatrixS(x1, x2, x3, x4)
            
            HS511 = self._shapeFns[511].HmatrixS(x1, x2, x3, x4)
            HS512 = self._shapeFns[512].HmatrixS(x1, x2, x3, x4)
            HS513 = self._shapeFns[513].HmatrixS(x1, x2, x3, x4)
            HS514 = self._shapeFns[514].HmatrixS(x1, x2, x3, x4)
            HS515 = self._shapeFns[515].HmatrixS(x1, x2, x3, x4)
            HS521 = self._shapeFns[521].HmatrixS(x1, x2, x3, x4)
            HS522 = self._shapeFns[522].HmatrixS(x1, x2, x3, x4)
            HS523 = self._shapeFns[523].HmatrixS(x1, x2, x3, x4)
            HS524 = self._shapeFns[524].HmatrixS(x1, x2, x3, x4)
            HS525 = self._shapeFns[525].HmatrixS(x1, x2, x3, x4)            
            HS531 = self._shapeFns[531].HmatrixS(x1, x2, x3, x4)
            HS532 = self._shapeFns[532].HmatrixS(x1, x2, x3, x4)
            HS533 = self._shapeFns[533].HmatrixS(x1, x2, x3, x4)
            HS534 = self._shapeFns[534].HmatrixS(x1, x2, x3, x4) 
            HS535 = self._shapeFns[535].HmatrixS(x1, x2, x3, x4)
            HS541 = self._shapeFns[541].HmatrixS(x1, x2, x3, x4)
            HS542 = self._shapeFns[542].HmatrixS(x1, x2, x3, x4)
            HS543 = self._shapeFns[543].HmatrixS(x1, x2, x3, x4)
            HS544 = self._shapeFns[544].HmatrixS(x1, x2, x3, x4)
            HS545 = self._shapeFns[545].HmatrixS(x1, x2, x3, x4)
            HS551 = self._shapeFns[551].HmatrixS(x1, x2, x3, x4)
            HS552 = self._shapeFns[552].HmatrixS(x1, x2, x3, x4)
            HS553 = self._shapeFns[553].HmatrixS(x1, x2, x3, x4)
            HS554 = self._shapeFns[554].HmatrixS(x1, x2, x3, x4)
            HS555 = self._shapeFns[555].HmatrixS(x1, x2, x3, x4)
            
            
            # create the third H matrix
            HT111 = self._shapeFns[111].HmatrixT(x1, x2, x3, x4)
            HT112 = self._shapeFns[112].HmatrixT(x1, x2, x3, x4)
            HT113 = self._shapeFns[113].HmatrixT(x1, x2, x3, x4)
            HT114 = self._shapeFns[114].HmatrixT(x1, x2, x3, x4)
            HT115 = self._shapeFns[115].HmatrixT(x1, x2, x3, x4)
            HT121 = self._shapeFns[121].HmatrixT(x1, x2, x3, x4)
            HT122 = self._shapeFns[122].HmatrixT(x1, x2, x3, x4)
            HT123 = self._shapeFns[123].HmatrixT(x1, x2, x3, x4)
            HT124 = self._shapeFns[124].HmatrixT(x1, x2, x3, x4)
            HT125 = self._shapeFns[125].HmatrixT(x1, x2, x3, x4)            
            HT131 = self._shapeFns[131].HmatrixT(x1, x2, x3, x4)
            HT132 = self._shapeFns[132].HmatrixT(x1, x2, x3, x4)
            HT133 = self._shapeFns[133].HmatrixT(x1, x2, x3, x4)
            HT134 = self._shapeFns[134].HmatrixT(x1, x2, x3, x4) 
            HT135 = self._shapeFns[135].HmatrixT(x1, x2, x3, x4)
            HT141 = self._shapeFns[141].HmatrixT(x1, x2, x3, x4)
            HT142 = self._shapeFns[142].HmatrixT(x1, x2, x3, x4)
            HT143 = self._shapeFns[143].HmatrixT(x1, x2, x3, x4)
            HT144 = self._shapeFns[144].HmatrixT(x1, x2, x3, x4)
            HT145 = self._shapeFns[145].HmatrixT(x1, x2, x3, x4)
            HT151 = self._shapeFns[151].HmatrixT(x1, x2, x3, x4)
            HT152 = self._shapeFns[152].HmatrixT(x1, x2, x3, x4)
            HT153 = self._shapeFns[153].HmatrixT(x1, x2, x3, x4)
            HT154 = self._shapeFns[154].HmatrixT(x1, x2, x3, x4)
            HT155 = self._shapeFns[155].HmatrixT(x1, x2, x3, x4)
            
            HT211 = self._shapeFns[211].HmatrixT(x1, x2, x3, x4)
            HT212 = self._shapeFns[212].HmatrixT(x1, x2, x3, x4)
            HT213 = self._shapeFns[213].HmatrixT(x1, x2, x3, x4)
            HT214 = self._shapeFns[214].HmatrixT(x1, x2, x3, x4)
            HT215 = self._shapeFns[215].HmatrixT(x1, x2, x3, x4)
            HT221 = self._shapeFns[221].HmatrixT(x1, x2, x3, x4)
            HT222 = self._shapeFns[222].HmatrixT(x1, x2, x3, x4)
            HT223 = self._shapeFns[223].HmatrixT(x1, x2, x3, x4)
            HT224 = self._shapeFns[224].HmatrixT(x1, x2, x3, x4)
            HT225 = self._shapeFns[225].HmatrixT(x1, x2, x3, x4)            
            HT231 = self._shapeFns[231].HmatrixT(x1, x2, x3, x4)
            HT232 = self._shapeFns[232].HmatrixT(x1, x2, x3, x4)
            HT233 = self._shapeFns[233].HmatrixT(x1, x2, x3, x4)
            HT234 = self._shapeFns[234].HmatrixT(x1, x2, x3, x4) 
            HT235 = self._shapeFns[235].HmatrixT(x1, x2, x3, x4)
            HT241 = self._shapeFns[241].HmatrixT(x1, x2, x3, x4)
            HT242 = self._shapeFns[242].HmatrixT(x1, x2, x3, x4)
            HT243 = self._shapeFns[243].HmatrixT(x1, x2, x3, x4)
            HT244 = self._shapeFns[244].HmatrixT(x1, x2, x3, x4)
            HT245 = self._shapeFns[245].HmatrixT(x1, x2, x3, x4)
            HT251 = self._shapeFns[251].HmatrixT(x1, x2, x3, x4)
            HT252 = self._shapeFns[252].HmatrixT(x1, x2, x3, x4)
            HT253 = self._shapeFns[253].HmatrixT(x1, x2, x3, x4)
            HT254 = self._shapeFns[254].HmatrixT(x1, x2, x3, x4)
            HT255 = self._shapeFns[255].HmatrixT(x1, x2, x3, x4)
            
            HT311 = self._shapeFns[311].HmatrixT(x1, x2, x3, x4)
            HT312 = self._shapeFns[312].HmatrixT(x1, x2, x3, x4)
            HT313 = self._shapeFns[313].HmatrixT(x1, x2, x3, x4)
            HT314 = self._shapeFns[314].HmatrixT(x1, x2, x3, x4)
            HT315 = self._shapeFns[315].HmatrixT(x1, x2, x3, x4)
            HT321 = self._shapeFns[321].HmatrixT(x1, x2, x3, x4)
            HT322 = self._shapeFns[322].HmatrixT(x1, x2, x3, x4)
            HT323 = self._shapeFns[323].HmatrixT(x1, x2, x3, x4)
            HT324 = self._shapeFns[324].HmatrixT(x1, x2, x3, x4)
            HT325 = self._shapeFns[325].HmatrixT(x1, x2, x3, x4)            
            HT331 = self._shapeFns[331].HmatrixT(x1, x2, x3, x4)
            HT332 = self._shapeFns[332].HmatrixT(x1, x2, x3, x4)
            HT333 = self._shapeFns[333].HmatrixT(x1, x2, x3, x4)
            HT334 = self._shapeFns[334].HmatrixT(x1, x2, x3, x4) 
            HT335 = self._shapeFns[335].HmatrixT(x1, x2, x3, x4)
            HT341 = self._shapeFns[341].HmatrixT(x1, x2, x3, x4)
            HT342 = self._shapeFns[342].HmatrixT(x1, x2, x3, x4)
            HT343 = self._shapeFns[343].HmatrixT(x1, x2, x3, x4)
            HT344 = self._shapeFns[344].HmatrixT(x1, x2, x3, x4)
            HT345 = self._shapeFns[345].HmatrixT(x1, x2, x3, x4)
            HT351 = self._shapeFns[351].HmatrixT(x1, x2, x3, x4)
            HT352 = self._shapeFns[352].HmatrixT(x1, x2, x3, x4)
            HT353 = self._shapeFns[353].HmatrixT(x1, x2, x3, x4)
            HT354 = self._shapeFns[354].HmatrixT(x1, x2, x3, x4)
            HT355 = self._shapeFns[355].HmatrixT(x1, x2, x3, x4)
            
            HT411 = self._shapeFns[411].HmatrixT(x1, x2, x3, x4)
            HT412 = self._shapeFns[412].HmatrixT(x1, x2, x3, x4)
            HT413 = self._shapeFns[413].HmatrixT(x1, x2, x3, x4)
            HT414 = self._shapeFns[414].HmatrixT(x1, x2, x3, x4)
            HT415 = self._shapeFns[415].HmatrixT(x1, x2, x3, x4)
            HT421 = self._shapeFns[421].HmatrixT(x1, x2, x3, x4)
            HT422 = self._shapeFns[422].HmatrixT(x1, x2, x3, x4)
            HT423 = self._shapeFns[423].HmatrixT(x1, x2, x3, x4)
            HT424 = self._shapeFns[424].HmatrixT(x1, x2, x3, x4)
            HT425 = self._shapeFns[425].HmatrixT(x1, x2, x3, x4)            
            HT431 = self._shapeFns[431].HmatrixT(x1, x2, x3, x4)
            HT432 = self._shapeFns[432].HmatrixT(x1, x2, x3, x4)
            HT433 = self._shapeFns[433].HmatrixT(x1, x2, x3, x4)
            HT434 = self._shapeFns[434].HmatrixT(x1, x2, x3, x4) 
            HT435 = self._shapeFns[435].HmatrixT(x1, x2, x3, x4)
            HT441 = self._shapeFns[441].HmatrixT(x1, x2, x3, x4)
            HT442 = self._shapeFns[442].HmatrixT(x1, x2, x3, x4)
            HT443 = self._shapeFns[443].HmatrixT(x1, x2, x3, x4)
            HT444 = self._shapeFns[444].HmatrixT(x1, x2, x3, x4)
            HT445 = self._shapeFns[445].HmatrixT(x1, x2, x3, x4)
            HT451 = self._shapeFns[451].HmatrixT(x1, x2, x3, x4)
            HT452 = self._shapeFns[452].HmatrixT(x1, x2, x3, x4)
            HT453 = self._shapeFns[453].HmatrixT(x1, x2, x3, x4)
            HT454 = self._shapeFns[454].HmatrixT(x1, x2, x3, x4)
            HT455 = self._shapeFns[455].HmatrixT(x1, x2, x3, x4)
            
            HT511 = self._shapeFns[511].HmatrixT(x1, x2, x3, x4)
            HT512 = self._shapeFns[512].HmatrixT(x1, x2, x3, x4)
            HT513 = self._shapeFns[513].HmatrixT(x1, x2, x3, x4)
            HT514 = self._shapeFns[514].HmatrixT(x1, x2, x3, x4)
            HT515 = self._shapeFns[515].HmatrixT(x1, x2, x3, x4)
            HT521 = self._shapeFns[521].HmatrixT(x1, x2, x3, x4)
            HT522 = self._shapeFns[522].HmatrixT(x1, x2, x3, x4)
            HT523 = self._shapeFns[523].HmatrixT(x1, x2, x3, x4)
            HT524 = self._shapeFns[524].HmatrixT(x1, x2, x3, x4)
            HT525 = self._shapeFns[525].HmatrixT(x1, x2, x3, x4)            
            HT531 = self._shapeFns[531].HmatrixT(x1, x2, x3, x4)
            HT532 = self._shapeFns[532].HmatrixT(x1, x2, x3, x4)
            HT533 = self._shapeFns[533].HmatrixT(x1, x2, x3, x4)
            HT534 = self._shapeFns[534].HmatrixT(x1, x2, x3, x4) 
            HT535 = self._shapeFns[535].HmatrixT(x1, x2, x3, x4)
            HT541 = self._shapeFns[541].HmatrixT(x1, x2, x3, x4)
            HT542 = self._shapeFns[542].HmatrixT(x1, x2, x3, x4)
            HT543 = self._shapeFns[543].HmatrixT(x1, x2, x3, x4)
            HT544 = self._shapeFns[544].HmatrixT(x1, x2, x3, x4)
            HT545 = self._shapeFns[545].HmatrixT(x1, x2, x3, x4)
            HT551 = self._shapeFns[551].HmatrixT(x1, x2, x3, x4)
            HT552 = self._shapeFns[552].HmatrixT(x1, x2, x3, x4)
            HT553 = self._shapeFns[553].HmatrixT(x1, x2, x3, x4)
            HT554 = self._shapeFns[554].HmatrixT(x1, x2, x3, x4)
            HT555 = self._shapeFns[555].HmatrixT(x1, x2, x3, x4)
            
            
            
            # the nonlinear stiffness matrix for 5 Gauss point
            f111 = (BL111.T.dot(M).dot(BN111) + BN111.T.dot(M).dot(BL111) + 
                   BN111.T.dot(M).dot(BN111))
            f112 = (BL112.T.dot(M).dot(BN112) + BN112.T.dot(M).dot(BL112) + 
                   BN112.T.dot(M).dot(BN112))
            f113 = (BL113.T.dot(M).dot(BN113) + BN113.T.dot(M).dot(BL113) + 
                   BN113.T.dot(M).dot(BN113))
            f114 = (BL114.T.dot(M).dot(BN114) + BN114.T.dot(M).dot(BL114) + 
                   BN114.T.dot(M).dot(BN114))   
            f115 = (BL115.T.dot(M).dot(BN115) + BN115.T.dot(M).dot(BL115) + 
                   BN115.T.dot(M).dot(BN115)) 
            f121 = (BL121.T.dot(M).dot(BN121) + BN121.T.dot(M).dot(BL121) + 
                   BN121.T.dot(M).dot(BN121))
            f122 = (BL122.T.dot(M).dot(BN122) + BN122.T.dot(M).dot(BL122) + 
                   BN122.T.dot(M).dot(BN122))
            f123 = (BL123.T.dot(M).dot(BN123) + BN123.T.dot(M).dot(BL123) + 
                   BN123.T.dot(M).dot(BN123))
            f124 = (BL124.T.dot(M).dot(BN124) + BN124.T.dot(M).dot(BL124) + 
                   BN124.T.dot(M).dot(BN124))
            f125 = (BL125.T.dot(M).dot(BN125) + BN125.T.dot(M).dot(BL125) + 
                   BN125.T.dot(M).dot(BN125))             
            f131 = (BL131.T.dot(M).dot(BN131) + BN131.T.dot(M).dot(BL131) + 
                   BN131.T.dot(M).dot(BN131))
            f132 = (BL132.T.dot(M).dot(BN132) + BN132.T.dot(M).dot(BL132) + 
                   BN132.T.dot(M).dot(BN132))
            f133 = (BL133.T.dot(M).dot(BN133) + BN133.T.dot(M).dot(BL133) + 
                   BN133.T.dot(M).dot(BN133))
            f134 = (BL134.T.dot(M).dot(BN134) + BN134.T.dot(M).dot(BL134) + 
                   BN134.T.dot(M).dot(BN134))
            f135 = (BL135.T.dot(M).dot(BN135) + BN135.T.dot(M).dot(BL135) + 
                   BN135.T.dot(M).dot(BN135))             
            f141 = (BL141.T.dot(M).dot(BN141) + BN141.T.dot(M).dot(BL141) + 
                   BN141.T.dot(M).dot(BN141))
            f142 = (BL142.T.dot(M).dot(BN142) + BN142.T.dot(M).dot(BL142) + 
                   BN142.T.dot(M).dot(BN142))
            f143 = (BL143.T.dot(M).dot(BN143) + BN143.T.dot(M).dot(BL143) + 
                   BN143.T.dot(M).dot(BN143))
            f144 = (BL144.T.dot(M).dot(BN144) + BN144.T.dot(M).dot(BL144) + 
                   BN144.T.dot(M).dot(BN144))
            f145 = (BL145.T.dot(M).dot(BN145) + BN145.T.dot(M).dot(BL145) + 
                   BN145.T.dot(M).dot(BN145)) 
            f151 = (BL151.T.dot(M).dot(BN151) + BN151.T.dot(M).dot(BL151) + 
                   BN151.T.dot(M).dot(BN151))
            f152 = (BL152.T.dot(M).dot(BN152) + BN152.T.dot(M).dot(BL152) + 
                   BN152.T.dot(M).dot(BN152))
            f153 = (BL153.T.dot(M).dot(BN153) + BN153.T.dot(M).dot(BL153) + 
                   BN153.T.dot(M).dot(BN153))
            f154 = (BL154.T.dot(M).dot(BN154) + BN154.T.dot(M).dot(BL154) + 
                   BN154.T.dot(M).dot(BN154))
            f155 = (BL155.T.dot(M).dot(BN155) + BN155.T.dot(M).dot(BL155) + 
                   BN155.T.dot(M).dot(BN155))
            
            f211 = (BL211.T.dot(M).dot(BN211) + BN211.T.dot(M).dot(BL211) + 
                   BN211.T.dot(M).dot(BN211))
            f212 = (BL212.T.dot(M).dot(BN212) + BN212.T.dot(M).dot(BL212) + 
                   BN212.T.dot(M).dot(BN212))
            f213 = (BL213.T.dot(M).dot(BN213) + BN213.T.dot(M).dot(BL213) + 
                   BN213.T.dot(M).dot(BN213))
            f214 = (BL214.T.dot(M).dot(BN214) + BN214.T.dot(M).dot(BL214) + 
                   BN214.T.dot(M).dot(BN214))   
            f215 = (BL215.T.dot(M).dot(BN215) + BN215.T.dot(M).dot(BL215) + 
                   BN215.T.dot(M).dot(BN215)) 
            f221 = (BL221.T.dot(M).dot(BN221) + BN221.T.dot(M).dot(BL221) + 
                   BN221.T.dot(M).dot(BN221))
            f222 = (BL222.T.dot(M).dot(BN222) + BN222.T.dot(M).dot(BL222) + 
                   BN222.T.dot(M).dot(BN222))
            f223 = (BL223.T.dot(M).dot(BN223) + BN223.T.dot(M).dot(BL223) + 
                   BN223.T.dot(M).dot(BN223))
            f224 = (BL224.T.dot(M).dot(BN224) + BN224.T.dot(M).dot(BL224) + 
                   BN224.T.dot(M).dot(BN224))
            f225 = (BL225.T.dot(M).dot(BN225) + BN225.T.dot(M).dot(BL225) + 
                   BN225.T.dot(M).dot(BN225))             
            f231 = (BL231.T.dot(M).dot(BN231) + BN231.T.dot(M).dot(BL231) + 
                   BN231.T.dot(M).dot(BN231))
            f232 = (BL232.T.dot(M).dot(BN232) + BN232.T.dot(M).dot(BL232) + 
                   BN232.T.dot(M).dot(BN232))
            f233 = (BL233.T.dot(M).dot(BN233) + BN233.T.dot(M).dot(BL233) + 
                   BN233.T.dot(M).dot(BN233))
            f234 = (BL234.T.dot(M).dot(BN234) + BN234.T.dot(M).dot(BL234) + 
                   BN234.T.dot(M).dot(BN234))
            f235 = (BL235.T.dot(M).dot(BN235) + BN235.T.dot(M).dot(BL235) + 
                   BN235.T.dot(M).dot(BN235))             
            f241 = (BL241.T.dot(M).dot(BN241) + BN241.T.dot(M).dot(BL241) + 
                   BN241.T.dot(M).dot(BN241))
            f242 = (BL242.T.dot(M).dot(BN242) + BN242.T.dot(M).dot(BL242) + 
                   BN242.T.dot(M).dot(BN242))
            f243 = (BL243.T.dot(M).dot(BN243) + BN243.T.dot(M).dot(BL243) + 
                   BN243.T.dot(M).dot(BN243))
            f244 = (BL244.T.dot(M).dot(BN244) + BN244.T.dot(M).dot(BL244) + 
                   BN244.T.dot(M).dot(BN244))
            f245 = (BL245.T.dot(M).dot(BN245) + BN245.T.dot(M).dot(BL245) + 
                   BN245.T.dot(M).dot(BN245)) 
            f251 = (BL251.T.dot(M).dot(BN251) + BN251.T.dot(M).dot(BL251) + 
                   BN251.T.dot(M).dot(BN251))
            f252 = (BL252.T.dot(M).dot(BN252) + BN252.T.dot(M).dot(BL252) + 
                   BN252.T.dot(M).dot(BN252))
            f253 = (BL253.T.dot(M).dot(BN253) + BN253.T.dot(M).dot(BL253) + 
                   BN253.T.dot(M).dot(BN253))
            f254 = (BL254.T.dot(M).dot(BN254) + BN254.T.dot(M).dot(BL254) + 
                   BN254.T.dot(M).dot(BN254))
            f255 = (BL255.T.dot(M).dot(BN255) + BN255.T.dot(M).dot(BL255) + 
                   BN255.T.dot(M).dot(BN255))
            
            f311 = (BL311.T.dot(M).dot(BN311) + BN311.T.dot(M).dot(BL311) + 
                   BN311.T.dot(M).dot(BN311))
            f312 = (BL312.T.dot(M).dot(BN312) + BN312.T.dot(M).dot(BL312) + 
                   BN312.T.dot(M).dot(BN312))
            f313 = (BL313.T.dot(M).dot(BN313) + BN313.T.dot(M).dot(BL313) + 
                   BN313.T.dot(M).dot(BN313))
            f314 = (BL314.T.dot(M).dot(BN314) + BN314.T.dot(M).dot(BL314) + 
                   BN314.T.dot(M).dot(BN314))   
            f315 = (BL315.T.dot(M).dot(BN315) + BN315.T.dot(M).dot(BL315) + 
                   BN315.T.dot(M).dot(BN315)) 
            f321 = (BL321.T.dot(M).dot(BN321) + BN321.T.dot(M).dot(BL321) + 
                   BN321.T.dot(M).dot(BN321))
            f322 = (BL322.T.dot(M).dot(BN322) + BN322.T.dot(M).dot(BL322) + 
                   BN322.T.dot(M).dot(BN322))
            f323 = (BL323.T.dot(M).dot(BN323) + BN323.T.dot(M).dot(BL323) + 
                   BN323.T.dot(M).dot(BN323))
            f324 = (BL324.T.dot(M).dot(BN324) + BN324.T.dot(M).dot(BL324) + 
                   BN324.T.dot(M).dot(BN324))
            f325 = (BL325.T.dot(M).dot(BN325) + BN325.T.dot(M).dot(BL325) + 
                   BN325.T.dot(M).dot(BN325))             
            f331 = (BL331.T.dot(M).dot(BN331) + BN331.T.dot(M).dot(BL331) + 
                   BN331.T.dot(M).dot(BN331))
            f332 = (BL332.T.dot(M).dot(BN332) + BN332.T.dot(M).dot(BL332) + 
                   BN332.T.dot(M).dot(BN332))
            f333 = (BL333.T.dot(M).dot(BN333) + BN333.T.dot(M).dot(BL333) + 
                   BN333.T.dot(M).dot(BN333))
            f334 = (BL334.T.dot(M).dot(BN334) + BN334.T.dot(M).dot(BL334) + 
                   BN334.T.dot(M).dot(BN334))
            f335 = (BL335.T.dot(M).dot(BN335) + BN335.T.dot(M).dot(BL335) + 
                   BN335.T.dot(M).dot(BN335))             
            f341 = (BL341.T.dot(M).dot(BN341) + BN341.T.dot(M).dot(BL341) + 
                   BN341.T.dot(M).dot(BN341))
            f342 = (BL342.T.dot(M).dot(BN342) + BN342.T.dot(M).dot(BL342) + 
                   BN342.T.dot(M).dot(BN342))
            f343 = (BL343.T.dot(M).dot(BN343) + BN343.T.dot(M).dot(BL343) + 
                   BN343.T.dot(M).dot(BN343))
            f344 = (BL344.T.dot(M).dot(BN344) + BN344.T.dot(M).dot(BL344) + 
                   BN344.T.dot(M).dot(BN344))
            f345 = (BL345.T.dot(M).dot(BN345) + BN345.T.dot(M).dot(BL345) + 
                   BN345.T.dot(M).dot(BN345)) 
            f351 = (BL351.T.dot(M).dot(BN351) + BN351.T.dot(M).dot(BL351) + 
                   BN351.T.dot(M).dot(BN351))
            f352 = (BL352.T.dot(M).dot(BN352) + BN352.T.dot(M).dot(BL352) + 
                   BN352.T.dot(M).dot(BN352))
            f353 = (BL353.T.dot(M).dot(BN353) + BN353.T.dot(M).dot(BL353) + 
                   BN353.T.dot(M).dot(BN353))
            f354 = (BL354.T.dot(M).dot(BN354) + BN354.T.dot(M).dot(BL354) + 
                   BN354.T.dot(M).dot(BN354))
            f355 = (BL355.T.dot(M).dot(BN355) + BN355.T.dot(M).dot(BL355) + 
                   BN355.T.dot(M).dot(BN355))
            
            f411 = (BL411.T.dot(M).dot(BN411) + BN411.T.dot(M).dot(BL411) + 
                   BN411.T.dot(M).dot(BN411))
            f412 = (BL412.T.dot(M).dot(BN412) + BN412.T.dot(M).dot(BL412) + 
                   BN412.T.dot(M).dot(BN412))
            f413 = (BL413.T.dot(M).dot(BN413) + BN413.T.dot(M).dot(BL413) + 
                   BN413.T.dot(M).dot(BN413))
            f414 = (BL414.T.dot(M).dot(BN414) + BN414.T.dot(M).dot(BL414) + 
                   BN414.T.dot(M).dot(BN414))   
            f415 = (BL415.T.dot(M).dot(BN415) + BN415.T.dot(M).dot(BL415) + 
                   BN415.T.dot(M).dot(BN415)) 
            f421 = (BL421.T.dot(M).dot(BN421) + BN421.T.dot(M).dot(BL421) + 
                   BN421.T.dot(M).dot(BN421))
            f422 = (BL422.T.dot(M).dot(BN422) + BN422.T.dot(M).dot(BL422) + 
                   BN422.T.dot(M).dot(BN422))
            f423 = (BL423.T.dot(M).dot(BN423) + BN423.T.dot(M).dot(BL423) + 
                   BN423.T.dot(M).dot(BN423))
            f424 = (BL424.T.dot(M).dot(BN424) + BN424.T.dot(M).dot(BL424) + 
                   BN424.T.dot(M).dot(BN424))
            f425 = (BL425.T.dot(M).dot(BN425) + BN425.T.dot(M).dot(BL425) + 
                   BN425.T.dot(M).dot(BN425))             
            f431 = (BL431.T.dot(M).dot(BN431) + BN431.T.dot(M).dot(BL431) + 
                   BN431.T.dot(M).dot(BN431))
            f432 = (BL432.T.dot(M).dot(BN432) + BN432.T.dot(M).dot(BL432) + 
                   BN432.T.dot(M).dot(BN432))
            f433 = (BL433.T.dot(M).dot(BN433) + BN433.T.dot(M).dot(BL433) + 
                   BN433.T.dot(M).dot(BN433))
            f434 = (BL434.T.dot(M).dot(BN434) + BN434.T.dot(M).dot(BL434) + 
                   BN434.T.dot(M).dot(BN434))
            f435 = (BL435.T.dot(M).dot(BN435) + BN435.T.dot(M).dot(BL435) + 
                   BN435.T.dot(M).dot(BN435))             
            f441 = (BL441.T.dot(M).dot(BN441) + BN441.T.dot(M).dot(BL441) + 
                   BN441.T.dot(M).dot(BN441))
            f442 = (BL442.T.dot(M).dot(BN442) + BN442.T.dot(M).dot(BL442) + 
                   BN442.T.dot(M).dot(BN442))
            f443 = (BL443.T.dot(M).dot(BN443) + BN443.T.dot(M).dot(BL443) + 
                   BN443.T.dot(M).dot(BN443))
            f444 = (BL444.T.dot(M).dot(BN444) + BN444.T.dot(M).dot(BL444) + 
                   BN444.T.dot(M).dot(BN444))
            f445 = (BL445.T.dot(M).dot(BN445) + BN445.T.dot(M).dot(BL445) + 
                   BN445.T.dot(M).dot(BN445)) 
            f451 = (BL451.T.dot(M).dot(BN451) + BN451.T.dot(M).dot(BL451) + 
                   BN451.T.dot(M).dot(BN451))
            f452 = (BL452.T.dot(M).dot(BN452) + BN452.T.dot(M).dot(BL452) + 
                   BN452.T.dot(M).dot(BN452))
            f453 = (BL453.T.dot(M).dot(BN453) + BN453.T.dot(M).dot(BL453) + 
                   BN453.T.dot(M).dot(BN453))
            f454 = (BL454.T.dot(M).dot(BN454) + BN454.T.dot(M).dot(BL454) + 
                   BN454.T.dot(M).dot(BN454))
            f455 = (BL455.T.dot(M).dot(BN455) + BN455.T.dot(M).dot(BL455) + 
                   BN455.T.dot(M).dot(BN455))
            
            f511 = (BL511.T.dot(M).dot(BN511) + BN511.T.dot(M).dot(BL511) + 
                   BN511.T.dot(M).dot(BN511))
            f512 = (BL512.T.dot(M).dot(BN512) + BN512.T.dot(M).dot(BL512) + 
                   BN512.T.dot(M).dot(BN512))
            f513 = (BL513.T.dot(M).dot(BN513) + BN513.T.dot(M).dot(BL513) + 
                   BN513.T.dot(M).dot(BN513))
            f514 = (BL514.T.dot(M).dot(BN514) + BN514.T.dot(M).dot(BL514) + 
                   BN514.T.dot(M).dot(BN514))   
            f515 = (BL515.T.dot(M).dot(BN515) + BN515.T.dot(M).dot(BL515) + 
                   BN515.T.dot(M).dot(BN515)) 
            f521 = (BL521.T.dot(M).dot(BN521) + BN521.T.dot(M).dot(BL521) + 
                   BN521.T.dot(M).dot(BN521))
            f522 = (BL522.T.dot(M).dot(BN522) + BN522.T.dot(M).dot(BL522) + 
                   BN522.T.dot(M).dot(BN522))
            f523 = (BL523.T.dot(M).dot(BN523) + BN523.T.dot(M).dot(BL523) + 
                   BN523.T.dot(M).dot(BN523))
            f524 = (BL524.T.dot(M).dot(BN524) + BN524.T.dot(M).dot(BL524) + 
                   BN524.T.dot(M).dot(BN524))
            f525 = (BL525.T.dot(M).dot(BN525) + BN525.T.dot(M).dot(BL525) + 
                   BN525.T.dot(M).dot(BN525))             
            f531 = (BL531.T.dot(M).dot(BN531) + BN531.T.dot(M).dot(BL531) + 
                   BN531.T.dot(M).dot(BN531))
            f532 = (BL532.T.dot(M).dot(BN532) + BN532.T.dot(M).dot(BL532) + 
                   BN532.T.dot(M).dot(BN532))
            f533 = (BL533.T.dot(M).dot(BN533) + BN533.T.dot(M).dot(BL533) + 
                   BN533.T.dot(M).dot(BN533))
            f534 = (BL534.T.dot(M).dot(BN534) + BN534.T.dot(M).dot(BL534) + 
                   BN534.T.dot(M).dot(BN534))
            f535 = (BL535.T.dot(M).dot(BN535) + BN535.T.dot(M).dot(BL535) + 
                   BN535.T.dot(M).dot(BN535))             
            f541 = (BL541.T.dot(M).dot(BN541) + BN541.T.dot(M).dot(BL541) + 
                   BN541.T.dot(M).dot(BN541))
            f542 = (BL542.T.dot(M).dot(BN542) + BN542.T.dot(M).dot(BL542) + 
                   BN542.T.dot(M).dot(BN542))
            f543 = (BL543.T.dot(M).dot(BN543) + BN543.T.dot(M).dot(BL543) + 
                   BN543.T.dot(M).dot(BN543))
            f544 = (BL544.T.dot(M).dot(BN544) + BN544.T.dot(M).dot(BL544) + 
                   BN544.T.dot(M).dot(BN544))
            f545 = (BL545.T.dot(M).dot(BN545) + BN545.T.dot(M).dot(BL545) + 
                   BN545.T.dot(M).dot(BN545)) 
            f551 = (BL551.T.dot(M).dot(BN551) + BN551.T.dot(M).dot(BL551) + 
                   BN551.T.dot(M).dot(BN551))
            f552 = (BL552.T.dot(M).dot(BN552) + BN552.T.dot(M).dot(BL552) + 
                   BN552.T.dot(M).dot(BN552))
            f553 = (BL553.T.dot(M).dot(BN553) + BN553.T.dot(M).dot(BL553) + 
                   BN553.T.dot(M).dot(BN553))
            f554 = (BL554.T.dot(M).dot(BN554) + BN554.T.dot(M).dot(BL554) + 
                   BN554.T.dot(M).dot(BN554))
            f555 = (BL555.T.dot(M).dot(BN555) + BN555.T.dot(M).dot(BL555) + 
                   BN555.T.dot(M).dot(BN555))
            
            
            
            KN = (detJ111 * w[0] * w[0] * w[0] * f111 +
                  detJ112 * w[0] * w[0] * w[1] * f112 +
                  detJ113 * w[0] * w[0] * w[2] * f113 +
                  detJ114 * w[0] * w[0] * w[3] * f114 +
                  detJ115 * w[0] * w[0] * w[4] * f115 +
                  detJ121 * w[0] * w[1] * w[0] * f121 +
                  detJ122 * w[0] * w[1] * w[1] * f122 +
                  detJ123 * w[0] * w[1] * w[2] * f123 +
                  detJ124 * w[0] * w[1] * w[3] * f124 +
                  detJ125 * w[0] * w[1] * w[4] * f125 +
                  detJ131 * w[0] * w[2] * w[0] * f131 +
                  detJ132 * w[0] * w[2] * w[1] * f132 +
                  detJ133 * w[0] * w[2] * w[2] * f133 +
                  detJ134 * w[0] * w[2] * w[3] * f134 +
                  detJ135 * w[0] * w[2] * w[4] * f135 +
                  detJ141 * w[0] * w[3] * w[0] * f141 +
                  detJ142 * w[0] * w[3] * w[1] * f142 +
                  detJ143 * w[0] * w[3] * w[2] * f143 +
                  detJ144 * w[0] * w[3] * w[3] * f144 + 
                  detJ145 * w[0] * w[3] * w[4] * f145 +
                  detJ151 * w[0] * w[4] * w[0] * f151 +
                  detJ152 * w[0] * w[4] * w[1] * f152 +
                  detJ153 * w[0] * w[4] * w[2] * f153 +
                  detJ154 * w[0] * w[4] * w[3] * f154 + 
                  detJ155 * w[0] * w[4] * w[4] * f155 +                  
                  detJ211 * w[1] * w[0] * w[0] * f211 +
                  detJ212 * w[1] * w[0] * w[1] * f212 +
                  detJ213 * w[1] * w[0] * w[2] * f213 +
                  detJ214 * w[1] * w[0] * w[3] * f214 +
                  detJ215 * w[1] * w[0] * w[4] * f215 +
                  detJ221 * w[1] * w[1] * w[0] * f221 +
                  detJ222 * w[1] * w[1] * w[1] * f222 +
                  detJ223 * w[1] * w[1] * w[2] * f223 +
                  detJ224 * w[1] * w[1] * w[3] * f224 +
                  detJ225 * w[1] * w[1] * w[4] * f225 +
                  detJ231 * w[1] * w[2] * w[0] * f231 +
                  detJ232 * w[1] * w[2] * w[1] * f232 +
                  detJ233 * w[1] * w[2] * w[2] * f233 +
                  detJ234 * w[1] * w[2] * w[3] * f234 +
                  detJ235 * w[1] * w[2] * w[4] * f235 +
                  detJ241 * w[1] * w[3] * w[0] * f241 +
                  detJ242 * w[1] * w[3] * w[1] * f242 +
                  detJ243 * w[1] * w[3] * w[2] * f243 +
                  detJ244 * w[1] * w[3] * w[3] * f244 + 
                  detJ245 * w[1] * w[3] * w[4] * f245 +
                  detJ251 * w[1] * w[4] * w[0] * f251 +
                  detJ252 * w[1] * w[4] * w[1] * f252 +
                  detJ253 * w[1] * w[4] * w[2] * f253 +
                  detJ254 * w[1] * w[4] * w[3] * f254 + 
                  detJ255 * w[1] * w[4] * w[4] * f255 +                  
                  detJ311 * w[2] * w[0] * w[0] * f311 +
                  detJ312 * w[2] * w[0] * w[1] * f312 +
                  detJ313 * w[2] * w[0] * w[2] * f313 +
                  detJ314 * w[2] * w[0] * w[3] * f314 +
                  detJ315 * w[2] * w[0] * w[4] * f315 +
                  detJ321 * w[2] * w[1] * w[0] * f321 +
                  detJ322 * w[2] * w[1] * w[1] * f322 +
                  detJ323 * w[2] * w[1] * w[2] * f323 +
                  detJ324 * w[2] * w[1] * w[3] * f324 +
                  detJ325 * w[2] * w[1] * w[4] * f325 +
                  detJ331 * w[2] * w[2] * w[0] * f331 +
                  detJ332 * w[2] * w[2] * w[1] * f332 +
                  detJ333 * w[2] * w[2] * w[2] * f333 +
                  detJ334 * w[2] * w[2] * w[3] * f334 +
                  detJ335 * w[2] * w[2] * w[4] * f335 +
                  detJ341 * w[2] * w[3] * w[0] * f341 +
                  detJ342 * w[2] * w[3] * w[1] * f342 +
                  detJ343 * w[2] * w[3] * w[2] * f343 +
                  detJ344 * w[2] * w[3] * w[3] * f344 + 
                  detJ345 * w[2] * w[3] * w[4] * f345 +
                  detJ351 * w[2] * w[4] * w[0] * f351 +
                  detJ352 * w[2] * w[4] * w[1] * f352 +
                  detJ353 * w[2] * w[4] * w[2] * f353 +
                  detJ354 * w[2] * w[4] * w[3] * f354 + 
                  detJ355 * w[2] * w[4] * w[4] * f355 +                  
                  detJ411 * w[3] * w[0] * w[0] * f411 +
                  detJ412 * w[3] * w[0] * w[1] * f412 +
                  detJ413 * w[3] * w[0] * w[2] * f413 +
                  detJ414 * w[3] * w[0] * w[3] * f414 +
                  detJ415 * w[3] * w[0] * w[4] * f415 +
                  detJ421 * w[3] * w[1] * w[0] * f421 +
                  detJ422 * w[3] * w[1] * w[1] * f422 +
                  detJ423 * w[3] * w[1] * w[2] * f423 +
                  detJ424 * w[3] * w[1] * w[3] * f424 +
                  detJ425 * w[3] * w[1] * w[4] * f425 +
                  detJ431 * w[3] * w[2] * w[0] * f431 +
                  detJ432 * w[3] * w[2] * w[1] * f432 +
                  detJ433 * w[3] * w[2] * w[2] * f433 +
                  detJ434 * w[3] * w[2] * w[3] * f434 +
                  detJ435 * w[3] * w[2] * w[4] * f435 +
                  detJ441 * w[3] * w[3] * w[0] * f441 +
                  detJ442 * w[3] * w[3] * w[1] * f442 +
                  detJ443 * w[3] * w[3] * w[2] * f443 +
                  detJ444 * w[3] * w[3] * w[3] * f444 + 
                  detJ445 * w[3] * w[3] * w[4] * f445 +
                  detJ451 * w[3] * w[4] * w[0] * f451 +
                  detJ452 * w[3] * w[4] * w[1] * f452 +
                  detJ453 * w[3] * w[4] * w[2] * f453 +
                  detJ454 * w[3] * w[4] * w[3] * f454 + 
                  detJ455 * w[3] * w[4] * w[4] * f455 +                  
                  detJ511 * w[4] * w[0] * w[0] * f511 +
                  detJ512 * w[4] * w[0] * w[1] * f512 +
                  detJ513 * w[4] * w[0] * w[2] * f513 +
                  detJ514 * w[4] * w[0] * w[3] * f514 +
                  detJ515 * w[4] * w[0] * w[4] * f515 +
                  detJ521 * w[4] * w[1] * w[0] * f521 +
                  detJ522 * w[4] * w[1] * w[1] * f522 +
                  detJ523 * w[4] * w[1] * w[2] * f523 +
                  detJ524 * w[4] * w[1] * w[3] * f524 +
                  detJ525 * w[4] * w[1] * w[4] * f525 +
                  detJ531 * w[4] * w[2] * w[0] * f531 +
                  detJ532 * w[4] * w[2] * w[1] * f532 +
                  detJ533 * w[4] * w[2] * w[2] * f533 +
                  detJ534 * w[4] * w[2] * w[3] * f534 +
                  detJ535 * w[4] * w[2] * w[4] * f535 +
                  detJ541 * w[4] * w[3] * w[0] * f541 +
                  detJ542 * w[4] * w[3] * w[1] * f542 +
                  detJ543 * w[4] * w[3] * w[2] * f543 +
                  detJ544 * w[4] * w[3] * w[3] * f544 + 
                  detJ545 * w[4] * w[3] * w[4] * f545 +
                  detJ551 * w[4] * w[4] * w[0] * f551 +
                  detJ552 * w[4] * w[4] * w[1] * f552 +
                  detJ553 * w[4] * w[4] * w[2] * f553 +
                  detJ554 * w[4] * w[4] * w[3] * f554 + 
                  detJ555 * w[4] * w[4] * w[4] * f555)
            
            
            # create the stress stiffness matrix            
            KS = (detJ111 * w[0] * w[0] * w[0] * HF111.T.dot(T).dot(HF111) +
                  detJ112 * w[0] * w[0] * w[1] * HF112.T.dot(T).dot(HF112) +
                  detJ113 * w[0] * w[0] * w[2] * HF113.T.dot(T).dot(HF113) +
                  detJ114 * w[0] * w[0] * w[3] * HF114.T.dot(T).dot(HF114) +
                  detJ115 * w[0] * w[0] * w[4] * HF115.T.dot(T).dot(HF115) +
                  detJ121 * w[0] * w[1] * w[0] * HF121.T.dot(T).dot(HF121) +
                  detJ122 * w[0] * w[1] * w[1] * HF122.T.dot(T).dot(HF122) +
                  detJ123 * w[0] * w[1] * w[2] * HF123.T.dot(T).dot(HF123) +
                  detJ124 * w[0] * w[1] * w[3] * HF124.T.dot(T).dot(HF124) +
                  detJ125 * w[0] * w[1] * w[4] * HF125.T.dot(T).dot(HF125) +
                  detJ131 * w[0] * w[2] * w[0] * HF131.T.dot(T).dot(HF131) +
                  detJ132 * w[0] * w[2] * w[1] * HF132.T.dot(T).dot(HF132) +
                  detJ133 * w[0] * w[2] * w[2] * HF133.T.dot(T).dot(HF133) +
                  detJ134 * w[0] * w[2] * w[3] * HF134.T.dot(T).dot(HF134) +
                  detJ135 * w[0] * w[2] * w[4] * HF135.T.dot(T).dot(HF135) +
                  detJ141 * w[0] * w[3] * w[0] * HF141.T.dot(T).dot(HF141) +
                  detJ142 * w[0] * w[3] * w[1] * HF142.T.dot(T).dot(HF142) +
                  detJ143 * w[0] * w[3] * w[2] * HF143.T.dot(T).dot(HF143) +
                  detJ144 * w[0] * w[3] * w[3] * HF144.T.dot(T).dot(HF144) + 
                  detJ145 * w[0] * w[3] * w[4] * HF145.T.dot(T).dot(HF145) +
                  detJ151 * w[0] * w[4] * w[0] * HF151.T.dot(T).dot(HF151) +
                  detJ152 * w[0] * w[4] * w[1] * HF152.T.dot(T).dot(HF152) +
                  detJ153 * w[0] * w[4] * w[2] * HF153.T.dot(T).dot(HF153) +
                  detJ154 * w[0] * w[4] * w[3] * HF154.T.dot(T).dot(HF154) + 
                  detJ155 * w[0] * w[4] * w[4] * HF155.T.dot(T).dot(HF155) +                  
                  detJ211 * w[1] * w[0] * w[0] * HF211.T.dot(T).dot(HF211) +
                  detJ212 * w[1] * w[0] * w[1] * HF212.T.dot(T).dot(HF212) +
                  detJ213 * w[1] * w[0] * w[2] * HF213.T.dot(T).dot(HF213) +
                  detJ214 * w[1] * w[0] * w[3] * HF214.T.dot(T).dot(HF214) +
                  detJ215 * w[1] * w[0] * w[4] * HF215.T.dot(T).dot(HF215) +
                  detJ221 * w[1] * w[1] * w[0] * HF221.T.dot(T).dot(HF221) +
                  detJ222 * w[1] * w[1] * w[1] * HF222.T.dot(T).dot(HF222) +
                  detJ223 * w[1] * w[1] * w[2] * HF223.T.dot(T).dot(HF223) +
                  detJ224 * w[1] * w[1] * w[3] * HF224.T.dot(T).dot(HF224) +
                  detJ225 * w[1] * w[1] * w[4] * HF225.T.dot(T).dot(HF225) +
                  detJ231 * w[1] * w[2] * w[0] * HF231.T.dot(T).dot(HF231) +
                  detJ232 * w[1] * w[2] * w[1] * HF232.T.dot(T).dot(HF232) +
                  detJ233 * w[1] * w[2] * w[2] * HF233.T.dot(T).dot(HF233) +
                  detJ234 * w[1] * w[2] * w[3] * HF234.T.dot(T).dot(HF234) +
                  detJ235 * w[1] * w[2] * w[4] * HF235.T.dot(T).dot(HF235) +
                  detJ241 * w[1] * w[3] * w[0] * HF241.T.dot(T).dot(HF241) +
                  detJ242 * w[1] * w[3] * w[1] * HF242.T.dot(T).dot(HF242) +
                  detJ243 * w[1] * w[3] * w[2] * HF243.T.dot(T).dot(HF243) +
                  detJ244 * w[1] * w[3] * w[3] * HF244.T.dot(T).dot(HF244) + 
                  detJ245 * w[1] * w[3] * w[4] * HF245.T.dot(T).dot(HF245) +
                  detJ251 * w[1] * w[4] * w[0] * HF251.T.dot(T).dot(HF251) +
                  detJ252 * w[1] * w[4] * w[1] * HF252.T.dot(T).dot(HF252) +
                  detJ253 * w[1] * w[4] * w[2] * HF253.T.dot(T).dot(HF253) +
                  detJ254 * w[1] * w[4] * w[3] * HF254.T.dot(T).dot(HF254) + 
                  detJ255 * w[1] * w[4] * w[4] * HF255.T.dot(T).dot(HF255) +                  
                  detJ311 * w[2] * w[0] * w[0] * HF311.T.dot(T).dot(HF311) +
                  detJ312 * w[2] * w[0] * w[1] * HF312.T.dot(T).dot(HF312) +
                  detJ313 * w[2] * w[0] * w[2] * HF313.T.dot(T).dot(HF313) +
                  detJ314 * w[2] * w[0] * w[3] * HF314.T.dot(T).dot(HF314) +
                  detJ315 * w[2] * w[0] * w[4] * HF315.T.dot(T).dot(HF315) +
                  detJ321 * w[2] * w[1] * w[0] * HF321.T.dot(T).dot(HF321) +
                  detJ322 * w[2] * w[1] * w[1] * HF322.T.dot(T).dot(HF322) +
                  detJ323 * w[2] * w[1] * w[2] * HF323.T.dot(T).dot(HF323) +
                  detJ324 * w[2] * w[1] * w[3] * HF324.T.dot(T).dot(HF324) +
                  detJ325 * w[2] * w[1] * w[4] * HF325.T.dot(T).dot(HF325) +
                  detJ331 * w[2] * w[2] * w[0] * HF331.T.dot(T).dot(HF331) +
                  detJ332 * w[2] * w[2] * w[1] * HF332.T.dot(T).dot(HF332) +
                  detJ333 * w[2] * w[2] * w[2] * HF333.T.dot(T).dot(HF333) +
                  detJ334 * w[2] * w[2] * w[3] * HF334.T.dot(T).dot(HF334) +
                  detJ335 * w[2] * w[2] * w[4] * HF335.T.dot(T).dot(HF335) +
                  detJ341 * w[2] * w[3] * w[0] * HF341.T.dot(T).dot(HF341) +
                  detJ342 * w[2] * w[3] * w[1] * HF342.T.dot(T).dot(HF342) +
                  detJ343 * w[2] * w[3] * w[2] * HF343.T.dot(T).dot(HF343) +
                  detJ344 * w[2] * w[3] * w[3] * HF344.T.dot(T).dot(HF344) + 
                  detJ345 * w[2] * w[3] * w[4] * HF345.T.dot(T).dot(HF345) +
                  detJ351 * w[2] * w[4] * w[0] * HF351.T.dot(T).dot(HF351) +
                  detJ352 * w[2] * w[4] * w[1] * HF352.T.dot(T).dot(HF352) +
                  detJ353 * w[2] * w[4] * w[2] * HF353.T.dot(T).dot(HF353) +
                  detJ354 * w[2] * w[4] * w[3] * HF354.T.dot(T).dot(HF354) + 
                  detJ355 * w[2] * w[4] * w[4] * HF355.T.dot(T).dot(HF355) +                  
                  detJ411 * w[3] * w[0] * w[0] * HF411.T.dot(T).dot(HF411) +
                  detJ412 * w[3] * w[0] * w[1] * HF412.T.dot(T).dot(HF412) +
                  detJ413 * w[3] * w[0] * w[2] * HF413.T.dot(T).dot(HF413) +
                  detJ414 * w[3] * w[0] * w[3] * HF414.T.dot(T).dot(HF414) +
                  detJ415 * w[3] * w[0] * w[4] * HF415.T.dot(T).dot(HF415) +
                  detJ421 * w[3] * w[1] * w[0] * HF421.T.dot(T).dot(HF421) +
                  detJ422 * w[3] * w[1] * w[1] * HF422.T.dot(T).dot(HF422) +
                  detJ423 * w[3] * w[1] * w[2] * HF423.T.dot(T).dot(HF423) +
                  detJ424 * w[3] * w[1] * w[3] * HF424.T.dot(T).dot(HF424) +
                  detJ425 * w[3] * w[1] * w[4] * HF425.T.dot(T).dot(HF425) +
                  detJ431 * w[3] * w[2] * w[0] * HF431.T.dot(T).dot(HF431) +
                  detJ432 * w[3] * w[2] * w[1] * HF432.T.dot(T).dot(HF432) +
                  detJ433 * w[3] * w[2] * w[2] * HF433.T.dot(T).dot(HF433) +
                  detJ434 * w[3] * w[2] * w[3] * HF434.T.dot(T).dot(HF434) +
                  detJ435 * w[3] * w[2] * w[4] * HF435.T.dot(T).dot(HF435) +
                  detJ441 * w[3] * w[3] * w[0] * HF441.T.dot(T).dot(HF441) +
                  detJ442 * w[3] * w[3] * w[1] * HF442.T.dot(T).dot(HF442) +
                  detJ443 * w[3] * w[3] * w[2] * HF443.T.dot(T).dot(HF443) +
                  detJ444 * w[3] * w[3] * w[3] * HF444.T.dot(T).dot(HF444) + 
                  detJ445 * w[3] * w[3] * w[4] * HF445.T.dot(T).dot(HF445) +
                  detJ451 * w[3] * w[4] * w[0] * HF451.T.dot(T).dot(HF451) +
                  detJ452 * w[3] * w[4] * w[1] * HF452.T.dot(T).dot(HF452) +
                  detJ453 * w[3] * w[4] * w[2] * HF453.T.dot(T).dot(HF453) +
                  detJ454 * w[3] * w[4] * w[3] * HF454.T.dot(T).dot(HF454) + 
                  detJ455 * w[3] * w[4] * w[4] * HF455.T.dot(T).dot(HF455) +                  
                  detJ511 * w[4] * w[0] * w[0] * HF511.T.dot(T).dot(HF511) +
                  detJ512 * w[4] * w[0] * w[1] * HF512.T.dot(T).dot(HF512) +
                  detJ513 * w[4] * w[0] * w[2] * HF513.T.dot(T).dot(HF513) +
                  detJ514 * w[4] * w[0] * w[3] * HF514.T.dot(T).dot(HF514) +
                  detJ515 * w[4] * w[0] * w[4] * HF515.T.dot(T).dot(HF515) +
                  detJ521 * w[4] * w[1] * w[0] * HF521.T.dot(T).dot(HF521) +
                  detJ522 * w[4] * w[1] * w[1] * HF522.T.dot(T).dot(HF522) +
                  detJ523 * w[4] * w[1] * w[2] * HF523.T.dot(T).dot(HF523) +
                  detJ524 * w[4] * w[1] * w[3] * HF524.T.dot(T).dot(HF524) +
                  detJ525 * w[4] * w[1] * w[4] * HF525.T.dot(T).dot(HF525) +
                  detJ531 * w[4] * w[2] * w[0] * HF531.T.dot(T).dot(HF531) +
                  detJ532 * w[4] * w[2] * w[1] * HF532.T.dot(T).dot(HF532) +
                  detJ533 * w[4] * w[2] * w[2] * HF533.T.dot(T).dot(HF533) +
                  detJ534 * w[4] * w[2] * w[3] * HF534.T.dot(T).dot(HF534) +
                  detJ535 * w[4] * w[2] * w[4] * HF535.T.dot(T).dot(HF535) +
                  detJ541 * w[4] * w[3] * w[0] * HF541.T.dot(T).dot(HF541) +
                  detJ542 * w[4] * w[3] * w[1] * HF542.T.dot(T).dot(HF542) +
                  detJ543 * w[4] * w[3] * w[2] * HF543.T.dot(T).dot(HF543) +
                  detJ544 * w[4] * w[3] * w[3] * HF544.T.dot(T).dot(HF544) + 
                  detJ545 * w[4] * w[3] * w[4] * HF545.T.dot(T).dot(HF545) +
                  detJ551 * w[4] * w[4] * w[0] * HF551.T.dot(T).dot(HF551) +
                  detJ552 * w[4] * w[4] * w[1] * HF552.T.dot(T).dot(HF552) +
                  detJ553 * w[4] * w[4] * w[2] * HF553.T.dot(T).dot(HF553) +
                  detJ554 * w[4] * w[4] * w[3] * HF554.T.dot(T).dot(HF554) + 
                  detJ555 * w[4] * w[4] * w[4] * HF555.T.dot(T).dot(HF555) +                  
                  detJ111 * w[0] * w[0] * w[0] * HS111.T.dot(T).dot(HS111) +
                  detJ112 * w[0] * w[0] * w[1] * HS112.T.dot(T).dot(HS112) +
                  detJ113 * w[0] * w[0] * w[2] * HS113.T.dot(T).dot(HS113) +
                  detJ114 * w[0] * w[0] * w[3] * HS114.T.dot(T).dot(HS114) +
                  detJ115 * w[0] * w[0] * w[4] * HS115.T.dot(T).dot(HS115) +
                  detJ121 * w[0] * w[1] * w[0] * HS121.T.dot(T).dot(HS121) +
                  detJ122 * w[0] * w[1] * w[1] * HS122.T.dot(T).dot(HS122) +
                  detJ123 * w[0] * w[1] * w[2] * HS123.T.dot(T).dot(HS123) +
                  detJ124 * w[0] * w[1] * w[3] * HS124.T.dot(T).dot(HS124) +
                  detJ125 * w[0] * w[1] * w[4] * HS125.T.dot(T).dot(HS125) +
                  detJ131 * w[0] * w[2] * w[0] * HS131.T.dot(T).dot(HS131) +
                  detJ132 * w[0] * w[2] * w[1] * HS132.T.dot(T).dot(HS132) +
                  detJ133 * w[0] * w[2] * w[2] * HS133.T.dot(T).dot(HS133) +
                  detJ134 * w[0] * w[2] * w[3] * HS134.T.dot(T).dot(HS134) +
                  detJ135 * w[0] * w[2] * w[4] * HS135.T.dot(T).dot(HS135) +
                  detJ141 * w[0] * w[3] * w[0] * HS141.T.dot(T).dot(HS141) +
                  detJ142 * w[0] * w[3] * w[1] * HS142.T.dot(T).dot(HS142) +
                  detJ143 * w[0] * w[3] * w[2] * HS143.T.dot(T).dot(HS143) +
                  detJ144 * w[0] * w[3] * w[3] * HS144.T.dot(T).dot(HS144) + 
                  detJ145 * w[0] * w[3] * w[4] * HS145.T.dot(T).dot(HS145) +
                  detJ151 * w[0] * w[4] * w[0] * HS151.T.dot(T).dot(HS151) +
                  detJ152 * w[0] * w[4] * w[1] * HS152.T.dot(T).dot(HS152) +
                  detJ153 * w[0] * w[4] * w[2] * HS153.T.dot(T).dot(HS153) +
                  detJ154 * w[0] * w[4] * w[3] * HS154.T.dot(T).dot(HS154) + 
                  detJ155 * w[0] * w[4] * w[4] * HS155.T.dot(T).dot(HS155) +                  
                  detJ211 * w[1] * w[0] * w[0] * HS211.T.dot(T).dot(HS211) +
                  detJ212 * w[1] * w[0] * w[1] * HS212.T.dot(T).dot(HS212) +
                  detJ213 * w[1] * w[0] * w[2] * HS213.T.dot(T).dot(HS213) +
                  detJ214 * w[1] * w[0] * w[3] * HS214.T.dot(T).dot(HS214) +
                  detJ215 * w[1] * w[0] * w[4] * HS215.T.dot(T).dot(HS215) +
                  detJ221 * w[1] * w[1] * w[0] * HS221.T.dot(T).dot(HS221) +
                  detJ222 * w[1] * w[1] * w[1] * HS222.T.dot(T).dot(HS222) +
                  detJ223 * w[1] * w[1] * w[2] * HS223.T.dot(T).dot(HS223) +
                  detJ224 * w[1] * w[1] * w[3] * HS224.T.dot(T).dot(HS224) +
                  detJ225 * w[1] * w[1] * w[4] * HS225.T.dot(T).dot(HS225) +
                  detJ231 * w[1] * w[2] * w[0] * HS231.T.dot(T).dot(HS231) +
                  detJ232 * w[1] * w[2] * w[1] * HS232.T.dot(T).dot(HS232) +
                  detJ233 * w[1] * w[2] * w[2] * HS233.T.dot(T).dot(HS233) +
                  detJ234 * w[1] * w[2] * w[3] * HS234.T.dot(T).dot(HS234) +
                  detJ235 * w[1] * w[2] * w[4] * HS235.T.dot(T).dot(HS235) +
                  detJ241 * w[1] * w[3] * w[0] * HS241.T.dot(T).dot(HS241) +
                  detJ242 * w[1] * w[3] * w[1] * HS242.T.dot(T).dot(HS242) +
                  detJ243 * w[1] * w[3] * w[2] * HS243.T.dot(T).dot(HS243) +
                  detJ244 * w[1] * w[3] * w[3] * HS244.T.dot(T).dot(HS244) + 
                  detJ245 * w[1] * w[3] * w[4] * HS245.T.dot(T).dot(HS245) +
                  detJ251 * w[1] * w[4] * w[0] * HS251.T.dot(T).dot(HS251) +
                  detJ252 * w[1] * w[4] * w[1] * HS252.T.dot(T).dot(HS252) +
                  detJ253 * w[1] * w[4] * w[2] * HS253.T.dot(T).dot(HS253) +
                  detJ254 * w[1] * w[4] * w[3] * HS254.T.dot(T).dot(HS254) + 
                  detJ255 * w[1] * w[4] * w[4] * HS255.T.dot(T).dot(HS255) +                  
                  detJ311 * w[2] * w[0] * w[0] * HS311.T.dot(T).dot(HS311) +
                  detJ312 * w[2] * w[0] * w[1] * HS312.T.dot(T).dot(HS312) +
                  detJ313 * w[2] * w[0] * w[2] * HS313.T.dot(T).dot(HS313) +
                  detJ314 * w[2] * w[0] * w[3] * HS314.T.dot(T).dot(HS314) +
                  detJ315 * w[2] * w[0] * w[4] * HS315.T.dot(T).dot(HS315) +
                  detJ321 * w[2] * w[1] * w[0] * HS321.T.dot(T).dot(HS321) +
                  detJ322 * w[2] * w[1] * w[1] * HS322.T.dot(T).dot(HS322) +
                  detJ323 * w[2] * w[1] * w[2] * HS323.T.dot(T).dot(HS323) +
                  detJ324 * w[2] * w[1] * w[3] * HS324.T.dot(T).dot(HS324) +
                  detJ325 * w[2] * w[1] * w[4] * HS325.T.dot(T).dot(HS325) +
                  detJ331 * w[2] * w[2] * w[0] * HS331.T.dot(T).dot(HS331) +
                  detJ332 * w[2] * w[2] * w[1] * HS332.T.dot(T).dot(HS332) +
                  detJ333 * w[2] * w[2] * w[2] * HS333.T.dot(T).dot(HS333) +
                  detJ334 * w[2] * w[2] * w[3] * HS334.T.dot(T).dot(HS334) +
                  detJ335 * w[2] * w[2] * w[4] * HS335.T.dot(T).dot(HS335) +
                  detJ341 * w[2] * w[3] * w[0] * HS341.T.dot(T).dot(HS341) +
                  detJ342 * w[2] * w[3] * w[1] * HS342.T.dot(T).dot(HS342) +
                  detJ343 * w[2] * w[3] * w[2] * HS343.T.dot(T).dot(HS343) +
                  detJ344 * w[2] * w[3] * w[3] * HS344.T.dot(T).dot(HS344) + 
                  detJ345 * w[2] * w[3] * w[4] * HS345.T.dot(T).dot(HS345) +
                  detJ351 * w[2] * w[4] * w[0] * HS351.T.dot(T).dot(HS351) +
                  detJ352 * w[2] * w[4] * w[1] * HS352.T.dot(T).dot(HS352) +
                  detJ353 * w[2] * w[4] * w[2] * HS353.T.dot(T).dot(HS353) +
                  detJ354 * w[2] * w[4] * w[3] * HS354.T.dot(T).dot(HS354) + 
                  detJ355 * w[2] * w[4] * w[4] * HS355.T.dot(T).dot(HS355) +                  
                  detJ411 * w[3] * w[0] * w[0] * HS411.T.dot(T).dot(HS411) +
                  detJ412 * w[3] * w[0] * w[1] * HS412.T.dot(T).dot(HS412) +
                  detJ413 * w[3] * w[0] * w[2] * HS413.T.dot(T).dot(HS413) +
                  detJ414 * w[3] * w[0] * w[3] * HS414.T.dot(T).dot(HS414) +
                  detJ415 * w[3] * w[0] * w[4] * HS415.T.dot(T).dot(HS415) +
                  detJ421 * w[3] * w[1] * w[0] * HS421.T.dot(T).dot(HS421) +
                  detJ422 * w[3] * w[1] * w[1] * HS422.T.dot(T).dot(HS422) +
                  detJ423 * w[3] * w[1] * w[2] * HS423.T.dot(T).dot(HS423) +
                  detJ424 * w[3] * w[1] * w[3] * HS424.T.dot(T).dot(HS424) +
                  detJ425 * w[3] * w[1] * w[4] * HS425.T.dot(T).dot(HS425) +
                  detJ431 * w[3] * w[2] * w[0] * HS431.T.dot(T).dot(HS431) +
                  detJ432 * w[3] * w[2] * w[1] * HS432.T.dot(T).dot(HS432) +
                  detJ433 * w[3] * w[2] * w[2] * HS433.T.dot(T).dot(HS433) +
                  detJ434 * w[3] * w[2] * w[3] * HS434.T.dot(T).dot(HS434) +
                  detJ435 * w[3] * w[2] * w[4] * HS435.T.dot(T).dot(HS435) +
                  detJ441 * w[3] * w[3] * w[0] * HS441.T.dot(T).dot(HS441) +
                  detJ442 * w[3] * w[3] * w[1] * HS442.T.dot(T).dot(HS442) +
                  detJ443 * w[3] * w[3] * w[2] * HS443.T.dot(T).dot(HS443) +
                  detJ444 * w[3] * w[3] * w[3] * HS444.T.dot(T).dot(HS444) + 
                  detJ445 * w[3] * w[3] * w[4] * HS445.T.dot(T).dot(HS445) +
                  detJ451 * w[3] * w[4] * w[0] * HS451.T.dot(T).dot(HS451) +
                  detJ452 * w[3] * w[4] * w[1] * HS452.T.dot(T).dot(HS452) +
                  detJ453 * w[3] * w[4] * w[2] * HS453.T.dot(T).dot(HS453) +
                  detJ454 * w[3] * w[4] * w[3] * HS454.T.dot(T).dot(HS454) + 
                  detJ455 * w[3] * w[4] * w[4] * HS455.T.dot(T).dot(HS455) +                  
                  detJ511 * w[4] * w[0] * w[0] * HS511.T.dot(T).dot(HS511) +
                  detJ512 * w[4] * w[0] * w[1] * HS512.T.dot(T).dot(HS512) +
                  detJ513 * w[4] * w[0] * w[2] * HS513.T.dot(T).dot(HS513) +
                  detJ514 * w[4] * w[0] * w[3] * HS514.T.dot(T).dot(HS514) +
                  detJ515 * w[4] * w[0] * w[4] * HS515.T.dot(T).dot(HS515) +
                  detJ521 * w[4] * w[1] * w[0] * HS521.T.dot(T).dot(HS521) +
                  detJ522 * w[4] * w[1] * w[1] * HS522.T.dot(T).dot(HS522) +
                  detJ523 * w[4] * w[1] * w[2] * HS523.T.dot(T).dot(HS523) +
                  detJ524 * w[4] * w[1] * w[3] * HS524.T.dot(T).dot(HS524) +
                  detJ525 * w[4] * w[1] * w[4] * HS525.T.dot(T).dot(HS525) +
                  detJ531 * w[4] * w[2] * w[0] * HS531.T.dot(T).dot(HS531) +
                  detJ532 * w[4] * w[2] * w[1] * HS532.T.dot(T).dot(HS532) +
                  detJ533 * w[4] * w[2] * w[2] * HS533.T.dot(T).dot(HS533) +
                  detJ534 * w[4] * w[2] * w[3] * HS534.T.dot(T).dot(HS534) +
                  detJ535 * w[4] * w[2] * w[4] * HS535.T.dot(T).dot(HS535) +
                  detJ541 * w[4] * w[3] * w[0] * HS541.T.dot(T).dot(HS541) +
                  detJ542 * w[4] * w[3] * w[1] * HS542.T.dot(T).dot(HS542) +
                  detJ543 * w[4] * w[3] * w[2] * HS543.T.dot(T).dot(HS543) +
                  detJ544 * w[4] * w[3] * w[3] * HS544.T.dot(T).dot(HS544) + 
                  detJ545 * w[4] * w[3] * w[4] * HS545.T.dot(T).dot(HS545) +
                  detJ551 * w[4] * w[4] * w[0] * HS551.T.dot(T).dot(HS551) +
                  detJ552 * w[4] * w[4] * w[1] * HS552.T.dot(T).dot(HS552) +
                  detJ553 * w[4] * w[4] * w[2] * HS553.T.dot(T).dot(HS553) +
                  detJ554 * w[4] * w[4] * w[3] * HS554.T.dot(T).dot(HS554) + 
                  detJ555 * w[4] * w[4] * w[4] * HS555.T.dot(T).dot(HS555) +                  
                  detJ111 * w[0] * w[0] * w[0] * HT111.T.dot(T).dot(HT111) +
                  detJ112 * w[0] * w[0] * w[1] * HT112.T.dot(T).dot(HT112) +
                  detJ113 * w[0] * w[0] * w[2] * HT113.T.dot(T).dot(HT113) +
                  detJ114 * w[0] * w[0] * w[3] * HT114.T.dot(T).dot(HT114) +
                  detJ115 * w[0] * w[0] * w[4] * HT115.T.dot(T).dot(HT115) +
                  detJ121 * w[0] * w[1] * w[0] * HT121.T.dot(T).dot(HT121) +
                  detJ122 * w[0] * w[1] * w[1] * HT122.T.dot(T).dot(HT122) +
                  detJ123 * w[0] * w[1] * w[2] * HT123.T.dot(T).dot(HT123) +
                  detJ124 * w[0] * w[1] * w[3] * HT124.T.dot(T).dot(HT124) +
                  detJ125 * w[0] * w[1] * w[4] * HT125.T.dot(T).dot(HT125) +
                  detJ131 * w[0] * w[2] * w[0] * HT131.T.dot(T).dot(HT131) +
                  detJ132 * w[0] * w[2] * w[1] * HT132.T.dot(T).dot(HT132) +
                  detJ133 * w[0] * w[2] * w[2] * HT133.T.dot(T).dot(HT133) +
                  detJ134 * w[0] * w[2] * w[3] * HT134.T.dot(T).dot(HT134) +
                  detJ135 * w[0] * w[2] * w[4] * HT135.T.dot(T).dot(HT135) +
                  detJ141 * w[0] * w[3] * w[0] * HT141.T.dot(T).dot(HT141) +
                  detJ142 * w[0] * w[3] * w[1] * HT142.T.dot(T).dot(HT142) +
                  detJ143 * w[0] * w[3] * w[2] * HT143.T.dot(T).dot(HT143) +
                  detJ144 * w[0] * w[3] * w[3] * HT144.T.dot(T).dot(HT144) + 
                  detJ145 * w[0] * w[3] * w[4] * HT145.T.dot(T).dot(HT145) +
                  detJ151 * w[0] * w[4] * w[0] * HT151.T.dot(T).dot(HT151) +
                  detJ152 * w[0] * w[4] * w[1] * HT152.T.dot(T).dot(HT152) +
                  detJ153 * w[0] * w[4] * w[2] * HT153.T.dot(T).dot(HT153) +
                  detJ154 * w[0] * w[4] * w[3] * HT154.T.dot(T).dot(HT154) + 
                  detJ155 * w[0] * w[4] * w[4] * HT155.T.dot(T).dot(HT155) +                  
                  detJ211 * w[1] * w[0] * w[0] * HT211.T.dot(T).dot(HT211) +
                  detJ212 * w[1] * w[0] * w[1] * HT212.T.dot(T).dot(HT212) +
                  detJ213 * w[1] * w[0] * w[2] * HT213.T.dot(T).dot(HT213) +
                  detJ214 * w[1] * w[0] * w[3] * HT214.T.dot(T).dot(HT214) +
                  detJ215 * w[1] * w[0] * w[4] * HT215.T.dot(T).dot(HT215) +
                  detJ221 * w[1] * w[1] * w[0] * HT221.T.dot(T).dot(HT221) +
                  detJ222 * w[1] * w[1] * w[1] * HT222.T.dot(T).dot(HT222) +
                  detJ223 * w[1] * w[1] * w[2] * HT223.T.dot(T).dot(HT223) +
                  detJ224 * w[1] * w[1] * w[3] * HT224.T.dot(T).dot(HT224) +
                  detJ225 * w[1] * w[1] * w[4] * HT225.T.dot(T).dot(HT225) +
                  detJ231 * w[1] * w[2] * w[0] * HT231.T.dot(T).dot(HT231) +
                  detJ232 * w[1] * w[2] * w[1] * HT232.T.dot(T).dot(HT232) +
                  detJ233 * w[1] * w[2] * w[2] * HT233.T.dot(T).dot(HT233) +
                  detJ234 * w[1] * w[2] * w[3] * HT234.T.dot(T).dot(HT234) +
                  detJ235 * w[1] * w[2] * w[4] * HT235.T.dot(T).dot(HT235) +
                  detJ241 * w[1] * w[3] * w[0] * HT241.T.dot(T).dot(HT241) +
                  detJ242 * w[1] * w[3] * w[1] * HT242.T.dot(T).dot(HT242) +
                  detJ243 * w[1] * w[3] * w[2] * HT243.T.dot(T).dot(HT243) +
                  detJ244 * w[1] * w[3] * w[3] * HT244.T.dot(T).dot(HT244) + 
                  detJ245 * w[1] * w[3] * w[4] * HT245.T.dot(T).dot(HT245) +
                  detJ251 * w[1] * w[4] * w[0] * HT251.T.dot(T).dot(HT251) +
                  detJ252 * w[1] * w[4] * w[1] * HT252.T.dot(T).dot(HT252) +
                  detJ253 * w[1] * w[4] * w[2] * HT253.T.dot(T).dot(HT253) +
                  detJ254 * w[1] * w[4] * w[3] * HT254.T.dot(T).dot(HT254) + 
                  detJ255 * w[1] * w[4] * w[4] * HT255.T.dot(T).dot(HT255) +                  
                  detJ311 * w[2] * w[0] * w[0] * HT311.T.dot(T).dot(HT311) +
                  detJ312 * w[2] * w[0] * w[1] * HT312.T.dot(T).dot(HT312) +
                  detJ313 * w[2] * w[0] * w[2] * HT313.T.dot(T).dot(HT313) +
                  detJ314 * w[2] * w[0] * w[3] * HT314.T.dot(T).dot(HT314) +
                  detJ315 * w[2] * w[0] * w[4] * HT315.T.dot(T).dot(HT315) +
                  detJ321 * w[2] * w[1] * w[0] * HT321.T.dot(T).dot(HT321) +
                  detJ322 * w[2] * w[1] * w[1] * HT322.T.dot(T).dot(HT322) +
                  detJ323 * w[2] * w[1] * w[2] * HT323.T.dot(T).dot(HT323) +
                  detJ324 * w[2] * w[1] * w[3] * HT324.T.dot(T).dot(HT324) +
                  detJ325 * w[2] * w[1] * w[4] * HT325.T.dot(T).dot(HT325) +
                  detJ331 * w[2] * w[2] * w[0] * HT331.T.dot(T).dot(HT331) +
                  detJ332 * w[2] * w[2] * w[1] * HT332.T.dot(T).dot(HT332) +
                  detJ333 * w[2] * w[2] * w[2] * HT333.T.dot(T).dot(HT333) +
                  detJ334 * w[2] * w[2] * w[3] * HT334.T.dot(T).dot(HT334) +
                  detJ335 * w[2] * w[2] * w[4] * HT335.T.dot(T).dot(HT335) +
                  detJ341 * w[2] * w[3] * w[0] * HT341.T.dot(T).dot(HT341) +
                  detJ342 * w[2] * w[3] * w[1] * HT342.T.dot(T).dot(HT342) +
                  detJ343 * w[2] * w[3] * w[2] * HT343.T.dot(T).dot(HT343) +
                  detJ344 * w[2] * w[3] * w[3] * HT344.T.dot(T).dot(HT344) + 
                  detJ345 * w[2] * w[3] * w[4] * HT345.T.dot(T).dot(HT345) +
                  detJ351 * w[2] * w[4] * w[0] * HT351.T.dot(T).dot(HT351) +
                  detJ352 * w[2] * w[4] * w[1] * HT352.T.dot(T).dot(HT352) +
                  detJ353 * w[2] * w[4] * w[2] * HT353.T.dot(T).dot(HT353) +
                  detJ354 * w[2] * w[4] * w[3] * HT354.T.dot(T).dot(HT354) + 
                  detJ355 * w[2] * w[4] * w[4] * HT355.T.dot(T).dot(HT355) +                  
                  detJ411 * w[3] * w[0] * w[0] * HT411.T.dot(T).dot(HT411) +
                  detJ412 * w[3] * w[0] * w[1] * HT412.T.dot(T).dot(HT412) +
                  detJ413 * w[3] * w[0] * w[2] * HT413.T.dot(T).dot(HT413) +
                  detJ414 * w[3] * w[0] * w[3] * HT414.T.dot(T).dot(HT414) +
                  detJ415 * w[3] * w[0] * w[4] * HT415.T.dot(T).dot(HT415) +
                  detJ421 * w[3] * w[1] * w[0] * HT421.T.dot(T).dot(HT421) +
                  detJ422 * w[3] * w[1] * w[1] * HT422.T.dot(T).dot(HT422) +
                  detJ423 * w[3] * w[1] * w[2] * HT423.T.dot(T).dot(HT423) +
                  detJ424 * w[3] * w[1] * w[3] * HT424.T.dot(T).dot(HT424) +
                  detJ425 * w[3] * w[1] * w[4] * HT425.T.dot(T).dot(HT425) +
                  detJ431 * w[3] * w[2] * w[0] * HT431.T.dot(T).dot(HT431) +
                  detJ432 * w[3] * w[2] * w[1] * HT432.T.dot(T).dot(HT432) +
                  detJ433 * w[3] * w[2] * w[2] * HT433.T.dot(T).dot(HT433) +
                  detJ434 * w[3] * w[2] * w[3] * HT434.T.dot(T).dot(HT434) +
                  detJ435 * w[3] * w[2] * w[4] * HT435.T.dot(T).dot(HT435) +
                  detJ441 * w[3] * w[3] * w[0] * HT441.T.dot(T).dot(HT441) +
                  detJ442 * w[3] * w[3] * w[1] * HT442.T.dot(T).dot(HT442) +
                  detJ443 * w[3] * w[3] * w[2] * HT443.T.dot(T).dot(HT443) +
                  detJ444 * w[3] * w[3] * w[3] * HT444.T.dot(T).dot(HT444) + 
                  detJ445 * w[3] * w[3] * w[4] * HT445.T.dot(T).dot(HT445) +
                  detJ451 * w[3] * w[4] * w[0] * HT451.T.dot(T).dot(HT451) +
                  detJ452 * w[3] * w[4] * w[1] * HT452.T.dot(T).dot(HT452) +
                  detJ453 * w[3] * w[4] * w[2] * HT453.T.dot(T).dot(HT453) +
                  detJ454 * w[3] * w[4] * w[3] * HT454.T.dot(T).dot(HT454) + 
                  detJ455 * w[3] * w[4] * w[4] * HT455.T.dot(T).dot(HT455) +                  
                  detJ511 * w[4] * w[0] * w[0] * HT511.T.dot(T).dot(HT511) +
                  detJ512 * w[4] * w[0] * w[1] * HT512.T.dot(T).dot(HT512) +
                  detJ513 * w[4] * w[0] * w[2] * HT513.T.dot(T).dot(HT513) +
                  detJ514 * w[4] * w[0] * w[3] * HT514.T.dot(T).dot(HT514) +
                  detJ515 * w[4] * w[0] * w[4] * HT515.T.dot(T).dot(HT515) +
                  detJ521 * w[4] * w[1] * w[0] * HT521.T.dot(T).dot(HT521) +
                  detJ522 * w[4] * w[1] * w[1] * HT522.T.dot(T).dot(HT522) +
                  detJ523 * w[4] * w[1] * w[2] * HT523.T.dot(T).dot(HT523) +
                  detJ524 * w[4] * w[1] * w[3] * HT524.T.dot(T).dot(HT524) +
                  detJ525 * w[4] * w[1] * w[4] * HT525.T.dot(T).dot(HT525) +
                  detJ531 * w[4] * w[2] * w[0] * HT531.T.dot(T).dot(HT531) +
                  detJ532 * w[4] * w[2] * w[1] * HT532.T.dot(T).dot(HT532) +
                  detJ533 * w[4] * w[2] * w[2] * HT533.T.dot(T).dot(HT533) +
                  detJ534 * w[4] * w[2] * w[3] * HT534.T.dot(T).dot(HT534) +
                  detJ535 * w[4] * w[2] * w[4] * HT535.T.dot(T).dot(HT535) +
                  detJ541 * w[4] * w[3] * w[0] * HT541.T.dot(T).dot(HT541) +
                  detJ542 * w[4] * w[3] * w[1] * HT542.T.dot(T).dot(HT542) +
                  detJ543 * w[4] * w[3] * w[2] * HT543.T.dot(T).dot(HT543) +
                  detJ544 * w[4] * w[3] * w[3] * HT544.T.dot(T).dot(HT544) + 
                  detJ545 * w[4] * w[3] * w[4] * HT545.T.dot(T).dot(HT545) +
                  detJ551 * w[4] * w[4] * w[0] * HT551.T.dot(T).dot(HT551) +
                  detJ552 * w[4] * w[4] * w[1] * HT552.T.dot(T).dot(HT552) +
                  detJ553 * w[4] * w[4] * w[2] * HT553.T.dot(T).dot(HT553) +
                  detJ554 * w[4] * w[4] * w[3] * HT554.T.dot(T).dot(HT554) + 
                  detJ555 * w[4] * w[4] * w[4] * HT555.T.dot(T).dot(HT555))   
            
            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS                        
        return stiffT



