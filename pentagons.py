#!/usr/bin/env python3       
# -*- coding: utf-8 -*-     

from chords import chord
import materialProperties as mp
import math as m
from membranes import membrane
import numpy as np
from ridder import findRoot
from shapeFnPentagons import shapeFunction
import spin as spinMtx
from numpy.linalg import inv
from numpy.linalg import det


"""
Module pentagons.py provides geometric information about irregular pentagons.

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
__version__ = "1.3.1"
__date__ = "08-08-2019"
__update__ = "10-06-2019"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

r"""

Change in version "1.3.0":

methods

    massM = p.massMatrix(rho, width)
        gaussPt  the Gauss point for which the mass matrix is to be supplied
        rho      the mass density with units of mass per unit volume
        width    the membrane thickness
    returns
        massM    a 10x10 mass matrix for the pentagon associated with 'gaussPt'

    stiffM = p.stiffnessMatrix()

    fFn = p.forcingFunction()

Overview of module pentagons.py:

Class pentagon in file pentagons.py allows for the creation of objects that are
to be used to represent irregular pentagons comprised of five connected chords.
The pentagon is regular whenever the dodecahedron is regular, otherwise it need
not be.  Each pentagon is assigned an unique number.

Initial coordinates that locate a vertex in a dodecahedron used to model the
alveoli of lung are assigned according to a reference configuration where the
pleural pressure (the pressure surrounding lung in the pleural cavity) and the
transpulmonary pressure (the difference between aleolar and pleural pressures)
are both at zero gauge pressure, i.e., all pressures are atmospheric pressure.
The pleural pressure is normally negative, sucking the pleural membrane against
the wall of the chest.  During expiration, the diaphragm is pushed up, reducing
the pleural pressure.  The pleural pressure remains negative during breating at
rest, but it can become positive during active expiration.  The surface tension
created by surfactant keeps most alveoli open during excursions into positive
pleural pressures, but not all will remain open.  Alveoli are their smallest at
max expiration.  Alveolar size is determined by the transpulmonary pressure.
The greater the transpulmonary pressure the greater the alveolar size will be.

Numerous methods have a string argument that is denoted as  state  which can
take on any of the following values:
    'c', 'curr', 'current'       gets the value for a current configuration
    'n', 'next'                  gets the value for a next configuration
    'p', 'prev', 'previous'      gets the value for a previous configuration
    'r', 'ref', 'reference'      gets the value for the reference configuration

class pentagon

constructor

    p = pentagon(number, chord1, chord2, chord3, chord4, chord5, h, gaussPts)
        number    immutable value that is unique to this pentagon
        chord1    unique edge of the pentagon, an instance of class chord
        chord2    unique edge of the pentagon, an instance of class chord
        chord3    unique edge of the pentagon, an instance of class chord
        chord4    unique edge of the pentagon, an instance of class chord
        chord5    unique edge of the pentagon, an instance of class chord
        h         timestep size between two successive calls to 'advance'
        gaussPts  number of Gauss points to be used: must be 1, 4 or 7

        when assigning these five chords, do so according to the following
        graphic when looking inward from the outside of the dodecahedron:

                            v1
                          /   \
                       c2       c1
                     /             \
                  v2                 v5
                   \                 /
                    c3              c5
                     \              /
                      v3 -- c4 -- v4

        by numbering the chords in a counterclockwise direction, the algorithm
        used to compute its area will be positive; otherwise, if the chords
        were numbered clockwise the derived area would be negative

methods

    s = p.toString()
        returns string representation for the pentagon in configuration 'state'

    n = p.number()
        returns the unique number affiated with this pentagon

    n1, n2, n3, n4, n5 = p.chordNumbers()
        returns the unique numbers associated with the chords of a pentagon

    n1, n2, n3, n4, n5 = p.vertexNumbers()
        returns the unique numbers associated with the vertices of a pentagon

    truth = p.hasChord(number)
        returns True if one of the five chords has this chord number

    truth = p.hasVertex(number)
        returns True if one of the five vertices has this vertex number

    c = p.getChord(number)
        returns a chord; typically called from within a p.hasChord if clause

    v = p.getVertex(number)
        returns a vertex; typically called from within a p.hasVertex if clause

    n = p.gaussPoints()
        returns the number of Gauss points assigned to the pentagon

    p.update()
        assigns new coordinate values to the pentagon for its next location
        and updates all effected fields.  To be called after all vertices have
        had their coordinates updated, and after all chords have been updated,
        too.  This may be called multiple times before freezing it with advance

    p.advance()
        assigns the current location into the previous location, and then it
        assigns the next location into the current location, thereby freezing
        the location of the present next-location in preparation to advance to
        the next step along a solution path

    Material properties that associate with this septum.  Except for the mass
    density, all are drawn randomly from a statistical distribution.

    rho = p.massDensity()
        returns the mass density of the chord (collagen and elastin fibers)

    w = p.width(state)
        returns the cross-sectional thickness or width of the membrane in
        configuration 'state'

    M1, M2, Me_t, N1, N2, Ne_t, G1, G2, Ge_t = c.matProp()
        returns the constitutive properties for this septal membrane where
            M1, M2, Me_t  pertain to the dilation response
            N1, N2, Ne_t  pertain to the squeeze  response
            G1, G2, Ge_t  pertain to the  shear   response
        where the first in these sets describes the compliant response
        the second in these sets describes the stiff response, and
        the third in these sets establishes the strain of transition

    Geometric fields associated with a pentagonal surface in 3 space

    a = p.area(state)
        returns the area of this irregular pentagon in configuration 'state'

    aLambda = p.arealStretch(state)
            returns the square root of area(state) divided by reference area

    aStrain = p.arealStrain(state)
            returns the logarithm of areal stretch evaluated at 'state'

    daStrain = p.dArealStrain(state)
            returns the time rate of change in areal strain at 'state'

    [nx, ny, nz] = p.normal(state)
        returns the unit normal to this pentagon in configuration 'state'

    [dnx, dny, dnz] = p.dNormal(state)
        returns the time rate of change of the unit normal at 'state'

    Kinematic fields associated with the centroid of a pentagon in 3 space

    [cx, cy, cz] = p.centroid(state)
        returns centroid of this irregular pentagon in configuration 'state'

    [ux, uy, uz] = p.displacement(state)
        returns the displacement at the centroid in configuration 'state'

    [vx, vy, vz] = p.velocity(state)
        returns the velocity at the centroid in configuration 'state'

    [ax, ay, az] = p.acceleration(state)
        returns the acceleration at the centroid in configuration 'state'

    pMtx = p.rotation(state)
        returns a 3x3 orthogonal matrix that rotates the reference base vectors
        of the dodecahedron into a set of local base vectors pertaining to an
        irregular pentagon whose normal aligns with the 3 direction, i.e.,
        the irregular pentagon resides in the local 1-2 plane.  The returned
        matrix associates with configuration 'state'

    omegaMtx = p.spin(state)
        returns a 3x3 skew symmetric matrix that describes the time rate of
        rotation, i.e., spin, of the local pentagonal coordinate system about
        the fixed dodecahedral coordinate system with reference base vectors.
        The returned matrix associates with configuration 'state'

    The fundamental fields of kinematics

    gMtx = p.G(gaussPt, state)
        returns 2x2 matrix describing the displacement gradient (it is not
        relabeled) of the pentagon at 'gaussPt' in configuration 'state'

    fMtx = p.F(gaussPt, state)
        returns 2x2 matrix describing the deformation gradient (it is not
        relabeled) of the pentagon at 'gaussPt' in configuration 'state'

    lMtx = c.L(gaussPt, state)
        returns the velocity gradient at the specified Gauss point for the
        specified configuration

    Reindexing of the coordinate used in the membrane analysis at a Gauss point

    qMtx = p.Q(gaussPt, state)
        returns 2x2 reindexing matrix applied to the deformation gradient
        prior to its Gram-Schmidt decomposition at 'gaussPt' in
        configuration 'state'

    Gram-Schmidt factorization of a reindexed deformation gradient

    rMtx = p.R(gaussPt, state)
        returns 2x2 rotation matrix 'Q' derived from a QR decomposition of the
        reindexed deformation gradient at 'gaussPt' in configuration 'state'

    omega = p.Omega(gaussPt, state)
        returns 2x2 spin matrix caused by planar deformation, i.e., dR R^t,
        at 'gaussPt' in configuration 'state'

    uMtx = p.U(gaussPt, state)
        returns 2x2 Laplace stretch 'R' derived from a QR decomposition of the
        reindexed deformation gradient at 'gaussPt' in configuration 'state'

    uInvMtx = p.UInv(gaussPt, state)
        returns 2x2 inverse Laplace stretch derived from a QR decomposition of
        reindexed deformation gradient at 'gaussPt' in configuration 'state'

    duMtx = p.dU(gaussPt, state)
        returns 2x2 matrix for differential change in Laplace stretch at
        'gaussPt' in configuration 'state'

    duInvMtx = p.dUInv(gaussPt, state)
        returns 2x2 matrix for differential change in the inverse Laplace
        stretch at 'gaussPt' in configuration 'state'

    The extensive thermodynamic variables for a membrane and their rates
    accuired from a reindexed deformation gradient of the membrane

    xi = p.dilation(gaussPt, state)
        returns the planar dilation derived from a QR decomposition of the
        reindexed deformation gradient in configuration 'state'

    epsilon = p.squeeze(gaussPt, state)
        returns the planar squeeze derived from a QR decomposition of the
        reindexed deformation gradient at 'gaussPt' in configuration 'state'

    gamma = p.shear(gaussPt, state)
        returns the planar shear derived from a QR decomposition of the
        reindexed deformation gradient at 'gaussPt' in configuration 'state'

    dXi = p.dDilation(gaussPt, state)
        returns the differential change in dilation acquired from the reindexed
        deformation gradient for the membrane model at 'gaussPt' in
        configuration 'state'

    dEpsilon = p.dSqueeze(gaussPt, state)
        returns the differential change in squeeze acquired from the reindexed
        deformation gradient for the membrane model at 'gaussPt' in
        configuration 'state'

    dGamma = p.dShear(gaussPt, state)
        returns the differential change in shear acquired from the reindexed
        deformation gradient for the membrane model at 'gaussPt' in
        configuration 'state'

    Fields needed to construct finite element representations

    sf = p.shapeFunction(gaussPt):
        returns the shape function associated with the specified Gauss point.

    mMtx = p.massMatrix(rho, width)
        returns an average of the lumped and consistent mass matrices (ensures
        the mass matrix is not singular) for the chosen number of Gauss points
        for a pentagon whose mass density, rho, and whose thickness, width, are
        specified.

    kMtx = p.stiffnessMatrix()
        returns a tangent stiffness matrix for the chosen number of Gauss
        points.

    fFn = p.forcingFunction()
        returns a vector for the forcing function on the right hand side.

References
    1) Freed, A. D., Erel, V., and Moreno, M. R. "Conjugate Stress/Strain Base
       Pairs for the Analysis of Planar Biologic Tissues", Journal of Mechanics
       of Materials and Structures, 12 (2017), 219-247.
    2) Freed, A. D., and Zamani, S.: “On the Use of Convected Coordinate
       Systems in the Mechanics of Continuous Media Derived from a QR
       Decomposition of F,” International Journal of Engineering Science, 127
       (2018), 145-161.
    3) Freed, A. D. and Rajagopal, K. R. "An Optimal Representation of the
       Laplace Stretch for Transparency of the Physics of Deformation",
       in review.
    4) Bourke, P., "Polygons, Meshes", http://paulbourke.net/geometry
"""


class pentagon(object):

    def __init__(self, number, chord1, chord2, chord3, chord4, chord5, h,
                 gaussPts):
        self._number = int(number)

        # verify the input
        if not isinstance(chord1, chord):
            raise RuntimeError('chord1 passed to the pentagon constructor ' +
                               'was invalid.')
        if not isinstance(chord2, chord):
            raise RuntimeError('chord2 passed to the pentagon constructor ' +
                               'was invalid.')
        if not isinstance(chord3, chord):
            raise RuntimeError('chord3 passed to the pentagon constructor ' +
                               'was invalid.')
        if not isinstance(chord4, chord):
            raise RuntimeError('chord4 passed to the pentagon constructor ' +
                               'was invalid.')
        if not isinstance(chord5, chord):
            raise RuntimeError('chord5 passed to the pentagon constructor ' +
                               'was invalid.')

        # verify that the chords connect to form a pentagon
        c1v1, c1v2 = chord1.vertexNumbers()
        c2v1, c2v2 = chord2.vertexNumbers()
        c3v1, c3v2 = chord3.vertexNumbers()
        c4v1, c4v2 = chord4.vertexNumbers()
        c5v1, c5v2 = chord5.vertexNumbers()
        if not (c1v1 == c2v1 or c1v1 == c2v2 or c1v2 == c2v1 or c1v2 == c2v2):
            raise RuntimeError('chord1 & chord2 do not have a common vertex.')
        if not (c2v1 == c3v1 or c2v1 == c3v2 or c2v2 == c3v1 or c2v2 == c3v2):
            raise RuntimeError('chord2 & chord3 do not have a common vertex.')
        if not (c3v1 == c4v1 or c3v1 == c4v2 or c3v2 == c4v1 or c3v2 == c4v2):
            raise RuntimeError('chord3 & chord4 do not have a common vertex.')
        if not (c4v1 == c5v1 or c4v1 == c5v2 or c4v2 == c5v1 or c4v2 == c5v2):
            raise RuntimeError('chord4 & chord5 do not have a common vertex.')
        if not (c1v1 == c5v1 or c1v1 == c5v2 or c1v2 == c5v1 or c1v2 == c5v2):
            raise RuntimeError('chord1 & chord5 do not have a common vertex.')

        # check the stepsize
        if h > np.finfo(float).eps:
            self._h = float(h)
        else:
            raise RuntimeError("The stepsize in the pentagon constructor " +
                               "wasn't positive.")

        # check the number of Gauss points to use
        if gaussPts == 1 or gaussPts == 4 or gaussPts == 7:
            self._gaussPts = gaussPts
        else:
            raise RuntimeError('{} Gauss points were '.format(gaussPts) +
                               'specified in the pentagon constructor; ' +
                               'it must be 1, 4 or 7.')

        # establish the set of chords
        self._chord = {
            1: chord1,
            2: chord2,
            3: chord3,
            4: chord4,
            5: chord5
        }
        self._setOfChords = {
            chord1.number(),
            chord2.number(),
            chord3.number(),
            chord4.number(),
            chord5.number()
        }
        if not len(self._setOfChords) == 5:
            raise RuntimeError('There were not 5 unique chords in a pentagon.')

        # establish the set of vertices
        if chord1.hasVertex(c1v1) and chord5.hasVertex(c1v1):
            v1 = chord1.getVertex(c1v1)
        else:
            v1 = chord1.getVertex(c1v2)
        if chord2.hasVertex(c2v1) and chord1.hasVertex(c2v1):
            v2 = chord2.getVertex(c2v1)
        else:
            v2 = chord2.getVertex(c2v2)
        if chord3.hasVertex(c3v1) and chord2.hasVertex(c3v1):
            v3 = chord3.getVertex(c3v1)
        else:
            v3 = chord3.getVertex(c3v2)
        if chord4.hasVertex(c4v1) and chord3.hasVertex(c4v1):
            v4 = chord4.getVertex(c4v1)
        else:
            v4 = chord4.getVertex(c4v2)
        if chord5.hasVertex(c5v1) and chord4.hasVertex(c5v1):
            v5 = chord5.getVertex(c5v1)
        else:
            v5 = chord5.getVertex(c5v2)
        self._vertex = {
            # these get permuted by 1 to place the correct vertex at the apex
            1: v2,
            2: v3,
            3: v4,
            4: v5,
            5: v1
        }
        self._setOfVertices = {
            v1.number(),
            v2.number(),
            v3.number(),
            v4.number(),
            v5.number()
        }
        if not len(self._setOfVertices) == 5:
            raise RuntimeError('There were not 5 unique vertices ' +
                               'in this pentagon.')

        # establish the shape functions located at the Gauss points (xi, eta)
        if gaussPts == 1:
            xi = 0.0000000000000000
            eta = 0.0000000000000000
            sf1 = shapeFunction(xi, eta)

            self._shapeFns = {
                1: sf1
            }
        elif gaussPts == 4:
            xi1 = -0.0349156305831802
            eta1 = 0.6469731019095136
            sf1 = shapeFunction(xi1, eta1)

            xi2 = -0.5951653065516678
            eta2 = -0.0321196846022659
            sf2 = shapeFunction(xi2, eta2)

            xi3 = 0.0349156305831798
            eta3 = -0.6469731019095134
            sf3 = shapeFunction(xi3, eta3)

            xi4 = 0.5951653065516677
            eta4 = 0.0321196846022661
            sf4 = shapeFunction(xi4, eta4)

            self._shapeFns = {
                1: sf1,
                2: sf2,
                3: sf3,
                4: sf4
            }
        else:  # gaussPts = 7
            xi1 = -0.0000000000000000
            eta1 = -0.0000000000000002
            sf1 = shapeFunction(xi1, eta1)

            xi2 = -0.1351253857178451
            eta2 = 0.7099621260052327
            sf2 = shapeFunction(xi2, eta2)

            xi3 = -0.6970858746672087
            eta3 = 0.1907259121533272
            sf3 = shapeFunction(xi3, eta3)

            xi4 = -0.4651171392611024
            eta4 = -0.5531465782166917
            sf4 = shapeFunction(xi4, eta4)

            xi5 = 0.2842948078559476
            eta5 = -0.6644407817506509
            sf5 = shapeFunction(xi5, eta5)

            xi6 = 0.7117958231685716
            eta6 = -0.1251071394727008
            sf6 = shapeFunction(xi6, eta6)

            xi7 = 0.5337947578638855
            eta7 = 0.4872045224587945
            sf7 = shapeFunction(xi7, eta7)

            self._shapeFns = {
                1: sf1,
                2: sf2,
                3: sf3,
                4: sf4,
                5: sf5,
                6: sf6,
                7: sf7
            }

        # get the vertex coordinates in the reference configuration
        x1, y1, z1 = v1.coordinates('ref')
        x2, y2, z2 = v2.coordinates('ref')
        x3, y3, z3 = v3.coordinates('ref')
        x4, y4, z4 = v4.coordinates('ref')
        x5, y5, z5 = v5.coordinates('ref')

        # base vector 1: connects the two shoulders of a pentagon
        x = x5 - x2
        y = y5 - y2
        z = z5 - z2
        mag = m.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 2: goes from the apex to a point along its base

        # establish the unit vector for the base of the pentagon
        x = x4 - x3
        y = y4 - y3
        z = z4 - z3
        mag = m.sqrt(x * x + y * y + z * z)
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the base of
        # the pentagon that results in a vector n2 which is normal to n1

        def getDelta(delta):
            nx = x1 - (x3 + delta * ex)
            ny = y1 - (y3 + delta * ey)
            nz = z1 - (z3 + delta * ez)
            # when the dot product is zero then the two vectors are orthogonal
            n1Dotn2 = n1x * nx + n1y * ny + n1z * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * mag
        deltaH = 5.0 * mag
        delta = findRoot(deltaL, deltaH, getDelta)

        # create base vector 2
        x = x1 - (x3 + delta * ex)
        y = y1 - (y3 + delta * ey)
        z = z1 - (z3 + delta * ez)
        mag = m.sqrt(x * x + y * y + z * z)
        n2x = x / mag
        n2y = y / mag
        n2z = z / mag

        # base vector 3 is obtained through the cross product
        # it is normal to the pentagon and points outward from the dodecahedron
        n3x = n1y * n2z - n1z * n2y
        n3y = n1z * n2x - n1x * n2z
        n3z = n1x * n2y - n1y * n2x

        # initialize the normal vector
        self._normalX0 = n3x
        self._normalY0 = n3y
        self._normalZ0 = n3z

        # create rotation matrix from dodecahedral to pentagonal coordinates
        self._Pr3D = np.zeros((3, 3), dtype=float)
        self._Pr3D[0, 0] = n1x
        self._Pr3D[0, 1] = n2x
        self._Pr3D[0, 2] = n3x
        self._Pr3D[1, 0] = n1y
        self._Pr3D[1, 1] = n2y
        self._Pr3D[1, 2] = n3y
        self._Pr3D[2, 0] = n1z
        self._Pr3D[2, 1] = n2z
        self._Pr3D[2, 2] = n3z

        # determine vertice coordinates in the pentagonal frame of reference
        self._v1x0 = n1x * x1 + n1y * y1 + n1z * z1
        self._v1y0 = n2x * x1 + n2y * y1 + n2z * z1
        self._v2x0 = n1x * x2 + n1y * y2 + n1z * z2
        self._v2y0 = n2x * x2 + n2y * y2 + n2z * z2
        self._v3x0 = n1x * x3 + n1y * y3 + n1z * z3
        self._v3y0 = n2x * x3 + n2y * y3 + n2z * z3
        self._v4x0 = n1x * x4 + n1y * y4 + n1z * z4
        self._v4y0 = n2x * x4 + n2y * y4 + n2z * z4
        self._v5x0 = n1x * x5 + n1y * y5 + n1z * z5
        self._v5y0 = n2x * x5 + n2y * y5 + n2z * z5

        # initialize current vertice coordinates in pentagonal frame of reference
        self._v1x = self._v1x0
        self._v1y = self._v1y0
        self._v2x = self._v2x0
        self._v2y = self._v2y0
        self._v3x = self._v3x0
        self._v3y = self._v3y0
        self._v4x = self._v4x0
        self._v4y = self._v4y0
        self._v5x = self._v5x0
        self._v5y = self._v5y0

        # z offsets for the pentagonal plane (which should all be the same)
        v1z = n3x * x1 + n3y * y1 + n3z * z1
        v2z = n3x * x2 + n3y * y2 + n3z * z2
        v3z = n3x * x3 + n3y * y3 + n3z * z3
        v4z = n3x * x4 + n3y * y4 + n3z * z4
        v5z = n3x * x5 + n3y * y5 + n3z * z5
        vz0 = (v1z + v2z + v3z + v4z + v5z) / 5.0

        # determine the initial areas of this irregular pentagon
        self._A0 = ((self._v1x0 * self._v2y0 - self._v2x0 * self._v1y0 +
                     self._v2x0 * self._v3y0 - self._v3x0 * self._v2y0 +
                     self._v3x0 * self._v4y0 - self._v4x0 * self._v3y0 +
                     self._v4x0 * self._v5y0 - self._v5x0 * self._v4y0 +
                     self._v5x0 * self._v1y0 - self._v1x0 * self._v5y0) / 2.0)
        self._Ap = self._A0
        self._Ac = self._A0
        self._An = self._A0
        # this area will be positive if the vertices index counterclockwise

        # determine the centroid of this pentagon in pentagonal coordinates
        self._cx0 = (((self._v1x0 + self._v2x0) *
                      (self._v1x0 * self._v2y0 - self._v2x0 * self._v1y0) +
                      (self._v2x0 + self._v3x0) *
                      (self._v2x0 * self._v3y0 - self._v3x0 * self._v2y0) +
                      (self._v3x0 + self._v4x0) *
                      (self._v3x0 * self._v4y0 - self._v4x0 * self._v3y0) +
                      (self._v4x0 + self._v5x0) *
                      (self._v4x0 * self._v5y0 - self._v5x0 * self._v4y0) +
                      (self._v5x0 + self._v1x0) *
                      (self._v5x0 * self._v1y0 - self._v1x0 * self._v5y0))
                     / (6.0 * self._A0))
        self._cy0 = (((self._v1y0 + self._v2y0) *
                      (self._v1x0 * self._v2y0 - self._v2x0 * self._v1y0) +
                      (self._v2y0 + self._v3y0) *
                      (self._v2x0 * self._v3y0 - self._v3x0 * self._v2y0) +
                      (self._v3y0 + self._v4y0) *
                      (self._v3x0 * self._v4y0 - self._v4x0 * self._v3y0) +
                      (self._v4y0 + self._v5y0) *
                      (self._v4x0 * self._v5y0 - self._v5x0 * self._v4y0) +
                      (self._v5y0 + self._v1y0) *
                      (self._v5x0 * self._v1y0 - self._v1x0 * self._v5y0))
                     / (6.0 * self._A0))
        self._cz0 = vz0

        # rotate this centroid back into the reference coordinate system
        self._centroidX0 = n1x * self._cx0 + n2x * self._cy0 + n3x * self._cz0
        self._centroidY0 = n1y * self._cx0 + n2y * self._cy0 + n3y * self._cz0
        self._centroidZ0 = n1z * self._cx0 + n2z * self._cy0 + n3z * self._cz0

        # initialize the centroids for all configurations
        self._centroidXp = self._centroidX0
        self._centroidYp = self._centroidY0
        self._centroidZp = self._centroidZ0
        self._centroidXc = self._centroidX0
        self._centroidYc = self._centroidY0
        self._centroidZc = self._centroidZ0
        self._centroidXn = self._centroidX0
        self._centroidYn = self._centroidY0
        self._centroidZn = self._centroidZ0

        # rotation matrices: from dodecahedral frame into pentagonal frame
        self._Pp3D = np.zeros((3, 3), dtype=float)
        self._Pc3D = np.zeros((3, 3), dtype=float)
        self._Pn3D = np.zeros((3, 3), dtype=float)
        self._Pp3D[:, :] = self._Pr3D[:, :]
        self._Pc3D[:, :] = self._Pr3D[:, :]
        self._Pn3D[:, :] = self._Pr3D[:, :]

        # create matrices for a pentagon at its Gauss points via dictionaries
        # p implies previous, c implies current, n implies next
        if gaussPts == 1:
            # displacement gradients located at the Gauss points of pentagon
            self._G0 = {
                1: np.zeros((2, 2), dtype=float)
            }
            self._Gp = {
                1: np.zeros((2, 2), dtype=float)
            }
            self._Gc = {
                1: np.zeros((2, 2), dtype=float)
            }
            self._Gn = {
                1: np.zeros((2, 2), dtype=float)
            }

            # deformation gradients located at the Gauss points of pentagon
            self._F0 = {
                1: np.identity(2, dtype=float)
            }
            self._Fp = {
                1: np.identity(2, dtype=float)
            }
            self._Fc = {
                1: np.identity(2, dtype=float)
            }
            self._Fn = {
                1: np.identity(2, dtype=float)
            }

        elif gaussPts == 4:
            # displacement gradients located at the Gauss points of pentagon
            self._G0 = {
                1: np.zeros((2, 2), dtype=float),
                2: np.zeros((2, 2), dtype=float),
                3: np.zeros((2, 2), dtype=float),
                4: np.zeros((2, 2), dtype=float)
            }
            self._Gp = {
                1: np.zeros((2, 2), dtype=float),
                2: np.zeros((2, 2), dtype=float),
                3: np.zeros((2, 2), dtype=float),
                4: np.zeros((2, 2), dtype=float)
            }
            self._Gc = {
                1: np.zeros((2, 2), dtype=float),
                2: np.zeros((2, 2), dtype=float),
                3: np.zeros((2, 2), dtype=float),
                4: np.zeros((2, 2), dtype=float)
            }
            self._Gn = {
                1: np.zeros((2, 2), dtype=float),
                2: np.zeros((2, 2), dtype=float),
                3: np.zeros((2, 2), dtype=float),
                4: np.zeros((2, 2), dtype=float)
            }

            # deformation gradients located at the Gauss points of pentagon
            self._F0 = {
                1: np.identity(2, dtype=float),
                2: np.identity(2, dtype=float),
                3: np.identity(2, dtype=float),
                4: np.identity(2, dtype=float)
            }
            self._Fp = {
                1: np.identity(2, dtype=float),
                2: np.identity(2, dtype=float),
                3: np.identity(2, dtype=float),
                4: np.identity(2, dtype=float)
            }
            self._Fc = {
                1: np.identity(2, dtype=float),
                2: np.identity(2, dtype=float),
                3: np.identity(2, dtype=float),
                4: np.identity(2, dtype=float)
            }
            self._Fn = {
                1: np.identity(2, dtype=float),
                2: np.identity(2, dtype=float),
                3: np.identity(2, dtype=float),
                4: np.identity(2, dtype=float)
            }

        else:  # gaussPts = 7
            # displacement gradients located at the Gauss points of pentagon
            self._G0 = {
                1: np.zeros((2, 2), dtype=float),
                2: np.zeros((2, 2), dtype=float),
                3: np.zeros((2, 2), dtype=float),
                4: np.zeros((2, 2), dtype=float),
                5: np.zeros((2, 2), dtype=float),
                6: np.zeros((2, 2), dtype=float),
                7: np.zeros((2, 2), dtype=float)
            }
            self._Gp = {
                1: np.zeros((2, 2), dtype=float),
                2: np.zeros((2, 2), dtype=float),
                3: np.zeros((2, 2), dtype=float),
                4: np.zeros((2, 2), dtype=float),
                5: np.zeros((2, 2), dtype=float),
                6: np.zeros((2, 2), dtype=float),
                7: np.zeros((2, 2), dtype=float)
            }
            self._Gc = {
                1: np.zeros((2, 2), dtype=float),
                2: np.zeros((2, 2), dtype=float),
                3: np.zeros((2, 2), dtype=float),
                4: np.zeros((2, 2), dtype=float),
                5: np.zeros((2, 2), dtype=float),
                6: np.zeros((2, 2), dtype=float),
                7: np.zeros((2, 2), dtype=float)
            }
            self._Gn = {
                1: np.zeros((2, 2), dtype=float),
                2: np.zeros((2, 2), dtype=float),
                3: np.zeros((2, 2), dtype=float),
                4: np.zeros((2, 2), dtype=float),
                5: np.zeros((2, 2), dtype=float),
                6: np.zeros((2, 2), dtype=float),
                7: np.zeros((2, 2), dtype=float)
            }

            # deformation gradients located at the Gauss points of pentagon
            self._F0 = {
                1: np.identity(2, dtype=float),
                2: np.identity(2, dtype=float),
                3: np.identity(2, dtype=float),
                4: np.identity(2, dtype=float),
                5: np.identity(2, dtype=float),
                6: np.identity(2, dtype=float),
                7: np.identity(2, dtype=float)
            }
            self._Fp = {
                1: np.identity(2, dtype=float),
                2: np.identity(2, dtype=float),
                3: np.identity(2, dtype=float),
                4: np.identity(2, dtype=float),
                5: np.identity(2, dtype=float),
                6: np.identity(2, dtype=float),
                7: np.identity(2, dtype=float)
            }
            self._Fc = {
                1: np.identity(2, dtype=float),
                2: np.identity(2, dtype=float),
                3: np.identity(2, dtype=float),
                4: np.identity(2, dtype=float),
                5: np.identity(2, dtype=float),
                6: np.identity(2, dtype=float),
                7: np.identity(2, dtype=float)
            }
            self._Fn = {
                1: np.identity(2, dtype=float),
                2: np.identity(2, dtype=float),
                3: np.identity(2, dtype=float),
                4: np.identity(2, dtype=float),
                5: np.identity(2, dtype=float),
                6: np.identity(2, dtype=float),
                7: np.identity(2, dtype=float)
            }

        # assign membrane objects to each Gauss point
        if gaussPts == 1:
            mem1 = membrane(h)

            self._septum = {
                1: mem1
            }
        elif gaussPts == 4:
            mem1 = membrane(h)
            mem2 = membrane(h)
            mem3 = membrane(h)
            mem4 = membrane(h)

            self._septum = {
                1: mem1,
                2: mem2,
                3: mem3,
                4: mem4
            }
        else:  # gaussPts = 7
            mem1 = membrane(h)
            mem2 = membrane(h)
            mem3 = membrane(h)
            mem4 = membrane(h)
            mem5 = membrane(h)
            mem6 = membrane(h)
            mem7 = membrane(h)

            self._septum = {
                1: mem1,
                2: mem2,
                3: mem3,
                4: mem4,
                5: mem5,
                6: mem6,
                7: mem7
            }

        self._rho = mp.rhoSepta()

        self._width = mp.septalWidth()

        M1, M2, Me_t, N1, N2, Ne_t, G1, G2, Ge_t = mp.septalMembrane()
        # the elastic moduli governing dilation
        self._M1 = M1
        self._M2 = M2
        self._Me_t = Me_t
        # the elastic moduli governing squeeze
        self._N1 = N1
        self._N2 = N2
        self._Ne_t = Ne_t
        # the elastic moduli governing shear
        self._G1 = G1
        self._G2 = G2
        self._Ge_t = Ge_t

    def toString(self, state):
        if self._number < 10:
            s = 'pentagon[0'
        else:
            s = 'pentagon['
        s = s + str(self._number)
        s = s + '] has vertices: \n'
        if isinstance(state, str):
            s = s + '   1: ' + self._vertex[1].toString(state) + '\n'
            s = s + '   2: ' + self._vertex[2].toString(state) + '\n'
            s = s + '   3: ' + self._vertex[3].toString(state) + '\n'
            s = s + '   4: ' + self._vertex[4].toString(state) + '\n'
            s = s + '   5: ' + self._vertex[5].toString(state)
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.toString.")
        return s

    def number(self):
        return self._number

    def chordNumbers(self):
        numbers = sorted(self._setOfChords)
        return numbers[0], numbers[1], numbers[2], numbers[3], numbers[4]

    def vertexNumbers(self):
        numbers = sorted(self._setOfVertices)
        return numbers[0], numbers[1], numbers[2], numbers[3], numbers[4]

    def hasChord(self, number):
        return number in self._setOfChords

    def hasVertex(self, number):
        return number in self._setOfVertices

    def getChord(self, number):
        if self._chord[1].number() == number:
            return self._chord[1]
        elif self._chord[2].number() == number:
            return self._chord[2]
        elif self._chord[3].number() == number:
            return self._chord[3]
        elif self._chord[4].number() == number:
            return self._chord[4]
        elif self._chord[5].number() == number:
            return self._chord[5]
        else:
            raise RuntimeError('The requested chord {} '.format(number) +
                               'is not in pentagon {}.'.format(self._number))

    def getVertex(self, number):
        if self._vertex[1].number() == number:
            return self._vertex[1]
        elif self._vertex[2].number() == number:
            return self._vertex[2]
        elif self._vertex[3].number() == number:
            return self._vertex[3]
        elif self._vertex[4].number() == number:
            return self._vertex[4]
        elif self._vertex[5].number() == number:
            return self._vertex[5]
        else:
            raise RuntimeError('The requested vertex {} '.format(number) +
                               'is not in pentagon {}.'.format(self._number))

    def gaussPoints(self):
        return self._gaussPts

    def update(self):
        # computes the fields positioned at the next time step

        # get the updated coordinates for the vetices of the pentagon
        x1, y1, z1 = self._vertex[1].coordinates('next')
        x2, y2, z2 = self._vertex[2].coordinates('next')
        x3, y3, z3 = self._vertex[3].coordinates('next')
        x4, y4, z4 = self._vertex[4].coordinates('next')
        x5, y5, z5 = self._vertex[5].coordinates('next')

        # base vector 1: connects the two shoulders of the pentagon
        x = x5 - x2
        y = y5 - y2
        z = z5 - z2
        mag = m.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 2: goes from the apex to a point along its base

        # establish the unit vector for the base of the pentagon
        x = x4 - x3
        y = y4 - y3
        z = z4 - z3
        mag = m.sqrt(x * x + y * y + z * z)
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the base of
        # the pentagon that results in a vector n2 which is normal to n1

        def getDelta(delta):
            nx = x1 - (x3 + delta * ex)
            ny = y1 - (y3 + delta * ey)
            nz = z1 - (z3 + delta * ez)
            # when the dot product is zero then the two vectors are orthogonal
            n1Dotn2 = n1x * nx + n1y * ny + n1z * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * mag
        deltaH = 5.0 * mag
        delta = findRoot(deltaL, deltaH, getDelta)

        # create base vector 2
        x = x1 - (x3 + delta * ex)
        y = y1 - (y3 + delta * ey)
        z = z1 - (z3 + delta * ez)
        mag = m.sqrt(x * x + y * y + z * z)
        n2x = x / mag
        n2y = y / mag
        n2z = z / mag

        # base vector 3 is obtained through the cross product
        # it is normal to the pentagon and points outward from the dodecahedron
        n3x = n1y * n2z - n1z * n2y
        n3y = n1z * n2x - n1x * n2z
        n3z = n1x * n2y - n1y * n2x

        # create the rotation matrix from reference to pentagonal coordinates
        self._Pn3D[0, 0] = n1x
        self._Pn3D[0, 1] = n2x
        self._Pn3D[0, 2] = n3x
        self._Pn3D[1, 0] = n1y
        self._Pn3D[1, 1] = n2y
        self._Pn3D[1, 2] = n3y
        self._Pn3D[2, 0] = n1z
        self._Pn3D[2, 1] = n2z
        self._Pn3D[2, 2] = n3z

        # determine vertice coordinates in the pentagonal frame of reference
        self._v1x = n1x * x1 + n1y * y1 + n1z * z1
        self._v1y = n2x * x1 + n2y * y1 + n2z * z1
        self._v2x = n1x * x2 + n1y * y2 + n1z * z2
        self._v2y = n2x * x2 + n2y * y2 + n2z * z2
        self._v3x = n1x * x3 + n1y * y3 + n1z * z3
        self._v3y = n2x * x3 + n2y * y3 + n2z * z3
        self._v4x = n1x * x4 + n1y * y4 + n1z * z4
        self._v4y = n2x * x4 + n2y * y4 + n2z * z4
        self._v5x = n1x * x5 + n1y * y5 + n1z * z5
        self._v5y = n2x * x5 + n2y * y5 + n2z * z5

        # z offsets for the pentagonal plane (which should all be the same)
        v1z = n3x * x1 + n3y * y1 + n3z * z1
        v2z = n3x * x2 + n3y * y2 + n3z * z2
        v3z = n3x * x3 + n3y * y3 + n3z * z3
        v4z = n3x * x4 + n3y * y4 + n3z * z4
        v5z = n3x * x5 + n3y * y5 + n3z * z5

        # determine the area of this irregular pentagon
        self._An = (self._v1x * self._v2y - self._v2x * self._v1y +
                    self._v2x * self._v3y - self._v3x * self._v2y +
                    self._v3x * self._v4y - self._v4x * self._v3y +
                    self._v4x * self._v5y - self._v5x * self._v4y +
                    self._v5x * self._v1y - self._v1x * self._v5y) / 2.0
        # the area will be positive if the vertices index counter clockwise

        # determine the centroid of this pentagon in pentagonal coordinates
        cx = ((self._v1x + self._v2x) * 
              (self._v1x * self._v2y - self._v2x * self._v1y) +
              (self._v2x + self._v3x) * 
              (self._v2x * self._v3y - self._v3x * self._v2y) +
              (self._v3x + self._v4x) * 
              (self._v3x * self._v4y - self._v4x * self._v3y) +
              (self._v4x + self._v5x) * 
              (self._v4x * self._v5y - self._v5x * self._v4y) +
              (self._v5x + self._v1x) * 
              (self._v5x * self._v1y - 
               self._v1x * self._v5y)) / (6.0 * self._An)
        cy = ((self._v1y + self._v2y) * 
              (self._v1x * self._v2y - self._v2x * self._v1y) +
              (self._v2y + self._v3y) * 
              (self._v2x * self._v3y - self._v3x * self._v2y) +
              (self._v3y + self._v4y) * 
              (self._v3x * self._v4y - self._v4x * self._v3y) +
              (self._v4y + self._v5y) * 
              (self._v4x * self._v5y - self._v5x * self._v4y) +
              (self._v5y + self._v1y) * 
              (self._v5x * self._v1y - 
               self._v1x * self._v5y)) / (6.0 * self._An)
        cz = (v1z + v2z + v3z + v4z + v5z) / 5.0

        # rotate this centroid back into the reference coordinate system
        self._centroidXn = n1x * cx + n2x * cy + n3x * cz
        self._centroidYn = n1y * cx + n2y * cy + n3y * cz
        self._centroidZn = n1z * cx + n2z * cy + n3z * cz

        # Determine the deformation gradient for this irregular pentagon

        # current vertex coordinates in pentagonal frame of reference
        x1 = (self._v1x, self._v1y)
        x2 = (self._v2x, self._v2y)
        x3 = (self._v3x, self._v3y)
        x4 = (self._v4x, self._v4y)
        x5 = (self._v5x, self._v5y)

        # reference vertex coordinates in pentagonal frame of reference
        x10 = (self._v1x0, self._v1y0)
        x20 = (self._v2x0, self._v2y0)
        x30 = (self._v3x0, self._v3y0)
        x40 = (self._v4x0, self._v4y0)
        x50 = (self._v5x0, self._v5y0)

        # establish the deformation and displacement gradients as dictionaries
        if self._gaussPts == 1:
            # displacement gradients located at the Gauss points of pentagon
            self._Gn[1] = self._shapeFns[1].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            # deformation gradients located at the Gauss points of pentagon
            self._Fn[1] = self._shapeFns[1].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
        elif self._gaussPts == 4:
            # displacement gradients located at the Gauss points of pentagon
            self._Gn[1] = self._shapeFns[1].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Gn[2] = self._shapeFns[2].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Gn[3] = self._shapeFns[3].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Gn[4] = self._shapeFns[4].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            # deformation gradients located at the Gauss points of pentagon
            self._Fn[1] = self._shapeFns[1].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Fn[2] = self._shapeFns[2].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Fn[3] = self._shapeFns[3].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Fn[4] = self._shapeFns[4].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
        else:  # gaussPts = 7
            # displacement gradients located at the Gauss points of pentagon
            self._Gn[1] = self._shapeFns[1].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Gn[2] = self._shapeFns[2].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Gn[3] = self._shapeFns[3].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Gn[4] = self._shapeFns[4].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Gn[5] = self._shapeFns[5].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Gn[6] = self._shapeFns[6].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Gn[7] = self._shapeFns[7].G(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            # deformation gradients located at the Gauss points of pentagon
            self._Fn[1] = self._shapeFns[1].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Fn[2] = self._shapeFns[2].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Fn[3] = self._shapeFns[3].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Fn[4] = self._shapeFns[4].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Fn[5] = self._shapeFns[5].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Fn[6] = self._shapeFns[6].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)
            self._Fn[7] = self._shapeFns[7].F(x1, x2, x3, x4, x5,
                                              x10, x20, x30, x40, x50)

        # update the membrane objects at each Gauss point
        for i in range(1, self._gaussPts+1):
            self._septum[i].update(self._Fn[i])

        return  # nothing

    def advance(self):
        # advance the geometric properties of the pentagon
        self._Ap = self._Ac
        self._Ac = self._An
        self._centroidXp = self._centroidXc
        self._centroidYp = self._centroidYc
        self._centroidZp = self._centroidZc
        self._centroidXc = self._centroidXn
        self._centroidYc = self._centroidYn
        self._centroidZc = self._centroidZn

        # advance rotation matrix: from the dodecaheral into the pentagonal
        self._Pp3D[:, :] = self._Pc3D[:, :]
        self._Pc3D[:, :] = self._Pn3D[:, :]

        # advance the matrix fields associated with each Gauss point
        for i in range(1, self._gaussPts+1):
            self._Fp[i][:, :] = self._Fc[i][:, :]
            self._Fc[i][:, :] = self._Fn[i][:, :]
            self._Gp[i][:, :] = self._Gc[i][:, :]
            self._Gc[i][:, :] = self._Gn[i][:, :]

        # advance the membrane objects at each Gauss point
        for i in range(1, self._gaussPts+1):
            self._septum[i].advance()

    # Material properties that associate with this tetrahedron.  Except for the
    # mass density, all are drawn randomly from a statistical distribution.

    def massDensity(self):
        # returns the mass density of the membrane
        return self._rho

    def width(self, state):
        # returns the cross-sectional thickness or width of the membrane
        # assuming volume is preserved
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._width * self._A0 / self._Ac
            elif state == 'n' or state == 'next':
                return self._width * self._A0 / self._An
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._width * self._A0 / self._Ap
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._width
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.width.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.width.")

    def matProp(self):
        # elastic properties describing planar dilation
        M1 = self._M1
        M2 = self._M2
        Me_t = self._Me_t
        # elastic properties describing planar squeeze
        N1 = self._N1
        N2 = self._N2
        Ne_t = self._Ne_t
        # elastic properties describing planar shear
        G1 = self._G1
        G2 = self._G2
        Ge_t = self._Ge_t
        return M1, M2, Me_t, N1, N2, Ne_t, G1, G2, Ge_t

    # geometric properties of a pentagon

    def area(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._Ac
            elif state == 'n' or state == 'next':
                return self._An
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._Ap
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._A0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.area.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.area.")

    def arealStretch(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return m.sqrt(self._Ac / self._A0)
            elif state == 'n' or state == 'next':
                return m.sqrt(self._An / self._A0)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return m.sqrt(self._Ap / self._A0)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 1.0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call pentagon.arealStretch.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.arealStretch.")

    def arealStrain(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return m.log(self._Ac / self._A0) / 2.0
            elif state == 'n' or state == 'next':
                return m.log(self._An / self._A0) / 2.0
            elif state == 'p' or state == 'prev' or state == 'previous':
                return m.log(self._Ap / self._A0) / 2.0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call pentagon.arealStrain.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.arealStrain.")

    def dArealStrain(self, state):
        if isinstance(state, str):
            h = 2.0 * self._h
            if state == 'c' or state == 'curr' or state == 'current':
                # use second-order central difference formula
                dArea = (self._An - self._Ap) / h
                return (dArea / self._Ac) / 2.0
            elif state == 'n' or state == 'next':
                # use second-order backward difference formula
                dArea = (3.0 * self._An - 4.0 * self._Ac + self._Ap) / h
                return (dArea / self._An) / 2.0
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use second-order forward difference formula
                dArea = (-self._An + 4.0 * self._Ac - 3.0 * self._Ap) / h
                return (dArea / self._Ap) / 2.0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.dArealStrain.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.dArealStrain.")

    def normal(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                nx = self._Pc3D[0, 2]
                ny = self._Pc3D[1, 2]
                nz = self._Pc3D[2, 2]
            elif state == 'n' or state == 'next':
                nx = self._Pn3D[0, 2]
                ny = self._Pn3D[1, 2]
                nz = self._Pn3D[2, 2]
            elif state == 'p' or state == 'prev' or state == 'previous':
                nx = self._Pp3D[0, 2]
                ny = self._Pp3D[1, 2]
                nz = self._Pp3D[2, 2]
            elif state == 'r' or state == 'ref' or state == 'reference':
                nx = self._Pr3D[0, 2]
                ny = self._Pr3D[1, 2]
                nz = self._Pr3D[2, 2]
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.normal.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.normal.")
        return np.array([nx, ny, nz])

    def dNormal(self, state):
        if isinstance(state, str):
            h = 2.0 * self._h
            if state == 'c' or state == 'curr' or state == 'current':
                # use second-order central difference formula
                dnx = (self._Pn3D[0, 2] - self._Pp3D[0, 2]) / h
                dny = (self._Pn3D[1, 2] - self._Pp3D[1, 2]) / h
                dnz = (self._Pn3D[2, 2] - self._Pp3D[2, 2]) / h
            elif state == 'n' or state == 'next':
                # use second-order backward difference formula
                dnx = (3.0 * self._Pn3D[0, 2] - 4.0 * self._Pc3D[0, 2]
                       + self._Pp3D[0, 2]) / h
                dny = (3.0 * self._Pn3D[1, 2] - 4.0 * self._Pc3D[1, 2]
                       + self._Pp3D[1, 2]) / h
                dnz = (3.0 * self._Pn3D[2, 2] - 4.0 * self._Pc3D[2, 2]
                       + self._Pp3D[2, 2]) / h
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use second-order forward difference formula
                dnx = (-self._Pn3D[0, 2] + 4.0 * self._Pc3D[0, 2]
                       - 3.0 * self._Pp3D[0, 2]) / h
                dny = (-self._Pn3D[1, 2] + 4.0 * self._Pc3D[1, 2]
                       - 3.0 * self._Pp3D[1, 2]) / h
                dnz = (-self._Pn3D[2, 2] + 4.0 * self._Pc3D[2, 2]
                       - 3.0 * self._Pp3D[2, 2]) / h
            elif state == 'r' or state == 'ref' or state == 'reference':
                dnx = 0.0
                dny = 0.0
                dnz = 0.0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.dNormal.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.dNormal.")
        return np.array([dnx, dny, dnz])

    # properties that associate with the centroid of the pentagon

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
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.centroid.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.centroid.")
        return np.array([cx, cy, cz])

    def displacement(self, state):
        x0, y0, z0 = self.centroid('ref')
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
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.velocity.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.velocity.")
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
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.acceleration.")
        return np.array([ax, ay, az])

    def rotation(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Pc3D)
            elif state == 'n' or state == 'next':
                return np.copy(self._Pn3D)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Pp3D)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._Pr3D)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.rotation.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.rotation.")

    def spin(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return spinMtx.currSpin(self._Pp3D, self._Pc3D,
                                        self._Pn3D, self._h)
            elif state == 'n' or state == 'next':
                return spinMtx.nextSpin(self._Pp3D, self._Pc3D,
                                        self._Pn3D, self._h)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return spinMtx.prevSpin(self._Pp3D, self._Pc3D,
                                        self._Pn3D, self._h)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.zeros((3, 3), dtype=float)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.spin.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.spin.")

    # fundamental fields from kinematics

    # displacement gradient at a Gauss point
    def G(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.G and you sent {}.".format(gaussPt))
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
                                   "in a call to pentagon.G.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.G.")

    # deformation gradient at a Gauss point
    def F(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.F and you sent {}.".format(gaussPt))
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
                                   "in a call to pentagon.F.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.F.")

    # velocity gradient at a Gauss point
    def L(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.L and you sent {}.".format(gaussPt))

        def FInv(fMtx):
            fInv = np.array((2, 2), dtype=float)
            det = fMtx[0, 0] * fMtx[1, 1] - fMtx[1, 0] * fMtx[0, 1]
            fInv[0, 0] = fMtx[1, 1] / det
            fInv[0, 1] = -fMtx[0, 1] / det
            fInv[1, 0] = -fMtx[1, 0] / det
            fInv[1, 1] = fMtx[0, 0] / det
            return fInv

        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                # use central difference scheme
                dF = ((self._Fn[gaussPt] - self._Fp[gaussPt])
                      / (2.0 * self._h))
                fInv = FInv(self._Fc[gaussPt])
            elif state == 'n' or state == 'next':
                # use backward difference scheme
                dF = ((3.0 * self._Fn[gaussPt] - 4.0 * self._Fc[gaussPt] +
                       self._Fp[gaussPt]) / (2.0 * self._h))
                fInv = FInv(self._Fn[gaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use forward difference scheme
                dF = ((-self._Fn[gaussPt] + 4.0 * self._Fc[gaussPt] -
                       3.0 * self._Fp[gaussPt]) / (2.0 * self._h))
                fInv = FInv(self._Fp[gaussPt])
            elif state == 'r' or state == 'ref' or state == 'reference':
                dF = np.zeros(2, dtype=float)
                fInv = np.zeros(2, dtype=float)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.L.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.L.")
        return np.dot(dF, fInv)

    # methods that associate with a QR decomposition

    # the orthogonal matrix that relabels the coordinate directions
    def Q(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.Q and you sent {}.".format(gaussPt))
        return self._septum[gaussPt].Q(state)

    # the orthogonal matrix from a Gram-Schmidt factorization of (relabeled) F
    def R(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.R and you sent {}.".format(gaussPt))
        return self._septum[gaussPt].R(state)

    # a skew-symmetric matrix for the spin associated with R
    def Omega(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.Omega, you sent {}.".format(gaussPt))
        return self._septum[gaussPt].spin(state)

    # Laplace stretch from a Gram-Schmidt factorization of (relabeled) F
    def U(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.U and you sent {}.".format(gaussPt))
        return self._septum[gaussPt].U(state)

    # inverse Laplace stretch from Gram-Schmidt factorization of (relabeled) F
    def UInv(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.UInv, you sent {}.".format(gaussPt))
        return self._septum[gaussPt].UInv(state)

    # differential in the Laplace stretch from Gram-Schmidt factorization of F
    def dU(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.dU and you sent {}.".format(gaussPt))
        return self._septum[gaussPt].dU(state)

    # inverse Laplace stretch from Gram-Schmidt factorization of (relabeled) F
    def dUInv(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.dUInv, you sent {}.".format(gaussPt))
        return self._septum[gaussPt].dUInv(state)

    # physical kinematic attributes at a Gauss point in the pentagon

    def dilation(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.dilation and you sent " +
                               "{}.".format(gaussPt))
        return self._septum[gaussPt].dilation(state)

    def squeeze(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.squeeze and you sent " +
                               "{}.".format(gaussPt))
        return self._septum[gaussPt].squeeze(state)

    def shear(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.shear and you sent " +
                               "{}.".format(gaussPt))
        return self._septum[gaussPt].shear(state)

    def dDilation(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.dDilation and you sent " +
                               "{}.".format(gaussPt))
        return self._septum[gaussPt].dDilation(state)

    def dSqueeze(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.dSqueeze and you sent " +
                               "{}.".format(gaussPt))
        return self._septum[gaussPt].dSqueeze(state)

    def dShear(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "pentagon.dShear and you sent " +
                               "{}.".format(gaussPt))
        return self._septum[gaussPt].dShear(state)

    # properties used in finite elements

    def shapeFunction(self, gaussPt):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("gaussPt must be in the range of " +
                               "[1, {}] in call ".format(self._gaussPts) +
                               "to pentagon.shapeFunction and you sent " +
                               "{}.".format(gaussPt))
            sf = self._shapeFns[gaussPt]
        return sf

    def massMatrix(self):
        # assign coordinates at the vertices in the reference configuration
        x01 = (self._v1x0, self._v1y0)
        x02 = (self._v2x0, self._v2y0)
        x03 = (self._v3x0, self._v3y0)
        x04 = (self._v4x0, self._v4y0)
        x05 = (self._v5x0, self._v5y0)

        # determine the mass matrix
        if self._gaussPts == 1:
            # 'natural' weight of the element
            w = np.array([2.3776412907378837])

            jacob = self._shapeFns[1].jacobian(x01, x02, x03, x04, x05)
            
            # determinant of the Jacobian matrix
            detJ = det(jacob)
            
            nn1 = np.dot(np.transpose(self._shapeFns[1].Nmatx),
                         self._shapeFns[1].Nmatx)

            # the consistent mass matrix for 1 Gauss point
            massC = self._rho * self._width * (detJ * w[0] * nn1)

            # the lumped mass matrix for 1 Gauss point
            massL = np.zeros((10, 10), dtype=float)
            row, col = np.diag_indices_from(massC)
            massL[row, col] = massC.sum(axis=1)

            # the mass matrix is the average of the above two mass matrices
            mass = 0.5 * (massC + massL)
        elif self._gaussPts == 4:
            # 'natural' weights of the element
            w = np.array([0.5449124407446143, 0.6439082046243272,
                            0.5449124407446146, 0.6439082046243275])

            jacob1 = self._shapeFns[1].jacobian(x01, x02, x03, x04, x05)
            jacob2 = self._shapeFns[2].jacobian(x01, x02, x03, x04, x05)
            jacob3 = self._shapeFns[3].jacobian(x01, x02, x03, x04, x05)
            jacob4 = self._shapeFns[4].jacobian(x01, x02, x03, x04, x05)

            # determinant of the Jacobian matrix
            detJ1 = det(jacob1)
            detJ2 = det(jacob2)
            detJ3 = det(jacob3)            
            detJ4 = det(jacob4)

            nn1 = np.dot(np.transpose(self._shapeFns[1].Nmatx),
                         self._shapeFns[1].Nmatx)
            nn2 = np.dot(np.transpose(self._shapeFns[2].Nmatx),
                         self._shapeFns[2].Nmatx)
            nn3 = np.dot(np.transpose(self._shapeFns[3].Nmatx),
                         self._shapeFns[3].Nmatx)
            nn4 = np.dot(np.transpose(self._shapeFns[4].Nmatx),
                         self._shapeFns[4].Nmatx)

            # the consistent mass matrix for 4 Gauss points
            massC = (self._rho * self._width * (detJ1 * w[0] * nn1 +
                                                detJ2 * w[1] * nn2 +
                                                detJ3 * w[2] * nn3 +
                                                detJ4 * w[3] * nn4))

            # the lumped mass matrix for 4 Gauss points
            massL = np.zeros((10, 10), dtype=float)
            row, col = np.diag_indices_from(massC)
            massL[row, col] = massC.sum(axis=1)

            # the mass matrix is the average of the above two mass matrices
            mass = 0.5 * (massC + massL)
        else:  # gaussPts = 7
            # 'natural' weights of the element
            w = np.array([0.6257871064166934, 0.3016384608809768,
                            0.3169910433902452, 0.3155445150066620,
                            0.2958801959111726, 0.2575426306970870,
                            0.2642573384350463])

            jacob1 = self._shapeFns[1].jacobian(x01, x02, x03, x04, x05)
            jacob2 = self._shapeFns[2].jacobian(x01, x02, x03, x04, x05)
            jacob3 = self._shapeFns[3].jacobian(x01, x02, x03, x04, x05)
            jacob4 = self._shapeFns[4].jacobian(x01, x02, x03, x04, x05)
            jacob5 = self._shapeFns[5].jacobian(x01, x02, x03, x04, x05)
            jacob6 = self._shapeFns[6].jacobian(x01, x02, x03, x04, x05)
            jacob7 = self._shapeFns[7].jacobian(x01, x02, x03, x04, x05)

            # determinant of the Jacobian matrix
            detJ1 = det(jacob1)
            detJ2 = det(jacob2)
            detJ3 = det(jacob3)            
            detJ4 = det(jacob4)
            detJ5 = det(jacob5)
            detJ6 = det(jacob6)            
            detJ7 = det(jacob7)
            
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
            nn6 = np.dot(np.transpose(self._shapeFns[6].Nmatx),
                         self._shapeFns[6].Nmatx)
            nn7 = np.dot(np.transpose(self._shapeFns[7].Nmatx),
                         self._shapeFns[7].Nmatx)

            # the consistent mass matrix for 7 Gauss points
            massC = (self._rho * self._width * (detJ1 * w[0] * nn1 +
                                                detJ2 * w[1] * nn2 +
                                                detJ3 * w[2] * nn3 +
                                                detJ4 * w[3] * nn4 +
                                                detJ5 * w[4] * nn5 +
                                                detJ6 * w[5] * nn6 +
                                                detJ7 * w[6] * nn7))

            # the lumped mass matrix for 7 Gauss points
            massL = np.zeros((10, 10), dtype=float)
            row, col = np.diag_indices_from(massC)
            massL[row, col] = massC.sum(axis=1)

            # the mass matrix is the average of the above two mass matrices
            mass = 0.5 * (massC + massL)
        return mass

    def stiffnessMatrix(self, M):
        # current vertex coordinates in pentagonal frame of reference
        x1 = (self._v1x, self._v1y)
        x2 = (self._v2x, self._v2y)
        x3 = (self._v3x, self._v3y)
        x4 = (self._v4x, self._v4y)
        x5 = (self._v5x, self._v5y)

        # assign coordinates at the vertices in the reference configuration
        x01 = (self._v1x0, self._v1y0)
        x02 = (self._v2x0, self._v2y0)
        x03 = (self._v3x0, self._v3y0)
        x04 = (self._v4x0, self._v4y0)
        x05 = (self._v5x0, self._v5y0)

        # displacement of each vertex
        u1 = x1[0] - x01[0]
        u2 = x2[0] - x02[0]
        u3 = x3[0] - x03[0]
        u4 = x4[0] - x04[0]
        u5 = x5[0] - x05[0]
        v1 = x1[1] - x01[1]
        v2 = x2[1] - x02[1]
        v3 = x3[1] - x03[1]
        v4 = x4[1] - x04[1]
        v5 = x5[1] - x05[1]
        
        # creat the displacement vector
        delta = np.array([[u1, u1, v1, v1, u2, u2, v2, v2, u3, u3, v3, v3, 
                           u4, u4, v4, v4, u5, u5, v5, v5]]).T
            
        # determine the stiffness matrix
        if self._gaussPts == 1:
            # 'natural' weight of the element
            w = np.array([2.3776412907378837])
            
            jacob = self._shapeFns[1].jacobian(x01, x02, x03, x04, x05)
            
            # determinant of the Jacobian matrix
            detJ = det(jacob)
            
            # create the linear Bmatrix
            BL = self._shapeFns[1].BLinear(x1, x2, x3, x4, x5) * inv(detJ)
            # the linear stiffness matrix for 1 Gauss point
            KL = self._width * (detJ * w[0] * BL.T.dot(M).dot(BL))
            
            # create the nonlinear Bmatrix
            BN = self._shapeFns[1].BNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            # creat the H matrix
            H = self._shapeFns[1].Hmatrix(x1, x2, x3, x4, x5)
            # creat the AP matrix
            AP = self._shapeFns[1].APmatrix(x1, x2, x3, x4, x5)
            # the nonlinear stiffness matrix for 1 Gauss point
            KN = self._width * (detJ * w[0] * (BL.T.dot(M).dot(BN) +
                                BN.T.dot(M).dot(BL) + BN.T.dot(M).dot(BN) +
                                0.5 * BL.T.dot(M).dot(AP).dot(H).dot(delta) +
                                0.5 * BN.T.dot(M).dot(AP).dot(H).dot(delta)))
            
            # create the stress matrix
            
            # create the stress stiffness matrix            
            KS = self._width * (detJ * w[0] * H.T.dot(st).dot(H))
            
            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS
            
        elif self._gaussPts == 4:
            # 'natural' weights of the element
            w = np.array([0.5449124407446143, 0.6439082046243272,
                            0.5449124407446146, 0.6439082046243275])

            jacob1 = self._shapeFns[1].jacobian(x01, x02, x03, x04, x05)
            jacob2 = self._shapeFns[2].jacobian(x01, x02, x03, x04, x05)
            jacob3 = self._shapeFns[3].jacobian(x01, x02, x03, x04, x05)
            jacob4 = self._shapeFns[4].jacobian(x01, x02, x03, x04, x05)

            # determinant of the Jacobian matrix
            detJ1 = det(jacob1)
            detJ2 = det(jacob2)
            detJ3 = det(jacob3)            
            detJ4 = det(jacob4)

            # create the linear Bmatrix
            BL1 = self._shapeFns[1].BLinear(x1, x2, x3, x4, x5) * inv(detJ1)
            BL2 = self._shapeFns[2].BLinear(x1, x2, x3, x4, x5) * inv(detJ2)
            BL3 = self._shapeFns[3].BLinear(x1, x2, x3, x4, x5) * inv(detJ3)
            BL4 = self._shapeFns[4].BLinear(x1, x2, x3, x4, x5) * inv(detJ4)

            # the linear stiffness matrix for 4 Gauss points
            KL = self._width * (detJ1 * w[0] * BL1.T.dot(M).dot(BL1) +
                                detJ2 * w[1] * BL2.T.dot(M).dot(BL2) +
                                detJ3 * w[2] * BL3.T.dot(M).dot(BL3) +
                                detJ4 * w[3] * BL4.T.dot(M).dot(BL4))
            
            # create the nonlinear Bmatrix
            BN1 = self._shapeFns[1].BNonLinear(x1, x2, x3, x4, x5, 
                                x01, x02, x03, x04, x05)
            BN2 = self._shapeFns[2].BNonLinear(x1, x2, x3, x4, x5, 
                                x01, x02, x03, x04, x05)
            BN3 = self._shapeFns[3].BNonLinear(x1, x2, x3, x4, x5, 
                                x01, x02, x03, x04, x05)
            BN4 = self._shapeFns[4].BNonLinear(x1, x2, x3, x4, x5, 
                                x01, x02, x03, x04, x05)

            # create the H matrix
            H1 = self._shapeFns[1].Hmatrix(x1, x2, x3, x4, x5)
            H2 = self._shapeFns[2].Hmatrix(x1, x2, x3, x4, x5)
            H3 = self._shapeFns[3].Hmatrix(x1, x2, x3, x4, x5)
            H4 = self._shapeFns[4].Hmatrix(x1, x2, x3, x4, x5)

            # create the AP matrix
            AP1 = self._shapeFns[1].APmatrix(x1, x2, x3, x4, x5)
            AP2 = self._shapeFns[2].APmatrix(x1, x2, x3, x4, x5)
            AP3 = self._shapeFns[3].APmatrix(x1, x2, x3, x4, x5)
            AP4 = self._shapeFns[4].APmatrix(x1, x2, x3, x4, x5)
            
            # the nonlinear stiffness matrix for 1 Gauss point
            KN = self._width * (detJ1 * w[0] * (BL1.T.dot(M).dot(BN1) + 
                                BN1.T.dot(M).dot(BL1) + BN1.T.dot(M).dot(BN1) +
                                0.5 * BL1.T.dot(M).dot(AP1).dot(H1).dot(delta) +
                                0.5 * BN1.T.dot(M).dot(AP1).dot(H1).dot(delta)) +
                                detJ2 * w[1] * (BL2.T.dot(M).dot(BN2) +
                                BN2.T.dot(M).dot(BL2) + BN2.T.dot(M).dot(BN2) +
                                0.5 * BL2.T.dot(M).dot(AP2).dot(H2).dot(delta) +
                                0.5 * BN2.T.dot(M).dot(AP2).dot(H2).dot(delta)) +
                                detJ3 * w[2] * (BL3.T.dot(M).dot(BN3) +
                                BN3.T.dot(M).dot(BL3) + BN3.T.dot(M).dot(BN3) +
                                0.5 * BL3.T.dot(M).dot(AP3).dot(H3).dot(delta) +
                                0.5 * BN3.T.dot(M).dot(AP3).dot(H3).dot(delta)) +
                                detJ4 * w[3] * (BL4.T.dot(M).dot(BN4) +
                                BN4.T.dot(M).dot(BL4) + BN4.T.dot(M).dot(BN4) +
                                0.5 * BL4.T.dot(M).dot(AP4).dot(H4).dot(delta) +
                                0.5 * BN4.T.dot(M).dot(AP4).dot(H4).dot(delta)))

            # create the stress matrix
            
            # create the stress stiffness matrix
            KS = self._width * (detJ1 * w[0] * H1.T.dot(st).dot(AP1) +
                                detJ2 * w[1] * H2.T.dot(st).dot(AP2) +
                                detJ3 * w[2] * H3.T.dot(st).dot(AP3) +
                                detJ4 * w[3] * H4.T.dot(st).dot(AP4))            
            
            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS

        else:  # gaussPts = 7
            # 'natural' weights of the element
            w = np.array([0.6257871064166934, 0.3016384608809768,
                            0.3169910433902452, 0.3155445150066620,
                            0.2958801959111726, 0.2575426306970870,
                            0.2642573384350463])

            jacob1 = self._shapeFns[1].jacobian(x01, x02, x03, x04, x05)
            jacob2 = self._shapeFns[2].jacobian(x01, x02, x03, x04, x05)
            jacob3 = self._shapeFns[3].jacobian(x01, x02, x03, x04, x05)
            jacob4 = self._shapeFns[4].jacobian(x01, x02, x03, x04, x05)
            jacob5 = self._shapeFns[5].jacobian(x01, x02, x03, x04, x05)
            jacob6 = self._shapeFns[6].jacobian(x01, x02, x03, x04, x05)
            jacob7 = self._shapeFns[7].jacobian(x01, x02, x03, x04, x05)

            # determinant of the Jacobian matrix
            detJ1 = det(jacob1)
            detJ2 = det(jacob2)
            detJ3 = det(jacob3)            
            detJ4 = det(jacob4)
            detJ5 = det(jacob5)
            detJ6 = det(jacob6)            
            detJ7 = det(jacob7)
            
            # create the linear Bmatrix
            BL1 = self._shapeFns[1].BLinear(x1, x2, x3, x4, x5) * inv(detJ1)
            BL2 = self._shapeFns[2].BLinear(x1, x2, x3, x4, x5) * inv(detJ2)         
            BL3 = self._shapeFns[3].BLinear(x1, x2, x3, x4, x5) * inv(detJ3)
            BL4 = self._shapeFns[4].BLinear(x1, x2, x3, x4, x5) * inv(detJ4)
            BL5 = self._shapeFns[5].BLinear(x1, x2, x3, x4, x5) * inv(detJ5)      
            BL6 = self._shapeFns[6].BLinear(x1, x2, x3, x4, x5) * inv(detJ6)
            BL7 = self._shapeFns[7].BLinear(x1, x2, x3, x4, x5) * inv(detJ7)

            # the consistent mass matrix for 7 Gauss points
            KL = self._width * (detJ1 * w[0] * BL1.T.dot(M).dot(BL1) +
                                detJ2 * w[1] * BL2.T.dot(M).dot(BL2) +
                                detJ3 * w[2] * BL3.T.dot(M).dot(BL3) +
                                detJ4 * w[3] * BL4.T.dot(M).dot(BL4) +
                                detJ5 * w[4] * BL5.T.dot(M).dot(BL5) +
                                detJ6 * w[5] * BL6.T.dot(M).dot(BL6) +
                                detJ7 * w[6] * BL7.T.dot(M).dot(BL7))

            # create the nonlinear Bmatrix
            BN1 = self._shapeFns[1].BNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BN2 = self._shapeFns[2].BNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BN3 = self._shapeFns[3].BNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BN4 = self._shapeFns[4].BNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BN5 = self._shapeFns[5].BNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BN6 = self._shapeFns[6].BNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BN7 = self._shapeFns[7].BNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)

            
            # create the H matrix
            H1 = self._shapeFns[1].Hmatrix(x1, x2, x3, x4, x5)
            H2 = self._shapeFns[2].Hmatrix(x1, x2, x3, x4, x5)
            H3 = self._shapeFns[3].Hmatrix(x1, x2, x3, x4, x5)
            H4 = self._shapeFns[4].Hmatrix(x1, x2, x3, x4, x5)
            H5 = self._shapeFns[5].Hmatrix(x1, x2, x3, x4, x5)
            H6 = self._shapeFns[6].Hmatrix(x1, x2, x3, x4, x5)
            H7 = self._shapeFns[7].Hmatrix(x1, x2, x3, x4, x5)

            # create the AP matrix
            AP1 = self._shapeFns[1].APmatrix(x1, x2, x3, x4, x5)
            AP2 = self._shapeFns[2].APmatrix(x1, x2, x3, x4, x5)
            AP3 = self._shapeFns[3].APmatrix(x1, x2, x3, x4, x5)
            AP4 = self._shapeFns[4].APmatrix(x1, x2, x3, x4, x5)
            AP5 = self._shapeFns[5].APmatrix(x1, x2, x3, x4, x5)
            AP6 = self._shapeFns[6].APmatrix(x1, x2, x3, x4, x5)
            AP7 = self._shapeFns[7].APmatrix(x1, x2, x3, x4, x5)
            
            # the nonlinear stiffness matrix for 1 Gauss point
            KN = self._width * (detJ1 * w[0] * (BL1.T.dot(M).dot(BN1) + 
                                BN1.T.dot(M).dot(BL1) + BN1.T.dot(M).dot(BN1) +
                                0.5 * BL1.T.dot(M).dot(AP1).dot(H1).dot(delta) +
                                0.5 * BN1.T.dot(M).dot(AP1).dot(H1).dot(delta)) +
                                detJ2 * w[1] * (BL2.T.dot(M).dot(BN2) +
                                BN2.T.dot(M).dot(BL2) + BN2.T.dot(M).dot(BN2) +
                                0.5 * BL2.T.dot(M).dot(AP2).dot(H2).dot(delta) +
                                0.5 * BN2.T.dot(M).dot(AP2).dot(H2).dot(delta)) +
                                detJ3 * w[2] * (BL3.T.dot(M).dot(BN3) +
                                BN3.T.dot(M).dot(BL3) + BN3.T.dot(M).dot(BN3) +
                                0.5 * BL3.T.dot(M).dot(AP3).dot(H3).dot(delta) +
                                0.5 * BN3.T.dot(M).dot(AP3).dot(H3).dot(delta)) +
                                detJ4 * w[3] * (BL4.T.dot(M).dot(BN4) +
                                BN4.T.dot(M).dot(BL4) + BN4.T.dot(M).dot(BN4) +
                                0.5 * BL4.T.dot(M).dot(AP4).dot(H4).dot(delta) +
                                0.5 * BN4.T.dot(M).dot(AP4).dot(H4).dot(delta)) +
                                detJ5 * w[4] * (BL5.T.dot(M).dot(BN5) +
                                BN5.T.dot(M).dot(BL5) + BN5.T.dot(M).dot(BN5) +
                                0.5 * BL5.T.dot(M).dot(AP5).dot(H5).dot(delta) +
                                0.5 * BN5.T.dot(M).dot(AP5).dot(H5).dot(delta)) +
                                detJ6 * w[5] * (BL6.T.dot(M).dot(BN6) +
                                BN6.T.dot(M).dot(BL6) + BN6.T.dot(M).dot(BN6) +
                                0.5 * BL6.T.dot(M).dot(AP6).dot(H6).dot(delta) +
                                0.5 * BN6.T.dot(M).dot(AP6).dot(H6).dot(delta)) +
                                detJ7 * w[6] * (BL7.T.dot(M).dot(BN7) +
                                BN7.T.dot(M).dot(BL7) + BN7.T.dot(M).dot(BN7) +
                                0.5 * BL7.T.dot(M).dot(AP7).dot(H7).dot(delta) +
                                0.5 * BN7.T.dot(M).dot(AP7).dot(H7).dot(delta)))          

            # create the stress matrix
            
            # create the stress stiffness matrix            
            KS = self._width * (detJ1 * w[0] * H1.T.dot(st).dot(AP1) +
                                detJ2 * w[1] * H2.T.dot(st).dot(AP2) +
                                detJ3 * w[2] * H3.T.dot(st).dot(AP3) +
                                detJ4 * w[3] * H4.T.dot(st).dot(AP4) +
                                detJ5 * w[4] * H5.T.dot(st).dot(AP5) +
                                detJ6 * w[5] * H6.T.dot(st).dot(AP6) +
                                detJ7 * w[6] * H7.T.dot(st).dot(AP7))   
            
            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS                        
        return stiffT

    def forcingFunction(self):
        return
