#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import chord
from ridder import findRoot
import math as m
from membranes import membrane
import numpy as np
from shapeFnPentagons import shapeFunction
import spin as spinMtx


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
__update__ = "09-27-2019"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

r"""

Change in version "1.3.0":

methods

    detjac = p.detJ(gaussPt, state)
        gaussPt  the  Gauss point  for which the Jacobian associates
        state    the configuration for which the Jacobian associates
    returns
        detjac   the determinant of the Jacobian matrix

    massM = p.massMatrix(gaussPt, rho, width)
        gaussPt  the Gauss point for which the mass matrix is to be supplied
        rho      the mass density with units of mass per unit volume
        width    the membrane thickness
    returns
        massM    a 10x10 mass matrix for the pentagon associated with 'gaussPt'

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

    Fields needed to construct finite element representations

    sf = p.shapeFunction(self, gaussPt):
        returns the shape function associated with the specified Gauss point

    massM = p.massMatrix(rho, width)
        rho      the mass density with units of mass per unit volume
        width    the membrane thickness
    returns
        massM    a 10x10 mass matrix for the pentagon

    The fundamental fields of kinematics

    gMtx = p.G(gaussPt, state)
        returns 2x2 matrix describing the displacement gradient (it is not
        relabeled) of the pentagon at 'gaussPt' in configuration 'state'

    fMtx = p.F(gaussPt, state)
        returns 2x2 matrix describing the deformation gradient (it is not
        relabeled) of the pentagon at 'gaussPt' in configuration 'state'

    Gram-Schmidt factorization of the deformation gradient

    qMtx = p.Q(gaussPt, state)
        returns 2x2 reindexing matrix applied to the deformation gradient
        prior to its Gram-Schmidt decomposition at 'gaussPt' in
        configuration 'state'

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
        returns the differential change in dilation at 'gaussPt' in
        configuration 'state'

    dEpsilon = p.dSqueeze(gaussPt, state)
        returns the differential change in squeeze at 'gaussPt' in
        configuration 'state'

    dGamma = p.dShear(gaussPt, state)
        returns the differential change in shear at 'gaussPt' in configuration
        'state'

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
            raise RuntimeError(
                   'Error: chord1 passed to pentagon constructor was invalid.')
        if not isinstance(chord2, chord):
            raise RuntimeError(
                   'Error: chord2 passed to pentagon constructor was invalid.')
        if not isinstance(chord3, chord):
            raise RuntimeError(
                   'Error: chord3 passed to pentagon constructor was invalid.')
        if not isinstance(chord4, chord):
            raise RuntimeError(
                   'Error: chord4 passed to pentagon constructor was invalid.')
        if not isinstance(chord5, chord):
            raise RuntimeError(
                   'Error: chord5 passed to pentagon constructor was invalid.')

        # verify that the chords connect to form a pentagon
        c1v1, c1v2 = chord1.vertexNumbers()
        c2v1, c2v2 = chord2.vertexNumbers()
        c3v1, c3v2 = chord3.vertexNumbers()
        c4v1, c4v2 = chord4.vertexNumbers()
        c5v1, c5v2 = chord5.vertexNumbers()
        if not (c1v1 == c2v1 or c1v1 == c2v2 or c1v2 == c2v1 or c1v2 == c2v2):
            raise RuntimeError(
                       'Error: chord1 and chord2 do not have a common vertex.')
        if not (c2v1 == c3v1 or c2v1 == c3v2 or c2v2 == c3v1 or c2v2 == c3v2):
            raise RuntimeError(
                       'Error: chord2 and chord3 do not have a common vertex.')
        if not (c3v1 == c4v1 or c3v1 == c4v2 or c3v2 == c4v1 or c3v2 == c4v2):
            raise RuntimeError(
                       'Error: chord3 and chord4 do not have a common vertex.')
        if not (c4v1 == c5v1 or c4v1 == c5v2 or c4v2 == c5v1 or c4v2 == c5v2):
            raise RuntimeError(
                       'Error: chord4 and chord5 do not have a common vertex.')
        if not (c1v1 == c5v1 or c1v1 == c5v2 or c1v2 == c5v1 or c1v2 == c5v2):
            raise RuntimeError(
                       'Error: chord1 and chord5 do not have a common vertex.')

        # check the stepsize
        if h > np.finfo(float).eps:
            self._h = float(h)
        else:
            raise RuntimeError(
                     "Error: stepsize in pentagon constructor isn't positive.")

        # check the number of Gauss points to use
        if gaussPts == 1 or gaussPts == 4 or gaussPts == 7:
            self._gaussPts = gaussPts
        else:
            raise RuntimeError('Error: {} Gauss points were specified in ' +
                               'the pentagon constructor; must be 1, 4 or 7.'
                               .format(gaussPts))

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
            raise RuntimeError(
                       'Error: there are not 5 unique chords in the pentagon.')

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
            raise RuntimeError(
                     'Error: there are not 5 unique vertices in the pentagon.')

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
            raise RuntimeError(
                        "Error: unknown state {} in call to pentagon.toString."
                        .format(str(state)))
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
            raise RuntimeError(
                         'Error: the requested chord {} is not in pentagon {}.'
                         .format(number, self._number))

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
            raise RuntimeError(
                        'Error: the requested vertex {} is not in pentagon {}.'
                        .format(number, self._number))

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
        v1x = n1x * x1 + n1y * y1 + n1z * z1
        v1y = n2x * x1 + n2y * y1 + n2z * z1
        v2x = n1x * x2 + n1y * y2 + n1z * z2
        v2y = n2x * x2 + n2y * y2 + n2z * z2
        v3x = n1x * x3 + n1y * y3 + n1z * z3
        v3y = n2x * x3 + n2y * y3 + n2z * z3
        v4x = n1x * x4 + n1y * y4 + n1z * z4
        v4y = n2x * x4 + n2y * y4 + n2z * z4
        v5x = n1x * x5 + n1y * y5 + n1z * z5
        v5y = n2x * x5 + n2y * y5 + n2z * z5

        # z offsets for the pentagonal plane (which should all be the same)
        v1z = n3x * x1 + n3y * y1 + n3z * z1
        v2z = n3x * x2 + n3y * y2 + n3z * z2
        v3z = n3x * x3 + n3y * y3 + n3z * z3
        v4z = n3x * x4 + n3y * y4 + n3z * z4
        v5z = n3x * x5 + n3y * y5 + n3z * z5

        # determine the area of this irregular pentagon
        self._An = (v1x * v2y - v2x * v1y +
                    v2x * v3y - v3x * v2y +
                    v3x * v4y - v4x * v3y +
                    v4x * v5y - v5x * v4y +
                    v5x * v1y - v1x * v5y) / 2.0
        # the area will be positive if the vertices index counter clockwise

        # determine the centroid of this pentagon in pentagonal coordinates
        cx = ((v1x + v2x) * (v1x * v2y - v2x * v1y) +
              (v2x + v3x) * (v2x * v3y - v3x * v2y) +
              (v3x + v4x) * (v3x * v4y - v4x * v3y) +
              (v4x + v5x) * (v4x * v5y - v5x * v4y) +
              (v5x + v1x) * (v5x * v1y - v1x * v5y)) / (6.0 * self._An)
        cy = ((v1y + v2y) * (v1x * v2y - v2x * v1y) +
              (v2y + v3y) * (v2x * v3y - v3x * v2y) +
              (v3y + v4y) * (v3x * v4y - v4x * v3y) +
              (v4y + v5y) * (v4x * v5y - v5x * v4y) +
              (v5y + v1y) * (v5x * v1y - v1x * v5y)) / (6.0 * self._An)
        cz = (v1z + v2z + v3z + v4z + v5z) / 5.0

        # rotate this centroid back into the reference coordinate system
        self._centroidXn = n1x * cx + n2x * cy + n3x * cz
        self._centroidYn = n1y * cx + n2y * cy + n3y * cz
        self._centroidZn = n1z * cx + n2z * cy + n3z * cz

        # Determine the deformation gradient for this irregular pentagon

        # current vertex coordinates in pentagonal frame of reference
        x1 = (v1x, v1y)
        x2 = (v2x, v2y)
        x3 = (v3x, v3y)
        x4 = (v4x, v4y)
        x5 = (v5x, v5y)

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
                raise RuntimeError(
                            "Error: unknown state {} in call to pentagon.area."
                            .format(state))
        else:
            raise RuntimeError(
                            "Error: unknown state {} in call to pentagon.area."
                            .format(str(state)))

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
                raise RuntimeError(
                       "Error: unknown state {} in call pentagon.arealStretch."
                       .format(state))
        else:
            raise RuntimeError(
                    "Error: unknown state {} in call to pentagon.arealStretch."
                    .format(str(state)))

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
                raise RuntimeError(
                        "Error: unknown state {} in call pentagon.arealStrain."
                        .format(state))
        else:
            raise RuntimeError(
                     "Error: unknown state {} in call to pentagon.arealStrain."
                     .format(str(state)))

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
                raise RuntimeError(
                            "Error: unknown state {} in pentagon.dArealStrain."
                            .format(state))
        else:
            raise RuntimeError(
                            "Error: unknown state {} in pentagon.dArealStrain."
                            .format(str(state)))

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
                raise RuntimeError(
                          "Error: unknown state {} in call to pentagon.normal."
                          .format(state))
        else:
            raise RuntimeError(
                          "Error: unknown state {} in call to pentagon.normal."
                          .format(str(state)))
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
                raise RuntimeError(
                         "Error: unknown state {} in call to pentagon.dNormal."
                         .format(state))
        else:
            raise RuntimeError(
                         "Error: unknown state {} in call to pentagon.dNormal."
                         .format(str(state)))
        return np.array([dnx, dny, dnz])

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
                raise RuntimeError(
                        "Error: unknown state {} in call to pentagon.centroid."
                        .format(state))
        else:
            raise RuntimeError(
                        "Error: unknown state {} in call to pentagon.centroid."
                        .format(str(state)))
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
                raise RuntimeError(
                        "Error: unknown state {} in call to pentagon.velocity."
                        .format(state))
        else:
            raise RuntimeError(
                        "Error: unknown state {} in call to pentagon.velocity."
                        .format(str(state)))
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
            raise RuntimeError(
                    "Error: unknown state {} in call to pentagon.acceleration."
                    .format(str(state)))
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
                raise RuntimeError(
                        "Error: unknown state {} in call to pentagon.rotation."
                        .format(state))
        else:
            raise RuntimeError(
                        "Error: unknown state {} in call to pentagon.rotation."
                        .format(str(state)))

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
                raise RuntimeError(
                            "Error: unknown state {} in call to pentagon.spin."
                            .format(state))
        else:
            raise RuntimeError(
                            "Error: unknown state {} in call to pentagon.spin."
                            .format(str(state)))

    def shapeFunction(self, gaussPt):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.shapeFunction and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.shapeFunction and you sent {}."
                                   .format(gaussPt))
            sf = self._shapeFns[gaussPt]
        return sf

    def massMatrix(self, rho, width):
        if rho <= 0.0:
            raise RuntimeError("Mass density rho must be positive, you " +
                               "sent {} to pentagon.massMatrix.".format(rho))
        if width <= 0.0:
            raise RuntimeError("The width sent to pentagon.massMatrix " +
                               "{}; it must be positive.".format(width))

        # assign coordinates at the vertices in the reference configuration
        x01 = (self._v1x0, self._v1y0)
        x02 = (self._v2x0, self._v2y0)
        x03 = (self._v3x0, self._v3y0)
        x04 = (self._v4x0, self._v4y0)
        x05 = (self._v5x0, self._v5y0)

        # determine the mass matrix
        if self._gaussPts == 1:
            # 'natural' weight of the element
            wel = np.array([2.3776412907378837])

            detJ = self._shapeFns[1].detJacobian(x01, x02, x03, x04, x05)

            nn1 = np.dot(np.transpose(self._shapeFns[1].Nmatx),
                         self._shapeFns[1].Nmatx)

            # Integration to get the mass matrix for 1 Gauss point
            mass = rho * width * (detJ * wel[0] * nn1)
        elif self._gaussPts == 4:
            # 'natural' weights of the element
            wel = np.array([0.5449124407446143, 0.6439082046243272,
                            0.5449124407446146, 0.6439082046243275])

            detJ1 = self._shapeFns[1].detJacobian(x01, x02, x03, x04, x05)
            detJ2 = self._shapeFns[2].detJacobian(x01, x02, x03, x04, x05)
            detJ3 = self._shapeFns[3].detJacobian(x01, x02, x03, x04, x05)
            detJ4 = self._shapeFns[4].detJacobian(x01, x02, x03, x04, x05)

            nn1 = np.dot(np.transpose(self._shapeFns[1].Nmatx),
                         self._shapeFns[1].Nmatx)
            nn2 = np.dot(np.transpose(self._shapeFns[2].Nmatx),
                         self._shapeFns[2].Nmatx)
            nn3 = np.dot(np.transpose(self._shapeFns[3].Nmatx),
                         self._shapeFns[3].Nmatx)
            nn4 = np.dot(np.transpose(self._shapeFns[4].Nmatx),
                         self._shapeFns[4].Nmatx)

            # Integration to get the mass matrix for 4 Gauss points
            mass = (rho * width * (detJ1 * wel[0] * nn1 +
                                   detJ2 * wel[1] * nn2 +
                                   detJ3 * wel[2] * nn3 +
                                   detJ4 * wel[3] * nn4))
        else:  # gaussPts = 7
            # 'natural' weights of the element
            wel = np.array([0.6257871064166934, 0.3016384608809768,
                            0.3169910433902452, 0.3155445150066620,
                            0.2958801959111726, 0.2575426306970870,
                            0.2642573384350463])

            detJ1 = self._shapeFns[1].detJacobian(x01, x02, x03, x04, x05)
            detJ2 = self._shapeFns[2].detJacobian(x01, x02, x03, x04, x05)
            detJ3 = self._shapeFns[3].detJacobian(x01, x02, x03, x04, x05)
            detJ4 = self._shapeFns[4].detJacobian(x01, x02, x03, x04, x05)
            detJ5 = self._shapeFns[5].detJacobian(x01, x02, x03, x04, x05)
            detJ6 = self._shapeFns[6].detJacobian(x01, x02, x03, x04, x05)
            detJ7 = self._shapeFns[7].detJacobian(x01, x02, x03, x04, x05)

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

            # Integration to get the mass Matrix for 7 Gauss points
            mass = (rho * width * (detJ1 * wel[0] * nn1 +
                                   detJ2 * wel[1] * nn2 +
                                   detJ3 * wel[2] * nn3 +
                                   detJ4 * wel[3] * nn4 +
                                   detJ5 * wel[4] * nn5 +
                                   detJ6 * wel[5] * nn6 +
                                   detJ7 * wel[6] * nn7))
        return mass

    # displacement gradient at a Gauss point
    def G(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.G and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.G and you sent {}."
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
                raise RuntimeError(
                               "Error: unknown state {} in call to pentagon.G."
                               .format(state))
        else:
            raise RuntimeError("Error: unknown state {} in call to pentagon.G."
                               .format(str(state)))

    # deformation gradient at a Gauss point
    def F(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.F and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.F and you sent {}."
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
                raise RuntimeError(
                               "Error: unknown state {} in call to pentagon.F."
                               .format(state))
        else:
            raise RuntimeError("Error: unknown state {} in call to pentagon.F."
                               .format(str(state)))

    # the orthogonal matrix that relabels the coordinate directions
    def Q(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.Q and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.Q and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].Q(state)

    # the orthogonal matrix from a Gram-Schmidt factorization of (relabeled) F
    def R(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.R and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.R and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].R(state)

    # a skew-symmetric matrix for the spin associated with R
    def Omega(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.Omega and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.Omega and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].spin(state)

    # Laplace stretch from a Gram-Schmidt factorization of (relabeled) F
    def U(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.U and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.U and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].U(state)

    # inverse Laplace stretch from Gram-Schmidt factorization of (relabeled) F
    def UInv(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.UInv and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.UInv and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].UInv(state)

    # differential in the Laplace stretch from Gram-Schmidt factorization of F
    def dU(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.dU and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.dU and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].dU(state)

    # inverse Laplace stretch from Gram-Schmidt factorization of (relabeled) F
    def dUInv(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.dUInv and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.dUInv and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].dUInv(state)

    # physical kinematic attributes at a Gauss point in the pentagon

    def dilation(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.dilation and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.dilation and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].dilation(state)

    def squeeze(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.squeeze and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.squeeze and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].squeeze(state)

    def shear(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.shear and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.shear and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].shear(state)

    def dDilation(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.dDilation and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.dDilation and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].dDilation(state)

    def dSqueeze(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.dSqueeze and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.dSqueeze and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].dSqueeze(state)

    def dShear(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "pentagon.dShear and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to pentagon.dShear and you sent {}."
                                   .format(gaussPt))
        return self._septum[gaussPt].dShear(state)
