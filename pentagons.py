#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ceMembranes import controlMembrane, ceMembrane
from chords import Chord
from pivotIncomingF import Pivot
import meanProperties as mp
import math as m
from membranes import membrane
import numpy as np
from ridder import findRoot
from shapeFnPentagons import ShapeFunction as pentShapeFunction
import spin as spinMtx
from shapeFnChords import ShapeFunction as chordShapeFunction
from gaussQuadChords import GaussQuadrature as chordGaussQuadrature
from gaussQuadPentagons import GaussQuadrature as pentGaussQuadrature


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
__update__ = "12-06-2020"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

r"""

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

    p = pentagon(number, chord1, chord2, chord3, chord4, chord5, h, pentGaussPts)
        number    immutable value that is unique to this pentagon
        chord1    unique edge of the pentagon, an instance of class chord
        chord2    unique edge of the pentagon, an instance of class chord
        chord3    unique edge of the pentagon, an instance of class chord
        chord4    unique edge of the pentagon, an instance of class chord
        chord5    unique edge of the pentagon, an instance of class chord
        h         timestep size between two successive calls to 'advance'
        pentGaussPts  number of Gauss points to be used: must be 1, 4 or 7

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

    w = p.width(state)
        returns the cross-sectional thickness or width of the membrane in
        configuration 'state'

    Geometric fields associated with a pentagonal surface in 3 space

    a = p.area(state)
        returns the area of this irregular pentagon in configuration 'state'

    aLambda = p.arealStretch(state)
            returns the square root of area(state) divided by reference area

    aStrain = p.arealStrain(state)
            returns the logarithm of areal stretch evaluated at 'state'

    daStrain = p.dArealStrain(state)
            returns the time rate of change in areal strain at 'state'

    Material properties that associate with this septum.  Except for the mass
    density, all are drawn randomly from a statistical distribution.

    rho = p.massDensity()
        returns the mass density of the chord (collagen and elastin fibers)

    M1, M2, Me_t, N1, N2, Ne_t, G1, G2, Ge_t = c.matProp()
        returns the constitutive properties for this septal membrane where
            M1, M2, Me_t  pertain to the dilation response
            N1, N2, Ne_t  pertain to the squeeze  response
            G1, G2, Ge_t  pertain to the  shear   response
        where the first in these sets describes the compliant response
        the second in these sets describes the stiff response, and
        the third in these sets establishes the strain of transition
        
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

    p.advance(reindex)
       input
            reindex is an instance of Pivot object from module pivotIncomingF      
        assigns the current location into the previous location, and then it
        assigns the next location into the current location, thereby freezing
        the location of the present next-location in preparation to advance to
        the next step along a solution path

    pMtx12 = c.rotation12(state)
        returns a 2x2 orthogonal matrix that rotates the reference base vectors
        into the set of local base vectors pertaining to 1-2 chord whose axis
        aligns with the 1 direction, while the 2 direction passes through the
        origin of the dodecahedral reference coordinate system.  The returned
        matrix associates with configuration 'state'

    pMtx23 = c.rotation23(state)
        returns a 2x2 orthogonal matrix that rotates the reference base vectors
        into the set of local base vectors pertaining to 2-3 chord whose axis
        aligns with the 1 direction, while the 2 direction passes through the
        origin of the dodecahedral reference coordinate system.  The returned
        matrix associates with configuration 'state'
        
    pMtx34 = c.rotation34(state)
        returns a 2x2 orthogonal matrix that rotates the reference base vectors
        into the set of local base vectors pertaining to 3-4 chord whose axis
        aligns with the 1 direction, while the 2 direction passes through the
        origin of the dodecahedral reference coordinate system.  The returned
        matrix associates with configuration 'state'

    pMtx45 = c.rotation45(state)
        returns a 2x2 orthogonal matrix that rotates the reference base vectors
        into the set of local base vectors pertaining to 4-5 chord whose axis
        aligns with the 1 direction, while the 2 direction passes through the
        origin of the dodecahedral reference coordinate system.  The returned
        matrix associates with configuration 'state'

    pMtx51 = c.rotation51(state)
        returns a 2x2 orthogonal matrix that rotates the reference base vectors
        into the set of local base vectors pertaining to 5-1 chord whose axis
        aligns with the 1 direction, while the 2 direction passes through the
        origin of the dodecahedral reference coordinate system.  The returned
        matrix associates with configuration 'state'        

    [nx, ny, nz] = p.normal(state)
        returns the unit normal to this pentagon in configuration 'state'

    [dnx, dny, dnz] = p.dNormal(state)
        returns the time rate of change of the unit normal at 'state'

    Kinematic fields associated with the centroid of a pentagon in 3 space

    [cx, cy, cz] = p.centroid(state)
        returns centroid of this irregular pentagon in configuration 'state'

    [ux, uy, uz] = p.centroidDisplacement(reindex, state)
        returns the displacement at the centroid in configuration 'state'
            
    [vx, vy, vz] = p.centroidVelocity(reindex, state)
        returns the velocity at the centroid in configuration 'state'

    [ax, ay, az] = p.centroidAcceleration(reindex, state)
        returns the acceleration at the centroid in configuration 'state'

    Dmtx1 = sf.dDisplacement1(reindex)
       input
            reindex is an instance of Pivot object from module pivotIncomingF      
       output
            Dmtx1 is change in displacement ( dA1 = L1 * D1 ) in the contribution 
            to the first nonlinear strain

    Dmtx2 = sf.dDisplacement2(reindex)
       input
            reindex is an instance of Pivot object from module pivotIncomingF      
       output
            Dmtx2 is change in displacement ( dA2 = L2 * D2 ) in the contribution 
            to the second nonlinear strain

    Dmtx3 = sf.dDisplacement3(reindex)
       input
            reindex is an instance of Pivot object from module pivotIncomingF      
       output
            Dmtx3 is change in displacement ( dA3 = L3 * D3 ) in the contribution 
            to the third nonlinear strain
            
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

    gMtx = p.G(pentGaussPt, state)
        returns 2x2 matrix describing the displacement gradient (it is not
        relabeled) of the pentagon at 'pentGaussPt' in configuration 'state'

    fMtx = p.F(pentGaussPt, state)
        returns 2x2 matrix describing the deformation gradient (it is not
        relabeled) of the pentagon at 'pentGaussPt' in configuration 'state'

    lMtx = c.L(pentGaussPt, state)
        returns the velocity gradient at the specified Gauss point for the
        specified configuration

    Reindexing of the coordinate used in the membrane analysis at a Gauss point

    qMtx = p.Q(pentGaussPt, state)
        returns 2x2 reindexing matrix applied to the deformation gradient
        prior to its Gram-Schmidt decomposition at 'pentGaussPt' in
        configuration 'state'

    Gram-Schmidt factorization of a reindexed deformation gradient

    rMtx = p.R(pentGaussPt, state)
        returns 2x2 rotation matrix 'Q' derived from a QR decomposition of the
        reindexed deformation gradient at 'pentGaussPt' in configuration 'state'

    omega = p.Omega(pentGaussPt, state)
        returns 2x2 spin matrix caused by planar deformation, i.e., dR R^t,
        at 'pentGaussPt' in configuration 'state'

    uMtx = p.U(pentGaussPt, state)
        returns 2x2 Laplace stretch 'R' derived from a QR decomposition of the
        reindexed deformation gradient at 'pentGaussPt' in configuration 'state'

    uInvMtx = p.UInv(pentGaussPt, state)
        returns 2x2 inverse Laplace stretch derived from a QR decomposition of
        reindexed deformation gradient at 'pentGaussPt' in configuration 'state'

    duMtx = p.dU(pentGaussPt, state)
        returns 2x2 matrix for differential change in Laplace stretch at
        'pentGaussPt' in configuration 'state'

    duInvMtx = p.dUInv(pentGaussPt, state)
        returns 2x2 matrix for differential change in the inverse Laplace
        stretch at 'pentGaussPt' in configuration 'state'

    The extensive thermodynamic variables for a membrane and their rates
    accuired from a reindexed deformation gradient of the membrane

    xi = p.dilation(pentGaussPt, state)
        returns the planar dilation derived from a QR decomposition of the
        reindexed deformation gradient in configuration 'state'

    epsilon = p.squeeze(pentGaussPt, state)
        returns the planar squeeze derived from a QR decomposition of the
        reindexed deformation gradient at 'pentGaussPt' in configuration 'state'

    gamma = p.shear(pentGaussPt, state)
        returns the planar shear derived from a QR decomposition of the
        reindexed deformation gradient at 'pentGaussPt' in configuration 'state'

    dXi = p.dDilation(pentGaussPt, state)
        returns the differential change in dilation acquired from the reindexed
        deformation gradient for the membrane model at 'pentGaussPt' in
        configuration 'state'

    dEpsilon = p.dSqueeze(pentGaussPt, state)
        returns the differential change in squeeze acquired from the reindexed
        deformation gradient for the membrane model at 'pentGaussPt' in
        configuration 'state'

    dGamma = p.dShear(pentGaussPt, state)
        returns the differential change in shear acquired from the reindexed
        deformation gradient for the membrane model at 'pentGaussPt' in
        configuration 'state'

    Fields needed to construct finite element representations

    Psf = p.pentShapeFunction(pentGaussPt):
        returns the shape function associated with the specified Gauss point 
        for pentagon

    csf = p.chordShapeFunction(chordGaussPt):
        returns the shape function associated with the specified Gauss point 
        for chord

    pgq = p.pentGaussQuadrature():
        returns the gauss Gauss quadrature rule to be used for pentaon

    cgq = p.chordGaussQuadrature():
        returns the gauss Gauss quadrature rule to be used for chord
        
    mMtx = p.massMatrix()
        returns an average of the lumped and consistent mass matrices (ensures
        the mass matrix is not singular) for the chosen number of Gauss points
        for a pentagon whose mass density, rho, and whose thickness, width, are
        specified.

    cMtx = p.tangentStiffnessMtxC()
        reindex is an instance of Pivot object from module pivotIncomingF
        returns a tangent stiffness matrix for the chosen number of Gauss
        points.
        
    kMtx = p.secantStiffnessMtxK(reindex)
        reindex is an instance of Pivot object from module pivotIncomingF
        returns a secant stiffness matrix for the chosen number of Gauss
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

    def __init__(self, number, chord1, chord2, chord3, chord4, chord5, h):
        self._number = int(number)

        # verify the input
        if not isinstance(chord1, Chord):
            raise RuntimeError('chord1 passed to the pentagon constructor ' +
                               'was invalid.')
        if not isinstance(chord2, Chord):
            raise RuntimeError('chord2 passed to the pentagon constructor ' +
                               'was invalid.')
        if not isinstance(chord3, Chord):
            raise RuntimeError('chord3 passed to the pentagon constructor ' +
                               'was invalid.')
        if not isinstance(chord4, Chord):
            raise RuntimeError('chord4 passed to the pentagon constructor ' +
                               'was invalid.')
        if not isinstance(chord5, Chord):
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
            
        # assign the Gauss quadrature rule to be used for pentaon
        self._pgq = pentGaussQuadrature()
        
        
        # assign the Gauss quadrature rule to be used for chord
        self._cgq = chordGaussQuadrature()
        

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

        # get the vertex coordinates in the reference configuration
        x1 = v1.coordinates('ref')
        x2 = v2.coordinates('ref')
        x3 = v3.coordinates('ref')
        x4 = v4.coordinates('ref')
        x5 = v5.coordinates('ref')

        # base vector 1: connects the two shoulders of a pentagon
        x = x5[0] - x2[0]
        y = x5[1] - x2[1]
        z = x5[2] - x2[2]
        mag = m.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 2: goes from the apex to a point along its base

        # establish the unit vector for the base of the pentagon
        x = x4[0] - x3[0]
        y = x4[1] - x3[1]
        z = x4[2] - x3[2]
        mag = m.sqrt(x * x + y * y + z * z)
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the base of
        # the pentagon that results in a vector n2 which is normal to n1

        def getDelta(delta):
            nx = x1[0] - (x3[0] + delta * ex)
            ny = x1[1] - (x3[1] + delta * ey)
            nz = x1[2] - (x3[2] + delta * ez)
            # when the dot product is zero then the two vectors are orthogonal
            n1Dotn2 = n1x * nx + n1y * ny + n1z * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * mag
        deltaH = 5.0 * mag
        delta = findRoot(deltaL, deltaH, getDelta)

        # create base vector 2
        x = x1[0] - (x3[0] + delta * ex)
        y = x1[1] - (x3[1] + delta * ey)
        z = x1[2] - (x3[2] + delta * ez)
        mag = m.sqrt(x * x + y * y + z * z)
        n2x = x / mag
        n2y = y / mag
        n2z = z / mag

        # base vector 3 is obtained through the cross product
        # it is normal to the pentagon and points outward from the dodecahedron
        n3x = n1y * n2z - n1z * n2y
        n3y = n1z * n2x - n1x * n2z
        n3z = n1x * n2y - n1y * n2x

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
        self._v1x0 = n1x * x1[0] + n1y * x1[1] + n1z * x1[2]
        self._v1y0 = n2x * x1[0] + n2y * x1[1] + n2z * x1[2]
        self._v2x0 = n1x * x2[0] + n1y * x2[1] + n1z * x2[2]
        self._v2y0 = n2x * x2[0] + n2y * x2[1] + n2z * x2[2]
        self._v3x0 = n1x * x3[0] + n1y * x3[1] + n1z * x3[2]
        self._v3y0 = n2x * x3[0] + n2y * x3[1] + n2z * x3[2]
        self._v4x0 = n1x * x4[0] + n1y * x4[1] + n1z * x4[2]
        self._v4y0 = n2x * x4[0] + n2y * x4[1] + n2z * x4[2]
        self._v5x0 = n1x * x5[0] + n1y * x5[1] + n1z * x5[2]
        self._v5y0 = n2x * x5[0] + n2y * x5[1] + n2z * x5[2]

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
        self._v1z = n3x * x1[0] + n3y * x1[1] + n3z * x1[2]
        self._v2z = n3x * x2[0] + n3y * x2[1] + n3z * x2[2]
        self._v3z = n3x * x3[0] + n3y * x3[1] + n3z * x3[2]
        self._v4z = n3x * x4[0] + n3y * x4[1] + n3z * x4[2]
        self._v5z = n3x * x5[0] + n3y * x5[1] + n3z * x5[2]
        self._vz0 = ((self._v1z + self._v2z + self._v3z + self._v4z +
                      self._v5z) / 5.0)

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
        self._cz0 = self._vz0

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
        
       
        # create the rotation matrices for all chords of the pentagon
        self._Pr2D12 = np.identity(2, dtype=float)
        self._Pp2D12 = np.identity(2, dtype=float)
        self._Pc2D12 = np.identity(2, dtype=float)
        self._Pn2D12 = np.identity(2, dtype=float)
        
        self._Pr2D23 = np.identity(2, dtype=float)
        self._Pp2D23 = np.identity(2, dtype=float)
        self._Pc2D23 = np.identity(2, dtype=float)
        self._Pn2D23 = np.identity(2, dtype=float)
        
        self._Pr2D34 = np.identity(2, dtype=float)
        self._Pp2D34 = np.identity(2, dtype=float)
        self._Pc2D34 = np.identity(2, dtype=float)
        self._Pn2D34 = np.identity(2, dtype=float)
        
        self._Pr2D45 = np.identity(2, dtype=float)
        self._Pp2D45 = np.identity(2, dtype=float)
        self._Pc2D45 = np.identity(2, dtype=float)
        self._Pn2D45 = np.identity(2, dtype=float)
        
        self._Pr2D51 = np.identity(2, dtype=float)
        self._Pp2D51 = np.identity(2, dtype=float)
        self._Pc2D51 = np.identity(2, dtype=float)
        self._Pn2D51 = np.identity(2, dtype=float)

        # initialize the 1-2 chordal lengths for all configurations
        x1 = v1.coordinates('ref')
        x2 = v2.coordinates('ref')
        L120 = m.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2 
                      + (x2[2] - x1[2])**2)
        self._L120 = L120
        self._L12p = L120
        self._L12c = L120
        self._L12n = L120
        
        # initialize the 2-3 chordal lengths for all configurations
        x2 = v2.coordinates('ref')
        x3 = v3.coordinates('ref')
        L230 = m.sqrt((x3[0] - x2[0])**2 + (x3[1] - x2[1])**2 
                      + (x3[2] - x2[2])**2)
        self._L230 = L230
        self._L23p = L230
        self._L23c = L230
        self._L23n = L230
        
        # initialize the 3-4 chordal lengths for all configurations
        x3 = v3.coordinates('ref')
        x4 = v4.coordinates('ref')
        L340 = m.sqrt((x4[0] - x3[0])**2 + (x4[1] - x3[1])**2 
                      + (x4[2] - x3[2])**2)
        self._L340 = L340
        self._L34p = L340
        self._L34c = L340
        self._L34n = L340
        
        # initialize the 4-5 chordal lengths for all configurations
        x4 = v4.coordinates('ref')
        x5 = v5.coordinates('ref')
        L450 = m.sqrt((x5[0] - x4[0])**2 + (x5[1] - x4[1])**2 
                      + (x5[2] - x4[2])**2)
        self._L450 = L450
        self._L45p = L450
        self._L45c = L450
        self._L45n = L450
        
        # initialize the 5-1 chordal lengths for all configurations
        x5 = v5.coordinates('ref')
        x1 = v1.coordinates('ref')
        L510 = m.sqrt((x5[0] - x1[0])**2 + (x5[1] - x1[1])**2 
                      + (x5[2] - x1[2])**2)
        self._L510 = L510
        self._L51p = L510
        self._L51c = L510
        self._L51n = L510
                
        # determine the rotation matrix for 1-2 chord 
        # base vector 1: aligns with the axis of the 1-2 chord
        x12 = x2[0] - x1[0]
        y12 = x2[1] - x1[1]
        z12 = x2[2] - x1[2]
        mag12 = m.sqrt(x12 * x12 + y12 * y12  + z12 * z12)
        n1x12 = x12 / mag12
        n1y12 = y12 / mag12
        n1z12 = z12 / mag12

        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1[0] + x2[0]) / 2.0
        y = (x1[1] + x2[1]) / 2.0
        z = (x1[2] + x2[2]) / 2.0
        mag = m.sqrt(x * x + y * y  + z * z)
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta12(delta12):
            nx = ex + delta12 * n1x12
            ny = ey + delta12 * n1y12
            nz = ez + delta12 * n1z12
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x12 * nx + n1y12 * ny + n1z12 * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L120
        deltaH = 4.0 * self._L120
        delta12 = findRoot(deltaL, deltaH, getDelta12)

        # create base vector 2 (the radial vector out to the chord)
        x = ex + delta12 * n1x12
        y = ey + delta12 * n1y12
        z = ez + delta12 * n1z12
        mag = m.sqrt(x * x + y * y + z * z)
        n2x12 = x / mag
        n2y12 = y / mag   
        n2z12 = z / mag
        
        # create the rotation matrix from dodecahedral to 1-2 chordal coordinates
        self._Pr2D12[0, 0] = n1x12
        self._Pr2D12[0, 1] = n2x12
        self._Pr2D12[1, 0] = n1y12
        self._Pr2D12[1, 1] = n2y12
        self._Pp2D12[:, :] = self._Pr2D12[:, :]
        self._Pc2D12[:, :] = self._Pr2D12[:, :]
        self._Pn2D12[:, :] = self._Pr2D12[:, :]

        # determine vertice coordinates in the chordal frame of reference
        self._v1x120 = n1x12 * x1[0] + n1y12 * x1[1] + n1z12 * x1[2] 
        self._v1y120 = n2x12 * x1[0] + n2y12 * x1[1] + n2z12 * x1[2]
        self._v2x120 = n1x12 * x2[0] + n1y12 * x2[1] + n1z12 * x2[2]
        self._v2y120 = n2x12 * x2[0] + n2y12 * x2[1] + n2z12 * x2[2]

        self._v1x12 = self._v1x120 
        self._v1y12 = self._v1y120 
        self._v2x12 = self._v2x120 
        self._v2y12 = self._v2y120 
        
                
        # determine the rotation matrix for 2-3 chord 
        # base vector 1: aligns with the axis of the 2-3 chord
        x23 = x3[0] - x2[0]
        y23 = x3[1] - x2[1]
        z23 = x3[2] - x2[2]
        mag23 = m.sqrt(x23 * x23 + y23 * y23 + z23 * z23)
        n1x23 = x23 / mag23
        n1y23 = y23 / mag23
        n1z23 = z23 / mag23

        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x3[0] + x2[0]) / 2.0
        y = (x3[1] + x2[1]) / 2.0
        z = (x3[2] + x2[2]) / 2.0
        mag = m.sqrt(x * x + y * y  + z * z )
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta23(delta23):
            nx = ex + delta23 * n1x23
            ny = ey + delta23 * n1y23
            nz = ez + delta23 * n1z23
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x23 * nx + n1y23 * ny + n1z23 * nz 
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L230
        deltaH = 4.0 * self._L230
        delta23 = findRoot(deltaL, deltaH, getDelta23)

        # create base vector 2 (the radial vector out to the chord)
        x = ex + delta23 * n1x23
        y = ey + delta23 * n1y23
        z = ez + delta23 * n1z23
        mag = m.sqrt(x * x + y * y + z * z)
        n2x23 = x / mag
        n2y23 = y / mag 
        n2z23 = z / mag          
                
        # create the rotation matrix from dodecahedral to 2-3 chordal coordinates
        self._Pr2D23[0, 0] = n1x23
        self._Pr2D23[0, 1] = n2x23
        self._Pr2D23[1, 0] = n1y23
        self._Pr2D23[1, 1] = n2y23
        self._Pp2D23[:, :] = self._Pr2D23[:, :]
        self._Pc2D23[:, :] = self._Pr2D23[:, :]
        self._Pn2D23[:, :] = self._Pr2D23[:, :]

        # determine vertice coordinates in the chordal frame of reference
        self._v2x230 = n1x23 * x2[0] + n1y23 * x2[1]  + n1z23 * x2[2] 
        self._v2y230 = n2x23 * x2[0] + n2y23 * x2[1]  + n2z23 * x2[2]
        self._v3x230 = n1x23 * x3[0] + n1y23 * x3[1]  + n1z23 * x3[2]
        self._v3y230 = n2x23 * x3[0] + n2y23 * x3[1]  + n2z23 * x3[2]
        
        self._v2x23 = self._v2x230 
        self._v2y23 = self._v2y230 
        self._v3x23 = self._v3x230 
        self._v3y23 = self._v3y230
        
        
        # determine the rotation matrix for 3-4 chord 
        # base vector 1: aligns with the axis of the 3-4 chord
        x34 = x4[0] - x3[0]
        y34 = x4[1] - x3[1]
        z34 = x4[2] - x3[2]
        mag34 = m.sqrt(x34 * x34 + y34 * y34 + z34 * z34)
        n1x34 = x34 / mag34
        n1y34 = y34 / mag34
        n1z34 = z34 / mag34
        
        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x4[0] + x3[0]) / 2.0
        y = (x4[1] + x3[1]) / 2.0
        z = (x4[2] + x3[2]) / 2.0
        mag = m.sqrt(x * x + y * y + z * z )
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta34(delta34):
            nx = ex + delta34 * n1x34
            ny = ey + delta34 * n1y34
            nz = ez + delta34 * n1z34
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x34 * nx + n1y34 * ny + n1z34 * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L340
        deltaH = 4.0 * self._L340
        delta34 = findRoot(deltaL, deltaH, getDelta34)

        # create base vector 2 (the radial vector out to the chord)
        x = ex + delta34 * n1x34
        y = ey + delta34 * n1y34
        z = ez + delta34 * n1z34
        mag = m.sqrt(x * x + y * y + z * z)
        n2x34 = x / mag
        n2y34 = y / mag   
        n2z34 = z / mag
        
        # create the rotation matrix from dodecahedral to 3-4 chordal coordinates
        self._Pr2D34[0, 0] = n1x34
        self._Pr2D34[0, 1] = n2x34
        self._Pr2D34[1, 0] = n1y34
        self._Pr2D34[1, 1] = n2y34
        self._Pp2D34[:, :] = self._Pr2D34[:, :]
        self._Pc2D34[:, :] = self._Pr2D34[:, :]
        self._Pn2D34[:, :] = self._Pr2D34[:, :]

        # determine vertice coordinates in the chordal frame of reference
        self._v3x340 = n1x34 * x3[0] + n1y34 * x3[1] + n1z34 * x3[2] 
        self._v3y340 = n2x34 * x3[0] + n2y34 * x3[1] + n2z34 * x3[2]
        self._v4x340 = n1x34 * x4[0] + n1y34 * x4[1] + n1z34 * x4[2]
        self._v4y340 = n2x34 * x4[0] + n2y34 * x4[1] + n2z34 * x4[2]
 
        self._v3x34 = self._v3x340 
        self._v3y34 = self._v3y340 
        self._v4x34 = self._v4x340 
        self._v4y34 = self._v4y340 
        
        
        # determine the rotation matrix for 4-5 chord 
        # base vector 1: aligns with the axis of the 4-5 chord
        x45 = x5[0] - x4[0]
        y45 = x5[1] - x4[1]
        z45 = x5[2] - x4[2]
        mag45 = m.sqrt(x45 * x45 + y45 * y45 + z45 * z45)
        n1x45 = x45 / mag45
        n1y45 = y45 / mag45
        n1z45 = z45 / mag45         
        
        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x5[0] + x4[0]) / 2.0
        y = (x5[1] + x4[1]) / 2.0
        z = (x5[2] + x4[2]) / 2.0
        mag = m.sqrt(x * x + y * y + z * z )
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta45(delta45):
            nx = ex + delta45 * n1x45
            ny = ey + delta45 * n1y45
            nz = ez + delta45 * n1z45
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x45 * nx + n1y45 * ny + n1z45 * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L450
        deltaH = 4.0 * self._L450
        delta45 = findRoot(deltaL, deltaH, getDelta45)

        # create base vector 2 (the radial vector out to the chord)
        x = ex + delta45 * n1x45
        y = ey + delta45 * n1y45
        z = ez + delta45 * n1z45
        mag = m.sqrt(x * x + y * y + z * z)
        n2x45 = x / mag
        n2y45 = y / mag   
        n2z45 = z / mag 
        
        # create the rotation matrix from dodecahedral to 4-5 chordal coordinates
        self._Pr2D45[0, 0] = n1x45
        self._Pr2D45[0, 1] = n2x45
        self._Pr2D45[1, 0] = n1y45
        self._Pr2D45[1, 1] = n2y45
        self._Pp2D45[:, :] = self._Pr2D45[:, :]
        self._Pc2D45[:, :] = self._Pr2D45[:, :]
        self._Pn2D45[:, :] = self._Pr2D45[:, :]

        # determine vertice coordinates in the chordal frame of reference
        self._v4x450 = n1x45 * x4[0] + n1y45 * x4[1] + n1z45 * x4[2] 
        self._v4y450 = n2x45 * x4[0] + n2y45 * x4[1] + n2z45 * x4[2]
        self._v5x450 = n1x45 * x5[0] + n1y45 * x5[1] + n1z45 * x5[2]
        self._v5y450 = n2x45 * x5[0] + n2y45 * x5[1] + n2z45 * x5[2]

        self._v4x45 = self._v4x450 
        self._v4y45 = self._v4y450 
        self._v5x45 = self._v5x450 
        self._v5y45 = self._v5y450 
        
        
        # determine the rotation matrix for 5-1 chord 
        # base vector 1: aligns with the axis of the 5-1 chord
        x51 = x1[0] - x5[0]
        y51 = x1[1] - x5[1]
        z51 = x1[2] - x5[2]
        mag51 = m.sqrt(x51 * x51 + y51 * y51 + z51 * z51)
        n1x51 = x51 / mag51
        n1y51 = y51 / mag51 
        n1z51 = z51 / mag51
        
        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1[0] + x5[0]) / 2.0
        y = (x1[1] + x5[1]) / 2.0
        z = (x1[2] + x5[2]) / 2.0
        mag = m.sqrt(x * x + y * y + z * z )
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta51(delta51):
            nx = ex + delta51 * n1x51
            ny = ey + delta51 * n1y51
            nz = ez + delta51 * n1z51
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x51 * nx + n1y51 * ny + n1z51 * nz 
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L510
        deltaH = 4.0 * self._L510
        delta51 = findRoot(deltaL, deltaH, getDelta51)

        # create base vector 2 (the radial vector out to the chord)
        x = ex + delta51 * n1x51
        y = ey + delta51 * n1y51
        z = ez + delta51 * n1z51
        mag = m.sqrt(x * x + y * y + z * z)
        n2x51 = x / mag
        n2y51 = y / mag   
        n2z51 = z / mag 
        
        # create the rotation matrix from dodecahedral to 5-1 chordal coordinates
        self._Pr2D51[0, 0] = n1x51
        self._Pr2D51[0, 1] = n2x51
        self._Pr2D51[1, 0] = n1y51
        self._Pr2D51[1, 1] = n2y51
        self._Pp2D51[:, :] = self._Pr2D51[:, :]
        self._Pc2D51[:, :] = self._Pr2D51[:, :]
        self._Pn2D51[:, :] = self._Pr2D51[:, :]

        # determine vertice coordinates in the chordal frame of reference
        self._v5x510 = n1x51 * x5[0] + n1y51 * x5[1] + n1z51 * x5[2]
        self._v5y510 = n2x51 * x5[0] + n2y51 * x5[1] + n2z51 * x5[2]
        self._v1x510 = n1x51 * x1[0] + n1y51 * x1[1] + n1z51 * x1[2]
        self._v1y510 = n2x51 * x1[0] + n2y51 * x1[1] + n2z51 * x1[2]

        self._v5x51 = self._v5x510 
        self._v5y51 = self._v5y510 
        self._v1x51 = self._v1x510 
        self._v1y51 = self._v1y510         
        
        # establish the shape functions located at the various Gauss points 
        # for pentagon
        pentAtGaussPt = 1
        Psf1 = pentShapeFunction(self._pgq.coordinates(pentAtGaussPt))
        pentAtGaussPt = 2
        Psf2 = pentShapeFunction(self._pgq.coordinates(pentAtGaussPt))
        pentAtGaussPt = 3
        Psf3 = pentShapeFunction(self._pgq.coordinates(pentAtGaussPt))
        pentAtGaussPt = 4
        Psf4 = pentShapeFunction(self._pgq.coordinates(pentAtGaussPt))
        pentAtGaussPt = 5
        Psf5 = pentShapeFunction(self._pgq.coordinates(pentAtGaussPt))
        self._pentShapeFns = {
            1: Psf1,
            2: Psf2,
            3: Psf3,
            4: Psf4,
            5: Psf5        
        }
        
        # establish the shape functions located at the various Gauss points
        # for chord
        chordAtGaussPt = 1
        csf1 = chordShapeFunction(self._cgq.coordinates(chordAtGaussPt))
        chordAtGaussPt = 2
        csf2 = chordShapeFunction(self._cgq.coordinates(chordAtGaussPt))
        self._chordShapeFns = {
            1: csf1,
            2: csf2
        }
        
        # create matrices for a pentagon at its Gauss points via dictionaries
        # p implies previous, c implies current, n implies next
            # displacement gradients located at the Gauss points of pentagon
        self._G0 = {
            1: np.zeros((2, 2), dtype=float),
            2: np.zeros((2, 2), dtype=float),
            3: np.zeros((2, 2), dtype=float),
            4: np.zeros((2, 2), dtype=float),
            5: np.zeros((2, 2), dtype=float)
        }
        self._Gp = {
            1: np.zeros((2, 2), dtype=float),
            2: np.zeros((2, 2), dtype=float),
            3: np.zeros((2, 2), dtype=float),
            4: np.zeros((2, 2), dtype=float),
            5: np.zeros((2, 2), dtype=float)
        }
        self._Gc = {
            1: np.zeros((2, 2), dtype=float),
            2: np.zeros((2, 2), dtype=float),
            3: np.zeros((2, 2), dtype=float),
            4: np.zeros((2, 2), dtype=float),
            5: np.zeros((2, 2), dtype=float),
        }
        self._Gn = {
            1: np.zeros((2, 2), dtype=float),
            2: np.zeros((2, 2), dtype=float),
            3: np.zeros((2, 2), dtype=float),
            4: np.zeros((2, 2), dtype=float),
            5: np.zeros((2, 2), dtype=float)
        }

        # deformation gradients located at the Gauss points of pentagon
        self._F0 = {
            1: np.identity(2, dtype=float),
            2: np.identity(2, dtype=float),
            3: np.identity(2, dtype=float),
            4: np.identity(2, dtype=float),
            5: np.identity(2, dtype=float)
        }
        self._Fp = {
            1: np.identity(2, dtype=float),
            2: np.identity(2, dtype=float),
            3: np.identity(2, dtype=float),
            4: np.identity(2, dtype=float),
            5: np.identity(2, dtype=float)
        }
        self._Fc = {
            1: np.identity(2, dtype=float),
            2: np.identity(2, dtype=float),
            3: np.identity(2, dtype=float),
            4: np.identity(2, dtype=float),
            5: np.identity(2, dtype=float)
        }
        self._Fn = {
            1: np.identity(2, dtype=float),
            2: np.identity(2, dtype=float),
            3: np.identity(2, dtype=float),
            4: np.identity(2, dtype=float),
            5: np.identity(2, dtype=float)
        }

        # assign membrane objects to each Gauss point of pentagon
        mem1 = membrane(h)
        mem2 = membrane(h)
        mem3 = membrane(h)
        mem4 = membrane(h)
        mem5 = membrane(h)

        self._septum = {
            1: mem1,
            2: mem2,
            3: mem3,
            4: mem4,
            5: mem5
        }

        self._rho = mp.rhoSepta()

        self._width = mp.septalWidth()

        M1, M2, M_t, N1, N2, N_t, G1, G2, G_t, xi_f, pi_0 = mp.septalMembrane()
        # the elastic moduli governing dilation
        self._M1 = M1
        self._M2 = M2
        self._Me_t = M_t
        # the elastic moduli governing squeeze
        self._N1 = N1
        self._N2 = N2
        self._Ne_t = N_t
        # the elastic moduli governing shear
        self._G1 = G1
        self._G2 = G2
        self._Ge_t = G_t
        # the maximum strain at rupture
        self._xi_f = xi_f 
        # the membrane pre-stressing of the surface tension
        self._pi_0 = pi_0

        nbrVars = 4   # for a chord they are: temperature and length
        respVars = 4
        T0 = 37.0     # body temperature in centigrade
        # thermodynamic strains (thermal and mechanical) are 0 at reference
        eVec0 = np.zeros((nbrVars,), dtype=float)
        # physical variables have reference values of
        xVec0 = np.zeros((nbrVars,), dtype=float)
        # vector of thermodynamic response variables
        yVec0 = np.zeros((respVars,), dtype=float)
        
        yVec0[1] = self._pi_0      # initial surface tension
        yVec0[2] = 0.0             # initial normal stress difference
        yVec0[3] = 0.0             # initial shear stress
        
        xVec0[0] = T0   # temperature in centigrade
        xVec0[1] = 1.0  # elongation in 1 direction 
        xVec0[2] = 1.0  # elongation in 2 direction 
        xVec0[3] = 0.0  # magnitude of shear 

        self._response = {
            1: ceMembrane(),
            2: ceMembrane(),
            3: ceMembrane(),
            4: ceMembrane(),
            5: ceMembrane()
        }

        self._Ms = {
            1: self._response[1].secMod(eVec0, xVec0, yVec0),
            2: self._response[2].secMod(eVec0, xVec0, yVec0),
            3: self._response[3].secMod(eVec0, xVec0, yVec0),
            4: self._response[4].secMod(eVec0, xVec0, yVec0),
            5: self._response[5].secMod(eVec0, xVec0, yVec0)
        }
        
        self._Mt = {
            1: self._response[1].tanMod(eVec0, xVec0, yVec0),
            2: self._response[2].tanMod(eVec0, xVec0, yVec0),
            3: self._response[3].tanMod(eVec0, xVec0, yVec0),
            4: self._response[4].tanMod(eVec0, xVec0, yVec0),
            5: self._response[5].tanMod(eVec0, xVec0, yVec0)
        }        

        self.Ss = {
            1: self._response[1].stressMtx(),
            2: self._response[2].stressMtx(),
            3: self._response[3].stressMtx(),
            4: self._response[4].stressMtx(),
            5: self._response[5].stressMtx()
        }  

        self.T = {
            1: self._response[1].intensiveStressVec(),
            2: self._response[2].intensiveStressVec(),
            3: self._response[3].intensiveStressVec(),
            4: self._response[4].intensiveStressVec(),
            5: self._response[5].intensiveStressVec()
        }
              
    def __str__(self):
        return self.toString()
    
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

    # Material properties that associate with this tetrahedron.  Except for the
    # mass density, all are drawn randomly from a statistical distribution.
    def massDensity(self):
        # returns the mass density of the membrane
        return self._rho
    
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
    
    # These FE arrays are evaluated at the beginning of the current step of
    # integration, i.e., they associate with an updated Lagrangian formulation.
    def _massMatrix(self):
        # create the returned mass matrix
        mMtx = np.zeros((10, 10), dtype=float)

        # construct the consistent mass matrix
        massC = np.zeros((10, 10), dtype=float)
        NtN = np.zeros((10, 10), dtype=float)
        for i in range(1, self._pgq.gaussPoints()+1):
            psfn = self._pentShapeFns[i]
            wgt = self._pgq.weight(i)
            NtN += wgt * np.matmul(np.transpose(psfn.Nmtx), psfn.Nmtx)
        massC[:, :] = NtN[:, :]

        # construct the lumped mass matrix in natural co-ordinates
        massL = np.zeros((10, 10), dtype=float)
        row, col = np.diag_indices_from(massC)
        massL[row, col] = massC.sum(axis=1)
        
        # constrcuct the averaged mass matrix in natural co-ordinates
        massA = np.zeros((10, 10), dtype=float)
        massA = 0.5 * (massC + massL)

        # the following print statements were used to verify the code
        # print("\nThe averaged mass matrix in natural co-ordinates is")
        # print(0.5 * massA)  

        # current vertex coordinates in pentagonal frame of reference
        x01 = (self._v1x0, self._v1y0)
        x02 = (self._v2x0, self._v2y0)
        x03 = (self._v3x0, self._v3y0)
        x04 = (self._v4x0, self._v4y0)
        x05 = (self._v5x0, self._v5y0)
        
        # convert average mass matrix from natural to physical co-ordinates
        Jdet = psfn.jacobianDeterminant(x01, x02, x03, x04, x05)
        rho = self.massDensity()
        mMtx = (rho * self._width * Jdet) * massA

        return mMtx


    def _tangentStiffnessMtxC(self):
        
        CMtx = np.zeros((10, 10), dtype=float)      
        # current vertex coordinates in pentagonal frame of reference
        xn1 = (self._v1x, self._v1y)
        xn2 = (self._v2x, self._v2y)
        xn3 = (self._v3x, self._v3y)
        xn4 = (self._v4x, self._v4y)
        xn5 = (self._v5x, self._v5y)

        # assign coordinates at the vertices in the reference configuration
        x01 = (self._v1x0, self._v1y0)
        x02 = (self._v2x0, self._v2y0)
        x03 = (self._v3x0, self._v3y0)
        x04 = (self._v4x0, self._v4y0)
        x05 = (self._v5x0, self._v5y0)
        
        Cs1 = np.zeros((10, 10), dtype=float)
        Ct1 = np.zeros((10, 10), dtype=float)
            
        for i in range(1, self._pgq.gaussPoints()+1):
            psfn = self._pentShapeFns[i]
            wgt = self._pgq.weight(i) 
            Mt = self._Mt[i]
            Ss = self.Ss[i]
            
            # determinant of jacobian matrix
            Jdet = psfn.jacobianDeterminant(x01, x02, x03, x04, x05)

            BLmtx = psfn.BL(xn1, xn2, xn3, xn4, xn5)

            Hmtx1 = psfn.H1(xn1, xn2, xn3, xn4, xn5)
            BNmtx1 = psfn.BN1(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)
            
            Hmtx2 = psfn.H2(xn1, xn2, xn3, xn4, xn5)
            BNmtx2 = psfn.BN2(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)

            Hmtx3 = psfn.H3(xn1, xn2, xn3, xn4, xn5)
            BNmtx3 = psfn.BN3(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)
            
            # total nonlinear Bmatrix
            BNmtx = BNmtx1 + BNmtx2 + BNmtx3
            
            # the tangent stiffness matrix Cs1
            Cs1 += (self._width * Jdet * wgt * ( Hmtx1.T.dot(Ss).dot(Hmtx1) 
                                               + Hmtx2.T.dot(Ss).dot(Hmtx2)
                                               + Hmtx3.T.dot(Ss).dot(Hmtx3)))
            # the tangent stiffness matrix Ct1
            Ct1 += (self._width * Jdet * wgt * ( BLmtx.T.dot(Mt).dot(BLmtx) 
                                               + BLmtx.T.dot(Mt).dot(BNmtx)
                                               + BNmtx.T.dot(Mt).dot(BLmtx) 
                                               + BNmtx.T.dot(Mt).dot(BNmtx) ))


        Cs = np.zeros((10, 10), dtype=float)
        Ct = np.zeros((10, 10), dtype=float)
        
        Cs[:, :] = Cs1[:, :]
        Ct[:, :] = Ct1[:, :]
              
        # determine the total tangent stiffness matrix
        CMtx = Cs + Ct

        return CMtx
    
    
    def _secantStiffnessMtxK(self, reindex):
        
        kMtx = np.zeros((10, 10), dtype=float)      
        # current vertex coordinates in pentagonal frame of reference
        xn1 = (self._v1x, self._v1y)
        xn2 = (self._v2x, self._v2y)
        xn3 = (self._v3x, self._v3y)
        xn4 = (self._v4x, self._v4y)
        xn5 = (self._v5x, self._v5y)

        # assign coordinates at the vertices in the reference configuration
        x01 = (self._v1x0, self._v1y0)
        x02 = (self._v2x0, self._v2y0)
        x03 = (self._v3x0, self._v3y0)
        x04 = (self._v4x0, self._v4y0)
        x05 = (self._v5x0, self._v5y0)
        
        Ks1 = np.zeros((10, 10), dtype=float)
        Kt1 = np.zeros((10, 10), dtype=float)
            
        for i in range(1, self._pgq.gaussPoints()+1):
            psfn = self._pentShapeFns[i]
            wgt = self._pgq.weight(i) 
            Ms = self._Ms[i]
            Mt = self._Mt[i]
            
            # determinant of jacobian matrix
            Jdet = psfn.jacobianDeterminant(x01, x02, x03, x04, x05)

            BLmtx = psfn.BL(xn1, xn2, xn3, xn4, xn5)

            Hmtx1 = psfn.H1(xn1, xn2, xn3, xn4, xn5)
            Lmtx1 = psfn.L1(xn1, xn2, xn3, xn4, xn5)
            BNmtx1 = psfn.BN1(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)
            A1 = psfn.A1(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)              
            Dmtx1 = self.dDisplacement1(reindex)      
            dA1 = np.dot(Lmtx1, np.transpose(Dmtx1))            
            dSt1 = A1.T.dot(Mt).dot(dA1)
            
            Hmtx2 = psfn.H2(xn1, xn2, xn3, xn4, xn5)
            Lmtx2 = psfn.L2(xn1, xn2, xn3, xn4, xn5)
            BNmtx2 = psfn.BN2(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)
            A2 = psfn.A2(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)              
            Dmtx2 = self.dDisplacement2(reindex)      
            dA2 = np.dot(Lmtx2, np.transpose(Dmtx2))            
            dSt2 = A2.T.dot(Mt).dot(dA2)

            Hmtx3 = psfn.H3(xn1, xn2, xn3, xn4, xn5)
            Lmtx3 = psfn.L3(xn1, xn2, xn3, xn4, xn5)
            BNmtx3 = psfn.BN3(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)
            A3 = psfn.A3(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)              
            Dmtx3 = self.dDisplacement3(reindex)      
            dA3 = np.dot(Lmtx3, np.transpose(Dmtx3))            
            dSt3 = A3.T.dot(Mt).dot(dA3)
            
            # total nonlinear Bmatrix
            BNmtx = BNmtx1 + BNmtx2 + BNmtx3
            
            # the secant stiffness matrix Ks1
            Ks1 += (self._width * Jdet * wgt * ( BLmtx.T.dot(Ms).dot(BLmtx) 
                                               + BLmtx.T.dot(Ms).dot(BNmtx) 
                                               + BNmtx.T.dot(Ms).dot(BLmtx) 
                                               + BNmtx.T.dot(Ms).dot(BNmtx) ))
            # the secant stiffness matrix Kt1
            Kt1 += (self._width * Jdet * wgt * ( Hmtx1.T.dot(dSt1).dot(Hmtx1) 
                                               + Hmtx2.T.dot(dSt2).dot(Hmtx2)
                                               + Hmtx3.T.dot(dSt3).dot(Hmtx3)))


        Ks = np.zeros((10, 10), dtype=float)
        Kt = np.zeros((10, 10), dtype=float)
        

        Ks[:, :] = Ks1[:, :]
        Kt[:, :] = Kt1[:, :]
              
        # determine the total secant stiffness matrix
        kMtx = Ks + Kt

        return kMtx

    def _forcingFunction(self):
        
        state = 'ref'
        fVec = np.zeros((10,1), dtype=float)
                   
        P12 = self.rotation12(state)
        P23 = self.rotation23(state)
        P34 = self.rotation34(state)
        P45 = self.rotation45(state)
        P51 = self.rotation51(state)
        
        # normal vector to each chord of pentagon
        n12 = np.zeros((1,2), dtype=float)
        n23 = np.zeros((1,2), dtype=float)
        n34 = np.zeros((1,2), dtype=float)
        n45 = np.zeros((1,2), dtype=float)
        n51 = np.zeros((1,2), dtype=float)
        
        n12[0, 0] = P12[0, 1]
        n12[0, 1] = P12[1, 1]
        
        n23[0, 0] = P23[0, 1]
        n23[0, 1] = P23[1, 1]
        
        n34[0, 0] = P34[0, 1]
        n34[0, 1] = P34[1, 1]
        
        n45[0, 0] = P45[0, 1]
        n45[0, 1] = P45[1, 1]
        
        n51[0, 0] = P51[0, 1]
        n51[0, 1] = P51[1, 1]

                
        # create the traction vector apply on each chord of pentagon 
        t12 = np.dot(self.Ss[1], np.transpose(n12))   
        t23 = np.dot(self.Ss[2], np.transpose(n23))  
        t34 = np.dot(self.Ss[3], np.transpose(n34))  
        t45 = np.dot(self.Ss[4], np.transpose(n45))  
        t51 = np.dot(self.Ss[5], np.transpose(n51)) 
        

        # current vertex coordinates in pentagonal frame of reference
        xn1 = (self._v1x, self._v1y)
        xn2 = (self._v2x, self._v2y)
        xn3 = (self._v3x, self._v3y)
        xn4 = (self._v4x, self._v4y)
        xn5 = (self._v5x, self._v5y)

        # assign coordinates at the vertices in the reference configuration
        x01 = (self._v1x0, self._v1y0)
        x02 = (self._v2x0, self._v2y0)
        x03 = (self._v3x0, self._v3y0)
        x04 = (self._v4x0, self._v4y0)
        x05 = (self._v5x0, self._v5y0)

        # assign coordinates at the vertices in the reference configuration
        x1012 = (self._v1x120, self._v1y120)
        x2012 = (self._v2x120, self._v2y120)

        x2023 = (self._v2x230, self._v2y230)
        x3023 = (self._v3x230, self._v3y230)
        
        x3034 = (self._v3x340, self._v3y340)
        x4034 = (self._v4x340, self._v4y340)

        x4045 = (self._v4x450, self._v4y450)
        x5045 = (self._v5x450, self._v5y450)

        x5051 = (self._v5x510, self._v5y510)
        x1051 = (self._v1x510, self._v1y510)

        
        Nmtx12 = np.zeros((10, 1), dtype=float)
        Nmtx23 = np.zeros((10, 1), dtype=float)
        Nmtx34 = np.zeros((10, 1), dtype=float)
        Nmtx45 = np.zeros((10, 1), dtype=float)
        Nmtx51 = np.zeros((10, 1), dtype=float)

        BL1 = np.zeros((10, 3), dtype=float)
        BN1 = np.zeros((10, 3), dtype=float)
        BN2 = np.zeros((10, 3), dtype=float)
        BN3 = np.zeros((10, 3), dtype=float)
                
        for i in range(1, self._pgq.gaussPoints()+1):
            Psfn = self._pentShapeFns[i]
            wgt = self._pgq.weight(i)
            T0 = self.T[i]

            BLmtx = Psfn.BL(xn1, xn2, xn3, xn4, xn5)
            BNmtx1 = Psfn.BN1(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)
            BNmtx2 = Psfn.BN2(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)
            BNmtx3 = Psfn.BN3(xn1, xn2, xn3, xn4, xn5, x01, x02, x03, x04, x05)
            
            BL1 += wgt * np.transpose(BLmtx)   
            BN1 += wgt * np.transpose(BNmtx1)   
            BN2 += wgt * np.transpose(BNmtx2)  
            BN3 += wgt * np.transpose(BNmtx3)
            B = BL1 + BN1 + BN2 + BN3
            BdotT0 = np.dot(B, T0)
        
        # determinant of jacobian matrix
        pJdet = Psfn.jacobianDeterminant(x01, x02, x03, x04, x05)
        
        F0 = np.zeros((10, 1), dtype=float)
        
        F0 = pJdet * self._width * BdotT0

        for i in range(1, self._cgq.gaussPoints()+1):
            csfn = self._chordShapeFns[i]
            wgt = self._cgq.weight(i)

            N1 = csfn.N1
            N2 = csfn.N2
            
            N12 = np.array([[N1, 0.0, N2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, N1, 0.0, N2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            N23 = np.array([[0.0, 0.0, N1, 0.0, N2, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [ 0.0, 0.0, 0.0, N1, 0.0, N2, 0.0, 0.0, 0.0, 0.0]])
            N34 = np.array([[0.0, 0.0, 0.0, 0.0, N1, 0.0, N2, 0.0, 0.0, 0.0],
                            [ 0.0, 0.0, 0.0, 0.0, 0.0, N1, 0.0, N2, 0.0, 0.0]])
            N45 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, N1, 0.0, N2, 0.0],
                            [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, N1, 0.0, N2]])
            N51 = np.array([[N2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, N1, 0.0],
                            [0.0, N2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, N1]])
                            
            Nmtx12 += wgt * N12.T.dot(t12) 
            Nmtx23 += wgt * N23.T.dot(t23) 
            Nmtx34 += wgt * N34.T.dot(t34) 
            Nmtx45 += wgt * N45.T.dot(t45) 
            Nmtx51 += wgt * N51.T.dot(t51)  
        
        # determinant of jacobian matrix
        cJdet12 = csfn.jacobianDeterminant(x1012, x2012)
        cJdet23 = csfn.jacobianDeterminant(x2023, x3023)
        cJdet34 = csfn.jacobianDeterminant(x3034, x4034)
        cJdet45 = csfn.jacobianDeterminant(x4045, x5045)
        cJdet51 = csfn.jacobianDeterminant(x5051, x1051)
        
        FBc = np.zeros((10, 1), dtype=float)
        
        FBc = (cJdet12 * Nmtx12 + cJdet23 * Nmtx23 + cJdet34 * Nmtx34 
               + cJdet45 * Nmtx45 + cJdet51 * Nmtx51)  


        fVec = FBc - F0
            
        return fVec
        
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

    def update(self):
        # computes the fields positioned at the next time step

        # get the updated coordinates for the vetices of the pentagon
        x1 = self._vertex[1].coordinates('next')
        x2 = self._vertex[2].coordinates('next')
        x3 = self._vertex[3].coordinates('next')
        x4 = self._vertex[4].coordinates('next')
        x5 = self._vertex[5].coordinates('next')

        # base vector 1: connects the two shoulders of the pentagon
        x = x5[0] - x2[0]
        y = x5[1] - x2[1]
        z = x5[2] - x2[2]
        mag = m.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 2: goes from the apex to a point along its base

        # establish the unit vector for the base of the pentagon
        x = x4[0] - x3[0]
        y = x4[1] - x3[1]
        z = x4[2] - x3[2]
        mag = m.sqrt(x * x + y * y + z * z)
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the base of
        # the pentagon that results in a vector n2 which is normal to n1

        def getDelta(delta):
            nx = x1[0] - (x3[0] + delta * ex)
            ny = x1[1] - (x3[1] + delta * ey)
            nz = x1[2] - (x3[2] + delta * ez)
            # when the dot product is zero then the two vectors are orthogonal
            n1Dotn2 = n1x * nx + n1y * ny + n1z * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * mag
        deltaH = 5.0 * mag
        delta = findRoot(deltaL, deltaH, getDelta)

        # create base vector 2
        x = x1[0] - (x3[0] + delta * ex)
        y = x1[1] - (x3[1] + delta * ey)
        z = x1[2] - (x3[2] + delta * ez)
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
        self._v1x0 = n1x * x1[0] + n1y * x1[1] + n1z * x1[2]
        self._v1y0 = n2x * x1[0] + n2y * x1[1] + n2z * x1[2]
        self._v2x0 = n1x * x2[0] + n1y * x2[1] + n1z * x2[2]
        self._v2y0 = n2x * x2[0] + n2y * x2[1] + n2z * x2[2]
        self._v3x0 = n1x * x3[0] + n1y * x3[1] + n1z * x3[2]
        self._v3y0 = n2x * x3[0] + n2y * x3[1] + n2z * x3[2]
        self._v4x0 = n1x * x4[0] + n1y * x4[1] + n1z * x4[2]
        self._v4y0 = n2x * x4[0] + n2y * x4[1] + n2z * x4[2]
        self._v5x0 = n1x * x5[0] + n1y * x5[1] + n1z * x5[2]
        self._v5y0 = n2x * x5[0] + n2y * x5[1] + n2z * x5[2]

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
        self._v1z = n3x * x1[0] + n3y * x1[1] + n3z * x1[2]
        self._v2z = n3x * x2[0] + n3y * x2[1] + n3z * x2[2]
        self._v3z = n3x * x3[0] + n3y * x3[1] + n3z * x3[2]
        self._v4z = n3x * x4[0] + n3y * x4[1] + n3z * x4[2]
        self._v5z = n3x * x5[0] + n3y * x5[1] + n3z * x5[2]

        # determine the area of this irregular pentagon
        self._An = (self._v1x * self._v2y - self._v2x * self._v1y +
                    self._v2x * self._v3y - self._v3x * self._v2y +
                    self._v3x * self._v4y - self._v4x * self._v3y +
                    self._v4x * self._v5y - self._v5x * self._v4y +
                    self._v5x * self._v1y - self._v1x * self._v5y) / 2.0
        # the area will be positive if the vertices index counter clockwise

        # determine the centroid of this pentagon in pentagonal coordinates
        self._cx = ((self._v1x + self._v2x) *
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
        self._cy = ((self._v1y + self._v2y) *
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
        self._cz = ((self._v1z + self._v2z + self._v3z + self._v4z +
                     self._v5z) / 5.0)

        # rotate this centroid back into the reference coordinate system
        self._centroidXn = n1x * self._cx + n2x * self._cy + n3x * self._cz
        self._centroidYn = n1y * self._cx + n2y * self._cy + n3y * self._cz
        self._centroidZn = n1z * self._cx + n2z * self._cy + n3z * self._cz
        
        
        

        # determine length of the 1-2 chord in the next configuration
        self._L12n = m.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2
                             + (x2[2] - x1[2])**2)
        
        # determine length of the 2-3 chord in the next configuration
        self._L23n = m.sqrt((x3[0] - x2[0])**2 + (x3[1] - x2[1])**2
                             + (x3[2] - x2[2])**2)
        
        # determine length of the 3-4 chord in the next configuration
        self._L34n = m.sqrt((x4[0] - x3[0])**2 + (x4[1] - x3[1])**2
                             + (x4[2] - x3[2])**2)
        
        # determine length of the 4-5 chord in the next configuration
        self._L45n = m.sqrt((x5[0] - x4[0])**2 + (x5[1] - x4[1])**2
                             + (x5[2] - x4[2])**2)
        
        # determine length of the 5-1 chord in the next configuration
        self._L51n = m.sqrt((x5[0] - x1[0])**2 + (x5[1] - x1[1])**2
                             + (x5[2] - x1[2])**2)

        # determine the rotation matrix for 1-2 chord 
        # base vector 1: aligns with the axis of the 1-2 chord
        x12 = x2[0] - x1[0]
        y12 = x2[1] - x1[1]
        z12 = x2[2] - x1[2]
        mag12 = m.sqrt(x12 * x12 + y12 * y12  + z12 * z12)
        n1x12 = x12 / mag12
        n1y12 = y12 / mag12
        n1z12 = z12 / mag12

        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1[0] + x2[0]) / 2.0
        y = (x1[1] + x2[1]) / 2.0
        z = (x1[2] + x2[2]) / 2.0
        mag = m.sqrt(x * x + y * y  + z * z)
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta12(delta12):
            nx = ex + delta12 * n1x12
            ny = ey + delta12 * n1y12
            nz = ez + delta12 * n1z12
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x12 * nx + n1y12 * ny + n1z12 * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L120
        deltaH = 4.0 * self._L120
        delta12 = findRoot(deltaL, deltaH, getDelta12)

        # create base vector 2 (the radial vector out to the chord)
        x = ex + delta12 * n1x12
        y = ey + delta12 * n1y12
        z = ez + delta12 * n1z12
        mag = m.sqrt(x * x + y * y + z * z)
        n2x12 = x / mag
        n2y12 = y / mag   
        n2z12 = z / mag
        
        # create the rotation matrix from dodecahedral to 1-2 chordal coordinates
        self._Pn2D12[0, 0] = n1x12
        self._Pn2D12[0, 1] = n2x12
        self._Pn2D12[1, 0] = n1y12
        self._Pn2D12[1, 1] = n2y12

        # determine vertice coordinates in the chordal frame of reference
        self._v1x120 = n1x12 * x1[0] + n1y12 * x1[1] + n1z12 * x1[2] 
        self._v1y120 = n2x12 * x1[0] + n2y12 * x1[1] + n2z12 * x1[2]
        self._v2x120 = n1x12 * x2[0] + n1y12 * x2[1] + n1z12 * x2[2]
        self._v2y120 = n2x12 * x2[0] + n2y12 * x2[1] + n2z12 * x2[2]

        self._v1x12 = self._v1x120 
        self._v1y12 = self._v1y120 
        self._v2x12 = self._v2x120 
        self._v2y12 = self._v2y120 



        # determine the rotation matrix for 2-3 chord 
        # base vector 1: aligns with the axis of the 2-3 chord
        x23 = x3[0] - x2[0]
        y23 = x3[1] - x2[1]
        z23 = x3[2] - x2[2]
        mag23 = m.sqrt(x23 * x23 + y23 * y23 + z23 * z23)
        n1x23 = x23 / mag23
        n1y23 = y23 / mag23
        n1z23 = z23 / mag23
        # create normal vector 

        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x3[0] + x2[0]) / 2.0
        y = (x3[1] + x2[1]) / 2.0
        z = (x3[2] + x2[2]) / 2.0
        mag = m.sqrt(x * x + y * y  + z * z )
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta23(delta23):
            nx = ex + delta23 * n1x23
            ny = ey + delta23 * n1y23
            nz = ez + delta23 * n1z23
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x23 * nx + n1y23 * ny + n1z23 * nz 
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L230
        deltaH = 4.0 * self._L230
        delta23 = findRoot(deltaL, deltaH, getDelta23)

        # create base vector 2 (the radial vector out to the chord)
        x = ex + delta23 * n1x23
        y = ey + delta23 * n1y23
        z = ez + delta23 * n1z23
        mag = m.sqrt(x * x + y * y + z * z)
        n2x23 = x / mag
        n2y23 = y / mag 
        n2z23 = z / mag     
        
        # create the rotation matrix from dodecahedral to 2-3 chordal coordinates
        self._Pn2D23[0, 0] = n1x23
        self._Pn2D23[0, 1] = n2x23
        self._Pn2D23[1, 0] = n1y23
        self._Pn2D23[1, 1] = n2y23

        # determine vertice coordinates in the chordal frame of reference
        self._v2x230 = n1x23 * x2[0] + n1y23 * x2[1]  + n1z23 * x2[2] 
        self._v2y230 = n2x23 * x2[0] + n2y23 * x2[1]  + n2z23 * x2[2]
        self._v3x230 = n1x23 * x3[0] + n1y23 * x3[1]  + n1z23 * x3[2]
        self._v3y230 = n2x23 * x3[0] + n2y23 * x3[1]  + n2z23 * x3[2]

        self._v2x23 = self._v2x230 
        self._v2y23 = self._v2y230 
        self._v3x23 = self._v3x230 
        self._v3y23 = self._v3y230 
        
        
        # determine the rotation matrix for 3-4 chord 
        # base vector 1: aligns with the axis of the 3-4 chord
        x34 = x4[0] - x3[0]
        y34 = x4[1] - x3[1]
        z34 = x4[2] - x3[2]
        mag34 = m.sqrt(x34 * x34 + y34 * y34 + z34 * z34)
        n1x34 = x34 / mag34
        n1y34 = y34 / mag34
        n1z34 = z34 / mag34
        
        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x4[0] + x3[0]) / 2.0
        y = (x4[1] + x3[1]) / 2.0
        z = (x4[2] + x3[2]) / 2.0
        mag = m.sqrt(x * x + y * y + z * z )
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta34(delta34):
            nx = ex + delta34 * n1x34
            ny = ey + delta34 * n1y34
            nz = ez + delta34 * n1z34
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x34 * nx + n1y34 * ny + n1z34 * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L340
        deltaH = 4.0 * self._L340
        delta34 = findRoot(deltaL, deltaH, getDelta34)

        # create base vector 2 (the radial vector out to the chord)
        x = ex + delta34 * n1x34
        y = ey + delta34 * n1y34
        z = ez + delta34 * n1z34
        mag = m.sqrt(x * x + y * y + z * z)
        n2x34 = x / mag
        n2y34 = y / mag   
        n2z34 = z / mag
        
        # create the rotation matrix from dodecahedral to 3-4 chordal coordinates
        self._Pn2D34[0, 0] = n1x34
        self._Pn2D34[0, 1] = n2x34
        self._Pn2D34[1, 0] = n1y34
        self._Pn2D34[1, 1] = n2y34

        # determine vertice coordinates in the chordal frame of reference
        self._v3x340 = n1x34 * x3[0] + n1y34 * x3[1] + n1z34 * x3[2] 
        self._v3y340 = n2x34 * x3[0] + n2y34 * x3[1] + n2z34 * x3[2]
        self._v4x340 = n1x34 * x4[0] + n1y34 * x4[1] + n1z34 * x4[2]
        self._v4y340 = n2x34 * x4[0] + n2y34 * x4[1] + n2z34 * x4[2]

        self._v3x34 = self._v3x340 
        self._v3y34 = self._v3y340 
        self._v4x34 = self._v4x340 
        self._v4y34 = self._v4y340 
        

        # determine the rotation matrix for 4-5 chord 
        # base vector 1: aligns with the axis of the 4-5 chord
        x45 = x5[0] - x4[0]
        y45 = x5[1] - x4[1]
        z45 = x5[2] - x4[2]
        mag45 = m.sqrt(x45 * x45 + y45 * y45 + z45 * z45)
        n1x45 = x45 / mag45
        n1y45 = y45 / mag45
        n1z45 = z45 / mag45         
        
        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x5[0] + x4[0]) / 2.0
        y = (x5[1] + x4[1]) / 2.0
        z = (x5[2] + x4[2]) / 2.0
        mag = m.sqrt(x * x + y * y + z * z )
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta45(delta45):
            nx = ex + delta45 * n1x45
            ny = ey + delta45 * n1y45
            nz = ez + delta45 * n1z45
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x45 * nx + n1y45 * ny + n1z45 * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L450
        deltaH = 4.0 * self._L450
        delta45 = findRoot(deltaL, deltaH, getDelta45)

        # create base vector 2 (the radial vector out to the chord)
        x = ex + delta45 * n1x45
        y = ey + delta45 * n1y45
        z = ez + delta45 * n1z45
        mag = m.sqrt(x * x + y * y + z * z)
        n2x45 = x / mag
        n2y45 = y / mag   
        n2z45 = z / mag 
        
        # create the rotation matrix from dodecahedral to 4-5 chordal coordinates
        self._Pn2D45[0, 0] = n1x45
        self._Pn2D45[0, 1] = n2x45
        self._Pn2D45[1, 0] = n1y45
        self._Pn2D45[1, 1] = n2y45

        # determine vertice coordinates in the chordal frame of reference
        self._v4x450 = n1x45 * x4[0] + n1y45 * x4[1] + n1z45 * x4[2] 
        self._v4y450 = n2x45 * x4[0] + n2y45 * x4[1] + n2z45 * x4[2]
        self._v5x450 = n1x45 * x5[0] + n1y45 * x5[1] + n1z45 * x5[2]
        self._v5y450 = n2x45 * x5[0] + n2y45 * x5[1] + n2z45 * x5[2]

        self._v4x45 = self._v4x450 
        self._v4y45 = self._v4y450 
        self._v5x45 = self._v5x450 
        self._v5y45 = self._v5y450 
        
                       
        # determine the rotation matrix for 5-1 chord 
        # base vector 1: aligns with the axis of the 5-1 chord
        x51 = x1[0] - x5[0]
        y51 = x1[1] - x5[1]
        z51 = x1[2] - x5[2]
        mag51 = m.sqrt(x51 * x51 + y51 * y51 + z51 * z51)
        n1x51 = x51 / mag51
        n1y51 = y51 / mag51 
        n1z51 = z51 / mag51
        
        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1[0] + x5[0]) / 2.0
        y = (x1[1] + x5[1]) / 2.0
        z = (x1[2] + x5[2]) / 2.0
        mag = m.sqrt(x * x + y * y + z * z )
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta51(delta51):
            nx = ex + delta51 * n1x51
            ny = ey + delta51 * n1y51
            nz = ez + delta51 * n1z51
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x51 * nx + n1y51 * ny + n1z51 * nz 
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L510
        deltaH = 4.0 * self._L510
        delta51 = findRoot(deltaL, deltaH, getDelta51)

        # create base vector 2 (the radial vector out to the chord)
        x = ex + delta51 * n1x51
        y = ey + delta51 * n1y51
        z = ez + delta51 * n1z51
        mag = m.sqrt(x * x + y * y + z * z)
        n2x51 = x / mag
        n2y51 = y / mag   
        n2z51 = z / mag 
        
        # create the rotation matrix from dodecahedral to 5-1 chordal coordinates
        self._Pn2D51[0, 0] = n1x51
        self._Pn2D51[0, 1] = n2x51
        self._Pn2D51[1, 0] = n1y51
        self._Pn2D51[1, 1] = n2y51

        # determine vertice coordinates in the chordal frame of reference
        self._v5x510 = n1x51 * x5[0] + n1y51 * x5[1] + n1z51 * x5[2]
        self._v5y510 = n2x51 * x5[0] + n2y51 * x5[1] + n2z51 * x5[2]
        self._v1x510 = n1x51 * x1[0] + n1y51 * x1[1] + n1z51 * x1[2]
        self._v1y510 = n2x51 * x1[0] + n2y51 * x1[1] + n2z51 * x1[2]

        self._v5x51 = self._v5x510 
        self._v5y51 = self._v5y510 
        self._v1x51 = self._v1x510 
        self._v1y51 = self._v1y510 
        

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
        # displacement gradients located at the Gauss points of pentagon
        self._Gn[1] = self._pentShapeFns[1].G(x1, x2, x3, x4, x5,
                                       x10, x20, x30, x40, x50)
        self._Gn[2] = self._pentShapeFns[2].G(x1, x2, x3, x4, x5,
                                       x10, x20, x30, x40, x50)
        self._Gn[3] = self._pentShapeFns[3].G(x1, x2, x3, x4, x5,
                                       x10, x20, x30, x40, x50)
        self._Gn[4] = self._pentShapeFns[4].G(x1, x2, x3, x4, x5,
                                       x10, x20, x30, x40, x50)
        self._Gn[5] = self._pentShapeFns[5].G(x1, x2, x3, x4, x5,
                                       x10, x20, x30, x40, x50)

        # deformation gradients located at the Gauss points of pentagon
        self._Fn[1] = self._pentShapeFns[1].F(x1, x2, x3, x4, x5,
                                       x10, x20, x30, x40, x50)
        self._Fn[2] = self._pentShapeFns[2].F(x1, x2, x3, x4, x5,
                                       x10, x20, x30, x40, x50)
        self._Fn[3] = self._pentShapeFns[3].F(x1, x2, x3, x4, x5,
                                       x10, x20, x30, x40, x50)
        self._Fn[4] = self._pentShapeFns[4].F(x1, x2, x3, x4, x5,
                                       x10, x20, x30, x40, x50)
        self._Fn[5] = self._pentShapeFns[5].F(x1, x2, x3, x4, x5,
                                       x10, x20, x30, x40, x50)

        # update the membrane objects at each Gauss point
        for i in range(1, self._pgq.gaussPoints()+1):
            self._septum[i].update(self._Fn[i])

        return  # nothing

    def advance(self, reindex):
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

        for i in range(1, self._pgq.gaussPoints()+1):
            self._Fp[i][:, :] = self._Fc[i][:, :]
            self._Fc[i][:, :] = self._Fn[i][:, :]
            self._Gp[i][:, :] = self._Gc[i][:, :]
            self._Gc[i][:, :] = self._Gn[i][:, :]

        # advance the membrane objects at each Gauss point
        for i in range(1, self._pgq.gaussPoints()+1):
            self._septum[i].advance()

        # assign current to previous values, and then next to current values
        self._L12p = self._L12c
        self._L12c = self._L12n
        self._Pp2D12[:, :] = self._Pc2D12[:, :]
        self._Pc2D12[:, :] = self._Pn2D12[:, :]        

        self._L23p = self._L23c
        self._L23c = self._L23n
        self._Pp2D23[:, :] = self._Pc2D23[:, :]
        self._Pc2D23[:, :] = self._Pn2D23[:, :]   
        
        self._L34p = self._L34c
        self._L34c = self._L34n
        self._Pp2D34[:, :] = self._Pc2D34[:, :]
        self._Pc2D34[:, :] = self._Pn2D34[:, :]  
        
        self._L45p = self._L45c
        self._L45c = self._L45n
        self._Pp2D45[:, :] = self._Pc2D45[:, :]
        self._Pc2D45[:, :] = self._Pn2D45[:, :]  
        
        self._L51p = self._L51c
        self._L51c = self._L51n
        self._Pp2D51[:, :] = self._Pc2D51[:, :]
        self._Pc2D51[:, :] = self._Pn2D51[:, :] 


        # compute the FE arrays needed for the next interval of integration
        self.mMtx = self._massMatrix()
        self.cMtx = self._tangentStiffnessMtxC()
        self.kMtx = self._secantStiffnessMtxK(reindex)
        self.fVec = self._forcingFunction()

           
    def rotation12(self, state):
        # rotation matrix for chord 1-2
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Pc2D12)
            elif state == 'n' or state == 'next':
                return np.copy(self._Pn2D12)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Pp2D12)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._Pr2D12)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "sent in a call to chord.rotation.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "sent in a call to chord.rotation.")
            
    def rotation23(self, state):
        # rotation matrix for chord 2-3
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Pc2D23)
            elif state == 'n' or state == 'next':
                return np.copy(self._Pn2D23)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Pp2D23)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._Pr2D23)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "sent in a call to chord.rotation.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "sent in a call to chord.rotation.")

    def rotation34(self, state):
        # rotation matrix for chord 3-4
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Pc2D34)
            elif state == 'n' or state == 'next':
                return np.copy(self._Pn2D34)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Pp2D34)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._Pr2D34)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "sent in a call to chord.rotation.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "sent in a call to chord.rotation.")

    def rotation45(self, state):
        # rotation matrix for chord 4-5
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Pc2D45)
            elif state == 'n' or state == 'next':
                return np.copy(self._Pn2D45)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Pp2D45)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._Pr2D45)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "sent in a call to chord.rotation.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "sent in a call to chord.rotation.")

    def rotation51(self, state):
        # rotation matrix for chord 5-1
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Pc2D51)
            elif state == 'n' or state == 'next':
                return np.copy(self._Pn2D51)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Pp2D51)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._Pr2D51)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "sent in a call to chord.rotation.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "sent in a call to chord.rotation.")

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
                self._cx = self._centroidXc
                self._cy = self._centroidYc
                self._cz = self._centroidZc
            elif state == 'n' or state == 'next':
                self._cx = self._centroidXn
                self._cy = self._centroidYn
                self._cz = self._centroidZn
            elif state == 'p' or state == 'prev' or state == 'previous':
                self._cx = self._centroidXp
                self._cy = self._centroidYp
                self._cz = self._centroidZp
            elif state == 'r' or state == 'ref' or state == 'reference':
                self._cx = self._centroidX0
                self._cy = self._centroidY0
                self._cz = self._centroidZ0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.centroid.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.centroid.")
        return np.array([self._cx, self._cy, self._cz])

    def centeroidDisplacement(self, reindex, state):
        # verify the input
        if not isinstance(reindex, Pivot):
            raise RuntimeError("The 'reindex' variable sent to " +
                               "pentagon.centroidDisplacement must be of type Pivot.")
        # calculate the displacement in the specified configuration
        u = np.zeros(3, dtype=float)
        x0 = np.zeros(3, dtype=float)
        
        x0, y0, z0 = self.centroid('ref')        
        xRef = np.array([x0, y0, z0])
        fromCase = reindex.pivotCase('ref')
        if isinstance(state, str):
            xp, yp, zp = self.centroid('prev')
            xc, yc, zc = self.centroid('curr')
            xn, yn, zn = self.centroid('next')
            
            if state == 'c' or state == 'curr' or state == 'current':
                x = np.array([xc, yc, zc])
                toCase = reindex.pivotCase('curr')
                x0 = reindex.reindexVector(xRef, fromCase, toCase)
            elif state == 'n' or state == 'next':
                x = np.array([xn, yn, zn])
                toCase = reindex.pivotCase('next')
                x0 = reindex.reindexVector(xRef, fromCase, toCase)
            elif state == 'p' or state == 'prev' or state == 'previous':
                x = np.array([xp, yp, zp])
                toCase = reindex.pivotCase('prev')
                x0 = reindex.reindexVector(xRef, fromCase, toCase)
            elif state == 'r' or state == 'ref' or state == 'reference':
                x = np.array([x0, y0, z0])
                x0 = np.array([x0, y0, z0])
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   "in a call to pentagon.centroidDisplacement.")
        else:
            raise RuntimeError("Unknown state {} ".format(str(state)) +
                               "in a call to pentagon.centroidDisplacement.")
        u = x - x0
        
        R = self.rotation(state)
        uxr = R[0, 0] * u[0] + R[1, 0] * u[1] + R[2, 0] * u[2]
        uyr = R[0, 1] * u[0] + R[1, 1] * u[1] + R[2, 1] * u[2]
        uzr = R[0, 2] * u[0] + R[1, 2] * u[1] + R[2, 2] * u[2]

        return np.array([uxr, uyr, uzr])
    
    def centeroidVelocity(self, reindex, state):
        # verify the input
        if not isinstance(reindex, Pivot):
            raise RuntimeError("The 'reindex' variable sent to " +
                               "pentagon.centroidVelocity must be of type Pivot.")
        # calculate the velocity in the specified configuration
        h = 2.0 * self._h
        v = np.zeros(3, dtype=float)
        if isinstance(state, str):
            xp, yp, zp = self.centroid('prev')
            xc, yc, zc = self.centroid('curr')
            xn, yn, zn = self.centroid('next')
            if state == 'c' or state == 'curr' or state == 'current':
                toCase = reindex.pivotCase('curr')
                # map vectors into co-ordinate system of current configuration
                xPrev = np.array([xp, yp, zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xNext = np.array([xn, yn, zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
                
                # use second-order central difference formula
                v = (xN - xP) / h
                
            elif state == 'n' or state == 'next':
                toCase = reindex.pivotCase('next')
                # map vectors into co-ordinate system of next configuration
                xPrev = np.array([xp, yp, zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xCurr = np.array([xc, yc, zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xN = np.array([xn, yn, zn])
                # use second-order backward difference formula
                v = (3.0 * xN - 4.0 * xC + xP) / h

            elif state == 'p' or state == 'prev' or state == 'previous':
                toCase = reindex.pivotCase('prev')
                # map vector into co-ordinate system of previous configuration
                xP = np.array([xp, yp, zp])
                xCurr = np.array([xc, yc, zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xNext = np.array([xn, yn, zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
                # use second-order forward difference formula
                v = (-xN + 4.0 * xC - 3.0 * xP) / h
                
            elif state == 'r' or state == 'ref' or state == 'reference':
                # velocity is zero
                pass
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   "in a call to pentagon.centroidVelocity.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.centroidVelocity.")

        R = self.rotation(state)
        vxr = R[0, 0] * v[0] + R[1, 0] * v[1] + R[2, 0] * v[2]
        vyr = R[0, 1] * v[0] + R[1, 1] * v[1] + R[2, 1] * v[2]
        vzr = R[0, 2] * v[0] + R[1, 2] * v[1] + R[2, 2] * v[2]

        return np.array([vxr, vyr, vzr])

   
    def centeroidCompInjCri(self, reindex, state):
        # verify the input
        if not isinstance(reindex, Pivot):
            raise RuntimeError("The 'reindex' variable sent to " +
                               "pentagon.centroidVelocity must be of type Pivot.")
        # calculate the velocity in the specified configuration
        h = 2.0 * self._h
        if isinstance(state, str):
            xp, yp, zp = self.centroid('prev')
            xc, yc, zc = self.centroid('curr')
            xn, yn, zn = self.centroid('next')
            if state == 'c' or state == 'curr' or state == 'current':
                toCase = reindex.pivotCase('curr')
                # map vectors into co-ordinate system of current configuration
                xPrev = np.array([xp, yp, zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xNext = np.array([xn, yn, zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
                
                # use second-order central difference formula
                c = (xP - xN) / h
                
            elif state == 'n' or state == 'next':
                toCase = reindex.pivotCase('next')
                # map vectors into co-ordinate system of next configuration
                xPrev = np.array([xp, yp, zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xCurr = np.array([xc, yc, zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xN = np.array([xn, yn, zn])
                # use second-order backward difference formula
                c = (3.0 * xN - 4.0 * xC + xP) / h

            elif state == 'p' or state == 'prev' or state == 'previous':
                toCase = reindex.pivotCase('prev')
                # map vector into co-ordinate system of previous configuration
                xP = np.array([xp, yp, zp])
                xCurr = np.array([xc, yc, zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xNext = np.array([xn, yn, zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
                # use second-order forward difference formula
                c = (-xN + 4.0 * xC - 3.0 * xP) / h
                
            elif state == 'r' or state == 'ref' or state == 'reference':
                # velocity is zero
                pass
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   "in a call to pentagon.centroidVelocity.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.centroidVelocity.")

        R = self.rotation(state)
        cxr = R[0, 0] * c[0] + R[1, 0] * c[1] + R[2, 0] * c[2]
        cyr = R[0, 1] * c[0] + R[1, 1] * c[1] + R[2, 1] * c[2]
        czr = R[0, 2] * c[0] + R[1, 2] * c[1] + R[2, 2] * c[2]

        return np.array([cxr, cyr, czr])
    
    
    def centeroidAcceleration(self, reindex, state):
        # verify the input
        if not isinstance(reindex, Pivot):
            raise RuntimeError("The 'reindex' variable sent to " +
                               "pentagon.centroidAcceleration must be of type Pivot.")
        # calculate the acceleration in the specified configuration
        h2 = self._h**2
        a = np.zeros(3, dtype=float)
        if isinstance(state, str):
            xp, yp, zp = self.centroid('prev')
            xc, yc, zc = self.centroid('curr')
            xn, yn, zn = self.centroid('next')
            if state == 'c' or state == 'curr' or state == 'current':
                toCase = reindex.pivotCase('curr')
                # map vectors into co-ordinate system of current configuration
                xPrev = np.array([xp, yp, zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xC = np.array([xc, yc, zc])
                xNext = np.array([xn, yn, zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
            elif state == 'n' or state == 'next':
                toCase = reindex.pivotCase('next')
                # map vectors into co-ordinate system of next configuration
                xPrev = np.array([xp, yp, zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xCurr = np.array([xc, yc, zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xN = np.array([xn, yn, zn])
            elif state == 'p' or state == 'prev' or state == 'previous':
                toCase = reindex.pivotCase('prev')
                # map vector into co-ordinate system of previous configuration
                xP = np.array([xp, yp, zp])
                xCurr = np.array([xc, yc, zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xNext = np.array([xn, yn, zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
            elif state == 'r' or state == 'ref' or state == 'reference':
                # acceleration is zero
                pass
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   "in a call to pentagon.centroidAcceleration.")
        else:
            raise RuntimeError("Unknown state {} ".format(str(state)) +
                               "in a call to pentagon.centroidAcceleration.")
        a = (xN - 2.0 * xC + xP) / h2

        R = self.rotation(state)
        axr = R[0, 0] * a[0] + R[1, 0] * a[1] + R[2, 0] * a[2]
        ayr = R[0, 1] * a[0] + R[1, 1] * a[1] + R[2, 1] * a[2]
        azr = R[0, 2] * a[0] + R[1, 2] * a[1] + R[2, 2] * a[2]

        return np.array([axr, ayr, azr])
    
    
    # change in displacementin contribution to the first nonlinear strain
    def dDisplacement1(self, reindex):
        v1 = self._vertex[1].velocity(reindex, 'curr')
        v2 = self._vertex[2].velocity(reindex, 'curr')
        v3 = self._vertex[3].velocity(reindex, 'curr')
        v4 = self._vertex[4].velocity(reindex, 'curr')
        v5 = self._vertex[5].velocity(reindex, 'curr')

        R = self.rotation('curr')        
        Dmtx1 = np.zeros((2, 10), dtype=float)
        Dmtx1[0, 0] = R[0, 0] * v1[0] + R[1, 0] * v1[1] + R[2, 0] * v1[2]
        Dmtx1[0, 2] = R[0, 0] * v2[0] + R[1, 0] * v2[1] + R[2, 0] * v2[2]
        Dmtx1[0, 4] = R[0, 0] * v3[0] + R[1, 0] * v3[1] + R[2, 0] * v3[2]
        Dmtx1[0, 6] = R[0, 0] * v4[0] + R[1, 0] * v4[1] + R[2, 0] * v4[2]
        Dmtx1[0, 8] = R[0, 0] * v5[0] + R[1, 0] * v5[1] + R[2, 0] * v5[2]


        Dmtx1[1, 1] = R[0, 1] * v1[0] + R[1, 1] * v1[1] + R[2, 1] * v1[2]
        Dmtx1[1, 3] = R[0, 1] * v2[0] + R[1, 1] * v2[1] + R[2, 1] * v2[2]
        Dmtx1[1, 5] = R[0, 1] * v3[0] + R[1, 1] * v3[1] + R[2, 1] * v3[2]
        Dmtx1[1, 7] = R[0, 1] * v4[0] + R[1, 1] * v4[1] + R[2, 1] * v4[2]
        Dmtx1[1, 9] = R[0, 1] * v5[0] + R[1, 1] * v5[1] + R[2, 1] * v5[2]
        
        return Dmtx1


    # change in displacementin contribution to the second nonlinear strain
    def dDisplacement2(self, reindex):
        v1 = self._vertex[1].velocity(reindex, 'curr')
        v2 = self._vertex[2].velocity(reindex, 'curr')
        v3 = self._vertex[3].velocity(reindex, 'curr')
        v4 = self._vertex[4].velocity(reindex, 'curr')
        v5 = self._vertex[5].velocity(reindex, 'curr')

        R = self.rotation('curr')        
        Dmtx2 = np.zeros((2, 10), dtype=float)
        Dmtx2[0, 0] = R[0, 1] * v1[0] + R[1, 1] * v1[1] + R[2, 1] * v1[2]
        Dmtx2[0, 2] = R[0, 1] * v2[0] + R[1, 1] * v2[1] + R[2, 1] * v2[2]
        Dmtx2[0, 4] = R[0, 1] * v3[0] + R[1, 1] * v3[1] + R[2, 1] * v3[2]
        Dmtx2[0, 6] = R[0, 1] * v4[0] + R[1, 1] * v4[1] + R[2, 1] * v4[2]
        Dmtx2[0, 8] = R[0, 1] * v5[0] + R[1, 1] * v5[1] + R[2, 1] * v5[2]


        Dmtx2[1, 1] = R[0, 0] * v1[0] + R[1, 0] * v1[1] + R[2, 0] * v1[2]
        Dmtx2[1, 3] = R[0, 0] * v2[0] + R[1, 0] * v2[1] + R[2, 0] * v2[2]
        Dmtx2[1, 5] = R[0, 0] * v3[0] + R[1, 0] * v3[1] + R[2, 0] * v3[2]
        Dmtx2[1, 7] = R[0, 0] * v4[0] + R[1, 0] * v4[1] + R[2, 0] * v4[2]
        Dmtx2[1, 9] = R[0, 0] * v5[0] + R[1, 0] * v5[1] + R[2, 0] * v5[2]
        
        return Dmtx2

    def dDisplacement3(self, reindex):
        v1 = self._vertex[1].velocity(reindex, 'curr')
        v2 = self._vertex[2].velocity(reindex, 'curr')
        v3 = self._vertex[3].velocity(reindex, 'curr')
        v4 = self._vertex[4].velocity(reindex, 'curr')
        v5 = self._vertex[5].velocity(reindex, 'curr')

        R = self.rotation('curr')        
        Dmtx3 = np.zeros((2, 10), dtype=float)
        Dmtx3[0, 0] = R[0, 0] * v1[0] + R[1, 0] * v1[1] + R[2, 0] * v1[2]
        Dmtx3[0, 2] = R[0, 0] * v2[0] + R[1, 0] * v2[1] + R[2, 0] * v2[2]
        Dmtx3[0, 4] = R[0, 0] * v3[0] + R[1, 0] * v3[1] + R[2, 0] * v3[2]
        Dmtx3[0, 6] = R[0, 0] * v4[0] + R[1, 0] * v4[1] + R[2, 0] * v4[2]
        Dmtx3[0, 8] = R[0, 0] * v5[0] + R[1, 0] * v5[1] + R[2, 0] * v5[2]


        Dmtx3[1, 1] = R[0, 1] * v1[0] + R[1, 1] * v1[1] + R[2, 1] * v1[2]
        Dmtx3[1, 3] = R[0, 1] * v2[0] + R[1, 1] * v2[1] + R[2, 1] * v2[2]
        Dmtx3[1, 5] = R[0, 1] * v3[0] + R[1, 1] * v3[1] + R[2, 1] * v3[2]
        Dmtx3[1, 7] = R[0, 1] * v4[0] + R[1, 1] * v4[1] + R[2, 1] * v4[2]
        Dmtx3[1, 9] = R[0, 1] * v5[0] + R[1, 1] * v5[1] + R[2, 1] * v5[2]
        
        return Dmtx3
    
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
    def G(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.G and you sent {}.".format(PentGaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Gc[PentGaussPt])
            elif state == 'n' or state == 'next':
                return np.copy(self._Gn[PentGaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Gp[PentGaussPt])
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._G0[PentGaussPt])
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.G.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.G.")

    # deformation gradient at a Gauss point
    def F(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.F and you sent {}.".format(PentGaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Fc[PentGaussPt])
            elif state == 'n' or state == 'next':
                return np.copy(self._Fn[PentGaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Fp[PentGaussPt])
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._F0[PentGaussPt])
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.F.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.F.")

    # velocity gradient at a Gauss point
    def L(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.L and you sent {}.".format(PentGaussPt))

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
                dF = ((self._Fn[PentGaussPt] - self._Fp[PentGaussPt])
                      / (2.0 * self._h))
                fInv = FInv(self._Fc[PentGaussPt])
            elif state == 'n' or state == 'next':
                # use backward difference scheme
                dF = ((3.0 * self._Fn[PentGaussPt] - 4.0 * self._Fc[PentGaussPt] +
                       self._Fp[PentGaussPt]) / (2.0 * self._h))
                fInv = FInv(self._Fn[PentGaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use forward difference scheme
                dF = ((-self._Fn[PentGaussPt] + 4.0 * self._Fc[PentGaussPt] -
                       3.0 * self._Fp[PentGaussPt]) / (2.0 * self._h))
                fInv = FInv(self._Fp[PentGaussPt])
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
    def Q(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.Q and you sent {}.".format(PentGaussPt))
        return self._septum[PentGaussPt].Q(state)

    # the orthogonal matrix from a Gram-Schmidt factorization of (relabeled) F
    def R(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be" +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.R and you sent {}.".format(PentGaussPt))
        return self._septum[PentGaussPt].R(state)

    # a skew-symmetric matrix for the spin associated with R
    def Omega(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.Omega, you sent {}.".format(PentGaussPt))
        return self._septum[PentGaussPt].spin(state)

    # Laplace stretch from a Gram-Schmidt factorization of (relabeled) F
    def U(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.U and you sent {}.".format(PentGaussPt))
        return self._septum[PentGaussPt].U(state)

    # inverse Laplace stretch from Gram-Schmidt factorization of (relabeled) F
    def UInv(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.UInv, you sent {}.".format(PentGaussPt))
        return self._septum[PentGaussPt].UInv(state)

    # differential in the Laplace stretch from Gram-Schmidt factorization of F
    def dU(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.dU and you sent {}.".format(PentGaussPt))
        return self._septum[PentGaussPt].dU(state)

    # inverse Laplace stretch from Gram-Schmidt factorization of (relabeled) F
    def dUInv(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.dUInv, you sent {}.".format(PentGaussPt))
        return self._septum[PentGaussPt].dUInv(state)

    # physical kinematic attributes at a Gauss point in the pentagon

    def dilation(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.dilation and you sent " +
                               "{}.".format(PentGaussPt))
        return self._septum[PentGaussPt].dilation(state)

    def squeeze(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.squeeze and you sent " +
                               "{}.".format(PentGaussPt))
        return self._septum[PentGaussPt].squeeze(state)

    def shear(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.shear and you sent " +
                               "{}.".format(PentGaussPt))
        return self._septum[PentGaussPt].shear(state)

    def dDilation(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pentGaussPts) +
                               "pentagon.dDilation and you sent " +
                               "{}.".format(PentGaussPt))
        return self._septum[PentGaussPt].dDilation(state)

    def dSqueeze(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.dSqueeze and you sent " +
                               "{}.".format(PentGaussPt))
        return self._septum[PentGaussPt].dSqueeze(state)

    def dShear(self, PentGaussPt, state):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("The pentGaussPt must be " +
                               "{} in call to ".format(self._pgq.gaussPoints()) +
                               "pentagon.dShear and you sent " +
                               "{}.".format(PentGaussPt))
        return self._septum[PentGaussPt].dShear(state)

    # properties used in finite elements

    def pentShapeFunction(self, PentGaussPt):
        if PentGaussPt != self._pgq.gaussPoints():
            raise RuntimeError("pentGaussPt must be " +
                               "{} in call ".format(self._pgq.gaussPoints()) +
                               "to pentagon.pentShapeFunction and you sent " +
                               "{}.".format(PentGaussPt))
            Psf = self._pentShapeFns[PentGaussPt]
        return Psf
    

    def chordShapeFunction(self, chordGaussPt):
        if chordGaussPt != self._cgq.gaussPoints():
            raise RuntimeError("gaussPt must be " +
                                   "{} ".format(self._cgq.gaussPoints()) +
                                   "in a call to chord.shapeFunction " +
                                   "and you sent {}.".format(chordGaussPt))
            csf = self._chordShapeFns[chordGaussPt]
        return csf

    def pentGaussQuadrature(self):
        return self._pgq
    
    def chordGaussQuadrature(self):
        return self._cgq
    
    def massMatrix(self):
        mMtx = np.copy(self._massMatrix())
        return mMtx

    def tangentStiffnessMtxC(self):
        cMtx = np.copy(self._tangentStiffnessMtxC())
        return cMtx
    
    def secantStiffnessMtxK(self, reindex):
        kMtx = np.copy(self._secantStiffnessMtxK(reindex))
        return kMtx

    def forcingFunction(self):
        fVec = np.copy(self._forcingFunction())
        return fVec    
    
    