#!/usr/bin/env python3       
# -*- coding: utf-8 -*-     

from chords import chord
import materialProperties as mp
import math as m
from membranes import membrane
import numpy as np
from ridder import findRoot
from shapeFnPentagons import pentShapeFunction
import spin as spinMtx
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

    sf = p.shapeFunction(pentGaussPt):
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
                 pentGaussPts, triaGaussPts):
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
        if pentGaussPts == 1 or pentGaussPts == 4 or pentGaussPts == 7:
            self._pentGaussPts = pentGaussPts
        else:
            raise RuntimeError('{} Gauss points were '.format(pentGaussPts) +
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

        # establish the shape functions located at the pentagon Gauss points 
        # (xi, eta)
        if pentGaussPts == 1:
            xi = 0.0000000000000000
            eta = 0.0000000000000000
            sf11 = pentShapeFunction(xi, eta)

            self._pentShapeFns = {
                11: sf11
            }
        elif pentGaussPts == 4:            
            xi1 = -0.0349156305831802
            eta1 = 0.6469731019095136
            sf11 = pentShapeFunction(xi1, eta1)
            
            xi1 = -0.0349156305831802
            eta2 = -0.0321196846022659
            sf12 = pentShapeFunction(xi1, eta2)

            xi1 = -0.0349156305831802
            eta3 = -0.6469731019095134
            sf13 = pentShapeFunction(xi1, eta3)

            xi1 = -0.0349156305831802
            eta4 = 0.0321196846022661
            sf14 = pentShapeFunction(xi1, eta4)
            
            
            
            xi2 = -0.5951653065516678
            eta1 = 0.6469731019095136
            sf21 = pentShapeFunction(xi2, eta1)
            
            xi2 = -0.5951653065516678
            eta2 = -0.0321196846022659
            sf22 = pentShapeFunction(xi2, eta2)

            xi2 = -0.5951653065516678
            eta3 = -0.6469731019095134
            sf23 = pentShapeFunction(xi2, eta3)

            xi2 = -0.5951653065516678
            eta4 = 0.0321196846022661
            sf24 = pentShapeFunction(xi2, eta4)
            
            
            
            xi3 = 0.0349156305831798
            eta1 = 0.6469731019095136
            sf31 = pentShapeFunction(xi3, eta1)

            xi3 = 0.0349156305831798
            eta2 = -0.0321196846022659
            sf32 = pentShapeFunction(xi3, eta2)
            
            xi3 = 0.0349156305831798
            eta3 = -0.6469731019095134
            sf33 = pentShapeFunction(xi3, eta3)

            xi3 = 0.0349156305831798
            eta4 = 0.0321196846022661
            sf34 = pentShapeFunction(xi3, eta4)
            
            

            xi4 = 0.5951653065516677
            eta1 = 0.6469731019095136
            sf41 = pentShapeFunction(xi4, eta1)

            xi4 = 0.5951653065516677
            eta2 = -0.0321196846022659
            sf42 = pentShapeFunction(xi4, eta2)

            xi4 = 0.5951653065516677
            eta3 = -0.6469731019095134
            sf43 = pentShapeFunction(xi4, eta3)
            
            xi4 = 0.5951653065516677
            eta4 = 0.0321196846022661
            sf44 = pentShapeFunction(xi4, eta4)


            self._pentShapeFns = {
                11: sf11,
                12: sf12,
                13: sf13,
                14: sf14,
                21: sf21,
                22: sf22,
                23: sf23,
                24: sf24,
                31: sf31,
                32: sf32,
                33: sf33,
                34: sf34,
                41: sf41,
                42: sf42,
                43: sf43,
                44: sf44
            }
        else:  # pentGaussPts = 7
            xi1 = -0.0000000000000000
            eta1 = -0.0000000000000002
            sf11 = pentShapeFunction(xi1, eta1)

            xi1 = -0.0000000000000000
            eta2 = 0.7099621260052327
            sf12 = pentShapeFunction(xi1, eta2)

            xi1 = -0.0000000000000000
            eta3 = 0.1907259121533272
            sf13 = pentShapeFunction(xi1, eta3)

            xi1 = -0.0000000000000000
            eta4 = -0.5531465782166917
            sf14 = pentShapeFunction(xi1, eta4)

            xi1 = -0.0000000000000000
            eta5 = -0.6644407817506509
            sf15 = pentShapeFunction(xi1, eta5)

            xi1 = -0.0000000000000000
            eta6 = -0.1251071394727008
            sf16 = pentShapeFunction(xi1, eta6)

            xi1 = -0.0000000000000000
            eta7 = 0.4872045224587945
            sf17 = pentShapeFunction(xi1, eta7)
            
            
            
            
            xi2 = -0.1351253857178451
            eta1 = -0.0000000000000002
            sf21 = pentShapeFunction(xi2, eta1)

            xi2 = -0.1351253857178451
            eta2 = 0.7099621260052327
            sf22 = pentShapeFunction(xi2, eta2)

            xi2 = -0.1351253857178451
            eta3 = 0.1907259121533272
            sf23 = pentShapeFunction(xi2, eta3)

            xi2 = -0.1351253857178451
            eta4 = -0.5531465782166917
            sf24 = pentShapeFunction(xi2, eta4)

            xi2 = -0.1351253857178451
            eta5 = -0.6644407817506509
            sf25 = pentShapeFunction(xi2, eta5)

            xi2 = -0.1351253857178451
            eta6 = -0.1251071394727008
            sf26 = pentShapeFunction(xi2, eta6)

            xi2 = -0.1351253857178451
            eta7 = 0.4872045224587945
            sf27 = pentShapeFunction(xi2, eta7)
            
            
                        
            
            xi3 = -0.6970858746672087
            eta1 = -0.0000000000000002
            sf31 = pentShapeFunction(xi3, eta1)

            xi3 = -0.6970858746672087
            eta2 = 0.7099621260052327
            sf32 = pentShapeFunction(xi3, eta2)

            xi3 = -0.6970858746672087
            eta3 = 0.1907259121533272
            sf33 = pentShapeFunction(xi3, eta3)

            xi3 = -0.6970858746672087
            eta4 = -0.5531465782166917
            sf34 = pentShapeFunction(xi3, eta4)

            xi3 = -0.6970858746672087
            eta5 = -0.6644407817506509
            sf35 = pentShapeFunction(xi3, eta5)

            xi3 = -0.6970858746672087
            eta6 = -0.1251071394727008
            sf36 = pentShapeFunction(xi3, eta6)

            xi3 = -0.6970858746672087
            eta7 = 0.4872045224587945
            sf37 = pentShapeFunction(xi3, eta7)
            
            
            
            
            xi4 = -0.4651171392611024
            eta1 = -0.0000000000000002
            sf41 = pentShapeFunction(xi4, eta1)

            xi4 = -0.4651171392611024
            eta2 = 0.7099621260052327
            sf42 = pentShapeFunction(xi4, eta2)

            xi4 = -0.4651171392611024
            eta3 = 0.1907259121533272
            sf43 = pentShapeFunction(xi4, eta3)

            xi4 = -0.4651171392611024
            eta4 = -0.5531465782166917
            sf44 = pentShapeFunction(xi4, eta4)

            xi4 = -0.4651171392611024
            eta5 = -0.6644407817506509
            sf45 = pentShapeFunction(xi4, eta5)

            xi4 = -0.4651171392611024
            eta6 = -0.1251071394727008
            sf46 = pentShapeFunction(xi4, eta6)

            xi4 = -0.4651171392611024
            eta7 = 0.4872045224587945
            sf47 = pentShapeFunction(xi4, eta7)
            
                        
            
            
            xi5 = 0.2842948078559476
            eta1 = -0.0000000000000002
            sf51 = pentShapeFunction(xi5, eta1)

            xi5 = 0.2842948078559476
            eta2 = 0.7099621260052327
            sf52 = pentShapeFunction(xi5, eta2)

            xi5 = 0.2842948078559476
            eta3 = 0.1907259121533272
            sf53 = pentShapeFunction(xi5, eta3)

            xi5 = 0.2842948078559476
            eta4 = -0.5531465782166917
            sf54 = pentShapeFunction(xi5, eta4)

            xi5 = 0.2842948078559476
            eta5 = -0.6644407817506509
            sf55 = pentShapeFunction(xi5, eta5)

            xi5 = 0.2842948078559476
            eta6 = -0.1251071394727008
            sf56 = pentShapeFunction(xi5, eta6)

            xi5 = 0.2842948078559476
            eta7 = 0.4872045224587945
            sf57 = pentShapeFunction(xi5, eta7)
            
            
            
            
            xi6 = 0.7117958231685716
            eta1 = -0.0000000000000002
            sf61 = pentShapeFunction(xi6, eta1)

            xi6 = 0.7117958231685716
            eta2 = 0.7099621260052327
            sf62 = pentShapeFunction(xi6, eta2)

            xi6 = 0.7117958231685716
            eta3 = 0.1907259121533272
            sf63 = pentShapeFunction(xi6, eta3)

            xi6 = 0.7117958231685716
            eta4 = -0.5531465782166917
            sf64 = pentShapeFunction(xi6, eta4)

            xi6 = 0.7117958231685716
            eta5 = -0.6644407817506509
            sf65 = pentShapeFunction(xi6, eta5)

            xi6 = 0.7117958231685716
            eta6 = -0.1251071394727008
            sf66 = pentShapeFunction(xi6, eta6)

            xi6 = 0.7117958231685716
            eta7 = 0.4872045224587945
            sf67 = pentShapeFunction(xi6, eta7)
            
            
            
            
            xi7 = 0.5337947578638855
            eta1 = -0.0000000000000002
            sf71 = pentShapeFunction(xi7, eta1)

            xi7 = 0.5337947578638855
            eta2 = 0.7099621260052327
            sf72 = pentShapeFunction(xi7, eta2)

            xi7 = 0.5337947578638855
            eta3 = 0.1907259121533272
            sf73 = pentShapeFunction(xi7, eta3)

            xi7 = 0.5337947578638855
            eta4 = -0.5531465782166917
            sf74 = pentShapeFunction(xi7, eta4)

            xi7 = 0.5337947578638855
            eta5 = -0.6644407817506509
            sf75 = pentShapeFunction(xi7, eta5)

            xi7 = 0.5337947578638855
            eta6 = -0.1251071394727008
            sf76 = pentShapeFunction(xi7, eta6)

            xi7 = 0.5337947578638855
            eta7 = 0.4872045224587945
            sf77 = pentShapeFunction(xi7, eta7)

            self._pentShapeFns = {
                11: sf11,
                12: sf12,
                13: sf13,
                14: sf14,
                15: sf15,
                16: sf16,
                17: sf17,
                21: sf21,
                22: sf22,
                23: sf23,
                24: sf24,
                25: sf25,
                26: sf26,
                27: sf27,
                31: sf31,
                32: sf32,
                33: sf33,
                34: sf34,
                35: sf35,
                36: sf36,
                37: sf37,
                41: sf41,
                42: sf42,
                43: sf43,
                44: sf44,
                45: sf45,
                46: sf46,
                47: sf47,
                51: sf51,
                52: sf52,
                53: sf53,
                54: sf54,
                55: sf55,
                56: sf56,
                57: sf57,
                61: sf61,
                62: sf62,
                63: sf63,
                64: sf64,
                65: sf65,
                66: sf66,
                67: sf67,
                71: sf71,
                72: sf72,
                73: sf73,
                74: sf74,
                75: sf75,
                76: sf76,
                77: sf77
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
        self._v1z = n3x * x1 + n3y * y1 + n3z * z1
        self._v2z = n3x * x2 + n3y * y2 + n3z * z2
        self._v3z = n3x * x3 + n3y * y3 + n3z * z3
        self._v4z = n3x * x4 + n3y * y4 + n3z * z4
        self._v5z = n3x * x5 + n3y * y5 + n3z * z5
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

        # create matrices for a pentagon at its Gauss points via dictionaries
        # p implies previous, c implies current, n implies next
        if pentGaussPts == 1:
            # displacement gradients located at the Gauss points of pentagon
            self._G0 = {
                11: np.zeros((2, 2), dtype=float)
            }
            self._Gp = {
                11: np.zeros((2, 2), dtype=float)
            }
            self._Gc = {
                11: np.zeros((2, 2), dtype=float)
            }
            self._Gn = {
                11: np.zeros((2, 2), dtype=float)
            }

            # deformation gradients located at the Gauss points of pentagon
            self._F0 = {
                11: np.identity(2, dtype=float)
            }
            self._Fp = {
                11: np.identity(2, dtype=float)
            }
            self._Fc = {
                11: np.identity(2, dtype=float)
            }
            self._Fn = {
                11: np.identity(2, dtype=float)
            }

        elif pentGaussPts == 4:
            # displacement gradients located at the Gauss points of pentagon
            self._G0 = {
                11: np.zeros((2, 2), dtype=float),
                12: np.zeros((2, 2), dtype=float),
                13: np.zeros((2, 2), dtype=float),
                14: np.zeros((2, 2), dtype=float),
                21: np.zeros((2, 2), dtype=float),
                22: np.zeros((2, 2), dtype=float),
                23: np.zeros((2, 2), dtype=float),
                24: np.zeros((2, 2), dtype=float),
                31: np.zeros((2, 2), dtype=float),
                32: np.zeros((2, 2), dtype=float),
                33: np.zeros((2, 2), dtype=float),
                34: np.zeros((2, 2), dtype=float),
                41: np.zeros((2, 2), dtype=float),
                42: np.zeros((2, 2), dtype=float),
                43: np.zeros((2, 2), dtype=float),
                44: np.zeros((2, 2), dtype=float)
            }
            self._Gp = {
                11: np.zeros((2, 2), dtype=float),
                12: np.zeros((2, 2), dtype=float),
                13: np.zeros((2, 2), dtype=float),
                14: np.zeros((2, 2), dtype=float),
                21: np.zeros((2, 2), dtype=float),
                22: np.zeros((2, 2), dtype=float),
                23: np.zeros((2, 2), dtype=float),
                24: np.zeros((2, 2), dtype=float),
                31: np.zeros((2, 2), dtype=float),
                32: np.zeros((2, 2), dtype=float),
                33: np.zeros((2, 2), dtype=float),
                34: np.zeros((2, 2), dtype=float),
                41: np.zeros((2, 2), dtype=float),
                42: np.zeros((2, 2), dtype=float),
                43: np.zeros((2, 2), dtype=float),
                44: np.zeros((2, 2), dtype=float)
            }
            self._Gc = {
                11: np.zeros((2, 2), dtype=float),
                12: np.zeros((2, 2), dtype=float),
                13: np.zeros((2, 2), dtype=float),
                14: np.zeros((2, 2), dtype=float),
                21: np.zeros((2, 2), dtype=float),
                22: np.zeros((2, 2), dtype=float),
                23: np.zeros((2, 2), dtype=float),
                24: np.zeros((2, 2), dtype=float),
                31: np.zeros((2, 2), dtype=float),
                32: np.zeros((2, 2), dtype=float),
                33: np.zeros((2, 2), dtype=float),
                34: np.zeros((2, 2), dtype=float),
                41: np.zeros((2, 2), dtype=float),
                42: np.zeros((2, 2), dtype=float),
                43: np.zeros((2, 2), dtype=float),
                44: np.zeros((2, 2), dtype=float)
            }
            self._Gn = {
                11: np.zeros((2, 2), dtype=float),
                12: np.zeros((2, 2), dtype=float),
                13: np.zeros((2, 2), dtype=float),
                14: np.zeros((2, 2), dtype=float),
                21: np.zeros((2, 2), dtype=float),
                22: np.zeros((2, 2), dtype=float),
                23: np.zeros((2, 2), dtype=float),
                24: np.zeros((2, 2), dtype=float),
                31: np.zeros((2, 2), dtype=float),
                32: np.zeros((2, 2), dtype=float),
                33: np.zeros((2, 2), dtype=float),
                34: np.zeros((2, 2), dtype=float),
                41: np.zeros((2, 2), dtype=float),
                42: np.zeros((2, 2), dtype=float),
                43: np.zeros((2, 2), dtype=float),
                44: np.zeros((2, 2), dtype=float)
            }

            # deformation gradients located at the Gauss points of pentagon
            self._F0 = {
                11: np.identity(2, dtype=float),
                12: np.identity(2, dtype=float),
                13: np.identity(2, dtype=float),
                14: np.identity(2, dtype=float),
                21: np.identity(2, dtype=float),
                22: np.identity(2, dtype=float),
                23: np.identity(2, dtype=float),
                24: np.identity(2, dtype=float),
                31: np.identity(2, dtype=float),
                32: np.identity(2, dtype=float),
                33: np.identity(2, dtype=float),
                34: np.identity(2, dtype=float),
                41: np.identity(2, dtype=float),
                42: np.identity(2, dtype=float),
                43: np.identity(2, dtype=float),
                44: np.identity(2, dtype=float)
            }
            self._Fp = {
                11: np.identity(2, dtype=float),
                12: np.identity(2, dtype=float),
                13: np.identity(2, dtype=float),
                14: np.identity(2, dtype=float),
                21: np.identity(2, dtype=float),
                22: np.identity(2, dtype=float),
                23: np.identity(2, dtype=float),
                24: np.identity(2, dtype=float),
                31: np.identity(2, dtype=float),
                32: np.identity(2, dtype=float),
                33: np.identity(2, dtype=float),
                34: np.identity(2, dtype=float),
                41: np.identity(2, dtype=float),
                42: np.identity(2, dtype=float),
                43: np.identity(2, dtype=float),
                44: np.identity(2, dtype=float)
            }
            self._Fc = {
                11: np.identity(2, dtype=float),
                12: np.identity(2, dtype=float),
                13: np.identity(2, dtype=float),
                14: np.identity(2, dtype=float),
                21: np.identity(2, dtype=float),
                22: np.identity(2, dtype=float),
                23: np.identity(2, dtype=float),
                24: np.identity(2, dtype=float),
                31: np.identity(2, dtype=float),
                32: np.identity(2, dtype=float),
                33: np.identity(2, dtype=float),
                34: np.identity(2, dtype=float),
                41: np.identity(2, dtype=float),
                42: np.identity(2, dtype=float),
                43: np.identity(2, dtype=float),
                44: np.identity(2, dtype=float)
            }
            self._Fn = {
                11: np.identity(2, dtype=float),
                12: np.identity(2, dtype=float),
                13: np.identity(2, dtype=float),
                14: np.identity(2, dtype=float),
                21: np.identity(2, dtype=float),
                22: np.identity(2, dtype=float),
                23: np.identity(2, dtype=float),
                24: np.identity(2, dtype=float),
                31: np.identity(2, dtype=float),
                32: np.identity(2, dtype=float),
                33: np.identity(2, dtype=float),
                34: np.identity(2, dtype=float),
                41: np.identity(2, dtype=float),
                42: np.identity(2, dtype=float),
                43: np.identity(2, dtype=float),
                44: np.identity(2, dtype=float)
            }

        else:  # pentGaussPts = 7
            # displacement gradients located at the Gauss points of pentagon
            self._G0 = {
                11: np.zeros((2, 2), dtype=float),
                12: np.zeros((2, 2), dtype=float),
                13: np.zeros((2, 2), dtype=float),
                14: np.zeros((2, 2), dtype=float),
                15: np.zeros((2, 2), dtype=float),
                16: np.zeros((2, 2), dtype=float),
                17: np.zeros((2, 2), dtype=float),
                21: np.zeros((2, 2), dtype=float),
                22: np.zeros((2, 2), dtype=float),
                23: np.zeros((2, 2), dtype=float),
                24: np.zeros((2, 2), dtype=float),
                25: np.zeros((2, 2), dtype=float),
                26: np.zeros((2, 2), dtype=float),
                27: np.zeros((2, 2), dtype=float),
                31: np.zeros((2, 2), dtype=float),
                32: np.zeros((2, 2), dtype=float),
                33: np.zeros((2, 2), dtype=float),
                34: np.zeros((2, 2), dtype=float),
                35: np.zeros((2, 2), dtype=float),
                36: np.zeros((2, 2), dtype=float),
                37: np.zeros((2, 2), dtype=float),
                41: np.zeros((2, 2), dtype=float),
                42: np.zeros((2, 2), dtype=float),
                43: np.zeros((2, 2), dtype=float),
                44: np.zeros((2, 2), dtype=float),
                45: np.zeros((2, 2), dtype=float),
                46: np.zeros((2, 2), dtype=float),
                47: np.zeros((2, 2), dtype=float),
                51: np.zeros((2, 2), dtype=float),
                52: np.zeros((2, 2), dtype=float),
                53: np.zeros((2, 2), dtype=float),
                54: np.zeros((2, 2), dtype=float),
                55: np.zeros((2, 2), dtype=float),
                56: np.zeros((2, 2), dtype=float),
                57: np.zeros((2, 2), dtype=float),
                61: np.zeros((2, 2), dtype=float),
                62: np.zeros((2, 2), dtype=float),
                63: np.zeros((2, 2), dtype=float),
                64: np.zeros((2, 2), dtype=float),
                65: np.zeros((2, 2), dtype=float),
                66: np.zeros((2, 2), dtype=float),
                67: np.zeros((2, 2), dtype=float),
                71: np.zeros((2, 2), dtype=float),
                72: np.zeros((2, 2), dtype=float),
                73: np.zeros((2, 2), dtype=float),
                74: np.zeros((2, 2), dtype=float),
                75: np.zeros((2, 2), dtype=float),
                76: np.zeros((2, 2), dtype=float),
                77: np.zeros((2, 2), dtype=float)
            }
            self._Gp = {
                11: np.zeros((2, 2), dtype=float),
                12: np.zeros((2, 2), dtype=float),
                13: np.zeros((2, 2), dtype=float),
                14: np.zeros((2, 2), dtype=float),
                15: np.zeros((2, 2), dtype=float),
                16: np.zeros((2, 2), dtype=float),
                17: np.zeros((2, 2), dtype=float),
                21: np.zeros((2, 2), dtype=float),
                22: np.zeros((2, 2), dtype=float),
                23: np.zeros((2, 2), dtype=float),
                24: np.zeros((2, 2), dtype=float),
                25: np.zeros((2, 2), dtype=float),
                26: np.zeros((2, 2), dtype=float),
                27: np.zeros((2, 2), dtype=float),
                31: np.zeros((2, 2), dtype=float),
                32: np.zeros((2, 2), dtype=float),
                33: np.zeros((2, 2), dtype=float),
                34: np.zeros((2, 2), dtype=float),
                35: np.zeros((2, 2), dtype=float),
                36: np.zeros((2, 2), dtype=float),
                37: np.zeros((2, 2), dtype=float),
                41: np.zeros((2, 2), dtype=float),
                42: np.zeros((2, 2), dtype=float),
                43: np.zeros((2, 2), dtype=float),
                44: np.zeros((2, 2), dtype=float),
                45: np.zeros((2, 2), dtype=float),
                46: np.zeros((2, 2), dtype=float),
                47: np.zeros((2, 2), dtype=float),
                51: np.zeros((2, 2), dtype=float),
                52: np.zeros((2, 2), dtype=float),
                53: np.zeros((2, 2), dtype=float),
                54: np.zeros((2, 2), dtype=float),
                55: np.zeros((2, 2), dtype=float),
                56: np.zeros((2, 2), dtype=float),
                57: np.zeros((2, 2), dtype=float),
                61: np.zeros((2, 2), dtype=float),
                62: np.zeros((2, 2), dtype=float),
                63: np.zeros((2, 2), dtype=float),
                64: np.zeros((2, 2), dtype=float),
                65: np.zeros((2, 2), dtype=float),
                66: np.zeros((2, 2), dtype=float),
                67: np.zeros((2, 2), dtype=float),
                71: np.zeros((2, 2), dtype=float),
                72: np.zeros((2, 2), dtype=float),
                73: np.zeros((2, 2), dtype=float),
                74: np.zeros((2, 2), dtype=float),
                75: np.zeros((2, 2), dtype=float),
                76: np.zeros((2, 2), dtype=float),
                77: np.zeros((2, 2), dtype=float)
            }
            self._Gc = {
                11: np.zeros((2, 2), dtype=float),
                12: np.zeros((2, 2), dtype=float),
                13: np.zeros((2, 2), dtype=float),
                14: np.zeros((2, 2), dtype=float),
                15: np.zeros((2, 2), dtype=float),
                16: np.zeros((2, 2), dtype=float),
                17: np.zeros((2, 2), dtype=float),
                21: np.zeros((2, 2), dtype=float),
                22: np.zeros((2, 2), dtype=float),
                23: np.zeros((2, 2), dtype=float),
                24: np.zeros((2, 2), dtype=float),
                25: np.zeros((2, 2), dtype=float),
                26: np.zeros((2, 2), dtype=float),
                27: np.zeros((2, 2), dtype=float),
                31: np.zeros((2, 2), dtype=float),
                32: np.zeros((2, 2), dtype=float),
                33: np.zeros((2, 2), dtype=float),
                34: np.zeros((2, 2), dtype=float),
                35: np.zeros((2, 2), dtype=float),
                36: np.zeros((2, 2), dtype=float),
                37: np.zeros((2, 2), dtype=float),
                41: np.zeros((2, 2), dtype=float),
                42: np.zeros((2, 2), dtype=float),
                43: np.zeros((2, 2), dtype=float),
                44: np.zeros((2, 2), dtype=float),
                45: np.zeros((2, 2), dtype=float),
                46: np.zeros((2, 2), dtype=float),
                47: np.zeros((2, 2), dtype=float),
                51: np.zeros((2, 2), dtype=float),
                52: np.zeros((2, 2), dtype=float),
                53: np.zeros((2, 2), dtype=float),
                54: np.zeros((2, 2), dtype=float),
                55: np.zeros((2, 2), dtype=float),
                56: np.zeros((2, 2), dtype=float),
                57: np.zeros((2, 2), dtype=float),
                61: np.zeros((2, 2), dtype=float),
                62: np.zeros((2, 2), dtype=float),
                63: np.zeros((2, 2), dtype=float),
                64: np.zeros((2, 2), dtype=float),
                65: np.zeros((2, 2), dtype=float),
                66: np.zeros((2, 2), dtype=float),
                67: np.zeros((2, 2), dtype=float),
                71: np.zeros((2, 2), dtype=float),
                72: np.zeros((2, 2), dtype=float),
                73: np.zeros((2, 2), dtype=float),
                74: np.zeros((2, 2), dtype=float),
                75: np.zeros((2, 2), dtype=float),
                76: np.zeros((2, 2), dtype=float),
                77: np.zeros((2, 2), dtype=float)
            }
            self._Gn = {
                11: np.zeros((2, 2), dtype=float),
                12: np.zeros((2, 2), dtype=float),
                13: np.zeros((2, 2), dtype=float),
                14: np.zeros((2, 2), dtype=float),
                15: np.zeros((2, 2), dtype=float),
                16: np.zeros((2, 2), dtype=float),
                17: np.zeros((2, 2), dtype=float),
                21: np.zeros((2, 2), dtype=float),
                22: np.zeros((2, 2), dtype=float),
                23: np.zeros((2, 2), dtype=float),
                24: np.zeros((2, 2), dtype=float),
                25: np.zeros((2, 2), dtype=float),
                26: np.zeros((2, 2), dtype=float),
                27: np.zeros((2, 2), dtype=float),
                31: np.zeros((2, 2), dtype=float),
                32: np.zeros((2, 2), dtype=float),
                33: np.zeros((2, 2), dtype=float),
                34: np.zeros((2, 2), dtype=float),
                35: np.zeros((2, 2), dtype=float),
                36: np.zeros((2, 2), dtype=float),
                37: np.zeros((2, 2), dtype=float),
                41: np.zeros((2, 2), dtype=float),
                42: np.zeros((2, 2), dtype=float),
                43: np.zeros((2, 2), dtype=float),
                44: np.zeros((2, 2), dtype=float),
                45: np.zeros((2, 2), dtype=float),
                46: np.zeros((2, 2), dtype=float),
                47: np.zeros((2, 2), dtype=float),
                51: np.zeros((2, 2), dtype=float),
                52: np.zeros((2, 2), dtype=float),
                53: np.zeros((2, 2), dtype=float),
                54: np.zeros((2, 2), dtype=float),
                55: np.zeros((2, 2), dtype=float),
                56: np.zeros((2, 2), dtype=float),
                57: np.zeros((2, 2), dtype=float),
                61: np.zeros((2, 2), dtype=float),
                62: np.zeros((2, 2), dtype=float),
                63: np.zeros((2, 2), dtype=float),
                64: np.zeros((2, 2), dtype=float),
                65: np.zeros((2, 2), dtype=float),
                66: np.zeros((2, 2), dtype=float),
                67: np.zeros((2, 2), dtype=float),
                71: np.zeros((2, 2), dtype=float),
                72: np.zeros((2, 2), dtype=float),
                73: np.zeros((2, 2), dtype=float),
                74: np.zeros((2, 2), dtype=float),
                75: np.zeros((2, 2), dtype=float),
                76: np.zeros((2, 2), dtype=float),
                77: np.zeros((2, 2), dtype=float)
            }

            # deformation gradients located at the Gauss points of pentagon
            self._F0 = {
                11: np.identity(2, dtype=float),
                12: np.identity(2, dtype=float),
                13: np.identity(2, dtype=float),
                14: np.identity(2, dtype=float),
                15: np.identity(2, dtype=float),
                16: np.identity(2, dtype=float),
                17: np.identity(2, dtype=float),
                21: np.identity(2, dtype=float),
                22: np.identity(2, dtype=float),
                23: np.identity(2, dtype=float),
                24: np.identity(2, dtype=float),
                25: np.identity(2, dtype=float),
                26: np.identity(2, dtype=float),
                27: np.identity(2, dtype=float),
                31: np.identity(2, dtype=float),
                32: np.identity(2, dtype=float),
                33: np.identity(2, dtype=float),
                34: np.identity(2, dtype=float),
                35: np.identity(2, dtype=float),
                36: np.identity(2, dtype=float),
                37: np.identity(2, dtype=float),
                41: np.identity(2, dtype=float),
                42: np.identity(2, dtype=float),
                43: np.identity(2, dtype=float),
                44: np.identity(2, dtype=float),
                45: np.identity(2, dtype=float),
                46: np.identity(2, dtype=float),
                47: np.identity(2, dtype=float),
                51: np.identity(2, dtype=float),
                52: np.identity(2, dtype=float),
                53: np.identity(2, dtype=float),
                54: np.identity(2, dtype=float),
                55: np.identity(2, dtype=float),
                56: np.identity(2, dtype=float),
                57: np.identity(2, dtype=float),
                61: np.identity(2, dtype=float),
                62: np.identity(2, dtype=float),
                63: np.identity(2, dtype=float),
                64: np.identity(2, dtype=float),
                65: np.identity(2, dtype=float),
                66: np.identity(2, dtype=float),
                67: np.identity(2, dtype=float),
                71: np.identity(2, dtype=float),
                72: np.identity(2, dtype=float),
                73: np.identity(2, dtype=float),
                74: np.identity(2, dtype=float),
                75: np.identity(2, dtype=float),
                76: np.identity(2, dtype=float),
                77: np.identity(2, dtype=float)
            }
            self._Fp = {
                11: np.identity(2, dtype=float),
                12: np.identity(2, dtype=float),
                13: np.identity(2, dtype=float),
                14: np.identity(2, dtype=float),
                15: np.identity(2, dtype=float),
                16: np.identity(2, dtype=float),
                17: np.identity(2, dtype=float),
                21: np.identity(2, dtype=float),
                22: np.identity(2, dtype=float),
                23: np.identity(2, dtype=float),
                24: np.identity(2, dtype=float),
                25: np.identity(2, dtype=float),
                26: np.identity(2, dtype=float),
                27: np.identity(2, dtype=float),
                31: np.identity(2, dtype=float),
                32: np.identity(2, dtype=float),
                33: np.identity(2, dtype=float),
                34: np.identity(2, dtype=float),
                35: np.identity(2, dtype=float),
                36: np.identity(2, dtype=float),
                37: np.identity(2, dtype=float),
                41: np.identity(2, dtype=float),
                42: np.identity(2, dtype=float),
                43: np.identity(2, dtype=float),
                44: np.identity(2, dtype=float),
                45: np.identity(2, dtype=float),
                46: np.identity(2, dtype=float),
                47: np.identity(2, dtype=float),
                51: np.identity(2, dtype=float),
                52: np.identity(2, dtype=float),
                53: np.identity(2, dtype=float),
                54: np.identity(2, dtype=float),
                55: np.identity(2, dtype=float),
                56: np.identity(2, dtype=float),
                57: np.identity(2, dtype=float),
                61: np.identity(2, dtype=float),
                62: np.identity(2, dtype=float),
                63: np.identity(2, dtype=float),
                64: np.identity(2, dtype=float),
                65: np.identity(2, dtype=float),
                66: np.identity(2, dtype=float),
                67: np.identity(2, dtype=float),
                71: np.identity(2, dtype=float),
                72: np.identity(2, dtype=float),
                73: np.identity(2, dtype=float),
                74: np.identity(2, dtype=float),
                75: np.identity(2, dtype=float),
                76: np.identity(2, dtype=float),
                77: np.identity(2, dtype=float)
            }
            self._Fc = {
                11: np.identity(2, dtype=float),
                12: np.identity(2, dtype=float),
                13: np.identity(2, dtype=float),
                14: np.identity(2, dtype=float),
                15: np.identity(2, dtype=float),
                16: np.identity(2, dtype=float),
                17: np.identity(2, dtype=float),
                21: np.identity(2, dtype=float),
                22: np.identity(2, dtype=float),
                23: np.identity(2, dtype=float),
                24: np.identity(2, dtype=float),
                25: np.identity(2, dtype=float),
                26: np.identity(2, dtype=float),
                27: np.identity(2, dtype=float),
                31: np.identity(2, dtype=float),
                32: np.identity(2, dtype=float),
                33: np.identity(2, dtype=float),
                34: np.identity(2, dtype=float),
                35: np.identity(2, dtype=float),
                36: np.identity(2, dtype=float),
                37: np.identity(2, dtype=float),
                41: np.identity(2, dtype=float),
                42: np.identity(2, dtype=float),
                43: np.identity(2, dtype=float),
                44: np.identity(2, dtype=float),
                45: np.identity(2, dtype=float),
                46: np.identity(2, dtype=float),
                47: np.identity(2, dtype=float),
                51: np.identity(2, dtype=float),
                52: np.identity(2, dtype=float),
                53: np.identity(2, dtype=float),
                54: np.identity(2, dtype=float),
                55: np.identity(2, dtype=float),
                56: np.identity(2, dtype=float),
                57: np.identity(2, dtype=float),
                61: np.identity(2, dtype=float),
                62: np.identity(2, dtype=float),
                63: np.identity(2, dtype=float),
                64: np.identity(2, dtype=float),
                65: np.identity(2, dtype=float),
                66: np.identity(2, dtype=float),
                67: np.identity(2, dtype=float),
                71: np.identity(2, dtype=float),
                72: np.identity(2, dtype=float),
                73: np.identity(2, dtype=float),
                74: np.identity(2, dtype=float),
                75: np.identity(2, dtype=float),
                76: np.identity(2, dtype=float),
                77: np.identity(2, dtype=float)
            }
            self._Fn = {
                11: np.identity(2, dtype=float),
                12: np.identity(2, dtype=float),
                13: np.identity(2, dtype=float),
                14: np.identity(2, dtype=float),
                15: np.identity(2, dtype=float),
                16: np.identity(2, dtype=float),
                17: np.identity(2, dtype=float),
                21: np.identity(2, dtype=float),
                22: np.identity(2, dtype=float),
                23: np.identity(2, dtype=float),
                24: np.identity(2, dtype=float),
                25: np.identity(2, dtype=float),
                26: np.identity(2, dtype=float),
                27: np.identity(2, dtype=float),
                31: np.identity(2, dtype=float),
                32: np.identity(2, dtype=float),
                33: np.identity(2, dtype=float),
                34: np.identity(2, dtype=float),
                35: np.identity(2, dtype=float),
                36: np.identity(2, dtype=float),
                37: np.identity(2, dtype=float),
                41: np.identity(2, dtype=float),
                42: np.identity(2, dtype=float),
                43: np.identity(2, dtype=float),
                44: np.identity(2, dtype=float),
                45: np.identity(2, dtype=float),
                46: np.identity(2, dtype=float),
                47: np.identity(2, dtype=float),
                51: np.identity(2, dtype=float),
                52: np.identity(2, dtype=float),
                53: np.identity(2, dtype=float),
                54: np.identity(2, dtype=float),
                55: np.identity(2, dtype=float),
                56: np.identity(2, dtype=float),
                57: np.identity(2, dtype=float),
                61: np.identity(2, dtype=float),
                62: np.identity(2, dtype=float),
                63: np.identity(2, dtype=float),
                64: np.identity(2, dtype=float),
                65: np.identity(2, dtype=float),
                66: np.identity(2, dtype=float),
                67: np.identity(2, dtype=float),
                71: np.identity(2, dtype=float),
                72: np.identity(2, dtype=float),
                73: np.identity(2, dtype=float),
                74: np.identity(2, dtype=float),
                75: np.identity(2, dtype=float),
                76: np.identity(2, dtype=float),
                77: np.identity(2, dtype=float)
            }

        # assign membrane objects to each Gauss point of pentagon
        if pentGaussPts == 1:
            mem11 = membrane(h)

            self._septum = {
                11: mem11
            }
        elif pentGaussPts == 4:
            mem11 = membrane(h)
            mem12 = membrane(h)
            mem13 = membrane(h)
            mem14 = membrane(h)
            mem21 = membrane(h)
            mem22 = membrane(h)
            mem23 = membrane(h)
            mem24 = membrane(h)
            mem31 = membrane(h)
            mem32 = membrane(h)
            mem33 = membrane(h)
            mem34 = membrane(h)
            mem41 = membrane(h)
            mem42 = membrane(h)
            mem43 = membrane(h)
            mem44 = membrane(h)

            self._septum = {
                11: mem11,
                12: mem12,
                13: mem13,
                14: mem14,
                21: mem21,
                22: mem22,
                23: mem23,
                24: mem24,
                31: mem31,
                32: mem32,
                33: mem33,
                34: mem34,
                41: mem41,
                42: mem42,
                43: mem43,
                44: mem44
            }
        else:  # pentGaussPts = 7
            mem11 = membrane(h)
            mem12 = membrane(h)
            mem13 = membrane(h)
            mem14 = membrane(h)
            mem15 = membrane(h)
            mem16 = membrane(h)
            mem17 = membrane(h)
            mem21 = membrane(h)
            mem22 = membrane(h)
            mem23 = membrane(h)
            mem24 = membrane(h)
            mem25 = membrane(h)
            mem26 = membrane(h)
            mem27 = membrane(h)
            mem31 = membrane(h)
            mem32 = membrane(h)
            mem33 = membrane(h)
            mem34 = membrane(h)
            mem35 = membrane(h)
            mem36 = membrane(h)
            mem37 = membrane(h)
            mem41 = membrane(h)
            mem42 = membrane(h)
            mem43 = membrane(h)
            mem44 = membrane(h)
            mem45 = membrane(h)
            mem46 = membrane(h)
            mem47 = membrane(h)
            mem51 = membrane(h)
            mem52 = membrane(h)
            mem53 = membrane(h)
            mem54 = membrane(h)
            mem55 = membrane(h)
            mem56 = membrane(h)
            mem57 = membrane(h)
            mem61 = membrane(h)
            mem62 = membrane(h)
            mem63 = membrane(h)
            mem64 = membrane(h)
            mem65 = membrane(h)
            mem66 = membrane(h)
            mem67 = membrane(h)
            mem71 = membrane(h)
            mem72 = membrane(h)
            mem73 = membrane(h)
            mem74 = membrane(h)
            mem75 = membrane(h)
            mem76 = membrane(h)
            mem77 = membrane(h)

            self._septum = {
                11: mem11,
                12: mem12,
                13: mem13,
                14: mem14,
                15: mem15,
                16: mem16,
                17: mem17,
                21: mem21,
                22: mem22,
                23: mem23,
                24: mem24,
                25: mem25,
                26: mem26,
                27: mem27,
                31: mem31,
                32: mem32,
                33: mem33,
                34: mem34,
                35: mem35,
                36: mem36,
                37: mem37,
                41: mem41,
                42: mem42,
                43: mem43,
                44: mem44,
                45: mem45,
                46: mem46,
                47: mem47,
                51: mem51,
                52: mem52,
                53: mem53,
                54: mem54,
                55: mem55,
                56: mem56,
                57: mem57,
                61: mem61,
                62: mem62,
                63: mem63,
                64: mem64,
                65: mem65,
                66: mem66,
                67: mem67,
                71: mem71,
                72: mem72,
                73: mem73,
                74: mem74,
                75: mem75,
                76: mem76,
                77: mem77
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

    def pentGaussPoints(self):
        return self._pentGaussPts

    def triaGaussPoints(self):
        return self._triaGaussPts
    
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
        self._v1z = n3x * x1 + n3y * y1 + n3z * z1
        self._v2z = n3x * x2 + n3y * y2 + n3z * z2
        self._v3z = n3x * x3 + n3y * y3 + n3z * z3
        self._v4z = n3x * x4 + n3y * y4 + n3z * z4
        self._v5z = n3x * x5 + n3y * y5 + n3z * z5

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
        if self._pentGaussPts == 1:
            # displacement gradients located at the Gauss points of pentagon
            self._Gn[11] = self._pentShapeFns[11].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            # deformation gradients located at the Gauss points of pentagon
            self._Fn[11] = self._pentShapeFns[11].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
        elif self._pentGaussPts == 4:
            # displacement gradients located at the Gauss points of pentagon
            self._Gn[11] = self._pentShapeFns[11].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[12] = self._pentShapeFns[12].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[13] = self._pentShapeFns[13].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[14] = self._pentShapeFns[14].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Gn[21] = self._pentShapeFns[21].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[22] = self._pentShapeFns[22].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[23] = self._pentShapeFns[23].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[24] = self._pentShapeFns[24].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Gn[31] = self._pentShapeFns[31].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[32] = self._pentShapeFns[32].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[33] = self._pentShapeFns[33].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[34] = self._pentShapeFns[34].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Gn[41] = self._pentShapeFns[41].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[42] = self._pentShapeFns[42].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[43] = self._pentShapeFns[43].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[44] = self._pentShapeFns[44].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            # deformation gradients located at the Gauss points of pentagon
            self._Fn[11] = self._pentShapeFns[11].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[12] = self._pentShapeFns[12].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[13] = self._pentShapeFns[13].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[14] = self._pentShapeFns[14].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Fn[21] = self._pentShapeFns[21].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[22] = self._pentShapeFns[22].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[23] = self._pentShapeFns[23].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[24] = self._pentShapeFns[24].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Fn[31] = self._pentShapeFns[31].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[32] = self._pentShapeFns[32].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[33] = self._pentShapeFns[33].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[34] = self._pentShapeFns[34].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Fn[41] = self._pentShapeFns[41].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[42] = self._pentShapeFns[42].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[43] = self._pentShapeFns[43].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[44] = self._pentShapeFns[44].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
        else:  # pentGaussPts = 7
            # displacement gradients located at the Gauss points of pentagon
            self._Gn[11] = self._pentShapeFns[11].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[12] = self._pentShapeFns[12].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[13] = self._pentShapeFns[13].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[14] = self._pentShapeFns[14].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[15] = self._pentShapeFns[15].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[16] = self._pentShapeFns[16].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[17] = self._pentShapeFns[17].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Gn[21] = self._pentShapeFns[21].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[22] = self._pentShapeFns[22].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[23] = self._pentShapeFns[23].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[24] = self._pentShapeFns[24].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[25] = self._pentShapeFns[25].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[26] = self._pentShapeFns[26].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[27] = self._pentShapeFns[27].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Gn[31] = self._pentShapeFns[31].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[32] = self._pentShapeFns[32].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[33] = self._pentShapeFns[33].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[34] = self._pentShapeFns[34].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[35] = self._pentShapeFns[35].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[36] = self._pentShapeFns[36].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[37] = self._pentShapeFns[37].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Gn[41] = self._pentShapeFns[41].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[42] = self._pentShapeFns[42].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[43] = self._pentShapeFns[43].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[44] = self._pentShapeFns[44].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[45] = self._pentShapeFns[45].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[46] = self._pentShapeFns[46].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[47] = self._pentShapeFns[47].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Gn[51] = self._pentShapeFns[51].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[52] = self._pentShapeFns[52].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[53] = self._pentShapeFns[53].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[54] = self._pentShapeFns[54].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[55] = self._pentShapeFns[55].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[56] = self._pentShapeFns[56].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[57] = self._pentShapeFns[57].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Gn[61] = self._pentShapeFns[61].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[62] = self._pentShapeFns[62].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[63] = self._pentShapeFns[63].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[64] = self._pentShapeFns[64].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[65] = self._pentShapeFns[65].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[66] = self._pentShapeFns[66].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[67] = self._pentShapeFns[67].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Gn[71] = self._pentShapeFns[71].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[72] = self._pentShapeFns[72].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[73] = self._pentShapeFns[73].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[74] = self._pentShapeFns[74].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[75] = self._pentShapeFns[75].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[76] = self._pentShapeFns[76].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Gn[77] = self._pentShapeFns[77].G(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            # deformation gradients located at the Gauss points of pentagon
            self._Fn[11] = self._pentShapeFns[11].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[12] = self._pentShapeFns[12].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[13] = self._pentShapeFns[13].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[14] = self._pentShapeFns[14].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[15] = self._pentShapeFns[15].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[16] = self._pentShapeFns[16].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[17] = self._pentShapeFns[17].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Fn[21] = self._pentShapeFns[21].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[22] = self._pentShapeFns[22].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[23] = self._pentShapeFns[23].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[24] = self._pentShapeFns[24].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[25] = self._pentShapeFns[25].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[26] = self._pentShapeFns[26].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[27] = self._pentShapeFns[27].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Fn[31] = self._pentShapeFns[31].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[32] = self._pentShapeFns[32].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[33] = self._pentShapeFns[33].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[34] = self._pentShapeFns[34].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[35] = self._pentShapeFns[35].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[36] = self._pentShapeFns[36].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[37] = self._pentShapeFns[37].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Fn[41] = self._pentShapeFns[41].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[42] = self._pentShapeFns[42].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[43] = self._pentShapeFns[43].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[44] = self._pentShapeFns[44].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[45] = self._pentShapeFns[45].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[46] = self._pentShapeFns[46].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[47] = self._pentShapeFns[47].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Fn[51] = self._pentShapeFns[51].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[52] = self._pentShapeFns[52].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[53] = self._pentShapeFns[53].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[54] = self._pentShapeFns[54].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[55] = self._pentShapeFns[55].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[56] = self._pentShapeFns[56].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[57] = self._pentShapeFns[57].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Fn[61] = self._pentShapeFns[61].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[62] = self._pentShapeFns[62].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[63] = self._pentShapeFns[63].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[64] = self._pentShapeFns[64].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[65] = self._pentShapeFns[65].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[66] = self._pentShapeFns[66].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[67] = self._pentShapeFns[67].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            
            self._Fn[71] = self._pentShapeFns[71].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[72] = self._pentShapeFns[72].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[73] = self._pentShapeFns[73].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[74] = self._pentShapeFns[74].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[75] = self._pentShapeFns[75].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[76] = self._pentShapeFns[76].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)
            self._Fn[77] = self._pentShapeFns[77].F(x1, x2, x3, x4, x5,
                                               x10, x20, x30, x40, x50)

        # update the membrane objects at each Gauss point
        if self._pentGaussPts == 1:
            for i in range(11, self._pentGaussPts+11):
                self._septum[i].update(self._Fn[i])
                
        elif self._pentGaussPts == 4:   
            for i in range(11, self._pentGaussPts+11):
                self._septum[i].update(self._Fn[i])            
            for i in range(21, self._pentGaussPts+21):
                self._septum[i].update(self._Fn[i])                
            for i in range(31, self._pentGaussPts+31):
                self._septum[i].update(self._Fn[i])                
            for i in range(41, self._pentGaussPts+41):
                self._septum[i].update(self._Fn[i])
                
        else:  # pentGaussPts = 7
            for i in range(11, self._pentGaussPts+11):
                self._septum[i].update(self._Fn[i])            
            for i in range(21, self._pentGaussPts+21):
                self._septum[i].update(self._Fn[i])                
            for i in range(31, self._pentGaussPts+31):
                self._septum[i].update(self._Fn[i])                
            for i in range(41, self._pentGaussPts+41):
                self._septum[i].update(self._Fn[i])
            for i in range(51, self._pentGaussPts+51):
                self._septum[i].update(self._Fn[i])                
            for i in range(61, self._pentGaussPts+61):
                self._septum[i].update(self._Fn[i])                
            for i in range(71, self._pentGaussPts+71):
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
        if self._pentGaussPts == 1:
            for i in range(11, self._pentGaussPts+11):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
                
        elif self._pentGaussPts == 4:   
            for i in range(11, self._pentGaussPts+11):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]         
            for i in range(21, self._pentGaussPts+21):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]             
            for i in range(31, self._pentGaussPts+31):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]             
            for i in range(41, self._pentGaussPts+41):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
                
        else:  # pentGaussPts = 7
            for i in range(11, self._pentGaussPts+11):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]           
            for i in range(21, self._pentGaussPts+21):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]              
            for i in range(31, self._pentGaussPts+31):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]               
            for i in range(41, self._pentGaussPts+41):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
            for i in range(51, self._pentGaussPts+51):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]               
            for i in range(61, self._pentGaussPts+61):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]              
            for i in range(71, self._pentGaussPts+71):
                self._Fp[i][:, :] = self._Fc[i][:, :]
                self._Fc[i][:, :] = self._Fn[i][:, :]
                self._Gp[i][:, :] = self._Gc[i][:, :]
                self._Gc[i][:, :] = self._Gn[i][:, :]
        

        # advance the membrane objects at each Gauss point
        if self._pentGaussPts == 1:
            for i in range(11, self._pentGaussPts+11):
                self._septum[i].advance()
                
        elif self._pentGaussPts == 4:   
            for i in range(11, self._pentGaussPts+11):
                self._septum[i].advance()
            for i in range(21, self._pentGaussPts+21):
                self._septum[i].advance()
            for i in range(31, self._pentGaussPts+31):
                self._septum[i].advance()
            for i in range(41, self._pentGaussPts+41):
                self._septum[i].advance()
                
        else:  # pentGaussPts = 7
            for i in range(11, self._pentGaussPts+11):
                self._septum[i].advance()
            for i in range(21, self._pentGaussPts+21):
                self._septum[i].advance()
            for i in range(31, self._pentGaussPts+31):
                self._septum[i].advance()
            for i in range(41, self._pentGaussPts+41):
                self._septum[i].advance()
            for i in range(51, self._pentGaussPts+51):
                self._septum[i].advance()
            for i in range(61, self._pentGaussPts+61):
                self._septum[i].advance()
            for i in range(71, self._pentGaussPts+71):
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
    def G(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.G and you sent {}.".format(pentGaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Gc[pentGaussPt])
            elif state == 'n' or state == 'next':
                return np.copy(self._Gn[pentGaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Gp[pentGaussPt])
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._G0[pentGaussPt])
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.G.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.G.")

    # deformation gradient at a Gauss point
    def F(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.F and you sent {}.".format(pentGaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Fc[pentGaussPt])
            elif state == 'n' or state == 'next':
                return np.copy(self._Fn[pentGaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Fp[pentGaussPt])
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._F0[pentGaussPt])
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to pentagon.F.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to pentagon.F.")

    # velocity gradient at a Gauss point
    def L(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.L and you sent {}.".format(pentGaussPt))

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
                dF = ((self._Fn[pentGaussPt] - self._Fp[pentGaussPt])
                      / (2.0 * self._h))
                fInv = FInv(self._Fc[pentGaussPt])
            elif state == 'n' or state == 'next':
                # use backward difference scheme
                dF = ((3.0 * self._Fn[pentGaussPt] - 4.0 * self._Fc[pentGaussPt] +
                       self._Fp[pentGaussPt]) / (2.0 * self._h))
                fInv = FInv(self._Fn[pentGaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use forward difference scheme
                dF = ((-self._Fn[pentGaussPt] + 4.0 * self._Fc[pentGaussPt] -
                       3.0 * self._Fp[pentGaussPt]) / (2.0 * self._h))
                fInv = FInv(self._Fp[pentGaussPt])
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
    def Q(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.Q and you sent {}.".format(pentGaussPt))
        return self._septum[pentGaussPt].Q(state)

    # the orthogonal matrix from a Gram-Schmidt factorization of (relabeled) F
    def R(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.R and you sent {}.".format(pentGaussPt))
        return self._septum[pentGaussPt].R(state)

    # a skew-symmetric matrix for the spin associated with R
    def Omega(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.Omega, you sent {}.".format(pentGaussPt))
        return self._septum[pentGaussPt].spin(state)

    # Laplace stretch from a Gram-Schmidt factorization of (relabeled) F
    def U(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.U and you sent {}.".format(pentGaussPt))
        return self._septum[pentGaussPt].U(state)

    # inverse Laplace stretch from Gram-Schmidt factorization of (relabeled) F
    def UInv(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.UInv, you sent {}.".format(pentGaussPt))
        return self._septum[pentGaussPt].UInv(state)

    # differential in the Laplace stretch from Gram-Schmidt factorization of F
    def dU(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.dU and you sent {}.".format(pentGaussPt))
        return self._septum[pentGaussPt].dU(state)

    # inverse Laplace stretch from Gram-Schmidt factorization of (relabeled) F
    def dUInv(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.dUInv, you sent {}.".format(pentGaussPt))
        return self._septum[pentGaussPt].dUInv(state)

    # physical kinematic attributes at a Gauss point in the pentagon

    def dilation(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.dilation and you sent " +
                               "{}.".format(pentGaussPt))
        return self._septum[pentGaussPt].dilation(state)

    def squeeze(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.squeeze and you sent " +
                               "{}.".format(pentGaussPt))
        return self._septum[pentGaussPt].squeeze(state)

    def shear(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.shear and you sent " +
                               "{}.".format(pentGaussPt))
        return self._septum[pentGaussPt].shear(state)

    def dDilation(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.dDilation and you sent " +
                               "{}.".format(pentGaussPt))
        return self._septum[pentGaussPt].dDilation(state)

    def dSqueeze(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.dSqueeze and you sent " +
                               "{}.".format(pentGaussPt))
        return self._septum[pentGaussPt].dSqueeze(state)

    def dShear(self, pentGaussPt, state):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("The pentGaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._pentGaussPts) +
                               "pentagon.dShear and you sent " +
                               "{}.".format(pentGaussPt))
        return self._septum[pentGaussPt].dShear(state)

    # properties used in finite elements

    def shapeFunction(self, pentGaussPt):
        if (pentGaussPt < 1) or (pentGaussPt > self._pentGaussPts):
            raise RuntimeError("pentGaussPt must be in the range of " +
                               "[1, {}] in call ".format(self._pentGaussPts) +
                               "to pentagon.shapeFunction and you sent " +
                               "{}.".format(pentGaussPt))
            sf = self._pentShapeFns[pentGaussPt]
        return sf

    def massMatrix(self):
        # current vertex coordinates in pentagonal frame of reference
        x1 = (self._v1x, self._v1y)
        x2 = (self._v2x, self._v2y)
        x3 = (self._v3x, self._v3y)
        x4 = (self._v4x, self._v4y)
        x5 = (self._v5x, self._v5y)

        # determine the mass matrix
        if self._pentGaussPts == 1:
            # 'natural' weight of the element
            w = np.array([2.3776412907378837])

            jacob = self._pentShapeFns[11].jacobian(x1, x2, x3, x4, x5)
            
            # determinant of the Jacobian matrix
            detJ = det(jacob)
            
            nn1 = np.dot(np.transpose(self._pentShapeFns[11].Nmatx),
                         self._pentShapeFns[11].Nmatx)

            # the consistent mass matrix for 1 Gauss point
            massC = self._rho * self._width * (detJ * w[0] * w[0] * nn1)

            # the lumped mass matrix for 1 Gauss point
            massL = np.zeros((10, 10), dtype=float)
            row, col = np.diag_indices_from(massC)
            massL[row, col] = massC.sum(axis=1)

            # the mass matrix is the average of the above two mass matrices
            mass = 0.5 * (massC + massL)
        elif self._pentGaussPts == 4:
            # 'natural' weights of the element
            w = np.array([0.5449124407446143, 0.6439082046243272,
                            0.5449124407446146, 0.6439082046243275])

            jacob11 = self._pentShapeFns[11].jacobian(x1, x2, x3, x4, x5)
            jacob12 = self._pentShapeFns[12].jacobian(x1, x2, x3, x4, x5)
            jacob13 = self._pentShapeFns[13].jacobian(x1, x2, x3, x4, x5)
            jacob14 = self._pentShapeFns[14].jacobian(x1, x2, x3, x4, x5)
            
            jacob21 = self._pentShapeFns[21].jacobian(x1, x2, x3, x4, x5)
            jacob22 = self._pentShapeFns[22].jacobian(x1, x2, x3, x4, x5)
            jacob23 = self._pentShapeFns[23].jacobian(x1, x2, x3, x4, x5)
            jacob24 = self._pentShapeFns[24].jacobian(x1, x2, x3, x4, x5)
            
            jacob31 = self._pentShapeFns[31].jacobian(x1, x2, x3, x4, x5)
            jacob32 = self._pentShapeFns[32].jacobian(x1, x2, x3, x4, x5)
            jacob33 = self._pentShapeFns[33].jacobian(x1, x2, x3, x4, x5)
            jacob34 = self._pentShapeFns[34].jacobian(x1, x2, x3, x4, x5)
            
            jacob41 = self._pentShapeFns[41].jacobian(x1, x2, x3, x4, x5)
            jacob42 = self._pentShapeFns[42].jacobian(x1, x2, x3, x4, x5)
            jacob43 = self._pentShapeFns[43].jacobian(x1, x2, x3, x4, x5)
            jacob44 = self._pentShapeFns[44].jacobian(x1, x2, x3, x4, x5)
            

            # determinant of the Jacobian matrix
            detJ11 = det(jacob11)
            detJ12 = det(jacob12)
            detJ13 = det(jacob13)            
            detJ14 = det(jacob14)
            
            detJ21 = det(jacob21)
            detJ22 = det(jacob22)
            detJ23 = det(jacob23)            
            detJ24 = det(jacob24)
            
            detJ31 = det(jacob31)
            detJ32 = det(jacob32)
            detJ33 = det(jacob33)            
            detJ34 = det(jacob34)
            
            detJ41 = det(jacob41)
            detJ42 = det(jacob42)
            detJ43 = det(jacob43)            
            detJ44 = det(jacob44)

            nn11 = np.dot(np.transpose(self._pentShapeFns[11].Nmatx),
                         self._pentShapeFns[11].Nmatx)
            nn12 = np.dot(np.transpose(self._pentShapeFns[12].Nmatx),
                         self._pentShapeFns[12].Nmatx)
            nn13 = np.dot(np.transpose(self._pentShapeFns[13].Nmatx),
                         self._pentShapeFns[13].Nmatx)
            nn14 = np.dot(np.transpose(self._pentShapeFns[14].Nmatx),
                         self._pentShapeFns[14].Nmatx)
            
            nn21 = np.dot(np.transpose(self._pentShapeFns[21].Nmatx),
                         self._pentShapeFns[21].Nmatx)
            nn22 = np.dot(np.transpose(self._pentShapeFns[22].Nmatx),
                         self._pentShapeFns[22].Nmatx)
            nn23 = np.dot(np.transpose(self._pentShapeFns[23].Nmatx),
                         self._pentShapeFns[23].Nmatx)
            nn24 = np.dot(np.transpose(self._pentShapeFns[24].Nmatx),
                         self._pentShapeFns[24].Nmatx)
            
            nn31 = np.dot(np.transpose(self._pentShapeFns[31].Nmatx),
                         self._pentShapeFns[31].Nmatx)
            nn32 = np.dot(np.transpose(self._pentShapeFns[32].Nmatx),
                         self._pentShapeFns[32].Nmatx)
            nn33 = np.dot(np.transpose(self._pentShapeFns[33].Nmatx),
                         self._pentShapeFns[33].Nmatx)
            nn34 = np.dot(np.transpose(self._pentShapeFns[34].Nmatx),
                         self._pentShapeFns[34].Nmatx)
            
            nn41 = np.dot(np.transpose(self._pentShapeFns[41].Nmatx),
                         self._pentShapeFns[41].Nmatx)
            nn42 = np.dot(np.transpose(self._pentShapeFns[42].Nmatx),
                         self._pentShapeFns[42].Nmatx)
            nn43 = np.dot(np.transpose(self._pentShapeFns[43].Nmatx),
                         self._pentShapeFns[43].Nmatx)
            nn44 = np.dot(np.transpose(self._pentShapeFns[44].Nmatx),
                         self._pentShapeFns[44].Nmatx)

            # the consistent mass matrix for 4 Gauss points
            massC = (self._rho * self._width * (detJ11 * w[0] * w[0] * nn11 +
                                                detJ12 * w[0] * w[1] * nn12 +
                                                detJ13 * w[0] * w[2] * nn13 +
                                                detJ14 * w[0] * w[3] * nn14 +
                                                detJ21 * w[1] * w[0] * nn21 +
                                                detJ22 * w[1] * w[1] * nn22 +
                                                detJ23 * w[1] * w[2] * nn23 +
                                                detJ24 * w[1] * w[3] * nn24 +
                                                detJ31 * w[2] * w[0] * nn31 +
                                                detJ32 * w[2] * w[1] * nn32 +
                                                detJ33 * w[2] * w[2] * nn33 +
                                                detJ34 * w[2] * w[3] * nn34 +
                                                detJ41 * w[3] * w[0] * nn41 +
                                                detJ42 * w[3] * w[1] * nn42 +
                                                detJ43 * w[3] * w[2] * nn43 +
                                                detJ44 * w[3] * w[3] * nn44))

            # the lumped mass matrix for 4 Gauss points
            massL = np.zeros((10, 10), dtype=float)
            row, col = np.diag_indices_from(massC)
            massL[row, col] = massC.sum(axis=1)

            # the mass matrix is the average of the above two mass matrices
            mass = 0.5 * (massC + massL)
        else:  # pentGaussPts = 7
            # 'natural' weights of the element
            w = np.array([0.6257871064166934, 0.3016384608809768,
                          0.3169910433902452, 0.3155445150066620,
                          0.2958801959111726, 0.2575426306970870,
                          0.2642573384350463])

            jacob11 = self._pentShapeFns[11].jacobian(x1, x2, x3, x4, x5)
            jacob12 = self._pentShapeFns[12].jacobian(x1, x2, x3, x4, x5)
            jacob13 = self._pentShapeFns[13].jacobian(x1, x2, x3, x4, x5)
            jacob14 = self._pentShapeFns[14].jacobian(x1, x2, x3, x4, x5)
            jacob15 = self._pentShapeFns[15].jacobian(x1, x2, x3, x4, x5)
            jacob16 = self._pentShapeFns[16].jacobian(x1, x2, x3, x4, x5)
            jacob17 = self._pentShapeFns[17].jacobian(x1, x2, x3, x4, x5)
            
            jacob21 = self._pentShapeFns[21].jacobian(x1, x2, x3, x4, x5)
            jacob22 = self._pentShapeFns[22].jacobian(x1, x2, x3, x4, x5)
            jacob23 = self._pentShapeFns[23].jacobian(x1, x2, x3, x4, x5)
            jacob24 = self._pentShapeFns[24].jacobian(x1, x2, x3, x4, x5)
            jacob25 = self._pentShapeFns[25].jacobian(x1, x2, x3, x4, x5)
            jacob26 = self._pentShapeFns[26].jacobian(x1, x2, x3, x4, x5)
            jacob27 = self._pentShapeFns[27].jacobian(x1, x2, x3, x4, x5)
            
            jacob31 = self._pentShapeFns[31].jacobian(x1, x2, x3, x4, x5)
            jacob32 = self._pentShapeFns[32].jacobian(x1, x2, x3, x4, x5)
            jacob33 = self._pentShapeFns[33].jacobian(x1, x2, x3, x4, x5)
            jacob34 = self._pentShapeFns[34].jacobian(x1, x2, x3, x4, x5)
            jacob35 = self._pentShapeFns[35].jacobian(x1, x2, x3, x4, x5)
            jacob36 = self._pentShapeFns[36].jacobian(x1, x2, x3, x4, x5)
            jacob37 = self._pentShapeFns[37].jacobian(x1, x2, x3, x4, x5)
            
            jacob41 = self._pentShapeFns[41].jacobian(x1, x2, x3, x4, x5)
            jacob42 = self._pentShapeFns[42].jacobian(x1, x2, x3, x4, x5)
            jacob43 = self._pentShapeFns[43].jacobian(x1, x2, x3, x4, x5)
            jacob44 = self._pentShapeFns[44].jacobian(x1, x2, x3, x4, x5)
            jacob45 = self._pentShapeFns[45].jacobian(x1, x2, x3, x4, x5)
            jacob46 = self._pentShapeFns[46].jacobian(x1, x2, x3, x4, x5)
            jacob47 = self._pentShapeFns[47].jacobian(x1, x2, x3, x4, x5)
            
            jacob51 = self._pentShapeFns[51].jacobian(x1, x2, x3, x4, x5)
            jacob52 = self._pentShapeFns[52].jacobian(x1, x2, x3, x4, x5)
            jacob53 = self._pentShapeFns[53].jacobian(x1, x2, x3, x4, x5)
            jacob54 = self._pentShapeFns[54].jacobian(x1, x2, x3, x4, x5)
            jacob55 = self._pentShapeFns[55].jacobian(x1, x2, x3, x4, x5)
            jacob56 = self._pentShapeFns[56].jacobian(x1, x2, x3, x4, x5)
            jacob57 = self._pentShapeFns[57].jacobian(x1, x2, x3, x4, x5)
            
            jacob61 = self._pentShapeFns[61].jacobian(x1, x2, x3, x4, x5)
            jacob62 = self._pentShapeFns[62].jacobian(x1, x2, x3, x4, x5)
            jacob63 = self._pentShapeFns[63].jacobian(x1, x2, x3, x4, x5)
            jacob64 = self._pentShapeFns[64].jacobian(x1, x2, x3, x4, x5)
            jacob65 = self._pentShapeFns[65].jacobian(x1, x2, x3, x4, x5)
            jacob66 = self._pentShapeFns[66].jacobian(x1, x2, x3, x4, x5)
            jacob67 = self._pentShapeFns[67].jacobian(x1, x2, x3, x4, x5)
            
            jacob71 = self._pentShapeFns[71].jacobian(x1, x2, x3, x4, x5)
            jacob72 = self._pentShapeFns[72].jacobian(x1, x2, x3, x4, x5)
            jacob73 = self._pentShapeFns[73].jacobian(x1, x2, x3, x4, x5)
            jacob74 = self._pentShapeFns[74].jacobian(x1, x2, x3, x4, x5)
            jacob75 = self._pentShapeFns[75].jacobian(x1, x2, x3, x4, x5)
            jacob76 = self._pentShapeFns[76].jacobian(x1, x2, x3, x4, x5)
            jacob77 = self._pentShapeFns[77].jacobian(x1, x2, x3, x4, x5)

            # determinant of the Jacobian matrix
            detJ11 = det(jacob11)
            detJ12 = det(jacob12)
            detJ13 = det(jacob13)            
            detJ14 = det(jacob14)
            detJ15 = det(jacob15)
            detJ16 = det(jacob16)            
            detJ17 = det(jacob17)
            
            detJ21 = det(jacob21)
            detJ22 = det(jacob22)
            detJ23 = det(jacob23)            
            detJ24 = det(jacob24)
            detJ25 = det(jacob25)
            detJ26 = det(jacob26)            
            detJ27 = det(jacob27)
            
            detJ31 = det(jacob31)
            detJ32 = det(jacob32)
            detJ33 = det(jacob33)            
            detJ34 = det(jacob34)
            detJ35 = det(jacob35)
            detJ36 = det(jacob36)            
            detJ37 = det(jacob37)
            
            detJ41 = det(jacob41)
            detJ42 = det(jacob42)
            detJ43 = det(jacob43)            
            detJ44 = det(jacob44)
            detJ45 = det(jacob45)
            detJ46 = det(jacob46)            
            detJ47 = det(jacob47)
            
            detJ51 = det(jacob51)
            detJ52 = det(jacob52)
            detJ53 = det(jacob53)            
            detJ54 = det(jacob54)
            detJ55 = det(jacob55)
            detJ56 = det(jacob56)            
            detJ57 = det(jacob57)
            
            detJ61 = det(jacob61)
            detJ62 = det(jacob62)
            detJ63 = det(jacob63)            
            detJ64 = det(jacob64)
            detJ65 = det(jacob65)
            detJ66 = det(jacob66)            
            detJ67 = det(jacob67)
            
            detJ71 = det(jacob71)
            detJ72 = det(jacob72)
            detJ73 = det(jacob73)            
            detJ74 = det(jacob74)
            detJ75 = det(jacob75)
            detJ76 = det(jacob76)            
            detJ77 = det(jacob77)
            
            nn11 = np.dot(np.transpose(self._pentShapeFns[11].Nmatx),
                         self._pentShapeFns[11].Nmatx)
            nn12 = np.dot(np.transpose(self._pentShapeFns[12].Nmatx),
                         self._pentShapeFns[12].Nmatx)
            nn13 = np.dot(np.transpose(self._pentShapeFns[13].Nmatx),
                         self._pentShapeFns[13].Nmatx)
            nn14 = np.dot(np.transpose(self._pentShapeFns[14].Nmatx),
                         self._pentShapeFns[14].Nmatx)
            nn15 = np.dot(np.transpose(self._pentShapeFns[15].Nmatx),
                         self._pentShapeFns[15].Nmatx)
            nn16 = np.dot(np.transpose(self._pentShapeFns[16].Nmatx),
                         self._pentShapeFns[16].Nmatx)
            nn17 = np.dot(np.transpose(self._pentShapeFns[17].Nmatx),
                         self._pentShapeFns[17].Nmatx)
            
            nn21 = np.dot(np.transpose(self._pentShapeFns[21].Nmatx),
                         self._pentShapeFns[21].Nmatx)
            nn22 = np.dot(np.transpose(self._pentShapeFns[22].Nmatx),
                         self._pentShapeFns[22].Nmatx)
            nn23 = np.dot(np.transpose(self._pentShapeFns[23].Nmatx),
                         self._pentShapeFns[23].Nmatx)
            nn24 = np.dot(np.transpose(self._pentShapeFns[24].Nmatx),
                         self._pentShapeFns[24].Nmatx)
            nn25 = np.dot(np.transpose(self._pentShapeFns[25].Nmatx),
                         self._pentShapeFns[25].Nmatx)
            nn26 = np.dot(np.transpose(self._pentShapeFns[26].Nmatx),
                         self._pentShapeFns[26].Nmatx)
            nn27 = np.dot(np.transpose(self._pentShapeFns[27].Nmatx),
                         self._pentShapeFns[27].Nmatx)
            
            nn31 = np.dot(np.transpose(self._pentShapeFns[31].Nmatx),
                         self._pentShapeFns[31].Nmatx)
            nn32 = np.dot(np.transpose(self._pentShapeFns[32].Nmatx),
                         self._pentShapeFns[32].Nmatx)
            nn33 = np.dot(np.transpose(self._pentShapeFns[33].Nmatx),
                         self._pentShapeFns[33].Nmatx)
            nn34 = np.dot(np.transpose(self._pentShapeFns[34].Nmatx),
                         self._pentShapeFns[34].Nmatx)
            nn35 = np.dot(np.transpose(self._pentShapeFns[35].Nmatx),
                         self._pentShapeFns[35].Nmatx)
            nn36 = np.dot(np.transpose(self._pentShapeFns[36].Nmatx),
                         self._pentShapeFns[36].Nmatx)
            nn37 = np.dot(np.transpose(self._pentShapeFns[37].Nmatx),
                         self._pentShapeFns[37].Nmatx)
            
            nn41 = np.dot(np.transpose(self._pentShapeFns[41].Nmatx),
                         self._pentShapeFns[41].Nmatx)
            nn42 = np.dot(np.transpose(self._pentShapeFns[42].Nmatx),
                         self._pentShapeFns[42].Nmatx)
            nn43 = np.dot(np.transpose(self._pentShapeFns[43].Nmatx),
                         self._pentShapeFns[43].Nmatx)
            nn44 = np.dot(np.transpose(self._pentShapeFns[44].Nmatx),
                         self._pentShapeFns[44].Nmatx)
            nn45 = np.dot(np.transpose(self._pentShapeFns[45].Nmatx),
                         self._pentShapeFns[45].Nmatx)
            nn46 = np.dot(np.transpose(self._pentShapeFns[46].Nmatx),
                         self._pentShapeFns[46].Nmatx)
            nn47 = np.dot(np.transpose(self._pentShapeFns[47].Nmatx),
                         self._pentShapeFns[47].Nmatx)
            
            nn51 = np.dot(np.transpose(self._pentShapeFns[51].Nmatx),
                         self._pentShapeFns[51].Nmatx)
            nn52 = np.dot(np.transpose(self._pentShapeFns[52].Nmatx),
                         self._pentShapeFns[52].Nmatx)
            nn53 = np.dot(np.transpose(self._pentShapeFns[53].Nmatx),
                         self._pentShapeFns[53].Nmatx)
            nn54 = np.dot(np.transpose(self._pentShapeFns[54].Nmatx),
                         self._pentShapeFns[54].Nmatx)
            nn55 = np.dot(np.transpose(self._pentShapeFns[55].Nmatx),
                         self._pentShapeFns[55].Nmatx)
            nn56 = np.dot(np.transpose(self._pentShapeFns[56].Nmatx),
                         self._pentShapeFns[56].Nmatx)
            nn57 = np.dot(np.transpose(self._pentShapeFns[57].Nmatx),
                         self._pentShapeFns[57].Nmatx)
            
            nn61 = np.dot(np.transpose(self._pentShapeFns[61].Nmatx),
                         self._pentShapeFns[61].Nmatx)
            nn62 = np.dot(np.transpose(self._pentShapeFns[62].Nmatx),
                         self._pentShapeFns[62].Nmatx)
            nn63 = np.dot(np.transpose(self._pentShapeFns[63].Nmatx),
                         self._pentShapeFns[63].Nmatx)
            nn64 = np.dot(np.transpose(self._pentShapeFns[64].Nmatx),
                         self._pentShapeFns[64].Nmatx)
            nn65 = np.dot(np.transpose(self._pentShapeFns[65].Nmatx),
                         self._pentShapeFns[65].Nmatx)
            nn66 = np.dot(np.transpose(self._pentShapeFns[66].Nmatx),
                         self._pentShapeFns[66].Nmatx)
            nn67 = np.dot(np.transpose(self._pentShapeFns[67].Nmatx),
                         self._pentShapeFns[67].Nmatx)
            
            nn71 = np.dot(np.transpose(self._pentShapeFns[71].Nmatx),
                         self._pentShapeFns[71].Nmatx)
            nn72 = np.dot(np.transpose(self._pentShapeFns[72].Nmatx),
                         self._pentShapeFns[72].Nmatx)
            nn73 = np.dot(np.transpose(self._pentShapeFns[73].Nmatx),
                         self._pentShapeFns[73].Nmatx)
            nn74 = np.dot(np.transpose(self._pentShapeFns[74].Nmatx),
                         self._pentShapeFns[74].Nmatx)
            nn75 = np.dot(np.transpose(self._pentShapeFns[75].Nmatx),
                         self._pentShapeFns[75].Nmatx)
            nn76 = np.dot(np.transpose(self._pentShapeFns[76].Nmatx),
                         self._pentShapeFns[76].Nmatx)
            nn77 = np.dot(np.transpose(self._pentShapeFns[77].Nmatx),
                         self._pentShapeFns[77].Nmatx)

            # the consistent mass matrix for 7 Gauss points
            massC = (self._rho * self._width * (detJ11 * w[0] * w[0] * nn11 +
                                                detJ12 * w[0] * w[1] * nn12 +
                                                detJ13 * w[0] * w[2] * nn13 +
                                                detJ14 * w[0] * w[3] * nn14 +
                                                detJ15 * w[0] * w[4] * nn15 +
                                                detJ16 * w[0] * w[5] * nn16 +
                                                detJ17 * w[0] * w[6] * nn17 +
                                                detJ21 * w[1] * w[0] * nn21 +
                                                detJ22 * w[1] * w[1] * nn22 +
                                                detJ23 * w[1] * w[2] * nn23 +
                                                detJ24 * w[1] * w[3] * nn24 +
                                                detJ25 * w[1] * w[4] * nn25 +
                                                detJ26 * w[1] * w[5] * nn26 +
                                                detJ27 * w[1] * w[6] * nn27 +
                                                detJ31 * w[2] * w[0] * nn31 +
                                                detJ32 * w[2] * w[1] * nn32 +
                                                detJ33 * w[2] * w[2] * nn33 +
                                                detJ34 * w[2] * w[3] * nn34 +
                                                detJ35 * w[2] * w[4] * nn35 +
                                                detJ36 * w[2] * w[5] * nn36 +
                                                detJ37 * w[2] * w[6] * nn37 +
                                                detJ41 * w[3] * w[0] * nn41 +
                                                detJ42 * w[3] * w[1] * nn42 +
                                                detJ43 * w[3] * w[2] * nn43 +
                                                detJ44 * w[3] * w[3] * nn44 +
                                                detJ45 * w[3] * w[4] * nn45 +
                                                detJ46 * w[3] * w[5] * nn46 +
                                                detJ47 * w[3] * w[6] * nn47 +
                                                detJ51 * w[4] * w[0] * nn51 +
                                                detJ52 * w[4] * w[1] * nn52 +
                                                detJ53 * w[4] * w[2] * nn53 +
                                                detJ54 * w[4] * w[3] * nn54 +
                                                detJ55 * w[4] * w[4] * nn55 +
                                                detJ56 * w[4] * w[5] * nn56 +
                                                detJ57 * w[4] * w[6] * nn57 +
                                                detJ61 * w[5] * w[0] * nn61 +
                                                detJ62 * w[5] * w[1] * nn62 +
                                                detJ63 * w[5] * w[2] * nn63 +
                                                detJ64 * w[5] * w[3] * nn64 +
                                                detJ65 * w[5] * w[4] * nn65 +
                                                detJ66 * w[5] * w[5] * nn66 +
                                                detJ67 * w[5] * w[6] * nn67 +
                                                detJ71 * w[6] * w[0] * nn71 +
                                                detJ72 * w[6] * w[1] * nn72 +
                                                detJ73 * w[6] * w[2] * nn73 +
                                                detJ74 * w[6] * w[3] * nn74 +
                                                detJ75 * w[6] * w[4] * nn75 +
                                                detJ76 * w[6] * w[5] * nn76 +
                                                detJ77 * w[6] * w[6] * nn77))

            # the lumped mass matrix for 7 Gauss points
            massL = np.zeros((10, 10), dtype=float)
            row, col = np.diag_indices_from(massC)
            massL[row, col] = massC.sum(axis=1)

            # the mass matrix is the average of the above two mass matrices
            mass = 0.5 * (massC + massL)
        return mass

    def stiffnessMatrix(self, M, sp, st, ss):
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
        
        # create the stress matrix
        T = np.zeros((2, 2), dtype=float)
        T[0, 0] = sp
        T[0, 1] = st
        T[1, 0] = st
        T[1, 1] = ss
                    
        # determine the stiffness matrix
        if self._pentGaussPts == 1:
            # 'natural' weight of the element
            w = np.array([2.3776412907378837])
            
            jacob = self._pentShapeFns[11].jacobian(x1, x2, x3, x4, x5)
            
            # determinant of the Jacobian matrix
            detJ = det(jacob)
            
            # create the linear Bmatrix
            BL = self._pentShapeFns[11].BLinear(x1, x2, x3, x4, x5)
            # the linear stiffness matrix for 1 Gauss point
            KL = self._width * (detJ * w[0] * w[0] * BL.T.dot(M).dot(BL))
            
            # create the nonlinear Bmatrix
            BNF = self._pentShapeFns[11].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS = self._pentShapeFns[11].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            # total nonlinear Bmatrix
            BN = np.add(BNF, BNS)
            
            # creat the H1 matrix
            HF = self._pentShapeFns[11].HmatrixF(x1, x2, x3, x4, x5)
            # creat the H2 matrix
            HS = self._pentShapeFns[11].HmatrixS(x1, x2, x3, x4, x5)
            
            # the nonlinear stiffness matrix for 1 Gauss point
            KN = self._width * (detJ * w[0] * w[0] * (BL.T.dot(M).dot(BN) +
                                BN.T.dot(M).dot(BL) + BN.T.dot(M).dot(BN)))
            
            # create the stress stiffness matrix            
            KS = self._width * (detJ * w[0] * w[0] * HF.T.dot(T).dot(HF) +
                                detJ * w[0] * w[0] * HS.T.dot(T).dot(HS))
            
            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS
            
        elif self._pentGaussPts == 4:
            # 'natural' weights of the element
            w = np.array([0.5449124407446143, 0.6439082046243272,
                          0.5449124407446146, 0.6439082046243275])

            jacob11 = self._pentShapeFns[11].jacobian(x1, x2, x3, x4, x5)
            jacob12 = self._pentShapeFns[12].jacobian(x1, x2, x3, x4, x5)
            jacob13 = self._pentShapeFns[13].jacobian(x1, x2, x3, x4, x5)
            jacob14 = self._pentShapeFns[14].jacobian(x1, x2, x3, x4, x5)
            
            jacob21 = self._pentShapeFns[21].jacobian(x1, x2, x3, x4, x5)
            jacob22 = self._pentShapeFns[22].jacobian(x1, x2, x3, x4, x5)
            jacob23 = self._pentShapeFns[23].jacobian(x1, x2, x3, x4, x5)
            jacob24 = self._pentShapeFns[24].jacobian(x1, x2, x3, x4, x5)
            
            jacob31 = self._pentShapeFns[31].jacobian(x1, x2, x3, x4, x5)
            jacob32 = self._pentShapeFns[32].jacobian(x1, x2, x3, x4, x5)
            jacob33 = self._pentShapeFns[33].jacobian(x1, x2, x3, x4, x5)
            jacob34 = self._pentShapeFns[34].jacobian(x1, x2, x3, x4, x5)
            
            jacob41 = self._pentShapeFns[41].jacobian(x1, x2, x3, x4, x5)
            jacob42 = self._pentShapeFns[42].jacobian(x1, x2, x3, x4, x5)
            jacob43 = self._pentShapeFns[43].jacobian(x1, x2, x3, x4, x5)
            jacob44 = self._pentShapeFns[44].jacobian(x1, x2, x3, x4, x5)

            # determinant of the Jacobian matrix
            detJ11 = det(jacob11)
            detJ12 = det(jacob12)
            detJ13 = det(jacob13)            
            detJ14 = det(jacob14)
            
            detJ21 = det(jacob21)
            detJ22 = det(jacob22)
            detJ23 = det(jacob23)            
            detJ24 = det(jacob24)
            
            detJ31 = det(jacob31)
            detJ32 = det(jacob32)
            detJ33 = det(jacob33)            
            detJ34 = det(jacob34)
            
            detJ41 = det(jacob41)
            detJ42 = det(jacob42)
            detJ43 = det(jacob43)            
            detJ44 = det(jacob44)

            # create the linear Bmatrix
            BL11 = self._pentShapeFns[11].BLinear(x1, x2, x3, x4, x5) 
            BL12 = self._pentShapeFns[12].BLinear(x1, x2, x3, x4, x5) 
            BL13 = self._pentShapeFns[13].BLinear(x1, x2, x3, x4, x5)
            BL14 = self._pentShapeFns[14].BLinear(x1, x2, x3, x4, x5) 
            
            BL21 = self._pentShapeFns[21].BLinear(x1, x2, x3, x4, x5)
            BL22 = self._pentShapeFns[22].BLinear(x1, x2, x3, x4, x5) 
            BL23 = self._pentShapeFns[23].BLinear(x1, x2, x3, x4, x5) 
            BL24 = self._pentShapeFns[24].BLinear(x1, x2, x3, x4, x5) 
            
            BL31 = self._pentShapeFns[31].BLinear(x1, x2, x3, x4, x5) 
            BL32 = self._pentShapeFns[32].BLinear(x1, x2, x3, x4, x5) 
            BL33 = self._pentShapeFns[33].BLinear(x1, x2, x3, x4, x5) 
            BL34 = self._pentShapeFns[34].BLinear(x1, x2, x3, x4, x5) 
            
            BL41 = self._pentShapeFns[41].BLinear(x1, x2, x3, x4, x5)
            BL42 = self._pentShapeFns[42].BLinear(x1, x2, x3, x4, x5) 
            BL43 = self._pentShapeFns[43].BLinear(x1, x2, x3, x4, x5) 
            BL44 = self._pentShapeFns[44].BLinear(x1, x2, x3, x4, x5) 

            # the linear stiffness matrix for 4 Gauss points
            KL = self._width * (detJ11 * w[0] * w[0] * BL11.T.dot(M).dot(BL11)+
                                detJ12 * w[0] * w[1] * BL12.T.dot(M).dot(BL12)+
                                detJ13 * w[0] * w[2] * BL13.T.dot(M).dot(BL13)+
                                detJ14 * w[0] * w[3] * BL14.T.dot(M).dot(BL14)+
                                detJ21 * w[1] * w[0] * BL21.T.dot(M).dot(BL21)+
                                detJ22 * w[1] * w[1] * BL22.T.dot(M).dot(BL22)+
                                detJ23 * w[1] * w[2] * BL23.T.dot(M).dot(BL23)+
                                detJ24 * w[1] * w[3] * BL24.T.dot(M).dot(BL24)+
                                detJ31 * w[2] * w[0] * BL31.T.dot(M).dot(BL31)+
                                detJ32 * w[2] * w[1] * BL32.T.dot(M).dot(BL32)+
                                detJ33 * w[2] * w[2] * BL33.T.dot(M).dot(BL33)+
                                detJ34 * w[2] * w[3] * BL34.T.dot(M).dot(BL34)+
                                detJ41 * w[3] * w[0] * BL41.T.dot(M).dot(BL41)+
                                detJ42 * w[3] * w[1] * BL42.T.dot(M).dot(BL42)+
                                detJ43 * w[3] * w[2] * BL43.T.dot(M).dot(BL43)+
                                detJ44 * w[3] * w[3] * BL44.T.dot(M).dot(BL44))
            
            # create the first nonlinear Bmatrix
            BNF11 = self._pentShapeFns[11].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF12 = self._pentShapeFns[12].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF13 = self._pentShapeFns[13].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF14 = self._pentShapeFns[14].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNF21 = self._pentShapeFns[21].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF22 = self._pentShapeFns[22].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF23 = self._pentShapeFns[23].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF24 = self._pentShapeFns[24].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNF31 = self._pentShapeFns[31].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF32 = self._pentShapeFns[32].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF33 = self._pentShapeFns[33].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF34 = self._pentShapeFns[34].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNF41 = self._pentShapeFns[41].FirstBNonLinear(x1, x2, x3, x4, x5,
                                  x01, x02, x03, x04, x05)
            BNF42 = self._pentShapeFns[42].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF43 = self._pentShapeFns[43].FirstBNonLinear(x1, x2, x3, x4, x5,
                                  x01, x02, x03, x04, x05)
            BNF44 = self._pentShapeFns[44].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            # create the second nonlinear Bmatrix
            BNS11 = self._pentShapeFns[11].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS12 = self._pentShapeFns[12].SecondBNonLinear(x1, x2, x3, x4, x5,
                                  x01, x02, x03, x04, x05)
            BNS13 = self._pentShapeFns[13].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS14 = self._pentShapeFns[14].SecondBNonLinear(x1, x2, x3, x4, x5,
                                  x01, x02, x03, x04, x05)
            
            BNS21 = self._pentShapeFns[21].SecondBNonLinear(x1, x2, x3, x4, x5,
                                  x01, x02, x03, x04, x05)
            BNS22 = self._pentShapeFns[22].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS23 = self._pentShapeFns[23].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS24 = self._pentShapeFns[24].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNS31 = self._pentShapeFns[31].SecondBNonLinear(x1, x2, x3, x4, x5,
                                  x01, x02, x03, x04, x05)
            BNS32 = self._pentShapeFns[32].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS33 = self._pentShapeFns[33].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS34 = self._pentShapeFns[34].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNS41 = self._pentShapeFns[41].SecondBNonLinear(x1, x2, x3, x4, x5,
                                  x01, x02, x03, x04, x05)
            BNS42 = self._pentShapeFns[42].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS43 = self._pentShapeFns[43].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS44 = self._pentShapeFns[44].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            # total nonlinear Bmatrix
            BN11 = np.add(BNF11, BNS11)
            BN12 = np.add(BNF12, BNS12)
            BN13 = np.add(BNF13, BNS13)
            BN14 = np.add(BNF14, BNS14)
            
            BN21 = np.add(BNF21, BNS21)
            BN22 = np.add(BNF22, BNS22)
            BN23 = np.add(BNF23, BNS23)
            BN24 = np.add(BNF24, BNS24)
            
            BN31 = np.add(BNF31, BNS31)
            BN32 = np.add(BNF32, BNS32)
            BN33 = np.add(BNF33, BNS33)
            BN34 = np.add(BNF34, BNS34)
            
            BN41 = np.add(BNF41, BNS41)
            BN42 = np.add(BNF42, BNS42)
            BN43 = np.add(BNF43, BNS43)
            BN44 = np.add(BNF44, BNS44)

            # create the first H matrix
            HF11 = self._pentShapeFns[11].HmatrixF(x1, x2, x3, x4, x5)
            HF12 = self._pentShapeFns[12].HmatrixF(x1, x2, x3, x4, x5)
            HF13 = self._pentShapeFns[13].HmatrixF(x1, x2, x3, x4, x5)
            HF14 = self._pentShapeFns[14].HmatrixF(x1, x2, x3, x4, x5)
            
            HF21 = self._pentShapeFns[21].HmatrixF(x1, x2, x3, x4, x5)
            HF22 = self._pentShapeFns[22].HmatrixF(x1, x2, x3, x4, x5)
            HF23 = self._pentShapeFns[23].HmatrixF(x1, x2, x3, x4, x5)
            HF24 = self._pentShapeFns[24].HmatrixF(x1, x2, x3, x4, x5)
            
            HF31 = self._pentShapeFns[31].HmatrixF(x1, x2, x3, x4, x5)
            HF32 = self._pentShapeFns[32].HmatrixF(x1, x2, x3, x4, x5)
            HF33 = self._pentShapeFns[33].HmatrixF(x1, x2, x3, x4, x5)
            HF34 = self._pentShapeFns[34].HmatrixF(x1, x2, x3, x4, x5)
            
            HF41 = self._pentShapeFns[41].HmatrixF(x1, x2, x3, x4, x5)
            HF42 = self._pentShapeFns[42].HmatrixF(x1, x2, x3, x4, x5)
            HF43 = self._pentShapeFns[43].HmatrixF(x1, x2, x3, x4, x5)
            HF44 = self._pentShapeFns[44].HmatrixF(x1, x2, x3, x4, x5)

            # create the second H matrix
            HS11 = self._pentShapeFns[11].HmatrixS(x1, x2, x3, x4, x5)
            HS12 = self._pentShapeFns[12].HmatrixS(x1, x2, x3, x4, x5)
            HS13 = self._pentShapeFns[13].HmatrixS(x1, x2, x3, x4, x5)
            HS14 = self._pentShapeFns[14].HmatrixS(x1, x2, x3, x4, x5)
            
            HS21 = self._pentShapeFns[21].HmatrixS(x1, x2, x3, x4, x5)
            HS22 = self._pentShapeFns[22].HmatrixS(x1, x2, x3, x4, x5)
            HS23 = self._pentShapeFns[23].HmatrixS(x1, x2, x3, x4, x5)
            HS24 = self._pentShapeFns[24].HmatrixS(x1, x2, x3, x4, x5)
            
            HS31 = self._pentShapeFns[31].HmatrixS(x1, x2, x3, x4, x5)
            HS32 = self._pentShapeFns[32].HmatrixS(x1, x2, x3, x4, x5)
            HS33 = self._pentShapeFns[33].HmatrixS(x1, x2, x3, x4, x5)
            HS34 = self._pentShapeFns[34].HmatrixS(x1, x2, x3, x4, x5)
            
            HS41 = self._pentShapeFns[41].HmatrixS(x1, x2, x3, x4, x5)
            HS42 = self._pentShapeFns[42].HmatrixS(x1, x2, x3, x4, x5)
            HS43 = self._pentShapeFns[43].HmatrixS(x1, x2, x3, x4, x5)
            HS44 = self._pentShapeFns[44].HmatrixS(x1, x2, x3, x4, x5)
         
            
            # the nonlinear stiffness matrix for 4 Gauss point
            f11 = (BL11.T.dot(M).dot(BN11) + BN11.T.dot(M).dot(BL11) + 
                   BN11.T.dot(M).dot(BN11))
            f12 = (BL12.T.dot(M).dot(BN12) + BN12.T.dot(M).dot(BL12) + 
                   BN12.T.dot(M).dot(BN12))
            f13 = (BL13.T.dot(M).dot(BN13) + BN13.T.dot(M).dot(BL13) + 
                   BN13.T.dot(M).dot(BN13))
            f14 = (BL14.T.dot(M).dot(BN14) + BN14.T.dot(M).dot(BL14) + 
                   BN14.T.dot(M).dot(BN14))
            
            f21 = (BL21.T.dot(M).dot(BN21) + BN21.T.dot(M).dot(BL21) + 
                   BN21.T.dot(M).dot(BN21))
            f22 = (BL22.T.dot(M).dot(BN22) + BN22.T.dot(M).dot(BL22) + 
                   BN22.T.dot(M).dot(BN22))
            f23 = (BL23.T.dot(M).dot(BN23) + BN23.T.dot(M).dot(BL23) + 
                   BN23.T.dot(M).dot(BN23))
            f24 = (BL24.T.dot(M).dot(BN24) + BN24.T.dot(M).dot(BL24) + 
                   BN24.T.dot(M).dot(BN24))
            
            f31 = (BL31.T.dot(M).dot(BN31) + BN31.T.dot(M).dot(BL31) + 
                   BN31.T.dot(M).dot(BN31))
            f32 = (BL32.T.dot(M).dot(BN32) + BN32.T.dot(M).dot(BL32) + 
                   BN32.T.dot(M).dot(BN32))
            f33 = (BL33.T.dot(M).dot(BN33) + BN33.T.dot(M).dot(BL33) + 
                   BN33.T.dot(M).dot(BN33))
            f34 = (BL34.T.dot(M).dot(BN34) + BN34.T.dot(M).dot(BL34) + 
                   BN34.T.dot(M).dot(BN34))
            
            f41 = (BL41.T.dot(M).dot(BN41) + BN41.T.dot(M).dot(BL41) + 
                   BN41.T.dot(M).dot(BN41))
            f42 = (BL42.T.dot(M).dot(BN42) + BN42.T.dot(M).dot(BL42) + 
                   BN42.T.dot(M).dot(BN42))
            f43 = (BL43.T.dot(M).dot(BN43) + BN43.T.dot(M).dot(BL43) + 
                   BN43.T.dot(M).dot(BN43))
            f44 = (BL44.T.dot(M).dot(BN44) + BN44.T.dot(M).dot(BL44) + 
                   BN44.T.dot(M).dot(BN44))
                                
            
            KN = (self._width * (detJ11 * w[0] * w[0] * f11 +
                                 detJ12 * w[0] * w[1] * f12 +
                                 detJ13 * w[0] * w[2] * f13 +
                                 detJ14 * w[0] * w[3] * f14 +
                                 detJ21 * w[1] * w[0] * f21 +
                                 detJ22 * w[1] * w[1] * f22 +
                                 detJ23 * w[1] * w[2] * f23 +
                                 detJ24 * w[1] * w[3] * f24 +
                                 detJ31 * w[2] * w[0] * f31 +
                                 detJ32 * w[2] * w[1] * f32 +
                                 detJ33 * w[2] * w[2] * f33 +
                                 detJ34 * w[2] * w[3] * f34 +
                                 detJ41 * w[3] * w[0] * f41 +
                                 detJ42 * w[3] * w[1] * f42 +
                                 detJ43 * w[3] * w[2] * f43 +
                                 detJ44 * w[3] * w[3] * f44))

            # create the stress stiffness matrix
            KS = self._width * (detJ11 * w[0] * w[0] * HF11.T.dot(T).dot(HF11)+
                                detJ12 * w[0] * w[1] * HF12.T.dot(T).dot(HF12)+
                                detJ13 * w[0] * w[2] * HF13.T.dot(T).dot(HF13)+
                                detJ14 * w[0] * w[3] * HF14.T.dot(T).dot(HF14)+
                                detJ11 * w[0] * w[0] * HS11.T.dot(T).dot(HS11)+
                                detJ12 * w[0] * w[1] * HS12.T.dot(T).dot(HS12)+
                                detJ13 * w[0] * w[2] * HS13.T.dot(T).dot(HS13)+
                                detJ14 * w[0] * w[3] * HS14.T.dot(T).dot(HS14)+
                                detJ21 * w[1] * w[0] * HF21.T.dot(T).dot(HF21)+
                                detJ22 * w[1] * w[1] * HF22.T.dot(T).dot(HF22)+
                                detJ23 * w[1] * w[2] * HF23.T.dot(T).dot(HF23)+
                                detJ24 * w[1] * w[3] * HF24.T.dot(T).dot(HF24)+
                                detJ21 * w[1] * w[0] * HS21.T.dot(T).dot(HS21)+
                                detJ22 * w[1] * w[1] * HS22.T.dot(T).dot(HS22)+
                                detJ23 * w[1] * w[2] * HS23.T.dot(T).dot(HS23)+
                                detJ24 * w[1] * w[3] * HS24.T.dot(T).dot(HS24)+
                                detJ31 * w[2] * w[0] * HF31.T.dot(T).dot(HF31)+
                                detJ32 * w[2] * w[1] * HF32.T.dot(T).dot(HF32)+
                                detJ33 * w[2] * w[2] * HF33.T.dot(T).dot(HF33)+
                                detJ34 * w[2] * w[3] * HF34.T.dot(T).dot(HF34)+
                                detJ31 * w[2] * w[0] * HS31.T.dot(T).dot(HS31)+
                                detJ32 * w[2] * w[1] * HS32.T.dot(T).dot(HS32)+
                                detJ33 * w[2] * w[2] * HS33.T.dot(T).dot(HS33)+
                                detJ34 * w[2] * w[3] * HS34.T.dot(T).dot(HS34)+
                                detJ41 * w[3] * w[0] * HF41.T.dot(T).dot(HF41)+
                                detJ42 * w[3] * w[1] * HF42.T.dot(T).dot(HF42)+
                                detJ43 * w[3] * w[2] * HF43.T.dot(T).dot(HF43)+
                                detJ44 * w[3] * w[3] * HF44.T.dot(T).dot(HF44)+
                                detJ41 * w[3] * w[0] * HS41.T.dot(T).dot(HS41)+
                                detJ42 * w[3] * w[1] * HS42.T.dot(T).dot(HS42)+
                                detJ43 * w[3] * w[2] * HS43.T.dot(T).dot(HS43)+
                                detJ44 * w[3] * w[3] * HS44.T.dot(T).dot(HS44))            
            
            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS

        else:  # pentGaussPts = 7
            # 'natural' weights of the element
            w = np.array([0.6257871064166934, 0.3016384608809768,
                            0.3169910433902452, 0.3155445150066620,
                            0.2958801959111726, 0.2575426306970870,
                            0.2642573384350463])

            jacob11 = self._pentShapeFns[11].jacobian(x1, x2, x3, x4, x5)
            jacob12 = self._pentShapeFns[12].jacobian(x1, x2, x3, x4, x5)
            jacob13 = self._pentShapeFns[13].jacobian(x1, x2, x3, x4, x5)
            jacob14 = self._pentShapeFns[14].jacobian(x1, x2, x3, x4, x5)
            jacob15 = self._pentShapeFns[15].jacobian(x1, x2, x3, x4, x5)
            jacob16 = self._pentShapeFns[16].jacobian(x1, x2, x3, x4, x5)
            jacob17 = self._pentShapeFns[17].jacobian(x1, x2, x3, x4, x5)
            
            jacob21 = self._pentShapeFns[21].jacobian(x1, x2, x3, x4, x5)
            jacob22 = self._pentShapeFns[22].jacobian(x1, x2, x3, x4, x5)
            jacob23 = self._pentShapeFns[23].jacobian(x1, x2, x3, x4, x5)
            jacob24 = self._pentShapeFns[24].jacobian(x1, x2, x3, x4, x5)
            jacob25 = self._pentShapeFns[25].jacobian(x1, x2, x3, x4, x5)
            jacob26 = self._pentShapeFns[26].jacobian(x1, x2, x3, x4, x5)
            jacob27 = self._pentShapeFns[27].jacobian(x1, x2, x3, x4, x5)
            
            jacob31 = self._pentShapeFns[31].jacobian(x1, x2, x3, x4, x5)
            jacob32 = self._pentShapeFns[32].jacobian(x1, x2, x3, x4, x5)
            jacob33 = self._pentShapeFns[33].jacobian(x1, x2, x3, x4, x5)
            jacob34 = self._pentShapeFns[34].jacobian(x1, x2, x3, x4, x5)
            jacob35 = self._pentShapeFns[35].jacobian(x1, x2, x3, x4, x5)
            jacob36 = self._pentShapeFns[36].jacobian(x1, x2, x3, x4, x5)
            jacob37 = self._pentShapeFns[37].jacobian(x1, x2, x3, x4, x5)
            
            jacob41 = self._pentShapeFns[41].jacobian(x1, x2, x3, x4, x5)
            jacob42 = self._pentShapeFns[42].jacobian(x1, x2, x3, x4, x5)
            jacob43 = self._pentShapeFns[43].jacobian(x1, x2, x3, x4, x5)
            jacob44 = self._pentShapeFns[44].jacobian(x1, x2, x3, x4, x5)
            jacob45 = self._pentShapeFns[45].jacobian(x1, x2, x3, x4, x5)
            jacob46 = self._pentShapeFns[46].jacobian(x1, x2, x3, x4, x5)
            jacob47 = self._pentShapeFns[47].jacobian(x1, x2, x3, x4, x5)
            
            jacob51 = self._pentShapeFns[51].jacobian(x1, x2, x3, x4, x5)
            jacob52 = self._pentShapeFns[52].jacobian(x1, x2, x3, x4, x5)
            jacob53 = self._pentShapeFns[53].jacobian(x1, x2, x3, x4, x5)
            jacob54 = self._pentShapeFns[54].jacobian(x1, x2, x3, x4, x5)
            jacob55 = self._pentShapeFns[55].jacobian(x1, x2, x3, x4, x5)
            jacob56 = self._pentShapeFns[56].jacobian(x1, x2, x3, x4, x5)
            jacob57 = self._pentShapeFns[57].jacobian(x1, x2, x3, x4, x5)
            
            jacob61 = self._pentShapeFns[61].jacobian(x1, x2, x3, x4, x5)
            jacob62 = self._pentShapeFns[62].jacobian(x1, x2, x3, x4, x5)
            jacob63 = self._pentShapeFns[63].jacobian(x1, x2, x3, x4, x5)
            jacob64 = self._pentShapeFns[64].jacobian(x1, x2, x3, x4, x5)
            jacob65 = self._pentShapeFns[65].jacobian(x1, x2, x3, x4, x5)
            jacob66 = self._pentShapeFns[66].jacobian(x1, x2, x3, x4, x5)
            jacob67 = self._pentShapeFns[67].jacobian(x1, x2, x3, x4, x5)
            
            jacob71 = self._pentShapeFns[71].jacobian(x1, x2, x3, x4, x5)
            jacob72 = self._pentShapeFns[72].jacobian(x1, x2, x3, x4, x5)
            jacob73 = self._pentShapeFns[73].jacobian(x1, x2, x3, x4, x5)
            jacob74 = self._pentShapeFns[74].jacobian(x1, x2, x3, x4, x5)
            jacob75 = self._pentShapeFns[75].jacobian(x1, x2, x3, x4, x5)
            jacob76 = self._pentShapeFns[76].jacobian(x1, x2, x3, x4, x5)
            jacob77 = self._pentShapeFns[77].jacobian(x1, x2, x3, x4, x5)

            # determinant of the Jacobian matrix
            detJ11 = det(jacob11)
            detJ12 = det(jacob12)
            detJ13 = det(jacob13)            
            detJ14 = det(jacob14)
            detJ15 = det(jacob15)
            detJ16 = det(jacob16)            
            detJ17 = det(jacob17)
            
            detJ21 = det(jacob21)
            detJ22 = det(jacob22)
            detJ23 = det(jacob23)            
            detJ24 = det(jacob24)
            detJ25 = det(jacob25)
            detJ26 = det(jacob26)            
            detJ27 = det(jacob27)
            
            detJ31 = det(jacob31)
            detJ32 = det(jacob32)
            detJ33 = det(jacob33)            
            detJ34 = det(jacob34)
            detJ35 = det(jacob35)
            detJ36 = det(jacob36)            
            detJ37 = det(jacob37)
            
            detJ41 = det(jacob41)
            detJ42 = det(jacob42)
            detJ43 = det(jacob43)            
            detJ44 = det(jacob44)
            detJ45 = det(jacob45)
            detJ46 = det(jacob46)            
            detJ47 = det(jacob47)
            
            detJ51 = det(jacob51)
            detJ52 = det(jacob52)
            detJ53 = det(jacob53)            
            detJ54 = det(jacob54)
            detJ55 = det(jacob55)
            detJ56 = det(jacob56)            
            detJ57 = det(jacob57)
            
            detJ61 = det(jacob61)
            detJ62 = det(jacob62)
            detJ63 = det(jacob63)            
            detJ64 = det(jacob64)
            detJ65 = det(jacob65)
            detJ66 = det(jacob66)            
            detJ67 = det(jacob67)
            
            detJ71 = det(jacob71)
            detJ72 = det(jacob72)
            detJ73 = det(jacob73)            
            detJ74 = det(jacob74)
            detJ75 = det(jacob75)
            detJ76 = det(jacob76)            
            detJ77 = det(jacob77)
            
            # create the linear Bmatrix
            BL11 = self._pentShapeFns[11].BLinear(x1, x2, x3, x4, x5)
            BL12 = self._pentShapeFns[12].BLinear(x1, x2, x3, x4, x5)        
            BL13 = self._pentShapeFns[13].BLinear(x1, x2, x3, x4, x5) 
            BL14 = self._pentShapeFns[14].BLinear(x1, x2, x3, x4, x5) 
            BL15 = self._pentShapeFns[15].BLinear(x1, x2, x3, x4, x5)      
            BL16 = self._pentShapeFns[16].BLinear(x1, x2, x3, x4, x5) 
            BL17 = self._pentShapeFns[17].BLinear(x1, x2, x3, x4, x5) 
            
            BL21 = self._pentShapeFns[21].BLinear(x1, x2, x3, x4, x5) 
            BL22 = self._pentShapeFns[22].BLinear(x1, x2, x3, x4, x5)        
            BL23 = self._pentShapeFns[23].BLinear(x1, x2, x3, x4, x5) 
            BL24 = self._pentShapeFns[24].BLinear(x1, x2, x3, x4, x5) 
            BL25 = self._pentShapeFns[25].BLinear(x1, x2, x3, x4, x5)     
            BL26 = self._pentShapeFns[26].BLinear(x1, x2, x3, x4, x5) 
            BL27 = self._pentShapeFns[27].BLinear(x1, x2, x3, x4, x5) 
            
            BL31 = self._pentShapeFns[31].BLinear(x1, x2, x3, x4, x5) 
            BL32 = self._pentShapeFns[32].BLinear(x1, x2, x3, x4, x5)        
            BL33 = self._pentShapeFns[33].BLinear(x1, x2, x3, x4, x5)
            BL34 = self._pentShapeFns[34].BLinear(x1, x2, x3, x4, x5)
            BL35 = self._pentShapeFns[35].BLinear(x1, x2, x3, x4, x5)      
            BL36 = self._pentShapeFns[36].BLinear(x1, x2, x3, x4, x5) 
            BL37 = self._pentShapeFns[37].BLinear(x1, x2, x3, x4, x5) 
            
            BL41 = self._pentShapeFns[41].BLinear(x1, x2, x3, x4, x5) 
            BL42 = self._pentShapeFns[42].BLinear(x1, x2, x3, x4, x5)    
            BL43 = self._pentShapeFns[43].BLinear(x1, x2, x3, x4, x5) 
            BL44 = self._pentShapeFns[44].BLinear(x1, x2, x3, x4, x5) 
            BL45 = self._pentShapeFns[45].BLinear(x1, x2, x3, x4, x5)     
            BL46 = self._pentShapeFns[46].BLinear(x1, x2, x3, x4, x5) 
            BL47 = self._pentShapeFns[47].BLinear(x1, x2, x3, x4, x5) 
            
            BL51 = self._pentShapeFns[51].BLinear(x1, x2, x3, x4, x5)
            BL52 = self._pentShapeFns[52].BLinear(x1, x2, x3, x4, x5)        
            BL53 = self._pentShapeFns[53].BLinear(x1, x2, x3, x4, x5) 
            BL54 = self._pentShapeFns[54].BLinear(x1, x2, x3, x4, x5) 
            BL55 = self._pentShapeFns[55].BLinear(x1, x2, x3, x4, x5)     
            BL56 = self._pentShapeFns[56].BLinear(x1, x2, x3, x4, x5) 
            BL57 = self._pentShapeFns[57].BLinear(x1, x2, x3, x4, x5) 
            
            BL61 = self._pentShapeFns[61].BLinear(x1, x2, x3, x4, x5)
            BL62 = self._pentShapeFns[62].BLinear(x1, x2, x3, x4, x5)         
            BL63 = self._pentShapeFns[63].BLinear(x1, x2, x3, x4, x5) 
            BL64 = self._pentShapeFns[64].BLinear(x1, x2, x3, x4, x5) 
            BL65 = self._pentShapeFns[65].BLinear(x1, x2, x3, x4, x5)      
            BL66 = self._pentShapeFns[66].BLinear(x1, x2, x3, x4, x5) 
            BL67 = self._pentShapeFns[67].BLinear(x1, x2, x3, x4, x5) 
            
            BL71 = self._pentShapeFns[71].BLinear(x1, x2, x3, x4, x5) 
            BL72 = self._pentShapeFns[72].BLinear(x1, x2, x3, x4, x5)        
            BL73 = self._pentShapeFns[73].BLinear(x1, x2, x3, x4, x5) 
            BL74 = self._pentShapeFns[74].BLinear(x1, x2, x3, x4, x5)
            BL75 = self._pentShapeFns[75].BLinear(x1, x2, x3, x4, x5)    
            BL76 = self._pentShapeFns[76].BLinear(x1, x2, x3, x4, x5)
            BL77 = self._pentShapeFns[77].BLinear(x1, x2, x3, x4, x5) 

            # the consistent mass matrix for 7 Gauss points
            KL = self._width * (detJ11 * w[0] * w[0] * BL11.T.dot(M).dot(BL11)+
                                detJ12 * w[0] * w[1] * BL12.T.dot(M).dot(BL12)+
                                detJ13 * w[0] * w[2] * BL13.T.dot(M).dot(BL13)+
                                detJ14 * w[0] * w[3] * BL14.T.dot(M).dot(BL14)+
                                detJ15 * w[0] * w[4] * BL15.T.dot(M).dot(BL15)+
                                detJ16 * w[0] * w[5] * BL16.T.dot(M).dot(BL16)+
                                detJ17 * w[0] * w[6] * BL17.T.dot(M).dot(BL17)+
                                detJ21 * w[1] * w[0] * BL21.T.dot(M).dot(BL21)+
                                detJ22 * w[1] * w[1] * BL22.T.dot(M).dot(BL22)+
                                detJ23 * w[1] * w[2] * BL23.T.dot(M).dot(BL23)+
                                detJ24 * w[1] * w[3] * BL24.T.dot(M).dot(BL24)+
                                detJ25 * w[1] * w[4] * BL25.T.dot(M).dot(BL25)+
                                detJ26 * w[1] * w[5] * BL26.T.dot(M).dot(BL26)+
                                detJ27 * w[1] * w[6] * BL27.T.dot(M).dot(BL27)+
                                detJ31 * w[2] * w[0] * BL31.T.dot(M).dot(BL31)+
                                detJ32 * w[2] * w[1] * BL32.T.dot(M).dot(BL32)+
                                detJ33 * w[2] * w[2] * BL33.T.dot(M).dot(BL33)+
                                detJ34 * w[2] * w[3] * BL34.T.dot(M).dot(BL34)+
                                detJ35 * w[2] * w[4] * BL35.T.dot(M).dot(BL35)+
                                detJ36 * w[2] * w[5] * BL36.T.dot(M).dot(BL36)+
                                detJ37 * w[2] * w[6] * BL37.T.dot(M).dot(BL37)+
                                detJ41 * w[3] * w[0] * BL41.T.dot(M).dot(BL41)+
                                detJ42 * w[3] * w[1] * BL42.T.dot(M).dot(BL42)+
                                detJ43 * w[3] * w[2] * BL43.T.dot(M).dot(BL43)+
                                detJ44 * w[3] * w[3] * BL44.T.dot(M).dot(BL44)+
                                detJ45 * w[3] * w[4] * BL45.T.dot(M).dot(BL45)+
                                detJ46 * w[3] * w[5] * BL46.T.dot(M).dot(BL46)+
                                detJ47 * w[3] * w[6] * BL47.T.dot(M).dot(BL47)+
                                detJ51 * w[4] * w[0] * BL51.T.dot(M).dot(BL51)+
                                detJ52 * w[4] * w[1] * BL52.T.dot(M).dot(BL52)+
                                detJ53 * w[4] * w[2] * BL53.T.dot(M).dot(BL53)+
                                detJ54 * w[4] * w[3] * BL54.T.dot(M).dot(BL54)+
                                detJ55 * w[4] * w[4] * BL55.T.dot(M).dot(BL55)+
                                detJ56 * w[4] * w[5] * BL56.T.dot(M).dot(BL56)+
                                detJ57 * w[4] * w[6] * BL57.T.dot(M).dot(BL57)+
                                detJ61 * w[5] * w[0] * BL61.T.dot(M).dot(BL61)+
                                detJ62 * w[5] * w[1] * BL62.T.dot(M).dot(BL62)+
                                detJ63 * w[5] * w[2] * BL63.T.dot(M).dot(BL63)+
                                detJ64 * w[5] * w[3] * BL64.T.dot(M).dot(BL64)+
                                detJ65 * w[5] * w[4] * BL65.T.dot(M).dot(BL65)+
                                detJ66 * w[5] * w[5] * BL66.T.dot(M).dot(BL66)+
                                detJ67 * w[5] * w[6] * BL67.T.dot(M).dot(BL67)+
                                detJ71 * w[6] * w[0] * BL71.T.dot(M).dot(BL71)+
                                detJ72 * w[6] * w[1] * BL72.T.dot(M).dot(BL72)+
                                detJ73 * w[6] * w[2] * BL73.T.dot(M).dot(BL73)+
                                detJ74 * w[6] * w[3] * BL74.T.dot(M).dot(BL74)+
                                detJ75 * w[6] * w[4] * BL75.T.dot(M).dot(BL75)+
                                detJ76 * w[6] * w[5] * BL76.T.dot(M).dot(BL76)+
                                detJ77 * w[6] * w[6] * BL77.T.dot(M).dot(BL77))

            # create the first nonlinear Bmatrix
            BNF11 = self._pentShapeFns[11].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF12 = self._pentShapeFns[12].FirstBNonLinear(x1, x2, x3, x4, x5,
                                  x01, x02, x03, x04, x05)
            BNF13 = self._pentShapeFns[13].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF14 = self._pentShapeFns[14].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF15 = self._pentShapeFns[15].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF16 = self._pentShapeFns[16].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF17 = self._pentShapeFns[17].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNF21 = self._pentShapeFns[21].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF22 = self._pentShapeFns[22].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF23 = self._pentShapeFns[23].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF24 = self._pentShapeFns[24].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF25 = self._pentShapeFns[25].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF26 = self._pentShapeFns[26].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF27 = self._pentShapeFns[27].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNF31 = self._pentShapeFns[31].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF32 = self._pentShapeFns[32].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF33 = self._pentShapeFns[33].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF34 = self._pentShapeFns[34].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF35 = self._pentShapeFns[35].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF36 = self._pentShapeFns[36].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF37 = self._pentShapeFns[37].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNF41 = self._pentShapeFns[41].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF42 = self._pentShapeFns[42].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF43 = self._pentShapeFns[43].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF44 = self._pentShapeFns[44].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF45 = self._pentShapeFns[45].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF46 = self._pentShapeFns[46].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF47 = self._pentShapeFns[47].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNF51 = self._pentShapeFns[51].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF52 = self._pentShapeFns[52].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF53 = self._pentShapeFns[53].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF54 = self._pentShapeFns[54].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF55 = self._pentShapeFns[55].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF56 = self._pentShapeFns[56].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF57 = self._pentShapeFns[57].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNF61 = self._pentShapeFns[61].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF62 = self._pentShapeFns[62].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF63 = self._pentShapeFns[63].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF64 = self._pentShapeFns[64].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF65 = self._pentShapeFns[65].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF66 = self._pentShapeFns[66].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF67 = self._pentShapeFns[67].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNF71 = self._pentShapeFns[71].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF72 = self._pentShapeFns[72].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF73 = self._pentShapeFns[73].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF74 = self._pentShapeFns[74].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF75 = self._pentShapeFns[75].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF76 = self._pentShapeFns[76].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNF77 = self._pentShapeFns[77].FirstBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)

            # create the first nonlinear Bmatrix
            BNS11 = self._pentShapeFns[11].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS12 = self._pentShapeFns[12].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS13 = self._pentShapeFns[13].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS14 = self._pentShapeFns[14].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS15 = self._pentShapeFns[15].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS16 = self._pentShapeFns[16].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS17 = self._pentShapeFns[17].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNS21 = self._pentShapeFns[21].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS22 = self._pentShapeFns[22].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS23 = self._pentShapeFns[23].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS24 = self._pentShapeFns[24].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS25 = self._pentShapeFns[25].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS26 = self._pentShapeFns[26].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS27 = self._pentShapeFns[27].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNS31 = self._pentShapeFns[31].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS32 = self._pentShapeFns[32].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS33 = self._pentShapeFns[33].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS34 = self._pentShapeFns[34].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS35 = self._pentShapeFns[35].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS36 = self._pentShapeFns[36].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS37 = self._pentShapeFns[37].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNS41 = self._pentShapeFns[41].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS42 = self._pentShapeFns[42].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS43 = self._pentShapeFns[43].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS44 = self._pentShapeFns[44].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS45 = self._pentShapeFns[45].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS46 = self._pentShapeFns[46].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS47 = self._pentShapeFns[47].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNS51 = self._pentShapeFns[51].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS52 = self._pentShapeFns[52].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS53 = self._pentShapeFns[53].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS54 = self._pentShapeFns[54].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS55 = self._pentShapeFns[55].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS56 = self._pentShapeFns[56].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS57 = self._pentShapeFns[57].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNS61 = self._pentShapeFns[61].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS62 = self._pentShapeFns[62].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS63 = self._pentShapeFns[63].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS64 = self._pentShapeFns[64].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS65 = self._pentShapeFns[65].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS66 = self._pentShapeFns[66].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS67 = self._pentShapeFns[67].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            
            BNS71 = self._pentShapeFns[71].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS72 = self._pentShapeFns[72].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS73 = self._pentShapeFns[73].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS74 = self._pentShapeFns[74].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS75 = self._pentShapeFns[75].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS76 = self._pentShapeFns[76].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)
            BNS77 = self._pentShapeFns[77].SecondBNonLinear(x1, x2, x3, x4, x5, 
                                  x01, x02, x03, x04, x05)

            # total nonlinear Bmatrix
            BN11 = np.add(BNF11, BNS11)
            BN12 = np.add(BNF12, BNS12)
            BN13 = np.add(BNF13, BNS13)
            BN14 = np.add(BNF14, BNS14)
            BN15 = np.add(BNF15, BNS15)
            BN16 = np.add(BNF16, BNS16)
            BN17 = np.add(BNF17, BNS17)
            
            BN21 = np.add(BNF21, BNS21)
            BN22 = np.add(BNF22, BNS22)
            BN23 = np.add(BNF23, BNS23)
            BN24 = np.add(BNF24, BNS24)
            BN25 = np.add(BNF25, BNS25)
            BN26 = np.add(BNF26, BNS26)
            BN27 = np.add(BNF27, BNS27)
            
            BN31 = np.add(BNF31, BNS31)
            BN32 = np.add(BNF32, BNS32)
            BN33 = np.add(BNF33, BNS33)
            BN34 = np.add(BNF34, BNS34)
            BN35 = np.add(BNF35, BNS35)
            BN36 = np.add(BNF36, BNS36)
            BN37 = np.add(BNF37, BNS37)
            
            BN41 = np.add(BNF41, BNS41)
            BN42 = np.add(BNF42, BNS42)
            BN43 = np.add(BNF43, BNS43)
            BN44 = np.add(BNF44, BNS44)
            BN45 = np.add(BNF45, BNS45)
            BN46 = np.add(BNF46, BNS46)
            BN47 = np.add(BNF47, BNS47)
            
            BN51 = np.add(BNF51, BNS51)
            BN52 = np.add(BNF52, BNS52)
            BN53 = np.add(BNF53, BNS53)
            BN54 = np.add(BNF54, BNS54)
            BN55 = np.add(BNF55, BNS55)
            BN56 = np.add(BNF56, BNS56)
            BN57 = np.add(BNF57, BNS57)
            
            BN61 = np.add(BNF61, BNS61)
            BN62 = np.add(BNF62, BNS62)
            BN63 = np.add(BNF63, BNS63)
            BN64 = np.add(BNF64, BNS64)
            BN65 = np.add(BNF65, BNS65)
            BN66 = np.add(BNF66, BNS66)
            BN67 = np.add(BNF67, BNS67)
            
            BN71 = np.add(BNF71, BNS71)
            BN72 = np.add(BNF72, BNS72)
            BN73 = np.add(BNF73, BNS73)
            BN74 = np.add(BNF74, BNS74)
            BN75 = np.add(BNF75, BNS75)
            BN76 = np.add(BNF76, BNS76)
            BN77 = np.add(BNF77, BNS77)

            # create the first H matrix
            HF11 = self._pentShapeFns[11].HmatrixF(x1, x2, x3, x4, x5)
            HF12 = self._pentShapeFns[12].HmatrixF(x1, x2, x3, x4, x5)
            HF13 = self._pentShapeFns[13].HmatrixF(x1, x2, x3, x4, x5)
            HF14 = self._pentShapeFns[14].HmatrixF(x1, x2, x3, x4, x5)
            HF15 = self._pentShapeFns[15].HmatrixF(x1, x2, x3, x4, x5)
            HF16 = self._pentShapeFns[16].HmatrixF(x1, x2, x3, x4, x5)
            HF17 = self._pentShapeFns[17].HmatrixF(x1, x2, x3, x4, x5)
            
            HF21 = self._pentShapeFns[21].HmatrixF(x1, x2, x3, x4, x5)
            HF22 = self._pentShapeFns[22].HmatrixF(x1, x2, x3, x4, x5)
            HF23 = self._pentShapeFns[23].HmatrixF(x1, x2, x3, x4, x5)
            HF24 = self._pentShapeFns[24].HmatrixF(x1, x2, x3, x4, x5)
            HF25 = self._pentShapeFns[25].HmatrixF(x1, x2, x3, x4, x5)
            HF26 = self._pentShapeFns[26].HmatrixF(x1, x2, x3, x4, x5)
            HF27 = self._pentShapeFns[27].HmatrixF(x1, x2, x3, x4, x5)
            
            HF31 = self._pentShapeFns[31].HmatrixF(x1, x2, x3, x4, x5)
            HF32 = self._pentShapeFns[32].HmatrixF(x1, x2, x3, x4, x5)
            HF33 = self._pentShapeFns[33].HmatrixF(x1, x2, x3, x4, x5)
            HF34 = self._pentShapeFns[34].HmatrixF(x1, x2, x3, x4, x5)
            HF35 = self._pentShapeFns[35].HmatrixF(x1, x2, x3, x4, x5)
            HF36 = self._pentShapeFns[36].HmatrixF(x1, x2, x3, x4, x5)
            HF37 = self._pentShapeFns[37].HmatrixF(x1, x2, x3, x4, x5)
            
            HF41 = self._pentShapeFns[41].HmatrixF(x1, x2, x3, x4, x5)
            HF42 = self._pentShapeFns[42].HmatrixF(x1, x2, x3, x4, x5)
            HF43 = self._pentShapeFns[43].HmatrixF(x1, x2, x3, x4, x5)
            HF44 = self._pentShapeFns[44].HmatrixF(x1, x2, x3, x4, x5)
            HF45 = self._pentShapeFns[45].HmatrixF(x1, x2, x3, x4, x5)
            HF46 = self._pentShapeFns[46].HmatrixF(x1, x2, x3, x4, x5)
            HF47 = self._pentShapeFns[47].HmatrixF(x1, x2, x3, x4, x5)
            
            HF51 = self._pentShapeFns[51].HmatrixF(x1, x2, x3, x4, x5)
            HF52 = self._pentShapeFns[52].HmatrixF(x1, x2, x3, x4, x5)
            HF53 = self._pentShapeFns[53].HmatrixF(x1, x2, x3, x4, x5)
            HF54 = self._pentShapeFns[54].HmatrixF(x1, x2, x3, x4, x5)
            HF55 = self._pentShapeFns[55].HmatrixF(x1, x2, x3, x4, x5)
            HF56 = self._pentShapeFns[56].HmatrixF(x1, x2, x3, x4, x5)
            HF57 = self._pentShapeFns[57].HmatrixF(x1, x2, x3, x4, x5)
            
            HF61 = self._pentShapeFns[61].HmatrixF(x1, x2, x3, x4, x5)
            HF62 = self._pentShapeFns[62].HmatrixF(x1, x2, x3, x4, x5)
            HF63 = self._pentShapeFns[63].HmatrixF(x1, x2, x3, x4, x5)
            HF64 = self._pentShapeFns[64].HmatrixF(x1, x2, x3, x4, x5)
            HF65 = self._pentShapeFns[65].HmatrixF(x1, x2, x3, x4, x5)
            HF66 = self._pentShapeFns[66].HmatrixF(x1, x2, x3, x4, x5)
            HF67 = self._pentShapeFns[67].HmatrixF(x1, x2, x3, x4, x5)
            
            HF71 = self._pentShapeFns[71].HmatrixF(x1, x2, x3, x4, x5)
            HF72 = self._pentShapeFns[72].HmatrixF(x1, x2, x3, x4, x5)
            HF73 = self._pentShapeFns[73].HmatrixF(x1, x2, x3, x4, x5)
            HF74 = self._pentShapeFns[74].HmatrixF(x1, x2, x3, x4, x5)
            HF75 = self._pentShapeFns[75].HmatrixF(x1, x2, x3, x4, x5)
            HF76 = self._pentShapeFns[76].HmatrixF(x1, x2, x3, x4, x5)
            HF77 = self._pentShapeFns[77].HmatrixF(x1, x2, x3, x4, x5)

            # create the second H matrix
            HS11 = self._pentShapeFns[11].HmatrixS(x1, x2, x3, x4, x5)
            HS12 = self._pentShapeFns[12].HmatrixS(x1, x2, x3, x4, x5)
            HS13 = self._pentShapeFns[13].HmatrixS(x1, x2, x3, x4, x5)
            HS14 = self._pentShapeFns[14].HmatrixS(x1, x2, x3, x4, x5)
            HS15 = self._pentShapeFns[15].HmatrixS(x1, x2, x3, x4, x5)
            HS16 = self._pentShapeFns[16].HmatrixS(x1, x2, x3, x4, x5)
            HS17 = self._pentShapeFns[17].HmatrixS(x1, x2, x3, x4, x5)   
            
            HS21 = self._pentShapeFns[21].HmatrixS(x1, x2, x3, x4, x5)
            HS22 = self._pentShapeFns[22].HmatrixS(x1, x2, x3, x4, x5)
            HS23 = self._pentShapeFns[23].HmatrixS(x1, x2, x3, x4, x5)
            HS24 = self._pentShapeFns[24].HmatrixS(x1, x2, x3, x4, x5)
            HS25 = self._pentShapeFns[25].HmatrixS(x1, x2, x3, x4, x5)
            HS26 = self._pentShapeFns[26].HmatrixS(x1, x2, x3, x4, x5)
            HS27 = self._pentShapeFns[27].HmatrixS(x1, x2, x3, x4, x5)
            
            HS31 = self._pentShapeFns[31].HmatrixS(x1, x2, x3, x4, x5)
            HS32 = self._pentShapeFns[32].HmatrixS(x1, x2, x3, x4, x5)
            HS33 = self._pentShapeFns[33].HmatrixS(x1, x2, x3, x4, x5)
            HS34 = self._pentShapeFns[34].HmatrixS(x1, x2, x3, x4, x5)
            HS35 = self._pentShapeFns[35].HmatrixS(x1, x2, x3, x4, x5)
            HS36 = self._pentShapeFns[36].HmatrixS(x1, x2, x3, x4, x5)
            HS37 = self._pentShapeFns[37].HmatrixS(x1, x2, x3, x4, x5) 
            
            HS41 = self._pentShapeFns[41].HmatrixS(x1, x2, x3, x4, x5)
            HS42 = self._pentShapeFns[42].HmatrixS(x1, x2, x3, x4, x5)
            HS43 = self._pentShapeFns[43].HmatrixS(x1, x2, x3, x4, x5)
            HS44 = self._pentShapeFns[44].HmatrixS(x1, x2, x3, x4, x5)
            HS45 = self._pentShapeFns[45].HmatrixS(x1, x2, x3, x4, x5)
            HS46 = self._pentShapeFns[46].HmatrixS(x1, x2, x3, x4, x5)
            HS47 = self._pentShapeFns[47].HmatrixS(x1, x2, x3, x4, x5) 
            
            HS51 = self._pentShapeFns[51].HmatrixS(x1, x2, x3, x4, x5)
            HS52 = self._pentShapeFns[52].HmatrixS(x1, x2, x3, x4, x5)
            HS53 = self._pentShapeFns[53].HmatrixS(x1, x2, x3, x4, x5)
            HS54 = self._pentShapeFns[54].HmatrixS(x1, x2, x3, x4, x5)
            HS55 = self._pentShapeFns[55].HmatrixS(x1, x2, x3, x4, x5)
            HS56 = self._pentShapeFns[56].HmatrixS(x1, x2, x3, x4, x5)
            HS57 = self._pentShapeFns[57].HmatrixS(x1, x2, x3, x4, x5) 
            
            HS61 = self._pentShapeFns[61].HmatrixS(x1, x2, x3, x4, x5)
            HS62 = self._pentShapeFns[62].HmatrixS(x1, x2, x3, x4, x5)
            HS63 = self._pentShapeFns[63].HmatrixS(x1, x2, x3, x4, x5)
            HS64 = self._pentShapeFns[64].HmatrixS(x1, x2, x3, x4, x5)
            HS65 = self._pentShapeFns[65].HmatrixS(x1, x2, x3, x4, x5)
            HS66 = self._pentShapeFns[66].HmatrixS(x1, x2, x3, x4, x5)
            HS67 = self._pentShapeFns[67].HmatrixS(x1, x2, x3, x4, x5) 
            
            HS71 = self._pentShapeFns[71].HmatrixS(x1, x2, x3, x4, x5)
            HS72 = self._pentShapeFns[72].HmatrixS(x1, x2, x3, x4, x5)
            HS73 = self._pentShapeFns[73].HmatrixS(x1, x2, x3, x4, x5)
            HS74 = self._pentShapeFns[74].HmatrixS(x1, x2, x3, x4, x5)
            HS75 = self._pentShapeFns[75].HmatrixS(x1, x2, x3, x4, x5)
            HS76 = self._pentShapeFns[76].HmatrixS(x1, x2, x3, x4, x5)
            HS77 = self._pentShapeFns[77].HmatrixS(x1, x2, x3, x4, x5) 
            
            
            # the nonlinear stiffness matrix for 7 Gauss point
            f11 = (BL11.T.dot(M).dot(BN11) + BN11.T.dot(M).dot(BL11) + 
                   BN11.T.dot(M).dot(BN11))
            f12 = (BL12.T.dot(M).dot(BN12) + BN12.T.dot(M).dot(BL12) + 
                   BN12.T.dot(M).dot(BN12))
            f13 = (BL13.T.dot(M).dot(BN13) + BN13.T.dot(M).dot(BL13) + 
                   BN13.T.dot(M).dot(BN13))
            f14 = (BL14.T.dot(M).dot(BN14) + BN14.T.dot(M).dot(BL14) +
                   BN14.T.dot(M).dot(BN14))
            f15 = (BL15.T.dot(M).dot(BN15) + BN15.T.dot(M).dot(BL15) + 
                   BN15.T.dot(M).dot(BN15))
            f16 = (BL16.T.dot(M).dot(BN16) + BN16.T.dot(M).dot(BL16) + 
                   BN16.T.dot(M).dot(BN16))
            f17 = (BL17.T.dot(M).dot(BN17) + BN17.T.dot(M).dot(BL17) + 
                   BN17.T.dot(M).dot(BN17))
            
            f21 = (BL21.T.dot(M).dot(BN21) + BN21.T.dot(M).dot(BL21) + 
                   BN21.T.dot(M).dot(BN21))
            f22 = (BL22.T.dot(M).dot(BN22) + BN22.T.dot(M).dot(BL22) + 
                   BN22.T.dot(M).dot(BN22))
            f23 = (BL23.T.dot(M).dot(BN23) + BN23.T.dot(M).dot(BL23) + 
                   BN23.T.dot(M).dot(BN23))
            f24 = (BL24.T.dot(M).dot(BN24) + BN24.T.dot(M).dot(BL24) +
                   BN24.T.dot(M).dot(BN24))
            f25 = (BL25.T.dot(M).dot(BN25) + BN25.T.dot(M).dot(BL25) + 
                   BN25.T.dot(M).dot(BN25))
            f26 = (BL26.T.dot(M).dot(BN26) + BN26.T.dot(M).dot(BL26) + 
                   BN26.T.dot(M).dot(BN26))
            f27 = (BL27.T.dot(M).dot(BN27) + BN27.T.dot(M).dot(BL27) + 
                   BN27.T.dot(M).dot(BN27))
            
            f31 = (BL31.T.dot(M).dot(BN31) + BN31.T.dot(M).dot(BL31) + 
                   BN31.T.dot(M).dot(BN31))
            f32 = (BL32.T.dot(M).dot(BN32) + BN32.T.dot(M).dot(BL32) + 
                   BN32.T.dot(M).dot(BN32))
            f33 = (BL33.T.dot(M).dot(BN33) + BN33.T.dot(M).dot(BL33) + 
                   BN33.T.dot(M).dot(BN33))
            f34 = (BL34.T.dot(M).dot(BN34) + BN34.T.dot(M).dot(BL34) +
                   BN34.T.dot(M).dot(BN34))
            f35 = (BL35.T.dot(M).dot(BN35) + BN35.T.dot(M).dot(BL35) + 
                   BN35.T.dot(M).dot(BN35))
            f36 = (BL36.T.dot(M).dot(BN36) + BN36.T.dot(M).dot(BL36) + 
                   BN36.T.dot(M).dot(BN36))
            f37 = (BL37.T.dot(M).dot(BN37) + BN37.T.dot(M).dot(BL37) + 
                   BN37.T.dot(M).dot(BN37))
            
            f41 = (BL41.T.dot(M).dot(BN41) + BN41.T.dot(M).dot(BL41) + 
                   BN41.T.dot(M).dot(BN41))
            f42 = (BL42.T.dot(M).dot(BN42) + BN42.T.dot(M).dot(BL42) + 
                   BN42.T.dot(M).dot(BN42))
            f43 = (BL43.T.dot(M).dot(BN43) + BN43.T.dot(M).dot(BL43) + 
                   BN43.T.dot(M).dot(BN43))
            f44 = (BL44.T.dot(M).dot(BN44) + BN44.T.dot(M).dot(BL44) +
                   BN44.T.dot(M).dot(BN44))
            f45 = (BL45.T.dot(M).dot(BN45) + BN45.T.dot(M).dot(BL45) + 
                   BN45.T.dot(M).dot(BN45))
            f46 = (BL46.T.dot(M).dot(BN46) + BN46.T.dot(M).dot(BL46) + 
                   BN46.T.dot(M).dot(BN46))
            f47 = (BL47.T.dot(M).dot(BN47) + BN47.T.dot(M).dot(BL47) + 
                   BN47.T.dot(M).dot(BN47))
            
            f51 = (BL51.T.dot(M).dot(BN51) + BN51.T.dot(M).dot(BL51) + 
                   BN51.T.dot(M).dot(BN51))
            f52 = (BL52.T.dot(M).dot(BN52) + BN52.T.dot(M).dot(BL52) + 
                   BN52.T.dot(M).dot(BN52))
            f53 = (BL53.T.dot(M).dot(BN53) + BN53.T.dot(M).dot(BL53) + 
                   BN53.T.dot(M).dot(BN53))
            f54 = (BL54.T.dot(M).dot(BN54) + BN54.T.dot(M).dot(BL54) +
                   BN54.T.dot(M).dot(BN54))
            f55 = (BL55.T.dot(M).dot(BN55) + BN55.T.dot(M).dot(BL55) + 
                   BN55.T.dot(M).dot(BN55))
            f56 = (BL56.T.dot(M).dot(BN56) + BN56.T.dot(M).dot(BL56) + 
                   BN56.T.dot(M).dot(BN56))
            f57 = (BL57.T.dot(M).dot(BN57) + BN57.T.dot(M).dot(BL57) + 
                   BN57.T.dot(M).dot(BN57))
            
            f61 = (BL61.T.dot(M).dot(BN61) + BN61.T.dot(M).dot(BL61) + 
                   BN61.T.dot(M).dot(BN61))
            f62 = (BL62.T.dot(M).dot(BN62) + BN62.T.dot(M).dot(BL62) + 
                   BN62.T.dot(M).dot(BN62))
            f63 = (BL63.T.dot(M).dot(BN63) + BN63.T.dot(M).dot(BL63) + 
                   BN63.T.dot(M).dot(BN63))
            f64 = (BL64.T.dot(M).dot(BN64) + BN64.T.dot(M).dot(BL64) +
                   BN64.T.dot(M).dot(BN64))
            f65 = (BL65.T.dot(M).dot(BN65) + BN65.T.dot(M).dot(BL65) + 
                   BN65.T.dot(M).dot(BN65))
            f66 = (BL66.T.dot(M).dot(BN66) + BN66.T.dot(M).dot(BL66) + 
                   BN66.T.dot(M).dot(BN66))
            f67 = (BL67.T.dot(M).dot(BN67) + BN67.T.dot(M).dot(BL67) + 
                   BN67.T.dot(M).dot(BN67))
            
            f71 = (BL71.T.dot(M).dot(BN71) + BN71.T.dot(M).dot(BL71) + 
                   BN71.T.dot(M).dot(BN71))
            f72 = (BL72.T.dot(M).dot(BN72) + BN72.T.dot(M).dot(BL72) + 
                   BN72.T.dot(M).dot(BN72))
            f73 = (BL73.T.dot(M).dot(BN73) + BN73.T.dot(M).dot(BL73) + 
                   BN73.T.dot(M).dot(BN73))
            f74 = (BL74.T.dot(M).dot(BN74) + BN74.T.dot(M).dot(BL74) +
                   BN74.T.dot(M).dot(BN74))
            f75 = (BL75.T.dot(M).dot(BN75) + BN75.T.dot(M).dot(BL75) + 
                   BN75.T.dot(M).dot(BN75))
            f76 = (BL76.T.dot(M).dot(BN76) + BN76.T.dot(M).dot(BL76) + 
                   BN76.T.dot(M).dot(BN76))
            f77 = (BL77.T.dot(M).dot(BN77) + BN77.T.dot(M).dot(BL77) + 
                   BN77.T.dot(M).dot(BN77))
            
            KN = self._width * (detJ11 * w[0] * w[0] * f11 +
                                detJ12 * w[0] * w[1] * f12 +
                                detJ13 * w[0] * w[2] * f13 +
                                detJ14 * w[0] * w[3] * f14 +
                                detJ15 * w[0] * w[4] * f15 +
                                detJ16 * w[0] * w[5] * f16 +
                                detJ17 * w[0] * w[6] * f17 +
                                detJ21 * w[1] * w[0] * f21 +
                                detJ22 * w[1] * w[1] * f22 +
                                detJ23 * w[1] * w[2] * f23 +
                                detJ24 * w[1] * w[3] * f24 +
                                detJ25 * w[1] * w[4] * f25 +
                                detJ26 * w[1] * w[5] * f26 +
                                detJ27 * w[1] * w[6] * f27 +
                                detJ31 * w[2] * w[0] * f31 +
                                detJ32 * w[2] * w[1] * f32 +
                                detJ33 * w[2] * w[2] * f33 +
                                detJ34 * w[2] * w[3] * f34 +
                                detJ35 * w[2] * w[4] * f35 +
                                detJ36 * w[2] * w[5] * f36 +
                                detJ37 * w[2] * w[6] * f37 +
                                detJ41 * w[3] * w[0] * f41 +
                                detJ42 * w[3] * w[1] * f42 +
                                detJ43 * w[3] * w[2] * f43 +
                                detJ44 * w[3] * w[3] * f44 +
                                detJ45 * w[3] * w[4] * f45 +
                                detJ46 * w[3] * w[5] * f46 +
                                detJ47 * w[3] * w[6] * f47 +
                                detJ51 * w[4] * w[0] * f51 +
                                detJ52 * w[4] * w[1] * f52 +
                                detJ53 * w[4] * w[2] * f53 +
                                detJ54 * w[4] * w[3] * f54 +
                                detJ55 * w[4] * w[4] * f55 +
                                detJ56 * w[4] * w[5] * f56 +
                                detJ57 * w[4] * w[6] * f57 +
                                detJ61 * w[5] * w[0] * f61 +
                                detJ62 * w[5] * w[1] * f62 +
                                detJ63 * w[5] * w[2] * f63 +
                                detJ64 * w[5] * w[3] * f64 +
                                detJ65 * w[5] * w[4] * f65 +
                                detJ66 * w[5] * w[5] * f66 +
                                detJ67 * w[5] * w[6] * f67 +
                                detJ71 * w[6] * w[0] * f71 +
                                detJ72 * w[6] * w[1] * f72 +
                                detJ73 * w[6] * w[2] * f73 +
                                detJ74 * w[6] * w[3] * f74 +
                                detJ75 * w[6] * w[4] * f75 +
                                detJ76 * w[6] * w[5] * f76 +
                                detJ77 * w[6] * w[6] * f77)          

            
            # create the stress stiffness matrix            
            KS = self._width * (detJ11 * w[0] * w[0] * HF11.T.dot(T).dot(HF11)+
                                detJ12 * w[0] * w[1] * HF12.T.dot(T).dot(HF12)+
                                detJ13 * w[0] * w[2] * HF13.T.dot(T).dot(HF13)+
                                detJ14 * w[0] * w[3] * HF14.T.dot(T).dot(HF14)+
                                detJ15 * w[0] * w[4] * HF15.T.dot(T).dot(HF15)+
                                detJ16 * w[0] * w[5] * HF16.T.dot(T).dot(HF16)+
                                detJ17 * w[0] * w[6] * HF17.T.dot(T).dot(HF17)+
                                detJ11 * w[0] * w[0] * HS11.T.dot(T).dot(HS11)+
                                detJ12 * w[0] * w[1] * HS12.T.dot(T).dot(HS12)+
                                detJ13 * w[0] * w[2] * HS13.T.dot(T).dot(HS13)+
                                detJ14 * w[0] * w[3] * HS14.T.dot(T).dot(HS14)+
                                detJ15 * w[0] * w[4] * HS15.T.dot(T).dot(HS15)+
                                detJ16 * w[0] * w[5] * HS16.T.dot(T).dot(HS16)+
                                detJ17 * w[0] * w[6] * HS17.T.dot(T).dot(HS17)+
                                detJ21 * w[1] * w[0] * HF21.T.dot(T).dot(HF21)+
                                detJ22 * w[1] * w[1] * HF22.T.dot(T).dot(HF22)+
                                detJ23 * w[1] * w[2] * HF23.T.dot(T).dot(HF23)+
                                detJ24 * w[1] * w[3] * HF24.T.dot(T).dot(HF24)+
                                detJ25 * w[1] * w[4] * HF25.T.dot(T).dot(HF25)+
                                detJ26 * w[1] * w[5] * HF26.T.dot(T).dot(HF26)+
                                detJ27 * w[1] * w[6] * HF27.T.dot(T).dot(HF27)+
                                detJ21 * w[1] * w[0] * HS21.T.dot(T).dot(HS21)+
                                detJ22 * w[1] * w[1] * HS22.T.dot(T).dot(HS22)+
                                detJ23 * w[1] * w[2] * HS23.T.dot(T).dot(HS23)+
                                detJ24 * w[1] * w[3] * HS24.T.dot(T).dot(HS24)+
                                detJ25 * w[1] * w[4] * HS25.T.dot(T).dot(HS25)+
                                detJ26 * w[1] * w[5] * HS26.T.dot(T).dot(HS26)+
                                detJ27 * w[1] * w[6] * HS27.T.dot(T).dot(HS27)+
                                detJ31 * w[2] * w[0] * HF31.T.dot(T).dot(HF31)+
                                detJ32 * w[2] * w[1] * HF32.T.dot(T).dot(HF32)+
                                detJ33 * w[2] * w[2] * HF33.T.dot(T).dot(HF33)+
                                detJ34 * w[2] * w[3] * HF34.T.dot(T).dot(HF34)+
                                detJ35 * w[2] * w[4] * HF35.T.dot(T).dot(HF35)+
                                detJ36 * w[2] * w[5] * HF36.T.dot(T).dot(HF36)+
                                detJ37 * w[2] * w[6] * HF37.T.dot(T).dot(HF37)+
                                detJ31 * w[2] * w[0] * HS31.T.dot(T).dot(HS31)+
                                detJ32 * w[2] * w[1] * HS32.T.dot(T).dot(HS32)+
                                detJ33 * w[2] * w[2] * HS33.T.dot(T).dot(HS33)+
                                detJ34 * w[2] * w[3] * HS34.T.dot(T).dot(HS34)+
                                detJ35 * w[2] * w[4] * HS35.T.dot(T).dot(HS35)+
                                detJ36 * w[2] * w[5] * HS36.T.dot(T).dot(HS36)+
                                detJ37 * w[2] * w[6] * HS37.T.dot(T).dot(HS37)+
                                detJ41 * w[3] * w[0] * HF41.T.dot(T).dot(HF31)+
                                detJ42 * w[3] * w[1] * HF42.T.dot(T).dot(HF42)+
                                detJ43 * w[3] * w[2] * HF43.T.dot(T).dot(HF43)+
                                detJ44 * w[3] * w[3] * HF44.T.dot(T).dot(HF44)+
                                detJ45 * w[3] * w[4] * HF45.T.dot(T).dot(HF45)+
                                detJ46 * w[3] * w[5] * HF46.T.dot(T).dot(HF46)+
                                detJ47 * w[3] * w[6] * HF47.T.dot(T).dot(HF47)+
                                detJ41 * w[3] * w[0] * HS41.T.dot(T).dot(HS41)+
                                detJ42 * w[3] * w[1] * HS42.T.dot(T).dot(HS42)+
                                detJ43 * w[3] * w[2] * HS43.T.dot(T).dot(HS43)+
                                detJ44 * w[3] * w[3] * HS44.T.dot(T).dot(HS44)+
                                detJ45 * w[3] * w[4] * HS45.T.dot(T).dot(HS45)+
                                detJ46 * w[3] * w[5] * HS46.T.dot(T).dot(HS46)+
                                detJ47 * w[3] * w[6] * HS47.T.dot(T).dot(HS47)+
                                detJ51 * w[4] * w[0] * HF51.T.dot(T).dot(HF51)+
                                detJ52 * w[4] * w[1] * HF52.T.dot(T).dot(HF52)+
                                detJ53 * w[4] * w[2] * HF53.T.dot(T).dot(HF53)+
                                detJ54 * w[4] * w[3] * HF54.T.dot(T).dot(HF54)+
                                detJ55 * w[4] * w[4] * HF55.T.dot(T).dot(HF55)+
                                detJ56 * w[4] * w[5] * HF56.T.dot(T).dot(HF56)+
                                detJ57 * w[4] * w[6] * HF57.T.dot(T).dot(HF57)+
                                detJ51 * w[4] * w[0] * HS51.T.dot(T).dot(HS51)+
                                detJ52 * w[4] * w[1] * HS52.T.dot(T).dot(HS52)+
                                detJ53 * w[4] * w[2] * HS53.T.dot(T).dot(HS53)+
                                detJ54 * w[4] * w[3] * HS54.T.dot(T).dot(HS54)+
                                detJ55 * w[4] * w[4] * HS55.T.dot(T).dot(HS55)+
                                detJ56 * w[4] * w[5] * HS56.T.dot(T).dot(HS56)+
                                detJ57 * w[4] * w[6] * HS57.T.dot(T).dot(HS57)+
                                detJ61 * w[5] * w[0] * HF61.T.dot(T).dot(HF61)+
                                detJ62 * w[5] * w[1] * HF62.T.dot(T).dot(HF62)+
                                detJ63 * w[5] * w[2] * HF63.T.dot(T).dot(HF63)+
                                detJ64 * w[5] * w[3] * HF64.T.dot(T).dot(HF64)+
                                detJ65 * w[5] * w[4] * HF65.T.dot(T).dot(HF65)+
                                detJ66 * w[5] * w[5] * HF66.T.dot(T).dot(HF66)+
                                detJ67 * w[5] * w[6] * HF67.T.dot(T).dot(HF67)+
                                detJ61 * w[5] * w[0] * HS61.T.dot(T).dot(HS61)+
                                detJ62 * w[5] * w[1] * HS62.T.dot(T).dot(HS62)+
                                detJ63 * w[5] * w[2] * HS63.T.dot(T).dot(HS63)+
                                detJ64 * w[5] * w[3] * HS64.T.dot(T).dot(HS64)+
                                detJ65 * w[5] * w[4] * HS65.T.dot(T).dot(HS65)+
                                detJ66 * w[5] * w[5] * HS66.T.dot(T).dot(HS66)+
                                detJ67 * w[5] * w[6] * HS67.T.dot(T).dot(HS67)+
                                detJ71 * w[6] * w[0] * HF71.T.dot(T).dot(HF71)+
                                detJ72 * w[6] * w[1] * HF72.T.dot(T).dot(HF72)+
                                detJ73 * w[6] * w[2] * HF73.T.dot(T).dot(HF73)+
                                detJ74 * w[6] * w[3] * HF74.T.dot(T).dot(HF74)+
                                detJ75 * w[6] * w[4] * HF75.T.dot(T).dot(HF75)+
                                detJ76 * w[6] * w[5] * HF76.T.dot(T).dot(HF76)+
                                detJ77 * w[6] * w[6] * HF77.T.dot(T).dot(HF77)+
                                detJ71 * w[6] * w[0] * HS71.T.dot(T).dot(HS71)+
                                detJ72 * w[6] * w[1] * HS72.T.dot(T).dot(HS72)+
                                detJ73 * w[6] * w[2] * HS73.T.dot(T).dot(HS73)+
                                detJ74 * w[6] * w[3] * HS74.T.dot(T).dot(HS74)+
                                detJ75 * w[6] * w[4] * HS75.T.dot(T).dot(HS75)+
                                detJ76 * w[6] * w[5] * HS76.T.dot(T).dot(HS76)+
                                detJ77 * w[6] * w[6] * HS77.T.dot(T).dot(HS77))   
            
            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS                        
        return stiffT
