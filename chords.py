#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ceChords import septalChord
from gaussQuadratures import gaussQuadChord1, gaussQuadChord3, gaussQuadChord5
import materialProperties as mp
import math
import numpy as np
from peceCE import control, pece
from ridder import findRoot
from shapeFnChords import shapeFunction
import spin as spinMtx
from vertices import vertex


"""
Module chords.py provides geometric information about a septal chord.

Copyright (c) 2020 Alan D. Freed

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
__version__ = "1.0.0"
__date__ = "08-08-2019"
__update__ = "05-20-2020"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


Class chord in file chords.py allows for the creation of objects that are to
be used to represent chords that connect vertices in a polyhedron.  A chord is
assigned an unique number, two distinct vertices that serve as its end points,
the time-step size used to approximate derivatives and integrals, plus the
number of Gauss points that are to be used for integrating over its length.

Initial co-ordinates that locate vertices in a dodecahedron used to model the
alveoli of lung are assigned according to a reference configuration where the
pleural pressure (the pressure surrounding lung in the pleural cavity) and the
transpulmonary pressure (the difference between aleolar and pleural pressures)
are both at zero gauge pressure, i.e., all pressures are atmospheric pressure.
The pleural pressure is normally negative, sucking the pleural membrane against
the wall of the chest.  During expiration, the diaphragm is pushed up, reducing
the pleural pressure.  The pleural pressure remains negative during breathing
at rest, but it can become positive during active expiration.  The surface
tension created by surfactant keeps most alveoli open during excursions into
the range of positive pleural pressures, but not all will remain open.  Alveoli
are their smallest at max expiration.  Alveolar size is determined by the
transpulmonary pressure.  The greater the transpulmonary pressure the greater
the alveolar size will be.

Numerous methods have a string argument that is denoted as  'state'  which can
take on any of the following values:
    'c', 'curr', 'current'       gets the value for a current configuration
    'n', 'next'                  gets the value for a next configuration
    'p', 'prev', 'previous'      gets the value for a previous configuration
    'r', 'ref', 'reference'      gets the value for the reference configuration

Co-ordinates are handled as tuples; vector fields are handled as arrays;
tensor fields are handled as matrices.


class chord

A chord object, say c, can be printed out to the command window using the
following command.  The object printed associates with the current state.

    print(c)

constructor

    c = chord(number, vertex1, vertex2, dt, polyDegree, m=1)
        number      is an immutable value that is unique to this chord
        vertex1     is an end point of the chord, an instance of class vertex
        vertex2     is an end point of the chord, an instance of class vertex
        dt          is the time seperating any two neighboring configurations
        polyDegree  order of polynomials integrated exactly: must be 1, 3 or 5
        m           is the number of CE iterations, i.e., PE(CE)^m, m in [0, 5]

methods

    s = c.toString(state)
        returns a string representation for this chord in configuration 'state'

    n = c.number()
        returns the unique indexing number affiliated with this chord

    v1, v2 = c.vertexNumbers()
        returns the unique numbers assigned to the two vertices of this chord

    truth = c.hasVertex(number)
        returns 'True' if one of the two vertices has this vertex number

    v = c.getVertex(number)
        returns a vertex; to be called inside, e.g., a c.hasVertex if clause

    n = c.gaussPoints()
        returns the number of Gauss points assigned to this chord

    c.update()
        assigns new co-ordinate values to the chord for its next location and
        updates all effected fields.  It is to be called AFTER all vertices
        have had their co-ordinates updated.  It may be called multiple times
        before freezing the co-ordinate values with a call to chord.advance

    c.advance()
        assigns the current fields to the previous fields, and then assigns
        the next fields to the current fields, thereby freezing the present
        next-fields in preparation for an advancment of a solution along its
        path of motion

    Material properties that associate with this chord.

    rho = c.massDensity()
        returns the mass density of the chord (collagen and elastin fibers)

    Geometric fields associated with a chord in 3 space.  For those fields that
    are constructed from difference formulae, it is necessary that they be
    rotated into the re-indexed co-ordinate frame for the 'state' of interest

    a = c.area(state)
        returns the cross-sectional area of the chord, i.e., both the collagen
        and elastin fibers in configuration 'state', deformed under an
        assumption that chordal volume is preserved

    ell = c.length(state)
        returns the chordal length in configuration 'state'

    lambda = c.stretch(state)
        returns the chordal stretch in configuration 'state'

    Kinematic fields associated with the centroid of a chord in 3 space are

    [x, y, z] = c.centroid(state)
        returns coordinates for the chordal mid-point in configuration 'state'

    [ux, uy, uz] = c.displacement(reindex, state)
        returns the displacement of the centroid in configuration 'state'

    [vx, vy, vz] = c.velocity(reindex, state)
        returns the velocity of the centroid in configuration 'state'

    [ax, ay, az] = c.acceleration(reindex, state)
        returns the acceleration of the centroid in configuration 'state'

    Rotation and spin of a chord wrt the dodecahedral coordinate system are

    pMtx = c.rotation(state)
        returns a 3x3 orthogonal matrix that rotates the reference base vectors
        (E_1, E_2, E_3) into a set of local base vectors (e_1, e_2, e_3) that
        pertain to a chord whose axis aligns with the e_1 direction, while the
        e_2 direction passes through the origin of the dodecahedral reference
        co-ordinate system.  The returned matrix associates with configuration
        'state'

    omegaMtx = c.spin(reindex, state)
        input
            reindex is an instance of class pivot from pivotIncomingF.py
        output
        returns a 3x3 skew symmetric matrix that describes the time rate of
        change in rotation, i.e., the spin of the local chordal co-ordinate
        system (e_1, e_2, e_3) about a fixed co-ordinate frame (E_1, E_2, E_3)
        belonging to the dodecahedron.  The returned matrix associates with
        configuration 'state'

    Thermodynamic fields evaluated in the chordal co-ordinate system:

    epsilon = c.strain(state)
        returns the logarithmic strain of the chord in configuration 'state'

    dEpsilon = c.dStrain(state)
        returns the logarithmic strain rate of the chord in 'state'

    sigma = c.stress(state)
        returns the stress carried by the chord in configuration 'state'

    f = c.force(state)
        returns the force carried by the chord in configuration 'state'

    truth = c.collagenHasRuptured(state)
        returns True if the collagen fiber in the chord has ruptured

    truth = c.elastinHasRuptured(state)
        returns True if the elastin fiber in the chord has ruptured

    Kinematic fields evaluated in the chordal co-ordinate system:

    gMtx = c.G(gaussPt, state)
        returns the displacement gradient G at a specified Gauss point for the
        specified configuration.  gMtx is a 1x1 matrix for a chord.

    fMtx = c.F(gaussPt, state)
        returns the deformation gradient F at a specified Gauss point for the
        specified configuration.  fMtx is a 1x1 matrix for a chord.

    lMtx = c.L(gaussPt, state)
        returns the velocity gradient L at a specified Gauss point for the
        specified configuration.  lMtx is a 1x1 matrix for a chord.

    Fields needed to construct a finite element solution strategy are:

    sf = c.shapeFunction(gaussPt):
        returns the shape function associated with a specified Gauss point.

    gq = c.gaussQuadrature()
        returns the Gauss quadrature rule being used for integration.

    mMtx = c.massMatrix()
        returns an average of the lumped and consistent mass matrices (thereby
        ensuring that the mass matrix is not singular) for a chosen number of
        Gauss points for a chord whose mass density, rho, and whose cross-
        sectional area are considered to be uniform over the length of the
        chord.  The mass matrix is constant and therefore independent of state.

    kMtx = c.stiffnessMatrix()
        returns a tangent stiffness matrix for the chosen number of Gauss
        points belonging to the current state.  An updated Lagrangian
        formulation is implemented.

    fVec = c.forcingFunction()
        returns a vector describing the forcing function on the right-hand side
        belonging to the current state.  An updated Lagrangian formulation is
        implemented.
"""

# create a control object for the PECE integrator


class chordCtrl(control):

    def __init__(self, ctrlVars, dt, temp0, len0):
        # Call the constructor of the base type.
        super().__init__(ctrlVars, dt)
        # This creates the counter  self.step  which may be useful.
        # Add any other information for your inhereted type, as required.
        self.TR = temp0    # initial temperature
        self.LR = len0     # initial length
        # create their fields for the previous, current and next steps
        self.TP = temp0
        self.TC = temp0
        self.TN = temp0
        self.LP = len0
        self.LC = len0
        self.LN = len0
        return  # a new instance of type chordCtrl

    def x(self, tN, tempN, lenN):
        # Call the base implementation of this method to create xVec.
        xVec = super().x(tN)
        # You will need to add your application's control functions here,
        # i.e., you will need to populate xVec before returning.
        # The temperature at time t in centigrade
        xVec[0] = tempN
        # The strain, i.e., log of stretch, at time t
        xVec[1] = math.log(lenN / self.LR)
        # update the data structure
        self.TN = tempN
        self.LN = lenN
        return xVec

    # call x before calling dxdt
    def dxdt(self, tN, restart=False):
        # Do not call the base implementation of this method to build dxdtVec.
        dxdtVec = np.zeros((2,), dtype=float)
        if restart is True:
            self.step = 1
        if self.step == 0:
            # a first-order forward finite difference
            dxdtVec[0] = (self.TN - self.TC) / self.dt
            dxdtVec[1] = (self.LN - self.LC) / (self.LC * self.dt)
        elif self.step == 1:
            # a first-order backward finite difference
            dxdtVec[0] = (self.TN - self.TC) / self.dt
            dxdtVec[1] = (self.LN - self.LC) / (self.LN * self.dt)
        else:
            # a second-order backward finite difference
            dxdtVec[0] = ((3.0 * self.TN - 4.0 * self.TC + self.TP) /
                          (2.0 * self.dt))
            dxdtVec[1] = ((3.0 * self.LN - 4.0 * self.LC + self.LP) /
                          (2.0 * self.LN * self.dt))
        return dxdtVec

    def advance(self):
        # Call the base implementation of this method
        super().advance()
        # Called internally by the pece integrator.  Do not call it yourself.
        # Update your object's data structure, if required
        self.TP = self.TC
        self.TC = self.TN
        self.LP = self.LC
        self.LC = self.LN
        return  # nothing


class chord(object):

    def __init__(self, number, vertex1, vertex2, dt, polyDegree, m=1):
        stressR = 0.0
        # verify the inputs

        # provide an unique identifier for the chord
        if isinstance(number, int):
            self._number = number
        else:
            raise RuntimeError("The chord number must be an integer.")

        if not isinstance(vertex1, vertex):
            raise RuntimeError('vertex1 sent to the chord ' +
                               'constructor was not of type vertex.')
        if not isinstance(vertex2, vertex):
            raise RuntimeError('vertex2 sent to the chord ' +
                               'constructor was not of type vertex.')

        # save the vertices in a dictionary
        if vertex1.number() < vertex2.number():
            self._vertex = {
                1: vertex1,
                2: vertex2
            }
        elif vertex1.number() > vertex2.number():
            self._vertex = {
                1: vertex2,
                2: vertex1
            }
        else:
            raise RuntimeError('A chord must have two distinct vertices.')

        # create the time variables
        if isinstance(dt, float) and dt > np.finfo(float).eps:
            self._h = dt
        else:
            raise RuntimeError("The timestep size dt sent to the chord " +
                               "constructor must exceed machine precision.")
        self._tP = -dt
        self._tC = 0.0
        self._tN = dt

        # assign the Gauss quadrature rule to be used
        if polyDegree == 1:
            self._gq = gaussQuadChord1
        elif polyDegree == 3:
            self._gq = gaussQuadChord3
        elif polyDegree == 5:
            self._gq = gaussQuadChord5
        else:
            raise RuntimeError('A Gauss quadrature rule capable of ' +
                               'integrating polynomials up to degree ' +
                               '{} \n'.format(polyDegree) +
                               'was specified in a call to the chord ' +
                               'constructor, but degree must be 1, 3 or 5.')

        # limit the range for m in our implementation of PE(CE)^m
        if isinstance(m, int):
            if m < 0:
                self.m = 0
            elif m > 5:
                self.m = 5
            else:
                self.m = m
        else:
            raise RuntimeError("Argument m must be an integer within [0, 5].")

        # create the four rotation matrices: rotate dodecahedral into chordal
        self._Pr3D = np.identity(3, dtype=float)
        self._Pp3D = np.identity(3, dtype=float)
        self._Pc3D = np.identity(3, dtype=float)
        self._Pn3D = np.identity(3, dtype=float)

        # initialize the chordal lengths for all configurations
        x1 = self._vertex[1].coordinates('ref')
        x2 = self._vertex[2].coordinates('ref')
        L0 = m.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2 +
                    (x2[2] - x1[2])**2)
        self._L0 = L0
        self._Lp = L0
        self._Lc = L0
        self._Ln = L0

        # initialize the centroids for all configurations
        self._centroidX0 = (x1[0] + x2[0]) / 2.0
        self._centroidY0 = (x1[1] + x2[1]) / 2.0
        self._centroidZ0 = (x1[2] + x2[2]) / 2.0
        self._centroidXp = self._centroidX0
        self._centroidYp = self._centroidY0
        self._centroidZp = self._centroidZ0
        self._centroidXc = self._centroidX0
        self._centroidYc = self._centroidY0
        self._centroidZc = self._centroidZ0
        self._centroidXn = self._centroidX0
        self._centroidYn = self._centroidY0
        self._centroidZn = self._centroidZ0

        # base vector 1: aligns with the axis of the chord
        x = x2[0] - x1[0]
        y = x2[1] - x1[1]
        z = x2[2] - x1[2]
        mag = m.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1[0] + x2[0]) / 2.0
        y = (x1[1] + x2[1]) / 2.0
        z = (x1[2] + x2[2]) / 2.0
        mag = m.sqrt(x * x + y * y + z * z)
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta(delta):
            nx = ex + delta * n1x
            ny = ey + delta * n1y
            nz = ez + delta * n1z
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x * nx + n1y * ny + n1z * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L0
        deltaH = 4.0 * self._L0
        delta = findRoot(deltaL, deltaH, getDelta)

        # create base vector 2
        x = ex + delta * n1x
        y = ey + delta * n1y
        z = ez + delta * n1z
        mag = m.sqrt(x * x + y * y + z * z)
        n2x = x / mag
        n2y = y / mag
        n2z = z / mag

        # base vector 3 (the binormal) is obtained through a cross product
        n3x = n1y * n2z - n1z * n2y
        n3y = n1z * n2x - n1x * n2z
        n3z = n1x * n2y - n1y * n2x

        # create the rotation matrix from dodecahedral to chordal co-ordinates
        self._Pr3D[0, 0] = n1x
        self._Pr3D[0, 1] = n2x
        self._Pr3D[0, 2] = n3x
        self._Pr3D[1, 0] = n1y
        self._Pr3D[1, 1] = n2y
        self._Pr3D[1, 2] = n3y
        self._Pr3D[2, 0] = n1z
        self._Pr3D[2, 1] = n2z
        self._Pr3D[2, 2] = n3z
        self._Pp3D[:, :] = self._Pr3D[:, :]
        self._Pc3D[:, :] = self._Pr3D[:, :]
        self._Pn3D[:, :] = self._Pr3D[:, :]

        # establish the shape functions located at the various Gauss points
        if polyDegree == 1:
            # this quadrature rule has a single Gauss point
            atGaussPt = 1
            sf1 = shapeFunction(self._gq.coordinates(atGaussPt))

            self._shapeFns = {
                1: sf1
            }
        elif polyDegree == 3:
            # this quadrature rule has two Gauss points
            atGaussPt = 1
            sf1 = shapeFunction(self._gq.coordinates(atGaussPt))
            atGaussPt = 2
            sf2 = shapeFunction(self._gq.coordinates(atGaussPt))

            self._shapeFns = {
                1: sf1,
                2: sf2
            }
        else:  # degree == 5
            # this quadrature rule has three Gauss points
            atGaussPt = 1
            sf1 = shapeFunction(self._gq.coordinates(atGaussPt))
            atGaussPt = 2
            sf2 = shapeFunction(self._gq.coordinates(atGaussPt))
            atGaussPt = 3
            sf3 = shapeFunction(self._gq.coordinates(atGaussPt))

            self._shapeFns = {
                1: sf1,
                2: sf2,
                3: sf3
            }

        # create the displacement and deformation gradients for a chord at
        # their Gauss points via dictionaries.  '0' implies reference,
        # 'p' implies previous, 'c' implies current, and 'n' implies next.
        if polyDegree == 1:
            # displacement gradients located at the Gauss points of a chord
            self._G0 = {
                1: np.zeros((1, 1), dtype=float)
            }
            self._Gp = {
                1: np.zeros((1, 1), dtype=float)
            }
            self._Gc = {
                1: np.zeros((1, 1), dtype=float)
            }
            self._Gn = {
                1: np.zeros((1, 1), dtype=float)
            }
            # deformation gradients located at the Gauss points of a chord
            self._F0 = {
                1: np.ones((1, 1), dtype=float)
            }
            self._Fp = {
                1: np.ones((1, 1), dtype=float)
            }
            self._Fc = {
                1: np.ones((1, 1), dtype=float)
            }
            self._Fn = {
                1: np.ones((1, 1), dtype=float)
            }
        elif polyDegree == 3:
            # displacement gradients located at the Gauss points of a chord
            self._G0 = {
                1: np.zeros((1, 1), dtype=float),
                2: np.zeros((1, 1), dtype=float)
            }
            self._Gp = {
                1: np.zeros((1, 1), dtype=float),
                2: np.zeros((1, 1), dtype=float)
            }
            self._Gc = {
                1: np.zeros((1, 1), dtype=float),
                2: np.zeros((1, 1), dtype=float)
            }
            self._Gn = {
                1: np.zeros((1, 1), dtype=float),
                2: np.zeros((1, 1), dtype=float)
            }
            # deformation gradients located at the Gauss points of a chord
            self._F0 = {
                1: np.ones((1, 1), dtype=float),
                2: np.ones((1, 1), dtype=float)
            }
            self._Fp = {
                1: np.ones((1, 1), dtype=float),
                2: np.ones((1, 1), dtype=float)
            }
            self._Fc = {
                1: np.ones((1, 1), dtype=float),
                2: np.ones((1, 1), dtype=float)
            }
            self._Fn = {
                1: np.ones((1, 1), dtype=float),
                2: np.ones((1, 1), dtype=float)
            }
        else:  # polyDegree == 5
            # displacement gradients located at the Gauss points of a chord
            self._G0 = {
                1: np.zeros((1, 1), dtype=float),
                2: np.zeros((1, 1), dtype=float),
                3: np.zeros((1, 1), dtype=float)
            }
            self._Gp = {
                1: np.zeros((1, 1), dtype=float),
                2: np.zeros((1, 1), dtype=float),
                3: np.zeros((1, 1), dtype=float)
            }
            self._Gc = {
                1: np.zeros((1, 1), dtype=float),
                2: np.zeros((1, 1), dtype=float),
                3: np.zeros((1, 1), dtype=float)
            }
            self._Gn = {
                1: np.zeros((1, 1), dtype=float),
                2: np.zeros((1, 1), dtype=float),
                3: np.zeros((1, 1), dtype=float)
            }
            # deformation gradients located at the Gauss points of a chord
            self._F0 = {
                1: np.ones((1, 1), dtype=float),
                2: np.ones((1, 1), dtype=float),
                3: np.ones((1, 1), dtype=float)
            }
            self._Fp = {
                1: np.ones((1, 1), dtype=float),
                2: np.ones((1, 1), dtype=float),
                3: np.ones((1, 1), dtype=float)
            }
            self._Fc = {
                1: np.ones((1, 1), dtype=float),
                2: np.ones((1, 1), dtype=float),
                3: np.ones((1, 1), dtype=float)
            }
            self._Fn = {
                1: np.ones((1, 1), dtype=float),
                2: np.ones((1, 1), dtype=float),
                3: np.ones((1, 1), dtype=float)
            }

        # create the constitutive model for this chord
        diaC = mp.fiberDiameterCollagen()
        diaE = mp.fiberDiameterElastin()
        self._sc = septalChord(L0, diaC, diaE)

        # create the control object
        nbrVars = 2  # temperature and strain (chordal lengths are passed)
        T0 = 37.0     # body temperature
        self._ctrl = chordCtrl(nbrVars, dt, T0, L0)

        # create the response objects
        self._respC = self._sc.bioFiberCollagen()
        self._respE = self._sc.bioFiberElastin()
        y0C = np.zeros((nbrVars,), dtype=float)
        y0E = np.zeros((nbrVars,), dtype=float)
        y0C[0] = mp.etaCollagen()
        y0C[1] = stressR
        y0E[0] = mp.etaElastin()
        y0E[1] = stressR

        # create their integrators
        self._peceC = pece(self._tC, y0C, dt, self._ctrl, self._respC, m)
        self._peceE = pece(self._tC, y0E, dt, self._ctrl, self._respE, m)

        # establish physical properties of this chord
        self._rho = self._sc.massDensityChord()

        # create the fiber forces
        self._forceP = y0C[1]
        self._forceC = y0C[1]
        self._forceN = y0C[1]

        return  # a new chord object

    def __str__(self):
        return self.toString('curr')

    def toString(self, state):
        if self._number < 10:
            s = 'chord[0'
        else:
            s = 'chord['
        s = s + str(self._number)
        s = s + '] has vertices: \n'
        if isinstance(state, str):
            s = s + '   ' + self._vertex[1].toString(state) + '\n'
            s = s + '   ' + self._vertex[2].toString(state)
        else:
            raise RuntimeError("Unknown state {} ".format(str(state)) +
                               "in a call to chord.toString.")
        return s

    def number(self):
        return self._number

    def vertexNumbers(self):
        return self._vertex[1].number(), self._vertex[2].number()

    def hasVertex(self, number):
        if self._vertex[1].number() == number:
            return True
        elif self._vertex[2].number() == number:
            return True
        else:
            return False

    def getVertex(self, number):
        if self._vertex[1].number() == number:
            return self._vertex[1]
        elif self._vertex[2].number() == number:
            return self._vertex[2]
        else:
            raise RuntimeError('Vertex {} '.format(number) + "does not " +
                               'belong to chord {}.'.format(self._number))

    def gaussPoints(self):
        return self._gq.nodes

    def update(self):
        # determine length of the chord in the next configuration
        x1 = self._vertex[1].coordinates('next')
        x2 = self._vertex[2].coordinates('next')
        self._Ln = m.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2 +
                          (x2[2] - x1[2])**2)

        # determine the centroid of this chord
        self._centroidXn = (x1[0] + x2[0]) / 2.0
        self._centroidYn = (x1[1] + x2[1]) / 2.0
        self._centroidZn = (x1[2] + x2[2]) / 2.0

        # base vector 1: aligns with the axis of the chord
        x = x2[0] - x1[0]
        y = x2[1] - x1[1]
        z = x2[2] - x1[2]
        mag = m.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1[0] + x2[0]) / 2.0
        y = (x1[1] + x2[1]) / 2.0
        z = (x1[2] + x2[2]) / 2.0
        mag = m.sqrt(x * x + y * y + z * z)
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to base vector n1
        def getDelta(delta):
            nx = ex + delta * n1x
            ny = ey + delta * n1y
            nz = ez + delta * n1z
            # when the dot product is zero, the two vectors are orthogonal
            n1Dotn2 = n1x * nx + n1y * ny + n1z * nz
            return n1Dotn2

        # use a root finder to secure a base vector n2 that is orthogonal to n1
        deltaL = -4.0 * self._L0
        deltaH = 4.0 * self._L0
        delta = findRoot(deltaL, deltaH, getDelta)

        # create base vector 2
        x = ex + delta * n1x
        y = ey + delta * n1y
        z = ez + delta * n1z
        mag = m.sqrt(x * x + y * y + z * z)
        n2x = x / mag
        n2y = y / mag
        n2z = z / mag

        # base vector 3 is obtained through a cross product
        n3x = n1y * n2z - n1z * n2y
        n3y = n1z * n2x - n1x * n2z
        n3z = n1x * n2y - n1y * n2x

        # create the rotation matrix from dodecahedral to chordal co-ordinates
        self._Pn3D[0, 0] = n1x
        self._Pn3D[0, 1] = n2x
        self._Pn3D[0, 2] = n3x
        self._Pn3D[1, 0] = n1y
        self._Pn3D[1, 1] = n2y
        self._Pn3D[1, 2] = n3y
        self._Pn3D[2, 0] = n1z
        self._Pn3D[2, 1] = n2z
        self._Pn3D[2, 2] = n3z

        # chordal co-ordinates for the chords, must be tuples
        x10 = (-self._L0 / 2.0,)
        x20 = (self._L0 / 2.0,)
        x1n = (-self._Ln / 2.0,)
        x2n = (self._Ln / 2.0,)

        # quantify the displacement and deformation gradients of the chord
        if self.gaussPoints() == 1:
            # displacement gradient located at the Gauss point of the chord
            self._Gn[1] = self._shapeFns[1].G(x1n, x2n, x10, x20)
            # deformation gradient located at the Gauss point of the chord
            self._Fn[1] = self._shapeFns[1].F(x1n, x2n, x10, x20)
        elif self.gaussPoints() == 2:
            # displacement gradients located at the Gauss points of a chord
            self._Gn[1] = self._shapeFns[1].G(x1n, x2n, x10, x20)
            self._Gn[2] = self._shapeFns[2].G(x1n, x2n, x10, x20)
            # deformation gradients located at the Gauss points of a chord
            self._Fn[1] = self._shapeFns[1].F(x1n, x2n, x10, x20)
            self._Fn[2] = self._shapeFns[2].F(x1n, x2n, x10, x20)
        else:  # gaussPoints == 3
            # displacement gradients located at the Gauss points of a chord
            self._Gn[1] = self._shapeFns[1].G(x1n, x2n, x10, x20)
            self._Gn[2] = self._shapeFns[2].G(x1n, x2n, x10, x20)
            self._Gn[3] = self._shapeFns[3].G(x1n, x2n, x10, x20)
            # deformation gradients located at the Gauss points of a chord
            self._Fn[1] = self._shapeFns[1].F(x1n, x2n, x10, x20)
            self._Fn[2] = self._shapeFns[2].F(x1n, x2n, x10, x20)
            self._Fn[3] = self._shapeFns[3].F(x1n, x2n, x10, x20)

        # integrate the constitutive equations
        stress = np.array([0.0])
        self._peceC.integrate()
        stress = self._peceC.getX()
        self._stressCn = stress[0]
        self._peceE.integrate()
        stress = self._peceE.getX()
        self._stressEn = stress[0]

        return  # nothing, the data structure has been updated

    def advance(self):
        # assign current to previous values, and then next to current values
        self._Lp = self._Lc
        self._Lc = self._Ln
        self._centroidXp = self._centroidXc
        self._centroidYp = self._centroidYc
        self._centroidZp = self._centroidZc
        self._centroidXc = self._centroidXn
        self._centroidYc = self._centroidYn
        self._centroidZc = self._centroidZn
        self._Pp3D[:, :] = self._Pc3D[:, :]
        self._Pc3D[:, :] = self._Pn3D[:, :]

        # advance the matrix fields associated with each Gauss point
        for i in range(1, self.gaussPoints()+1):
            self._Fp[i] = self._Fc[i]
            self._Fc[i] = self._Fn[i]
            self._Gp[i] = self._Gc[i]
            self._Gc[i] = self._Gn[i]

        # advance the independent variable of integration, i.e., time
        self._tP = self._tC
        self._tC = self._tN
        self._tN += self._h

        # advance the dependent variables of integration, i.e., the stresses
        self._stressCp = self._stressCc
        self._stressEp = self._stressEc
        self._stressCc = self._stressCn
        self._stressEc = self._stressEn

        # advance the integrators
        self._peceC.advance()
        self._peceE.advance()

        return  # nothing, the data structure has been advanced

    # Material properties that associate with this chord.  Except for the mass
    # density, all are drawn randomly from various statistical distributions.

    def massDensity(self):
        # returns the mass density of the chord (collagen and elastin fibers)
        return self._rho

    def areaCollagen(self, state):
        # returns the cross-sectional area of the collagen fiber, assuming
        # volume is preserved
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._areaC * self._L0 / self._Lc
            elif state == 'n' or state == 'next':
                return self._areaC * self._L0 / self._Ln
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._areaC * self._L0 / self._Lp
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._areaC
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.areaCollagen.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in call a to chord.areaCollagen.")

    def areaElastin(self, state):
        # returns the cross-sectional area of the elastin fiber, assuming
        # volume is preserved
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._areaE * self._L0 / self._Lc
            elif state == 'n' or state == 'next':
                return self._areaE * self._L0 / self._Ln
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._areaE * self._L0 / self._Lp
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._areaE
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.areaElastin.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in call a to chord.areaElastin.")














    def tangentModulusCollagen(self, temperature, fiberStress, fiberStrain):
        # tangent modulus of a Freed-Rajagopal elastic fiber
        if fiberStress > 0.0:
            nonLinStrain = (fiberStrain -
                            mp.alphaCollagen() * (temperature + 273 - 310) -
                            fiberStress / self._Ec2)
            if nonLinStrain < self._ec_t:
                compliance = ((self._ec_t - nonLinStrain) /
                              (self._Ec1 * self._ec_t + 2.0 * fiberStress) +
                              1.0 / self._Ec2)
                return 1.0 / compliance
            else:
                return self._Ec2
        else:
            return self._Ec1 * self._Ec2 / (self._Ec1 + self._Ec2)

    def tangentModulusElastin(self, temperature, fiberStress, fiberStrain):
        # tangent modulus of a Freed-Rajagopal elastic fiber
        if fiberStress > 0.0:
            nonLinStrain = (fiberStrain -
                            mp.alphaElastin() * (temperature + 273 - 310) -
                            fiberStress / self._Ee2)
            if nonLinStrain < self._ee_t:
                compliance = ((self._ee_t - nonLinStrain) /
                              (self._Ee1 * self._ee_t + 2.0 * fiberStress) +
                              1.0 / self._Ee2)
                return 1.0 / compliance
            else:
                return self._Ee2
        else:
            return self._Ee1 * self._Ee2 / (self._Ee1 + self._Ee2)

    def odeCollagen(self, time, entropyStress):
        # verify the input
        if (time < 0.99999 * self._tP) or (time > 1.00001 * self._tN):
            raise RuntimeError("Time is being extrapolated instead of being " +
                               "interpolated in a call to chord.odeCollagen.")
        if not isinstance(entropyStress, np.ndarray):
            raise RuntimeError("Variable entropyStress must be a NumPy array.")

        # obtain the Lagrange interpolation weights
        wp = ((time - self._tC)*(time - self._tN) /
              ((self._tP - self._tC)*(self._tP - self._tN)))
        wc = ((time - self._tP)*(time - self._tN) /
              ((self._tC - self._tP)*(self._tC - self._tN)))
        wn = ((time - self._tP)*(time - self._tC) /
              ((self._tN - self._tP)*(self._tN - self._tC)))

        # interpolate strain and its rate
        strain = (wp * self.strain("prev") + wc * self.strain("curr") +
                  wn * self.strain("next"))
        strainRate = (wp * self.dStrain("prev") +
                      wc * self.dStrain("curr") +
                      wn * self.dStrain("next"))

        # construct the governing ODE to be integrated by the PECE method
        C = mp.CpCollagen()
        alpha = mp.alphaCollagen()
        rho = mp.rhoCollagen()
        E = self.tangentModulusCollagen(stress, strain)
        M = np.zeros((2, 2), dtype=float)
        M[0, 0] = (mp.CpCollage() / 310.0 -
                   mp.alphaCollagen()**2 * E / mp.rhoCollagen())
        return stressRate

    def odeElastin(self, time, stress):
        if (time < 0.99999 * self._tP) or (time > 1.00001 * self._tN):
            raise RuntimeError("Time is being extrapolated instead of being " +
                               "interpolated in a call to chord.odeElastin.")
        # obtain the Lagrange interpolation weights
        wp = ((time - self._tC)*(time - self._tN) /
              ((self._tP - self._tC)*(self._tP - self._tN)))
        wc = ((time - self._tP)*(time - self._tN) /
              ((self._tC - self._tP)*(self._tC - self._tN)))
        wn = ((time - self._tP)*(time - self._tC) /
              ((self._tN - self._tP)*(self._tN - self._tC)))

        # interpolate strain and its rate
        strain = (wp * self.strain("prev") + wc * self.strain("curr") +
                  wn * self.strain("next"))
        strainRate = (wp * self.dStrain("prev") +
                      wc * self.dStrain("curr") +
                      wn * self.dStrain("next"))

        # construct the governing ODE to be integrated by the PECE method
        dStressdStrain = self.tangentModulusElastin(stress, strain)
        stressRate = dStressdStrain * strainRate
        return stressRate


















    # geometric properties of the chord

    def area(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return (self._areaC + self._areaE) * self._L0 / self._Lc
            elif state == 'n' or state == 'next':
                return (self._areaC + self._areaE) * self._L0 / self._Ln
            elif state == 'p' or state == 'prev' or state == 'previous':
                return (self._areaC + self._areaE) * self._L0 / self._Lp
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._areaC + self._areaE
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.area.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in call a to chord.area.")

    def length(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._Lc
            elif state == 'n' or state == 'next':
                return self._Ln
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._Lp
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._L0
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.length.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in call a to chord.length.")

    def stretch(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._Lc / self._L0
            elif state == 'n' or state == 'next':
                return self._Ln / self._L0
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._Lp / self._L0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 1.0
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.stretch.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in a call to chord.stretch.")

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
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.centroid.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in a call to chord.centroid.")
        return np.array([cx, cy, cz])

    def displacement(self, reindex, state):
        u1 = self._vertex[1].displacement(reindex, state)
        u2 = self._vertex[2].displacement(reindex, state)
        u = 0.5 * (u1 + u2)
        return u

    def velocity(self, reindex, state):
        v1 = self._vertex[1].velocity(reindex, state)
        v2 = self._vertex[2].velocity(reindex, state)
        v = 0.5 * (v1 + v2)
        return v

    def acceleration(self, reindex, state):
        a1 = self._vertex[1].acceleration(reindex, state)
        a2 = self._vertex[2].acceleration(reindex, state)
        a = 0.5 * (a1 + a2)
        return a

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
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.rotation.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in a call to chord.rotation.")

    def spin(self, reindex, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return spinMtx.currSpin(self._Pp3D, self._Pc3D,
                                        self._Pn3D, reindex, self._h)
            elif state == 'n' or state == 'next':
                return spinMtx.nextSpin(self._Pp3D, self._Pc3D,
                                        self._Pn3D, reindex, self._h)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return spinMtx.prevSpin(self._Pp3D, self._Pc3D,
                                        self._Pn3D, reindex, self._h)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.zeros((3, 3), dtype=float)
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.spin.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in a call to chord.spin.")

    # thermodynamic fields associated with a chord

    def strain(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return m.log(self._Lc / self._L0)
            elif state == 'n' or state == 'next':
                return m.log(self._Ln / self._L0)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return m.log(self._Lp / self._L0)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.strain.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in a call to chord.strain.")

    def dStrain(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return (self._Ln - self._Lp) / (2.0 * self._h * self._Lc)
            elif state == 'n' or state == 'next':
                return ((3.0 * self._Ln - 4.0 * self._Lc + self._Lp) /
                        (2.0 * self._h * self._Ln))
            elif state == 'p' or state == 'prev' or state == 'previous':
                return ((-self._Ln + 4.0 * self._Lc - 3.0 * self._Lp) /
                        (2.0 * self._h * self._Lp))
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.dStrain.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in a call to chord.dStrain.")

    def stress(self, state):
        if isinstance(state, str):
            ac = self.areaCollagen(state)
            ae = self.areaElastin(state)
            if state == 'c' or state == 'curr' or state == 'current':
                return (self._stressCc * ac + self._stressEc * ae) / (ac + ae)
            elif state == 'n' or state == 'next':
                return (self._stressCn * ac + self._stressEn * ae) / (ac + ae)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return (self._stressCp * ac + self._stressEp * ae) / (ac + ae)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.stress.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in a call to chord.stress.")

    def force(self, state):
        if isinstance(state, str):
            ac = self.areaCollagen(state)
            ae = self.areaElastin(state)
            if state == 'c' or state == 'curr' or state == 'current':
                return (self._stressCc * ac + self._stressEc * ae)
            elif state == 'n' or state == 'next':
                return (self._stressCn * ac + self._stressEn * ae)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return (self._stressCp * ac + self._stressEp * ae)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.force.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in a call to chord.force.")

    # fundamental kinematic fields

    # displacement gradient at a Gauss point
    def G(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self.gaussPoints()):
            if self.gaussPoints() == 1:
                raise RuntimeError("gaussPt can only be 1 in a call to " +
                                   "chord.G and you sent " +
                                   "{}.".format(gaussPt))
            else:
                raise RuntimeError("gaussPt must be in the range of " +
                                   "[1, {}] ".format(self.gaussPoints()) +
                                   "in a call to chord.G and you sent " +
                                   "{}.".format(gaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._Gc[gaussPt]
            elif state == 'n' or state == 'next':
                return self._Gn[gaussPt]
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._Gp[gaussPt]
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._G0[gaussPt]
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "sent in a call to chord.G.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "sent in a call to chord.G.")

    # deformation gradient at a Gauss point
    def F(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self.gaussPoints()):
            if self.gaussPoints() == 1:
                raise RuntimeError("gaussPt can only be 1 in a call to " +
                                   "chord.F and you sent " +
                                   "{}.".format(gaussPt))
            else:
                raise RuntimeError("gaussPt must be in the range of " +
                                   "[1, {}] ".format(self.gaussPoints()) +
                                   "in a call to chord.F and you sent " +
                                   "{}.".format(gaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._Fc[gaussPt]
            elif state == 'n' or state == 'next':
                return self._Fn[gaussPt]
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._Fp[gaussPt]
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._F0[gaussPt]
            else:
                raise RuntimeError("Error: unknown state {} ".format(state) +
                                   "in a call to chord.F.")
        else:
            raise RuntimeError("Error: unknown state {} ".format(str(state)) +
                               "in a call to chord.F.")

    # velocity gradient at a Gauss point in a specified state
    def L(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self.gaussPoints()):
            if self._gaussPts == 1:
                raise RuntimeError("gaussPt can only be 1 in a call to " +
                                   "chord.L and you sent " +
                                   "{}.".format(gaussPt))
            else:
                raise RuntimeError("gaussPt must be in the range of " +
                                   "[1, {}] ".format(self.gaussPoints()) +
                                   "in a call to chord.L and you sent " +
                                   "{}.".format(gaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                # use central difference scheme
                velGrad = ((self._Fn[gaussPt] - self._Fp[gaussPt])
                           / (2.0 * self._h * self._Fc[gaussPt]))
            elif state == 'n' or state == 'next':
                # use backward difference scheme
                velGrad = ((3.0 * self._Fn[gaussPt] - 4.0 * self._Fc[gaussPt] +
                            self._Fp[gaussPt])
                           / (2.0 * self._h * self._Fn[gaussPt]))
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use forward difference scheme
                velGrad = ((-self._Fn[gaussPt] + 4.0 * self._Fc[gaussPt] -
                            3.0 * self._Fp[gaussPt])
                           / (2.0 * self._h * self._Fp[gaussPt]))
            elif state == 'r' or state == 'ref' or state == 'reference':
                velGrad = 0.0
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was in a call to chord.L.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was in a call to chord.L.")
        return velGrad

    # fields needed to construct a finite element solution strategy

    def shapeFunction(self, gaussPt):
        if (gaussPt < 1) or (gaussPt > self.gaussPoints()):
            if self.gaussPoints() == 1:
                raise RuntimeError("gaussPt can only be 1 in a call to " +
                                   "chord.shapeFunction and you sent " +
                                   "{}.".format(gaussPt))
            else:
                raise RuntimeError("gaussPt must be in the range of " +
                                   "[1, {}] ".format(self.gaussPoints()) +
                                   "in a call to chord.shapeFunction " +
                                   "and you sent {}.".format(gaussPt))
        sf = self._shapeFns[gaussPt]
        return sf

    def gaussQuadrature(self):
        return self._gq

    def massMatrix(self):
        # use the following rule for Gauss quadrature
        gq = self.gaussQuadrature()

        # construct the consistent mass matrix in natural co-ordinates
        massC = np.zeros((2, 2), dtype=float)
        NtN = np.zeros((2, 2), dtype=float)
        for i in range(self.gaussPoints()):
            sf = self.shapeFunction(i+1)
            NtN += gq.weight(i+1) * np.matmul(np.transpose(sf.Nmtx), sf.Nmtx)
        massC[:, :] = NtN[:, :]

        # construct the lumped mass matrix in natural co-ordinates
        massL = np.zeros((2, 2), dtype=float)
        row, col = np.diag_indices_from(massC)
        massL[row, col] = massC.sum(axis=1)

        # construct the averaged mass matrix in natural co-ordinates
        massA = np.zeros((2, 2), dtype=float)
        massA = 0.5 * (massC + massL)

        # the following print statements were used to verify the code
        # print("\nThe averaged mass matrix in natural co-ordinates is")
        # print(0.5 * massA)  # the half is the Jacobian for span [-1, 1]

        # convert average mass matrix from natural to physical co-ordinates
        length = self.length("ref")
        area = self.area('ref')
        xn1 = (-length / 2,)
        xn2 = (length / 2,)
        mass = np.zeros((2, 2), dtype=float)
        Jdet = sf.jacobianDet(xn1, xn2)
        rho = self.massDensity()
        mass = (rho * area * Jdet) * massA
        return mass

    """
    def stiffnessMatrix(self, M, se, sc):
        # cross-sectional area of the chord (both collagen and elastin fibers)
        area = self._areaC + self._areaE

        # initial natural coordinates for a chord
        x01 = -self._L0 / 2.0
        x02 = self._L0 / 2.0
        xn1 = -self._Ln / 2.0
        xn2 = self._Ln / 2.0

        # creat the stress matrix
        T = se + sc

        # determine the stiffness matrix
        if self._gaussPts == 1:
            # 'natural' weight of the element
            wgt = 2.0
            w = np.array([wgt])

            # Jacobian matrix for
            J = self._shapeFns[1].jacobian(xn1, xn2)

            # create the linear Bmatrix
            BL = self._shapeFns[1].dNdximat() / J
            # the linear stiffness matrix for 1 Gauss point
            KL = area * (J * w[0] * BL.T.dot(M).dot(BL))

            # create the matrix of derivative of shape functions (H matrix)
            H = self._shapeFns[1].dNdximat() / J
            # create the matrix of derivative of displacements (A matrix)
            A = - self._shapeFns[1].G(xn1, xn2, x01, x02) / J
            # create the nonlinear Bmatrix
            BN = np.dot(A, H)
            # the nonlinear stiffness matrix for 1 Gauss point
            KN = (area * J * w[0] * (BL.T.dot(M).dot(BN) +
                  BN.T.dot(M).dot(BL) + BN.T.dot(M).dot(BN) ))

            # the stress stiffness matrix for 1 Gauss point
            KS = area * (J * w[0] * H.T.dot(T).dot(H))

            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS

        elif self._gaussPts == 2:
            # 'natural' weights of the element
            wgt = 1.0
            w = np.array([wgt, wgt])

            # at Gauss point 1
            J1 = self._shapeFns[1].jacobian(xn1, xn2)
            # create the linear Bmatrix
            BL1 = self._shapeFns[1].dNdximat() / J1

            # at Gauss point 2
            J2 = self._shapeFns[2].jacobian(xn1, xn2)
            # create the linear  Bmatrix
            BL2 = self._shapeFns[2].dNdximat() / J2

            # the linear stiffness matrix for 2 Gauss points
            KL = (area * (J1 * w[0] * BL1.T.dot(M).dot(BL1) +
                          J2 * w[1] * BL2.T.dot(M).dot(BL2)))

            # create the matrix of derivative of shape functions (H matrix)
            H1 = self._shapeFns[1].dNdximat() / J
            H2 = self._shapeFns[2].dNdximat() / J
            # create the matrix of derivative of displacements (A matrix)
            A1 = - self._shapeFns[1].G(xn1, xn2, x01, x02) / J1
            A2 = - self._shapeFns[2].G(xn1, xn2, x01, x02) / J2
            # create the nonlinear Bmatrix
            BN1 = np.dot(A1, H1)
            BN2 = np.dot(A2, H2)

            # the nonlinear stiffness matrix for 2 Gauss point
            KN = (area * (J1 * w[0] * (BL1.T.dot(M).dot(BN1) +
                         BN1.T.dot(M).dot(BL1) + BN1.T.dot(M).dot(BN1)) +
                         J2 * w[1] * (BL2.T.dot(M).dot(BN2) +
                         BN2.T.dot(M).dot(BL2) + BN2.T.dot(M).dot(BN2))))

            # the stress stiffness matrix for 1 Gauss point
            KS = (area * (J1 * w[0] * H1.T.dot(T).dot(H1) +
                          J2 * w[1] * H2.T.dot(T).dot(H2)))

            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS

        else:  # gaussPts = 3
            # 'natural' weights of the element
            wgt1 = 5.0 / 9.0
            wgt2 = 8.0 / 9.0
            w = np.array([wgt1, wgt2, wgt1])

            # at Gauss point 1
            J1 = self._shapeFns[1].jacobian(xn1, xn2)
            # create the linear Bmatrix
            BL1 = self._shapeFns[1].dNdximat() / J1

            # at Gauss point 2
            J2 = self._shapeFns[2].jacobian(xn1, xn2)
            # create the linear Bmatrix
            BL2 = self._shapeFns[2].dNdximat() / J2

            # at Gauss point 3
            J3 = self._shapeFns[3].jacobian(xn1, xn2)
            # create the linear Bmatrix
            BL3 = self._shapeFns[3].dNdximat() / J3

            # the linear stiffness matrix for 3 Gauss points
            KL = (area * (J1 * w[0] * BL1.T.dot(M).dot(BL1) +
                          J2 * w[1] * BL2.T.dot(M).dot(BL2) +
                          J3 * w[2] * BL3.T.dot(M).dot(BL3)))

            # create the matrix of derivative of shape functions (H matrix)
            H1 = self._shapeFns[1].dNdximat() / J1
            H2 = self._shapeFns[2].dNdximat() / J2
            H3 = self._shapeFns[3].dNdximat() / J3
            # create the matrix of derivative of displacements (A matrix)
            A1 = - self._shapeFns[1].G(xn1, xn2, x01, x02) / J1
            A2 = - self._shapeFns[2].G(xn1, xn2, x01, x02) / J2
            A3 = - self._shapeFns[3].G(xn1, xn2, x01, x02) / J3
            # create the nonlinear Bmatrix
            BN1 = np.dot(A1, H1)
            BN2 = np.dot(A2, H2)
            BN3 = np.dot(A3, H3)

            # the nonlinear stiffness matrix for 3 Gauss point
            KN = (area * (J1 * w[0] * (BL1.T.dot(M).dot(BN1) +
                          BN1.T.dot(M).dot(BL1) + BN1.T.dot(M).dot(BN1)) +
                          J2 * w[1] * (BL2.T.dot(M).dot(BN2) +
                          BN2.T.dot(M).dot(BL2) + BN2.T.dot(M).dot(BN2) ) +
                          J3 * w[2] * (BL3.T.dot(M).dot(BN3) +
                          BN3.T.dot(M).dot(BL3) + BN3.T.dot(M).dot(BN3))))

            # the stress stiffness matrix for 1 Gauss point
            KS = (area * (J1 * w[0] * H1.T.dot(T).dot(H1) +
                          J2 * w[1] * H2.T.dot(T).dot(H2) +
                          J3 * w[2] * H3.T.dot(T).dot(H3)))

            # determine the total tangent stiffness matrix
            stiffT = KL + KN + KS

        return stiffT

    def forcingFunction(self, se, sc):

        # create the traction
        T = sc + se
        t = T

        # initial natural coordinates for a chord
        xn1 = -self._Ln / 2.0
        xn2 = self._Ln / 2.0

        # determine the force vector
        if self._gaussPts == 1:
            # 'natural' weight of the element
            wgt = 2.0
            we = np.array([wgt])

            N1 = self._shapeFns[1].N1
            N2 = self._shapeFns[1].N2
            n = np.array([[N1, N2]])
            nMat1 = np.transpose(n)

            J = self._shapeFns[1].jacobian(xn1, xn2)

            # the force vector for 1 Gauss point
            FVec = J * we[0] * nMat1 * t

        elif self._gaussPts == 2:
            # 'natural' weights of the element
            wgt = 1.0
            we = np.array([wgt, wgt])

            # at Gauss point 1
            N1 = self._shapeFns[1].N1
            N2 = self._shapeFns[1].N2
            n1 = np.array([[N1, N2]])
            nMat1 = np.transpose(n1)

            # at Gauss point 2
            N1 = self._shapeFns[2].N1
            N2 = self._shapeFns[2].N2
            n2 = np.array([[N1, N2]])
            nMat2 = np.transpose(n2)

            J1 = self._shapeFns[1].jacobian(xn1, xn2)
            J2 = self._shapeFns[2].jacobian(xn1, xn2)

            # the force vector for 2 Gauss points
            FVec = J1 * we[0] * nMat1 * t + J2 * we[1] * nMat2 * t

        else:  # gaussPts = 3
            # 'natural' weights of the element
            wgt1 = 5.0 / 9.0
            wgt2 = 8.0 / 9.0
            we = np.array([wgt1, wgt2, wgt1])

            # at Gauss point 1
            N1 = self._shapeFns[1].N1
            N2 = self._shapeFns[1].N2
            n1 = np.array([[N1, N2]])
            nMat1 = np.transpose(n1)

            # at Gauss point 2
            N1 = self._shapeFns[2].N1
            N2 = self._shapeFns[2].N2
            n2 = np.array([[N1, N2]])
            nMat2 = np.transpose(n2)

            # at Gauss point 3
            N1 = self._shapeFns[3].N1
            N2 = self._shapeFns[3].N2
            n3 = np.array([[N1, N2]])
            nMat3 = np.transpose(n3)

            J1 = self._shapeFns[1].jacobian(xn1, xn2)
            J2 = self._shapeFns[2].jacobian(xn1, xn2)
            J3 = self._shapeFns[3].jacobian(xn1, xn2)

            # the force vector for 3 Gauss points
            FVec = (J1 * we[0] * nMat1 * t + J2 * we[1] * nMat2 * t +
                    J3 * we[2] * nMat3 * t)

        return FVec
    """


"""
Changes made in version "1.0.0":


A chord object can now be printed using the print(object) command.


The interface for the constructor was changed from

    c = chord(number, vertex1, vertex2, h, gaussPts)

to

    c = chord(number, vertex1, vertex2, h, degree)

in an effort to make the constructors of all geometric types the same.

New methods that allow for solving the constitutive equations

    E = c.tangentModulusCollagen(fiberStress, fiberStrain)

    stressRate = c.odeCollagen(time, stress)

    E = c.tangentModulusElastin(fiberStress, fiberStrain)

    stressRate = c.odeElastin(time, stress)

Other methods added

    gq = c.gaussQuadrature()

    sigma = c.stress(state)

    f = c.force(state)

Removed methods

    E1, E2, e_t = c.matPropCollagen()

    E1, E2, e_t = c.matPropElastin()

Methods c.G and c.F now return 1x1 matrices instead of floats


Changes made were not kept track of in the beta versions.
"""
