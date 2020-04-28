#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gaussQuadratures import gaussQuadChord1, gaussQuadChord3, gaussQuadChord5
import materialProperties as mp
import math as m
import numpy as np
from peceVtoX import pece
from ridder import findRoot
from shapeFnChords import shapeFunction
import spin as spinMtx
from vertices import vertex


"""
Module chords.py provides geometric information about a septal chord.

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
__date__ = "08-08-2019"
__update__ = "04-17-2020"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


Class chord in file chords.py allows for the creation of objects that are to
be used to represent chords that connect vertices in a polyhedron.  A chord is
assigned an unique number, two distinct vertices that serve as end points, the
time step size used to approximate derivatives and integrals, and the number
of Gauss points to be used for integration.

Initial coordinates that locate vertices in a dodecahedron used to model the
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


class chord

A chord object, say c, can be printed out to the command window using the
following command.  The objected printed associates with the current state.

    print(c)

constructor

    c = chord(number, vertex1, vertex2, h, degree)
        number    immutable value that is unique to this chord
        vertex1   an end point of the chord, an instance of class vertex
        vertex2   an end point of the chord, an instance of class vertex
        h         timestep size between two successive calls to 'advance'
        degree    order of polynomials integrated exactly: must be 1, 3 or 5

methods

    s = c.toString(state)
        returns a string representation for this chord in configuration 'state'

    n = c.number()
        returns the unique indexing number affiated with this chord

    v1, v2 = c.vertexNumbers()
        returns the unique numbers assigned to the two vertices of this chord

    truth = c.hasVertex(number)
        returns 'True' if one of the two vertices has this vertex number

    v = c.getVertex(number)
        returns a vertex; to be called inside, e.g., a c.hasVertex if clause

    n = c.gaussPoints()
        returns the number of Gauss points assigned to this chord

    c.update()
        assigns new coordinate values to the chord for its next location and
        updates all effected fields.  It is to be called after all vertices
        have had their coordinates updated.  This may be called multiple times
        before freezing its values with a call to 'advance'

    c.advance()
        assigns the current fields to the previous fields, and then it assigns
        the next fields to the current fields, thereby freezing the present
        next-fields in preparation for an advancment of the solution along its
        path

    Material properties that associate with this chord.  The areas are drawn
    from a random statistical distribution.

    rho = c.massDensity()
        returns the mass density of the chord (collagen and elastin fibers)

    a = c.areaCollagen(state)
        returns the cross-sectional area of the collagen fiber in
        configuration 'state'

    a = c.areaElastin(state)
        returns the cross-sectional area of the elastin fiber in
        configuration 'state'

    E = c.tangentModulusCollagen(fiberStress, fiberStrain)
        returns the elastic tangent modulus for the collagen fiber

    E = c.tangentModulusElastin(fiberStress, fiberStrain)
        returns the elastic tangent modulus for the elastin fiber

    stressRate = c.odeCollagen(time, stress)
        called by the PECE integrator for stress in the collagen fiber

    stressRate = c.odeElastin(time, stress)
        called by the PECE integrator for stress in the elastin fiber

    Geometric fields associated with a chord in 3 space are:

    a = c.area(state)
        returns the cross-sectional area of the chord, i.e., both the collagen
        and elastin fibers in configuration 'state' under the assumption that
        volume is preserved

    ell = c.length(state)
        returns the chordal length in configuration 'state'

    lambda = c.stretch(state)
        returns the chordal stretch in configuration 'state'

    Kinematic fields associated with the centroid of a chord in 3 space are:

    [x, y, z] = c.centroid(state)
        returns coordinates for the chordal mid-point in configuration 'state'

    [ux, uy, uz] = c.displacement(state)
        returns the displacement of the centroid in configuration 'state'

    [vx, vy, vz] = c.velocity(state)
        returns the velocity of the centroid in configuration 'state'

    [ax, ay, az] = c.acceleration(state)
        returns the acceleration of the centroid in configuration 'state'

    Rotation and spin of a chord wrt the dodecahedral coordinate system are:

    pMtx = c.rotation(state)
        returns a 3x3 orthogonal matrix that rotates the reference base vectors
        into the set of local base vectors pertaining to a chord whose axis
        aligns with the 1 direction, while the 2 direction passes through the
        origin of the dodecahedral reference coordinate system.  The returned
        matrix associates with configuration 'state'

    omegaMtx = c.spin(state)
        returns a 3x3 skew symmetric matrix that describes the time rate of
        change in rotation, i.e., the spin of the local chordal coordinate
        system about the fixed coordinate system of the dodecahedron.  The
        returned matrix associates with configuration 'state'

    Thermodynamic fields associated with a chord are:

    epsilon = c.strain(state)
        returns the logarithmic strain of the chord in configuration 'state'

    dEpsilon = c.dStrain(state)
        returns the logarithmic strain rate of the chord in 'state'

    sigma = c.stress(state)
        returns the stress carried by the chord in configuration 'state'

    f = c.force(state)
        returns the force carried by the chord in configuration 'state'

    The fundamental kinematic fields are:

    gMtx = c.G(gaussPt, state)
        returns the displacement gradient at the specified Gauss point for the
        specified configuration.  gMtx is scalar valued for a chord.

    fMtx = c.F(gaussPt, state)
        returns the deformation gradient at the specified Gauss point for the
        specified configuration.  fMtx is scalar valued for a chord.

    lMtx = c.L(gaussPt, state)
        returns the velocity gradient at the specified Gauss point for the
        specified configuration.  lMtx is scalar valued for a chord.

    Fields needed to construct a finite element solution strategy are:

    sf = c.shapeFunction(gaussPt):
        returns the shape function associated with the specified Gauss point.

    gq = c.gaussQuadrature()
        returns the Gauss quadrature rule being used for integration.

    mMtx = c.massMatrix()
        returns an average of the lumped and consistent mass matrices (ensures
        the mass matrix is not singular) for the chosen number of Gauss points
        for a chord whose mass density, rho, and whose cross-sectional area
        are considered to be uniform over the length of the chord.

    kMtx = c.stiffnessMatrix()
        returns a tangent stiffness matrix for the chosen number of Gauss
        points belonging to the current state.

    fVec = c.forcingFunction()
        returns a vector for the forcing function on the right-hand side
        belonging to the current state.
"""


class chord(object):

    def __init__(self, number, vertex1, vertex2, h, degree):
        self._number = int(number)

        # verify the input
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
        if h > np.finfo(float).eps:
            self._h = float(h)
        else:
            raise RuntimeError("The stepsize sent to the chord " +
                               "constructor must exceed machine precision.")
        self._tp = -self._h
        self._tc = 0.0
        self._tn = self._h

        # assign the Gauss quadrature rule to be used
        if degree == 1:
            self._gq = gaussQuadChord1
        elif degree == 3:
            self._gq = gaussQuadChord3
        elif degree == 5:
            self._gq = gaussQuadChord5
        else:
            raise RuntimeError('A Gauss quadrature capable of integrating ' +
                               'polynomials up to degree {}'.format(degree) +
                               '\nwas specified in a call to the chord ' +
                               'constructor; degree must be 1, 3 or 5.')

        # create the four rotation matrices
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

        # base vector 2: goes from the coordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1[0] + x2[0]) / 2.0
        y = (x1[1] + x2[1]) / 2.0
        z = (x1[2] + x2[2]) / 2.0
        mag = m.sqrt(x * x + y * y + z * z)
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to n1
        def getDelta(delta):
            nx = ex + delta * n1x
            ny = ey + delta * n1y
            nz = ez + delta * n1z
            # when the dot product is zero then the two vectors are orthogonal
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

        # base vector 3 is obtained through the cross product
        n3x = n1y * n2z - n1z * n2y
        n3y = n1z * n2x - n1x * n2z
        n3z = n1x * n2y - n1y * n2x

        # create the rotation matrix from dodecahedral to chordal coordinates
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
        if degree == 1:
            # this quadrature rule has a single Gauss point
            atGaussPt = 1
            sf1 = shapeFunction(self._gq.coordinates(atGaussPt))

            self._shapeFns = {
                1: sf1
            }
        elif degree == 3:
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

        # create chord gradients at their Gauss points via dictionaries
        # 'p' implies previous, 'c' implies current, 'n' implies next
        if degree == 1:
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
        elif degree == 3:
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
        else:  # degree == 5
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

        # establish the physical properties for this chord
        dia = mp.fiberDiameterCollagen()
        self._areaC = np.pi * dia**2 / 4.0
        dia = mp.fiberDiameterElastin()
        self._areaE = np.pi * dia**2 / 4.0
        self._rho = ((self._areaC * mp.rhoCollagen() +
                      self._areaE * mp.rhoElastin()) /
                     (self._areaC + self._areaE))

        # create the ODE solvers for the two fiber constitutive equations
        time0 = 0.0
        stress0 = np.array([0.0])
        self._peceC = pece(self.odeCollagen, time0, stress0, h)
        self._peceE = pece(self.odeElastin, time0, stress0, h)

        # create the fiber stresses
        self._stressCp = stress0[0]
        self._stressCc = stress0[0]
        self._stressCn = stress0[0]
        self._stressEp = stress0[0]
        self._stressEc = stress0[0]
        self._stressEn = stress0[0]

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
            raise RuntimeError("Error: unknown state {} ".format(str(state)) +
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

        # base vector 2: goes from the coordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1[0] + x2[0]) / 2.0
        y = (x1[1] + x2[1]) / 2.0
        z = (x1[2] + x2[2]) / 2.0
        mag = m.sqrt(x * x + y * y + z * z)
        ex = x / mag
        ey = y / mag
        ez = z / mag

        # an internal function is used to locate that point along the chordal
        # axis which results in a vector n2 that is normal to n1
        def getDelta(delta):
            nx = ex + delta * n1x
            ny = ey + delta * n1y
            nz = ez + delta * n1z
            # when the dot product is zero then the two vectors are orthogonal
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

        # base vector 3 is obtained through the cross product
        n3x = n1y * n2z - n1z * n2y
        n3y = n1z * n2x - n1x * n2z
        n3z = n1x * n2y - n1y * n2x

        # create the rotation matrix from dodecahedral to chordal coordinates
        self._Pn3D[0, 0] = n1x
        self._Pn3D[0, 1] = n2x
        self._Pn3D[0, 2] = n3x
        self._Pn3D[1, 0] = n1y
        self._Pn3D[1, 1] = n2y
        self._Pn3D[1, 2] = n3y
        self._Pn3D[2, 0] = n1z
        self._Pn3D[2, 1] = n2z
        self._Pn3D[2, 2] = n3z

        # chordal coordinates for the chords
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
        self._tp = self._tc
        self._tc = self._tn
        self._tn += self._h

        # advance the dependent variables of integration, i.e., the stresses
        self._stressCp = self._stressCc
        self._stressEp = self._stressEc
        self._stressCc = self._stressCn
        self._stressEc = self._stressEn

        # advance the integrators
        self._peceC.advance()
        self._peceE.advance()

    # Material properties that associate with this chord.  Except for the mass
    # density, all are drawn randomly from a statistical distribution.

    def massDensity(self):
        # returns the mass density of the chord (collagen and elastin fibers)
        return self._rho

    def areaCollagen(self, state):
        # returns the cross-sectional area of the collagen fiber assuming
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
        # returns the cross-sectional area of the elastin fiber assuming
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

    def tangentModulusCollagen(self, fiberStress, fiberStrain):
        # returns the tangent modulus of Freed & Rajagopal for elastic fibers
        E1, E2, e_t = mp.collagenFiber()
        nonLinStrain = fiberStrain - fiberStress / E2
        if nonLinStrain < e_t:
            compliance = ((e_t - nonLinStrain) /
                          (E1 * e_t + 2.0 * fiberStress) + 1.0 / E2)
            return 1 / compliance
        else:
            return E2

    def tangentModulusElastin(self, fiberStress, fiberStrain):
        # returns the tangent modulus of Freed & Rajagopal for elastic fibers
        E1, E2, e_t = mp.elastinFiber()
        nonLinStrain = fiberStrain - fiberStress / E2
        if nonLinStrain < e_t:
            compliance = ((e_t - nonLinStrain) /
                          (E1 * e_t + 2.0 * fiberStress) + 1.0 / E2)
            return 1 / compliance
        else:
            return E2

    def odeCollagen(self, time, stress):
        if (time < 0.99999 * self._tp) or (time > 1.00001 * self._tn):
            raise RuntimeError("Time is being extrapolated instead of being " +
                               "interpolated in a call to chord.odeCollagen.")
        # obtain the Lagrange interpolation weights
        wp = ((time - self._tc)*(time - self._tn) /
              ((self._tp - self._tc)*(self._tp - self._tn)))
        wc = ((time - self._tp)*(time - self._tn) /
              ((self._tc - self._tp)*(self._tc - self._tn)))
        wn = ((time - self._tp)*(time - self._tc) /
              ((self._tn - self._tp)*(self._tn - self._tc)))

        # interpolate strain and its rate
        strain = (wp * self.strain("prev") + wc * self.strain("curr") +
                  wn * self.strain("next"))
        strainRate = (wp * self.dStrain("prev") +
                      wc * self.dStrain("curr") +
                      wn * self.dStrain("next"))

        # construct the governing ODE to be integrated by the PECE method
        dStressdStrain = self.tangentModulusCollagen(stress, strain)
        stressRate = dStressdStrain * strainRate
        return stressRate

    def odeElastin(self, time, stress):
        if (time < 0.99999 * self._tp) or (time > 1.00001 * self._tn):
            raise RuntimeError("Time is being extrapolated instead of being " +
                               "interpolated in a call to chord.odeElastin.")
        # obtain the Lagrange interpolation weights
        wp = ((time - self._tc)*(time - self._tn) /
              ((self._tp - self._tc)*(self._tp - self._tn)))
        wc = ((time - self._tp)*(time - self._tn) /
              ((self._tc - self._tp)*(self._tc - self._tn)))
        wn = ((time - self._tp)*(time - self._tc) /
              ((self._tn - self._tp)*(self._tn - self._tc)))

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

    def displacement(self, state):
        x0 = self.centroid('ref')
        x = self.centroid(state)
        u = x - x0
        return u

    def velocity(self, state):
        twoH = 2.0 * self._h
        xp = self.centroid('prev')
        xc = self.centroid('curr')
        xn = self.centroid('next')
        if isinstance(state, str):
            v = np.zeros(3, dtype=float)
            if state == 'c' or state == 'curr' or state == 'current':
                # use second-order central difference formula
                v = (xn - xp) / twoH
            elif state == 'n' or state == 'next':
                # use second-order backward difference formula
                v = (3.0 * xn - 4.0 * xc + xp) / twoH
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use second-order forward difference formula
                v = (-xn + 4.0 * xc - 3.0 * xp) / twoH
            elif state == 'r' or state == 'ref' or state == 'reference':
                pass
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.velocity.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in call a to chord.velocity.")
        return v

    def acceleration(self, state):
        if isinstance(state, str):
            a = np.zeros(3, dtype=float)
            if state == 'r' or state == 'ref' or state == 'reference':
                pass
            else:
                h2 = self._h**2
                xp = self.centroid('prev')
                xc = self.centroid('curr')
                xn = self.centroid('next')
                # use second-order central differenc formula
                a = (xn - 2.0 * xc + xp) / h2
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in a call to chord.acceleration.")
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

    # fundamental kinematic fields

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

        # initial natural coordinates for a chord
        xn1 = -self.length("ref") / 2.0
        xn2 = self.length("ref") / 2.0

        # construct the consistent mass matrix assuming conservation of volume
        NtN = np.zeros((2, 2), dtype=float)
        for i in range(self.gaussPoints()):
            sf = self.shapeFunction(i+1)
            NtN += gq.weight(i+1) * np.matmul(np.transpose(sf.Nmtx), sf.Nmtx)
        Jdet = sf.jacobianDet(xn1, xn2)
        massC = np.zeros((2, 2), dtype=float)
        massC = (self.massDensity() * self.area("ref") * Jdet) * NtN

        # construct the lumped mass matrix
        massL = np.zeros((2, 2), dtype=float)
        row, col = np.diag_indices_from(massC)
        massL[row, col] = massC.sum(axis=1)

        # construct the mass matrix
        # an average of the consistent and lumped mass matrices
        mass = np.zeros((2, 2), dtype=float)
        mass = 0.5 * (massC + massL)

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
Changes made in version "1.4.0":


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


Changes made were not kept track of prior to version "1.3.0":
"""
