#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math as m
import numpy as np
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
__version__ = "1.3.0"
__date__ = "08-08-2019"
__update__ = "09-27-2019"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
Class chord in file chords.py allows for the creation of objects that are to
be used to represent chords that connect vertices in a polyhedron.  A chord is
assigned an unique number and two distinct vertices that serve as end points.

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

class chord

constructor

    c = chord(number, vertex1, vertex2, h, gaussPts)
        number    immutable value that is unique to this chord
        vertex1   an end point of the chord, an instance of class vertex
        vertex2   an end point of the chord, an instance of class vertex
        h         timestep size between two successive calls to 'advance'
        gaussPts  number of Gauss points to be used: must be 1, 2 or 3

methods

    s = c.toString(state)
        returns a string representation for this chord in configuration 'state'

    n = c.number()
        returns the unique number affiated with this chord

    v1, v2 = c.vertexNumbers()
        returns the unique numbers assigned to the two vertices of this chord

    truth = c.hasVertex(number)
        returns True if one of the two vertices has this vertex number

    v = c.getVertex(number)
        returns a vertex; to be called inside, e.g., a c.hasVertex if clause

    n = c.gaussPoints()
        returns the number of Gauss points assigned to the chord

    c.update()
        assigns new coordinate values to the chord for its next location and
        updates all effected fields.  It is to be called after all vertices
        have had their coordinates updated.  This may be called multiple times
        before freezing its values with a call to advance

    c.advance()
        assigns the current fields to the previous fields, and then it assigns
        the next fields to the current fields, thereby freezing the present
        next-fields in preparation to advance the solution along its path

    The geometric fields associated with a chord in 3 space

    ell = c.length(state)
        returns the chordal length in configuration 'state'

    lambda = c.stretch(state)
        returns the chordal stretch in configuration 'state'

    The kinematic fields associated with the centroid of a chord in 3 space

    [x, y, z] = c.centroid(state)
        returns coordinates for the chordal mid-point in configuration 'state'

    [ux, uy, uz] = c.displacement(state)
        returns the displacement at the centroid in configuration 'state'

    [vx, vy, vz] = c.velocity(state)
        returns the velocity at the centroid in configuration 'state'

    [ax, ay, az] = c.acceleration(state)
        returns the acceleration at the centroid in configuration 'state'

    The rotation and spin of a chord wrt the dodecahedral coordinate system

    pMtx = c.rotation(state)
        returns a 3x3 orthogonal matrix that rotates the reference base vectors
        into the set of local base vectors pertaining to a chord whose axis
        aligns with the 1 direction while the 2 direction passes through the
        origin of the dodecahedral reference coordinate system.  The returned
        matrix associates with configuration 'state'

    omegaMtx = c.spin(state)
        returns a 3x3 skew symmetric matrix that describes the time rate of
        rotation, i.e., spin, of the local chordal coordinate system about the
        fixed coordinate system of the dodecahedron.  The returned matrix
        associates with configuration 'state'

    The thermodynamic strain and strain-rate fields associated with a chord

    epsilon = c.strain(state)
        returns the logarithmic strain in configuration 'state'

    dEpsilon = c.dStrain(state)
        returns the logarithmic strain rate in configuration 'state'

    Fields needed to construct finite element representations

    sf = c.shapeFunction(self, gaussPt):
        returns the shape function associated with the specified Gauss point

    mMtx = c.massMatrix(rho, diameter)
        returns a consistent mass matrix for the chosen number of Gauss points
        for a chord whose mass density is rho and whose diameter is specified

    The fundamental fields of kinematics

    gMtx = c.G(gaussPt, state)
        returns the displacement gradient at the specified Gauss point for the
        specified configuration

    fMtx = c.F(gaussPt, state)
        returns the deformation gradient at the specified Gauss point for the
        specified configuration
"""


class chord(object):

    def __init__(self, number, vertex1, vertex2, h, gaussPts):
        self._number = int(number)

        # verify the input
        if not isinstance(vertex1, vertex):
            raise RuntimeError('Error: vertex1 sent to the chord ' +
                               'constructor was not of type vertex.')
        if not isinstance(vertex2, vertex):
            raise RuntimeError('Error: vertex2 sent to the chord ' +
                               'constructor was not of type vertex.')
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
            raise RuntimeError('Error: a chord must have ' +
                               'two distinct vertices.')
        if h > np.finfo(float).eps:
            self._h = float(h)
        else:
            raise RuntimeError("Error: the stepsize sent to the chord " +
                               "constructor wasn't positive.")
        # check the number of Gauss points to use
        if gaussPts == 1 or gaussPts == 2 or gaussPts == 3:
            self._gaussPts = gaussPts
        else:
            raise RuntimeError('Error: {} Gauss points specified in ' +
                               'the chord constructor must be 1, 2 or 3.'
                               .format(gaussPts))

        # create the three rotation matrices
        self._Pr3D = np.identity(3, dtype=float)
        self._Pp3D = np.identity(3, dtype=float)
        self._Pc3D = np.identity(3, dtype=float)
        self._Pn3D = np.identity(3, dtype=float)

        # initialize the chordal lengths for all configurations
        x1, y1, z1 = self._vertex[1].coordinates('r')
        x2, y2, z2 = self._vertex[2].coordinates('r')
        L0 = m.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        self._L0 = L0
        self._Lp = L0
        self._Lc = L0
        self._Ln = L0

        # initialize the centroids for all configurations
        self._centroidX0 = (x1 + x2) / 2.0
        self._centroidY0 = (y1 + y2) / 2.0
        self._centroidZ0 = (z1 + z2) / 2.0
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
        x = x2 - x1
        y = y2 - y1
        z = z2 - z1
        mag = m.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 2: goes from the coordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        z = (z1 + z2) / 2.0
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
        for i in range(3):
            for j in range(3):
                self._Pp3D[i, j] = self._Pr3D[i, j]
                self._Pc3D[i, j] = self._Pr3D[i, j]
                self._Pn3D[i, j] = self._Pr3D[i, j]

        # establish the shape functions located at the various Gauss points
        if gaussPts == 1:
            # this single Gauss point has a weight of 2
            xi = 0.0
            sf1 = shapeFunction(xi)

            self._shapeFns = {
                1: sf1
            }
        elif gaussPts == 2:
            # each of these two Gauss points has a weight of 1
            xi1 = -0.577350269189626
            sf1 = shapeFunction(xi1)

            xi2 = 0.577350269189626
            sf2 = shapeFunction(xi2)

            self._shapeFns = {
                1: sf1,
                2: sf2
            }
        else:  # gaussPts = 3
            # Gauss points 1 & 3 have weights of 5/9
            xi1 = -0.774596669241
            sf1 = shapeFunction(xi1)

            # Gauss point 2 (the centroid) has a weight of 8/9
            xi2 = 0.0
            sf2 = shapeFunction(xi2)

            xi3 = 0.774596669241
            sf3 = shapeFunction(xi3)

            self._shapeFns = {
                1: sf1,
                2: sf2,
                3: sf3
            }

        # create chord gradients at their Gauss points via dictionaries
        # p implies previous, c implies current, n implies next
        if gaussPts == 1:
            # displacement gradients located at the Gauss points of a chord
            self._G0 = {
                1: 0.0
            }
            self._Gp = {
                1: 0.0
            }
            self._Gc = {
                1: 0.0
            }
            self._Gn = {
                1: 0.0
            }
            # deformation gradients located at the Gauss points of a chord
            self._F0 = {
                1: 1.0
            }
            self._Fp = {
                1: 1.0
            }
            self._Fc = {
                1: 1.0
            }
            self._Fn = {
                1: 1.0
            }
        elif gaussPts == 2:
            # displacement gradients located at the Gauss points of a chord
            self._G0 = {
                1: 0.0,
                2: 0.0
            }
            self._Gp = {
                1: 0.0,
                2: 0.0
            }
            self._Gc = {
                1: 0.0,
                2: 0.0
            }
            self._Gn = {
                1: 0.0,
                2: 0.0
            }
            # deformation gradients located at the Gauss points of a chord
            self._F0 = {
                1: 1.0,
                2: 1.0
            }
            self._Fp = {
                1: 1.0,
                2: 1.0
            }
            self._Fc = {
                1: 1.0,
                2: 1.0
            }
            self._Fn = {
                1: 1.0,
                2: 1.0
            }
        else:  # gaussPts = 3
            # displacement gradients located at the Gauss points of a chord
            self._G0 = {
                1: 0.0,
                2: 0.0,
                3: 0.0
            }
            self._Gp = {
                1: 0.0,
                2: 0.0,
                3: 0.0
            }
            self._Gc = {
                1: 0.0,
                2: 0.0,
                3: 0.0
            }
            self._Gn = {
                1: 0.0,
                2: 0.0,
                3: 0.0
            }
            # deformation gradients located at the Gauss points of a chord
            self._F0 = {
                1: 1.0,
                2: 1.0,
                3: 1.0
            }
            self._Fp = {
                1: 1.0,
                2: 1.0,
                3: 1.0
            }
            self._Fc = {
                1: 1.0,
                2: 1.0,
                3: 1.0
            }
            self._Fn = {
                1: 1.0,
                2: 1.0,
                3: 1.0
            }

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
            raise RuntimeError(
                           "Error: unknown state {} in call to chord.toString."
                           .format(str(state)))
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
            raise RuntimeError('Error: vertex {} does not belong to chord {}.'
                               .format(number, self._number))

    def gaussPoints(self):
        return self._gaussPts

    def update(self):
        # determine length of the chord in the next configuration
        x1, y1, z1 = self._vertex[1].coordinates('n')
        x2, y2, z2 = self._vertex[2].coordinates('n')
        self._Ln = m.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

        # determine the centroid of this chord
        self._centroidXn = (x1 + x2) / 2.0
        self._centroidYn = (y1 + y2) / 2.0
        self._centroidZn = (z1 + z2) / 2.0

        # base vector 1: aligns with the axis of the chord
        x = x2 - x1
        y = y2 - y1
        z = z2 - z1
        mag = m.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 2: goes from the coordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        z = (z1 + z2) / 2.0
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
        x10 = -self._L0 / 2.0
        x20 = self._L0 / 2.0
        x1 = -self._Ln / 2.0
        x2 = self._Ln / 2.0

        # establish the deformation and displacement gradients as dictionaries
        if self._gaussPts == 1:
            # displacement gradients located at the Gauss points of the chord
            self._Gn[1] = self._shapeFns[1].G(x1, x2, x10, x20)
            # deformation gradients located at the Gauss points of the chord
            self._Fn[1] = self._shapeFns[1].F(x1, x2, x10, x20)
        elif self._gaussPts == 2:
            # displacement gradients located at the Gauss points of a chord
            self._Gn[1] = self._shapeFns[1].G(x1, x2, x10, x20)
            self._Gn[2] = self._shapeFns[2].G(x1, x2, x10, x20)
            # deformation gradients located at the Gauss points of a chord
            self._Fn[1] = self._shapeFns[1].F(x1, x2, x10, x20)
            self._Fn[2] = self._shapeFns[2].F(x1, x2, x10, x20)
        else:  # gaussPts = 3
            # displacement gradients located at the Gauss points of a chord
            self._Gn[1] = self._shapeFns[1].G(x1, x2, x10, x20)
            self._Gn[2] = self._shapeFns[2].G(x1, x2, x10, x20)
            self._Gn[3] = self._shapeFns[3].G(x1, x2, x10, x20)
            # deformation gradients located at the Gauss points of a chord
            self._Fn[1] = self._shapeFns[1].F(x1, x2, x10, x20)
            self._Fn[2] = self._shapeFns[2].F(x1, x2, x10, x20)
            self._Fn[3] = self._shapeFns[3].F(x1, x2, x10, x20)

        return  # nothing

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
        for i in range(1, self._gaussPts+1):
            self._Fp[i] = self._Fc[i]
            self._Fc[i] = self._Fn[i]
            self._Gp[i] = self._Gc[i]
            self._Gc[i] = self._Gn[i]

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
                raise RuntimeError("Error: unknown state {} ".format(state) +
                                   "in call to chord.length.")
        else:
            raise RuntimeError("Error: unknown state {} ".format(str(state)) +
                               "in call to chord.length.")

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
                raise RuntimeError("Error: unknown state {} ".format(state) +
                                   "in call to chord.stretch.")
        else:
            raise RuntimeError("Error: unknown state {} ".format(str(state)) +
                               "in call to chord.stretch.")

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
                raise RuntimeError("Error: unknown state {} ".format(state) +
                                   "in call to chord.centroid.")
        else:
            raise RuntimeError("Error: unknown state {} ".format(str(state)) +
                               "in call to chord.centroid.")
        return np.array([cx, cy, cz])

    def displacement(self, state):
        x0, y0, z0 = self.centroid('ref')
        x, y, z = self.centroid(state)
        return np.array([x - x0, y - y0, z - z0])

    def velocity(self, state):
        h = 2.0 * self._h
        xp, yp, zp = self.centroid('prev')
        xc, yc, zc = self.centroid('curr')
        xn, yn, zn = self.centroid('next')
        if isinstance(state, str):
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
                raise RuntimeError("Error: unknown state {} ".format(state) +
                                   "in call to chord.velocity.")
        else:
            raise RuntimeError("Error: unknown state {} ".format(str(state)) +
                               "in call to chord.velocity.")
        return np.array([vx, vy, vz])

    def acceleration(self, state):
        if isinstance(state, str):
            if state == 'r' or state == 'ref' or state == 'reference':
                ax = 0.0
                ay = 0.0
                az = 0.0
            else:
                h2 = self._h**2
                xp, yp, zp = self.centroid('p')
                xc, yc, zc = self.centroid('c')
                xn, yn, zn = self.centroid('n')
                # use second-order central differenc formula
                ax = (xn - 2.0 * xc + xp) / h2
                ay = (yn - 2.0 * yc + yp) / h2
                az = (zn - 2.0 * zc + zp) / h2
        else:
            raise RuntimeError("Error: unknown state {} ".format(str(state)) +
                               "in call to chord.acceleration.")
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
                raise RuntimeError("Error: unknown state {} ".format(state) +
                                   "in call to chord.rotation.")
        else:
            raise RuntimeError("Error: unknown state {} ".format(str(state)) +
                               "in call to chord.rotation.")

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
                raise RuntimeError("Error: unknown state {} ".format(state) +
                                   "in call to chord.spin.")
        else:
            raise RuntimeError("Error: unknown state {} in call to chord.spin."
                               .format(str(state)))

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
                raise RuntimeError("Error: unknown state {} ".format(state) +
                                   "in call to chord.strain.")
        else:
            raise RuntimeError("Error: unknown state {} ".format(str(state)) +
                               "in call to chord.strain.")

    def dStrain(self, state):
        if isinstance(state, str):
            lamP = self._Lp / self._L0
            lamC = self._Lc / self._L0
            lamN = self._Ln / self._L0
            if state == 'c' or state == 'curr' or state == 'current':
                return (lamN - lamP) / (self._h * lamC)
            elif state == 'n' or state == 'next':
                return (3.0 * lamN - 4.0 * lamC + lamP) / (self._h * lamN)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return (-lamN + 4.0 * lamC - 3.0 * lamP) / (self._h * lamP)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("Error: unknown state {} ".format(state) +
                                   "in call to chord.dStrain.")
        else:
            raise RuntimeError("Error: unknown state {} ".format(str(state)) +
                               "in call to chord.dStrain.")

    def shapeFunction(self, gaussPt):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "chord.shapeFunction and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to chord.shapeFunction and you sent {}."
                                   .format(gaussPt))
            sf = self._shapeFns[gaussPt]
        return sf

    def massMatrix(self, rho, diameter):
        if rho <= 0.0:
            raise RuntimeError("Mass density rho must be positive, you " +
                               "sent {} to chord.massMatrix.".format(rho))
        if diameter <= 0.0:
            raise RuntimeError("The diameter sent to chord.massMatrix " +
                               "{}; it must be positive.".format(diameter))

        # initial natural coordinates for a chord
        x01 = -self._L0 / 2.0
        x02 = -self._L0 / 2.0

        # determine the mass matrix
        if self._gaussPts == 1:
            # 'natural' weight of the element
            wgt = 2.0
            wel = np.array([wgt])

            nn1 = np.dot(np.transpose(self._shapeFns[1].Nmatx),
                         self._shapeFns[1].Nmatx)

            detJ = self._shapeFns[1].detJacobian(x01, x02)

            # Integration to get the mass matrix for 1 Gauss point
            mass = rho * diameter * (detJ * wel[0] * nn1)
        elif self._gaussPts == 2:
            # 'natural' weights of the element
            wgt = 1.0
            wel = np.array([wgt, wgt])

            nn1 = np.dot(np.transpose(self._shapeFns[1].Nmatx),
                         self._shapeFns[1].Nmatx)
            nn2 = np.dot(np.transpose(self._shapeFns[2].Nmatx),
                         self._shapeFns[2].Nmatx)

            detJ1 = self._shapeFns[1].detJacobian(x01, x02)
            detJ2 = self._shapeFns[2].detJacobian(x01, x02)

            # Integration to get the mass matrix for 2 Gauss points
            mass = rho * diameter * (detJ1 * wel[0] * nn1 +
                                     detJ2 * wel[1] * nn2)
        else:  # gaussPts = 3
            # 'natural' weights of the element
            wgt1 = 5.0 / 9.0
            wgt2 = 8.0 / 9.0
            wel = np.array([wgt1, wgt2, wgt1])

            nn1 = np.dot(np.transpose(self._shapeFns[1].Nmatx),
                         self._shapeFns[1].Nmatx)
            nn2 = np.dot(np.transpose(self._shapeFns[2].Nmatx),
                         self._shapeFns[2].Nmatx)
            nn3 = np.dot(np.transpose(self._shapeFns[3].Nmatx),
                         self._shapeFns[3].Nmatx)

            detJ1 = self._shapeFns[1].detJacobian(x01, x02)
            detJ2 = self._shapeFns[2].detJacobian(x01, x02)
            detJ3 = self._shapeFns[3].detJacobian(x01, x02)

            # Integration to get the mass Matrix for 3 Gauss points
            mass = rho * diameter * (detJ1 * wel[0] * nn1 +
                                     detJ2 * wel[1] * nn2 +
                                     detJ3 * wel[2] * nn3)
        return mass

    # displacement gradient at a Gauss point
    def G(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "chord.G and you sent {}.".format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPts) +
                                   "to chord.G and you sent {}."
                                   .format(gaussPt))
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
                raise RuntimeError("Error: unknown state {} in call a to " +
                                   "chord.G.".format(state))
        else:
            raise RuntimeError("Error: unknown state {} in a call to " +
                               "chord.G.".format(str(state)))

    # deformation gradient at a Gauss point
    def F(self, gaussPt, state):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            if self._gaussPts == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in a call " +
                                   "to chord.F and you sent {}."
                                   .format(gaussPt))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in a "
                                   .format(self._gaussPts) +
                                   "call to chord.F and you sent {}."
                                   .format(gaussPt))
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
                raise RuntimeError("Error: unknown state {} in a call to " +
                                   "chord.F.".format(state))
        else:
            raise RuntimeError("Error: unknown state {} in a call to " +
                               "chord.F.".format(str(state)))
