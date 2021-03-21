#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pivotIncomingF import Pivot

"""
Module vertices.py provides an object to manage its geometric information.

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
__date__ = "04-27-2019"
__update__ = "07-17-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


Class Vertex in file vertices.py allows for the creation of objects that are
to be used to locate a vertex in a polyhedron, in our case, an irregular
dodecahedron.  Vertices can have co-ordinates reassigned, but not their number.

Initial co-ordinates that locate a vertex in a dodecahedron used to model the
alveoli of lung are assigned according to a reference configuration where the
pleural pressure (the pressure surrounding lung in the pleural cavity) and the
transpulmonary pressure (the difference between aleolar and pleural pressures)
are both at zero gauge pressure, i.e., all pressures are atmospheric pressure.
The pleural pressure is normally negative, sucking the pleural membrane against
the wall of the chest.  During expiration, the diaphragm is pushed up, reducing
the pleural pressure.  Pleural pressure remains negative during breathing at
rest, but it can become positive during active expiration.  The surface tension
created by surfactant keeps most alveoli open during excursions into positive
pleural pressures, but not all will remain open.  Alveoli are their smallest at
max expiration.  Alveolar size is determined by the transpulmonary pressure.
The greater the transpulmonary pressure the greater the alveolar size will be.

Numerous methods have a string argument that is denoted as  'state'  which can
take on any of the following values:
    'c', 'curr', 'current'       gets the value for a current configuration
    'n', 'next'                  gets the value for a next configuration
    'p', 'prev', 'previous'      gets the value for a previous configuration
    'r', 'ref', 'reference'      gets the value for the reference configuration

Co-ordinates are handled as tuples; vector fields are handled as arrays.

procedures

s = coordinatesToString(coordinates)
    coordinates = (x, y, z) is a tuple with values
        x   the 1 co-ordinate
        y   the 2 co-ordinate
        z   the 3 co-ordinate
    returns formatted string representation for the assigned co-ordinates

s = vectorToString(array)
    returns formatted string representation for the assigned vector

s = matrixToString(matrix)
    returns formatted string representation for the assigned matrix


class Vertex

A Vertex object, say v, can be printed out to the command window using the
following command.  The object printed associates with the current state.

    print(v)

constructor

    v = Vertex(number, coordinates, dTime)
        number  immutable value unique to this vertex
        coordinates = (x0, y0, z0) is a tuple with values
            x0  initial x co-ordinate at zero pleural pressure
            y0  initial y co-ordinate at zero pleural pressure
            z0  initial z co-ordinate at zero pleural pressure
        dTime   is the time seperating any two neighboring configurations

methods

    s = v.toString(state)
        returns string representation of this vertex for configuration 'state'

    n = v.number()
        returns the unique number affiliated with this vertex

    coordinates = v.coordinates(state)
        coordinates = (x, y, z) is a tuple with values
            x   next value assigned to the x co-ordinate for the vertex
            y   next value assigned to the y co-ordinate for the vertex
            z   next value assigned to the z co-ordinate for the vertex
        These co-ordinates locate the vertex in a dodecahedral frame of
        reference (E_1, E_2, E_3) for configuration 'state'

    v.update(coordinates)
        coordinates = (x, y, z) is a tuple with values
            x   next value assigned to the x co-ordinate for the vertex
            y   next value assigned to the y co-ordinate for the vertex
            z   next value assigned to the z co-ordinate for the vertex
        v.update may be called multiple times before freezing its co-ordinate
        value with a call to v.advance

    v.advance()
        assigns the current location into the previous location, and then it
        assigns the next location into the current location, thereby freezing
        the location of the present next-location in preparation for advancing
        a solution to its next place along its path of motion

    Kinematic vector fields associated with a point (vertex) in 3 space.
    Because these are constructed from difference formulae, it is necessary
    that they be rotated into the re-indexed co-ordinate frame for the 'state'
    of interest

    [ux, uy, uz] = v.displacement(reindex, state)
        reindex     is an instance of Pivot object from module pivotIncomingF
        returns the displacement of this vertex for configuration 'state'

    [vx, vy, vz] = v.velocity(reindex, state)
        reindex     is an instance of Pivot object from module pivotIncomingF
        returns the velocity of this vertex for configuration 'state'

    [ax, ay, az] = v.acceleration(reindex, state)
        reindex     is an instance of Pivot object from module pivotIncomingF
        returns the acceleration of this vertex for configuration 'state'
"""


def coordinatesToString(coordinates):
    # verify the input
    if isinstance(coordinates, tuple):
        x = float(coordinates[0])
        y = float(coordinates[1])
        z = float(coordinates[2])
    else:
        raise RuntimeError("Coordinates sent to coordinatesToString " +
                           "must be a tuple, e.g., (x, y, z).")
    if x < +0.0:
        s = '({:7.4e}'.format(x)
    else:
        s = '( {:7.4e}'.format(x)
    if y < +0.0:
        s = s + ' {:7.4e}'.format(y)
    else:
        s = s + '  {:7.4e}'.format(y)
    if z < +0.0:
        s = s + ' {:7.4e})'.format(z)
    else:
        s = s + '  {:7.4e})'.format(z)
    return s


def vectorToString(vector):
    # verify the input
    if not isinstance(vector, np.ndarray):
        raise RuntimeError("The vector sent to vectorToString " +
                           "was not a numpy array.")
    (length,) = np.shape(vector)
    x = float(vector[0])
    if x < +0.0:
        s = '[{:7.4e}'.format(x)
    else:
        s = '[ {:7.4e}'.format(x)
    for i in range(1, length):
        x = float(vector[i])
        if x < +0.0:
            s += ' {:7.4e}'.format(x)
        else:
            s += '  {:7.4e}'.format(x)
    s += ']'
    return s


def matrixToString(matrix):
    if not isinstance(matrix, np.ndarray):
        raise RuntimeError("The matrix sent to matrixToString " +
                           "was not a numpy array.")
    (rows, cols) = np.shape(matrix)
    for i in range(rows):
        if i == 0:
            s = '[['
        else:
            s += '\n ['
        x = float(matrix[i, 0])
        if x < +0.0:
            s += '{:7.4e}'.format(x)
        else:
            s += ' {:7.4e}'.format(x)
        for j in range(1, cols):
            x = float(matrix[i, j])
            if x < +0.0:
                s += ' {:7.4e}'.format(x)
            else:
                s += '  {:7.4e}'.format(x)
        s += ']'
    s += ']'
    return s


class Vertex(object):

    def __init__(self, number, coordinates, dTime):
        # verify the input
        if isinstance(coordinates, tuple):
            # assign the reference co-ordinates
            self._x0 = float(coordinates[0])
            self._y0 = float(coordinates[1])
            self._z0 = float(coordinates[2])
        else:
            raise RuntimeError("Co-ordinates sent to the vertex constructor " +
                               "must be a tuple, e.g., (x0, y0, z0).")

        if dTime > np.finfo(float).eps:
            self._h = float(dTime)
        else:
            raise RuntimeError("The stepsize sent to the vertex constructor " +
                               "wasn't positive.")

        self._number = int(number)

        # initialize co-ordinates for the previous step
        self._xp = self._x0
        self._yp = self._y0
        self._zp = self._z0
        # initialize co-ordinates for the current step
        self._xc = self._x0
        self._yc = self._y0
        self._zc = self._z0
        # initialize co-ordinates for the next step
        self._xn = self._x0
        self._yn = self._y0
        self._zn = self._z0

        return  # new vertex object

    def __str__(self):
        return self.toString('curr')

    def toString(self, state):
        if self._number < 10:
            s = 'vertex[0'
        else:
            s = 'vertex['
        s = s + str(self._number) + '] = '
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return s + coordinatesToString((self._xc, self._yc, self._zc))
            elif state == 'n' or state == 'next':
                return s + coordinatesToString((self._xn, self._yn, self._zn))
            elif state == 'p' or state == 'prev' or state == 'previous':
                return s + coordinatesToString((self._xp, self._yp, self._zp))
            elif state == 'r' or state == 'ref' or state == 'reference':
                return s + coordinatesToString((self._x0, self._y0, self._z0))
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   " in a call tovertex.toString.")
        else:
            raise RuntimeError("Unknown state {} in a call to vertex.toString."
                               .format(str(state)))

    def number(self):
        return self._number

    def coordinates(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return (self._xc, self._yc, self._zc)
            elif state == 'n' or state == 'next':
                return (self._xn, self._yn, self._zn)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return (self._xp, self._yp, self._zp)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return (self._x0, self._y0, self._z0)
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   "in a call to vertex.coordinates.")
        else:
            raise RuntimeError("Unknown state {} ".format(str(state)) +
                               "in call to vertex.coordinates.")

    def update(self, coordinates):
        # verify the input
        if isinstance(coordinates, tuple):
            self._xn = float(coordinates[0])
            self._yn = float(coordinates[1])
            self._zn = float(coordinates[2])
        else:
            raise RuntimeError("Co-ordinates sent to vertex.update " +
                               "must be a tuple, e.g., (x, y, z).")

    def advance(self):
        # current values are moved to previous values
        self._xp = self._xc
        self._yp = self._yc
        self._zp = self._zc
        # next values are moved to current values
        self._xc = self._xn
        self._yc = self._yn
        self._zc = self._zn

    def displacement(self, reindex, state):
        # verify the input
        if not isinstance(reindex, Pivot):
            raise RuntimeError("The 'reindex' variable sent to " +
                               "vertex.displacement must be of type Pivot.")
        # calculate the displacement in the specified configuration
        u = np.zeros(3, dtype=float)
        x0 = np.zeros(3, dtype=float)
        xRef = np.array([self._x0, self._y0, self._z0])
        fromCase = reindex.pivotCase('ref')
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                x = np.array([self._xc, self._yc, self._zc])
                toCase = reindex.pivotCase('curr')
                x0 = reindex.reindexVector(xRef, fromCase, toCase)
            elif state == 'n' or state == 'next':
                x = np.array([self._xn, self._yn, self._zn])
                toCase = reindex.pivotCase('next')
                x0 = reindex.reindexVector(xRef, fromCase, toCase)
            elif state == 'p' or state == 'prev' or state == 'previous':
                x = np.array([self._xp, self._yp, self._zp])
                toCase = reindex.pivotCase('prev')
                x0 = reindex.reindexVector(xRef, fromCase, toCase)
            elif state == 'r' or state == 'ref' or state == 'reference':
                x = np.array([self._x0, self._y0, self._z0])
                x0 = np.array([self._x0, self._y0, self._z0])
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   "in a call to vertex.displacement.")
        else:
            raise RuntimeError("Unknown state {} ".format(str(state)) +
                               "in a call to vertex.displacement.")
        u = x - x0
        return u

    def velocity(self, reindex, state):
        # verify the input
        if not isinstance(reindex, Pivot):
            raise RuntimeError("The 'reindex' variable sent to " +
                               "vertex.velocity must be of type Pivot.")
        # calculate the velocity in the specified configuration
        h = 2.0 * self._h
        v = np.zeros(3, dtype=float)
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                toCase = reindex.pivotCase('curr')
                # map vectors into co-ordinate system of current configuration
                xPrev = np.array([self._xp, self._yp, self._zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xNext = np.array([self._xn, self._yn, self._zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
                # use second-order central difference formula
                v = (xN - xP) / h
            elif state == 'n' or state == 'next':
                toCase = reindex.pivotCase('next')
                # map vectors into co-ordinate system of next configuration
                xPrev = np.array([self._xp, self._yp, self._zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xCurr = np.array([self._xc, self._yc, self._zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xN = np.array([self._xn, self._yn, self._zn])
                # use second-order backward difference formula
                v = (3.0 * xN - 4.0 * xC + xP) / h
            elif state == 'p' or state == 'prev' or state == 'previous':
                toCase = reindex.pivotCase('prev')
                # map vector into co-ordinate system of previous configuration
                xP = np.array([self._xp, self._yp, self._zp])
                xCurr = np.array([self._xc, self._yc, self._zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xNext = np.array([self._xn, self._yn, self._zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
                # use second-order forward difference formula
                v = (-xN + 4.0 * xC - 3.0 * xP) / h
            elif state == 'r' or state == 'ref' or state == 'reference':
                # velocity is zero
                pass
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   "in a call to vertex.velocity.")
        else:
            raise RuntimeError("Unknown state {} ".format(str(state)) +
                               "in a call to vertex.velocity.")
        return v

    def acceleration(self, reindex, state):
        # verify the input
        if not isinstance(reindex, Pivot):
            raise RuntimeError("The 'reindex' variable sent to " +
                               "vertex.acceleration must be of type Pivot.")
        # calculate the acceleration in the specified configuration
        h2 = self._h**2
        a = np.zeros(3, dtype=float)
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                toCase = reindex.pivotCase('curr')
                # map vectors into co-ordinate system of current configuration
                xPrev = np.array([self._xp, self._yp, self._zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xC = np.array([self._xc, self._yc, self._zc])
                xNext = np.array([self._xn, self._yn, self._zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
            elif state == 'n' or state == 'next':
                toCase = reindex.pivotCase('next')
                # map vectors into co-ordinate system of next configuration
                xPrev = np.array([self._xp, self._yp, self._zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xCurr = np.array([self._xc, self._yc, self._zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xN = np.array([self._xn, self._yn, self._zn])
            elif state == 'p' or state == 'prev' or state == 'previous':
                toCase = reindex.pivotCase('prev')
                # map vector into co-ordinate system of previous configuration
                xP = np.array([self._xp, self._yp, self._zp])
                xCurr = np.array([self._xc, self._yc, self._zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xNext = np.array([self._xn, self._yn, self._zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
            elif state == 'r' or state == 'ref' or state == 'reference':
                # acceleration is zero
                pass
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   "in a call to vertex.acceleration.")
        else:
            raise RuntimeError("Unknown state {} ".format(str(state)) +
                               "in a call to vertex.acceleration.")
        a = (xN - 2.0 * xC + xP) / h2
        return a


"""
Changes made in version "1.0.0":

Added procedures
    s = vectorToString(vector)
    s = matrixToString(matrix)

All co-ordinates  are now handled as tuples of three floating point numbers.
All vector fields are now handled as arrays of three floating point numbers.

A vertex object can now be printed using the print(object) command.

The vertex methods that return vector fields now account for the possibility
of a re-indexing in the co-ordinate system; specifically
    [ux, uy, uz] = v.displacement(reindex, state)
    [vx, vy, vz] = v.velocity(reindex, state)
    [ax, ay, az] = v.acceleration(reindex, state)
all require a reindex variable, which is an object of type pivot.

Changes made were not kept track of in the beta versions.
"""
