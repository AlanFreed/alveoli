#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
Module vertices.py provides geometric info about a septal vertex.

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
__date__ = "04-27-2019"
__update__ = "04-27-2019"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
Class vertex in file vertices.py allows for the creation of objects that are
to be used to locate a vertex in a polyhedron, specifically, an irregular
dodecahedron.  A vertex can have its coordinates reassigned but not its number.

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

procedure

s = coordinatesToString(x, y, z)
        x   the 1 coordinate
        y   the 2 coordiante
        z   the 3 coordiante

    returns a formatted string representation for assigned set of coordinates

class vertex

constructor

    v = vertex(number, x0, y0, z0, h)
        number   immutable value unique to this vertex
        x0       initial x coordinate at zero pleural pressure
        y0       initial y coordinate at zero pleural pressure
        z0       initial z coordinate at zero pleural pressure
        h        timestep size between two neighboring configurations

methods

    s = v.toString(state)
        returns string representation of this vertex for configuration 'state'

    n = v.number()
        returns unique number affiated with this vertex

    x, y, z = v.coordinates(state)
        returns the location of this vertex for configuration 'state'

    v.update(x, y, z)
        assigns new coordinate values to the vertex for its next location,
        which may be called multiple times before freezing its value with a
        call to advance

    v.advance()
        assigns the current location into the previous location, and then it
        assigns the next location into the current location, thereby freezing
        the location of the present next-location in preparation to advance the
        solution to its next place along its path of motion

    Kinematic fields associated with a point (vertex) in 3 space

    [ux, uy, uz] = v.displacement(state)
        returns the displacement of this vertex for configuration 'state'

    [vx, vy, vz] = v.velocity(state)
        returns the velocity of this vertex for configuration 'state'

    [ax, ay, az] = v.acceleration(state)
        returns the acceleration of this vertex for configuration 'state'
"""


def coordinatesToString(x, y, z):
    if x < +0.0:
        s = '[{:8.5e}'.format(x)
    else:
        s = '[ {:8.5e}'.format(x)
    if y < +0.0:
        s = s + ' {:8.5e}'.format(y)
    else:
        s = s + '  {:8.5e}'.format(y)
    if z < +0.0:
        s = s + ' {:8.5e}]'.format(z)
    else:
        s = s + '  {:8.5e}]'.format(z)
    return s


class vertex(object):

    def __init__(self, number, x0, y0, z0, h):
        if h > np.finfo(float).eps:
            self._h = float(h)
        else:
            raise RuntimeError(
                 "Error: stepsize sent to vertex constructor wasn't positive.")
        self._number = int(number)
        # reference coordinates
        self._x0 = float(x0)
        self._y0 = float(y0)
        self._z0 = float(z0)
        # coordinates of the previous step
        self._xp = self._x0
        self._yp = self._y0
        self._zp = self._z0
        # coordinates of the current step
        self._xc = self._x0
        self._yc = self._y0
        self._zc = self._z0
        # coordinates of the next step
        self._xn = self._x0
        self._yn = self._y0
        self._zn = self._z0

    def toString(self, state):
        if self._number < 10:
            s = 'vertex[0'
        else:
            s = 'vertex['
        s = s + str(self._number) + '] = '
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return s + coordinatesToString(self._xc, self._yc, self._zc)
            elif state == 'n' or state == 'next':
                return s + coordinatesToString(self._xn, self._yn, self._zn)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return s + coordinatesToString(self._xp, self._yp, self._zp)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return s + coordinatesToString(self._x0, self._y0, self._z0)
            else:
                raise RuntimeError(
                          "Error: unknown state {} in call to vertex.toString."
                          .format(state))
        else:
            raise RuntimeError(
                          "Error: unknown state {} in call to vertex.toString."
                          .format(str(state)))

    def number(self):
        return self._number

    def coordinates(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._xc, self._yc, self._zc
            elif state == 'n' or state == 'next':
                return self._xn, self._yn, self._zn
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._xp, self._yp, self._zp
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._x0, self._y0, self._z0
            else:
                raise RuntimeError(
                       "Error: unknown state {} in call to vertex.coordinates."
                       .format(state))
        else:
            raise RuntimeError(
                       "Error: unknown state {} in call to vertex.coordinates."
                       .format(str(state)))

    def update(self, x, y, z):
        self._xn = float(x)
        self._yn = float(y)
        self._zn = float(z)

    def advance(self):
        # current values are moved to previous values
        self._xp = self._xc
        self._yp = self._yc
        self._zp = self._zc
        # next values are moved to current values
        self._xc = self._xn
        self._yc = self._yn
        self._zc = self._zn

    def displacement(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                ux = self._xc - self._x0
                uy = self._yc - self._y0
                uz = self._zc - self._z0
            elif state == 'n' or state == 'next':
                ux = self._xn - self._x0
                uy = self._yn - self._y0
                uz = self._zn - self._z0
            elif state == 'p' or state == 'prev' or state == 'previous':
                ux = self._xp - self._x0
                uy = self._yp - self._y0
                uz = self._zp - self._z0
            elif state == 'r' or state == 'ref' or state == 'reference':
                ux = 0.0
                uy = 0.0
                uz = 0.0
            else:
                raise RuntimeError(
                      "Error: unknown state {} in call to vertex.displacement."
                      .format(state))
        else:
            raise RuntimeError(
                      "Error: unknown state {} in call to vertex.displacement."
                      .format(str(state)))
        return np.array([ux, uy, uz])

    def velocity(self, state):
        h = 2.0 * self._h
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                # use second-order central difference formula
                vx = (self._xn - self._xp) / h
                vy = (self._yn - self._yp) / h
                vz = (self._zn - self._zp) / h
            elif state == 'n' or state == 'next':
                # use second-order backward difference formula
                vx = (3.0 * self._xn - 4.0 * self._xc + self._xp) / h
                vy = (3.0 * self._yn - 4.0 * self._yc + self._yp) / h
                vz = (3.0 * self._zn - 4.0 * self._zc + self._zp) / h
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use second-order forward difference formula
                vx = (-self._xn + 4.0 * self._xc - 3.0 * self._xp) / h
                vy = (-self._yn + 4.0 * self._yc - 3.0 * self._yp) / h
                vz = (-self._zn + 4.0 * self._zc - 3.0 * self._zp) / h
            elif state == 'r' or state == 'ref' or state == 'reference':
                vx = 0.0
                vy = 0.0
                vz = 0.0
            else:
                raise RuntimeError(
                          "Error: unknown state {} in call to vertex.velocity."
                          .format(state))
        else:
            raise RuntimeError(
                          "Error: unknown state {} in call to vertex.velocity."
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
                # use second-order central difference formula
                ax = (self._xn - 2.0 * self._xc + self._xp) / h2
                ay = (self._yn - 2.0 * self._yc + self._yp) / h2
                az = (self._zn - 2.0 * self._zc + self._zp) / h2
        else:
            raise RuntimeError(
                      "Error: unknown state {} in call to vertex.acceleration."
                      .format(str(state)))
        return np.array([ax, ay, az])
