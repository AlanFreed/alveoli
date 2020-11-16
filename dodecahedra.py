#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import Chord
import math as m
import numpy as np
from pentagons import pentagon
from tetrahedra import tetrahedron
from vertices import Vertex

"""
Module dodecahedra.py provides geometric info about a deforming dodecahedron.

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
__update__ = "08-07-2020"
__authors__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, zamani.shahla@tamu.edu"

r"""
A listing of changes made wrt version release can be found at the end of file.


Overview of module Dodecahedra.py:


Class dodecahedron in file dodecahedra.py allows for the creation of objects
that are to be used to represent irregular dodecahedra comprised of sixty
irregular tetrahedra, twelve irregular pentagons, thirty chords of differing
lengths, and twenty vertices that connect the chord and pentagons, plus
thirteen more vertices that locate the centroids of the pentagons and the
dodecahedron.  Typically, the dodecahedron is regular in its reference
configuration, but this is not necessary.

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

Numerous methods have a string argument that is denoted as 'state' which can
take on any of the following values:
    'c', 'curr', 'current'       gets the value for a current configuration
    'n', 'next'                  gets the value for a next configuration
    'p', 'prev', 'previous'      gets the value for a previous configuration
    'r', 'ref', 'reference'      gets the value for the reference configuration

class dodecahedron

constructor

    d = dodecahedron(F0, h=1.0, alveolarDiameter=1.9524008028984345)
        F0                   far-field deformation gradient for reference shape
        h                    time seperating two successive calls to 'advance'
        alveolarDiameter     mean diameter of an alveolar sac (in cm)

    The default alveolar diameter, with F0 = I, results in vertices of the
    dodecahedron that take on coordinate values which associate with its
    natural configuration, i.e., eight of the twenty vertices take on the
    coordinates of cube whose coordinate origin resides at cube's centroid,
    with all non-centroidal vertices touching a sphere of unit radius.

    The deformation gradient 'F0' allows for an irregular dodecahedron in its
    reference configuration, with  F0 = I  producing a regular dodecahedron
    in its reference state.

methods

    s = d.vertices2string(state)
        returns a string description for the set of all vertices that describe
        a dodecahedron in its configuration 'state'

    s = d.chords2string(state)
        returns a string description for the set of all chords that describe a
        dodecahedron in its configuration 'state'

    s = d.pentagons2string(state)
        returns a string description for the set of all pentagons that describe
        a dodecahedron in its configuration 'state'

    s = d.tetrahedra2string(state)
        returns a string description for the set of all tetrahedra that
        describe the volume of a dodecahedron in its configuration 'state'

    v = d.getVertex(number)
        returns the vertex with number 'number', which must lie in [1, 33]

    c = d.getChord(number)
        returns the chord with number 'number', which must lie in [1, 30]

    p = d.getPentagon(number)
        returns the pentagon with number 'number', which must lie in [1, 12]

    t = d.getTetrahedron(number)
        returns the tetrahedron with number 'number', which must lie in [1, 60]

    d.update(nextF)
        assuming that the deformation imposed on an alveolus is homogeneous,
        described by a deformation gradient 'nextF', this procedure assigns
        new coordinate values to all of the vertices of the dodecahedron for
        this next configuration.  It calls the update method for all of its
        vertices, chords, pentagons and tetrahera, and then updates the local
        fields of the dodecahedron itself.  This method may be called multiple
        times before freezing its values with a call to the method 'advance'

        the actual deformation being imposed on the dodecahedron is the dot
        product between 'nextF' and 'F0', i.e., F = nextF.F0, as 'F0' is taken
        to describe its reference shape.  'F0' is not a shape change caused by
        imposed tractions

    d.advance()
        calls method 'advance' for all of the vertices, chords, pentagons and
        tetrahedra comprising the dodecahedron, where current fields are
        assigned to previous fields, and then next fields are assigned to
        current fields for these objects.  Afterwords, it assigns the next
        fields to the current fields for the dodecahedron, thereby freezing
        the present next-fields into their associated current fields in
        preparation for advancing the solution along its path of trajectory

    The geometric fields associated with a dodecahedron

    v = d.volume(state)
        returns the volume of this dodecahedron for configuration 'state'

    vLambda = d.volumetricStretch(state)
        returns the cube root of volume(state) divided by volume('reference')
        for this dodecahedron

    vStrain = d.volumetricStrain(state)
        returns the logarithm of volumetric stretch for this dodecahedron

    dvStrain = d.dVolumetricStrain(state)
        returns the rate of change in volumetric strain

Reference
    1) Freed, A.D., Einstein, D.R., Carson, J.P. and Jacob, R.E. “Viscoelastic
       Model for Lung Parenchyma for Multi-Scale Modeling of Respiratory System
       Phase II: Dodecahedral Micro-Model.” PNNL-21287, March, 2012.
    2) Colins, K. D. "Cayley-Menger Determinant." From MathWorld--A Wolfram Web
       Resource, created by Eric W. Weisstein. http://mathworld.wolfram.com/
       Cayley-MengerDeterminant.html
"""


class dodecahedron(object):

    def __init__(self, F0, h=1.0, alveolarDiameter=1.9524008028984345):
        # verify the inputs
        if (not isinstance(F0, np.ndarray)) or (F0.shape != (3, 3)):
            raise RuntimeError("Error: F0 sent to the dodecahedron " +
                               "constructor must be a 3x3 numpy array.")
        if h < np.finfo(float).eps:
            raise RuntimeError('The stepsize sent to the dodecahedron ' +
                               'constructor must be greater than ' +
                               'machine precision.')
        if alveolarDiameter < np.finfo(float).eps:
            raise RuntimeError('The alveolar diameter sent to the ' +
                               'dodecahedron constructor must be ' +
                               'greater than machine precision.')
        # the default diameter places the dodecahedron within an unit sphere
        # provided that F0 = I

        # projecting a dodecahedron onto a plane produces 20 triangles so
        alpha = np.pi / 20.0
        # omega is half the inside angle of a regular pentagon, i.e., 54 deg
        omega = 54.0 * np.pi / 180.0
        # phi is the golden ratio, which appears in dodecahedra geometry
        phi = (1.0 + m.sqrt(5.0)) / 2.0
        # septal chord length for dodecahedron that inscribes an unit sphere
        sqrt3 = m.sqrt(3.0)
        len0 = 2.0 / (sqrt3 * phi)
        # average projected diameter of dodecahedron that inscribes unit sphere
        dia0 = m.tan(omega) * (1.0 + m.cos(alpha)) * len0

        # determine the scale factor relating actual to reference diameters
        sf = alveolarDiameter / dia0

        # create vertices of a scaled & distorted dodecahedron via a dictionary
        v1x = sf * (F0[0, 0] + F0[0, 1] + F0[0, 2]) / sqrt3
        v1y = sf * (F0[1, 0] + F0[1, 1] + F0[1, 2]) / sqrt3
        v1z = sf * (F0[2, 0] + F0[2, 1] + F0[2, 2]) / sqrt3
        v2x = sf * (F0[0, 0] + F0[0, 1] - F0[0, 2]) / sqrt3
        v2y = sf * (F0[1, 0] + F0[1, 1] - F0[1, 2]) / sqrt3
        v2z = sf * (F0[2, 0] + F0[2, 1] - F0[2, 2]) / sqrt3
        v3x = sf * (-F0[0, 0] + F0[0, 1] - F0[0, 2]) / sqrt3
        v3y = sf * (-F0[1, 0] + F0[1, 1] - F0[1, 2]) / sqrt3
        v3z = sf * (-F0[2, 0] + F0[2, 1] - F0[2, 2]) / sqrt3
        v4x = sf * (-F0[0, 0] + F0[0, 1] + F0[0, 2]) / sqrt3
        v4y = sf * (-F0[1, 0] + F0[1, 1] + F0[1, 2]) / sqrt3
        v4z = sf * (-F0[2, 0] + F0[2, 1] + F0[2, 2]) / sqrt3
        v5x = sf * (F0[0, 0] - F0[0, 1] + F0[0, 2]) / sqrt3
        v5y = sf * (F0[1, 0] - F0[1, 1] + F0[1, 2]) / sqrt3
        v5z = sf * (F0[2, 0] - F0[2, 1] + F0[2, 2]) / sqrt3
        v6x = sf * (F0[0, 0] - F0[0, 1] - F0[0, 2]) / sqrt3
        v6y = sf * (F0[1, 0] - F0[1, 1] - F0[1, 2]) / sqrt3
        v6z = sf * (F0[2, 0] - F0[2, 1] - F0[2, 2]) / sqrt3
        v7x = sf * (-F0[0, 0] - F0[0, 1] - F0[0, 2]) / sqrt3
        v7y = sf * (-F0[1, 0] - F0[1, 1] - F0[1, 2]) / sqrt3
        v7z = sf * (-F0[2, 0] - F0[2, 1] - F0[2, 2]) / sqrt3
        v8x = sf * (-F0[0, 0] - F0[0, 1] + F0[0, 2]) / sqrt3
        v8y = sf * (-F0[1, 0] - F0[1, 1] + F0[1, 2]) / sqrt3
        v8z = sf * (-F0[2, 0] - F0[2, 1] + F0[2, 2]) / sqrt3
        v9x = sf * (0 * F0[0, 0] + phi * F0[0, 1] + F0[0, 2] / phi) / sqrt3
        v9y = sf * (0 * F0[1, 0] + phi * F0[1, 1] + F0[1, 2] / phi) / sqrt3
        v9z = sf * (0 * F0[2, 0] + phi * F0[2, 1] + F0[2, 2] / phi) / sqrt3
        v10x = sf * (0 * F0[0, 0] + phi * F0[0, 1] - F0[0, 2] / phi) / sqrt3
        v10y = sf * (0 * F0[1, 0] + phi * F0[1, 1] - F0[1, 2] / phi) / sqrt3
        v10z = sf * (0 * F0[2, 0] + phi * F0[2, 1] - F0[2, 2] / phi) / sqrt3
        v11x = sf * (phi * F0[0, 0] + F0[0, 1] / phi + 0 * F0[0, 2]) / sqrt3
        v11y = sf * (phi * F0[1, 0] + F0[1, 1] / phi + 0 * F0[1, 2]) / sqrt3
        v11z = sf * (phi * F0[2, 0] + F0[2, 1] / phi + 0 * F0[2, 2]) / sqrt3
        v12x = sf * (phi * F0[0, 0] - F0[0, 1] / phi + 0 * F0[0, 2]) / sqrt3
        v12y = sf * (phi * F0[1, 0] - F0[1, 1] / phi + 0 * F0[1, 2]) / sqrt3
        v12z = sf * (phi * F0[2, 0] - F0[2, 1] / phi + 0 * F0[2, 2]) / sqrt3
        v13x = sf * (-phi * F0[0, 0] + F0[0, 1] / phi + 0 * F0[0, 2]) / sqrt3
        v13y = sf * (-phi * F0[1, 0] + F0[1, 1] / phi + 0 * F0[1, 2]) / sqrt3
        v13z = sf * (-phi * F0[2, 0] + F0[2, 1] / phi + 0 * F0[2, 2]) / sqrt3
        v14x = sf * (-phi * F0[0, 0] - F0[0, 1] / phi + 0 * F0[0, 2]) / sqrt3
        v14y = sf * (-phi * F0[1, 0] - F0[1, 1] / phi + 0 * F0[1, 2]) / sqrt3
        v14z = sf * (-phi * F0[2, 0] - F0[2, 1] / phi + 0 * F0[2, 2]) / sqrt3
        v15x = sf * (F0[0, 0] / phi + 0 * F0[0, 1] + phi * F0[0, 2]) / sqrt3
        v15y = sf * (F0[1, 0] / phi + 0 * F0[1, 1] + phi * F0[1, 2]) / sqrt3
        v15z = sf * (F0[2, 0] / phi + 0 * F0[2, 1] + phi * F0[2, 2]) / sqrt3
        v16x = sf * (-F0[0, 0] / phi + 0 * F0[0, 1] + phi * F0[0, 2]) / sqrt3
        v16y = sf * (-F0[1, 0] / phi + 0 * F0[1, 1] + phi * F0[1, 2]) / sqrt3
        v16z = sf * (-F0[2, 0] / phi + 0 * F0[2, 1] + phi * F0[2, 2]) / sqrt3
        v17x = sf * (F0[0, 0] / phi + 0 * F0[0, 1] - phi * F0[0, 2]) / sqrt3
        v17y = sf * (F0[1, 0] / phi + 0 * F0[1, 1] - phi * F0[1, 2]) / sqrt3
        v17z = sf * (F0[2, 0] / phi + 0 * F0[2, 1] - phi * F0[2, 2]) / sqrt3
        v18x = sf * (-F0[0, 0] / phi + 0 * F0[0, 1] - phi * F0[0, 2]) / sqrt3
        v18y = sf * (-F0[1, 0] / phi + 0 * F0[1, 1] - phi * F0[1, 2]) / sqrt3
        v18z = sf * (-F0[2, 0] / phi + 0 * F0[2, 1] - phi * F0[2, 2]) / sqrt3
        v19x = sf * (0 * F0[0, 0] - phi * F0[0, 1] + F0[0, 2] / phi) / sqrt3
        v19y = sf * (0 * F0[1, 0] - phi * F0[1, 1] + F0[1, 2] / phi) / sqrt3
        v19z = sf * (0 * F0[2, 0] - phi * F0[2, 1] + F0[2, 2] / phi) / sqrt3
        v20x = sf * (0 * F0[0, 0] - phi * F0[0, 1] - F0[0, 2] / phi) / sqrt3
        v20y = sf * (0 * F0[1, 0] - phi * F0[1, 1] - F0[1, 2] / phi) / sqrt3
        v20z = sf * (0 * F0[2, 0] - phi * F0[2, 1] - F0[2, 2] / phi) / sqrt3
        # add the vertices that associate with the pentagonal centroids later
        self._vertex = {
            1: Vertex(1, (v1x, v1y, v1z), h),
            2: Vertex(2, (v2x, v2y, v2z), h),
            3: Vertex(3, (v3x, v3y, v3z), h),
            4: Vertex(4, (v4x, v4y, v4z), h),
            5: Vertex(5, (v5x, v5y, v5z), h),
            6: Vertex(6, (v6x, v6y, v6z), h),
            7: Vertex(7, (v7x, v7y, v7z), h),
            8: Vertex(8, (v8x, v8y, v8z), h),
            9: Vertex(9, (v9x, v9y, v9z), h),
            10: Vertex(10, (v10x, v10y, v10z), h),
            11: Vertex(11, (v11x, v11y, v11z), h),
            12: Vertex(12, (v12x, v12y, v12z), h),
            13: Vertex(13, (v13x, v13y, v13z), h),
            14: Vertex(14, (v14x, v14y, v14z), h),
            15: Vertex(15, (v15x, v15y, v15z), h),
            16: Vertex(16, (v16x, v16y, v16z), h),
            17: Vertex(17, (v17x, v17y, v17z), h),
            18: Vertex(18, (v18x, v18y, v18z), h),
            19: Vertex(19, (v19x, v19y, v19z), h),
            20: Vertex(20, (v20x, v20y, v20z), h),
            # set aside 13 more vertices to be created later
            21: None,
            22: None,
            23: None,
            24: None,
            25: None,
            26: None,
            27: None,
            28: None,
            29: None,
            30: None,
            31: None,
            32: None,
            33: None
        }

        # create the chords of a dodecahedron as a dictionary
        self._chord = {
            1: Chord(1, self._vertex[9], self._vertex[10], h),
            2: Chord(2, self._vertex[1], self._vertex[9], h),
            3: Chord(3, self._vertex[2], self._vertex[10], h),
            4: Chord(4, self._vertex[3], self._vertex[10], h),
            5: Chord(5, self._vertex[4], self._vertex[9], h),
            6: Chord(6, self._vertex[1], self._vertex[11], h),
            7: Chord(7, self._vertex[2], self._vertex[11], h),
            8: Chord(8, self._vertex[3], self._vertex[13], h),
            9: Chord(9, self._vertex[4], self._vertex[13], h),
            10: Chord(10, self._vertex[2], self._vertex[17], h),
            11: Chord(11, self._vertex[17], self._vertex[18], h),
            12: Chord(12, self._vertex[3], self._vertex[18], h),
            13: Chord(13, self._vertex[4], self._vertex[16], h),
            14: Chord(14, self._vertex[15], self._vertex[16], h),
            15: Chord(15, self._vertex[1], self._vertex[15], h),
            16: Chord(16, self._vertex[5], self._vertex[15], h),
            17: Chord(17, self._vertex[5], self._vertex[12], h),
            18: Chord(18, self._vertex[11], self._vertex[12], h),
            19: Chord(19, self._vertex[6], self._vertex[12], h),
            20: Chord(20, self._vertex[6], self._vertex[17], h),
            21: Chord(21, self._vertex[7], self._vertex[18], h),
            22: Chord(22, self._vertex[7], self._vertex[14], h),
            23: Chord(23, self._vertex[13], self._vertex[14], h),
            24: Chord(24, self._vertex[8], self._vertex[14], h),
            25: Chord(25, self._vertex[8], self._vertex[16], h),
            26: Chord(26, self._vertex[5], self._vertex[19], h),
            27: Chord(27, self._vertex[6], self._vertex[20], h),
            28: Chord(28, self._vertex[7], self._vertex[20], h),
            29: Chord(29, self._vertex[8], self._vertex[19], h),
            30: Chord(30, self._vertex[19], self._vertex[20], h)
        }

        # create the pentagons of a dodecahedron as a dictionary
        self._pentagon = {
            1: pentagon(1, self._chord[6], self._chord[7], self._chord[3],
                        self._chord[1], self._chord[2], h),
            2: pentagon(2, self._chord[4], self._chord[3], self._chord[10],
                        self._chord[11], self._chord[12], h),
            3: pentagon(3, self._chord[8], self._chord[9], self._chord[5],
                        self._chord[1], self._chord[4], h),
            4: pentagon(4, self._chord[2], self._chord[5], self._chord[13],
                        self._chord[14], self._chord[15], h),
            5: pentagon(5, self._chord[15], self._chord[16], self._chord[17],
                        self._chord[18], self._chord[6], h),
            6: pentagon(6, self._chord[20], self._chord[10], self._chord[7],
                        self._chord[18], self._chord[19], h),
            7: pentagon(7, self._chord[12], self._chord[21], self._chord[22],
                        self._chord[23], self._chord[8], h),
            8: pentagon(8, self._chord[25], self._chord[13], self._chord[9],
                        self._chord[23], self._chord[24], h),
            9: pentagon(9, self._chord[19], self._chord[17], self._chord[26],
                        self._chord[30], self._chord[27], h),
            10: pentagon(10, self._chord[24], self._chord[22], self._chord[28],
                         self._chord[30], self._chord[29], h),
            11: pentagon(11, self._chord[27], self._chord[28], self._chord[21],
                         self._chord[11], self._chord[20], h),
            12: pentagon(12, self._chord[29], self._chord[26], self._chord[16],
                         self._chord[14], self._chord[25], h)
        }

        # asign the remaining vertices; they are the pentagonal centroids
        # and the centroid of the dodecahedron
        c = np.array(3, dtype=float)
        c = self._pentagon[1].centroid('ref')
        self._vertex[21] = Vertex(21, (c[0], c[1], c[2]), h)
        c = self._pentagon[2].centroid('ref')
        self._vertex[22] = Vertex(22, (c[0], c[1], c[2]), h)
        c = self._pentagon[3].centroid('ref')
        self._vertex[23] = Vertex(23, (c[0], c[1], c[2]), h)
        c = self._pentagon[4].centroid('ref')
        self._vertex[24] = Vertex(24, (c[0], c[1], c[2]), h)
        c = self._pentagon[5].centroid('ref')
        self._vertex[25] = Vertex(25, (c[0], c[1], c[2]), h)
        c = self._pentagon[6].centroid('ref')
        self._vertex[26] = Vertex(26, (c[0], c[1], c[2]), h)
        c = self._pentagon[7].centroid('ref')
        self._vertex[27] = Vertex(27, (c[0], c[1], c[2]), h)
        c = self._pentagon[8].centroid('ref')
        self._vertex[28] = Vertex(28, (c[0], c[1], c[2]), h)
        c = self._pentagon[9].centroid('ref')
        self._vertex[29] = Vertex(29, (c[0], c[1], c[2]), h)
        c = self._pentagon[10].centroid('ref')
        self._vertex[30] = Vertex(30, (c[0], c[1], c[2]), h)
        c = self._pentagon[11].centroid('ref')
        self._vertex[31] = Vertex(31, (c[0], c[1], c[2]), h)
        c = self._pentagon[12].centroid('ref')
        self._vertex[32] = Vertex(32, (c[0], c[1], c[2]), h)
        self._vertex[33] = Vertex(33, (0.0, 0.0, 0.0), h)

        # create the tetrahedra that fill the volume as a dictionary
        self._tetrahedron = {
            1: tetrahedron(1, self._vertex[21], self._vertex[2],
                           self._vertex[11], self._vertex[33], h),
            2: tetrahedron(2, self._vertex[21], self._vertex[10],
                           self._vertex[2], self._vertex[33], h),
            3: tetrahedron(3, self._vertex[21], self._vertex[9],
                           self._vertex[10], self._vertex[33], h),
            4: tetrahedron(4, self._vertex[21], self._vertex[1],
                           self._vertex[9], self._vertex[33], h),
            5: tetrahedron(5, self._vertex[21], self._vertex[11],
                           self._vertex[1], self._vertex[33], h),
            6: tetrahedron(6, self._vertex[22], self._vertex[2],
                           self._vertex[10], self._vertex[33], h),
            7: tetrahedron(7, self._vertex[22], self._vertex[17],
                           self._vertex[2], self._vertex[33], h),
            8: tetrahedron(8, self._vertex[22], self._vertex[18],
                           self._vertex[17], self._vertex[33], h),
            9: tetrahedron(9, self._vertex[22], self._vertex[3],
                           self._vertex[18], self._vertex[33], h),
            10: tetrahedron(10, self._vertex[22], self._vertex[10],
                            self._vertex[3], self._vertex[33], h),
            11: tetrahedron(11, self._vertex[23], self._vertex[4],
                            self._vertex[13], self._vertex[33], h),
            12: tetrahedron(12, self._vertex[23], self._vertex[9],
                            self._vertex[4], self._vertex[33], h),
            13: tetrahedron(13, self._vertex[23], self._vertex[10],
                            self._vertex[9], self._vertex[33], h),
            14: tetrahedron(14, self._vertex[23], self._vertex[3],
                            self._vertex[10], self._vertex[33], h),
            15: tetrahedron(15, self._vertex[23], self._vertex[13],
                            self._vertex[3], self._vertex[33], h),
            16: tetrahedron(16, self._vertex[24], self._vertex[4],
                            self._vertex[9], self._vertex[33], h),
            17: tetrahedron(17, self._vertex[24], self._vertex[16],
                            self._vertex[4], self._vertex[33], h),
            18: tetrahedron(18, self._vertex[24], self._vertex[15],
                            self._vertex[16], self._vertex[33], h),
            19: tetrahedron(19, self._vertex[24], self._vertex[1],
                            self._vertex[15], self._vertex[33], h),
            20: tetrahedron(20, self._vertex[24], self._vertex[9],
                            self._vertex[1], self._vertex[33], h),
            21: tetrahedron(21, self._vertex[25], self._vertex[5],
                            self._vertex[15], self._vertex[33], h),
            22: tetrahedron(22, self._vertex[25], self._vertex[12],
                            self._vertex[5], self._vertex[33], h),
            23: tetrahedron(23, self._vertex[25], self._vertex[11],
                            self._vertex[12], self._vertex[33], h),
            24: tetrahedron(24, self._vertex[25], self._vertex[1],
                            self._vertex[11], self._vertex[33], h),
            25: tetrahedron(25, self._vertex[25], self._vertex[15],
                            self._vertex[1], self._vertex[33], h),
            26: tetrahedron(26, self._vertex[26], self._vertex[2],
                            self._vertex[17], self._vertex[33], h),
            27: tetrahedron(27, self._vertex[26], self._vertex[11],
                            self._vertex[2], self._vertex[33], h),
            28: tetrahedron(28, self._vertex[26], self._vertex[12],
                            self._vertex[11], self._vertex[33], h),
            29: tetrahedron(29, self._vertex[26], self._vertex[6],
                            self._vertex[12], self._vertex[33], h),
            30: tetrahedron(30, self._vertex[26], self._vertex[17],
                            self._vertex[6], self._vertex[33], h),
            31: tetrahedron(31, self._vertex[27], self._vertex[7],
                            self._vertex[18], self._vertex[33], h),
            32: tetrahedron(32, self._vertex[27], self._vertex[14],
                            self._vertex[7], self._vertex[33], h),
            33: tetrahedron(33, self._vertex[27], self._vertex[13],
                            self._vertex[14], self._vertex[33], h),
            34: tetrahedron(34, self._vertex[27], self._vertex[3],
                            self._vertex[13], self._vertex[33], h),
            35: tetrahedron(35, self._vertex[27], self._vertex[18],
                            self._vertex[3], self._vertex[33], h),
            36: tetrahedron(36, self._vertex[28], self._vertex[4],
                            self._vertex[16], self._vertex[33], h),
            37: tetrahedron(37, self._vertex[28], self._vertex[13],
                            self._vertex[4], self._vertex[33], h),
            38: tetrahedron(38, self._vertex[28], self._vertex[14],
                            self._vertex[13], self._vertex[33], h),
            39: tetrahedron(39, self._vertex[28], self._vertex[8],
                            self._vertex[14], self._vertex[33], h),
            40: tetrahedron(40, self._vertex[28], self._vertex[16],
                            self._vertex[8], self._vertex[33], h),
            41: tetrahedron(41, self._vertex[29], self._vertex[5],
                            self._vertex[12], self._vertex[33], h),
            42: tetrahedron(42, self._vertex[29], self._vertex[19],
                            self._vertex[5], self._vertex[33], h),
            43: tetrahedron(43, self._vertex[29], self._vertex[20],
                            self._vertex[19], self._vertex[33], h),
            44: tetrahedron(44, self._vertex[29], self._vertex[6],
                            self._vertex[20], self._vertex[33], h),
            45: tetrahedron(45, self._vertex[29], self._vertex[12],
                            self._vertex[6], self._vertex[33], h),
            46: tetrahedron(46, self._vertex[30], self._vertex[7],
                            self._vertex[14], self._vertex[33], h),
            47: tetrahedron(47, self._vertex[30], self._vertex[20],
                            self._vertex[7], self._vertex[33], h),
            48: tetrahedron(48, self._vertex[30], self._vertex[19],
                            self._vertex[20], self._vertex[33], h),
            49: tetrahedron(49, self._vertex[30], self._vertex[8],
                            self._vertex[19], self._vertex[33], h),
            50: tetrahedron(50, self._vertex[30], self._vertex[14],
                            self._vertex[8], self._vertex[33], h),
            51: tetrahedron(51, self._vertex[31], self._vertex[7],
                            self._vertex[20], self._vertex[33], h),
            52: tetrahedron(52, self._vertex[31], self._vertex[18],
                            self._vertex[7], self._vertex[33], h),
            53: tetrahedron(53, self._vertex[31], self._vertex[17],
                            self._vertex[18], self._vertex[33], h),
            54: tetrahedron(54, self._vertex[31], self._vertex[6],
                            self._vertex[17], self._vertex[33], h),
            55: tetrahedron(55, self._vertex[31], self._vertex[20],
                            self._vertex[6], self._vertex[33], h),
            56: tetrahedron(56, self._vertex[32], self._vertex[5],
                            self._vertex[19], self._vertex[33], h),
            57: tetrahedron(57, self._vertex[32], self._vertex[15],
                            self._vertex[5], self._vertex[33], h),
            58: tetrahedron(58, self._vertex[32], self._vertex[16],
                            self._vertex[15], self._vertex[33], h),
            59: tetrahedron(59, self._vertex[32], self._vertex[8],
                            self._vertex[16], self._vertex[33], h),
            60: tetrahedron(60, self._vertex[32], self._vertex[19],
                            self._vertex[8], self._vertex[33], h)
        }

        # add up the volumes associated with the sixty tetrahedra
        vol = 0.0
        for i in range(1, 61):
            vol += self._tetrahedron[i].volume('ref')
        self._refVol = vol
        self._prevVol = vol
        self._currVol = vol
        self._nextVol = vol

    def verticesToString(self, state):
        s = 'In state ' + state + ', the dodecahedron has vertices of: \n'
        for i in range(1, 21):
            s += '   ' + self._vertex[i].toString(state) + '\n'
        'The following extra vertices associate with the various centroids:\n'
        for i in range(21, 33):
            s += '   ' + self._vertex[i].toString(state) + '\n'
        s += '   ' + self._vertex[33].toString(state)
        return s

    def chordsToString(self, state):
        s = 'In state ' + state + ', the dodecahedron has chords of: \n'
        for i in range(1, 30):
            s += self._chord[i].toString(state) + '\n'
        s += self._chord[30].toString(state)
        return s

    def pentagonsToString(self, state):
        s = 'In state ' + state + ', the dodecahedron has pentagons of: \n'
        for i in range(1, 12):
            s += self._pentagon[i].toString(state) + '\n'
        s += self._pentagon[12].toString(state)
        return s

    def tetrahedraToString(self, state):
        s = 'In state ' + state + ', the dodecahedron has tetrahedra of: \n'
        for i in range(1, 60):
            s += self._tetrahedron[i].toString(state) + '\n'
        s += self._tetrahedron[60].toString(state)
        return s

    def getVertex(self, number):
        if number > 0 and number < 34:
            return self._vertex[number]
        else:
            raise RuntimeError('The requested vertex {} '.format(number) +
                               'does not exist.')

    def getChord(self, number):
        if number > 0 and number < 31:
            return self._chord[number]
        else:
            raise RuntimeError('The requested chord {} '.format(number) +
                               'does not exist.')

    def getPentagon(self, number):
        if number > 0 and number < 13:
            return self._pentagon[number]
        else:
            raise RuntimeError('The requested pentagon {} '.format(number) +
                               'does not exist.')

    def getTetrahedron(self, number):
        if number > 0 and number < 61:
            return self._tetrahedron[number]
        else:
            raise RuntimeError('The requested tetrahedron {} '.format(number) +
                               'does not exist.')

    def update(self, nextF):
        if (not isinstance(nextF, np.ndarray)) or (not nextF.shape == (3, 3)):
            raise RuntimeError("The nextF sent to dodecahedron.update " +
                               "must be a 3x3 numpy array.")

        # update the vertices for the lattice of the dodecahedron
        for i in range(1, 21):
            x0, y0, z0 = self._vertex[i].coordinates('ref')
            # deformation of the dodecahedron is taken to be homogeneous
            x = nextF[0, 0] * x0 + nextF[0, 1] * y0 + nextF[0, 2] * z0
            y = nextF[1, 0] * x0 + nextF[1, 1] * y0 + nextF[1, 2] * z0
            z = nextF[2, 0] * x0 + nextF[2, 1] * y0 + nextF[2, 2] * z0
            self._vertex[i].update((x, y, z))
        # vertices 21-33 are updated after the pentagons are updated

        # update the chords of the dodecahedron
        for i in range(1, 31):
            self._chord[i].update()

        # update the pentagons of the dodecahedron
        for i in range(1, 13):
            self._pentagon[i].update()

        # update the vertices that comprise the centroids of the dodecahedron
        c = np.array(3, dtype=float)
        for i in range(21, 33):
            c = self._pentagon[i-20].centroid('next')
            self._vertex[i].update((c[0], c[1], c[2]))
        # the 33rd vertex is at the centroid of the dodecahedron; it is fixed

        # update the tetrahedra of the dodecahedron
        for i in range(1, 61):
            self._tetrahedron[i].update()

        # update volume
        vol = 0.0
        for i in range(1, 61):
            vol += self._tetrahedron[i].volume('next')
        self._nextVol = vol

    def advance(self, reindex):
        # advance kinematic fields of the vertices
        for i in range(1, 34):
            self._vertex[i].advance()

        # advance kinematic fields of the chords
        for i in range(1, 31):
            self._chord[i].advance(reindex)

        # advance kinematic fields of the pentagons
        for i in range(1, 13):
            self._pentagon[i].advance(reindex)

        # advance kinematic fields of the tetrahedra
        for i in range(1, 61):
            self._tetrahedron[i].advance(reindex)

        # advance kinematic fields of the dodecahedron
        self._prevVol = self._currVol
        self._currVol = self._nextVol

    # properties of volume

    def volume(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._currVol
            elif state == 'n' or state == 'next':
                return self._nextVol
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._prevVol
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._refVol
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to dodecahedron.volume.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to dodecahedron.volume.")


    def V0V(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return (self._refVol / self._currVol)
            elif state == 'n' or state == 'next':
                return (self._refVol / self._nextVol)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return (self._refVol / self._prevVol)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 1.0
            else:
                raise RuntimeError("An unknown state {} in a ".format(state) +
                                   "call to dodecahedron.V0V.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to dodecahedron.V0V.")


    def volumetricStretch(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return (self._currVol / self._refVol)**(1.0 / 3.0)
            elif state == 'n' or state == 'next':
                return (self._nextVol / self._refVol)**(1.0 / 3.0)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return (self._prevVol / self._refVol)**(1.0 / 3.0)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 1.0
            else:
                raise RuntimeError("An unknown state {} in a ".format(state) +
                                   "call to dodecahedron.volumetricStretch.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to dodecahedron.volumetricStretch.")

    def volumetricStrain(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return m.log(self._currVol / self._refVol) / 3.0
            elif state == 'n' or state == 'next':
                return m.log(self._nextVol / self._refVol) / 3.0
            elif state == 'p' or state == 'prev' or state == 'previous':
                return m.log(self._prevVol / self._refVol) / 3.0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} in a ".format(state) +
                                   "call to dodecahedron.volumetricStrain.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to dodecahedron.volumetricStrain.")

    def dVolumetricStrain(self, state):
        if isinstance(state, str):
            h = 2.0 * self._h
            if state == 'c' or state == 'curr' or state == 'current':
                # use second-order central difference formula
                dVolume = (self._nextVol - self._prevVol) / h
                return (dVolume / self._currVol) / 3.0
            elif state == 'n' or state == 'next':
                # use second-order backward difference formula
                dVolume = (3.0 * self._nextVol - 4.0 * self._currVol
                           + self._prevVol) / h
                return (dVolume / self._nextVolume) / 3.0
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use second-order forward difference formula
                dVolume = (-self._nextVol + 4.0 * self._currVol
                           - 3.0 * self._prevVol) / h
                return (dVolume / self._prevVolume) / 3.0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} in a ".format(state) +
                                   "call to dodecahedron.dVolumetricStrain.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to dodecahedron.dVolumetricStrain.")


"""
Changes in version "1.0.0":

    Tetrahedra are included in the construction.  They are used to represent
    the volume of a dodecahedron and its properties.

Changes were not kept track of in the beta versions.
"""
