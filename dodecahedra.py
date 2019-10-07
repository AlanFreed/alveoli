#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import chord
import math as m
import numpy as np
from pentagons import pentagon
from tetrahedra import tetrahedron
from vertices import vertex

"""
Module dodecahedra.py provides geometric info about a deforming dodecahedron.

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
__update__ = "09-30-2019"
__authors__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, zamani.shahla@tamu.edu"

r"""

Changes in version "1.3.0":

    Tetrahedra are included in the construction.  They are used to represent
    the volume of a dodecahedron.

methods added

    detjac = d.detJacobianTet(gaussPt, state)
        gaussPt  the  Gauss point  for which the Jacobian associates
        state    the configuration for which the Jacobian associates
    returns
        detjac   the determinant of the Jacobian matrix in tetrahedron

    massM = p.massMatrixTet(gaussPt, rho)
        gaussPt  the Gauss point for which the mass matrix is to be supplied
        rho      the mass density
    returns
        massM    a 12x12 mass matrix for the Tetrahedron associated with
                 Gauss point 'gaussPt'


Overview of module Dodecahedra.py:


Class dodecahedron in file dodecahedra.py allows for the creation of objects
that are to be used to represent irregular dodecahedra comprised of twelve
irregular pentagons, thirty chords of differing lengths, and twenty vertices.
The dodecahedron is regular in its reference configuration, otherwise not.

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

class dodecahedron

constructor

    d = dodecahedron(diaChord, widthSepta, rhoChord, rhoSepta, rhoAir,
                     gaussPtsChord, gaussPtsPent, gaussPtsTet, F0, h,
                     alveolarDiameter=1.9524008028984345)
        diaChord           diameter of the septal chords (in cm)
        widthSepta         thickness of the septal membranes (in cm)
        rhoChord           mass density of the septal chords (in gr/cm^3)
        rhoSepta           mass density of the septal membranes (in gr/cm^3)
        rhoAir             mass density of the air (in gr/cm^3)
        gaussPtsBar        number of Gauss points in a chord: 1, 2 or 3
        gaussPtsPent       number of Gauss points in a pentagon: 1, 4 or 7
        gaussPtsTet        number of Gauss points in a tetrahedron: 1, 4 or 5
        F0                 far-field deformation gradient for a reference shape
        h                  time seperating two successive calls to 'advance'
        alveolarDiameter   mean diameter of an alveolar sac (in cm)

    The default alveolar diameter results in vertices of the dodecahedron
    taking on coordinate values that associate with its natural configuration,
    i.e., eight of the twenty vertices take on the coordinates of cube whose
    coordinate origin resides at cube's centroid with all dodecahedral vertices
    touching a sphere of unit radius.

    The deformation gradient 'F0' allows for irregular dodecahedra in their
    reference configuration with  F0 = I  producing regular dodecahedra in
    their reference state.

methods

    s = d.vertices2string(state)
        returns a string description for the set of vertices that describe a
        dodecahedron in configuration 'state'

    s = d.chords2string(state)
        returns a string description for the set of chords that describe a
        dodecahedron in configuration 'state'

    s = d.pentagons2string(state)
        returns a string description for the set of pentagons that describe a
        dodecahedron in configuration 'state'

    s = d.tetrahedra2string(state)
        returns a string description for the set of tetrahedra that describe
        the volume of a dodecahedron in configuration 'state'

    v = d.getVertex(number)
        returns the vertex with number 'number', which must be in [1, 33]

    c = d.getChord(number)
        returns the chord with number 'number', which must be in [1, 30]

    p = d.getPentagon(number)
        returns the pentagon with number 'number', which must be in [1, 12]

    t = d.getTetrahedron(number)
        returns the tetrahedron with number 'number', which must be in [1, 60]

    d.update(nextF)
        assuming that the deformation imposed on an alveolus is homogeneous,
        described by a deformation gradient 'nextF', this procedure assigns
        new coordinate values to all of the vertices of the dodecahedron for
        this next configuration.  It calls the update method for all of its
        vertices, chords, pentagons and tetrahera, and then updates the local
        fields of the dodecahedron.  This method may be called multiple times
        before freezing its values with a call to advance

        the actual deformation being imposed on the dodecahedron is the dot
        product between 'nextF' and 'F0', i.e., F = nextF.F0, as 'F0' is taken
        to describe the reference shape.  'F0' is not a shape change caused by
        imposed tractions

    d.advance()
        calls method 'advance' for all of the vertices, chords, pentagons and
        tetrahedra comprising the dodecahedron, where current fields are
        assigned to previous fields, and then next fields are assigned to
        current fields for these objects.  Afterwords, it assigns the next
        fields to the current fields of the dodecahedron, thereby freezing
        the present next-fields in preparation for advancing the solution
        along its path

    The geometric fields associated with a dodecahedron

    v = d.volume(state)
        returns the volume of this dodecahedron in configuration 'state'

    vLambda = d.volumetricStretch(state)
        returns the cube root of volume(state) divided by its reference volume
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

    def __init__(self, diaChord, widthSepta, rhoChord, rhoSepta, rhoAir,
                 gaussPtsBar, gaussPtsPent, gaussPtsTet, F0, h,
                 alveolarDiameter=1.9524008028984345):
        # verify the inputs
        if diaChord <= 0.0:
            raise RuntimeError("The diameter sent to the constructor was " +
                               "{}; it must be positive.".format(diaChord))
        if widthSepta <= 0.0:
            raise RuntimeError("The width sent to the constructor was " +
                               "{}; it must be positive.".format(widthSepta))
        if rhoChord <= 0.0:
            raise RuntimeError("Mass density rhoChord must be positive, you " +
                               "sent {} to the constructor..".format(rhoChord))
        if rhoSepta <= 0.0:
            raise RuntimeError("Mass density rhoSepta must be positive, you " +
                               "sent {} to the constructor..".format(rhoSepta))
        if rhoAir <= 0.0:
            raise RuntimeError("Mass density rhoAir must be positive, you " +
                               "sent {} to the constructor..".format(rhoAir))
        if gaussPtsBar != 1 and gaussPtsBar != 2 and gaussPtsBar != 3:
            raise RuntimeError('Gauss points for the chords were specified ' +
                               'at {}; it must be 1, 2 or 3.'
                               .format(gaussPtsBar))
        if gaussPtsPent != 1 and gaussPtsPent != 2 and gaussPtsPent != 3:
            raise RuntimeError('Gauss points for the pentagons were ' +
                               'specified at {}; it must be 1, 4 or 7.'
                               .format(gaussPtsPent))
        if gaussPtsTet != 1 and gaussPtsTet != 2 and gaussPtsTet != 3:
            raise RuntimeError('Gauss points for the tetrahedra were ' +
                               'specified at {}; it must be 1, 4 or 5.'
                               .format(gaussPtsTet))
        if (not isinstance(F0, np.ndarray)) or (F0.shape != (3, 3)):
            raise RuntimeError("Error: F0 sent to the dodecahedron " +
                               "constructor must be a 3x3 numpy array.")
        if h < np.finfo(float).eps:
            raise RuntimeError('Error: the stepsize sent to the dodecahedron' +
                               "constructor wasn't positive.")
        if alveolarDiameter < np.finfo(float).eps:
            raise RuntimeError('Error: the alveolar diameter must be ' +
                               'positive valued.')
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
            1: vertex(1, v1x, v1y, v1z, h),
            2: vertex(2, v2x, v2y, v2z, h),
            3: vertex(3, v3x, v3y, v3z, h),
            4: vertex(4, v4x, v4y, v4z, h),
            5: vertex(5, v5x, v5y, v5z, h),
            6: vertex(6, v6x, v6y, v6z, h),
            7: vertex(7, v7x, v7y, v7z, h),
            8: vertex(8, v8x, v8y, v8z, h),
            9: vertex(9, v9x, v9y, v9z, h),
            10: vertex(10, v10x, v10y, v10z, h),
            11: vertex(11, v11x, v11y, v11z, h),
            12: vertex(12, v12x, v12y, v12z, h),
            13: vertex(13, v13x, v13y, v13z, h),
            14: vertex(14, v14x, v14y, v14z, h),
            15: vertex(15, v15x, v15y, v15z, h),
            16: vertex(16, v16x, v16y, v16z, h),
            17: vertex(17, v17x, v17y, v17z, h),
            18: vertex(18, v18x, v18y, v18z, h),
            19: vertex(19, v19x, v19y, v19z, h),
            20: vertex(20, v20x, v20y, v20z, h),
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
            1: chord(1, self._vertex[9], self._vertex[10], h, gaussPtsBar),
            2: chord(2, self._vertex[1], self._vertex[9], h, gaussPtsBar),
            3: chord(3, self._vertex[2], self._vertex[10], h, gaussPtsBar),
            4: chord(4, self._vertex[3], self._vertex[10], h, gaussPtsBar),
            5: chord(5, self._vertex[4], self._vertex[9], h, gaussPtsBar),
            6: chord(6, self._vertex[1], self._vertex[11], h, gaussPtsBar),
            7: chord(7, self._vertex[2], self._vertex[11], h, gaussPtsBar),
            8: chord(8, self._vertex[3], self._vertex[13], h, gaussPtsBar),
            9: chord(9, self._vertex[4], self._vertex[13], h, gaussPtsBar),
            10: chord(10, self._vertex[2], self._vertex[17], h, gaussPtsBar),
            11: chord(11, self._vertex[17], self._vertex[18], h, gaussPtsBar),
            12: chord(12, self._vertex[3], self._vertex[18], h, gaussPtsBar),
            13: chord(13, self._vertex[4], self._vertex[16], h, gaussPtsBar),
            14: chord(14, self._vertex[15], self._vertex[16], h, gaussPtsBar),
            15: chord(15, self._vertex[1], self._vertex[15], h, gaussPtsBar),
            16: chord(16, self._vertex[5], self._vertex[15], h, gaussPtsBar),
            17: chord(17, self._vertex[5], self._vertex[12], h, gaussPtsBar),
            18: chord(18, self._vertex[11], self._vertex[12], h, gaussPtsBar),
            19: chord(19, self._vertex[6], self._vertex[12], h, gaussPtsBar),
            20: chord(20, self._vertex[6], self._vertex[17], h, gaussPtsBar),
            21: chord(21, self._vertex[7], self._vertex[18], h, gaussPtsBar),
            22: chord(22, self._vertex[7], self._vertex[14], h, gaussPtsBar),
            23: chord(23, self._vertex[13], self._vertex[14], h, gaussPtsBar),
            24: chord(24, self._vertex[8], self._vertex[14], h, gaussPtsBar),
            25: chord(25, self._vertex[8], self._vertex[16], h, gaussPtsBar),
            26: chord(26, self._vertex[5], self._vertex[19], h, gaussPtsBar),
            27: chord(27, self._vertex[6], self._vertex[20], h, gaussPtsBar),
            28: chord(28, self._vertex[7], self._vertex[20], h, gaussPtsBar),
            29: chord(29, self._vertex[8], self._vertex[19], h, gaussPtsBar),
            30: chord(30, self._vertex[19], self._vertex[20], h, gaussPtsBar)
        }

        # create the pentagons of a dodecahedron as a dictionary
        self._pentagon = {
            1: pentagon(1, self._chord[6], self._chord[7], self._chord[3],
                        self._chord[1], self._chord[2], h, gaussPtsPent),
            2: pentagon(2, self._chord[4], self._chord[3], self._chord[10],
                        self._chord[11], self._chord[12], h, gaussPtsPent),
            3: pentagon(3, self._chord[8], self._chord[9], self._chord[5],
                        self._chord[1], self._chord[4], h, gaussPtsPent),
            4: pentagon(4, self._chord[2], self._chord[5], self._chord[13],
                        self._chord[14], self._chord[15], h, gaussPtsPent),
            5: pentagon(5, self._chord[15], self._chord[16], self._chord[17],
                        self._chord[18], self._chord[6], h, gaussPtsPent),
            6: pentagon(6, self._chord[20], self._chord[10], self._chord[7],
                        self._chord[18], self._chord[19], h, gaussPtsPent),
            7: pentagon(7, self._chord[12], self._chord[21], self._chord[22],
                        self._chord[23], self._chord[8], h, gaussPtsPent),
            8: pentagon(8, self._chord[25], self._chord[13], self._chord[9],
                        self._chord[23], self._chord[24], h, gaussPtsPent),
            9: pentagon(9, self._chord[19], self._chord[17], self._chord[26],
                        self._chord[30], self._chord[27], h, gaussPtsPent),
            10: pentagon(10, self._chord[24], self._chord[22], self._chord[28],
                         self._chord[30], self._chord[29], h, gaussPtsPent),
            11: pentagon(11, self._chord[27], self._chord[28], self._chord[21],
                         self._chord[11], self._chord[20], h, gaussPtsPent),
            12: pentagon(12, self._chord[29], self._chord[26], self._chord[16],
                         self._chord[14], self._chord[25], h, gaussPtsPent)
        }

        # asign the remaining vertices; they are the pentagonal centroids
        # and the centroid of the dodecahedron
        c = np.array(3, dtype=float)
        c = self._pentagon[1].centroid('ref')
        self._vertex[21] = vertex(21, c[0], c[1], c[2], h)
        c = self._pentagon[2].centroid('ref')
        self._vertex[22] = vertex(22, c[0], c[1], c[2], h)
        c = self._pentagon[3].centroid('ref')
        self._vertex[23] = vertex(23, c[0], c[1], c[2], h)
        c = self._pentagon[4].centroid('ref')
        self._vertex[24] = vertex(24, c[0], c[1], c[2], h)
        c = self._pentagon[5].centroid('ref')
        self._vertex[25] = vertex(25, c[0], c[1], c[2], h)
        c = self._pentagon[6].centroid('ref')
        self._vertex[26] = vertex(26, c[0], c[1], c[2], h)
        c = self._pentagon[7].centroid('ref')
        self._vertex[27] = vertex(27, c[0], c[1], c[2], h)
        c = self._pentagon[8].centroid('ref')
        self._vertex[28] = vertex(28, c[0], c[1], c[2], h)
        c = self._pentagon[9].centroid('ref')
        self._vertex[29] = vertex(29, c[0], c[1], c[2], h)
        c = self._pentagon[10].centroid('ref')
        self._vertex[30] = vertex(30, c[0], c[1], c[2], h)
        c = self._pentagon[11].centroid('ref')
        self._vertex[31] = vertex(31, c[0], c[1], c[2], h)
        c = self._pentagon[12].centroid('ref')
        self._vertex[32] = vertex(32, c[0], c[1], c[2], h)
        self._vertex[33] = vertex(33, 0.0, 0.0, 0.0, h)

        # create the tetrahedra that fill the volume as a dictionary
        self._tetrahedron = {
            1: tetrahedron(1, self._vertex[11], self._vertex[2],
                           self._vertex[21], self._vertex[33], h, gaussPtsTet),
            2: tetrahedron(2, self._vertex[2], self._vertex[10],
                           self._vertex[21], self._vertex[33], h, gaussPtsTet),
            3: tetrahedron(3, self._vertex[10], self._vertex[9],
                           self._vertex[21], self._vertex[33], h, gaussPtsTet),
            4: tetrahedron(4, self._vertex[9], self._vertex[1],
                           self._vertex[21], self._vertex[33], h, gaussPtsTet),
            5: tetrahedron(5, self._vertex[1], self._vertex[11],
                           self._vertex[21], self._vertex[33], h, gaussPtsTet),
            6: tetrahedron(6, self._vertex[10], self._vertex[2],
                           self._vertex[22], self._vertex[33], h, gaussPtsTet),
            7: tetrahedron(7, self._vertex[2], self._vertex[17],
                           self._vertex[22], self._vertex[33], h, gaussPtsTet),
            8: tetrahedron(8, self._vertex[17], self._vertex[18],
                           self._vertex[22], self._vertex[33], h, gaussPtsTet),
            9: tetrahedron(9, self._vertex[18], self._vertex[3],
                           self._vertex[22], self._vertex[33], h, gaussPtsTet),
            10: tetrahedron(10, self._vertex[3], self._vertex[10],
                            self._vertex[22], self._vertex[33], h,
                            gaussPtsTet),
            11: tetrahedron(11, self._vertex[13], self._vertex[4],
                            self._vertex[23], self._vertex[33], h,
                            gaussPtsTet),
            12: tetrahedron(12, self._vertex[4], self._vertex[9],
                            self._vertex[23], self._vertex[33], h,
                            gaussPtsTet),
            13: tetrahedron(13, self._vertex[9], self._vertex[10],
                            self._vertex[23], self._vertex[33], h,
                            gaussPtsTet),
            14: tetrahedron(14, self._vertex[10], self._vertex[3],
                            self._vertex[23], self._vertex[33], h,
                            gaussPtsTet),
            15: tetrahedron(15, self._vertex[3], self._vertex[13],
                            self._vertex[23], self._vertex[33], h,
                            gaussPtsTet),
            16: tetrahedron(16, self._vertex[9], self._vertex[4],
                            self._vertex[24], self._vertex[33], h,
                            gaussPtsTet),
            17: tetrahedron(17, self._vertex[4], self._vertex[16],
                            self._vertex[24], self._vertex[33], h,
                            gaussPtsTet),
            18: tetrahedron(18, self._vertex[16], self._vertex[15],
                            self._vertex[24], self._vertex[33], h,
                            gaussPtsTet),
            19: tetrahedron(19, self._vertex[15], self._vertex[1],
                            self._vertex[24], self._vertex[33], h,
                            gaussPtsTet),
            20: tetrahedron(20, self._vertex[1], self._vertex[9],
                            self._vertex[24], self._vertex[33], h,
                            gaussPtsTet),
            21: tetrahedron(21, self._vertex[15], self._vertex[5],
                            self._vertex[25], self._vertex[33], h,
                            gaussPtsTet),
            22: tetrahedron(22, self._vertex[5], self._vertex[12],
                            self._vertex[25], self._vertex[33], h,
                            gaussPtsTet),
            23: tetrahedron(23, self._vertex[12], self._vertex[11],
                            self._vertex[25], self._vertex[33], h,
                            gaussPtsTet),
            24: tetrahedron(24, self._vertex[11], self._vertex[1],
                            self._vertex[25], self._vertex[33], h,
                            gaussPtsTet),
            25: tetrahedron(25, self._vertex[1], self._vertex[15],
                            self._vertex[25], self._vertex[33], h,
                            gaussPtsTet),
            26: tetrahedron(26, self._vertex[17], self._vertex[2],
                            self._vertex[26], self._vertex[33], h,
                            gaussPtsTet),
            27: tetrahedron(27, self._vertex[2], self._vertex[11],
                            self._vertex[26], self._vertex[33], h,
                            gaussPtsTet),
            28: tetrahedron(28, self._vertex[11], self._vertex[12],
                            self._vertex[26], self._vertex[33], h,
                            gaussPtsTet),
            29: tetrahedron(29, self._vertex[12], self._vertex[6],
                            self._vertex[26], self._vertex[33], h,
                            gaussPtsTet),
            30: tetrahedron(30, self._vertex[6], self._vertex[17],
                            self._vertex[26], self._vertex[33], h,
                            gaussPtsTet),
            31: tetrahedron(31, self._vertex[18], self._vertex[7],
                            self._vertex[27], self._vertex[33], h,
                            gaussPtsTet),
            32: tetrahedron(32, self._vertex[7], self._vertex[14],
                            self._vertex[27], self._vertex[33], h,
                            gaussPtsTet),
            33: tetrahedron(33, self._vertex[14], self._vertex[13],
                            self._vertex[27], self._vertex[33], h,
                            gaussPtsTet),
            34: tetrahedron(34, self._vertex[13], self._vertex[3],
                            self._vertex[27], self._vertex[33], h,
                            gaussPtsTet),
            35: tetrahedron(35, self._vertex[3], self._vertex[18],
                            self._vertex[27], self._vertex[33], h,
                            gaussPtsTet),
            36: tetrahedron(36, self._vertex[16], self._vertex[4],
                            self._vertex[28], self._vertex[33], h,
                            gaussPtsTet),
            37: tetrahedron(37, self._vertex[4], self._vertex[13],
                            self._vertex[28], self._vertex[33], h,
                            gaussPtsTet),
            38: tetrahedron(38, self._vertex[13], self._vertex[14],
                            self._vertex[28], self._vertex[33], h,
                            gaussPtsTet),
            39: tetrahedron(39, self._vertex[14], self._vertex[8],
                            self._vertex[28], self._vertex[33], h,
                            gaussPtsTet),
            40: tetrahedron(40, self._vertex[8], self._vertex[16],
                            self._vertex[28], self._vertex[33], h,
                            gaussPtsTet),
            41: tetrahedron(41, self._vertex[12], self._vertex[5],
                            self._vertex[29], self._vertex[33], h,
                            gaussPtsTet),
            42: tetrahedron(42, self._vertex[5], self._vertex[19],
                            self._vertex[29], self._vertex[33], h,
                            gaussPtsTet),
            43: tetrahedron(43, self._vertex[19], self._vertex[20],
                            self._vertex[29], self._vertex[33], h,
                            gaussPtsTet),
            44: tetrahedron(44, self._vertex[20], self._vertex[6],
                            self._vertex[29], self._vertex[33], h,
                            gaussPtsTet),
            45: tetrahedron(45, self._vertex[6], self._vertex[12],
                            self._vertex[29], self._vertex[33], h,
                            gaussPtsTet),
            46: tetrahedron(46, self._vertex[14], self._vertex[7],
                            self._vertex[30], self._vertex[33], h,
                            gaussPtsTet),
            47: tetrahedron(47, self._vertex[7], self._vertex[20],
                            self._vertex[30], self._vertex[33], h,
                            gaussPtsTet),
            48: tetrahedron(48, self._vertex[20], self._vertex[19],
                            self._vertex[30], self._vertex[33], h,
                            gaussPtsTet),
            49: tetrahedron(49, self._vertex[19], self._vertex[8],
                            self._vertex[30], self._vertex[33], h,
                            gaussPtsTet),
            50: tetrahedron(50, self._vertex[8], self._vertex[14],
                            self._vertex[30], self._vertex[33], h,
                            gaussPtsTet),
            51: tetrahedron(51, self._vertex[20], self._vertex[7],
                            self._vertex[31], self._vertex[33], h,
                            gaussPtsTet),
            52: tetrahedron(52, self._vertex[7], self._vertex[18],
                            self._vertex[31], self._vertex[33], h,
                            gaussPtsTet),
            53: tetrahedron(53, self._vertex[18], self._vertex[17],
                            self._vertex[31], self._vertex[33], h,
                            gaussPtsTet),
            54: tetrahedron(54, self._vertex[17], self._vertex[6],
                            self._vertex[31], self._vertex[33], h,
                            gaussPtsTet),
            55: tetrahedron(55, self._vertex[6], self._vertex[20],
                            self._vertex[31], self._vertex[33], h,
                            gaussPtsTet),
            56: tetrahedron(56, self._vertex[19], self._vertex[5],
                            self._vertex[32], self._vertex[33], h,
                            gaussPtsTet),
            57: tetrahedron(57, self._vertex[5], self._vertex[15],
                            self._vertex[32], self._vertex[33], h,
                            gaussPtsTet),
            58: tetrahedron(58, self._vertex[15], self._vertex[16],
                            self._vertex[32], self._vertex[33], h,
                            gaussPtsTet),
            59: tetrahedron(59, self._vertex[16], self._vertex[8],
                            self._vertex[32], self._vertex[33], h,
                            gaussPtsTet),
            60: tetrahedron(60, self._vertex[8], self._vertex[19],
                            self._vertex[32], self._vertex[33], h, gaussPtsTet)
        }

        # add up the volumes associated with the sixty tetrahedra
        vol = 0.0
        for i in range(1, 61):
            vol = vol + self._tetrahedron[i].volume('ref')
        self._refVol = vol
        self._prevVol = vol
        self._currVol = vol
        self._nextVol = vol

    def verticesToString(self, state):
        s = 'In state ' + state + ', the dodecahedron has vertices of: \n'
        for i in range(1, 20):
            s = s + '   ' + self._vertex[i].toString(state) + '\n'
        s = s + '   ' + self._vertex[20].toString(state)
        return s

    def chordsToString(self, state):
        s = 'In state ' + state + ', the dodecahedron has chords of: \n'
        for i in range(1, 30):
            s = s + self._chord[i].toString(state) + '\n'
        s = s + self._chord[30].toString(state)
        return s

    def pentagonsToString(self, state):
        s = 'In state ' + state + ', the dodecahedron has pentagons of: \n'
        for i in range(1, 12):
            s = s + self._pentagon[i].toString(state) + '\n'
        s = s + self._pentagon[12].toString(state)
        return s

    def getVertex(self, number):
        if number > 0 and number < 21:
            return self._vertex[number]
        else:
            raise RuntimeError('Error: requested vertex {} does not exist.'
                               .format(number))

    def getChord(self, number):
        if number > 0 and number < 31:
            return self._chord[number]
        else:
            raise RuntimeError('Error: requested chord {} does not exist.'
                               .format(number))

    def getPentagon(self, number):
        if number > 0 and number < 13:
            return self._pentagon[number]
        else:
            raise RuntimeError('Error: requested pentagon {} does not exist'
                               .format(number))

    def update(self, nextF, rho):
        if (not isinstance(nextF, np.ndarray)) or (nextF.shape != (3, 3)):
            raise RuntimeError("Error: nextF sent to dodecahedron.update " +
                               "must be a 3x3 numpy array.")

        # update the vertices for the lattice of the dodecahedron
        for i in range(1, 21):
            x0, y0, z0 = self._vertex[i].coordinates('reference')
            # deformation of the dodecahedron is taken to be homogeneous
            x = nextF[0, 0] * x0 + nextF[0, 1] * y0 + nextF[0, 2] * z0
            y = nextF[1, 0] * x0 + nextF[1, 1] * y0 + nextF[1, 2] * z0
            z = nextF[2, 0] * x0 + nextF[2, 1] * y0 + nextF[2, 2] * z0
            self._vertex[i].update(x, y, z)

        # update the chords of the dodecahedron
        for i in range(1, 31):
            self._chord[i].update()

        # update the pentagons of the dodecahedron
        for i in range(1, 13):
            self._pentagon[i].update()

        # update the vertices that comprise the centroids of the dodecahedron
        c = np.array(3, dtype=float)
        for i in range(21, 33):
            c = self._pentagon[1].centroid('next')
            self._vertex[i].update(c[0], c[1], c[2])

        # update the tetrahedra of the dodecahedron
        for i in range(1, 61):
            self._tetrahedron[i].update()

        # update volume: this is costly because the dodecahedron is irregular
        # add up the volumes associated with the twelve pentangonal faces
        vol = 0.0
        for i in range(1, 61):
            vol = vol + self._tetrahedron[i].volume('next')
        self._nextVol = vol





        def detJacTet(x1, x2, x3, x4):
            if self._gaussPtsTet == 1:
                # determinant of the Jacobian Matrix
                detJ1 = self._shapeFnsT[1].jacob(x1, x2, x3, x4)

                detJn = {
                        1: detJ1
                        }

            elif self._gaussPtsTet == 4:
                # determinant of the Jacobian Matrix
                detJ1 = self._shapeFnsT[1].jacob(x1, x2, x3, x4)
                detJ2 = self._shapeFnsT[2].jacob(x1, x2, x3, x4)
                detJ3 = self._shapeFnsT[3].jacob(x1, x2, x3, x4)
                detJ4 = self._shapeFnsT[4].jacob(x1, x2, x3, x4)

                detJn = {
                        1: detJ1,
                        2: detJ2,
                        3: detJ3,
                        4: detJ4
                        }

            else:  # gaussPtsTet = 5
                # determinant of the Jacobian Matrix
                detJ1 = self._shapeFnsT[1].jacob(x1, x2, x3, x4)
                detJ2 = self._shapeFnsT[2].jacob(x1, x2, x3, x4)
                detJ3 = self._shapeFnsT[3].jacob(x1, x2, x3, x4)
                detJ4 = self._shapeFnsT[4].jacob(x1, x2, x3, x4)
                detJ5 = self._shapeFnsT[5].jacob(x1, x2, x3, x4)

                detJn = {
                        1: detJ1,
                        2: detJ2,
                        3: detJ3,
                        4: detJ4,
                        5: detJ5
                        }

            return detJn

        def nextDetJacob(p):
            # two vertices are common to all five tetrahedons
            # the origin and the centroid of the pentagon
            cx, cy, cz = p.centroid('next')

            # get the chords of the pentagon
            n1, n2, n3, n4, n5 = p.chordNumbers()
            c1 = p.getChord(n1)

            # chord 1
            n1, n2 = c1.vertexNumbers()
            v1 = c1.getVertex(n1)
            x1, y1, z1 = v1.coordinates('next')
            v2 = c1.getVertex(n2)
            x2, y2, z2 = v2.coordinates('next')
            # current vertex coordinates of tetrahedron
            x1 = (x1, y1, z1)
            x2 = (x2, y2, z2)
            x3 = (cx, cy, cz)
            x4 = (0.0, 0.0, 0.0)
            detJn = detJacTet(x1, x2, x3, x4)

            return detJn

        self._detJn = nextDetJacob(self._pentagon[1])

    def advance(self):
        # advance kinematic fields of the vertices
        for i in range(1, 21):
            self._vertex[i].advance()

        # advance kinematic fields of the chords
        for i in range(1, 31):
            self._chord[i].advance()

        # advance kinematic fields of the pentagons
        for i in range(1, 13):
            self._pentagon[i].advance()

        # advance kinematic fields of the dodecahedron
        self._prevVol = self._currVol
        self._currVol = self._nextVol

        # advance the fields associated with each Gauss point
        for i in range(1, self._gaussPtsTet+1):
            self._detJp[i] = self._detJc[i]
            self._detJc[i] = self._detJn[i]

    # properties of the tetrahedra within the dodecahedron

    # determinant of Jacobian at a Gauss point
    def detJacobianTet(self, gaussPtTet, state):
        if (gaussPtTet < 1) or (gaussPtTet > self._gaussPtsTet):
            if self._gaussPtsTet == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "dodecahedron.detJacobianTet and you " +
                                   "sent {}.".format(gaussPtTet))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPtsTet) +
                                   "to dodecahedron.detJacobianTet and you " +
                                   "sent {}.".format(gaussPtTet))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._detJc
            elif state == 'n' or state == 'next':
                return self._detJn
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._detJp
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._detJ0
            else:
                raise RuntimeError("Error: unknown state {} in call to " +
                                   "dodecahedron.detJT.".format(state))
        else:
            raise RuntimeError("Error: unknown state {} in call to " +
                               "dodecahedron.detJT.".format(str(state)))

    def massMatrixTet(self, gaussPtTet, rho):
        if (gaussPtTet < 1) or (gaussPtTet > self._gaussPtsTet):
            if self._gaussPtsTet == 1:
                raise RuntimeError("Error: gaussPt can only be 1 in call to " +
                                   "dodecahedron.massMatrixTet and you " +
                                   "sent {}.".format(gaussPtTet))
            else:
                raise RuntimeError("Error: gaussPt must be in [1, {}] in call "
                                   .format(self._gaussPtsTet) + "to " +
                                   "dodecahedron.massMatrixTet and you " +
                                   "sent {}.".format(gaussPtTet))

        # determine the mass matrix
        if self._gaussPtsTet == 1:
            # 'natural' weight of the element
            wel = np.array([1 / 6])

            nn1 = np.dot(np.transpose(self._shapeFnsT[1].Nmatx),
                         self._shapeFnsT[1].Nmatx)

            # Integration to get the mass Matrix for 1 Gauss points
            mass = rho * (nn1 * self._detJc[1] * wel[0])
            return mass

        elif self._gaussPtsTet == 4:
            # 'natural' weight of the element
            wel = np.array([1 / 24, 1 / 24, 1 / 24, 1 / 24])

            nn1 = np.dot(np.transpose(self._shapeFnsT[1].Nmatx),
                         self._shapeFnsT[1].Nmatx)
            nn2 = np.dot(np.transpose(self._shapeFnsT[2].Nmatx),
                         self._shapeFnsT[2].Nmatx)
            nn3 = np.dot(np.transpose(self._shapeFnsT[3].Nmatx),
                         self._shapeFnsT[3].Nmatx)
            nn4 = np.dot(np.transpose(self._shapeFnsT[4].Nmatx),
                         self._shapeFnsT[4].Nmatx)

            # Integration to get the mass Matrix for 4 Gauss points
            mass = (rho * (nn1 * self._detJc[1] * wel[0] +
                           nn2 * self._detJc[2] * wel[1] +
                           nn3 * self._detJc[3] * wel[2] +
                           nn4 * self._detJc[4] * wel[3]))
            return mass

        else:  # gaussPtsTet = 5
            # 'natural' weight of the element
            wel = np.array([-2 / 15, 3 / 40, 3 / 40, 3 / 40, 3 / 40])

            nn1 = np.dot(np.transpose(self._shapeFnsT[1].Nmatx),
                         self._shapeFnsT[1].Nmatx)
            nn2 = np.dot(np.transpose(self._shapeFnsT[2].Nmatx),
                         self._shapeFnsT[2].Nmatx)
            nn3 = np.dot(np.transpose(self._shapeFnsT[3].Nmatx),
                         self._shapeFnsT[3].Nmatx)
            nn4 = np.dot(np.transpose(self._shapeFnsT[4].Nmatx),
                         self._shapeFnsT[4].Nmatx)
            nn5 = np.dot(np.transpose(self._shapeFnsT[5].Nmatx),
                         self._shapeFnsT[5].Nmatx)

            # Integration to get the mass Matrix for 5 Gauss points
            mass = (rho * (nn1 * self._detJc[1] * wel[0] +
                           nn2 * self._detJc[2] * wel[1] +
                           nn3 * self._detJc[3] * wel[2] +
                           nn4 * self._detJc[4] * wel[3] +
                           nn5 * self._detJc[5] * wel[4]))
            return mass

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
                raise RuntimeError(
                      "Error: unknown state {} in call to dodecahedron.volume."
                      .format(state))
        else:
            raise RuntimeError(
                      "Error: unknown state {} in call to dodecahedron.volume."
                      .format(str(state)))

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
                raise RuntimeError("Error: unknown state {} in".format(state) +
                                   " call to dodecahedron.volumetricStretch.")
        else:
            raise RuntimeError("Error: unknown state {} in".format(str(state))
                               + " call to dodecahedron.volumetricStretch.")

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
                raise RuntimeError("Error: unknown state {} in".format(state) +
                                   " call to dodecahedron.volumetricStretch.")
        else:
            raise RuntimeError("Error: unknown state {}".format(str(state)) +
                               " in call to dodecahedron.volumetricStretch.")

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
                raise RuntimeError("Error: unknown state {}".format(state) +
                                   " in dodecahedron.dVolumetricStrain.")
        else:
            raise RuntimeError("Error: unknown state {}".format(str(state)) +
                               " in dodecahedron.dVolumetricStrain.")
