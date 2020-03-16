#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import chord
import math as m
import numpy as np
from shapeFnChords import shapeFunction

"""
Module pentForce.py provides force vector of the pentagon.

Copyright (c) 2020 Shahla Zamani

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
__date__ = "03-10-2020"
__update__ = "03-16-2020"
__author__ = "Shahla Zamani"
__author_email__ = "Zamani.Shahla@tamu.edu"

"""
Class pentForce in file pentForces.py allows for the creation of objects that 
are to be used to represent the force on the boundaries of an irregular 
pentagons comprised of five connected chords.  A chord is assigned an unique 
number, two distinct vertices that serve as end points, the time step size 
used to approximate derivatives and  integrals, and the number of Gauss points 
to be used for integration.

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

constructor

    c = pentForce(number, chord1, chord2, chord3, chord4, chord5, h,
                 chordGaussPts)
        number         immutable value that is unique to this pentagon
        chord1         unique edge of the pentagon, an instance of class chord
        chord2         unique edge of the pentagon, an instance of class chord
        chord3         unique edge of the pentagon, an instance of class chord
        chord4         unique edge of the pentagon, an instance of class chord
        chord5         unique edge of the pentagon, an instance of class chord
        h              timestep size between two successive calls to 'advance'
        chordGaussPts  number of Gauss points to be used: must be 1, 2 or 3

methods

    n = p.number()
        returns the unique indexing number affiated with this pentagon

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
        returns the number of Gauss points assigned to this chord

    c.update()
        assigns new coordinate values to the pentagon for its next location and
        updates all effected fields.  It is to be called after all vertices
        have had their coordinates updated.  This may be called multiple times
        before freezing its values with a call to 'advance'

    c.advance()
        assigns the current fields to the previous fields, and then it assigns
        the next fields to the current fields, thereby freezing the present
        next-fields in preparation for an advancment of the solution along its
        path


    ell12 = p.length12(state)
        returns the 1-2 chordal length in configuration 'state'

    ell23 = p.length23(state)
        returns the 2-3 chordal length in configuration 'state'

    ell34 = p.length34(state)
        returns the 3-4 chordal length in configuration 'state'

    ell45 = p.length45(state)
        returns the 4-5 chordal length in configuration 'state'

    ell51 = p.length51(state)
        returns the 5-1 chordal length in configuration 'state'        

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
    Fields needed to construct a finite element solution strategy are:

    sf = p.shapeFunction(gaussPt):
        returns the shape function associated with the specified Gauss point.

    fVec = p.pentagonForcingFunction()
        returns a vector for the forcing function on the right-hand side
        belonging to the current state.
"""

class pentForce(object):

    def __init__(self, number, chord1, chord2, chord3, chord4, chord5, h,
                 chordGaussPts):
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
        if chordGaussPts == 1 or chordGaussPts == 2 or chordGaussPts == 3:
            self._chordGaussPts = chordGaussPts
        else:
            raise RuntimeError('{} Gauss points were '.format(chordGaussPts) +
                               'specified in a call to the chord ' +
                               'constructor; it must be 1, 2 or 3.')

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
        x1, y1, z1 = v1.coordinates('ref')
        x2, y2, z2 = v2.coordinates('ref')
        L120 = m.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        self._L120 = L120
        self._L12p = L120
        self._L12c = L120
        self._L12n = L120
        
        # initialize the 2-3 chordal lengths for all configurations
        x2, y2, z2 = v2.coordinates('ref')
        x3, y3, z3 = v3.coordinates('ref')
        L230 = m.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)
        self._L230 = L230
        self._L23p = L230
        self._L23c = L230
        self._L23n = L230
        
        # initialize the 3-4 chordal lengths for all configurations
        x3, y3, z3 = v3.coordinates('ref')
        x4, y4, z4 = v4.coordinates('ref')
        L340 = m.sqrt((x4 - x3)**2 + (y4 - y3)**2 + (z4 - z3)**2)
        self._L340 = L340
        self._L34p = L340
        self._L34c = L340
        self._L23n = L230
        
        # initialize the 4-5 chordal lengths for all configurations
        x4, y4, z4 = v4.coordinates('ref')
        x5, y5, z5 = v5.coordinates('ref')
        L450 = m.sqrt((x5 - x4)**2 + (y5 - y4)**2 + (z5 - z4)**2)
        self._L450 = L450
        self._L45p = L450
        self._L45c = L450
        self._L45n = L450
        
        # initialize the 5-1 chordal lengths for all configurations
        x5, y5, z5 = v5.coordinates('ref')
        x1, y1, z1 = v1.coordinates('ref')
        L510 = m.sqrt((x5 - x1)**2 + (y5 - y1)**2 + (z5 - z1)**2)
        self._L510 = L510
        self._L51p = L510
        self._L51c = L510
        self._L51n = L510
                
        # determine the rotation matrix for 1-2 chord 
        # base vector 1: aligns with the axis of the 1-2 chord
        x12 = x2 - x1
        y12 = y2 - y1
        mag12 = m.sqrt(x12 * x12 + y12 * y12)
        n1x12 = x12 / mag12
        n1y12 = y12 / mag12
        # create normal vector
        n2x12 = - n1y12
        n2y12 = n1x12
        # create the rotation matrix from dodecahedral to 1-2 chordal coordinates
        self._Pr2D12[0, 0] = n1x12
        self._Pr2D12[0, 1] = n2x12
        self._Pr2D12[1, 0] = n1y12
        self._Pr2D12[1, 1] = n2y12
        self._Pp2D12[:, :] = self._Pr2D12[:, :]
        self._Pc2D12[:, :] = self._Pr2D12[:, :]
        self._Pn2D12[:, :] = self._Pr2D12[:, :]
                
        # determine the rotation matrix for 2-3 chord 
        # base vector 1: aligns with the axis of the 2-3 chord
        x23 = x3 - x2
        y23 = y3 - y2
        mag23 = m.sqrt(x23 * x23 + y23 * y23)
        n1x23 = x23 / mag23
        n1y23 = y23 / mag23
        # create normal vector
        n2x23 = - n1y23
        n2y23 = n1x23  
        # create the rotation matrix from dodecahedral to 2-3 chordal coordinates
        self._Pr2D23[0, 0] = n1x23
        self._Pr2D23[0, 1] = n2x23
        self._Pr2D23[1, 0] = n1y23
        self._Pr2D23[1, 1] = n2y23
        self._Pp2D23[:, :] = self._Pr2D23[:, :]
        self._Pc2D23[:, :] = self._Pr2D23[:, :]
        self._Pn2D23[:, :] = self._Pr2D23[:, :]

        # determine the rotation matrix for 3-4 chord 
        # base vector 1: aligns with the axis of the 3-4 chord
        x34 = x4 - x3
        y34 = y4 - y3
        mag34 = m.sqrt(x34 * x34 + y34 * y34)
        n1x34 = x34 / mag34
        n1y34 = y34 / mag34
        # create normal vector
        n2x34 = - n1y34
        n2y34 = n1x34
        # create the rotation matrix from dodecahedral to 3-4 chordal coordinates
        self._Pr2D34[0, 0] = n1x34
        self._Pr2D34[0, 1] = n2x34
        self._Pr2D34[1, 0] = n1y34
        self._Pr2D34[1, 1] = n2y34
        self._Pp2D34[:, :] = self._Pr2D34[:, :]
        self._Pc2D34[:, :] = self._Pr2D34[:, :]
        self._Pn2D34[:, :] = self._Pr2D34[:, :]

        # determine the rotation matrix for 4-5 chord 
        # base vector 1: aligns with the axis of the 4-5 chord
        x45 = x5 - x4
        y45 = y5 - y4
        mag45 = m.sqrt(x45 * x45 + y45 * y45)
        n1x45 = x45 / mag45
        n1y45 = y45 / mag45        
        # create normal vector
        n2x45 = - n1y45
        n2y45 = n1x45 
        # create the rotation matrix from dodecahedral to 4-5 chordal coordinates
        self._Pr2D45[0, 0] = n1x45
        self._Pr2D45[0, 1] = n2x45
        self._Pr2D45[1, 0] = n1y45
        self._Pr2D45[1, 1] = n2y45
        self._Pp2D45[:, :] = self._Pr2D45[:, :]
        self._Pc2D45[:, :] = self._Pr2D45[:, :]
        self._Pn2D45[:, :] = self._Pr2D45[:, :]
        
        # determine the rotation matrix for 5-1 chord 
        # base vector 1: aligns with the axis of the 5-1 chord
        x51 = x1 - x5
        y51 = y1 - y5
        mag51 = m.sqrt(x51 * x51 + y51 * y51)
        n1x51 = x51 / mag51
        n1y51 = y51 / mag51       
        # create normal vector
        n2x51 = - n1y51
        n2y51 = n1x51
        # create the rotation matrix from dodecahedral to 5-1 chordal coordinates
        self._Pr2D51[0, 0] = n1x51
        self._Pr2D51[0, 1] = n2x51
        self._Pr2D51[1, 0] = n1y51
        self._Pr2D51[1, 1] = n2y51
        self._Pp2D51[:, :] = self._Pr2D51[:, :]
        self._Pc2D51[:, :] = self._Pr2D51[:, :]
        self._Pn2D51[:, :] = self._Pr2D51[:, :]
        
        # establish the shape functions located at the various Gauss points
        if chordGaussPts == 1:
            # this single Gauss point has a weight of 2
            xi = 0.0
            sf1 = shapeFunction(xi)

            self._shapeFns = {
                1: sf1
            }
        elif chordGaussPts == 2:
            # each of these two Gauss points has a weight of 1
            xi1 = -0.577350269189626
            sf1 = shapeFunction(xi1)

            xi2 = 0.577350269189626
            sf2 = shapeFunction(xi2)

            self._shapeFns = {
                1: sf1,
                2: sf2
            }
        else:  # chordGaussPts = 3
            # Gauss points 1 & 3 have weights of 5/9
            xi1 = -0.7745966692414834
            sf1 = shapeFunction(xi1)

            # Gauss point 2 (the centroid) has a weight of 8/9
            xi2 = 0.0
            sf2 = shapeFunction(xi2)

            xi3 = 0.7745966692414834
            sf3 = shapeFunction(xi3)

            self._shapeFns = {
                1: sf1,
                2: sf2,
                3: sf3
            }

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

    def gaussPoints(self):
        return self._chordGaussPts
       
    def update(self):
        # get the updated coordinates for the vetices of the pentagon
        x1, y1, z1 = self._vertex[1].coordinates('next')
        x2, y2, z2 = self._vertex[2].coordinates('next')
        x3, y3, z3 = self._vertex[3].coordinates('next')
        x4, y4, z4 = self._vertex[4].coordinates('next')
        x5, y5, z5 = self._vertex[5].coordinates('next')
        
        # determine length of the 1-2 chord in the next configuration
        self._L12n = m.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        
        # determine length of the 2-3 chord in the next configuration
        self._L23n = m.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)
        
        # determine length of the 3-4 chord in the next configuration
        self._L34n = m.sqrt((x4 - x3)**2 + (y4 - y3)**2 + (z4 - z3)**2)
        
        # determine length of the 4-5 chord in the next configuration
        self._L45n = m.sqrt((x5 - x4)**2 + (y5 - y4)**2 + (z5 - z4)**2)
        
        # determine length of the 5-1 chord in the next configuration
        self._L51n = m.sqrt((x5 - x1)**2 + (y5 - y1)**2 + (z5 - z1)**2)

        # determine the rotation matrix for 1-2 chord 
        # base vector 1: aligns with the axis of the 1-2 chord
        x12 = x2 - x1
        y12 = y2 - y1
        mag12 = m.sqrt(x12 * x12 + y12 * y12)
        n1x12 = x12 / mag12
        n1y12 = y12 / mag12
        # create normal vector
        n2x12 = - n1y12
        n2y12 = n1x12
        # create the rotation matrix from dodecahedral to 1-2 chordal coordinates
        self._Pn2D12[0, 0] = n1x12
        self._Pn2D12[0, 1] = n2x12
        self._Pn2D12[1, 0] = n1y12
        self._Pn2D12[1, 1] = n2y12

        # determine the rotation matrix for 2-3 chord 
        # base vector 1: aligns with the axis of the 2-3 chord
        x23 = x3 - x2
        y23 = y3 - y2
        mag23 = m.sqrt(x23 * x23 + y23 * y23)
        n1x23 = x23 / mag23
        n1y23 = y23 / mag23
        # create normal vector
        n2x23 = - n1y23
        n2y23 = n1x23  
        # create the rotation matrix from dodecahedral to 2-3 chordal coordinates
        self._Pn2D23[0, 0] = n1x23
        self._Pn2D23[0, 1] = n2x23
        self._Pn2D23[1, 0] = n1y23
        self._Pn2D23[1, 1] = n2y23
        
        # determine the rotation matrix for 3-4 chord 
        # base vector 1: aligns with the axis of the 3-4 chord
        x34 = x4 - x3
        y34 = y4 - y3
        mag34 = m.sqrt(x34 * x34 + y34 * y34)
        n1x34 = x34 / mag34
        n1y34 = y34 / mag34
        # create normal vector
        n2x34 = - n1y34
        n2y34 = n1x34
        # create the rotation matrix from dodecahedral to 3-4 chordal coordinates
        self._Pn2D34[0, 0] = n1x34
        self._Pn2D34[0, 1] = n2x34
        self._Pn2D34[1, 0] = n1y34
        self._Pn2D34[1, 1] = n2y34

        # determine the rotation matrix for 4-5 chord 
        # base vector 1: aligns with the axis of the 4-5 chord
        x45 = x5 - x4
        y45 = y5 - y4
        mag45 = m.sqrt(x45 * x45 + y45 * y45)
        n1x45 = x45 / mag45
        n1y45 = y45 / mag45
        # create normal vector
        n2x45 = - n1y45
        n2y45 = n1x45
        # create the rotation matrix from dodecahedral to 4-5 chordal coordinates
        self._Pn2D45[0, 0] = n1x45
        self._Pn2D45[0, 1] = n2x45
        self._Pn2D45[1, 0] = n1y45
        self._Pn2D45[1, 1] = n2y45
                       
        # determine the rotation matrix for 5-1 chord 
        # base vector 1: aligns with the axis of the 5-1 chord
        x51 = x1 - x5
        y51 = y1 - y5
        mag51 = m.sqrt(x51 * x51 + y51 * y51)
        n1x51 = x51 / mag51
        n1y51 = y51 / mag51
        # create normal vector
        n2x51 = - n1y51
        n2y51 = n1x51
        # create the rotation matrix from dodecahedral to 5-1 chordal coordinates
        self._Pn2D51[0, 0] = n1x51
        self._Pn2D51[0, 1] = n2x51
        self._Pn2D51[1, 0] = n1y51
        self._Pn2D51[1, 1] = n2y51
        
    def advance(self):
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
               
    def length12(self, state):
        # length of chord 1-2
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._L12c
            elif state == 'n' or state == 'next':
                return self._L12n
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._L12p
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._L120
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "sent in a call to chord.length.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "sent in call a to chord.length.")
    def length23(self, state):
        # length of chord 2-3
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._L23c
            elif state == 'n' or state == 'next':
                return self._L23n
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._L23p
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._L230
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "sent in a call to chord.length.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "sent in call a to chord.length.")

    def length34(self, state):
        # length of chord 3-4
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._L34c
            elif state == 'n' or state == 'next':
                return self._L34n
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._L34p
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._L340
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "sent in a call to chord.length.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "sent in call a to chord.length.")

    def length45(self, state):
        # length of chord 4-5
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._L45c
            elif state == 'n' or state == 'next':
                return self._L45n
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._L45p
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._L450
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "sent in a call to chord.length.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "sent in call a to chord.length.")

    def length51(self, state):
        # length of chord 5-1
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._L51c
            elif state == 'n' or state == 'next':
                return self._L51n
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._L51p
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._L510
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "sent in a call to chord.length.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "sent in call a to chord.length.")
           
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

    def shapeFunction(self, gaussPt):
        if (gaussPt < 1) or (gaussPt > self._chordGaussPts):
            if self._chordGaussPts == 1:
                raise RuntimeError("gaussPt can only be 1 in a call to " +
                                   "chord.shapeFunction and you sent " +
                                   "{}.".format(gaussPt))
            else:
                raise RuntimeError("gaussPt must be in the range of " +
                                   "[1, {}] ".format(self._chordGaussPts) +
                                   "in a call to chord.shapeFunction " +
                                   "and you sent {}.".format(gaussPt))
            sf = self._shapeFns[gaussPt]
        return sf
        
    def pentagonForcingFunction(self, cauchyStress, state):
        P12 = self.rotation12(state)
        P23 = self.rotation23(state)
        P34 = self.rotation34(state)
        P45 = self.rotation45(state)
        P51 = self.rotation51(state)
        
        # normal vector to each chord of pentagon
        n12 = np.array([P12[0, 1], P12[1, 1]])
        n23 = np.array([P23[0, 1], P23[1, 1]])
        n34 = np.array([P34[0, 1], P34[1, 1]])
        n45 = np.array([P45[0, 1], P45[1, 1]])
        n51 = np.array([P51[0, 1], P51[1, 1]])
                
        # create the traction vector apply on each chord of pentagon 
        t12 = np.dot(cauchyStress, np.transpose(n12))   
        t23 = np.dot(cauchyStress, np.transpose(n23))  
        t34 = np.dot(cauchyStress, np.transpose(n34))  
        t45 = np.dot(cauchyStress, np.transpose(n45))  
        t51 = np.dot(cauchyStress, np.transpose(n51))  
                        
        # chordal coordinates for the chords
        x1n12 = -self._L12n / 2.0
        x2n12 = self._L12n / 2.0
        
        x1n23 = -self._L23n / 2.0
        x2n23 = self._L23n / 2.0
        
        x1n34 = -self._L34n / 2.0
        x2n34 = self._L34n / 2.0
        
        x1n45 = -self._L45n / 2.0
        x2n45 = self._L45n / 2.0
        
        x1n51 = -self._L51n / 2.0
        x2n51 = self._L51n / 2.0
                
        # determine the force vector
        if self._chordGaussPts == 1:
            # 'natural' weight of the element
            wgt = 2.0
            we = np.array([wgt])

            N1 = self._shapeFns[1].N1
            N2 = self._shapeFns[1].N2
            
            N12 = np.array([[N1, 0, N2, 0, 0, 0, 0, 0, 0, 0],
                            [0, N1, 0, N2, 0, 0, 0, 0, 0, 0]])
            N23 = np.array([[0, 0, N1, 0, N2, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, N1, 0, N2, 0, 0, 0, 0]])
            N34 = np.array([[0, 0, 0, 0, N1, 0, N2, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, N1, 0, N2, 0, 0]])
            N45 = np.array([[0, 0, 0, 0, 0, 0, N1, 0, N2, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, N1, 0, N2]])
            N51 = np.array([[N2, 0, 0, 0, 0, 0, 0, 0, N1, 0],
                            [0, N2, 0, 0, 0, 0, 0, 0, 0, N1]])
                
            n1Mat12 = N12.transpose()
            n1Mat23 = N23.transpose()
            n1Mat34 = N34.transpose()
            n1Mat45 = N45.transpose()
            n1Mat51 = N51.transpose()

            J112 = self._shapeFns[1].jacobian(x1n12, x2n12)
            J123 = self._shapeFns[1].jacobian(x1n23, x2n23)
            J134 = self._shapeFns[1].jacobian(x1n34, x2n34)
            J145 = self._shapeFns[1].jacobian(x1n45, x2n45)
            J151 = self._shapeFns[1].jacobian(x1n51, x2n51)

            # the force vector for 1 Gauss point
            FVec = (we[0] * J112 * np.dot(n1Mat12, t12) + 
                    we[0] * J123 * np.dot(n1Mat23, t23) +
                    we[0] * J134 * np.dot(n1Mat34, t34) + 
                    we[0] * J145 * np.dot(n1Mat45, t45) +
                    we[0] * J151 * np.dot(n1Mat51, t51))

        elif self._chordGaussPts == 2:
            # 'natural' weights of the element
            wgt = 1.0
            we = np.array([wgt, wgt])

            # at Gauss point 1
            N1 = self._shapeFns[1].N1
            N2 = self._shapeFns[1].N2
            
            N12 = np.array([[N1, 0, N2, 0, 0, 0, 0, 0, 0, 0],
                                [0, N1, 0, N2, 0, 0, 0, 0, 0, 0]])
            N23 = np.array([[0, 0, N1, 0, N2, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, N1, 0, N2, 0, 0, 0, 0]])
            N34 = np.array([[0, 0, 0, 0, N1, 0, N2, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, N1, 0, N2, 0, 0]])
            N45 = np.array([[0, 0, 0, 0, 0, 0, N1, 0, N2, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, N1, 0, N2]])
            N51 = np.array([[N2, 0, 0, 0, 0, 0, 0, 0, N1, 0],
                                [0, N2, 0, 0, 0, 0, 0, 0, 0, N1]])
                
            n1Mat12 = N12.transpose()
            n1Mat23 = N23.transpose()
            n1Mat34 = N34.transpose()
            n1Mat45 = N45.transpose()
            n1Mat51 = N51.transpose()

            J112 = self._shapeFns[1].jacobian(x1n12, x2n12)
            J123 = self._shapeFns[1].jacobian(x1n23, x2n23)
            J134 = self._shapeFns[1].jacobian(x1n34, x2n34)
            J145 = self._shapeFns[1].jacobian(x1n45, x2n45)
            J151 = self._shapeFns[1].jacobian(x1n51, x2n51)


            # at Gauss point 2
            N1 = self._shapeFns[2].N1
            N2 = self._shapeFns[2].N2
            
            N12 = np.array([[N1, 0, N2, 0, 0, 0, 0, 0, 0, 0],
                            [0, N1, 0, N2, 0, 0, 0, 0, 0, 0]])
            N23 = np.array([[0, 0, N1, 0, N2, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, N1, 0, N2, 0, 0, 0, 0]])
            N34 = np.array([[0, 0, 0, 0, N1, 0, N2, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, N1, 0, N2, 0, 0]])
            N45 = np.array([[0, 0, 0, 0, 0, 0, N1, 0, N2, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, N1, 0, N2]])
            N51 = np.array([[N2, 0, 0, 0, 0, 0, 0, 0, N1, 0],
                            [0, N2, 0, 0, 0, 0, 0, 0, 0, N1]])
                
            n2Mat12 = N12.transpose()
            n2Mat23 = N23.transpose()
            n2Mat34 = N34.transpose()
            n2Mat45 = N45.transpose()
            n2Mat51 = N51.transpose()

            J212 = self._shapeFns[2].jacobian(x1n12, x2n12)
            J223 = self._shapeFns[2].jacobian(x1n23, x2n23)
            J234 = self._shapeFns[2].jacobian(x1n34, x2n34)
            J245 = self._shapeFns[2].jacobian(x1n45, x2n45)
            J251 = self._shapeFns[2].jacobian(x1n51, x2n51)


            # the force vector for 2 Gauss points
            FVec = (we[0] * J112 * np.dot(n1Mat12, t12) + 
                    we[0] * J123 * np.dot(n1Mat23, t23) +
                    we[0] * J134 * np.dot(n1Mat34, t34) + 
                    we[0] * J145 * np.dot(n1Mat45, t45) +
                    we[0] * J151 * np.dot(n1Mat51, t51) + 
                    we[1] * J212 * np.dot(n2Mat12, t12) + 
                    we[1] * J223 * np.dot(n2Mat23, t23) +
                    we[1] * J234 * np.dot(n2Mat34, t34) + 
                    we[1] * J245 * np.dot(n2Mat45, t45) +
                    we[1] * J251 * np.dot(n2Mat51, t51))

        else:  # chordGaussPts = 3
            # 'natural' weights of the element
            wgt1 = 5.0 / 9.0
            wgt2 = 8.0 / 9.0
            we = np.array([wgt1, wgt2, wgt1])

            # at Gauss point 1
            N1 = self._shapeFns[1].N1
            N2 = self._shapeFns[1].N2
            
            N12 = np.array([[N1, 0, N2, 0, 0, 0, 0, 0, 0, 0],
                            [0, N1, 0, N2, 0, 0, 0, 0, 0, 0]])
            N23 = np.array([[0, 0, N1, 0, N2, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, N1, 0, N2, 0, 0, 0, 0]])
            N34 = np.array([[0, 0, 0, 0, N1, 0, N2, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, N1, 0, N2, 0, 0]])
            N45 = np.array([[0, 0, 0, 0, 0, 0, N1, 0, N2, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, N1, 0, N2]])
            N51 = np.array([[N2, 0, 0, 0, 0, 0, 0, 0, N1, 0],
                            [0, N2, 0, 0, 0, 0, 0, 0, 0, N1]])
                
            n1Mat12 = N12.transpose()
            n1Mat23 = N23.transpose()
            n1Mat34 = N34.transpose()
            n1Mat45 = N45.transpose()
            n1Mat51 = N51.transpose()

            J112 = self._shapeFns[1].jacobian(x1n12, x2n12)
            J123 = self._shapeFns[1].jacobian(x1n23, x2n23)
            J134 = self._shapeFns[1].jacobian(x1n34, x2n34)
            J145 = self._shapeFns[1].jacobian(x1n45, x2n45)
            J151 = self._shapeFns[1].jacobian(x1n51, x2n51)

            # at Gauss point 2
            N1 = self._shapeFns[2].N1
            N2 = self._shapeFns[2].N2
            
            N12 = np.array([[N1, 0, N2, 0, 0, 0, 0, 0, 0, 0],
                                [0, N1, 0, N2, 0, 0, 0, 0, 0, 0]])
            N23 = np.array([[0, 0, N1, 0, N2, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, N1, 0, N2, 0, 0, 0, 0]])
            N34 = np.array([[0, 0, 0, 0, N1, 0, N2, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, N1, 0, N2, 0, 0]])
            N45 = np.array([[0, 0, 0, 0, 0, 0, N1, 0, N2, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, N1, 0, N2]])
            N51 = np.array([[N2, 0, 0, 0, 0, 0, 0, 0, N1, 0],
                                [0, N2, 0, 0, 0, 0, 0, 0, 0, N1]])
                
            n2Mat12 = N12.transpose()
            n2Mat23 = N23.transpose()
            n2Mat34 = N34.transpose()
            n2Mat45 = N45.transpose()
            n2Mat51 = N51.transpose()

            J212 = self._shapeFns[2].jacobian(x1n12, x2n12)
            J223 = self._shapeFns[2].jacobian(x1n23, x2n23)
            J234 = self._shapeFns[2].jacobian(x1n34, x2n34)
            J245 = self._shapeFns[2].jacobian(x1n45, x2n45)
            J251 = self._shapeFns[2].jacobian(x1n51, x2n51)

            # at Gauss point 3
            N1 = self._shapeFns[3].N1
            N2 = self._shapeFns[3].N2
            
            N12 = np.array([[N1, 0, N2, 0, 0, 0, 0, 0, 0, 0],
                            [0, N1, 0, N2, 0, 0, 0, 0, 0, 0]])
            N23 = np.array([[0, 0, N1, 0, N2, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, N1, 0, N2, 0, 0, 0, 0]])
            N34 = np.array([[0, 0, 0, 0, N1, 0, N2, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, N1, 0, N2, 0, 0]])
            N45 = np.array([[0, 0, 0, 0, 0, 0, N1, 0, N2, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, N1, 0, N2]])
            N51 = np.array([[N2, 0, 0, 0, 0, 0, 0, 0, N1, 0],
                            [0, N2, 0, 0, 0, 0, 0, 0, 0, N1]])
                
            n3Mat12 = N12.transpose()
            n3Mat23 = N23.transpose()
            n3Mat34 = N34.transpose()
            n3Mat45 = N45.transpose()
            n3Mat51 = N51.transpose()

            J312 = self._shapeFns[3].jacobian(x1n12, x2n12)
            J323 = self._shapeFns[3].jacobian(x1n23, x2n23)
            J334 = self._shapeFns[3].jacobian(x1n34, x2n34)
            J345 = self._shapeFns[3].jacobian(x1n45, x2n45)
            J351 = self._shapeFns[3].jacobian(x1n51, x2n51)

            # the force vector for 3 Gauss points
            FVec = (we[0] * J112 * np.dot(n1Mat12, t12) + 
                    we[0] * J123 * np.dot(n1Mat23, t23) +
                    we[0] * J134 * np.dot(n1Mat34, t34) + 
                    we[0] * J145 * np.dot(n1Mat45, t45) +
                    we[0] * J151 * np.dot(n1Mat51, t51) + 
                    we[1] * J212 * np.dot(n2Mat12, t12) + 
                    we[1] * J223 * np.dot(n2Mat23, t23) +
                    we[1] * J234 * np.dot(n2Mat34, t34) + 
                    we[1] * J245 * np.dot(n2Mat45, t45) +
                    we[1] * J251 * np.dot(n2Mat51, t51) +
                    we[2] * J312 * np.dot(n3Mat12, t12) + 
                    we[2] * J323 * np.dot(n3Mat23, t23) +
                    we[2] * J334 * np.dot(n3Mat34, t34) + 
                    we[2] * J345 * np.dot(n3Mat45, t45) +
                    we[2] * J351 * np.dot(n3Mat51, t51))

        return FVec         
