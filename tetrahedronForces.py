#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as m
from numpy.linalg import det
from shapeFnTriangles import triaShapeFunction
from vertices import vertex


"""
Module tetrahedronForces.py provides force vector of the tetrahedra.

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
__date__ = "03-10-2020"
__update__ = "03-16-2020"
__author__ = "Shahla Zamani"
__author_email__ = "Zamani.Shahla@tamu.edu"

r"""

Change in version "1.3.0":

Created

Overview of module tetrahedronForces.py:

Class tetforce in file tetrahedronForces.py allows for the creation of objects 
that are to be used to represent the force applied to the tetrahedron. We 
compute the integral over one of the tetrahedrone’s surfaces on which
makes one triangles of a pentagon because by internal stress equilibrium, 
those portions cancel with like contributions from the neighboring elements in 
the assembled force vector of the structure. The vertices of triangle are 
located at
    vertex1: = (1, 0, 0)
    vertex2: = (0, 1, 0)
    vertex3: = (0, 0, 1)

Numerous methods have a string argument that is denoted as  state  which can
take on any of the following values:
    'c', 'curr', 'current'       gets the value for a current configuration
    'n', 'next'                  gets the value for a next configuration
    'p', 'prev', 'previous'      gets the value for a previous configuration
    'r', 'ref', 'reference'      gets the value for the reference configuration

class tetrahedron

constructor

    tr = tetrahedron(number, vertex1, vertex2, vertex3, h, triaGaussPts)
      number        immutable value that is unique to this triangle
      vertex1       unique node of the tetrahedron, an instance of class vertex
      vertex2       unique node of the tetrahedron, an instance of class vertex
      vertex3       unique node of the tetrahedron, an instance of class vertex
      h             timestep size between two successive calls to 'advance'
      triaGaussPts  number of Gauss points to be used: must be 1, 3 or 4

methods

    s = tr.toString()
        returns string representation for triangle in configuration 'state'

    n = tr.number()
        returns the unique number affiated with this triangle

    n1, n2, n3 = tr.vertexNumbers()
        returns unique numbers associated with the vertices of a triangle

    truth = tr.hasVertex(number)
        returns True if one of the four vertices has this vertex number

    v = tr.getVertex(number)
        returns a vertex; typically called from within a t.hasVertex if clause

    n = tr.gaussPoints()
        returns the number of Gauss points assigned to the tetrahedron

    tr.update()
        assigns new coordinate values to the tetrahedorn for its next location
        and updates all affected fields.  To be called after all vertices have
        had their coordinates updated.  This may be called multiple times
        before freezing it with a call to advance

    tr.advance()
        assigns fields belonging to the current location into their cournter-
        parts in the previous location, and then it assigns their next values
        into the current location, thereby freezing the location of the present
        next-location in preparation to advance to the next step along a
        solution path
    
    [nx, ny, nz] = p.normal(state)
        returns the unit normal to this triangle in configuration 'state'

    sf = tr.shapeFunction(gaussPt):
        returns the shape function associated with the specified Gauss point

    fFn = tr.tetrahedronforcingFunction()
        returns a vector for the forcing function on the right hand side.

"""

class tetforce(object):

    def __init__(self, number, vertex1, vertex2, vertex3, h, triaGaussPts):
        # verify the input
        self._number = int(number)
        # place the vertices into their data structure
        if not isinstance(vertex1, vertex):
            raise RuntimeError("vertex1 must be an instance of type vertex.")
        if not isinstance(vertex2, vertex):
            raise RuntimeError("vertex2 must be an instance of type vertex.")
        if not isinstance(vertex3, vertex):
            raise RuntimeError("vertex3 must be an instance of type vertex.")
        self._vertex = {
            1: vertex1,
            2: vertex2,
            3: vertex3
        }
        self._setOfVertices = {
            vertex1.number(),
            vertex2.number(),
            vertex3.number()
        }
        # check the stepsize
        if h > np.finfo(float).eps:
            self._h = float(h)
        else:
            raise RuntimeError("The stepsize sent to the triangle " +
                               "constructor wasn't positive.")
        # check the number of Gauss points to use
        if triaGaussPts == 1 or triaGaussPts == 3 or triaGaussPts == 4:
            self._triaGaussPts = triaGaussPts
        else:
            raise RuntimeError('{} Gauss points were '.format(triaGaussPts) +
                               'specified in the triangle constructor; ' +
                               'it must be 1, 3 or 4.')

        # establish the shape functions located at the triangle Gauss points 
        # (xi, eta)
        if triaGaussPts == 1:
            xi = 1/3
            eta = 1/3
            sf11 = triaShapeFunction(xi, eta)

            self._shapeFns = {
                11: sf11
            }
        elif triaGaussPts == 3:            
            xi1 = 2/3
            eta1 = 1/6
            sf11 = triaShapeFunction(xi1, eta1)
            
            xi1 = 2/3
            eta2 = 1/6
            sf12 = triaShapeFunction(xi1, eta2)

            xi1 = 2/3
            eta3 = 2/3
            sf13 = triaShapeFunction(xi1, eta3)
                       
            
            xi2 = 1/6
            eta1 = 1/6
            sf21 = triaShapeFunction(xi2, eta1)
            
            xi2 = 1/6
            eta2 = 1/6
            sf22 = triaShapeFunction(xi2, eta2)

            xi2 = 1/6
            eta3 = 2/3
            sf23 = triaShapeFunction(xi2, eta3)
            
            
            xi3 = 1/6
            eta1 = 1/6
            sf31 = triaShapeFunction(xi3, eta1)

            xi3 = 1/6
            eta2 = 1/6
            sf32 = triaShapeFunction(xi3, eta2)
            
            xi3 = 1/6
            eta3 = 2/3
            sf33 = triaShapeFunction(xi3, eta3)


            self._shapeFns = {
                11: sf11,
                12: sf12,
                13: sf13,
                21: sf21,
                22: sf22,
                23: sf23,
                31: sf31,
                32: sf32,
                33: sf33
            }
        else:  # triaGaussPts = 4
            xi1 = 1/3
            eta1 = 1/3
            sf11 = triaShapeFunction(xi1, eta1)

            xi1 = 1/3
            eta2 = 1/5
            sf12 = triaShapeFunction(xi1, eta2)

            xi1 = 1/3
            eta3 = 1/5
            sf13 = triaShapeFunction(xi1, eta3)

            xi1 = 1/3
            eta4 = 3/5
            sf14 = triaShapeFunction(xi1, eta4)
            
            
            xi2 = 3/5
            eta1 = 1/3
            sf21 = triaShapeFunction(xi2, eta1)

            xi2 = 3/5
            eta2 = 1/5
            sf22 = triaShapeFunction(xi2, eta2)

            xi2 = 3/5
            eta3 = 1/5
            sf23 = triaShapeFunction(xi2, eta3)

            xi2 = 3/5
            eta4 = 3/5
            sf24 = triaShapeFunction(xi2, eta4)

                       
            xi3 = 1/5
            eta1 = 1/3
            sf31 = triaShapeFunction(xi3, eta1)

            xi3 = 1/5
            eta2 = 1/5
            sf32 = triaShapeFunction(xi3, eta2)

            xi3 = 1/5
            eta3 = 1/5
            sf33 = triaShapeFunction(xi3, eta3)

            xi3 = 1/5
            eta4 = 3/5
            sf34 = triaShapeFunction(xi3, eta4)
            
                        
            xi4 = 1/5
            eta1 = 1/3
            sf41 = triaShapeFunction(xi4, eta1)

            xi4 = 1/5
            eta2 = 1/5
            sf42 = triaShapeFunction(xi4, eta2)

            xi4 = 1/5
            eta3 = 1/5
            sf43 = triaShapeFunction(xi4, eta3)

            xi4 = 1/5
            eta4 = 3/5
            sf44 = triaShapeFunction(xi4, eta4)
 
            self._shapeFns = {
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
            
        # get the reference coordinates for the vetices of the triangle
        x10, y10, z10 = self._vertex[1].coordinates('ref')
        x20, y20, z20 = self._vertex[2].coordinates('ref')
        x30, y30, z30 = self._vertex[3].coordinates('ref')            

        # normal vector to the triangle
        # vector v = P2 - P1 
        vx = x20 - x10
        vy = y20 - y10
        vz = z20 - z10
        
        # vector v = P3 - P1 
        wx = x30 - x10
        wy = y30 - y10
        wz = z30 - z10
        
        # create the normal vector
        x = (vy * wz) - (vz * wy)
        y = (vz * wx) - (vx * wz)
        z = (vx * wy) - (vy * wx)
        
        # create the unit normal vector       
        mag = m.sqrt(x * x + y * y + z * z)
        nx = x / mag
        ny = y / mag
        nz = z / mag        
        
        # initialize the normal vector in the refference coordinate system
        self._normalXr = nx
        self._normalYr = ny
        self._normalZr = nz
        
        # initialize the normal vector in the previous, current, and next 
        # coordinate system
        self._normalXp = self._normalXr 
        self._normalYp = self._normalYr 
        self._normalZp = self._normalZr
        
        self._normalXc = self._normalXr 
        self._normalYc = self._normalYr 
        self._normalZc = self._normalZr
        
        self._normalXn = self._normalXr 
        self._normalYn = self._normalYr 
        self._normalZn = self._normalZr
        
        # initialize current vertice coordinates 
        self._x1 = x10
        self._y1 = y10
        self._z1 = z10
        self._x2 = x20
        self._y2 = y20
        self._z2 = z20
        self._x3 = x30
        self._y3 = y30
        self._z3 = z30       
        
    def toString(self, state):
        if self._number < 10:
            s = 'triangle[0'
        else:
            s = 'trinagle['
        s = s + str(self._number)
        s = s + '] has vertices: \n'
        if isinstance(state, str):
            s = s + '   1: ' + self._vertex[1].toString(state) + '\n'
            s = s + '   2: ' + self._vertex[2].toString(state) + '\n'
            s = s + '   3: ' + self._vertex[3].toString(state) 
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to triangle.toString.")
        return s            
            
    def number(self):
        return self._number

    def vertexNumbers(self):
        numbers = sorted(self._setOfVertices)
        return numbers[0], numbers[1], numbers[2]

    def hasVertex(self, number):
        return number in self._setOfVertices

    def getVertex(self, number):
        if self._vertex[1].number() == number:
            return self._vertex[1]
        elif self._vertex[2].number() == number:
            return self._vertex[2]
        elif self._vertex[3].number() == number:
            return self._vertex[3]
        else:
            raise RuntimeError('The requested vertex {} is '.format(number) +
                               'not in triangle {}.'.format(self._number))

    def gaussPoints(self):
        return self._triaGaussPts
                        
    def update(self):
        # computes the fields positioned at the next time step

        # get the updated coordinates for the vetices of the triangle
        self._x1, self._y1, self._z1 = self._vertex[1].coordinates('next')
        self._x2, self._y2, self._z2 = self._vertex[2].coordinates('next')
        self._x3, self._y3, self._z3 = self._vertex[3].coordinates('next')
                
        # normal vector to the triangle
        # vector v = P2 - P1 
        vx = self._x2 - self._x1
        vy = self._y2 - self._y1
        vz = self._z2 - self._z1
        
        # vector v = P3 - P1 
        wx = self._x3 - self._x1
        wy = self._y3 - self._y1
        wz = self._z3 - self._z1
        
        # create the normal vector
        x = (vy * wz) - (vz * wy)
        y = (vz * wx) - (vx * wz)
        z = (vx * wy) - (vy * wx)
        
        # create the unit normal vector       
        mag = m.sqrt(x * x + y * y + z * z)
        nx = x / mag
        ny = y / mag
        nz = z / mag        
        
        # initialize the normal vector
        self._normalXn = nx
        self._normalYn = ny
        self._normalZn = nz

        return  # nothing   
         
            
    def advance(self):
        # advance the normal vector
        self._normalXp = self._normalXc 
        self._normalYp = self._normalYc 
        self._normalZp = self._normalZc
        
        self._normalXc = self._normalXn 
        self._normalYc = self._normalYn 
        self._normalZc = self._normalZn
            
        
    def normal(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                nx = self._normalXc 
                ny = self._normalYc
                nz = self._normalZc
            elif state == 'n' or state == 'next':
                nx = self._normalXn 
                ny = self._normalYn
                nz = self._normalZn
            elif state == 'p' or state == 'prev' or state == 'previous':
                nx = self._normalXp 
                ny = self._normalYp
                nz = self._normalZp
            elif state == 'r' or state == 'ref' or state == 'reference':
                nx = self._normalXr 
                ny = self._normalYr
                nz = self._normalZr
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to triangle.normal.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to triangle.normal.")
        return np.array([nx, ny, nz])
            
    
    def shapeFunction(self, gaussPt):
        if (gaussPt < 1) or (gaussPt > self._gaussPts):
            raise RuntimeError("The gaussPt must be in the range of " +
                               "[1, {}] in call to ".format(self._gaussPts) +
                               "triangle.shapeFunction and " +
                               " you sent {}.".format(gaussPt))
            sf = self._shapeFns[gaussPt]
        return sf            
            
    def tetrahedronforcingFunction(self, cauchyStress, state):
        nx, ny, nz = self.normal(state)
        
        # create the normal vector
        n = np.array([nx, ny, nz])

        # create the traction vector
        t = np.dot(cauchyStress, np.transpose(n))
        
        # assign coordinates at the vertices in the reference configuration
        x1 = (self._x1, self._y1)
        x2 = (self._x2, self._y2)
        x3 = (self._x3, self._y3)

        # determine the force vector
        if self._triaGaussPts == 1:
            # 'natural' weight of the element
            wgt = 1.0 
            w = np.array([wgt])
            
            jacob11 = self._shapeFns[11].jacobian(x1, x2, x3)
            
            # determinant of the Jacobian matrix
            detJ = det(jacob11)
                        
            nMat1 = np.transpose(self._shapeFns[11].Nmatx)
                         
            # the force vector for 1 Gauss point
            FVec = detJ * w[0] * w[0] * np.dot(nMat1, np.transpose(t))

        elif self._triaGaussPts == 3:
            # 'natural' weights of the element
            wgt = 1.0 / 3.0
            w = np.array([wgt, wgt, wgt])

            jacob11 = self._shapeFns[11].jacobian(x1, x2, x3)
            jacob12 = self._shapeFns[12].jacobian(x1, x2, x3)
            jacob13 = self._shapeFns[13].jacobian(x1, x2, x3)
    
            jacob21 = self._shapeFns[21].jacobian(x1, x2, x3)
            jacob22 = self._shapeFns[22].jacobian(x1, x2, x3)
            jacob23 = self._shapeFns[23].jacobian(x1, x2, x3)
            
            jacob31 = self._shapeFns[31].jacobian(x1, x2, x3)
            jacob32 = self._shapeFns[32].jacobian(x1, x2, x3)
            jacob33 = self._shapeFns[33].jacobian(x1, x2, x3)
            
            # determinant of the Jacobian matrix
            detJ11 = det(jacob11)
            detJ12 = det(jacob12)
            detJ13 = det(jacob13)
            
            detJ21 = det(jacob21)
            detJ22 = det(jacob22)
            detJ23 = det(jacob23)
            
            detJ31 = det(jacob31)
            detJ32 = det(jacob32)
            detJ33 = det(jacob33)

            nMat11 = np.transpose(self._shapeFns[11].Nmatx)
            nMat12 = np.transpose(self._shapeFns[12].Nmatx)
            nMat13 = np.transpose(self._shapeFns[13].Nmatx)
            nMat21 = np.transpose(self._shapeFns[21].Nmatx)
            nMat22 = np.transpose(self._shapeFns[22].Nmatx)
            nMat23 = np.transpose(self._shapeFns[23].Nmatx)
            nMat31 = np.transpose(self._shapeFns[31].Nmatx)
            nMat32 = np.transpose(self._shapeFns[32].Nmatx)
            nMat33 = np.transpose(self._shapeFns[33].Nmatx)
            
            # the force vector for 3 Gauss points
            FVec = (detJ11 * w[0] * w[0] * np.dot(nMat11, np.transpose(t)) + 
                    detJ12 * w[0] * w[1] * np.dot(nMat12, np.transpose(t)) + 
                    detJ13 * w[0] * w[2] * np.dot(nMat13, np.transpose(t)) +
                    detJ21 * w[1] * w[0] * np.dot(nMat21, np.transpose(t)) + 
                    detJ22 * w[1] * w[1] * np.dot(nMat22, np.transpose(t)) + 
                    detJ23 * w[1] * w[2] * np.dot(nMat23, np.transpose(t)) +
                    detJ31 * w[2] * w[0] * np.dot(nMat31, np.transpose(t)) + 
                    detJ32 * w[2] * w[1] * np.dot(nMat32, np.transpose(t)) + 
                    detJ33 * w[2] * w[2] * np.dot(nMat33, np.transpose(t)))

        else:  # triaGaussPts = 4
            # 'natural' weights of the element
            wgt1 = -27.0 / 48.0
            wgt2 = 25.0 / 48.0
            w = np.array([wgt1, wgt2, wgt2, wgt2])

            jacob11 = self._shapeFns[11].jacobian(x1, x2, x3)
            jacob12 = self._shapeFns[12].jacobian(x1, x2, x3)
            jacob13 = self._shapeFns[13].jacobian(x1, x2, x3)
            jacob14 = self._shapeFns[14].jacobian(x1, x2, x3)
            
            jacob21 = self._shapeFns[21].jacobian(x1, x2, x3)
            jacob22 = self._shapeFns[22].jacobian(x1, x2, x3)
            jacob23 = self._shapeFns[23].jacobian(x1, x2, x3)
            jacob24 = self._shapeFns[24].jacobian(x1, x2, x3)
            
            jacob31 = self._shapeFns[31].jacobian(x1, x2, x3)
            jacob32 = self._shapeFns[32].jacobian(x1, x2, x3)
            jacob33 = self._shapeFns[33].jacobian(x1, x2, x3)
            jacob34 = self._shapeFns[34].jacobian(x1, x2, x3)
            
            jacob41 = self._shapeFns[41].jacobian(x1, x2, x3)
            jacob42 = self._shapeFns[42].jacobian(x1, x2, x3)
            jacob43 = self._shapeFns[43].jacobian(x1, x2, x3)
            jacob44 = self._shapeFns[44].jacobian(x1, x2, x3)
            
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
            
            nMat11 = np.transpose(self._shapeFns[11].Nmatx)
            nMat12 = np.transpose(self._shapeFns[12].Nmatx)
            nMat13 = np.transpose(self._shapeFns[13].Nmatx)
            nMat14 = np.transpose(self._shapeFns[14].Nmatx)
            nMat21 = np.transpose(self._shapeFns[21].Nmatx)
            nMat22 = np.transpose(self._shapeFns[22].Nmatx)
            nMat23 = np.transpose(self._shapeFns[23].Nmatx)
            nMat24 = np.transpose(self._shapeFns[24].Nmatx)
            nMat31 = np.transpose(self._shapeFns[31].Nmatx)
            nMat32 = np.transpose(self._shapeFns[32].Nmatx)
            nMat33 = np.transpose(self._shapeFns[33].Nmatx)
            nMat34 = np.transpose(self._shapeFns[34].Nmatx)
            nMat41 = np.transpose(self._shapeFns[41].Nmatx)
            nMat42 = np.transpose(self._shapeFns[42].Nmatx)
            nMat43 = np.transpose(self._shapeFns[43].Nmatx)
            nMat44 = np.transpose(self._shapeFns[44].Nmatx)
            
            # the force vector for 4 Gauss points
            FVec = (detJ11 * w[0] * w[0] * np.dot(nMat11, np.transpose(t)) + 
                    detJ12 * w[0] * w[1] * np.dot(nMat12, np.transpose(t)) + 
                    detJ13 * w[0] * w[2] * np.dot(nMat13, np.transpose(t)) +
                    detJ14 * w[0] * w[3] * np.dot(nMat14, np.transpose(t)) +
                    detJ21 * w[1] * w[0] * np.dot(nMat21, np.transpose(t)) + 
                    detJ22 * w[1] * w[1] * np.dot(nMat22, np.transpose(t)) + 
                    detJ23 * w[1] * w[2] * np.dot(nMat23, np.transpose(t)) +
                    detJ24 * w[1] * w[3] * np.dot(nMat24, np.transpose(t)) +
                    detJ31 * w[2] * w[0] * np.dot(nMat31, np.transpose(t)) + 
                    detJ32 * w[2] * w[1] * np.dot(nMat32, np.transpose(t)) + 
                    detJ33 * w[2] * w[2] * np.dot(nMat33, np.transpose(t)) +
                    detJ34 * w[2] * w[3] * np.dot(nMat34, np.transpose(t)) + 
                    detJ41 * w[3] * w[0] * np.dot(nMat41, np.transpose(t)) + 
                    detJ42 * w[3] * w[1] * np.dot(nMat42, np.transpose(t)) + 
                    detJ43 * w[3] * w[2] * np.dot(nMat43, np.transpose(t)) +
                    detJ44 * w[3] * w[3] * np.dot(nMat44, np.transpose(t)))

        return FVec        
        