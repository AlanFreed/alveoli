#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ceSacs import controlSac, ceSac
import meanProperties as mp
import math as m
from pivotIncomingF import Pivot
import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
from shapeFnTetrahedra import ShapeFunction as tetShapeFunction
from shapeFnTriangles import ShapeFunction as triShapeFunction
from vertices import Vertex
from gaussQuadTetrahedra import GaussQuadrature as tetGaussQuadrature
from gaussQuadTriangles import GaussQuadrature as triGaussQuadrature

"""
Module tetrahedra.py provides geometric information about irregular tetrahedra.

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
__date__ = "09-25-2019"
__update__ = "12-06-2020"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

r"""

Change in version "1.3.0":

Created

Overview of module tetrahedra.py:

Class tetrahedron in file tetrahedra.py allows for the creation of objects that
are to be used to represent irregular tetrahedra.  Its vertices are located at
    Vertex1: (xi, eta, zeta) = (0, 0, 0)
    Vertex2: (xi, eta, zeta) = (1, 0, 0)
    Vertex3: (xi, eta, zeta) = (0, 1, 0)
    Vertex4: (xi, eta, zeta) = (0, 0, 1)
wherein xi, eta, and zeta denote the element's natural coordinates.  The volume
of such a tetrahedron is 1/6.

Numerous methods have a string argument that is denoted as  state  which can
take on any of the following values:
    'c', 'curr', 'current'       gets the value for a current configuration
    'n', 'next'                  gets the value for a next configuration
    'p', 'prev', 'previous'      gets the value for a previous configuration
    'r', 'ref', 'reference'      gets the value for the reference configuration

class tetrahedron

constructor

    t = tetrahedron(number, Vertex1, Vertex2, Vertex3, Vertex4, h, gaussPts)
        number    immutable value that is unique to this tetrahedron
        Vertex1   unique node of the tetrahedron, an instance of class Vertex
        Vertex2   unique node of the tetrahedron, an instance of class Vertex
        Vertex3   unique node of the tetrahedron, an instance of class Vertex
        Vertex4   unique node of the tetrahedron, an instance of class Vertex
        h         timestep size between two successive calls to 'advance'
        gaussPts  number of Gauss points to be used: must be 1, 4 or 5

methods

    s = t.toString()
        returns string representation for tetrahedron in configuration 'state'

    n = t.number()
        returns the unique number affiated with this tetrahedron

    n1, n2, n3, n4 = t.VertexNumbers()
        returns unique numbers associated with the vertices of a tetrahedron

    truth = t.hasVertex(number)
        returns True if one of the four vertices has this Vertex number

    v = t.getVertex(number)
        returns a Vertex; typically called from within a t.hasVertex if clause

    n = t.gaussPoints()
        returns the number of Gauss points assigned to the tetrahedron

    t.update()
        assigns new coordinate values to the tetrahedorn for its next location
        and updates all affected fields.  To be called after all vertices have
        had their coordinates updated.  This may be called multiple times
        before freezing it with a call to advance

    t.advance(reindex)
       input
            reindex is an instance of Pivot object from module pivotIncomingF      
        assigns fields belonging to the current location into their cournter-
        parts in the previous location, and then it assigns their next values
        into the current location, thereby freezing the location of the present
        next-location in preparation to advance to the next step along a
        solution path

    [nx, ny, nz] = p.normal(state)
        returns the unit normal to this triangle in configuration 'state'

    Geometric fields associated with a tetrahedral volume in 3 space

    a = t.volume(state)
        returns the volume of the tetrahedron in configuration 'state'

    vLambda = t.volumetricStretch(state)
        returns the cube root of: volume(state) divided by its reference volume

    vStrain = t.volumetricStrain(state)
        returns the logarithm of volumetric stretch evaluated at 'state'

    dvStrain = t.dVolumetricStrain(state)
        returns the time rate of change in volumetric strain at 'state'

    Kinematic fields associated with the centroid of a tetrahedron in 3 space

    [cx, cy, cz] = t.centroid(state)
        returns centroid of this tetrahedron in configuration 'state'

    [ux, uy, uz] = t.centroidDisplacement(reindex, state)
        returns the displacement at the centroid in configuration 'state'

    [vx, vy, vz] = t.centroidVelocity(reindex, state)
        returns the velocity at the centroid in configuration 'state'

    [ax, ay, az] = t.centroidAcceleration(reindex, state)
        returns the acceleration at the centroid in configuration 'state'

    Dmtx1 = sf.dDisplacement1(reindex)
      input
        reindex is an instance of Pivot object from module pivotIncomingF      
      output
        Dmtx1 is change in displacement ( dA1 = L1 * D1 ) in the contribution to the 
        nonlinear strain 

    Dmtx2 = sf.dDisplacement2(reindex)
      input
        reindex is an instance of Pivot object from module pivotIncomingF      
      output
        Dmtx2 is change in displacement ( dA2 = L2 * D2 ) in the contribution to the 
        nonlinear strain 

    Dmtx3 = sf.dDisplacement3(reindex)
      input
        reindex is an instance of Pivot object from module pivotIncomingF      
      output
        Dmtx3 is change in displacement ( dA3 = L3 * D3 ) in the contribution to the 
        nonlinear strain 
        
    Dmtx4 = sf.dDisplacement4(reindex)
      input
        reindex is an instance of Pivot object from module pivotIncomingF      
      output
        Dmtx4 is change in displacement ( dA4 = L4 * D4 ) in the contribution to the 
        nonlinear strain         
        
    Dmtx5 = sf.dDisplacement5(reindex)
      input
        reindex is an instance of Pivot object from module pivotIncomingF      
      output
        Dmtx5 is change in displacement ( dA5 = L5 * D5 ) in the contribution to the 
        nonlinear strain 
            
    The fundamental fields of kinematics

    gMtx = t.G(gaussPt, state)
        returns 3x3 matrix describing the displacement gradient for the
        tetrahedron at 'gaussPt' in configuration 'state'

    fMtx = t.F(gaussPt, state)
        returns 3x3 matrix describing the deformation gradient for the
        tetrahedron at 'gaussPt' in configuration 'state'

    lMtx = t.L(gaussPt, state)
        returns the velocity gradient at the specified Gauss point for the
        specified configuration

    Fields needed to construct finite element representations.  The mass and
    stiffness matrices are 12x12

    tesf = t.tetshapeFunction(gaussPt):
        returns the shape function associated with the specified Gauss point 
        for tetrahedron

    trsf = t.trishapeFunction(gaussPt):
        returns the shape function associated with the specified Gauss point
        for triangle

    massM = t.massMatrix()
        returns an average of the lumped and consistent mass matrices (ensures
        the mass matrix is not singular) of dimension 12x12 for the chosen
        number of Gauss points for a tetrahedron whose mass density, rho,
        is specified.

    cMtx = t.tangentStiffnessMtxC()
        returns a tangent stiffness matrix for the chosen number of Gauss
        points.
        
    kMtx = t.secantStiffnessMtxK(reindex)
        returns a secant stiffness matrix for the chosen number of Gauss
        points.

    fFn = t.forcingFunction()
        returns a vector for the forcing function on the right hand side. 
        We compute the integral over one of the tetrahedrone’s surfaces on 
        which makes one triangles of a pentagon because by internal stress 
        equilibrium, those portions cancel with like contributions from the 
        neighboring elements in the assembled force vector of the structure.

Reference
    1) Guido Dhondt, "The Finite Element Method for Three-dimensional
       Thermomechanical Applications", John Wiley & Sons Ltd, 2004.
    2) Colins, K. D. "Cayley-Menger Determinant." From MathWorld--A Wolfram Web
       Resource, created by Eric W. Weisstein. http://mathworld.wolfram.com/
       Cayley-MengerDeterminant.html
"""


class tetrahedron(object):

    def __init__(self, number, Vertex1, Vertex2, Vertex3, Vertex4, h):
        # verify the input
        self._number = int(number)
        # place the vertices into their data structure
        if not isinstance(Vertex1, Vertex):
            raise RuntimeError("Vertex1 must be an instance of type Vertex.")
        if not isinstance(Vertex2, Vertex):
            raise RuntimeError("Vertex2 must be an instance of type Vertex.")
        if not isinstance(Vertex3, Vertex):
            raise RuntimeError("Vertex3 must be an instance of type Vertex.")
        if not isinstance(Vertex4, Vertex):
            raise RuntimeError("Vertex4 must be an instance of type Vertex.")
        self._vertex = {
            1: Vertex1,
            2: Vertex2,
            3: Vertex3,
            4: Vertex4
        }
        self._setOfVertices = {
            Vertex1.number(),
            Vertex2.number(),
            Vertex3.number(),
            Vertex4.number()
        }
        # check the stepsize
        if h > np.finfo(float).eps:
            self._h = float(h)
        else:
            raise RuntimeError("The stepsize sent to the tetrahedron " +
                               "constructor wasn't positive.")
        
        # assign the Gauss quadrature rule to be used for triangle
        self._trgq = triGaussQuadrature()
        
        
        # assign the Gauss quadrature rule to be used for tetrahedron
        self._tegq = tetGaussQuadrature()

        # create a shape function for the centroid of the tetrahedron
        self._centroidSF = tetShapeFunction((0.25, 0.25, 0.25))

        # establish the shape functions located at the various Gauss points
        # for tetrahedron
        tetAtGaussPt = 1
        tesf1 = tetShapeFunction(self._tegq.coordinates(tetAtGaussPt))
        tetAtGaussPt = 2
        tesf2 = tetShapeFunction(self._tegq.coordinates(tetAtGaussPt))
        tetAtGaussPt = 3
        tesf3 = tetShapeFunction(self._tegq.coordinates(tetAtGaussPt))
        tetAtGaussPt = 4
        tesf4 = tetShapeFunction(self._tegq.coordinates(tetAtGaussPt))
        self._tetShapeFns = {
            1: tesf1,
            2: tesf2,
            3: tesf3,
            4: tesf4       
        }

        # establish the shape functions located at the various Gauss points
        # for triangle
        triAtGaussPt = 1
        trsf1 = triShapeFunction(self._trgq.coordinates(triAtGaussPt))
        triAtGaussPt = 2
        trsf2 = triShapeFunction(self._trgq.coordinates(triAtGaussPt))
        triAtGaussPt = 3
        trsf3 = triShapeFunction(self._trgq.coordinates(triAtGaussPt))
        self._triShapeFns = {
            1: trsf1,
            2: trsf2,
            3: trsf3       
        }        
        
        # create matrices for tetrahedron at its Gauss points via dictionaries
        # p implies previous, c implies current, n implies next

        # displacement gradients located at the Gauss points of tetrahedron
        self._G0 = {
            1: np.zeros((3, 3), dtype=float),
            2: np.zeros((3, 3), dtype=float),
            3: np.zeros((3, 3), dtype=float),
            4: np.zeros((3, 3), dtype=float)
        }
        self._Gp = {
            1: np.zeros((3, 3), dtype=float),
            2: np.zeros((3, 3), dtype=float),
            3: np.zeros((3, 3), dtype=float),
            4: np.zeros((3, 3), dtype=float)
        }
        self._Gc = {
            1: np.zeros((3, 3), dtype=float),
            2: np.zeros((3, 3), dtype=float),
            3: np.zeros((3, 3), dtype=float),
            4: np.zeros((3, 3), dtype=float)
        }
        self._Gn = {
            1: np.zeros((3, 3), dtype=float),
            2: np.zeros((3, 3), dtype=float),
            3: np.zeros((3, 3), dtype=float),
            4: np.zeros((3, 3), dtype=float)
        }
        # deformation gradients located at the Gauss points of tetrahedron
        self._F0 = {
            1: np.identity(3, dtype=float),
            2: np.identity(3, dtype=float),
            3: np.identity(3, dtype=float),
            4: np.identity(3, dtype=float)
        }
        self._Fp = {
            1: np.identity(3, dtype=float),
            2: np.identity(3, dtype=float),
            3: np.identity(3, dtype=float),
            4: np.identity(3, dtype=float)
        }
        self._Fc = {
            1: np.identity(3, dtype=float),
            2: np.identity(3, dtype=float),
            3: np.identity(3, dtype=float),
            4: np.identity(3, dtype=float)
        }
        self._Fn = {
            1: np.identity(3, dtype=float),
            2: np.identity(3, dtype=float),
            3: np.identity(3, dtype=float),
            4: np.identity(3, dtype=float)
        }

        # get the reference coordinates for the vetices of the tetrahedron
        x1 = self._vertex[1].coordinates('ref')
        x2 = self._vertex[2].coordinates('ref')
        x3 = self._vertex[3].coordinates('ref')
        x4 = self._vertex[4].coordinates('ref')

        # base vector 1: connects the centroid of pentagon with one of the 
        # pentagon’s vertices
        x = x2[0] - x1[0]
        y = x2[1] - x1[1]
        z = x2[2] - x1[2]
        mag = m.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 3: connects the centroid of the dodecahedron with the 
        # centroid of a pentagon
        x = x4[0] - x1[0]
        y = x4[1] - x1[1]
        z = x4[2] - x1[2]
        mag = m.sqrt(x * x + y * y + z * z)
        n3x = x / mag
        n3y = y / mag
        n3z = z / mag

        # base vector 2 is obtained through the cross product
        n2x = n3y * n1z - n3z * n1y
        n2y = n3z * n1x - n3x * n1z
        n2z = n3x * n1y - n3y * n1x

        # create rotation matrix from dodecahedral to tetrahedron coordinates
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


        # determine vertice coordinates 
        self._v1x0 = n1x * x1[0] + n1y * x1[1] + n1z * x1[2]
        self._v1y0 = n2x * x1[0] + n2y * x1[1] + n2z * x1[2]
        self._v2x0 = n1x * x2[0] + n1y * x2[1] + n1z * x2[2]
        self._v2y0 = n2x * x2[0] + n2y * x2[1] + n2z * x2[2]
        self._v3x0 = n1x * x3[0] + n1y * x3[1] + n1z * x3[2]
        self._v3y0 = n2x * x3[0] + n2y * x3[1] + n2z * x3[2]
        self._v4x0 = n1x * x4[0] + n1y * x4[1] + n1z * x4[2]
        self._v4y0 = n2x * x4[0] + n2y * x4[1] + n2z * x4[2]

        # z offsets 
        self._v1z0 = n3x * x1[0] + n3y * x1[1] + n3z * x1[2]
        self._v2z0 = n3x * x2[0] + n3y * x2[1] + n3z * x2[2]
        self._v3z0 = n3x * x3[0] + n3y * x3[1] + n3z * x3[2]
        self._v4z0 = n3x * x4[0] + n3y * x4[1] + n3z * x4[2]
        
        
        # initialize current vertice coordinates
        self._v1x = self._v1x0
        self._v1y = self._v1y0
        self._v1z = self._v1z0
        self._v2x = self._v2x0
        self._v2y = self._v2y0
        self._v2z = self._v2z0
        self._v3x = self._v3x0
        self._v3y = self._v3y0
        self._v3z = self._v3z0
        self._v4x = self._v4x0
        self._v4y = self._v4y0
        self._v4z = self._v4z0


        self._vz0 = (self._v1z + self._v2z + self._v3z + self._v4z) / 4.0

        # determine the volume of this tetrahedron in its reference state
        self._V0 = self._volTet(self._v1x0, self._v1y0, self._v1z0,
                                self._v2x0, self._v2y0, self._v2z0,
                                self._v3x0, self._v3y0, self._v3z0,
                                self._v4x0, self._v4y0, self._v4z0)
        self._Vp = self._V0
        self._Vc = self._V0
        self._Vn = self._V0

        # establish the centroidal location of this tetrahedron
        self._cx0 = self._centroidSF.interpolate(self._v1x0, self._v2x0,
                                                 self._v3x0, self._v4x0)
        self._cy0 = self._centroidSF.interpolate(self._v1y0, self._v2y0,
                                                 self._v3y0, self._v4y0)
        self._cz0 = self._centroidSF.interpolate(self._v1z0, self._v2z0,
                                                 self._v3z0, self._v4z0)
      
        # rotate this centroid back into the reference coordinate system
        self._centroidX0 = n1x * self._cx0 + n2x * self._cy0 + n3x * self._cz0
        self._centroidY0 = n1y * self._cx0 + n2y * self._cy0 + n3y * self._cz0
        self._centroidZ0 = n1z * self._cx0 + n2z * self._cy0 + n3z * self._cz0

        
        self._centroidXp = self._centroidX0
        self._centroidXc = self._centroidX0
        self._centroidXn = self._centroidX0
        self._centroidYp = self._centroidY0
        self._centroidYc = self._centroidY0
        self._centroidYn = self._centroidY0
        self._centroidZp = self._centroidZ0
        self._centroidZc = self._centroidZ0
        self._centroidZn = self._centroidZ0

        # rotation matrices
        self._Pp3D = np.zeros((3, 3), dtype=float)
        self._Pc3D = np.zeros((3, 3), dtype=float)
        self._Pn3D = np.zeros((3, 3), dtype=float)
        self._Pp3D[:, :] = self._Pr3D[:, :]
        self._Pc3D[:, :] = self._Pr3D[:, :]
        self._Pn3D[:, :] = self._Pr3D[:, :]
        
        self._rho = mp.rhoAir()

        # get the material properties
        p_0, v0v = mp.septalSac()
        self._p_0 = p_0        
        nbrVars = 7   # for a chord they are: temperature and length
        respVars = 7
        T0 = 37.0     # body temperature in centigrade
        # thermodynamic strains (thermal and mechanical) are 0 at reference
        eVec0 = np.zeros((nbrVars,), dtype=float)
        # physical variables have reference values of
        xVec0 = np.zeros((nbrVars,), dtype=float)
        # vector of thermodynamic response variables
        yVec0 = np.zeros((respVars,), dtype=float)


        yVec0[1] = -3 * self._p_0   # pressure                   'pi'      (barye)
        yVec0[2] = 0.0              # normal stress difference   'sigma1'  (barye)
        yVec0[3] = 0.0              # normal stress difference   'sigma2'  (barye)
        yVec0[4] = 0.0              # shear stress               'tau1'    (barye)
        yVec0[5] = 0.0              # shear stress               'tau2'    (barye)
        yVec0[6] = 0.0              # shear stress               'tau3'

        xVec0[0] = T0    # temperature                          'T'   (centigrade)
        xVec0[1] = 1.0   # elongation in 1 direction            'a'   (dimensionless)
        xVec0[2] = 1.0   # elongation in 2 direction            'b'   (dimensionless)
        xVec0[3] = 1.0   # elongation in 3 direction            'c'   (dimensionless)
        xVec0[4] = 0.0   # magnitude of shear in the 23 plane   'alp' (dimensionless)
        xVec0[5] = 0.0   # magnitude of shear in the 13 plane   'bet' (dimensionless)
        xVec0[6] = 0.0   # magnitude of shear in the 12 plane   'gam'
        
        self._response = {
            1:ceSac(),
            2:ceSac(),
            3:ceSac(),
            4:ceSac()
            }
        
        self._Ms = {
            1: self._response[1].secMod(eVec0, xVec0, yVec0),
            2: self._response[2].secMod(eVec0, xVec0, yVec0),
            3: self._response[3].secMod(eVec0, xVec0, yVec0),
            4: self._response[4].secMod(eVec0, xVec0, yVec0),
        }
        
        self._Mt = {
            1: self._response[1].tanMod(eVec0, xVec0, yVec0),
            2: self._response[2].tanMod(eVec0, xVec0, yVec0),
            3: self._response[3].tanMod(eVec0, xVec0, yVec0),
            4: self._response[4].tanMod(eVec0, xVec0, yVec0),
        }       

        self.Ss = {
            1: self._response[1].stressMtx(),
            2: self._response[2].stressMtx(),
            3: self._response[3].stressMtx(),
            4: self._response[4].stressMtx()
        }  

        self.T = {
            1: self._response[1].intensiveStressVec(),
            2: self._response[2].intensiveStressVec(),
            3: self._response[3].intensiveStressVec(),
            4: self._response[4].intensiveStressVec()
        }


        return  # a new instance of type tetrahedron

    # volume of an irregular tetrahedron
    def _volTet(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
        # compute the square of the lengths of its six edges
        l12 = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2
        l13 = (x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2
        l14 = (x4 - x1)**2 + (y4 - y1)**2 + (z4 - z1)**2
        l23 = (x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2
        l24 = (x4 - x2)**2 + (y4 - y2)**2 + (z4 - z2)**2
        l34 = (x4 - x3)**2 + (y4 - y3)**2 + (z4 - z3)**2

        # prepare matrix from which the volume is computed (Ref. 2 above)
        A = np.array([[0.0, 1.0, 1.0, 1.0, 1.0],
                      [1.0, 0.0, l12, l13, l14],
                      [1.0, l12, 0.0, l23, l24],
                      [1.0, l13, l23, 0.0, l34],
                      [1.0, l14, l24, l34, 0.0]])
        volT = m.sqrt(det(A) / 288.0)
        return volT

    # These FE arrays are evaluated at the beginning of the current step of
    # integration, i.e., they associate with an updated Lagrangian formulation.
    def _massMatrix(self):
        # create the returned mass matrix
        mMtx = np.zeros((12, 12), dtype=float)

        # construct the consistent mass matrix
        massC = np.zeros((12, 12), dtype=float)
        NtN = np.zeros((12, 12), dtype=float)
        for i in range(1, self._tegq.gaussPoints()+1):
            tesfn = self._tetShapeFns[i]
            wgt = self._tegq.weight(i)
            NtN += wgt * np.matmul(np.transpose(tesfn.Nmtx), tesfn.Nmtx)
        massC[:, :] = NtN[:, :]

        # construct the lumped mass matrix in natural co-ordinates
        massL = np.zeros((12, 12), dtype=float)
        row, col = np.diag_indices_from(massC)
        massL[row, col] = massC.sum(axis=1)
        
        # constrcuct the averaged mass matrix in natural co-ordinates
        massA = np.zeros((12, 12), dtype=float)
        massA = 0.5 * (massC + massL)

        # the following print statements were used to verify the code
        # print("\nThe averaged mass matrix in natural co-ordinates is")
        # print(0.5 * massA)  

        # current Vertex coordinates in pentagonal frame of reference
        # coordinates for the reference vertices as tuples
        x01 = (self._v1x0, self._v1y0, self._v1z0)
        x02 = (self._v2x0, self._v2y0, self._v2z0)
        x03 = (self._v3x0, self._v3y0, self._v3z0)
        x04 = (self._v4x0, self._v4y0, self._v4z0)
        
        # convert average mass matrix from natural to physical co-ordinates
        Jdet = tesfn.jacobianDeterminant(x01, x02, x03, x04)
        rho = self.massDensity()
        mMtx = (rho * Jdet) * massA

        return mMtx  

    def _tangentStiffnessMtxC(self, ):
        
        cMtx = np.zeros((12, 12), dtype=float)
        
        # assign coordinates at the vertices in the reference configuration
        xn1 = (self._v1x, self._v1y, self._v1z)
        xn2 = (self._v2x, self._v2y, self._v2z)
        xn3 = (self._v3x, self._v3y, self._v3z)
        xn4 = (self._v4x, self._v4y, self._v4z)

        # coordinates for the reference vertices as tuples
        x01 = (self._v1x0, self._v1y0, self._v1z0)
        x02 = (self._v2x0, self._v2y0, self._v2z0)
        x03 = (self._v3x0, self._v3y0, self._v3z0)
        x04 = (self._v4x0, self._v4y0, self._v4z0)

        Cs1 = np.zeros((12, 12), dtype=float)
        Ct1 = np.zeros((12, 12), dtype=float)

        for i in range(1, self._tegq.gaussPoints()+1):
            tesfn = self._tetShapeFns[i]
            wgt = self._tegq.weight(i)  
            Mt = self._Mt[i]
            Ss = self.Ss[i]
            
            # determinant of jacobian matrix
            Jdet = tesfn.jacobianDeterminant(x01, x02, x03, x04)

            BLmtx = tesfn.BL(xn1, xn2, xn3, xn4)

            Hmtx1 = tesfn.H1(xn1, xn2, xn3, xn4)
            BNmtx1 = tesfn.BN1(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            
            Hmtx2 = tesfn.H2(xn1, xn2, xn3, xn4)
            BNmtx2 = tesfn.BN2(xn1, xn2, xn3, xn4, x01, x02, x03, x04)

            Hmtx3 = tesfn.H3(xn1, xn2, xn3, xn4)
            BNmtx3 = tesfn.BN3(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            
            Hmtx4 = tesfn.H4(xn1, xn2, xn3, xn4)
            BNmtx4 = tesfn.BN4(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            
            Hmtx5 = tesfn.H5(xn1, xn2, xn3, xn4)
            BNmtx5 = tesfn.BN5(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            
            # total nonlinear Bmatrix
            BNmtx = BNmtx1 + BNmtx2 + BNmtx3 + BNmtx4 + BNmtx5
            
            # the tangent stiffness matrix Cs1
            Cs1 += (Jdet * wgt * ( Hmtx1.T.dot(Ss).dot(Hmtx1) 
                                 + Hmtx2.T.dot(Ss).dot(Hmtx2)  
                                 + Hmtx3.T.dot(Ss).dot(Hmtx3)
                                 + Hmtx4.T.dot(Ss).dot(Hmtx4)
                                 + Hmtx5.T.dot(Ss).dot(Hmtx5) ))
            # the tangent stiffness matrix Ct1
            Ct1 += (Jdet * wgt * ( BLmtx.T.dot(Mt).dot(BLmtx) 
                                 + BLmtx.T.dot(Mt).dot(BNmtx)
                                 + BNmtx.T.dot(Mt).dot(BLmtx) 
                                 + BNmtx.T.dot(Mt).dot(BNmtx) ))

        Cs = np.zeros((12, 12), dtype=float)
        Ct = np.zeros((12, 12), dtype=float)
        
        Cs[:, :] = Cs1[:, :]
        Ct[:, :] = Ct1[:, :]
              
        # determine the total tangent stiffness matrix
        cMtx = Cs + Ct

        return cMtx
    
    def _secantStiffnessMtxK(self, reindex):
        
        kMtx = np.zeros((12, 12), dtype=float)
        
        # assign coordinates at the vertices in the reference configuration
        xn1 = (self._v1x, self._v1y, self._v1z)
        xn2 = (self._v2x, self._v2y, self._v2z)
        xn3 = (self._v3x, self._v3y, self._v3z)
        xn4 = (self._v4x, self._v4y, self._v4z)

        # coordinates for the reference vertices as tuples
        x01 = (self._v1x0, self._v1y0, self._v1z0)
        x02 = (self._v2x0, self._v2y0, self._v2z0)
        x03 = (self._v3x0, self._v3y0, self._v3z0)
        x04 = (self._v4x0, self._v4y0, self._v4z0)

        Ks1 = np.zeros((12, 12), dtype=float)
        Kt1 = np.zeros((12, 12), dtype=float)

        for i in range(1, self._tegq.gaussPoints()+1):
            tesfn = self._tetShapeFns[i]
            wgt = self._tegq.weight(i)  
            Ms = self._Ms[i]
            Mt = self._Mt[i]
            
            # determinant of jacobian matrix
            Jdet = tesfn.jacobianDeterminant(x01, x02, x03, x04)

            BLmtx = tesfn.BL(xn1, xn2, xn3, xn4)

            Hmtx1 = tesfn.H1(xn1, xn2, xn3, xn4)
            Lmtx1 = tesfn.L1(xn1, xn2, xn3, xn4)
            BNmtx1 = tesfn.BN1(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            A1 = tesfn.A1(xn1, xn2, xn3, xn4, x01, x02, x03, x04)              
            Dmtx1 = self.dDisplacement1(reindex)       
            dA1 = np.dot(Lmtx1, np.transpose(Dmtx1))            
            dSt1 = A1.T.dot(Mt).dot(dA1)
            
            Hmtx2 = tesfn.H2(xn1, xn2, xn3, xn4)
            Lmtx2 = tesfn.L2(xn1, xn2, xn3, xn4)
            BNmtx2 = tesfn.BN2(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            A2 = tesfn.A2(xn1, xn2, xn3, xn4, x01, x02, x03, x04)              
            Dmtx2 = self.dDisplacement2(reindex)        
            dA2 = np.dot(Lmtx2, np.transpose(Dmtx2))            
            dSt2 = A2.T.dot(Mt).dot(dA2)

            Hmtx3 = tesfn.H3(xn1, xn2, xn3, xn4)
            Lmtx3 = tesfn.L3(xn1, xn2, xn3, xn4)
            BNmtx3 = tesfn.BN3(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            A3 = tesfn.A3(xn1, xn2, xn3, xn4, x01, x02, x03, x04)              
            Dmtx3 = self.dDisplacement3(reindex)        
            dA3 = np.dot(Lmtx3, np.transpose(Dmtx3))            
            dSt3 = A3.T.dot(Mt).dot(dA3)  
            
            Hmtx4 = tesfn.H4(xn1, xn2, xn3, xn4)
            Lmtx4 = tesfn.L4(xn1, xn2, xn3, xn4)
            BNmtx4 = tesfn.BN4(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            A4 = tesfn.A4(xn1, xn2, xn3, xn4, x01, x02, x03, x04)              
            Dmtx4 = self.dDisplacement4(reindex)       
            dA4 = np.dot(Lmtx4, np.transpose(Dmtx4))            
            dSt4 = A4.T.dot(Mt).dot(dA4)
            
            Hmtx5 = tesfn.H5(xn1, xn2, xn3, xn4)
            Lmtx5 = tesfn.L5(xn1, xn2, xn3, xn4)
            BNmtx5 = tesfn.BN5(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            A5 = tesfn.A5(xn1, xn2, xn3, xn4, x01, x02, x03, x04)              
            Dmtx5 = self.dDisplacement5(reindex)        
            dA5 = np.dot(Lmtx5, np.transpose(Dmtx5))            
            dSt5 = A5.T.dot(Mt).dot(dA5)            
            
            # total nonlinear Bmatrix
            BNmtx = BNmtx1 + BNmtx2 + BNmtx3 + BNmtx4 + BNmtx5
            
            # the secant stiffness matrix Ks1
            Ks1 += (Jdet * wgt * ( BLmtx.T.dot(Ms).dot(BLmtx) 
                                 + BLmtx.T.dot(Ms).dot(BNmtx) 
                                 + BNmtx.T.dot(Ms).dot(BLmtx) 
                                 + BNmtx.T.dot(Ms).dot(BNmtx) ))
            # the secant stiffness matrix Kt1
            Kt1 += (Jdet * wgt * ( Hmtx1.T.dot(dSt1).dot(Hmtx1) 
                                 + Hmtx2.T.dot(dSt2).dot(Hmtx2)
                                 + Hmtx3.T.dot(dSt3).dot(Hmtx3)
                                 + Hmtx4.T.dot(dSt4).dot(Hmtx4)
                                 + Hmtx5.T.dot(dSt5).dot(Hmtx5) ))

        Ks = np.zeros((12, 12), dtype=float)
        Kt = np.zeros((12, 12), dtype=float)
        
        Ks[:, :] = Ks1[:, :]
        Kt[:, :] = Kt1[:, :]
              
        # determine the total secant stiffness matrix
        kMtx = Ks + Kt

        return kMtx
    
    
    def _forcingFunction(self):
        
        state = 'curr'
        
        
        fVec = np.zeros((12,1), dtype=float)            

        nx, ny, nz = self.normal(state)
        
        # create the normal vector
        n = np.zeros((1, 3), dtype=float)
        n[0, 0] = nx
        n[0, 1] = ny
        n[0, 2] = nz

        # assign coordinates at the vertices in the reference configuration
        # assign coordinates at the vertices in the reference configuration
        xn1 = (self._v1x, self._v1y, self._v1z)
        xn2 = (self._v2x, self._v2y, self._v2z)
        xn3 = (self._v3x, self._v3y, self._v3z)
        xn4 = (self._v4x, self._v4y, self._v4z)

        # coordinates for the reference vertices as tuples
        x01 = (self._v1x0, self._v1y0, self._v1z0)
        x02 = (self._v2x0, self._v2y0, self._v2z0)
        x03 = (self._v3x0, self._v3y0, self._v3z0)
        x04 = (self._v4x0, self._v4y0, self._v4z0)            

        Nt = np.zeros((12, 1), dtype=float)

        BLte1 = np.zeros((12, 6), dtype=float)
        BNte1 = np.zeros((12, 6), dtype=float)
        BNte2 = np.zeros((12, 6), dtype=float)
        BNte3 = np.zeros((12, 6), dtype=float)
        BNte4 = np.zeros((12, 6), dtype=float)
        BNte5 = np.zeros((12, 6), dtype=float)
                
        for i in range(1, self._tegq.gaussPoints()+1):
            tesfn = self._tetShapeFns[i]
            wgt = self._tegq.weight(i)
            T0 = self.T[i]
            
            BLmtx = tesfn.BL(xn1, xn2, xn3, xn4)
            BNmtx1 = tesfn.BN1(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            BNmtx2 = tesfn.BN2(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            BNmtx3 = tesfn.BN3(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            BNmtx4 = tesfn.BN4(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            BNmtx5 = tesfn.BN5(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
            
            BLte1 += wgt * np.transpose(BLmtx)   
            BNte1 += wgt * np.transpose(BNmtx1)   
            BNte2 += wgt * np.transpose(BNmtx2)   
            BNte3 += wgt * np.transpose(BNmtx3)   
            BNte4 += wgt * np.transpose(BNmtx4) 
            BNte5 += wgt * np.transpose(BNmtx5)   

            B = BLte1 + BNte1 + BNte2 + BNte3 + BNte4 + BNte5
            BdotT0 = np.dot(B, T0)
            
        # determinant of jacobian matrix
        teJdet = tesfn.jacobianDeterminant(x01, x02, x03, x04)
        
        F0 = np.zeros((12, 1), dtype=float)
        
        F0 = teJdet * BdotT0
 
        Nt = np.zeros((12, 1), dtype=float)
        for i in range(1, self._trgq.gaussPoints()+1):
            trsfn = self._triShapeFns[i]
            wgt = self._trgq.weight(i)
            # create the traction vector
            t = np.dot(self.Ss[i], np.transpose(n))            
            Nt += wgt * np.dot( np.transpose(trsfn.Nmtx), t )
        
        # determinant of jacobian matrix
        trJdet = trsfn.jacobianDeterminant(x01, x02, x03)
        
        FBc = np.zeros((12, 1), dtype=float)
        
        FBc = (trJdet * Nt )  

        fVec = FBc - F0
            
        return fVec 

    def toString(self, state):
        if self._number < 10:
            s = 'tetrahedron[0'
        else:
            s = 'tetrahedron['
        s = s + str(self._number)
        s = s + '] has vertices: \n'
        if isinstance(state, str):
            s = s + '   1: ' + self._vertex[1].toString(state) + '\n'
            s = s + '   2: ' + self._vertex[2].toString(state) + '\n'
            s = s + '   3: ' + self._vertex[3].toString(state) + '\n'
            s = s + '   4: ' + self._vertex[4].toString(state)
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.toString.")
        return s

    def number(self):
        return self._number

    def VertexNumbers(self):
        numbers = sorted(self._setOfVertices)
        return numbers[0], numbers[1], numbers[2], numbers[3]

    def hasVertex(self, number):
        return number in self._setOfVertices

    def getVertex(self, number):
        if self._vertex[1].number() == number:
            return self._vertex[1]
        elif self._vertex[2].number() == number:
            return self._vertex[2]
        elif self._vertex[3].number() == number:
            return self._vertex[3]
        elif self._vertex[4].number() == number:
            return self._vertex[4]
        else:
            raise RuntimeError('The requested Vertex {} is '.format(number) +
                               'not in tetrhaderon {}.'.format(self._number))

    def update(self):
        # computes the fields positioned at the next time step

        # get the updated coordinates for the vetices of the tetrahedron
        x1 = self._vertex[1].coordinates('next')
        x2 = self._vertex[2].coordinates('next')
        x3 = self._vertex[3].coordinates('next')
        x4 = self._vertex[4].coordinates('next')


        # base vector 1: connects the centroid of pentagon with one of the 
        # pentagon’s vertices
        x = x2[0] - x1[0]
        y = x2[1] - x1[1]
        z = x2[2] - x1[2]
        mag = m.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 3: connects the centroid of the dodecahedron with the 
        # centroid of a pentagon
        x = x4[0] - x1[0]
        y = x4[1] - x1[1]
        z = x4[2] - x1[2]
        mag = m.sqrt(x * x + y * y + z * z)
        n3x = x / mag
        n3y = y / mag
        n3z = z / mag

        # base vector 2 is obtained through the cross product
        n2x = n3y * n1z - n3z * n1y
        n2y = n3z * n1x - n3x * n1z
        n2z = n3x * n1y - n3y * n1x

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
        self._v1x0 = n1x * x1[0] + n1y * x1[1] + n1z * x1[2]
        self._v1y0 = n2x * x1[0] + n2y * x1[1] + n2z * x1[2]
        self._v2x0 = n1x * x2[0] + n1y * x2[1] + n1z * x2[2]
        self._v2y0 = n2x * x2[0] + n2y * x2[1] + n2z * x2[2]
        self._v3x0 = n1x * x3[0] + n1y * x3[1] + n1z * x3[2]
        self._v3y0 = n2x * x3[0] + n2y * x3[1] + n2z * x3[2]
        self._v4x0 = n1x * x4[0] + n1y * x4[1] + n1z * x4[2]
        self._v4y0 = n2x * x4[0] + n2y * x4[1] + n2z * x4[2]

        # z offsets 
        self._v1z0 = n3x * x1[0] + n3y * x1[1] + n3z * x1[2]
        self._v2z0 = n3x * x2[0] + n3y * x2[1] + n3z * x2[2]
        self._v3z0 = n3x * x3[0] + n3y * x3[1] + n3z * x3[2]
        self._v4z0 = n3x * x4[0] + n3y * x4[1] + n3z * x4[2]    
        
        # initialize current vertice coordinates
        self._v1x = self._v1x0
        self._v1y = self._v1y0
        self._v1z = self._v1z0
        self._v2x = self._v2x0
        self._v2y = self._v2y0
        self._v2z = self._v2z0
        self._v3x = self._v3x0
        self._v3y = self._v3y0
        self._v3z = self._v3z0
        self._v4x = self._v4x0
        self._v4y = self._v4y0
        self._v4z = self._v4z0

        # determine the volume of this tetrahedron 
        self._Vn = self._volTet(self._v1x, self._v1y, self._v1z,
                                self._v2x, self._v2y, self._v2z,
                                self._v3x, self._v3y, self._v3z,
                                self._v4x, self._v4y, self._v4z)

        # establish the centroidal location of this tetrahedron
        self._cx = self._centroidSF.interpolate(self._v1x, self._v2x,
                                                self._v3x, self._v4x)
        self._cy = self._centroidSF.interpolate(self._v1y, self._v2y,
                                                self._v3y, self._v4y)
        self._cz = self._centroidSF.interpolate(self._v1z, self._v2z,
                                                self._v3z, self._v4z)
      
        # rotate this centroid back into the reference coordinate system
        self._centroidX = n1x * self._cx + n2x * self._cy + n3x * self._cz
        self._centroidY = n1y * self._cx + n2y * self._cy + n3y * self._cz
        self._centroidZ = n1z * self._cx + n2z * self._cy + n3z * self._cz

        # assign coordinates at the vertices in the reference configuration
        xn1 = (self._v1x, self._v1y, self._v1z)
        xn2 = (self._v2x, self._v2y, self._v2z)
        xn3 = (self._v3x, self._v3y, self._v3z)
        xn4 = (self._v4x, self._v4y, self._v4z)

        # coordinates for the reference vertices as tuples
        x01 = (self._v1x0, self._v1y0, self._v1z0)
        x02 = (self._v2x0, self._v2y0, self._v2z0)
        x03 = (self._v3x0, self._v3y0, self._v3z0)
        x04 = (self._v4x0, self._v4y0, self._v4z0)

        # establish the deformation and displacement gradients as dictionaries
        self._Gn[1] = self._tetShapeFns[1].G(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
        self._Gn[2] = self._tetShapeFns[2].G(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
        self._Gn[3] = self._tetShapeFns[3].G(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
        self._Gn[4] = self._tetShapeFns[4].G(xn1, xn2, xn3, xn4, x01, x02, x03, x04)

        # deformation gradients located at the Gauss points of pentagon
        self._Fn[1] = self._tetShapeFns[1].F(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
        self._Fn[2] = self._tetShapeFns[2].F(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
        self._Fn[3] = self._tetShapeFns[3].F(xn1, xn2, xn3, xn4, x01, x02, x03, x04)
        self._Fn[4] = self._tetShapeFns[4].F(xn1, xn2, xn3, xn4, x01, x02, x03, x04)

        return  # nothing

    def advance(self, reindex):
        # advance the geometric properties of the pentagon
        self._Vp = self._Vc
        self._Vc = self._Vn
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
        for i in range(1, self._tegq.gaussPoints()+1):
            self._Fp[i][:, :] = self._Fc[i][:, :]
            self._Fc[i][:, :] = self._Fn[i][:, :]
            self._Gp[i][:, :] = self._Gc[i][:, :]
            self._Gc[i][:, :] = self._Gn[i][:, :]

        # compute the FE arrays needed for the next interval of integration
        self.mMtx = self._massMatrix()
        self.cMtx = self._tangentStiffnessMtxC()
        self.kMtx = self._secantStiffnessMtxK(reindex)
        self.fVec = self._forcingFunction()
        
        return  # nothing

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
                                   "in a call to tetrahedron.normal.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.normal.")
        return np.array([nx, ny, nz])
    
    
    # Material properties that associate with this tetrahedron.

    def massDensity(self):
        return self._rho

    # Geometric properties of this tetrahedron

    def volume(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._Vc
            elif state == 'n' or state == 'next':
                return self._Vn
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._Vp
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._V0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to tetrahedron.volume.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.volume.")

    def volumetricStretch(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return (self._Vc / self._V0)**(1.0 / 3.0)
            elif state == 'n' or state == 'next':
                return (self._Vn / self._V0)**(1.0 / 3.0)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return (self._Vp / self._V0)**(1.0 / 3.0)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 1.0
            else:
                raise RuntimeError("An unknown state {} in ".format(state) +
                                   "a call to tetrahedron.volumetricStretch.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to tetrahedron.volumetricStretch.")

    def volumetricStrain(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return m.log(self._Vc / self._V0) / 3.0
            elif state == 'n' or state == 'next':
                return m.log(self._Vn / self._V0) / 3.0
            elif state == 'p' or state == 'prev' or state == 'previous':
                return m.log(self._Vp / self._V0) / 3.0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} in ".format(state) +
                                   "a call to tetrahedron.volumetricStrain.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to tetrahedron.volumetricStrain.")

    def dVolumetricStrain(self, state):
        if isinstance(state, str):
            h = 2.0 * self._h
            if state == 'c' or state == 'curr' or state == 'current':
                # use second-order central difference formula
                dVol = (self._Vn - self._Vp) / h
                return (dVol / self._Vc) / 3.0
            elif state == 'n' or state == 'next':
                # use second-order backward difference formula
                dVol = (3.0 * self._Vn - 4.0 * self._Vc + self._Vp) / h
                return (dVol / self._Vn) / 3.0
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use second-order forward difference formula
                dVol = (-self._Vn + 4.0 * self._Vc - 3.0 * self._Vp) / h
                return (dVol / self._Vp) / 3.0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} in ".format(state) +
                                   "a call to tetrahedron.dVolumetricStrain.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to tetrahedron.dVolumetricStrain.")

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
                raise RuntimeError("An unknown state {} in ".format(state) +
                                   "a call to tetrahedron.centroid.")
        else:
            raise RuntimeError("An unknown state {} in ".format(str(state)) +
                               "a call to tetrahedron.centroid.")
        return np.array([cx, cy, cz])

    def centeroidDisplacement(self, reindex, state):
        # verify the input
        if not isinstance(reindex, Pivot):
            raise RuntimeError("The 'reindex' variable sent to " +
                               "tetrahedron.centroidDisplacement must be of type Pivot.")
        # calculate the displacement in the specified configuration
        u = np.zeros(3, dtype=float)
        x0 = np.zeros(3, dtype=float)
        
        x0, y0, z0 = self.centroid('ref')        
        xRef = np.array([x0, y0, z0])
        fromCase = reindex.pivotCase('ref')
        if isinstance(state, str):
            xp, yp, zp = self.centroid('prev')
            xc, yc, zc = self.centroid('curr')
            xn, yn, zn = self.centroid('next')
            
            if state == 'c' or state == 'curr' or state == 'current':
                x = np.array([xc, yc, zc])
                toCase = reindex.pivotCase('curr')
                x0 = reindex.reindexVector(xRef, fromCase, toCase)
            elif state == 'n' or state == 'next':
                x = np.array([xn, yn, zn])
                toCase = reindex.pivotCase('next')
                x0 = reindex.reindexVector(xRef, fromCase, toCase)
            elif state == 'p' or state == 'prev' or state == 'previous':
                x = np.array([xp, yp, zp])
                toCase = reindex.pivotCase('prev')
                x0 = reindex.reindexVector(xRef, fromCase, toCase)
            elif state == 'r' or state == 'ref' or state == 'reference':
                x = np.array([x0, y0, z0])
                x0 = np.array([x0, y0, z0])
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   "in a call to tetrahedron.centroidDisplacement.")
        else:
            raise RuntimeError("Unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.centroidDisplacement.")
        u = x - x0
        
        R = self.rotation(state)
        uxr = R[0, 0] * u[0] + R[1, 0] * u[1] + R[2, 0] * u[2]
        uyr = R[0, 1] * u[0] + R[1, 1] * u[1] + R[2, 1] * u[2]
        uzr = R[0, 2] * u[0] + R[1, 2] * u[1] + R[2, 2] * u[2]

        return np.array([uxr, uyr, uzr])

    def centeroidVelocity(self, reindex, state):
        # verify the input
        if not isinstance(reindex, Pivot):
            raise RuntimeError("The 'reindex' variable sent to " +
                               "tetrahedron.centroidVelocity must be of type Pivot.")
        # calculate the velocity in the specified configuration
        h = 2.0 * self._h
        v = np.zeros(3, dtype=float)
        if isinstance(state, str):
            xp, yp, zp = self.centroid('prev')
            xc, yc, zc = self.centroid('curr')
            xn, yn, zn = self.centroid('next')
            if state == 'c' or state == 'curr' or state == 'current':
                toCase = reindex.pivotCase('curr')
                # map vectors into co-ordinate system of current configuration
                xPrev = np.array([xp, yp, zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xNext = np.array([xn, yn, zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
                
                # use second-order central difference formula
                v = (xN - xP) / h
                
            elif state == 'n' or state == 'next':
                toCase = reindex.pivotCase('next')
                # map vectors into co-ordinate system of next configuration
                xPrev = np.array([xp, yp, zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xCurr = np.array([xc, yc, zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xN = np.array([xn, yn, zn])
                # use second-order backward difference formula
                v = (3.0 * xN - 4.0 * xC + xP) / h

            elif state == 'p' or state == 'prev' or state == 'previous':
                toCase = reindex.pivotCase('prev')
                # map vector into co-ordinate system of previous configuration
                xP = np.array([xp, yp, zp])
                xCurr = np.array([xc, yc, zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xNext = np.array([xn, yn, zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
                # use second-order forward difference formula
                v = (-xN + 4.0 * xC - 3.0 * xP) / h
                
            elif state == 'r' or state == 'ref' or state == 'reference':
                # velocity is zero
                pass
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   "in a call to tetrahedron.centroidVelocity.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.centroidVelocity.")

        R = self.rotation(state)
        vxr = R[0, 0] * v[0] + R[1, 0] * v[1] + R[2, 0] * v[2]
        vyr = R[0, 1] * v[0] + R[1, 1] * v[1] + R[2, 1] * v[2]
        vzr = R[0, 2] * v[0] + R[1, 2] * v[1] + R[2, 2] * v[2]

        return np.array([vxr, vyr, vzr])

    def centroidAcceleration(self, reindex, state):
        # verify the input
        if not isinstance(reindex, Pivot):
            raise RuntimeError("The 'reindex' variable sent to " +
                               "tetrahedron.centroidAcceleration must be of type Pivot.")
        # calculate the acceleration in the specified configuration
        h2 = self._h**2
        a = np.zeros(3, dtype=float)
        if isinstance(state, str):
            xp, yp, zp = self.centroid('prev')
            xc, yc, zc = self.centroid('curr')
            xn, yn, zn = self.centroid('next')
            if state == 'c' or state == 'curr' or state == 'current':
                toCase = reindex.pivotCase('curr')
                # map vectors into co-ordinate system of current configuration
                xPrev = np.array([xp, yp, zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xC = np.array([xc, yc, zc])
                xNext = np.array([xn, yn, zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
            elif state == 'n' or state == 'next':
                toCase = reindex.pivotCase('next')
                # map vectors into co-ordinate system of next configuration
                xPrev = np.array([xp, yp, zp])
                fromCase = reindex.pivotCase('prev')
                xP = reindex.reindexVector(xPrev, fromCase, toCase)
                xCurr = np.array([xc, yc, zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xN = np.array([xn, yn, zn])
            elif state == 'p' or state == 'prev' or state == 'previous':
                toCase = reindex.pivotCase('prev')
                # map vector into co-ordinate system of previous configuration
                xP = np.array([xp, yp, zp])
                xCurr = np.array([xc, yc, zc])
                fromCase = reindex.pivotCase('curr')
                xC = reindex.reindexVector(xCurr, fromCase, toCase)
                xNext = np.array([xn, yn, zn])
                fromCase = reindex.pivotCase('next')
                xN = reindex.reindexVector(xNext, fromCase, toCase)
            elif state == 'r' or state == 'ref' or state == 'reference':
                # acceleration is zero
                pass
            else:
                raise RuntimeError("Unknown state {} ".format(state) +
                                   "in a call to pentagon.centroidAcceleration.")
        else:
            raise RuntimeError("Unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.centroidAcceleration.")
        a = (xN - 2.0 * xC + xP) / h2

        R = self.rotation(state)
        axr = R[0, 0] * a[0] + R[1, 0] * a[1] + R[2, 0] * a[2]
        ayr = R[0, 1] * a[0] + R[1, 1] * a[1] + R[2, 1] * a[2]
        azr = R[0, 2] * a[0] + R[1, 2] * a[1] + R[2, 2] * a[2]

        return np.array([axr, ayr, azr])

    # change in displacementin contribution to the nonlinear strain
    def dDisplacement1(self, reindex):
        v1 = self._vertex[1].velocity(reindex, 'curr')
        v2 = self._vertex[2].velocity(reindex, 'curr')
        v3 = self._vertex[3].velocity(reindex, 'curr')
        v4 = self._vertex[4].velocity(reindex, 'curr')

        R = self.rotation('curr')        
        Dmtx1 = np.zeros((3, 12), dtype=float)
        Dmtx1[0, 0] = R[0, 0] * v1[0] + R[1, 0] * v1[1] + R[2, 0] * v1[2]
        Dmtx1[0, 3] = R[0, 0] * v2[0] + R[1, 0] * v2[1] + R[2, 0] * v2[2]
        Dmtx1[0, 6] = R[0, 0] * v3[0] + R[1, 0] * v3[1] + R[2, 0] * v3[2]
        Dmtx1[0, 9] = R[0, 0] * v4[0] + R[1, 0] * v4[1] + R[2, 0] * v4[2]


        Dmtx1[1, 1] = R[0, 1] * v1[0] + R[1, 1] * v1[1] + R[2, 1] * v1[2]
        Dmtx1[1, 4] = R[0, 1] * v2[0] + R[1, 1] * v2[1] + R[2, 1] * v2[2]
        Dmtx1[1, 7] = R[0, 1] * v3[0] + R[1, 1] * v3[1] + R[2, 1] * v3[2]
        Dmtx1[1, 10] = R[0, 1] * v4[0] + R[1, 1] * v4[1] + R[2, 1] * v4[2]
 
        
        Dmtx1[2, 2] = R[0, 2] * v1[0] + R[1, 2] * v1[1] + R[2, 2] * v1[2]
        Dmtx1[2, 5] = R[0, 2] * v2[0] + R[1, 2] * v2[1] + R[2, 2] * v2[2]
        Dmtx1[2, 8] = R[0, 2] * v3[0] + R[1, 2] * v3[1] + R[2, 2] * v3[2]
        Dmtx1[2, 11] = R[0, 2] * v4[0] + R[1, 2] * v4[1] + R[2, 2] * v4[2]
     
        return Dmtx1

    def dDisplacement2(self, reindex):
        v1 = self._vertex[1].velocity(reindex, 'curr')
        v2 = self._vertex[2].velocity(reindex, 'curr')
        v3 = self._vertex[3].velocity(reindex, 'curr')
        v4 = self._vertex[4].velocity(reindex, 'curr')

        R = self.rotation('curr')        
        Dmtx2 = np.zeros((3, 12), dtype=float)
        Dmtx2[0, 0] = R[0, 0] * v1[0] + R[1, 0] * v1[1] + R[2, 0] * v1[2]
        Dmtx2[0, 3] = R[0, 0] * v2[0] + R[1, 0] * v2[1] + R[2, 0] * v2[2]
        Dmtx2[0, 6] = R[0, 0] * v3[0] + R[1, 0] * v3[1] + R[2, 0] * v3[2]
        Dmtx2[0, 9] = R[0, 0] * v4[0] + R[1, 0] * v4[1] + R[2, 0] * v4[2]


        Dmtx2[1, 1] = R[0, 1] * v1[0] + R[1, 1] * v1[1] + R[2, 1] * v1[2]
        Dmtx2[1, 4] = R[0, 1] * v2[0] + R[1, 1] * v2[1] + R[2, 1] * v2[2]
        Dmtx2[1, 7] = R[0, 1] * v3[0] + R[1, 1] * v3[1] + R[2, 1] * v3[2]
        Dmtx2[1, 10] = R[0, 1] * v4[0] + R[1, 1] * v4[1] + R[2, 1] * v4[2]
 
        
        Dmtx2[2, 2] = R[0, 2] * v1[0] + R[1, 2] * v1[1] + R[2, 2] * v1[2]
        Dmtx2[2, 5] = R[0, 2] * v2[0] + R[1, 2] * v2[1] + R[2, 2] * v2[2]
        Dmtx2[2, 8] = R[0, 2] * v3[0] + R[1, 2] * v3[1] + R[2, 2] * v3[2]
        Dmtx2[2, 11] = R[0, 2] * v4[0] + R[1, 2] * v4[1] + R[2, 2] * v4[2]
     
        return Dmtx2
    
    def dDisplacement3(self, reindex):
        v1 = self._vertex[1].velocity(reindex, 'curr')
        v2 = self._vertex[2].velocity(reindex, 'curr')
        v3 = self._vertex[3].velocity(reindex, 'curr')
        v4 = self._vertex[4].velocity(reindex, 'curr')

        R = self.rotation('curr')        
        Dmtx3 = np.zeros((3, 12), dtype=float)
        Dmtx3[0, 0] = R[0, 1] * v1[0] + R[1, 1] * v1[1] + R[2, 1] * v1[2]
        Dmtx3[0, 3] = R[0, 1] * v2[0] + R[1, 1] * v2[1] + R[2, 1] * v2[2]
        Dmtx3[0, 6] = R[0, 1] * v3[0] + R[1, 1] * v3[1] + R[2, 1] * v3[2]
        Dmtx3[0, 9] = R[0, 1] * v4[0] + R[1, 1] * v4[1] + R[2, 1] * v4[2]


        Dmtx3[1, 1] = R[0, 2] * v1[0] + R[1, 2] * v1[1] + R[2, 2] * v1[2]
        Dmtx3[1, 4] = R[0, 2] * v2[0] + R[1, 2] * v2[1] + R[2, 2] * v2[2]
        Dmtx3[1, 7] = R[0, 2] * v3[0] + R[1, 2] * v3[1] + R[2, 2] * v3[2]
        Dmtx3[1, 10] = R[0, 2] * v4[0] + R[1, 2] * v4[1] + R[2, 2] * v4[2]
 
        
        Dmtx3[2, 2] = R[0, 2] * v1[0] + R[1, 2] * v1[1] + R[2, 2] * v1[2]
        Dmtx3[2, 5] = R[0, 2] * v2[0] + R[1, 2] * v2[1] + R[2, 2] * v2[2]
        Dmtx3[2, 8] = R[0, 2] * v3[0] + R[1, 2] * v3[1] + R[2, 2] * v3[2]
        Dmtx3[2, 11] = R[0, 2] * v4[0] + R[1, 2] * v4[1] + R[2, 2] * v4[2]
     
        return Dmtx3

    def dDisplacement4(self, reindex):
        v1 = self._vertex[1].velocity(reindex, 'curr')
        v2 = self._vertex[2].velocity(reindex, 'curr')
        v3 = self._vertex[3].velocity(reindex, 'curr')
        v4 = self._vertex[4].velocity(reindex, 'curr')

        R = self.rotation('curr')        
        Dmtx4 = np.zeros((3, 12), dtype=float)
        Dmtx4[0, 0] = R[0, 1] * v1[0] + R[1, 1] * v1[1] + R[2, 1] * v1[2]
        Dmtx4[0, 3] = R[0, 1] * v2[0] + R[1, 1] * v2[1] + R[2, 1] * v2[2]
        Dmtx4[0, 6] = R[0, 1] * v3[0] + R[1, 1] * v3[1] + R[2, 1] * v3[2]
        Dmtx4[0, 9] = R[0, 1] * v4[0] + R[1, 1] * v4[1] + R[2, 1] * v4[2]


        Dmtx4[1, 1] = R[0, 1] * v1[0] + R[1, 1] * v1[1] + R[2, 1] * v1[2]
        Dmtx4[1, 4] = R[0, 1] * v2[0] + R[1, 1] * v2[1] + R[2, 1] * v2[2]
        Dmtx4[1, 7] = R[0, 1] * v3[0] + R[1, 1] * v3[1] + R[2, 1] * v3[2]
        Dmtx4[1, 10] = R[0, 1] * v4[0] + R[1, 1] * v4[1] + R[2, 1] * v4[2]
 
        
        Dmtx4[2, 2] = R[0, 0] * v1[0] + R[1, 0] * v1[1] + R[2, 0] * v1[2]
        Dmtx4[2, 5] = R[0, 0] * v2[0] + R[1, 0] * v2[1] + R[2, 0] * v2[2]
        Dmtx4[2, 8] = R[0, 0] * v3[0] + R[1, 0] * v3[1] + R[2, 0] * v3[2]
        Dmtx4[2, 11] = R[0, 0] * v4[0] + R[1, 0] * v4[1] + R[2, 0] * v4[2]
     
        return Dmtx4

    def dDisplacement5(self, reindex):
        v1 = self._vertex[1].velocity(reindex, 'curr')
        v2 = self._vertex[2].velocity(reindex, 'curr')
        v3 = self._vertex[3].velocity(reindex, 'curr')
        v4 = self._vertex[4].velocity(reindex, 'curr')

        R = self.rotation('curr')        
        Dmtx5 = np.zeros((3, 12), dtype=float)
        Dmtx5[0, 0] = R[0, 1] * v1[0] + R[1, 1] * v1[1] + R[2, 1] * v1[2]
        Dmtx5[0, 3] = R[0, 1] * v2[0] + R[1, 1] * v2[1] + R[2, 1] * v2[2]
        Dmtx5[0, 6] = R[0, 1] * v3[0] + R[1, 1] * v3[1] + R[2, 1] * v3[2]
        Dmtx5[0, 9] = R[0, 1] * v4[0] + R[1, 1] * v4[1] + R[2, 1] * v4[2]


        Dmtx5[1, 1] = R[0, 2] * v1[0] + R[1, 2] * v1[1] + R[2, 2] * v1[2]
        Dmtx5[1, 4] = R[0, 2] * v2[0] + R[1, 2] * v2[1] + R[2, 2] * v2[2]
        Dmtx5[1, 7] = R[0, 2] * v3[0] + R[1, 2] * v3[1] + R[2, 2] * v3[2]
        Dmtx5[1, 10] = R[0, 2] * v4[0] + R[1, 2] * v4[1] + R[2, 2] * v4[2]
 
        
        Dmtx5[2, 2] = R[0, 0] * v1[0] + R[1, 0] * v1[1] + R[2, 0] * v1[2]
        Dmtx5[2, 5] = R[0, 0] * v2[0] + R[1, 0] * v2[1] + R[2, 0] * v2[2]
        Dmtx5[2, 8] = R[0, 0] * v3[0] + R[1, 0] * v3[1] + R[2, 0] * v3[2]
        Dmtx5[2, 11] = R[0, 0] * v4[0] + R[1, 0] * v4[1] + R[2, 0] * v4[2]
     
        return Dmtx5
    
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
                                   "in a call to tetrahedron.rotation.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.rotation.")


    # displacement gradient at a Gauss point
    def G(self, tetGaussPt, state):
        if tetGaussPt != self._tegq.gaussPoints():
            raise RuntimeError("The tetgaussPt must be " +
                               "{} in call to ".format(self._tegq.gaussPoints()) +
                               "tetrahedron.G, you sent {}.".format(tetGaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Gc[tetGaussPt])
            elif state == 'n' or state == 'next':
                return np.copy(self._Gn[tetGaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Gp[tetGaussPt])
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._G0[tetGaussPt])
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to tetrahedron.G.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.G.")

    # deformation gradient at a Gauss point
    def F(self, tetGaussPt, state):
        if tetGaussPt != self._tegq.gaussPoints():
            raise RuntimeError("The tetGaussPt must be " +
                               "{} in call to ".format(self._tegq.gaussPoints()) +
                               "tetrahedron.F, you sent {}.".format(tetGaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Fc[tetGaussPt])
            elif state == 'n' or state == 'next':
                return np.copy(self._Fn[tetGaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Fp[tetGaussPt])
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._F0[tetGaussPt])
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to tetrahedron.F.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.F.")

    def L(self, tetGaussPt, state):
        if tetGaussPt != self._tegq.gaussPoints():
            raise RuntimeError("The gaussPt must be " +
                               "{} in call to ".format(self._tegq.gaussPoints()) +
                               "tetrahedron.L, you sent {}.".format(tetGaussPt))
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                # use central difference scheme
                dF = ((self._Fn[tetGaussPt] - self._Fp[tetGaussPt])
                      / (2.0 * self._h))
                fInv = inv(self._Fc[tetGaussPt])
            elif state == 'n' or state == 'next':
                # use backward difference scheme
                dF = ((3.0 * self._Fn[tetGaussPt] - 4.0 * self._Fc[tetGaussPt] +
                       self._Fp[tetGaussPt]) / (2.0 * self._h))
                fInv = inv(self._Fn[tetGaussPt])
            elif state == 'p' or state == 'prev' or state == 'previous':
                # use forward difference scheme
                dF = ((-self._Fn[tetGaussPt] + 4.0 * self._Fc[tetGaussPt] -
                       3.0 * self._Fp[tetGaussPt]) / (2.0 * self._h))
                fInv = inv(self._Fp[tetGaussPt])
            elif state == 'r' or state == 'ref' or state == 'reference':
                dF = np.zeros(3, dtype=float)
                fInv = np.identity(3, dtype=float)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to tetrahedron.L.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to tetrahedron.L.")
        return np.dot(dF, fInv)

    def tetshapeFunction(self, tetGaussPt):
        if tetGaussPt != self._tegq.gaussPoints():
            raise RuntimeError("The gaussPt must be " +
                               "{} in call to ".format(self._tegq.gaussPoints()) +
                               "tetrahedron.tetshapeFunction and " +
                               " you sent {}.".format(tetGaussPt))
            sf = self._shapeFns[tetGaussPt]
        return sf

    def trishapeFunction(self, triGaussPt):
        if triGaussPt != self._trgq.gaussPoints():
            raise RuntimeError("The gaussPt must be " +
                               "{} in call to ".format(self._trgq.gaussPoints()) +
                               "tetrahedron.trishapeFunction and " +
                               " you sent {}.".format(triGaussPt))
            sf = self._shapeFns[triGaussPt]
        return sf
    
    def massMatrix(self):
        mMtx = np.copy(self._massMatrix())
        return mMtx

    def tangentStiffnessMtxC(self):
        cMtx = np.copy(self._tangentStiffnessMtxC())
        return cMtx
    
    def secantStiffnessMtxK(self, reindex):
        kMtx = np.copy(self._secantStiffnessMtxK(reindex))
        return kMtx

    def forcingFunction(self):
        fVec = np.copy(self._forcingFunction())
        return fVec   