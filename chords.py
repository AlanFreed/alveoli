#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ceChords import ControlFiber, SeptalChord
from gaussQuadChords import GaussQuadrature
import math
import numpy as np
from peceHE import PECE
from ridder import findRoot
from shapeFnChords import ShapeFunction  
import spin as spinMtx
from vertices import Vertex
import math as m
import meanProperties as mp




"""
Module chords.py provides geometric information about a septal chord.

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
__date__ = "08-08-2019"
__update__ = "11-06-2020"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


Class Chord in file chords.py allows for the creation of objects that are to
be used to represent chords that connect vertices in a polyhedron.  Each chord
is assigned an unique number, two distinct vertices that serve as its end
points, the time-step size used to approximate derivatives and integrals, plus
the number of Gauss points that are to be used for integrating over its length.

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

Co-ordinates are handled as tuples; vector fields are handled as NumPy arrays;
tensor fields are handled as NumPy arrays with matrix dimensions.

The CGS system of physical units adopted:
    length          centimeters (cm)
    mass            grams       (g)
    time            seconds     (s)
    temperature     centigrade  (C)
where
    force           dynes       [g.cm/s^2]      1 Newton = 10^5 dyne
    pressure        barye       [dyne/cm^2]     1 Pascal = 10 barye
    energy          erg         [dyne.cm]       1 Joule  = 10^7 ergs

Unless explicitly stated otherwise, all fields are evaluated in their current
configuration, in the sense of an updated Lagrangian formulation.  Their next
configuration associates with their Eulerian description.


class Chord


A Chord object, say c, can be printed out to the command window using the
standard print command.

    print(c)

constructor

    c = Chord(number, vertex1, vertex2, dt,
              diaCollagen=None, diaElastin=None, m=1)
        number      is an immutable value that is unique to this chord
        vertex1     is an end point of the chord, an instance of class vertex
        vertex2     is an end point of the chord, an instance of class vertex
        dt          is the time seperating any two neighboring configurations
        diaCollagen is the diameter of the collagen fiber within the chord
        diaElastin  is the diameter of the elastin  fiber within the chord
        m           is the number of CE iterations, i.e., PE(CE)^m, m in [0, 5]
    If the fiber diameters retain their default setting of None, then they are
    assigned via a random distribution; otherwise, they must lie between
    2 and 7.5 microns.

constants

    GAUSS_PTS

methods

    s = c.toString()
        returns a string representation for this chord

    n = c.number()
        returns the unique indexing number affiliated with this chord

    v1, v2 = c.vertexNumbers()
        returns the unique numbers assigned to the two vertices of this chord

    truth = c.hasVertex(number)
        returns 'True' if one of the two vertices has this vertex number

    v = c.getVertex(number)
        returns a vertex; to be called inside, e.g., a c.hasVertex if clause

    pece = c.solver(atGaussPt)
        returns a PECE solver used for integrating the chordal response at the
        specified Gauss point, as extablished in ceChords.py using the PECE
        integrator of peceHE.py

    c.update()
        assigns new co-ordinate values to the chord for its next location and
        updates all affected fields.  It is to be called AFTER all vertices
        have had their co-ordinates updated.  It may be called multiple times
        before freezing the co-ordinate values with a call to c.advance.  This
        method calls its PECE integrator to update the constitutive state.

    c.advance(reindex)
       input
            reindex     is an instance of Pivot object from module pivotIncomingF      
        assigns the current fields to the previous fields, and then assigns
        the next fields to the current fields, thereby freezing the present
        next-fields in preparation for an advancment of a solution along its
        path of motion. It calls the analogous method for its PECE integrator.

    Material properties that associate with this chord.

    rho = c.massDensity()
        returns the mass density of the chord (collagen and elastin fibers)

    truth = c.collagenIsRuptured()
        returns True if the collagen fiber in the chord has ruptured

    truth = c.elastinIsRuptured()
        returns True if the elastin fiber in the chord has ruptured

    truth = c.isRuptured()
        returns True if either the collagen or elastin fibers has ruptured

    Uniform geometric fields associated with a chord in 3 space.
    For those fields that are constructed from difference formulae, it is
    necessary that they be rotated into the re-indexed co-ordinate frame for
    the 'state' of interest.

    a = c.area()
        returns the current cross-sectional area of the chord, i.e., both the
        collagen and elastin fibers, deformed under an assumption that chordal
        volume is preserved

    ell = c.length()
        returns the current chordal length

    lambda_ = c.stretch()
        returns the current chordal stretch

    Kinematic fields associated with the centroid of a chord in 3 space are

    [x, y, z] = c.centroid()
        returns the current coordinates for the chordal mid-point

    [ux, uy, uz] = c.displacement(reindex)
    output
            current displacement of the centroid

    Dmtx = sf.dDisplacement(reindex)
       input
            reindex is an instance of Pivot object from module pivotIncomingF      
       output
            Dmtx is change in displacement ( dA = L * D ) in the contribution 
            to nonlinear strain
    
    [vx, vy, vz] = c.velocity(reindex)
       input
            reindex is an instance of Pivot object from module pivotIncomingF      
        output
            current velocity of the centroid

    [ax, ay, az] = c.acceleration(reindex)
       input
            reindex is an instance of Pivot object from module pivotIncomingF      
        output
            current acceleration of the centroid

    Rotation and spin of a chord wrt their dodecahedral coordinate system are

    pMtx = c.rotation()
        returns a 3x3 orthogonal matrix that rotates the reference base vectors
        (E_1, E_2, E_3) into a set of local base vectors (e_1, e_2, e_3) that
        pertain to a chord whose axis aligns with the e_1 direction, while the
        e_2 direction passes through the origin of the dodecahedral reference
        co-ordinate system (E_1, E_2, E_3).

    omegaMtx = c.spin()
        output
            a 3x3 skew symmetric matrix that describes the time rate of change
            in rotation, i.e., the spin of the local chordal co-ordinate system
            (e_1, e_2, e_3) about a fixed co-ordinate frame (E_1, E_2, E_3)
            belonging to the dodecahedron.

    Thermodynamic fields evaluated in the chordal co-ordinate system:

    Fields uniform along the length of chord:

    epsilon = c.strain()
        returns the current logarithmic strain of the chord

    f = c.force()
        returns the current force carried by the chord

    T = c.temperature()
        returns the current temperature of the chord in centegrade

    Fields extrapolated out to the nodal points

    sigmaN1, sigmaN2 = c.stress()
        returns stress carried by the chord extrapolated to its nodal points

    etaN1, etaN2 = c.entropy()
        returns entropy (actual, not density) of the chord extrpolated to its
        nodal points

    Kinematic fields evaluated in the chordal co-ordinate system:

    gMtx = c.G(atGaussPt)
        returns the current displacement gradient G at a specified Gauss point.
        gMtx is a 1x1 matrix for a chord.

    fMtx = c.F(atGaussPt)
        returns the current deformation gradient F at a specified Gauss point.
        fMtx is a 1x1 matrix for a chord.

    lMtx = c.L(atGaussPt)
        returns the current velocity gradient L at a specified Gauss point.
        lMtx is a 1x1 matrix for a chord.

    Fields needed to construct a finite element solution strategy are:

    sf = c.shapeFunction(atGaussPt):
        returns the shape function associated with a specified Gauss point.

    gq = c.gaussQuadrature()
        returns the Gauss quadrature rule being used for spatial integration.

    mMtx = c.massMatrix()
        returns an average of the lumped and consistent mass matrices (thereby
        ensuring that the mass matrix is not singular) for a chord whose mass
        density, rho, and whose cross-sectional area are considered to be
        uniform over the length of the chord. This mass matrix is constant and
        therefore independent of state.

    cMtx = c.tangentStiffnessMtxC()
        reindex is an instance of Pivot object from module pivotIncomingF
        returns a tangent stiffness matrix for the chosen number of Gauss
        points belonging to the current state.  
        
    kMtx = c.secantStiffnessMtxK(reindex)
        reindex is an instance of Pivot object from module pivotIncomingF
        returns a secant stiffness matrix for the chosen number of Gauss
        points belonging to the current state. 

    fVec = c.forcingFunction()
        returns a vector describing the forcing function on the right-hand side
        belonging to the current state.  An updated Lagrangian formulation is
        implemented.
"""


class Chord(object):

    # constructor

    def __init__(self, number, vertex1, vertex2, dt,
                 diaCollagen=None, diaElastin=None, m=1):
        # verify the inputs
        if isinstance(number, int):
            self._number = number
        else:
            raise RuntimeError("The chord number must be an integer.")

        if not isinstance(vertex1, Vertex):
            raise RuntimeError('vertex1 sent to the chord '
                               + 'constructor was not of type Vertex.')
        if not isinstance(vertex2, Vertex):
            raise RuntimeError('vertex2 sent to the chord '
                               + 'constructor was not of type Vertex.')

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
            raise RuntimeError("The timestep size dt sent to the chord "
                               + "constructor must exceed machine precision.")

        # assign the Gauss quadrature rule to be used
        self._gq = GaussQuadrature()

        # initialize the chordal lengths for all configurations
        x1 = self._vertex[1].coordinates('ref')
        x2 = self._vertex[2].coordinates('ref')

        L0 = math.sqrt((x2[0] - x1[0])**2 
                       + (x2[1] - x1[1])**2
                       + (x2[2] - x1[2])**2)
        self._L0 = L0
        self._Lc = L0
        self._Ln = L0
        
        # base vector 1: aligns with the axis of the chord
        x = x2[0] - x1[0]
        y = x2[1] - x1[1]
        z = x2[2] - x1[2]
        mag = math.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1[0] + x2[0]) / 2.0
        y = (x1[1] + x2[1]) / 2.0
        z = (x1[2] + x2[2]) / 2.0
        mag = math.sqrt(x * x + y * y + z * z)
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

        # create base vector 2 (the radial vector out to the chord)
        x = ex + delta * n1x
        y = ey + delta * n1y
        z = ez + delta * n1z
        mag = math.sqrt(x * x + y * y + z * z)
        n2x = x / mag
        n2y = y / mag
        n2z = z / mag

        # base vector 3 (the binormal) is obtained through a cross product
        n3x = n1y * n2z - n1z * n2y
        n3y = n1z * n2x - n1x * n2z
        n3z = n1x * n2y - n1y * n2x

        # create the rotation matrices: rotate dodecahedral into chordal
        self._Pr3D = np.identity(3, dtype=float)
        
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

        # determine vertice coordinates in the chordal frame of reference
        self._v1x0 = n1x * x1[0] + n1y * x1[1] + n1z * x1[2]
        self._v1y0 = n2x * x1[0] + n2y * x1[1] + n2z * x1[2]
        self._v2x0 = n1x * x2[0] + n1y * x2[1] + n1z * x2[2]
        self._v2y0 = n2x * x2[0] + n2y * x2[1] + n2z * x2[2]

        # initialize current vertice coordinates in chordal frame of reference
        self._v1x = self._v1x0
        self._v1y = self._v1y0
        self._v2x = self._v2x0
        self._v2y = self._v2y0
        
        # z offsets
        self._v1z = n3x * x1[0] + n3y * x1[1] + n3z * x1[2]
        self._v2z = n3x * x2[0] + n3y * x2[1] + n3z * x2[2]

        self._vz0 = ((self._v1z + self._v2z) / 2.0)        
        
        # determine the centroid of this tetrahedron       
        self._cx0 = (self._v1x0 + self._v2x0) / 2.0
        self._cy0 = (self._v1y0 + self._v2y0) / 2.0
        self._cz0 = self._vz0 

        # rotate this centroid back into the reference coordinate system
        self._centroidX0 = n1x * self._cx0 + n2x * self._cy0 + n3x * self._cz0
        self._centroidY0 = n1y * self._cx0 + n2y * self._cy0 + n3y * self._cz0
        self._centroidZ0 = n1z * self._cx0 + n2z * self._cy0 + n3z * self._cz0

        # initialize the centroids for all configurations              
        self._centroidXp = self._centroidX0
        self._centroidYp = self._centroidY0
        self._centroidZp = self._centroidZ0
        self._centroidXc = self._centroidX0
        self._centroidYc = self._centroidY0
        self._centroidZc = self._centroidZ0
        self._centroidXn = self._centroidX0
        self._centroidYn = self._centroidY0
        self._centroidZn = self._centroidZ0

        # rotation matrices: from dodecahedral frame into pentagonal frame
        self._Pp3D = np.zeros((3, 3), dtype=float)
        self._Pc3D = np.zeros((3, 3), dtype=float)
        self._Pn3D = np.zeros((3, 3), dtype=float)
        self._Pp3D[:, :] = self._Pr3D[:, :]
        self._Pc3D[:, :] = self._Pr3D[:, :]
        self._Pn3D[:, :] = self._Pr3D[:, :]

        
        # establish the shape functions located at the various Gauss points
        atGaussPt = 1
        sf1 = ShapeFunction(self._gq.coordinates(atGaussPt))
        atGaussPt = 2
        sf2 = ShapeFunction(self._gq.coordinates(atGaussPt))
        self._shapeFns = {
            1: sf1,
            2: sf2
        }

        # create the displacement and deformation gradients for a chord at
        # their Gauss points via dictionaries.  '0' implies reference,
        # 'p' implies previous, 'c' implies current, and 'n' implies next.

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
        
        E1_c, E2_c, et_c, ef_c, s0_c = mp.collagenFiber()
        E1_e, E2_e, et_e, s0_e = mp.elastinFiber()

        # create constitutive solvers for each Gauss point of a chord
        nbrVars = 2   # for a chord they are: temperature and length
        respVars = 4
        T0 = 37.0     # body temperature in centigrade
        # thermodynamic strains (thermal and mechanical) are 0 at reference
        eVec0 = np.zeros((nbrVars,), dtype=float)
        # physical variables have reference values of
        xVec0 = np.zeros((nbrVars,), dtype=float)
        # vector of thermodynamic response variables
        yVec0 = np.zeros((respVars,), dtype=float)
        yVec0[1] = s0_c
        yVec0[3] = s0_e
        
        xVec0[0] = T0  # temperature in centigrade
        xVec0[1] = L0  # length in centimeters
        self._control = {
            1: ControlFiber(eVec0, xVec0, dt),
            2: ControlFiber(eVec0, xVec0, dt)
        }
        self._response = {
            1: SeptalChord(eVec0, xVec0, yVec0),
            2: SeptalChord(eVec0, xVec0, yVec0)
        }
        self._solver = {
            1: PECE(self._control[1], self._response[1], m),
            2: PECE(self._control[2], self._response[2], m)
        }

        self._Ms = {
            1: self._response[1].mixedSecantModulus(eVec0, xVec0, yVec0),
            2: self._response[2].mixedSecantModulus(eVec0, xVec0, yVec0)
        }
        
        self._Mt = {
            1: self._response[1].mixedtangentModulus(eVec0, xVec0, yVec0),
            2: self._response[2].mixedtangentModulus(eVec0, xVec0, yVec0)
        }        

        self._trac = {
            1: self._response[1].traction(),
            2: self._response[2].traction()
        }  

        
        return  # a new chord object

    # local methods

    def __str__(self):
        return self.toString()
    
   # uniform or averaged geometric properties of the chord

    def area(self):
        a = ((self._response[1].A0_c + self._response[2].A0_c
              + self._response[1].A0_e + self._response[2].A0_e)
             * self._L0 / (2.0 * self._Lc))
        return a

    def length(self):
        return self._Lc

    def massDensity(self):
        v1 = self._response[1].volume()
        v2 = self._response[2].volume()
        m1 = self._response[1].massDensity() * v1
        m2 = self._response[2].massDensity() * v2
        rho = (m1 + m2) / (v1 + v2)
        return rho
    

    # These FE arrays are evaluated at the beginning of the current step of
    # integration, i.e., they associate with an updated Lagrangian formulation.

    def _massMatrix(self):
        # create the returned mass matrix
        mMtx = np.zeros((2, 2), dtype=float)

        # construct the consistent mass matrix in natural co-ordinates
        massC = np.zeros((2, 2), dtype=float)
        NtN = np.zeros((2, 2), dtype=float)
        for i in range(1, self._gq.gaussPoints()+1):
            sfn = self._shapeFns[i]
            wgt = self._gq.weight(i)
            NtN += wgt * np.matmul(np.transpose(sfn.Nmtx), sfn.Nmtx)
        massC[:, :] = NtN[:, :]

        # construct the lumped mass matrix in natural co-ordinates
        massL = np.zeros((2, 2), dtype=float)
        row, col = np.diag_indices_from(massC)
        massL[row, col] = massC.sum(axis=1)

        # constrcuct the averaged mass matrix in natural co-ordinates
        massA = np.zeros((2, 2), dtype=float)
        massA = 0.5 * (massC + massL)

        # the following print statements were used to verify the code
        # print("\nThe averaged mass matrix in natural co-ordinates is")
        # print(0.5 * massA)  # the half is the Jacobian for span [-1, 1]

        # convert average mass matrix from natural to physical co-ordinates
        
        area = self.area()

        # assign coordinates at the vertices in the reference configuration
        x01 = (self._v1x0, self._v1y0)
        x02 = (self._v2x0, self._v2y0)
        
        Jdet = sfn.jacobianDeterminant(x01, x02)
        rho = self.massDensity()
        mMtx = (rho * area * Jdet) * massA     

        return mMtx

    def _tangentStiffnessMtxC(self):
        cMtx = np.zeros((2, 2), dtype=float)

        # current vertex coordinates in pentagonal frame of reference
        xn1 = (self._v1x, self._v1y)
        xn2 = (self._v2x, self._v2y)

        # assign coordinates at the vertices in the reference configuration
        x01 = (self._v1x0, self._v1y0)
        x02 = (self._v2x0, self._v2y0)
        
        # cross-sectional area of the chord (both collagen and elastin fibers)
        area = self.area()
        
        cs1 = np.zeros((2, 2), dtype=float)
        ct1 = np.zeros((2, 2), dtype=float)
            
        for i in range(1, self._gq.gaussPoints()+1):
            sfn = self._shapeFns[i]
            wgt = self._gq.weight(i)
            Mt = self._Mt[i]
            Ss = self._trac[i]

            Hmtx = sfn.H(xn1, xn2)
            BLmtx = sfn.BL(xn1, xn2)
            BNmtx = sfn.BN(xn1, xn2, x01, x02)
                    
            
            cs1 += wgt * Hmtx.T.dot(Ss).dot(Hmtx)
            
            ct1 += wgt * (BLmtx.T.dot(Mt[1, 1]).dot(BLmtx) 
                        + BLmtx.T.dot(Mt[1, 1]).dot(BNmtx)
                        + BNmtx.T.dot(Mt[1, 1]).dot(BLmtx) 
                        + BNmtx.T.dot(Mt[1, 1]).dot(BNmtx) )
            
        # determinant of jacobian matrix
        Jdet = sfn.jacobianDeterminant(x01, x02)
        
        Cs = np.zeros((2, 2), dtype=float)
        Ct = np.zeros((2, 2), dtype=float)
    
        # the tangent stiffness matrix Cs
        Cs = area * Jdet * cs1

        # the tangent stiffness matrix Ct
        Ct = area * Jdet * ct1

        # determine the total tangent stiffness matrix
        cMtx = Cs + Ct

        return cMtx
    
    
    def _secantStiffnessMtxK(self, reindex):
        kMtx = np.zeros((2, 2), dtype=float)

        # current vertex coordinates in pentagonal frame of reference
        xn1 = (self._v1x, self._v1y)
        xn2 = (self._v2x, self._v2y)

        # assign coordinates at the vertices in the reference configuration
        x01 = (self._v1x0, self._v1y0)
        x02 = (self._v2x0, self._v2y0)
        
        # cross-sectional area of the chord (both collagen and elastin fibers)
        area = self.area()
        
        ks1 = np.zeros((2, 2), dtype=float)
        kt1 = np.zeros((2, 2), dtype=float)
            
        for i in range(1, self._gq.gaussPoints()+1):
            sfn = self._shapeFns[i]
            wgt = self._gq.weight(i)
            Ms = self._Ms[i]
            Mt = self._Mt[i]

            Hmtx = sfn.H(xn1, xn2)
            Lmtx = sfn.L(xn1, xn2)
            BLmtx = sfn.BL(xn1, xn2)
            BNmtx = sfn.BN(xn1, xn2, x01, x02)
            A = sfn.A(xn1, xn2, x01, x02)  
            Dmtx = self.dDisplacement(reindex)
                    
            dA = np.dot(Lmtx, np.transpose(Dmtx))            
            dSt = A.T.dot(Mt[1, 1]).dot(dA)
            
            
            ks1 += wgt * (BLmtx.T.dot(Ms[1, 1]).dot(BLmtx) 
                        + BLmtx.T.dot(Ms[1, 1]).dot(BNmtx) 
                        + BNmtx.T.dot(Ms[1, 1]).dot(BLmtx) 
                        + BNmtx.T.dot(Ms[1, 1]).dot(BNmtx) )
            
            kt1 += wgt * Hmtx.T.dot(dSt).dot(Hmtx)
            
        # determinant of jacobian matrix
        Jdet = sfn.jacobianDeterminant(x01, x02)
        
        Ks = np.zeros((2, 2), dtype=float)
        Kt = np.zeros((2, 2), dtype=float)

        # the secant stiffness matrix Ks
        Ks = area * Jdet * ks1

        # the stress stiffness matrix for 1 Gauss point
        Kt = area * Jdet * kt1

        # determine the total secant stiffness matrix
        kMtx = Ks + Kt

        return kMtx

    def _forcingFunction(self):
        fVec = np.zeros((2,1), dtype=float)
        
        # current vertex coordinates in pentagonal frame of reference
        xn1 = (self._v1x, self._v1y)
        xn2 = (self._v2x, self._v2y)

        # assign coordinates at the vertices in the reference configuration
        x01 = (self._v1x0, self._v1y0)
        x02 = (self._v2x0, self._v2y0)
        
        area = self.area()
        N1 = np.zeros((2, 1), dtype=float)
        BL1 = np.zeros((2, 1), dtype=float)
        BN1 = np.zeros((2, 1), dtype=float)
        
        for i in range(1, self._gq.gaussPoints()+1):
            sfn = self._shapeFns[i]
            wgt = self._gq.weight(i)
            t = self._trac[i]
            N1 += wgt * np.transpose(sfn.Nmtx) * t 
            BL1 += wgt * np.transpose(sfn.BL(xn1, xn2))   
            BN1 += wgt * np.transpose(sfn.BN(xn1, xn2, x01, x02))
            B = np.add(BL1, BN1) * t
        
        # determinant of jacobian matrix
        Jdet = sfn.jacobianDeterminant(x01, x02)
        
        FBc = np.zeros((2, 1), dtype=float)
        F0 = np.zeros((2, 1), dtype=float)
        
        FBc = Jdet * N1
                    
        F0 = Jdet * area * B
        
        fVec = FBc - F0
            
        return fVec

    def toString(self, state):
        if self._number < 10:
            s = 'chord[0'
        else:
            s = 'chord['
        s = s + str(self._number)
        s = s + '] has vertices: \n'
        s = s + '   ' + self._vertex[1].toString(state) + '\n'
        s = s + '   ' + self._vertex[2].toString(state)
        return s

    def number(self):
        return self._number

    def vertexNumbers(self):
        n1 = self._vertex[1].number()
        n2 = self._vertex[2].number()
        return n1, n2

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
            raise RuntimeError('Vertex {} '.format(number) + "does not "
                               + 'belong to chord {}.'.format(self._number))

    def solver(self, atGaussPt):
        if atGaussPt == 1 or atGaussPt == 2:
            return self._solver[atGaussPt]
        else:
            raise RuntimeError("Argument atGaussPt must be either 1 or 2.")

    def update(self, temperature=37.0, restart=False):
        # determine length of the chord in the next configuration
        x1 = self._vertex[1].coordinates('next')
        x2 = self._vertex[2].coordinates('next')

        # base vector 1: aligns with the axis of the chord
        x = x2[0] - x1[0]
        y = x2[1] - x1[1]
        z = x2[2] - x1[2]
        mag = math.sqrt(x * x + y * y + z * z)
        n1x = x / mag
        n1y = y / mag
        n1z = z / mag

        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1[0] + x2[0]) / 2.0
        y = (x1[1] + x2[1]) / 2.0
        z = (x1[2] + x2[2]) / 2.0
        mag = math.sqrt(x * x + y * y + z * z)
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
        mag = math.sqrt(x * x + y * y + z * z)
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

        # determine vertice coordinates 
        self._v1x = n1x * x1[0] + n1y * x1[1] + n1z * x1[2]
        self._v1y = n2x * x1[0] + n2y * x1[1] + n2z * x1[2]
        self._v2x = n1x * x2[0] + n1y * x2[1] + n1z * x2[2]
        self._v2y = n2x * x2[0] + n2y * x2[1] + n2z * x2[2]


        # z offsets 
        self._v1z = n3x * x1[0] + n3y * x1[1] + n3z * x1[2]
        self._v2z = n3x * x2[0] + n3y * x2[1] + n3z * x2[2]

        self._Ln  = math.sqrt((self._v2x - self._v1x)**2 
                              + (self._v2y - self._v1y)**2
                              + (self._v2z - self._v1z)**2)
        
        # determine the centroid of this tetrahedron       
        self._cx = (self._v1x + self._v2x) / 2.0
        self._cy = (self._v1y + self._v2y) / 2.0
        self._cz = (self._v1z + self._v2z) / 2.0  

        # rotate this centroid back into the reference coordinate system
        self._centroidXn = n1x * self._cx + n2x * self._cy + n3x * self._cz
        self._centroidYn = n1y * self._cx + n2y * self._cy + n3y * self._cz
        self._centroidZn = n1z * self._cx + n2z * self._cy + n3z * self._cz
        
        # current vertex coordinates 
        x1 = (self._v1x, self._v1y)
        x2 = (self._v2x, self._v2y)

        # reference vertex coordinates 
        x10 = (self._v1x0, self._v1y0)
        x20 = (self._v2x0, self._v2y0)
        

        # quantify the displacement and deformation gradients of the chord
        # displacement gradients located at the Gauss points of a chord
        self._Gn[1] = self._shapeFns[1].G(x1, x2, x10, x20)
        self._Gn[2] = self._shapeFns[2].G(x1, x2, x10, x20)
        # deformation gradients located at the Gauss points of a chord
        self._Fn[1] = self._shapeFns[1].F(x1, x2, x10, x20)
        self._Fn[2] = self._shapeFns[2].F(x1, x2, x10, x20)

        # integrate the constitutive equations
        xVec = np.zeros((2,), dtype=float)
        xVec[0] = temperature
        xVec[1] = self._Ln
        self._solver[1].integrate(xVec, restart)
        self._solver[2].integrate(xVec, restart)
        return  # nothing, the data structure has been updated

    def advance(self, reindex):
        # assign current to previous values, and then next to current values
        self._centroidXp = self._centroidXc
        self._centroidYp = self._centroidYc
        self._centroidZp = self._centroidZc
        self._centroidXc = self._centroidXn
        self._centroidYc = self._centroidYn
        self._centroidZc = self._centroidZn
        self._L0 = self._Lc
        self._Lc = self._Ln
        self._Pp3D[:, :] = self._Pc3D[:, :]
        self._Pc3D[:, :] = self._Pn3D[:, :]

        # advance the matrix fields associated with each Gauss point
        for i in range(1, self._gq.gaussPoints()+1):
            self._Fp[i] = self._Fc[i]
            self._Fc[i] = self._Fn[i]

        # advance the integrators
        self._solver[1].advance()
        self._solver[2].advance()

        # compute the FE arrays needed for the next interval of integration
        self.mMtx = self._massMatrix()
        self.cMtx = self._tangentStiffnessMtxC()
        self.kMtx = self._secantStiffnessMtxK(reindex)
        self.fVec = self._forcingFunction()
        return  # nothing, the data structure has been advanced

    # material properties of the chord

    def collagenIsRuptured(self):
        (ruptured,) = self._response[1].bioFiberCollagen.isRuptured()
        if ruptured:
            return ruptured
        (ruptured,) = self._response[2].bioFiberCollagen.isRuptured()
        return ruptured

    def elastinIsRuptured(self):
        (ruptured,) = self._response[1].bioFiberElastin.isRuptured()
        if ruptured:
            return ruptured
        (ruptured,) = self._response[2].bioFiberElastin.isRuptured()
        return ruptured

    def isRuptured(self):
        if (self.collagenIsRuptured() is True
           or self.elastinIsRuptured() is True):
            return True
        else:
            return False

    def stretch(self):
        lambda_ = self._Lc / self._L0
        return lambda_

    # kinematic fields associated with the chordal centroid in 3 space

    def centroid(self):
        cx = self._centroidXn
        cy = self._centroidYn
        cz = self._centroidZn
        return np.array([cx, cy, cz])

    def displacement(self, reindex):
        
        u1 = self._vertex[1].displacement(reindex, 'curr')
        u2 = self._vertex[2].displacement(reindex, 'curr')
        
        R = self.rotation()
        u1x = R[0,0] * u1[0] + R[1,0] * u1[1] + R[2,0] * u1[2]
        u1y = R[0,1] * u1[0] + R[1,1] * u1[1] + R[2,1] * u1[2]
        u1z = R[0,2] * u1[0] + R[1,2] * u1[1] + R[2,2] * u1[2]
        u2x = R[0,0] * u2[0] + R[1,0] * u2[1] + R[2,0] * u2[2]
        u2y = R[0,1] * u2[0] + R[1,1] * u2[1] + R[2,1] * u2[2]
        u2z = R[0,2] * u2[0] + R[1,2] * u2[1] + R[2,2] * u2[2]
        
        u1r = np.zeros((1, 3), dtype=float)
        u2r = np.zeros((1, 3), dtype=float)
        
        u1r[0,0] = u1x
        u1r[0,1] = u1y
        u1r[0,2] = u1z
        
        u2r[0,0] = u2x
        u2r[0,1] = u2y
        u2r[0,2] = u2z
        
        u = 0.5 * (u1r + u2r)        
        return u
    
    # determine the change in displacement ( dA = L * D )
    def dDisplacement(self, reindex):
        v1 = self._vertex[1].velocity(reindex, 'curr')
        v2 = self._vertex[2].velocity(reindex, 'curr')
        
        Dmtx = np.zeros((1, 2), dtype=float)
        
        R = self.rotation()
        Dmtx[0, 0] = R[0,0] * v1[0] + R[1,0] * v1[1] + R[2,0] * v1[2]
        Dmtx[0, 1] = R[0,0] * v2[0] + R[1,0] * v2[1] + R[2,0] * v2[2]
        
        return Dmtx
    
    def velocity(self, reindex):
        v1 = self._vertex[1].velocity(reindex, 'curr')
        v2 = self._vertex[2].velocity(reindex, 'curr')
        
        R = self.rotation()
        v1x = R[0,0] * v1[0] + R[1,0] * v1[1] + R[2,0] * v1[2]
        v1y = R[0,1] * v1[0] + R[1,1] * v1[1] + R[2,1] * v1[2]
        v1z = R[0,2] * v1[0] + R[1,2] * v1[1] + R[2,2] * v1[2]
        v2x = R[0,0] * v2[0] + R[1,0] * v2[1] + R[2,0] * v2[2]
        v2y = R[0,1] * v2[0] + R[1,1] * v2[1] + R[2,1] * v2[2]
        v2z = R[0,2] * v2[0] + R[1,2] * v2[1] + R[2,2] * v2[2]
        
        v1r = np.zeros((1, 3), dtype=float)
        v2r = np.zeros((1, 3), dtype=float)
        
        v1r[0,0] = v1x
        v1r[0,1] = v1y
        v1r[0,2] = v1z
        
        v2r[0,0] = v2x
        v2r[0,1] = v2y
        v2r[0,2] = v2z
        
        v = 0.5 * (v1r + v2r)         
        
        return v

    def acceleration(self, reindex):
        a1 = self._vertex[1].acceleration(reindex, 'curr')
        a2 = self._vertex[2].acceleration(reindex, 'curr')
        
        R = self.rotation()
        a1x = R[0,0] * a1[0] + R[1,0] * a1[1] + R[2,0] * a1[2]
        a1y = R[0,1] * a1[0] + R[1,1] * a1[1] + R[2,1] * a1[2]
        a1z = R[0,2] * a1[0] + R[1,2] * a1[1] + R[2,2] * a1[2]
        a2x = R[0,0] * a2[0] + R[1,0] * a2[1] + R[2,0] * a2[2]
        a2y = R[0,1] * a2[0] + R[1,1] * a2[1] + R[2,1] * a2[2]
        a2z = R[0,2] * a2[0] + R[1,2] * a2[1] + R[2,2] * a2[2]
        
        a1r = np.zeros((1, 3), dtype=float)
        a2r = np.zeros((1, 3), dtype=float)
        
        a1r[0,0] = a1x
        a1r[0,1] = a1y
        a1r[0,2] = a1z
        
        a2r[0,0] = a2x
        a2r[0,1] = a2y
        a2r[0,2] = a2z
        
        a = 0.5 * (a1r + a2r)         
        
        return a

    # rotation and spin of chord wrt dodecahedral coordinate system

    def rotation(self):
        return np.copy(self._Pc3D)

    def spin(self):
        return spinMtx.currSpin(self._Pp3D, self._Pc3D, self._Pn3D, self._h)

    # thermodynamic fields associated with a chord that are uniform in space

    def strain(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return m.log(self._Lc / self._L0)
            elif state == 'n' or state == 'next':
                return m.log(self._Ln / self._L0) 
            elif state == 'p' or state == 'prev' or state == 'previous':
                return m.log(self._Ln / self._L0) 
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state of {} ".format(state)
                                   + "was sent in a call to Chord.strain.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state))
                               + "was sent in a call to Chord.strain.")

    def force(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._fc
            elif state == 'n' or state == 'next':
                return self._fn
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._fp
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._f0
            else:
                raise RuntimeError("An unknown state of {} ".format(state)
                                   + "was sent in a call to Chord.force.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state))
                               + "was sent in a call to Chord.force.")

    def temperature(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._Tc
            elif state == 'n' or state == 'next':
                return self._Tn
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._Tp
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._T0
            else:
                raise RuntimeError("An unknown state of {} was ".format(state)
                                   + "sent in a call to Chord.temperature.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state))
                               + "was sent in a call to Chord.temperature.")

    # non-uniform fields that are extrapolated out to the nodal points

    def entropy(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._etaC
            elif state == 'n' or state == 'next':
                return self._etaN
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._etaP
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._eta0
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to chord.entropy.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in a call to chord.entropy.")

    # fundamental kinematic fields

    # displacement gradient at a Gauss point
    def G(self, gaussPt, state):
        if gaussPt != self._gq.gaussPoints():
            raise RuntimeError("gaussPt can only be {} in a " +
                               "{} in call to " .format(self._gq.gaussPoints()) +
                                   "chord.G and you sent " +
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
        if gaussPt != self._gq.gaussPoints():
            raise RuntimeError("gaussPt can only be {} in a " +
                               "{} in call to " .format(self._gq.gaussPoints()) +
                                   "chord.G and you sent " +
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
        if gaussPt != self._gq.gaussPoints():
            raise RuntimeError("gaussPt can only be {} in a " +
                               "{} in call to " .format(self._gq.gaussPoints()) +
                                   "chord.G and you sent " +
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
        if gaussPt != self._gq.gaussPoints():
            raise RuntimeError("gaussPt can only be {} in a " +
                               "{} in call to " .format(self._gq.gaussPoints()) +
                                   "chord.G and you sent " +
                                   "{}.".format(gaussPt))
        sf = self._shapeFns[gaussPt]
        return sf

    def gaussQuadrature(self):
        return self._gq

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
