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
__update__ = "07-21-2020"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


Class Chord in file chords.py allows for the creation of objects that are to
be used to represent chords that connect vertices in a polyhedron.  Each chord
is assigned an unique number, two distinct vertices that serve as its end
points, and the time-step size used to approximate derivatives and integrals.

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

Most fields (methods) are evaluated at the end of the current interval of
integration, i.e., at the next configuration.  However, the mass matrix,
stiffness matrix, and forcing function, which are information supplied to the
finite element solver, are evaluated at the beginning of the current interval
of integration, viz., in the sense of an updated Lagrangian formulation.


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
    If the fiber diameters retain their default setting of None, then they will
    be assigned via a random distribution; otherwise, they must lie between
    values of 2 and 7.5 microns, which are their physiologic range.

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
        returns a PECE solver object used for integrating the chordal response
        at the specified Gauss point, as extablished in ceChords.py using the
        PECE integrator of peceHE.py

    c.update(temperature, restart=False)
        assigns the temperature (in centigrade) for the end of the integration
        step, and specifies if the integrators are to be restarted, e.g.,
        because of a discontinuity in a control field.  It also assigns new
        co-ordinate values to the chord for its next spatial location and
        updates all affected fields.  Method update is to be called AFTER all
        vertices have had their co-ordinates updated.  This method may be
        called multiple times before freezing its co-ordinate values with a
        call to c.advance.  This method calls its PECE integrator to integrate
        the constitutive state at the two Gauss points of the chord.

    c.advance()
        assigns the current fields to the previous fields, and then assigns
        the next fields to the current fields, thereby freezing the present
        next-fields in preparation for an advancment of a solution along its
        path of motion.  Also, the mass matrix, stiffness matrix, and forcing
        function are created for the next step of integration.  This method
        calls the analogous method for its two PECE integrators.

    Material properties that associate with a chord.

    rho = c.massDensity()
        returns the mass density of the chord (its collagen and elastin fibers)

    truth = c.isRuptured()
        returns True if either the collagen or elastin fibers has ruptured

    Uniform fields associated with a chord in 3 space.

    ell = c.length()
        returns chordal length at the end of the integration step

    lambda_ = c.stretch()
        returns chordal stretch at the end of the integration step

    e = c.strain()
        returns logarithmic chordal strain at the end of the integration step

    f = c.force()
        returns force carried by the chord at the end of the integration step

    T = c.temperature()
        returns temperature of the chord at the end of the integration step

    Kinematic fields associated with the centroid of a chord in 3 space.
    For those fields that are constructed from difference formulae, it is
    necessary that they be rotated into the re-indexed co-ordinate frame
    appropriate for the end of the integration step.

    [x, y, z] = c.centroid()
        returns the co=ordinates for the chordal mid-point at the end of the
        integration step

    [ux, uy, uz] = c.displacement(reindex)
        input
            reindex is an instance of class Pivot from pivotIncomingF.py
        output
            centroid's displacement vector at the end of the integration step

    [vx, vy, vz] = c.velocity(reindex)
        input
            reindex is an instance of class Pivot from pivotIncomingF.py
        output
            centroid's velocity vector at the end of the integration step

    [ax, ay, az] = c.acceleration(reindex)
        input
            reindex is an instance of class Pivot from pivotIncomingF.py
        output
            centroid's acceleration vector at the end of the integration step

    Rotation and spin of a chord wrt their dodecahedral co-ordinate system are

    pMtx = c.rotation()
        returns a 3x3 orthogonal matrix that rotates the reference base vectors
        (E_1, E_2, E_3) into a set of local base vectors (e_1, e_2, e_3) that
        pertain to a chord whose axis aligns with the e_1 direction, while the
        e_2 direction passes through the origin of the dodecahedral reference
        co-ordinate system (E_1, E_2, E_3).  It is the rotation at the end of
        the current interval of integration.

    omegaMtx = c.spin(reindex)
        input
            reindex is an instance of class Pivot from pivotIncomingF.py
        output
            a 3x3 skew symmetric matrix that describes the time rate of change
            in rotation, i.e., the spin of the local chordal co-ordinate system
            (e_1, e_2, e_3) about a fixed co-ordinate frame (E_1, E_2, E_3)
            belonging to the dodecahedron.  It is the spin at the end of the
            current interval of integration.

    Thermodynamic fields evaluated in the chordal co-ordinate system:

    Fields extrapolated out to the nodal points

    aN1, aN2 = c.nodalAreas()
        returns the nominal cross-sectional area of the chord extrapolated to
        its nodal points, i.e., a mixture for the areas of collagen and elastin
        fibers that are load bearing, deformed under an assumption that chordal
        volume is preserved

    sigmaN1, sigmaN2 = c.nodalStresses()
        returns the nominal stress carried by the chord extrapolated to its
        nodal points, i.e., a mixture of the stresses evaluated at the end of
        the current interval of integration carried by the collagen and elastin
        fibers, constructed such that: force = sigmaN1*aN1 = simgaN2*aN2

    etaN1, etaN2 = c.nodalEntropies()
        returns the nominal entropy (actual, not density) of the chord extrpo-
        lated to its nodal points, i.e., a mixture of the entropies within its
        collagen and elastin fibers

    Kinematic fields of a continum evaluated in the chordal co-ordinate system:

    gMtx = c.G(atGaussPt)
        returns the displacement gradient G at a specified Gauss point at the
        end of the current interval of integration.  gMtx is a 1x1 matrix for
        a chord

    fMtx = c.F(atGaussPt)
        returns the deformation gradient F at a specified Gauss point at the
        end of the current interval of integration.  fMtx is a 1x1 matrix for
        a chord

    lMtx = c.L(atGaussPt)
        returns the velocity gradient L at a specified Gauss point at the end
        of the current interval of integration.  lMtx is a 1x1 matrix for a
        chord.

    Fields needed to construct a finite element model of chords.  These fields
    are evaluated at the beginning of the interval of integration, consistent
    with an updated Lagrangian formulation.

    sf = c.shapeFunction(atGaussPt):
        returns the object managing shape functions for a chord, as associated
        with a specified Gauss point

    gq = c.gaussQuadrature()
        returns the object managing the Gauss quadrature rule for a chord and
        used for spatial integration

    mMtx = c.massMatrix()
        returns an average of the lumped and consistent mass matrices (thereby
        ensuring that the mass matrix is not singular) for a chord whose mass
        density, rho, and whose cross-sectional area are considered to be
        uniform (nominal) over the length of the chord.  This mass matrix is
        constant and therefore independent of state, therefore, it does not
        need to be updated

    kMtx = c.stiffnessMatrix()
        returns a tangent stiffness matrix for the chosen number of Gauss
        points belonging to the current state.  The constitutive response
        comes from module ceChords.py

    fVec = c.forcingFunction()
        returns a vector describing the forcing function on the right-hand side
        of the FE system of equations
"""


class Chord(object):

    # constructor

    def __init__(self, number, vertex1, vertex2, dt,
                 diaCollagen=None, diaElastin=None, m=1):
        self.GAUSS_PTS = 2
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

        # create the four rotation matrices: rotate dodecahedral into chordal
        self._Pr3D = np.identity(3, dtype=float)
        self._Pp3D = np.identity(3, dtype=float)
        self._Pc3D = np.identity(3, dtype=float)
        self._Pn3D = np.identity(3, dtype=float)

        # initialize the chordal lengths for all configurations
        x1 = self._vertex[1].coordinates('ref')
        x2 = self._vertex[2].coordinates('ref')
        L0 = m.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2
                    + (x2[2] - x1[2])**2)
        self._L0 = L0
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

        # base vector 2: goes from the co-ordinate origin to the chord
        # initial guess: base vector 2 points to the midpoint of the chord
        x = (x1[0] + x2[0]) / 2.0
        y = (x1[1] + x2[1]) / 2.0
        z = (x1[2] + x2[2]) / 2.0
        mag = m.sqrt(x * x + y * y + z * z)
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
        mag = m.sqrt(x * x + y * y + z * z)
        n2x = x / mag
        n2y = y / mag
        n2z = z / mag

        # base vector 3 (the binormal) is obtained through a cross product
        n3x = n1y * n2z - n1z * n2y
        n3y = n1z * n2x - n1x * n2z
        n3z = n1x * n2y - n1y * n2x

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
        self._G = {
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

        # create constitutive solvers for each Gauss point of a chord
        nCtrl = 2     # for a chord the controls are: temperature and length
        T0 = 37.0     # body temperature in degrees centigrade
        # thermo strains (both thermal and mechanical) are 0 at reference state
        eVec0 = np.zeros((nCtrl,), dtype=float)
        # physical control variables have reference values of zero
        xVec0 = np.zeros((nCtrl,), dtype=float)
        xVec0[0] = T0  # temperature in centigrade
        xVec0[1] = L0  # length in centimeters
        self._control = {
            1: ControlFiber(eVec0, xVec0, dt),
            2: ControlFiber(eVec0, xVec0, dt)
        }
        self._response = {
            # because fiber properties are assigned from random distributions
            # different constitutive parameters assign to the Gauss points
            1: SeptalChord(diaCollagen, diaElastin),
            2: SeptalChord(diaCollagen, diaElastin)
        }
        self._solver = {
            1: PECE(self._control[1], self._response[1], m),
            2: PECE(self._control[2], self._response[2], m)
        }

        self._massMtx = self._massMatrix()
        self._stiffMtx = self._stiffnessMatrix()
        self._forceFn = self._forcingFunction()
        return  # a new chord object

    # local methods

    def __str__(self):
        return self.toString()

    # These FE fields are evaluated at the beginning of the current step of
    # integration, i.e., they associate with an updated Lagrangian formulation.

    def _massMatrix(self):
        # create the returned mass matrix
        mMtx = np.zeros((2, 2), dtype=float)

        # construct the consistent mass matrix in natural co-ordinates
        massC = np.zeros((2, 2), dtype=float)
        NtN = np.zeros((2, 2), dtype=float)
        for i in range(1, 3):
            sfn = self._shapeFns[i]
            wgt = self._gq.weight(i)
            NtN += wgt * np.matmul(np.transpose(sfn.Nmtx), sfn.Nmtx)
        massC[:, :] = NtN[:, :]

        # construct the lumped mass matrix in natural co-ordinates
        massL = np.zeros((2, 2), dtype=float)
        row, col = np.diag_indices_from(massC)
        massL[row, col] = massC.sum(axis=1)

        # construct the averaged mass matrix in natural co-ordinates
        massA = np.zeros((2, 2), dtype=float)
        massA = 0.5 * (massC + massL)

        # the following print statements were used to verify the code
        # print("\nThe averaged mass matrix in natural co-ordinates is")
        # print(0.5 * massA)  # the half is the Jacobian for span [-1, 1]

        # convert average mass matrix from natural to physical co-ordinates
        length = self.length("ref")
        area = self.area('ref')
        xn1 = (-length / 2,)
        xn2 = (length / 2,)
        Jdet = self._shapeFns[1].jacobianDeterminant(xn1, xn2)


        rho = self.massDensity()
        mMtx = (rho * area * Jdet) * massA

        return mMtx

    def _stiffnessMatrix(self):
        kMtx = np.zeros((2, 2), dtype=float)
        return kMtx

    def _forcingFunction(self):
        fVec = np.zeros((2,), dtype=float)
        return fVec

    """
    def massMatrix(self):
        # use the following rule for Gauss quadrature
        gq = self.gaussQuadrature()

        # construct the consistent mass matrix in natural co-ordinates
        massC = np.zeros((2, 2), dtype=float)
        NtN = np.zeros((2, 2), dtype=float)
        for i in range(self.gaussPoints()):
            sf = self.shapeFunction(i+1)
            NtN += gq.weight(i+1) * np.matmul(np.transpose(sf.Nmtx), sf.Nmtx)
        massC[:, :] = NtN[:, :]

        # construct the lumped mass matrix in natural co-ordinates
        massL = np.zeros((2, 2), dtype=float)
        row, col = np.diag_indices_from(massC)
        massL[row, col] = massC.sum(axis=1)

        # construct the averaged mass matrix in natural co-ordinates
        massA = np.zeros((2, 2), dtype=float)
        massA = 0.5 * (massC + massL)

        # the following print statements were used to verify the code
        # print("\nThe averaged mass matrix in natural co-ordinates is")
        # print(0.5 * massA)  # the half is the Jacobian for span [-1, 1]

        # convert average mass matrix from natural to physical co-ordinates
        length = self.length("ref")
        area = self.area('ref')
        xn1 = (-length / 2,)
        xn2 = (length / 2,)
        mass = np.zeros((2, 2), dtype=float)
        Jdet = sf.jacobianDet(xn1, xn2)
        rho = self.massDensity()
        mass = (rho * area * Jdet) * massA
        return mass

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

    # methods

    def toString(self):
        if self._number < 10:
            s = 'chord[0'
        else:
            s = 'chord['
        s = s + str(self._number)
        s = s + '] has vertices: \n'
        s = s + '   ' + self._vertex[1].toString('next') + '\n'
        s = s + '   ' + self._vertex[2].toString('next')
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
        if atGaussPt == 1:
            return self._solver[1]
        elif atGaussPt == 2:
            return self._solver[2]
        else:
            raise RuntimeError("Argument atGaussPt must be either 1 or 2.")

    def update(self, temperature=37.0, restart=False):
        # determine length of the chord in the next configuration
        x1 = self._vertex[1].coordinates('next')
        x2 = self._vertex[2].coordinates('next')
        self._Ln = math.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2
                             + (x2[2] - x1[2])**2)

        # determine the centroid of this chord
        self._centroidXn = (x1[0] + x2[0]) / 2.0
        self._centroidYn = (x1[1] + x2[1]) / 2.0
        self._centroidZn = (x1[2] + x2[2]) / 2.0

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

        # chordal co-ordinates for the chords, must be tuples
        x10 = (-self._L0 / 2.0,)
        x20 = (self._L0 / 2.0,)
        x1n = (-self._Ln / 2.0,)
        x2n = (self._Ln / 2.0,)

        # quantify the displacement and deformation gradients of the chord
        # displacement gradients located at the Gauss points of a chord
        self._G[1] = self._shapeFns[1].G(x1n, x2n, x10, x20)
        self._G[2] = self._shapeFns[2].G(x1n, x2n, x10, x20)
        # deformation gradients located at the Gauss points of a chord
        self._Fn[1] = self._shapeFns[1].F(x1n, x2n, x10, x20)
        self._Fn[2] = self._shapeFns[2].F(x1n, x2n, x10, x20)

        # integrate the constitutive equations
        xVec = np.zeros((2,), dtype=float)
        xVec[0] = temperature
        xVec[1] = self._Ln
        self._solver[1].integrate(xVec, restart)
        self._solver[2].integrate(xVec, restart)
        return  # nothing, the data structure has been updated

    def advance(self):
        # assign current to previous values, and then next to current values
        self._centroidXp = self._centroidXc
        self._centroidYp = self._centroidYc
        self._centroidZp = self._centroidZc
        self._centroidXc = self._centroidXn
        self._centroidYc = self._centroidYn
        self._centroidZc = self._centroidZn
        self._Pp3D[:, :] = self._Pc3D[:, :]
        self._Pc3D[:, :] = self._Pn3D[:, :]

        # advance the matrix fields associated with each Gauss point
        for i in range(1, self.GAUSS_PTS+1):
            self._Fp[i] = self._Fc[i]
            self._Fc[i] = self._Fn[i]
            # advance the integrators
            self._solver[i].advance()

        # update the stiffness matrix and forcing function for the FE solver
        self._stiffMtx = self._stiffnessMatrix()
        self._forceFn = self._forcingFunction()
        return  # nothing, the data structure has been advanced

    # material properties of the chord

    def massDensity(self):
        v1 = self._response[1].volume()
        v2 = self._response[2].volume()
        m1 = self._response[1].massDensity() * v1
        m2 = self._response[2].massDensity() * v2
        rho = (m1 + m2) / (v1 + v2)
        return rho

    def isRuptured(self):
        if self._response[1].isRuptured() or self._response[2].isRuptured():
            return True
        else:
            return False

    # uniform or averaged geometric properties of the chord

    def area(self):
        a = (self._response[1].area() + self._response[2].area()) / 2.0
        return a

    def length(self):
        return self._Ln

    def stretch(self):
        lambda_ = self._Ln / self._L0
        return lambda_

    def strain(self):
        e = math.log(self._Ln / self._L0)
        return e

    def force(self):
        f1 = self._response[1].force()
        f2 = self._response[2].force()
        if self._response[1].isRuptured():
            if self._response[2].isRuptured():
                f = (f1 + f2) / 2.0
            else:
                f = f1
        elif self._response[2].isRuptured():
            f = f2
        else:
            f = (f1 + f2) / 2.0
        return f

    def temperature(self):
        T = self._response[1].temperature()
        return T

    # kinematic fields associated with the chordal centroid in 3 space

    def centroid(self):
        cx = self._centroidXn
        cy = self._centroidYn
        cz = self._centroidZn
        return np.array([cx, cy, cz])

    def displacement(self, reindex):
        u1 = self._vertex[1].displacement(reindex, 'next')
        u2 = self._vertex[2].displacement(reindex, 'next')
        u = (u1 + u2) / 2.0
        return u

    def velocity(self, reindex):
        v1 = self._vertex[1].velocity(reindex, 'next')
        v2 = self._vertex[2].velocity(reindex, 'next')
        v = (v1 + v2) / 2.0
        return v

    def acceleration(self, reindex):
        a1 = self._vertex[1].acceleration(reindex, 'next')
        a2 = self._vertex[2].acceleration(reindex, 'next')
        a = (a1 + a2) / 2.0
        return a

    # rotation and spin of chord wrt dodecahedral coordinate system

    def rotation(self):
        return np.copy(self._Pn3D)

    def spin(self, reindex):
        return spinMtx.nextSpin(self._Pp3D, self._Pc3D, self._Pn3D, reindex,
                                self._h)

    # fields extrapolated out to the nodal points

    def nodalAreas(self):
        aG1 = self._response[1].area()
        aG2 = self._response[2].area()
        aN1, aN2 = self._gq.extrapolate(aG1, aG2)
        return aN1, aN2

    def nodalStresses(self):
        f = self.force()
        aN1, aN2 = self.nodalAreas()
        sN1 = f / aN1
        sN2 = f / aN2
        return sN1, sN2

    def nodalEntropies(self, state):
        etaG1 = self._response[1].entropy()
        etaG2 = self._response[2].entropy()
        etaN1, etaN2 = self._gq.extrapolate(etaG1, etaG2)
        return etaN1, etaN2

    # fundamental kinematic fields

    # displacement gradient at the specified Gauss point
    def G(self, atGaussPt):
        if atGaussPt == 1 or atGaussPt == 2:
            return np.copy(self._G[atGaussPt])
        else:
            raise RuntimeError("atGaussPt can only be 1 or 2 in a call to "
                               + "Chord.G, and you sent {}.".format(atGaussPt))

    # deformation gradient at the specified Gauss point
    def F(self, atGaussPt):
        if atGaussPt == 1 or atGaussPt == 2:
            return np.copy(self._Fn[atGaussPt])
        else:
            raise RuntimeError("atGaussPt can only be 1 or 2 in a call to "
                               + "Chord.F, and you sent {}.".format(atGaussPt))

    # velocity gradient at the specified Gauss point
    def L(self, atGaussPt):
        if atGaussPt == 1 or atGaussPt == 2:
            # use backward difference scheme
            Finv = np.inverse(self._Fn[atGaussPt])
            velGrad = (((3.0 * self._Fn[atGaussPt] - 4.0 * self._Fc[atGaussPt]
                         + self._Fp[atGaussPt])
                       / (2.0 * self._h)) * Finv)
        else:
            raise RuntimeError("atGaussPt can only be 1 or 2 in a call to "
                               + "Chord.L, and you sent {}.".format(atGaussPt))
        return velGrad

    # fields needed to construct a finite element solution strategy

    def shapeFunction(self, atGaussPt):
        if atGaussPt == 1 or atGaussPt == 2:
            return self._shapeFns[atGaussPt]
        else:
            raise RuntimeError("atGaussPt can only be 1 or 2 in a call to "
                               + "Chord.shapeFunction, "
                               + "and you sent {}.".format(atGaussPt))

    def gaussQuadrature(self):
        return self._gq

    def massMatrix(self):
        mMtx = np.copy(self._massMtx)
        return mMtx

    def stiffnessMatrix(self):
        kMtx = np.copy(self._stiffMtx)
        return kMtx

    def forcingFunction(self):
        fVec = np.copy(self._forceFn)
        return fVec


"""
Changes made in version "1.0.0":


This is the original version.
"""
