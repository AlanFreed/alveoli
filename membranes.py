#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math as m
import numpy as np
import spin as spinMtx

"""
Module membranes.py provides kinematic properties/attributes for a membrane.

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
__date__ = "05-02-2019"
__update__ = "10-05-2019"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

r"""
Class membrane in file membranes.py allows for the creation of objects that are
to be used to describe the kinematics of a membrane; in particular, the septal
planes of an alveolus that are taken to be pentagonal in shape.

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

class membrane

constructor

    m = membrane(h)
        h    uniform timestep size between any two neighboring configurations

    The deformation gradient in an initial configuration is the identity matrix

methods

    m.update(F)
        assigns a new 2x2 deformation gradient associated with the next state,
        which may be reassigned multiple times before advancing the solution

    m.advance()
        assigns current values into previous values, and next values into
        current values, thereby freezing these previous and current states in
        preparation to advance a solution to its next location along its
        solution path

    Reindex the coordinate indices if necessary

    qMtx = m.Q(state)
        returns 2x2 reindexing matrix applied to the deformation gradient
        prior to its Gram-Schmidt decomposition in configuration 'state'

    Gram-Schmidt factorization of a reindexed deformation gradient

    rMtx = m.R(state)
        returns 2x2 rotation matrix 'Q' derived from a QR decomposition of the
        reindexed deformation gradient in configuration 'state'

    omega = m.spin(state)
        returns 2x2 spin matrix caused by planar deformation, i.e., dR R^t,
        in the reindexed coordinate system for configuration 'state'

    uMtx = m.U(state)
        returns 2x2 Laplace stretch 'R' derived from a QR decomposition of the
        reindexed deformation gradient in configuration 'state'

    uInvMtx = m.UInv(state)
        returns 2x2 inverse Laplace stretch derived from a QR decomposition of
        the reindexed deformation gradient in configuration 'state'

    duMtx = m.dU(state)
        returns differential of 2x2 Laplace stretch 'R' derived from a QR
        decomposition of the reindexed deformation gradient in configuration
        'state'

    duInvMtx = m.dUInv(state)
        returns differential of 2x2 inverse Laplace stretch derived from a QR
        decomposition of the reindexed deformation gradient in configuration
        'state'

    The extensive thermodynamic variables for a membrane and their rates
    accuired from a reindexed deformation gradient

    delta = m.dilation(state)
        returns the planar dilation derived from a QR decomposition of the
        reindexed deformation gradient in configuration 'state'

    epsilon = m.squeeze(state)
        returns the planar squeeze derived from a QR decomposition of the
        reindexed deformation gradient in configuration 'state'

    gamma = m.shear(state)
        returns the planar shear derived from a QR decomposition of the
        reindexed deformation gradient in configuration 'state'

    dDelta = m.dDilation(state)
        returns the differential change in dilation in configuration 'state'

    dEpsilon = m.dSqueeze(state)
        returns the differential change in squeeze in configuration 'state'

    dGamma = m.dShear(state)
        returns the differential change in shear in configuration 'state'

References
    1) Freed, A. D., Erel, V., and Moreno, M. R. "Conjugate Stress/Strain Base
       Pairs for the Analysis of Planar Biologic Tissues", Journal of Mechanics
       of Materials and Structures, 12 (2017), 219-247.
    2) Freed, A. D., and Zamani, S.: “On the Use of Convected Coordinate
       Systems in the Mechanics of Continuous Media Derived from a QR
       Decomposition of F”, International Journal of Engineering Science, 127
       (2018), 145-161.
    3) Pual, S., Rajagopal, K. R., and Freed, A. D. "Optimal Representation
       of Laplace Stretch for Transparency with the Physics of Deformation",
       in review.
"""


class membrane(object):

    def __init__(self, h):
        # verify the stepsize
        if h > np.finfo(float).eps:
            self._h = float(h)
        else:
            raise RuntimeError("The stepsize sent to the membrane " +
                               "constructor wasn't positive.")

        # create the required matrices
        self._Qr = np.zeros((2, 2), dtype=float)
        self._Q0 = np.eye(2, dtype=float)
        self._Q1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        self._R0 = np.eye(2, dtype=float)
        self._U0 = np.eye(2, dtype=float)
        self._U0Inv = np.eye(2, dtype=float)
        self._dU0 = np.zeros((2, 2), dtype=float)
        self._dU0Inv = np.zeros((2, 2), dtype=float)

        # establish the reference physical variables for a planar deformation
        self._a0 = 1.0
        self._b0 = 1.0
        self._g0 = 0.0

        # initialize the remaining physical variables for a planar deformation
        self._ap = self._a0
        self._ac = self._a0
        self._an = self._a0
        self._bp = self._b0
        self._bc = self._b0
        self._bn = self._b0
        self._gp = self._g0
        self._gc = self._g0
        self._gn = self._g0
        self._dap = 0.0
        self._dac = 0.0
        self._dan = 0.0
        self._dbp = 0.0
        self._dbc = 0.0
        self._dbn = 0.0
        self._dgp = 0.0
        self._dgc = 0.0
        self._dgn = 0.0
        self._pivotedPrev = False
        self._pivotedCurr = False
        self._pivotedNext = False
        self._Qp = np.copy(self._Qr)
        self._Qc = np.copy(self._Qr)
        self._Qn = np.copy(self._Qr)
        self._Rp = np.copy(self._R0)
        self._Rc = np.copy(self._R0)
        self._Rn = np.copy(self._R0)
        self._Up = np.copy(self._U0)
        self._Uc = np.copy(self._U0)
        self._Un = np.copy(self._U0)
        self._UpInv = np.copy(self._U0Inv)
        self._UcInv = np.copy(self._U0Inv)
        self._UnInv = np.copy(self._U0Inv)
        self._dUp = np.copy(self._dU0)
        self._dUc = np.copy(self._dU0)
        self._dUn = np.copy(self._dU0)
        self._dUpInv = np.copy(self._dU0Inv)
        self._dUcInv = np.copy(self._dU0Inv)
        self._dUnInv = np.copy(self._dU0Inv)

    def update(self, F):
        # verify the initial deformation gradient
        if (not isinstance(F, np.ndarray)) or (not F.shape == (2, 2)):
            raise RuntimeError("F must be 2x2 numpy array in call to " +
                               "membrane.update.")

        # extract kinematic variables out of the deformation gradient
        x = F[0, 0]
        y = F[1, 1]
        alpha = F[1, 0] / x
        beta = F[0, 1] / y

        # if necessary, pivot the deformation gradient before calculating the
        # physical variables for a planar deformation
        gammaTilde = y * (alpha + beta) / (x * (1.0 + alpha**2))
        gammaHat = x * (alpha + beta) / (y * (1.0 + beta**2))
        if abs(gammaHat) < 1.000000001 * abs(gammaTilde):
            # don't pivot
            self._pivotedNext = False
            self._Qn[:, :] = self._Q0[:, :]
            self._an = x * m.sqrt(1.0 + alpha**2)
            self._bn = y * (1.0 - alpha * beta) / m.sqrt(1.0 + alpha**2)
            self._gn = gammaTilde
            theta = m.atan(-alpha)
        else:
            # pivot
            self._pivotedNext = True
            self._Qn[:, :] = self._Q1[:, :]
            self._an = y * m.sqrt(1.0 + beta**2)
            self._bn = x * (1.0 - alpha * beta) / m.sqrt(1.0 + beta**2)
            self._gn = gammaHat
            theta = m.atan(-beta)

        # determine rates for the physical variables (cf. appendix in Ref. 2)
        self._dan = ((3.0 * self._an - 4.0 * self._ac + self._ap) /
                     (2.0 * self._h))
        self._dbn = ((3.0 * self._bn - 4.0 * self._bc + self._bp) /
                     (2.0 * self._h))
        self._dgn = ((2.0 * self._ac / self._an) *
                     (self._gn - self._gc) / self._h -
                     (self._ap / self._an) *
                     (self._gn - self._gp) / (2.0 * self._h))

        # determine the Gram-Schmidt tensor fields
        cosine = m.cos(theta)
        sine = m.sin(theta)
        self._Rn[0, 0] = cosine
        self._Rn[0, 1] = -sine
        self._Rn[1, 0] = sine
        self._Rn[1, 1] = cosine

        self._Un[0, 0] = self._an
        self._Un[0, 1] = self._an * self._gn
        self._Un[1, 0] = 0.0
        self._Un[1, 1] = self._bn

        self._UnInv[0, 0] = 1.0 / self._an
        self._UnInv[0, 1] = -self._gn / self._bn
        self._UnInv[1, 0] = 0.0
        self._UnInv[1, 1] = 1.0 / self._bn

        self._dUn[0, 0] = self._dan
        self._dUn[0, 1] = self._an * self._dgn + self._gn * self._dan
        self._dUn[1, 0] = 0.0
        self._dUn[1, 1] = self._dbn

        self._dUnInv[0, 0] = -self._dan / self._an**2
        self._dUnInv[0, 1] = -(self._dgn / self._bn -
                               self._gn * self._dbn / self._bn**2)
        self._dUnInv[1, 0] = 0.0
        self._dUnInv[1, 1] = -self._dbn / self._bn**2

    def advance(self):
        # assign fields from the current state into their previous state
        self._pivotedPrev = self._pivotedCurr
        self._ap = self._ac
        self._bp = self._bc
        self._gp = self._gc
        self._dap = self._dac
        self._dbp = self._dbc
        self._dgp = self._dgc
        self._Qp[:, :] = self._Qc[:, :]
        self._Rp[:, :] = self._Rc[:, :]
        self._Up[:, :] = self._Uc[:, :]
        self._UpInv[:, :] = self._UcInv[:, :]
        self._dUp[:, :] = self._dUc[:, :]
        self._dUpInv[:, :] = self._dUcInv[:, :]

        # assign fields from the next state into their current state
        self._pivotedCurr = self._pivotedNext
        self._ac = self._an
        self._bc = self._bn
        self._gc = self._gn
        self._dac = self._dan
        self._dbc = self._dbn
        self._dgc = self._dgn
        self._Qc[:, :] = self._Qn[:, :]
        self._Rc[:, :] = self._Rn[:, :]
        self._Uc[:, :] = self._Un[:, :]
        self._UcInv[:, :] = self._UnInv[:, :]
        self._dUc[:, :] = self._dUn[:, :]
        self._dUcInv[:, :] = self._dUnInv[:, :]

    # Gram-Schmidt factorization of the deformation gradient

    def Q(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Qc)
            elif state == 'n' or state == 'next':
                return np.copy(self._Qn)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Qp)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._Qr)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.Q.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.Q.")

    def R(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Rc)
            elif state == 'n' or state == 'next':
                return np.copy(self._Rn)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Rp)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._R0)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.R.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.R.")

    def spin(self, state):
        if isinstance(state, str):
            if ((self._pivotedNext == self._pivotedCurr) and
               (self._pivotedCurr == self._pivotedPrev)):
                Rp = np.zeros((3, 3), dtype=float)
                Rc = np.zeros((3, 3), dtype=float)
                Rn = np.zeros((3, 3), dtype=float)
                for i in range(2):
                    for j in range(2):
                        Rp[i, j] = self._Rp[i, j]
                        Rc[i, j] = self._Rc[i, j]
                        Rn[i, j] = self._Rc[i, j]
                Rp[2, 2] = 1.0
                Rc[2, 2] = 1.0
                Rn[2, 2] = 1.0
                if state == 'c' or state == 'curr' or state == 'current':
                    omega3D = spinMtx.currSpin(Rp, Rc, Rn, self._h)
                elif state == 'n' or state == 'next':
                    omega3D = spinMtx.nextSpin(Rp, Rc, Rn, self._h)
                elif state == 'p' or state == 'prev' or state == 'previous':
                    omega3D = spinMtx.prevSpin(Rp, Rc, Rn, self._h)
                elif state == 'r' or state == 'ref' or state == 'reference':
                    omega3D = np.zeros((3, 3), dtype=float)
                else:
                    raise RuntimeError("An unknown state {} ".format(state) +
                                       "in a call to membrane.spin.")
            else:
                # rotations are likely to be disontinuous because of a
                # coordinate re-indexing that took place
                omega3D = np.zeros((3, 3), dtype=float)
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.spin.")
        omega2D = np.zeros((2, 2), dtype=float)
        omega2D[0, 1] = omega3D[0, 1]
        omega2D[1, 0] = omega3D[1, 0]
        return omega2D

    def U(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._Uc)
            elif state == 'n' or state == 'next':
                return np.copy(self._Un)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._Up)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._U0)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.U.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.U.")

    def UInv(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._UcInv)
            elif state == 'n' or state == 'next':
                return np.copy(self._UnInv)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._UpInv)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._U0Inv)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.UInv.")
        else:
            raise RuntimeError("An unknown state {}.format(str(state)) " +
                               "in a call to membrane.UInv.")

    def dU(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._dUc)
            elif state == 'n' or state == 'next':
                return np.copy(self._dUn)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._dUp)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._dU0)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.dU.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.dU.")

    def dUInv(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return np.copy(self._dUcInv)
            elif state == 'n' or state == 'next':
                return np.copy(self._dUnInv)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return np.copy(self._dUpInv)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return np.copy(self._dU0Inv)
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.dUInv.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.dUInv.")

    # The extensive thermodynamic variables and their rates

    def dilation(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return m.log(m.sqrt((self._ac * self._bc) /
                                    (self._a0 * self._b0)))
            elif state == 'n' or state == 'next':
                return m.log(m.sqrt((self._an * self._bn) /
                                    (self._a0 * self._b0)))
            elif state == 'p' or state == 'prev' or state == 'previous':
                return m.log(m.sqrt((self._ap * self._bp) /
                                    (self._a0 * self._b0)))
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.dilation.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.dilation.")

    def squeeze(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                if not self._pivotedCurr:
                    return m.log(m.sqrt((self._ac * self._b0) /
                                        (self._a0 * self._bc)))
                else:
                    return m.log(m.sqrt((self._a0 * self._bc) /
                                        (self._ac * self._b0)))
            elif state == 'n' or state == 'next':
                if not self._pivotedNext:
                    return m.log(m.sqrt((self._an * self._b0) /
                                        (self._a0 * self._bn)))
                else:
                    return m.log(m.sqrt((self._a0 * self._bn) /
                                        (self._an * self._b0)))
            elif state == 'p' or state == 'prev' or state == 'previous':
                if not self._pivotedPrev:
                    return m.log(m.sqrt((self._ap * self._b0) /
                                        (self._a0 * self._bp)))
                else:
                    return m.log(m.sqrt((self._a0 * self._bp) /
                                        (self._ap * self._b0)))
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.squeeze.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.squeeze.")

    def shear(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._gc - self._g0
            elif state == 'n' or state == 'next':
                return self._gn - self._g0
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._gp - self._g0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.shear.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.shear.")

    def dDilation(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return (self._dac / self._ac + self._dbc / self._bc) / 2.0
            elif state == 'n' or state == 'next':
                return (self._dan / self._an + self._dbn / self._bn) / 2.0
            elif state == 'p' or state == 'prev' or state == 'previous':
                return (self._dap / self._ap + self._dbp / self._bp) / 2.0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.dDilation.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.dDilation.")

    def dSqueeze(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                if not self._pivotedCurr:
                    return (self._dac / self._ac - self._dbc / self._bc) / 2.0
                else:
                    return (self._dbc / self._bc - self._dac / self._ac) / 2.0
            elif state == 'n' or state == 'next':
                if not self._pivotedNext:
                    return (self._dan / self._an - self._dbn / self._bn) / 2.0
                else:
                    return (self._dbn / self._bn - self._dan / self._an) / 2.0
            elif state == 'p' or state == 'prev' or state == 'previous':
                if not self._pivotedPrev:
                    return (self._dap / self._ap - self._dbp / self._bp) / 2.0
                else:
                    return (self._dbp / self._bp - self._dap / self._ap) / 2.0
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.dSqueeze.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.dSqueeze.")

    def dShear(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._dgc
            elif state == 'n' or state == 'next':
                return self._dgn
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._dgp
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("An unknown state {} ".format(state) +
                                   "in a call to membrane.dShear.")
        else:
            raise RuntimeError("An unknown state {} ".format(str(state)) +
                               "in a call to membrane.dShear.")
