#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import sqrt
import numpy as np

"""
Module pivotIncomingF.py provides a deformation gradient that remains
physically compatible with the Laplace stretch.

Copyright (c) 2020 Alan D. Freed

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
Copyright (c) 2020 Alan D. Freed
"""

# Module metadata
__version__ = "1.0.0"
__date__ = "04-17-2020"
__update__ = "07-06-2020"
__author__ = "Alan D. Freed and Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


Class Pivot in file pivotIncomingF.py pivots an incoming deformation gradient
so that in its re-indexed frame of reference the physics of Laplace stretch are
adhered to, viz., the invariant properties that the 1st co-ordinate direction
and the 12 co-ordinate plane remain invariant under transformations of the
Laplace stretch are not violated.

Localization assumes that the deformation gradient at a continuum mass point,
in our case specifying a mass point in the parenchyma of lung, which need not
be homogeneous at the macroscopic scale of the lung, is taken to be homogeneous
over the microscopic scale of an alveolus.  This module provides the linear
transformations that map fields between these macro and micro domains.

Numerous methods have a string argument that is denoted as  state  which can
take on any of the following values:
    'c', 'curr', 'current'       gets the value for a current configuration
    'n', 'next'                  gets the value for a next configuration
    'p', 'prev', 'previous'      gets the value for a previous configuration
    'r', 'ref', 'reference'      gets the value for the reference configuration


variables

    P1  maps global co-ordinate frame (i, j, k) into local frame (e1, e2, e3)
    P2  maps global co-ordinate frame (i, k, j) into local frame (e1, e2, e3)
    P3  maps global co-ordinate frame (j, i, k) into local frame (e1, e2, e3)
    P4  maps global co-ordinate frame (j, k, i) into local frame (e1, e2, e3)
    P5  maps global co-ordinate frame (k, i, j) into local frame (e1, e2, e3)
    P6  maps global co-ordinate frame (k, j, i) into local frame (e1, e2, e3)

class Pivot

constructor

    p = Pivot(F0)
        F0  is a deformation gradient in its global co-ordinate frame that
        pertains to a reference state or configuration

methods

    p.update(F)
        F   is a deformation gradient in its global co-ordinate frame that
            affiliates with the end of the next step of integration
        This may be reassigned many times before advancing a solution along
        its path of motion

    p.advance()
        assigns the current fields to their respective previous fields, and
        then assigns the next fields to their respective current fields, all
        in preparation for advancing a solution to the next location along its
        path of motion

    c = p.pivotedCase(state)
        c   is an integer within the interval [1, 6] that denotes which of the
            six possible co-ordinate re-labelings has been applied to the
            configuration 'state'

    F = p.pivotedF(state)
        F   is the pivoted or re-indexed deformation gradient to be used for
            analysis as it applies to the configuration 'state'

    P = p.pivotMatrix(state)
        P   is the orthogonal matrix used for re-indexing between the user's
            co-ordinate frame and the analysis frame of configuration 'state'
            it is one of p.P1, p.P2, p.P3, p.P4, p.P5 or p.P6

    In the following set of four methods, local fields are quantified in the
    co-ordinate frame of the dodecahedron with basis (E1, E2, E3), while
    global fields are quantified in the co-ordinate frame of the user with
    basis (i, j, k).  Typically, the local frame is at a microscopic level,
    while the global frame is at a macroscopic level.  In our application,
    (i, j, k) is the co-ordinate basis of an FEA of a lung, e.g., while
    (E1, E2, E3) is the co-ordinate basis of our dodecahedron.

    lVec = p.globalToLocalVector(gVec, state)
        gVec    a vector whose components are in the global (i, j, k) frame
        lVec    a vector whose components are in a local (E1, E2, E3) frame
        maps a global vector into a local vector at configuration 'state'

    gVec = p.localToGlobalVector(lVec, state)
        gVec    a vector whose components are in the global (i, j, k) frame
        lVec    a vector whose components are in a local (E1, E2, E3) frame
        maps a local vector into a global vector at configuration 'state'

    lTen = p.globalToLocalTensor(gTen, state)
        gTen    a tensor whose components are in the global (i, j, k) frame
        lTen    a tensor whose components are in a local (E1, E2, E3) frame
        maps a global tensor into a local tensor at configuration 'state'

    gTen = p.localToGlobalTensor(lTen, state)
        gTen    a tensor whose components are in the global (i, j, k) frame
        lTen    a tensor whose components are in a local (E1, E2, E3) frame
        maps a local tensor into a global tensor at configuration 'state'

    These two methods map a pivoted field from one re-indexed co-ordinate
    system to another re-indexed co-ordinate system, viz., they are maps
    between re-indexed frames for basis (E1, E2, E3).

    toVec = p.reindexVector(fromVec, fromCase, toCase)
        fromVec     is the vector before it has been mapped
        toVec       is the vector after  it has been mapped
        fromCase    is the p.pivotedCase out of which it is to be rotated
        toCase      is the p.pivotedCase in to  which it is to be rotated

    toTen = p.reindexTensor(fromTen, fromCase, toCase)
        fromTen     is the tensor before it has been mapped
        toTen       is the tensor after  it has been mapped
        fromCase    is the p.pivotedCase out of which it is to be rotated
        toCase      is the p.pivotedCase in to  which it is to be rotated

Reference:
    Paul, S., Rajagopal, K. R., and Freed, A. D., "On coordinate indexing when
    using Laplace stretch", in review.
"""


class Pivot(object):

    def __init__(self, F0):
        # assess admissibility of input
        (rows, cols) = np.shape(F0)
        if (rows != 3) or (cols != 3):
            raise RuntimeError("Matrix dimension of F0 was not 3x3 in call " +
                               "to the pivot constructor.")

        # initialize the object's internal fields
        self._caseR = 1
        self._caseP = 1
        self._caseC = 1
        self._caseN = 1
        self._Fr = np.eye(3, dtype=float)
        self._Fp = np.eye(3, dtype=float)
        self._Fc = np.eye(3, dtype=float)
        self._Fn = np.eye(3, dtype=float)
        self._Pr = np.eye(3, dtype=float)
        self._Pp = np.eye(3, dtype=float)
        self._Pc = np.eye(3, dtype=float)
        self._Pn = np.eye(3, dtype=float)

        # export the six admissible orthogonal mappings
        # (i, j, k) -> (e1, e2, e3)
        self.P1 = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]], dtype=float)
        # (i, j, k) -> (e1, e3, e2)
        self.P2 = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]], dtype=float)
        # (i, j, k) -> (e2, e1, e3)
        self.P3 = np.array([[0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0]], dtype=float)
        # (i, j, k) -> (e2, e3, e1)
        self.P4 = np.array([[0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]], dtype=float)
        # (i, j, k) -> (e3, e1, e2)
        self.P5 = np.array([[0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0]], dtype=float)
        # (i, j, k) -> (e3, e2, e1)
        self.P6 = np.array([[0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0]], dtype=float)

        # establish which pivoting case applies
        g1 = sqrt(F0[1, 0]**2 + F0[2, 0]**2) / F0[0, 0]
        g2 = sqrt(F0[0, 1]**2 + F0[2, 1]**2) / F0[1, 1]
        g3 = sqrt(F0[0, 2]**2 + F0[1, 2]**2) / F0[2, 2]

        f1 = np.array([F0[0, 0], F0[1, 0], F0[2, 0]])
        f2 = np.array([F0[0, 1], F0[1, 1], F0[2, 1]])
        f3 = np.array([F0[0, 2], F0[1, 2], F0[2, 2]])

        if (g1 <= g2) and (g1 <= g3):
            if np.dot(f1, f2) <= np.dot(f1, f3):
                self._caseC = 1
                self._Pc[:, :] = self.P1[:, :]
            else:
                self._caseC = 2
                self._Pc[:, :] = self.P2[:, :]
        elif (g2 <= g1) and (g2 <= g3):
            if np.dot(f2, f1) <= np.dot(f2, f3):
                self._caseC = 3
                self._Pc[:, :] = self.P3[:, :]
            else:
                self._caseC = 4
                self._Pc[:, :] = self.P4[:, :]
        else:
            if np.dot(f3, f1) <= np.dot(f3, f2):
                self._caseC = 5
                self._Pc[:, :] = self.P5[:, :]
            else:
                self._caseC = 6
                self._Pc[:, :] = self.P6[:, :]

        # pivot the deformation gradient as required
        self._Fc = np.matmul(np.transpose(self._Pc), np.matmul(F0, self._Pc))

        # make all cases the same when the current state is the reference state
        self._caseR = self._caseC
        self._caseP = self._caseC
        self._caseN = self._caseC
        self._Pr[:, :] = self._Pc[:, :]
        self._Pp[:, :] = self._Pc[:, :]
        self._Pn[:, :] = self._Pc[:, :]
        self._Fr[:, :] = self._Fc[:, :]
        self._Fp[:, :] = self._Fc[:, :]
        self._Fn[:, :] = self._Fc[:, :]

        return  # a new object

    def update(self, F):
        # assess admissibility of input
        (rows, cols) = np.shape(F)
        if (rows != 3) or (cols != 3):
            raise RuntimeError("Dimension of F was not 3x3 in call to " +
                               "pivot.update(F).")

        # establish which pivoting case applies
        g1 = sqrt(F[1, 0]**2 + F[2, 0]**2) / F[0, 0]
        g2 = sqrt(F[0, 1]**2 + F[2, 1]**2) / F[1, 1]
        g3 = sqrt(F[0, 2]**2 + F[1, 2]**2) / F[2, 2]

        f1 = np.array([F[0, 0], F[1, 0], F[2, 0]])
        f2 = np.array([F[0, 1], F[1, 1], F[2, 1]])
        f3 = np.array([F[0, 2], F[1, 2], F[2, 2]])

        if (g1 <= g2) and (g1 <= g3):
            if np.dot(f1, f2) <= np.dot(f1, f3):
                self._caseN = 1
                self._Pn[:, :] = self.P1[:, :]
            else:
                self._caseN = 2
                self._Pn[:, :] = self.P2[:, :]
        elif (g2 <= g1) and (g2 <= g3):
            if np.dot(f2, f1) <= np.dot(f2, f3):
                self._caseN = 3
                self._Pn[:, :] = self.P3[:, :]
            else:
                self._caseN = 4
                self._Pn[:, :] = self.P4[:, :]
        else:
            if np.dot(f3, f1) <= np.dot(f3, f2):
                self._caseN = 5
                self._Pn[:, :] = self.P5[:, :]
            else:
                self._caseN = 6
                self._Pn[:, :] = self.P6[:, :]

        # pivot the deformation gradient as required
        self._Fn = np.matmul(np.transpose(self._Pn), np.matmul(F, self._Pn))

        return  # nothing

    def advance(self):
        self._caseP = self._caseC
        self._caseC = self._caseN
        self._Pp[:, :] = self._Pc[:, :]
        self._Pc[:, :] = self._Pn[:, :]
        self._Fp[:, :] = self._Fc[:, :]
        self._Fc[:, :] = self._Fn[:, :]

        return  # nothing

    def pivotCase(self, state):
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self._caseC
            elif state == 'n' or state == 'next':
                return self._caseN
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self._caseP
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self._caseR
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to pivot.pivotCase.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in call a to pivot.pivotCase.")

    def pivotedF(self, state):
        F = np.zeros((3, 3), dtype=float)
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                F[:, :] = self._Fc[:, :]
            elif state == 'n' or state == 'next':
                F[:, :] = self._Fn[:, :]
            elif state == 'p' or state == 'prev' or state == 'previous':
                F[:, :] = self._Fp[:, :]
            elif state == 'r' or state == 'ref' or state == 'reference':
                F[:, :] = self._Fr[:, :]
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to pivot.pivotedF.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in call a to pivot.pivotedF.")
        return F

    def pivotMatrix(self, state):
        P = np.zeros((3, 3), dtype=float)
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                P[:, :] = self._Pc[:, :]
            elif state == 'n' or state == 'next':
                P[:, :] = self._Pn[:, :]
            elif state == 'p' or state == 'prev' or state == 'previous':
                P[:, :] = self._Pp[:, :]
            elif state == 'r' or state == 'ref' or state == 'reference':
                P[:, :] = self._Pr[:, :]
            else:
                raise RuntimeError("An unknown state of {} ".format(state) +
                                   "was sent in a call to pivot.pivotMtx.")
        else:
            raise RuntimeError("An unknown state of {} ".format(str(state)) +
                               "was sent in call a to pivot.pivotMtx.")
        return P

    def globalToLocalVector(self, gVec, state):
        lVec = np.zeros(3, dtype=float)
        P = self.pivotMatrix(state)
        lVec = np.matmul(np.transpose(P), gVec)
        return lVec

    def localToGlobalVector(self, lVec, state):
        gVec = np.zeros(3, dtype=float)
        P = self.pivotMatrix(state)
        gVec = np.matmul(P, lVec)
        return gVec

    def globalToLocalTensor(self, gTen, state):
        lTen = np.zeros((3, 3), dtype=float)
        P = self.pivotMatrix(state)
        lTen = np.matmul(np.transpose(P), np.matmul(gTen, P))
        return lTen

    def localToGlobalTensor(self, lTen, state):
        gTen = np.zeros((3, 3), dtype=float)
        P = self.pivotMatrix(state)
        gTen = np.matmul(P, np.matmul(lTen, np.transpose(P)))
        return gTen

    def reindexVector(self, fromVec, fromCase, toCase):
        toVec = np.zeros(3, dtype=float)
        if fromCase == toCase:
            toVec[:] = fromVec[:]
        else:
            if fromCase == 1:
                fromP = self.P1
            elif fromCase == 2:
                fromP = self.P2
            elif fromCase == 3:
                fromP = self.P3
            elif fromCase == 4:
                fromP = self.P4
            elif fromCase == 5:
                fromP = self.P5
            elif fromCase == 6:
                fromP = self.P6
            else:
                raise RuntimeError('fromCase in reindexTensor must belong ' +
                                   'to [1, 6], you sent {}.'.format(fromCase))
            if toCase == 1:
                toP = self.P1
            elif toCase == 2:
                toP = self.P2
            elif toCase == 3:
                toP = self.P3
            elif toCase == 4:
                toP = self.P4
            elif toCase == 5:
                toP = self.P5
            elif toCase == 6:
                toP = self.P6
            else:
                raise RuntimeError('toCase in reindexTensor must belong ' +
                                   'to [1, 6], you sent {}.'.format(toCase))
            pMtx = np.matmul(np.transpose(toP), fromP)
            toVec = np.matmul(pMtx, fromVec)
        return toVec

    def reindexTensor(self, fromTen, fromCase, toCase):
        toTen = np.zeros((3, 3), dtype=float)
        if fromCase == toCase:
            toTen[:, :] = fromTen[:, :]
        else:
            if fromCase == 1:
                fromP = self.P1
            elif fromCase == 2:
                fromP = self.P2
            elif fromCase == 3:
                fromP = self.P3
            elif fromCase == 4:
                fromP = self.P4
            elif fromCase == 5:
                fromP = self.P5
            elif fromCase == 6:
                fromP = self.P6
            else:
                raise RuntimeError('fromCase in reindexTensor must belong ' +
                                   'to [1, 6], you sent {}.'.format(fromCase))
            if toCase == 1:
                toP = self.P1
            elif toCase == 2:
                toP = self.P2
            elif toCase == 3:
                toP = self.P3
            elif toCase == 4:
                toP = self.P4
            elif toCase == 5:
                toP = self.P5
            elif toCase == 6:
                toP = self.P6
            else:
                raise RuntimeError('toCase in reindexTensor must belong ' +
                                   'to [1, 6], you sent {}.'.format(toCase))
            pMtx = np.matmul(np.transpose(toP), fromP)
            toTen = np.matmul(pMtx, np.matmul(fromTen, np.transpose(pMtx)))
        return toTen


"""
Version 1.0.0 is the original version.
"""
