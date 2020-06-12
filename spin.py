#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import acos, cos, sin, sqrt
import numpy as np
from pivotIncomingF import pivot

"""
Module spin.py provides the co-ordinate spin tensor.

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
__date__ = "04-30-2019"
__update__ = "05-20-2020"
__author__ = "Alan D. Freed"
__author_email__ = "adfreed@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


This module supplies four functions: one that returns the quaternions of
Rodrigues given an orthogonal rotation matrix, and the others that return the
the skew-symmetric spin matrices at three successive moments in time.

All matrices in the following procedures are to be 3x3 instances of type NumPy
darray, while the stepsize must be greater than machine epsilon.

procedures

    n0, n1, n2, n3 = quaternion(rotationMtx)
        where
            rotationMtx is an orthogonal 3x3 NumPy matrix
        This procedure returns the quaternion of Rodrigous according to the
        algorithm of Spurrier, which assures the angles or rotation are not
        close to zero, thereby avoiding potential problems when computing spins

    The following three procedures compute spin from three rotation matrices
    evaluated at consecutive time steps along a solution path separated in time
    by a stepsize dTime.  They return the skew symmetric spin for these three
    three instances in time.  All derivative estimates are second-order
    accurate.  The two neighboring states have their co-ordinates re-indexed
    to that of the requested state, as required, so their is continuity in
    their values.  Each spin procedure calls the quaternion procedure.

    spin = prevSpin(prevRotationMtx, currRotationMtx, nextRotationMtx,
                    reindex, dTime)
        inputs
            prevRotationMtx  is an orthogonal matrix at the previous time step
            currRotationMtx  is an orthogonal matrix at the current time step
            nextRotationMtx  is an orthogonal matrix at the next time step
            reindex          is an instance of pivot object from pivotIncomingF
            dTime            is the size of the time step
        output
            spin             is the skew symmetric spin at the previous time

    spin = currSpin(prevRotationMtx, currRotationMtx, nextRotationMtx,
                    reindex, dTime)
        inputs
            prevRotationMtx  is an orthogonal matrix at the previous time step
            currRotationMtx  is an orthogonal matrix at the current time step
            nextRotationMtx  is an orthogonal matrix at the next time step
            reindex          is an instance of pivot object from pivotIncomingF
            dTime            is the size of the time step
        output
            spin             is the skew symmetric spin at the current time

    spin = prevSpin(prevRotationMtx, currRotationMtx, nextRotationMtx,
                    reindex, dTime)
        inputs
            prevRotationMtx  is an orthogonal matrix at the previous time step
            currRotationMtx  is an orthogonal matrix at the current time step
            nextRotationMtx  is an orthogonal matrix at the next time step
            reindex          is an instance of pivot object from pivotIncomingF
            dTime            is the size of the time step
        output
            spin             is the skew symmetric spin at the next time

References:
    1) R. A. Spurrier, "Comment on 'Singularity-free extraction of a quaternion
    from a direction-cosine matrix'", Journal of Spacecraft and Rockets, 15,
    255 (1978).
    2) A. D. Freed, J.-B. le Graverend and K. R. Rajagopal, "A Decomposition of
    Laplace Stretch with Applications in Inelasticity", Acta Mechanica, 230,
    3423–3429 (2019).
"""


def quaternion(rotationMtx):
    # verify input
    if not isinstance(rotationMtx, np.ndarray):
        raise RuntimeError("rotationMtx sent to quaternion isn't numpy.")
    (rows, cols) = np.shape(rotationMtx)
    if (rows != 3) or (cols != 3):
        raise RuntimeError('Function quaternion requires 3x3 rotation matrix.')
    trQ = np.trace(rotationMtx)

    # Uses the algorithm of Spurrier to compute Rodrigues' quaternion
    if (trQ >= rotationMtx[0, 0] and trQ >= rotationMtx[1, 1] and
            trQ >= rotationMtx[2, 2]):
        n0 = sqrt(1.0 + trQ) / 2.0
        n1 = (rotationMtx[2, 1] - rotationMtx[1, 2]) / (4.0 * n0)
        n2 = (rotationMtx[0, 2] - rotationMtx[2, 0]) / (4.0 * n0)
        n3 = (rotationMtx[1, 0] - rotationMtx[0, 1]) / (4.0 * n0)
    elif (rotationMtx[0, 0] >= rotationMtx[1, 1] and
            rotationMtx[0, 0] >= rotationMtx[2, 2]):
        n1 = sqrt(2.0 * rotationMtx[0, 0] + (1.0 - trQ)) / 2.0
        n0 = (rotationMtx[2, 1] - rotationMtx[1, 2]) / (4.0 * n1)
        n2 = (rotationMtx[1, 0] + rotationMtx[0, 1]) / (4.0 * n1)
        n3 = (rotationMtx[2, 0] + rotationMtx[0, 2]) / (4.0 * n1)
    elif (rotationMtx[1, 1] >= rotationMtx[0, 0] and
            rotationMtx[1, 1] >= rotationMtx[2, 2]):
        n2 = sqrt(2.0 * rotationMtx[1, 1] + (1.0 - trQ)) / 2.0
        n0 = (rotationMtx[0, 2] - rotationMtx[2, 0]) / (4.0 * n2)
        n1 = (rotationMtx[0, 1] + rotationMtx[1, 0]) / (4.0 * n2)
        n3 = (rotationMtx[2, 1] + rotationMtx[1, 2]) / (4.0 * n2)
    else:
        n3 = sqrt(2.0 * rotationMtx[2, 2] + (1.0 - trQ)) / 2.0
        n0 = (rotationMtx[1, 0] - rotationMtx[0, 1]) / (4.0 * n3)
        n1 = (rotationMtx[0, 2] + rotationMtx[2, 0]) / (4.0 * n3)
        n2 = (rotationMtx[1, 2] + rotationMtx[2, 1]) / (4.0 * n3)
    return n0, n1, n2, n3


def prevSpin(prevRotationMtx, currRotationMtx, nextRotationMtx,
             reindex, dTime):
    # verify the input
    if not isinstance(reindex, pivot):
        raise RuntimeError("The 'reindex' variable sent to prevSpin " +
                           "must be of type pivot.")
    if dTime < 100 * np.finfo(float).eps:
        raise RuntimeError("The 'dTime' sent to prevSpin must be greater " +
                           "than 100 times machine epsilon.")

    # get quaternions for the three rotations in the previous co-ordinate frame
    n0p, n1p, n2p, n3p = quaternion(prevRotationMtx)
    # handle potential changes in the re-indexing of our co-ordinate frame
    toCase = reindex.pivotCase('prev')
    # map from current co-ordinate frame into previous co-ordinate frame
    fromCase = reindex.pivotCase('curr')
    currR = reindex.reindexTensor(currRotationMtx, fromCase, toCase)
    n0c, n1c, n2c, n3c = quaternion(currR)
    # map from next co-ordinate frame into previous co-ordinate frame
    fromCase = reindex.pivotCase('next')
    nextR = reindex.reindexTensor(nextRotationMtx, fromCase, toCase)
    n0n, n1n, n2n, n3n = quaternion(nextR)

    # determine the angles and axes of rotation
    if n0p < 1.0:
        prevRotAngle = 2.0 * acos(n0p)
    else:
        prevRotAngle = np.pi / 2.0
    if n0c < 1.0:
        currRotAngle = 2.0 * acos(n0c)
    else:
        currRotAngle = np.pi / 2.0
    if n0n < 1.0:
        nextRotAngle = 2.0 * acos(n0n)
    else:
        nextRotAngle = np.pi / 2.0

    prevRotAngle = 2.0 * acos(n0p)
    currRotAngle = 2.0 * acos(n0c)
    nextRotAngle = 2.0 * acos(n0n)
    sine = sin(prevRotAngle / 2.0)
    prevRotVec = np.array([n1p / sine, n2p / sine, n3p / sine])
    sine = sin(currRotAngle / 2.0)
    currRotVec = np.array([n1c / sine, n2c / sine, n3c / sine])
    sine = sin(nextRotAngle / 2.0)
    nextRotVec = np.array([n1n / sine, n2n / sine, n3n / sine])

    # derivatives for the angle and axis of rotation for the previous state
    dRotAngle = ((-nextRotAngle + 4.0 * currRotAngle - 3.0 * prevRotAngle) /
                 (2.0 * dTime))
    dRotVec = np.array([0.0, 0.0, 0.0])
    dRotVec = ((-nextRotVec + 4.0 * currRotVec - 3.0 * prevRotVec) /
               (2.0 * dTime))

    # Euler's formula for computing the axis of spin
    spinVec = np.array([0.0, 0.0, 0.0])
    spinVec = (dRotAngle * prevRotVec + sin(prevRotAngle) * dRotVec +
               (1.0 - cos(prevRotAngle)) * np.cross(prevRotVec, dRotVec))

    # create the skew symmetric spin matrix and return it
    spinMtx = np.zeros((3, 3), dtype=float)
    spinMtx[0, 1] = -spinVec[2]
    spinMtx[0, 2] = spinVec[1]
    spinMtx[1, 0] = spinVec[2]
    spinMtx[1, 2] = -spinVec[0]
    spinMtx[2, 0] = -spinVec[1]
    spinMtx[2, 1] = spinVec[0]
    return spinMtx


def currSpin(prevRotationMtx, currRotationMtx, nextRotationMtx,
             reindex, dTime):
    # verify the input
    if not isinstance(reindex, pivot):
        raise RuntimeError("The 'reindex' variable sent to currSpin " +
                           "must be of type pivot.")
    if dTime < 100 * np.finfo(float).eps:
        raise RuntimeError("The 'dTime' sent to currSpin must be greater " +
                           "than 100 times machine epsilon.")

    # get quaternions for the three rotations in the current co-ordinate frame
    n0c, n1c, n2c, n3c = quaternion(currRotationMtx)
    # handle potential changes in the re-indexing of our co-ordinate frame
    toCase = reindex.pivotCase('curr')
    # map from previous co-ordinate frame into current co-ordinate frame
    fromCase = reindex.pivotCase('prev')
    prevR = reindex.reindexTensor(prevRotationMtx, fromCase, toCase)
    n0p, n1p, n2p, n3p = quaternion(prevR)
    # map from next co-ordinate frame into current co-ordinate frame
    fromCase = reindex.pivotCase('next')
    nextR = reindex.reindexTensor(nextRotationMtx, fromCase, toCase)
    n0n, n1n, n2n, n3n = quaternion(nextR)

    # determine the angles and axes of rotation
    if n0p < 1.0:
        prevRotAngle = 2.0 * acos(n0p)
    else:
        prevRotAngle = np.pi / 2.0
    if n0c < 1.0:
        currRotAngle = 2.0 * acos(n0c)
    else:
        currRotAngle = np.pi / 2.0
    if n0n < 1.0:
        nextRotAngle = 2.0 * acos(n0n)
    else:
        nextRotAngle = np.pi / 2.0
    sine = sin(prevRotAngle / 2.0)
    prevRotVec = np.array([n1p / sine, n2p / sine, n3p / sine])
    sine = sin(currRotAngle / 2.0)
    currRotVec = np.array([n1c / sine, n2c / sine, n3c / sine])
    sine = sin(nextRotAngle / 2.0)
    nextRotVec = np.array([n1n / sine, n2n / sine, n3n / sine])

    # derivatives for the angle and axis of rotation for the current state
    dRotAngle = (nextRotAngle - prevRotAngle) / (2.0 * dTime)
    dRotVec = np.array([0.0, 0.0, 0.0])
    dRotVec = (nextRotVec - prevRotVec) / (2.0 * dTime)

    # Euler's formula for computing the axis of spin
    spinVec = np.array([0.0, 0.0, 0.0])
    spinVec = (dRotAngle * currRotVec + sin(currRotAngle) * dRotVec +
               (1.0 - cos(currRotAngle)) * np.cross(currRotVec, dRotVec))

    # create the skew symmetric spin matrix and return it
    spinMtx = np.zeros((3, 3), dtype=float)
    spinMtx[0, 1] = -spinVec[2]
    spinMtx[0, 2] = spinVec[1]
    spinMtx[1, 0] = spinVec[2]
    spinMtx[1, 2] = -spinVec[0]
    spinMtx[2, 0] = -spinVec[1]
    spinMtx[2, 1] = spinVec[0]
    return spinMtx


def nextSpin(prevRotationMtx, currRotationMtx, nextRotationMtx,
             reindex, dTime):
    # verify the input
    if not isinstance(reindex, pivot):
        raise RuntimeError("The 'reindex' variable sent to nextSpin " +
                           "must be of type pivot.")
    if dTime < 100 * np.finfo(float).eps:
        raise RuntimeError("The 'dTime' sent to nextSpin must be greater " +
                           "than 100 times machine epsilon.")

    # get quaternions for the three rotations in the next co-ordinate frame
    n0n, n1n, n2n, n3n = quaternion(nextRotationMtx)
    # handle potential changes in the re-indexing of our co-ordinate frame
    toCase = reindex.pivotCase('next')
    # map from previous co-ordinate frame into next co-ordinate frame
    fromCase = reindex.pivotCase('prev')
    prevR = reindex.reindexTensor(prevRotationMtx, fromCase, toCase)
    n0p, n1p, n2p, n3p = quaternion(prevR)
    # map from current co-ordinate frame into next co-ordinate frame
    fromCase = reindex.pivotCase('curr')
    currR = reindex.reindexTensor(currRotationMtx, fromCase, toCase)
    n0c, n1c, n2c, n3c = quaternion(currR)

    # determine the angles and axes of rotation
    if n0p < 1.0:
        prevRotAngle = 2.0 * acos(n0p)
    else:
        prevRotAngle = np.pi / 2.0
    if n0c < 1.0:
        currRotAngle = 2.0 * acos(n0c)
    else:
        currRotAngle = np.pi / 2.0
    if n0n < 1.0:
        nextRotAngle = 2.0 * acos(n0n)
    else:
        nextRotAngle = np.pi / 2.0
    prevRotAngle = 2.0 * acos(n0p)
    currRotAngle = 2.0 * acos(n0c)
    nextRotAngle = 2.0 * acos(n0n)
    sine = sin(prevRotAngle / 2.0)
    prevRotVec = np.array([n1p / sine, n2p / sine, n3p / sine])
    sine = sin(currRotAngle / 2.0)
    currRotVec = np.array([n1c / sine, n2c / sine, n3c / sine])
    sine = sin(nextRotAngle / 2.0)
    nextRotVec = np.array([n1n / sine, n2n / sine, n3n / sine])

    # derivatives for the angle and axis of rotation for the next state
    dRotAngle = ((3.0 * nextRotAngle - 4.0 * currRotAngle + prevRotAngle) /
                 (2.0 * dTime))
    dRotVec = np.array([0.0, 0.0, 0.0])
    dRotVec = ((3.0 * nextRotVec - 4.0 * currRotVec + prevRotVec) /
               (2.0 * dTime))

    # Euler's formula for computing the axis of spin
    spinVec = np.array([0.0, 0.0, 0.0])
    spinVec = (dRotAngle * nextRotVec + sin(nextRotAngle) * dRotVec +
               (1.0 - cos(nextRotAngle)) * np.cross(nextRotVec, dRotVec))

    # create the skew symmetric spin matrix and return it
    spinMtx = np.zeros((3, 3), dtype=float)
    spinMtx[0, 1] = -spinVec[2]
    spinMtx[0, 2] = spinVec[1]
    spinMtx[1, 0] = spinVec[2]
    spinMtx[1, 2] = -spinVec[0]
    spinMtx[2, 0] = -spinVec[1]
    spinMtx[2, 1] = spinVec[0]
    return spinMtx


"""
Changes made in version "1.0.0":

This is the initial version of this module.  Changes were not documented in the
beta versions of this software.
"""