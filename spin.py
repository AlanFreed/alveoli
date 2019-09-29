#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import acos, cos, sin, sqrt
import numpy as np
import sys

"""
Module spin.py provides the coordinate spin tensor.

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
__date__ = "04-30-2019"
__update__ = "09-24-2019"
__author__ = "Alan D. Freed"
__author_email__ = "adfreed@tamu.edu"

"""
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
    by a stepsize of h.  They return the skew symmetric spin for these three
    instances in time.  All derivative estimates are second order accurate.
    These spin procedures each call the quaternion procedure.

    spin = prevSpin(prevRotationMtx, currRotationMtx, nextRotationMtx, h)
        where
            prevRotationMtx  is an orthogonal matrix at the previous time step
            currRotationMtx  is an orthogonal matrix at the current time step
            nextRotationMtx  is an orthogonal matrix at the next time step
            h                is the size of the time step
            spin             is the skew symmetric spin at the previous time

    spin = currSpin(prevRotationMtx, currRotationMtx, nextRotationMtx, h)
        where
            prevRotationMtx  is an orthogonal matrix at the previous time step
            currRotationMtx  is an orthogonal matrix at the current time step
            nextRotationMtx  is an orthogonal matrix at the next time step
            h                is the size of the time step
            spin             is the skew symmetric spin at the current time

    spin = prevSpin(prevRotationMtx, currRotationMtx, nextRotationMtx, h)
        where
            prevRotationMtx  is an orthogonal matrix at the previous time step
            currRotationMtx  is an orthogonal matrix at the current time step
            nextRotationMtx  is an orthogonal matrix at the next time step
            h                is the size of the time step
            spin             is the skew symmetric spin at the next time

References:
    1) R. A. Spurrier, "Comment on 'Singularity-free extraction of a quaternion
    from a direction-cosine matrix'", Journal of Spacecraft and Rockets (1978),
    255.
    2) A. D. Freed, J.-B. le~Graverend and K. R. Rajagopal, "A technical note:
    A Decomposition of Laplace Stretch with Applications in Inelasticity", in
    review.
"""


def quaternion(rotationMtx):
    # verify input
    if not isinstance(rotationMtx, np.ndarray):
        print('Error: rotationMtx sent to quaternion must be numpy.ndarray.')
        sys.exit()
    (rows, cols) = np.shape(rotationMtx)
    if (rows != 3) or (cols != 3):
        print('Error: function quaternion requires a 3x3 rotation matrix.')
        sys.exit()
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


def prevSpin(prevRotationMtx, currRotationMtx, nextRotationMtx, h):
    # verify the input
    if h < np.finfo(float).eps:
        print("Error: stepsize sent to chord constructor wasn't positive.")
        sys.exit()
    # get the quaternions for these three rotations
    n0p, n1p, n2p, n3p = quaternion(prevRotationMtx)
    n0c, n1c, n2c, n3c = quaternion(currRotationMtx)
    n0n, n1n, n2n, n3n = quaternion(nextRotationMtx)

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
                 (2.0 * h))
    dRotVec = np.array([0.0, 0.0, 0.0])
    dRotVec = (-nextRotVec + 4.0 * currRotVec - 3.0 * prevRotVec) / (2.0 * h)

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


def currSpin(prevRotationMtx, currRotationMtx, nextRotationMtx, h):
    # verify the input
    if h < np.finfo(float).eps:
        print("Error: stepsize sent to chord constructor wasn't positive.")
        sys.exit()
    # get the quaternions for these three rotations
    n0p, n1p, n2p, n3p = quaternion(prevRotationMtx)
    n0c, n1c, n2c, n3c = quaternion(currRotationMtx)
    n0n, n1n, n2n, n3n = quaternion(nextRotationMtx)

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
    dRotAngle = (nextRotAngle - prevRotAngle) / (2.0 * h)
    dRotVec = np.array([0.0, 0.0, 0.0])
    dRotVec = (nextRotVec - prevRotVec) / (2.0 * h)

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


def nextSpin(prevRotationMtx, currRotationMtx, nextRotationMtx, h):
    # verify the input
    if h < np.finfo(float).eps:
        print("Error: stepsize sent to chord constructor wasn't positive.")
        sys.exit()
    # get the quaternions for these three rotations
    n0p, n1p, n2p, n3p = quaternion(prevRotationMtx)
    n0c, n1c, n2c, n3c = quaternion(currRotationMtx)
    n0n, n1n, n2n, n3n = quaternion(nextRotationMtx)

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
                 (2.0 * h))
    dRotVec = np.array([0.0, 0.0, 0.0])
    dRotVec = (3.0 * nextRotVec - 4.0 * currRotVec + prevRotVec) / (2.0 * h)

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
