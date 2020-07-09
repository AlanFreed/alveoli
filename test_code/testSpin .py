#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import cos, sin
import numpy as np
from pivotIncomingF import Pivot
from spin import prevSpin, currSpin, nextSpin

"""
Created on Tue Jan 22 2019
Updated on Tue Jul 07 2020

A test file for computing the skew symmetric spin matrix.

author Prof. Alan Freed
"""


def run():
    np.set_printoptions(precision=6, suppress=True)
    # the time step size
    h = 0.1

    # the angles of rotation
    theta1 = 7.0 * np.pi / 180.0   # 7 degrees
    theta2 = 8.0 * np.pi / 180.0   # 8 degrees
    theta3 = 9.0 * np.pi / 180.0   # 9 degrees

    # no re-indexing required to test spin
    F0 = np.eye(3, dtype=float)
    reindex = Pivot(F0)
    reindex.update(F0)
    reindex.advance()
    reindex.update(F0)
    reindex.advance()

    # rotate around the 3 axis

    # rotation at the previous time step
    sine = sin(theta1)
    cosine = cos(theta1)
    prevQ = np.array([[cosine, sine, 0.0],
                      [-sine, cosine, 0.0],
                      [0.0, 0.0, 1.0]])

    # rotation at the current time step
    sine = sin(theta2)
    cosine = cos(theta2)
    currQ = np.array([[cosine, sine, 0.0],
                      [-sine, cosine, 0.0],
                      [0.0, 0.0, 1.0]])

    # rotation at the next time step
    sine = sin(theta3)
    cosine = cos(theta3)
    nextQ = np.array([[cosine, sine, 0.0],
                      [-sine, cosine, 0.0],
                      [0.0, 0.0, 1.0]])

    # retrieve their associated spin matrices
    prevOmega = prevSpin(prevQ, currQ, nextQ, reindex, h)
    currOmega = currSpin(prevQ, currQ, nextQ, reindex, h)
    nextOmega = nextSpin(prevQ, currQ, nextQ, reindex, h)

    print('At the previous time step:')
    print('   rotation =')
    print(np.array2string(prevQ))
    print('   spin =')
    print(np.array2string(prevOmega))

    print('At the current time step:')
    print('   rotation =')
    print(np.array2string(currQ))
    print('   spin =')
    print(np.array2string(currOmega))

    print('At the next time step:')
    print('   rotation =')
    print(np.array2string(nextQ))
    print('   spin =')
    print(np.array2string(nextOmega))
    print()

    # rotate around the 2 axis

    # rotation at the previous time step
    sine = sin(theta1)
    cosine = cos(theta1)
    prevQ = np.array([[cosine, 0.0, sine],
                      [0.0, 1.0, 0.0],
                      [-sine, 0.0, cosine]])

    # rotation at the current time step
    sine = sin(theta2)
    cosine = cos(theta2)
    currQ = np.array([[cosine, 0.0, sine],
                      [0.0, 1.0, 0.0],
                      [-sine, 0.0, cosine]])

    # rotation at the next time step
    sine = sin(theta3)
    cosine = cos(theta3)
    nextQ = np.array([[cosine, 0.0, sine],
                      [0.0, 1.0, 0.0],
                      [-sine, 0.0, cosine]])

    # retrieve their associated spin matrices
    prevOmega = prevSpin(prevQ, currQ, nextQ, reindex, h)
    currOmega = currSpin(prevQ, currQ, nextQ, reindex, h)
    nextOmega = nextSpin(prevQ, currQ, nextQ, reindex, h)

    print('At the previous time step:')
    print('   rotation =')
    print(np.array2string(prevQ))
    print('   spin =')
    print(np.array2string(prevOmega))

    print('At the current time step:')
    print('   rotation =')
    print(np.array2string(currQ))
    print('   spin =')
    print(np.array2string(currOmega))

    print('At the next time step:')
    print('   rotation =')
    print(np.array2string(nextQ))
    print('   spin =')
    print(np.array2string(nextOmega))
    print()

    # rotate around the 1 axis

    # rotation at the previous time step
    sine = sin(theta1)
    cosine = cos(theta1)
    prevQ = np.array([[1.0, 0.0, 0.0],
                      [0.0, cosine, sine],
                      [0.0, -sine, cosine]])

    # rotation at the current time step
    sine = sin(theta2)
    cosine = cos(theta2)
    currQ = np.array([[1.0, 0.0, 0.0],
                      [0.0, cosine, sine],
                      [0.0, -sine, cosine]])

    # rotation at the next time step
    sine = sin(theta3)
    cosine = cos(theta3)
    nextQ = np.array([[1.0, 0.0, 0.0],
                      [0.0, cosine, sine],
                      [0.0, -sine, cosine]])

    # retrieve their associated spin matrices
    prevOmega = prevSpin(prevQ, currQ, nextQ, reindex, h)
    currOmega = currSpin(prevQ, currQ, nextQ, reindex, h)
    nextOmega = nextSpin(prevQ, currQ, nextQ, reindex, h)

    print('At the previous time step:')
    print('   rotation =')
    print(np.array2string(prevQ))
    print('   spin =')
    print(np.array2string(prevOmega))

    print('At the current time step:')
    print('   rotation =')
    print(np.array2string(currQ))
    print('   spin =')
    print(np.array2string(currOmega))

    print('At the next time step:')
    print('   rotation =')
    print(np.array2string(nextQ))
    print('   spin =')
    print(np.array2string(nextOmega))


run()
