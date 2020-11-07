#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import Chord
import math as m
import numpy as np
from pivotIncomingF import Pivot
from vertices import Vertex

"""
Created on Mon Jan 28 2019
Updated on Mon Nov 06 2020

A test file for class Chord in file chords.py.

author: Shahla Zamani
"""


def run():
    print('\nThis test case considers a deforming Chord.')
    print('Be patient, the constitutive equations are being integrated.\n')

    # time step size used for numeric integration
    dTime = 1.0e-3

    # tolerance for controlling local trunction error during integration
    tol = 0.001

    # impose a far-field deformation history
    F0 = np.eye(3, dtype=float)
    F1 = np.copy(F0)
    F1[0, 0] += 0.01
    F1[1, 1] -= 0.01
    F1[1, 0] -= 0.01
    F1[2, 0] += 0.01
    F2 = np.copy(F1)
    F2[0, 0] += 0.01
    F2[1, 1] -= 0.01
    F2[0, 1] += 0.02
    F2[2, 0] += 0.01
    F3 = np.copy(F2)
    F3[0, 0] += 0.02
    F3[1, 1] -= 0.02
    F3[0, 2] -= 0.01
    F3[2, 1] += 0.02

    # re-index the co-ordinate systems according to pivot in pivotIncomingF.py
    p = Pivot(F0)
    p.update(F1)
    p.advance()
    p.update(F2)
    p.advance()
    p.update(F3)
    # get this histories re-indexed deformation gradients
    pF0 = p.pivotedF('ref')
    pF1 = p.pivotedF('prev')
    pF2 = p.pivotedF('curr')
    pF3 = p.pivotedF('next')

    # quantities used when assigning a vertice's natrual co-ordinates
    sqrt3 = m.sqrt(3.0)
    # phi is the golden ratio which appears frequently in dodecahedral geometry
    phi = (1.0 + m.sqrt(5.0)) / 2.0
    # co-ordinates for vertex 1 in its natural co-ordinate frame of reference
    v1_0 = np.array([1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3])
    # co-ordinates for vertex 9 in its natural co-ordinate frame of reference
    v9_0 = np.array([0.0, phi / sqrt3, 1.0 / (sqrt3 * phi)])

    # create the two vertices of a chord and then create the chord
    pv1_0 = np.matmul(pF0, v1_0)
    v1 = Vertex(1, (pv1_0[0], pv1_0[1], pv1_0[2]), dTime)
    pv9_0 = np.matmul(pF0, v9_0)
    v9 = Vertex(9, (pv9_0[0], pv9_0[1], pv9_0[2]), dTime)
    c = Chord(2, v1, v9, dTime, tol)

    # update and advance these three objects for the deformation history given
    pv1_1 = np.matmul(pF1, v1_0)
    v1.update((pv1_1[0], pv1_1[1], pv1_1[2]))
    pv9_1 = np.matmul(pF1, v9_0)
    v9.update((pv9_1[0], pv9_1[1], pv9_1[2]))
    c.update()
    v1.advance()
    v9.advance()
    c.advance(p)
    pv1_2 = np.matmul(pF2, v1_0)
    v1.update((pv1_2[0], pv1_2[1], pv1_2[2]))
    pv9_2 = np.matmul(pF2, v9_0)
    v9.update((pv9_2[0], pv9_2[1], pv9_2[2]))
    c.update()
    v1.advance()
    v9.advance()
    c.advance(p)
    pv1_3 = np.matmul(pF3, v1_0)
    v1.update((pv1_3[0], pv1_3[1], pv1_3[2]))
    pv9_3 = np.matmul(pF3, v9_0)
    v9.update((pv9_3[0], pv9_3[1], pv9_3[2]))
    c.update()

    # get this histories re-indexed deformation gradients
    pF0 = p.pivotedF('ref')
    pF1 = p.pivotedF('prev')
    pF2 = p.pivotedF('curr')
    pF3 = p.pivotedF('next')

    mMtx = c.massMatrix()
    
    # normalize this matrix
    maxEle = np.amax(mMtx)
    mass = mMtx / maxEle
    
    # mass = np.around(mMtx, decimals = 10)
    
    print("\nThe normalized mass matrix for this chord is:")
    print(mass)
    print("whose determinant is {:8.5e}".format(np.linalg.det(mass)))
    print("and whose inverse is ")
    print(np.linalg.inv(mass))

    print("\nThe stiffness matrix for this chord is:")
    stiff = c.stiffnessMatrix(p)
    print(stiff)

    print("\nThe force vector for this chord is:")
    force = c.forcingFunction()
    print(force)

run()
