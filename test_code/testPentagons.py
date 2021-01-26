#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import Chord
import math as m
import numpy as np
from pentagons import pentagon
from vertices import Vertex
from pivotIncomingF import Pivot


"""
Created on Tue Jan 29 2019
Updated on Fri Nov 06 2020

A test file for class pentagon in file pentagons.py.

author: Shahla Zamani
"""


def run():
    print('')

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
    pi = Pivot(F0)
    pi.update(F1)
    pi.advance()
    pi.update(F2)
    pi.advance()
    pi.update(F3)
    # get this histories re-indexed deformation gradients
    piF0 = pi.pivotedF('ref')
    piF1 = pi.pivotedF('prev')
    piF2 = pi.pivotedF('curr')
    piF3 = pi.pivotedF('next')
    
    
    
    # omega is half the inside angle of a regular pentagon, i.e., 54 deg
    omega = 54.0 * np.pi / 180.0
    # normalized area of such a regular pentagon is
    area = 5.0 * m.sin(omega) * m.cos(omega)
    # the timestep size
    h = 0.1

    v1_0 = np.array([m.cos(m.pi/2), m.sin(m.pi/2), 0.0])
    v2_0 = np.array([m.cos(9*m.pi/10), m.sin(9*m.pi/10), 0.0])
    v3_0 = np.array([m.cos(13*m.pi/10), m.sin(13*m.pi/10), 0.0])
    v4_0 = np.array([m.cos(17*m.pi/10), m.sin(17*m.pi/10), 0.0])
    v5_0 = np.array([m.cos(21*m.pi/10), m.sin(21*m.pi/10), 0.0])

    pv1_0 = np.matmul(piF0, v1_0)
    pv2_0 = np.matmul(piF0, v2_0)
    pv3_0 = np.matmul(piF0, v3_0)
    pv4_0 = np.matmul(piF0, v4_0)
    pv5_0 = np.matmul(piF0, v5_0)

    # assign the vertices for pentagon 1 in the dodecahedron
    v1 = Vertex(1, (pv1_0[0], pv1_0[1], pv1_0[2]), h)
    v2 = Vertex(2, (pv2_0[0], pv2_0[1], pv2_0[2]), h)
    v3 = Vertex(3, (pv3_0[0], pv3_0[1], pv3_0[2]), h)
    v4 = Vertex(4, (pv4_0[0], pv4_0[1], pv4_0[2]), h)
    v5 = Vertex(5, (pv5_0[0], pv5_0[1], pv5_0[2]), h)

    # assign the cords for a pentagon that inscribes an unit circle
    c1 = Chord(1, v5, v1, h, tol)
    c2 = Chord(2, v1, v2, h, tol)
    c3 = Chord(3, v2, v3, h, tol)
    c4 = Chord(6, v3, v4, h, tol)
    c5 = Chord(7, v4, v5, h, tol)

    # create the pentagon
    p = pentagon(1, c1, c2, c3, c4, c5, h)

    # update and advance these three objects for the deformation history given
    pv1_1 = np.matmul(piF1, v1_0)
    pv2_1 = np.matmul(piF1, v2_0)
    pv3_1 = np.matmul(piF1, v3_0)
    pv4_1 = np.matmul(piF1, v4_0)
    pv5_1 = np.matmul(piF1, v5_0)
    
    v1.update((pv1_1[0], pv1_1[1], pv1_1[2]))
    v2.update((pv2_1[0], pv2_1[1], pv2_1[2]))
    v3.update((pv3_1[0], pv3_1[1], pv3_1[2]))
    v4.update((pv4_1[0], pv4_1[1], pv4_1[2]))
    v5.update((pv5_1[0], pv5_1[1], pv5_1[2]))
    
    c1.update()
    c2.update()
    c3.update()
    c4.update()
    c5.update()

    p.update()
    
    v1.advance()
    v2.advance()
    v3.advance()
    v4.advance()
    v5.advance()
    
    c1.advance(pi)
    c2.advance(pi)
    c3.advance(pi)
    c4.advance(pi)
    c5.advance(pi)
    
    p.advance(pi)





    pv1_2 = np.matmul(piF2, v1_0)
    pv2_2 = np.matmul(piF2, v2_0)
    pv3_2 = np.matmul(piF2, v3_0)
    pv4_2 = np.matmul(piF2, v4_0)
    pv5_2 = np.matmul(piF2, v5_0)
    
    v1.update((pv1_2[0], pv1_2[1], pv1_2[2]))
    v2.update((pv2_2[0], pv2_2[1], pv2_2[2]))
    v3.update((pv3_2[0], pv3_2[1], pv3_2[2]))
    v4.update((pv4_2[0], pv4_2[1], pv4_2[2]))
    v5.update((pv5_2[0], pv5_2[1], pv5_2[2]))
    
    c1.update()
    c2.update()
    c3.update()
    c4.update()
    c5.update()
    
    p.update()
    
    v1.advance()
    v2.advance()
    v3.advance()
    v4.advance()
    v5.advance()
    
    c1.advance(pi)
    c2.advance(pi)
    c3.advance(pi)
    c4.advance(pi)
    c5.advance(pi)
    
    p.advance(pi)
    



    pv1_3 = np.matmul(piF3, v1_0)
    pv2_3 = np.matmul(piF3, v2_0)
    pv3_3 = np.matmul(piF3, v3_0)
    pv4_3 = np.matmul(piF3, v4_0)
    pv5_3 = np.matmul(piF3, v5_0)
    
    v1.update((pv1_3[0], pv1_3[1], pv1_3[2]))
    v2.update((pv2_3[0], pv2_3[1], pv2_3[2]))
    v3.update((pv3_3[0], pv3_3[1], pv3_3[2]))
    v4.update((pv4_3[0], pv4_3[1], pv4_3[2]))
    v5.update((pv5_3[0], pv5_3[1], pv5_3[2]))
    
    c1.update()
    c2.update()
    c3.update()
    c4.update()
    c5.update()
    
    p.update()
    
    v1.advance()
    v2.advance()
    v3.advance()
    v4.advance()
    v5.advance()
    
    c1.advance(pi)
    c2.advance(pi)
    c3.advance(pi)
    c4.advance(pi)
    c5.advance(pi)
    
    p.advance(pi)
    
    
    # get this histories re-indexed deformation gradients
    piF0 = pi.pivotedF('ref')
    piF1 = pi.pivotedF('prev')
    piF2 = pi.pivotedF('curr')
    piF3 = pi.pivotedF('next')
    
    print('The area of this pentagon should be {:8.6F}; it is {:8.6F}'
          .format(area, p.area('ref')))

    
    mass = p.massMatrix()
    
    # normalize this matrix
    maxEle = np.amax(mass)
    mass = mass / maxEle
    
    
    print("\nThe normalized mass matrix for this pentagon is:")
    print(mass)
    print("whose determinant is {:8.5e}".format(np.linalg.det(mass)))
    print("and whose inverse is ")
    print(np.linalg.inv(mass))

    print("\nThe tangent stiffness matrix (C) for this pentagon is:")
    C = p.tangentStiffnessMtxC()
    print(C)
    
    print("\nThe secant stiffness matrix (K) for this pentagon is:")
    K = p.secantStiffnessMtxK(pi)
    print(K)

    print("\nThe force vector for this pentagon is:")
    F = p.forcingFunction()
    print(F)

run()
