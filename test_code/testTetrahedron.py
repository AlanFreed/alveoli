#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from vertices import Vertex
from pivotIncomingF import Pivot
from tetrahedra import tetrahedron



"""
Created on Tue Oct 16 2020
Updated on Fri Oct 19 2020

A test file for class tetrahedron in file tetrahedra.py.

author: Shahla Zamani
"""


def run():
    print('')

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
    
    
    # the timestep size
    h = 0.1

    # assign the vertices for a natural tetrahedron
    v1_0 = np.array([0.0, 0.0, 0.0])
    v2_0 = np.array([1.0, 0.0, 0.0])
    v3_0 = np.array([0.0, 1.0, 0.0])
    v4_0 = np.array([0.0, 0.0, 1.0])

    pv1_0 = np.matmul(piF0, v1_0)
    pv2_0 = np.matmul(piF0, v2_0)
    pv3_0 = np.matmul(piF0, v3_0)
    pv4_0 = np.matmul(piF0, v4_0)
    
    # assign the vertices for tetrahedron 1 in the dodecahedron
    v1 = Vertex(1, (pv1_0[0], pv1_0[1], pv1_0[2]), h)
    v2 = Vertex(2, (pv2_0[0], pv2_0[1], pv2_0[2]), h)
    v3 = Vertex(3, (pv3_0[0], pv3_0[1], pv3_0[2]), h)
    v4 = Vertex(4, (pv4_0[0], pv4_0[1], pv4_0[2]), h)



    # create the tetrahedron
    t = tetrahedron(1, v1, v2, v3, v4, h)

    # update and advance these three objects for the deformation history given
    pv1_1 = np.matmul(piF1, v1_0)
    pv2_1 = np.matmul(piF1, v2_0)
    pv3_1 = np.matmul(piF1, v3_0)
    pv4_1 = np.matmul(piF1, v4_0)
    
    v1.update((pv1_1[0], pv1_1[1], pv1_1[2]))
    v2.update((pv2_1[0], pv2_1[1], pv2_1[2]))
    v3.update((pv3_1[0], pv3_1[1], pv3_1[2]))
    v4.update((pv4_1[0], pv4_1[1], pv4_1[2]))
    
    t.update()
    
    v1.advance()
    v2.advance()
    v3.advance()
    v4.advance()
    
    t.advance(pi)





    pv1_2 = np.matmul(piF2, v1_0)
    pv2_2 = np.matmul(piF2, v2_0)
    pv3_2 = np.matmul(piF2, v3_0)
    pv4_2 = np.matmul(piF2, v4_0)
    
    v1.update((pv1_2[0], pv1_2[1], pv1_2[2]))
    v2.update((pv2_2[0], pv2_2[1], pv2_2[2]))
    v3.update((pv3_2[0], pv3_2[1], pv3_2[2]))
    v4.update((pv4_2[0], pv4_2[1], pv4_2[2]))
    
    t.update()
    
    v1.advance()
    v2.advance()
    v3.advance()
    v4.advance()
    
    t.advance(pi)
    



    pv1_3 = np.matmul(piF3, v1_0)
    pv2_3 = np.matmul(piF3, v2_0)
    pv3_3 = np.matmul(piF3, v3_0)
    pv4_3 = np.matmul(piF3, v4_0)
    
    v1.update((pv1_3[0], pv1_3[1], pv1_3[2]))
    v2.update((pv2_3[0], pv2_3[1], pv2_3[2]))
    v3.update((pv3_3[0], pv3_3[1], pv3_3[2]))
    v4.update((pv4_3[0], pv4_3[1], pv4_3[2]))

    t.update()
    
    v1.advance()
    v2.advance()
    v3.advance()
    v4.advance()
    
    t.advance(pi)
    
    
    # get this histories re-indexed deformation gradients
    piF0 = pi.pivotedF('ref')
    piF1 = pi.pivotedF('prev')
    piF2 = pi.pivotedF('curr')
    piF3 = pi.pivotedF('next')
    
    mass = t.massMatrix()
    
    # normalize this matrix
    maxEle = np.amax(mass)
    mass = mass / maxEle

    
    print("\nThe Noralized mass matrix for this tetrahedron is:")
    print(mass)
    print("whose determinant is {:8.5e}".format(np.linalg.det(mass)))
    print("and whose inverse is ")
    print(np.linalg.inv(mass))
    
    print("\nThe stiffness matrix for this tetrahedron is:")
    stiff = t.stiffnessMatrix(pi)
    print(stiff)

    print("\nThe force vector for this tetrahedron is:")
    force = t.forcingFunction()
    print(force)

run()


