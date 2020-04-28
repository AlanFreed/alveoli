#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tetrahedra import tetrahedron
from vertices import vertex


"""
Created on Mon Feb 28 2020
Updated on Wed Feb 28 2020

A test file for class pentagon in file tetrahedra.py to find the stiffness matrix

author: Shahla Zamani
"""


def run(gaussPts):
    # the timestep size
    h = 0.1

    # assign the vertices for a natural tetrahedron
    v1 = vertex(1, 0.0, 0.0, 0.0, h)
    v2 = vertex(2, 1.0, 0.0, 0.0, h)
    v3 = vertex(3, 0.0, 1.0, 0.0, h)
    v4 = vertex(4, 0.0, 0.0, 1.0, h)

    # create the tetrahedron
    t = tetrahedron(1, v1, v2, v3, v4, h, gaussPts)

    t.update()
    t.advance()
    
    p = 1
    s1 = 1
    s2 = 1
    t1 = 1
    t2 = 1
    t3 = 1
    
    M = np.ones((7, 7), dtype=float)

    

    stiff = t.stiffnessMatrix(M, p, s1, s2, t1, t2, t3)


    print('A tetrahedron with {} Gauss points has a stiffness matrix of:\n'
          .format(gaussPts))
    print(stiff)

    


print("\nFor 1 Gauss point.\n")
run(1)
print("\nFor 4 Gauss points.\n")
run(4)
print("\nFor 5 Gauss points.\n")
run(5)
