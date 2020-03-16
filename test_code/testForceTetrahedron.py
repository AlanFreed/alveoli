#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tetrahedronForces import tetforce
from vertices import vertex


"""
Created on March 13 2020
Updated on March 16 2020

A test file for class tetforce in file tetrahedronForces.py to find the force 
vector

author: Shahla Zamani
"""


def run(triaGaussPts, state):
    # the timestep size
    h = 0.1

    # assign the vertices for a natural tetrahedron
    v1 = vertex(1, 1.0, 0.0, 0.0, h)
    v2 = vertex(2, 0.0, 1.0, 0.0, h)
    v3 = vertex(3, 0.0, 0.0, 1.0, h)

    # create the triangle
    tr = tetforce(1, v1, v2, v3, h, triaGaussPts)

    tr.update()
    tr.advance()
    
    cauchystress = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])    

    force = tr.tetrahedronforcingFunction(cauchystress, state)

    print('A tetrahedron with {} triangular Gauss points has a force vector of:\n'
          .format(triaGaussPts))
    print(force)


run(1,'n')
run(3,'n')
run(4,'n')
