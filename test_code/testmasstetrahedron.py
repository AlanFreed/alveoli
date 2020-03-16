#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tetrahedra import tetrahedron
from vertices import vertex


"""
Created on Mon Feb 25 2020
Updated on Wed Feb 25 2020

A test file for class pentagon in file tetrahedra.py to find the mass matrix

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

    # create the pentagon
    t = tetrahedron(1, v1, v2, v3, v4, h, gaussPts)

    t.update()
    t.advance()

    mass = t.massMatrix()
    # normalize this matrix
    maxEle = np.amax(mass)
    mass = mass / maxEle

    print('A pentagon with {} Gauss points has a normalized mass matrix of:\n'
          .format(gaussPts))
    print(mass)
    print('\nwith a determinant of {:6.4e}.\n'.format(np.linalg.det(mass)))
    


print("\nFor 1 Gauss point.\n")
run(1)
print("\nFor 4 Gauss points.\n")
run(4)
print("\nFor 5 Gauss points.\n")
run(5)
