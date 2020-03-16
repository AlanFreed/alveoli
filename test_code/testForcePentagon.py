#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import chord
import math as m
from pentForces import pentForce
from vertices import vertex
import numpy as np


"""
Created on March 07 2020
Updated on March 10 2020

A test file for class pentForce in file pentForces.py to find the force vector

author: Shahla Zamani
"""


def run(chordGaussPts, state):

    h = 0.1
    # phi is the golden ratio which appears in dodecahedra geometry
    phi = (1.0 + m.sqrt(5.0)) / 2.0
    sqrt3 = m.sqrt(3.0)


    # assign the vertices for pentagon 2
    v2 = vertex(2, 1.0 / sqrt3, 1.0 / sqrt3, -1.0 / sqrt3, h)
    v3 = vertex(3, -1.0 / sqrt3, 1.0 / sqrt3, -1.0 / sqrt3, h)
    v10 = vertex(10, 0.0, phi / sqrt3, -1.0 / (sqrt3 * phi), h)
    v17 = vertex(17, 1.0 / (sqrt3 * phi), 0.0, -phi / sqrt3, h)
    v18 = vertex(18, -1.0 / (sqrt3 * phi), 0.0, -phi / sqrt3, h)


    # assign the cords for a pentagon that inscribes an unit circle
    c1 = chord(4, v3, v10, h, chordGaussPts)
    c2 = chord(3, v2, v10, h, chordGaussPts)
    c3 = chord(10, v2, v17, h, chordGaussPts)
    c4 = chord(11, v17, v18, h, chordGaussPts)
    c5 = chord(12, v3, v18, h, chordGaussPts)


    # create the pentagon
    p = pentForce(2, c1, c2, c3, c4, c5, h, chordGaussPts)
    p.update()
    p.advance()
    
    cauchystress = np.array([[1, 1], [1, 1]])
    
    
    force = p.pentagonForcingFunction(cauchystress, state)

    print('A pentagon with {} chordal Gauss points has a force vector of:\n'
          .format(chordGaussPts))
    print(force)


print("")
run(1,'n')
run(2,'n')
run(3,'n')

