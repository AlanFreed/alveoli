#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import chord
import math as m
from pentagons import pentagon
from vertices import vertex

"""
Created on Feb 09 2020
Updated on Feb 10 2020

A test file for class pentagon in file pentagon.py to find the force vector

author: Prof. Alan Freed, Shahla Zamani
"""


def run(pentagonGaussPts, state):

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
    chordGaussPts = 3
    c1 = chord(4, v3, v10, h, chordGaussPts)
    c2 = chord(3, v2, v10, h, chordGaussPts)
    c3 = chord(10, v2, v17, h, chordGaussPts)
    c4 = chord(11, v17, v18, h, chordGaussPts)
    c5 = chord(12, v3, v18, h, chordGaussPts)


    # create the pentagon
    p = pentagon(2, c1, c2, c3, c4, c5, h, pentagonGaussPts)
    p.update()
    p.advance()
    
    sp = 1
    st = 1
    ss = 1
    state
    
    force = p.forcingFunction(sp, st, ss, pentagonGaussPts, state)

    print('A pentagon with {} Gauss points has a force vector of:\n'
          .format(pentagonGaussPts, state))
    print(force)


print("")
run(1, 'n')
run(4, 'n')
run(7, 'n')

