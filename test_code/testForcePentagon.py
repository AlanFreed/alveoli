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

    # assign the vertices for pentagon 1 in the dodecahedron
    v1 = vertex(1, m.cos(m.pi/2), m.sin(m.pi/2), 0.0, h)
    v2 = vertex(2, m.cos(9*m.pi/10), m.sin(9*m.pi/10), 0.0, h)
    v3 = vertex(3, m.cos(13*m.pi/10), m.sin(13*m.pi/10), 0.0, h)
    v4 = vertex(4, m.cos(17*m.pi/10), m.sin(17*m.pi/10), 0.0, h)
    v5 = vertex(5, m.cos(21*m.pi/10), m.sin(21*m.pi/10), 0.0, h)

    # assign the cords for a pentagon that inscribes an unit circle
    chordGaussPts = 3
    c1 = chord(1, v5, v1, h, chordGaussPts)
    c2 = chord(2, v1, v2, h, chordGaussPts)
    c3 = chord(3, v2, v3, h, chordGaussPts)
    c4 = chord(6, v3, v4, h, chordGaussPts)
    c5 = chord(7, v4, v5, h, chordGaussPts)

    # create the pentagon
    p = pentagon(1, c1, c2, c3, c4, c5, h, pentagonGaussPts)
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

