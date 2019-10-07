#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import chord
import math as m
import numpy as np
from pentagons import pentagon
from vertices import coordinatesToString
from vertices import vertex

"""
Created on Tue Jan 29 2019
Updated on Fri Jul 05 2019

A test file for class pentagon in file pentagons.py.

author: Prof. Alan Freed
"""


def run():
    print('')
    # omega is half the inside angle of a regular pentagon, i.e., 54 deg
    omega = 54.0 * np.pi / 180.0
    # septal cord length for a pentagon inscribed in an unit circle
    len0 = 2.0 * m.cos(omega)
    # normalized area of such a regular pentagon is
    area = 5.0 * m.sin(omega) * m.cos(omega)
    # the timestep size
    h = 0.1
    # the number of Gauss points
    chordGaussPts = 2
    pentagonGaussPts = 4

    # assign the vertices for pentagon 1 in the dodecahedron
    v1 = vertex(1, m.cos(m.pi/2), m.sin(m.pi/2), 0.0, h)
    v2 = vertex(2, m.cos(9*m.pi/10), m.sin(9*m.pi/10), 0.0, h)
    v3 = vertex(3, m.cos(13*m.pi/10), m.sin(13*m.pi/10), 0.0, h)
    v4 = vertex(4, m.cos(17*m.pi/10), m.sin(17*m.pi/10), 0.0, h)
    v5 = vertex(5, m.cos(21*m.pi/10), m.sin(21*m.pi/10), 0.0, h)

    # assign the cords for a pentagon that inscribes an unit circle
    c1 = chord(1, v5, v1, h, chordGaussPts)
    c2 = chord(2, v1, v2, h, chordGaussPts)
    c3 = chord(3, v2, v3, h, chordGaussPts)
    c4 = chord(6, v3, v4, h, chordGaussPts)
    c5 = chord(7, v4, v5, h, chordGaussPts)

    # create the pentagon
    p = pentagon(1, c1, c2, c3, c4, c5, h, pentagonGaussPts)
    print('Edge length of a pentagon inscribed in an unit circle is {:8.6F}'
          .format(len0))
    print('The area of this pentagon should be {:8.6F}; it is {:8.6F}'
          .format(area, p.area('ref')))
    n1, n2, n3 = p.normal('ref')
    x1, x2, x3 = p.centroid('ref')
    print('The normal to this pentagon is: ' + coordinatesToString(n1, n2, n3))
    print('that is rooted at the centroid: ' + coordinatesToString(x1, x2, x3))


run()
