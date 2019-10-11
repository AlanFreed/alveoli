#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tetrahedra import tetrahedron
import numpy as np
from vertices import vertex
# for creating graphics
from matplotlib import pyplot as plt
from matplotlib import rc
from pylab import rcParams

"""
Created on Mon Oct 07 2019
Updated on Wed Oct 07 2019

Tests the module tetrahedra.py.

author: Prof. Alan Freed
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

    print('Edge length of a pentagon inscribed in an unit circle is {:8.6F}'
          .format(len0))
    print('The area of this pentagon should be {:8.6F}; it is {:8.6F}'
          .format(area, p.area('ref')))
    n1, n2, n3 = p.normal('ref')
    x1, x2, x3 = p.centroid('ref')
    print('The normal to this pentagon is: ' + coordinatesToString(n1, n2, n3))
    print('that is rooted at the centroid: ' + coordinatesToString(x1, x2, x3))


print("\nFor 1 Gauss point.\n")
run(1)
print("\nFor 4 Gauss points.\n")
run(4)
print("\nFor 5 Gauss points.\n")
run(5)
