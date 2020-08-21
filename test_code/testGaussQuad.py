#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gaussQuadChords import GaussQuadrature as GQChords
from gaussQuadTriangles import GaussQuadrature as GQTris
from gaussQuadTetrahedra import GaussQuadrature as GQTets
from gaussQuadPentagons import GaussQuadrature as GQPents
from math import cos, pi, sin
from shapeFnPentagons import ShapeFunction

"""
Created on Wed Apr 15 2020
Updated on Wed Jul 10 2020

A test file for exported objects in files: gaussQuadChords.py,
gaussQuadTriangles.py, gaussQuadTetrahedra.py and gaussQuadPentagons.py.

author: Prof. Alan Freed
"""


def run():
    print("Gauss quadrature for 1D chords.")
    quad = GQChords()
    print("      node    co-ordinate    weight")
    for i in range(1, quad.gaussPoints()+1):
        x = quad.coordinates(i)
        w = quad.weight(i)
        print("        {}       {:7.4f}     {:7.4f}".format(i, x[0], w))
    print("")
    print("Gauss quadrature for 2D triangles.")
    quad = GQTris()
    print("      node          co-ordinates         weight")
    for i in range(1, quad.gaussPoints()+1):
        x = quad.coordinates(i)
        w = quad.weight(i)
        print("        {}       {:7.4f}     {:7.4f}     {:7.4f}"
              .format(i, x[0], x[1], w))
    print("")
    print("Gauss quadrature for 3D tetrahedra.")
    quad = GQTets()
    print("      node                co-ordinates               weight")
    for i in range(1, quad.gaussPoints()+1):
        x = quad.coordinates(i)
        w = quad.weight(i)
        print("       {:2d}       {:7.4f}     {:7.4f}     {:7.4f}     {:7.4f}"
              .format(i, x[0], x[1], x[2], w))
    print("")
    print("Gauss quadrature for 2D pentagons.")
    quad = GQPents()
    print("      node          co-ordinates         weight")
    for i in range(1, quad.gaussPoints()+1):
        x = quad.coordinates(i)
        w = quad.weight(i)
        print("        {}       {:7.4f}     {:7.4f}     {:7.4f}"
              .format(i, x[0], x[1], w))
    print("")
    print("Shape functions evaluated at the Gauss points of a pentagon are:")
    for i in range(1, quad.gaussPoints()+1):
        print("")
        sf = ShapeFunction(quad.coordinates(i))
        print("For Gauss point {}:".format(i))
        print("   N1 = {}".format(sf.N1))
        print("   N2 = {}".format(sf.N2))
        print("   N3 = {}".format(sf.N3))
        print("   N4 = {}".format(sf.N4))
        print("   N5 = {}".format(sf.N5))
    print("")
    print("The nodal co-ordinates for a pentagon are:")
    print("      node          co-ordinates")
    x = cos(pi/2)
    y = sin(pi/2)
    print("        {}       {}     {}".format(1, x, y))
    x = cos(9*pi/10)
    y = sin(9*pi/10)
    print("        {}       {}     {}".format(2, x, y))
    x = cos(13*pi/10)
    y = sin(13*pi/10)
    print("        {}       {}     {}".format(3, x, y))
    x = cos(17*pi/10)
    y = sin(17*pi/10)
    print("        {}       {}     {}".format(4, x, y))
    x = cos(21*pi/10)
    y = sin(21*pi/10)
    print("        {}       {}     {}".format(5, x, y))


run()
