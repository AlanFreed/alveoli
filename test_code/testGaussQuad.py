#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gaussQuadratures as gq

"""
Created on Wed Apr 15 2020
Updated on Wed Apr 15 2020

A test file for exported objects in file gaussQuadratures.py.

author: Prof. Alan Freed
"""


def run():
    print("Gauss quadratures for 1D chords.")
    print("   For 1st degree polynomials:")
    quad = gq.gaussQuadChord1
    print("      node    co-ordinate    weight")
    for i in range(quad.nodes):
        x = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("        {}       {:7.4f}     {:7.4f}".format(i+1, x, w))
    print("   For 3rd degree polynomials:")
    quad = gq.gaussQuadChord3
    print("      node    co-ordinate    weight")
    for i in range(quad.nodes):
        x = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("        {}       {:7.4f}     {:7.4f}".format(i+1, x, w))
    print("   For 5th degree polynomials:")
    quad = gq.gaussQuadChord5
    print("      node    co-ordinate    weight")
    for i in range(quad.nodes):
        x = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("        {}       {:7.4f}     {:7.4f}".format(i+1, x, w))
    print("")
    print("Gauss quadratures for 2D triangles.")
    print("   For 1st degree polynomials:")
    quad = gq.gaussQuadTriangle1
    print("      node          co-ordinates         weight")
    for i in range(quad.nodes):
        (x, y) = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("        {}       {:7.4f}     {:7.4f}     {:7.4f}"
              .format(i+1, x, y, w))
    print("   For 3rd degree polynomials:")
    quad = gq.gaussQuadTriangle3
    print("      node          co-ordinates         weight")
    for i in range(quad.nodes):
        (x, y) = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("        {}       {:7.4f}     {:7.4f}     {:7.4f}"
              .format(i+1, x, y, w))
    print("   For 5th degree polynomials:")
    quad = gq.gaussQuadTriangle5
    print("      node          co-ordinates         weight")
    for i in range(quad.nodes):
        (x, y) = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("        {}       {:7.4f}     {:7.4f}     {:7.4f}"
              .format(i+1, x, y, w))
    print("")
    print("Gauss quadratures for 2D pentagons.")
    print("   For 1st degree polynomials:")
    quad = gq.gaussQuadPentagon1
    print("      node          co-ordinates         weight")
    for i in range(quad.nodes):
        (x, y) = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("        {}       {:7.4f}     {:7.4f}     {:7.4f}"
              .format(i+1, x, y, w))
    print("   For 3rd degree polynomials:")
    quad = gq.gaussQuadPentagon3
    print("      node          co-ordinates         weight")
    for i in range(quad.nodes):
        (x, y) = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("        {}       {:7.4f}     {:7.4f}     {:7.4f}"
              .format(i+1, x, y, w))
    print("   For 5th degree polynomials:")
    quad = gq.gaussQuadPentagon5
    print("      node          co-ordinates         weight")
    for i in range(quad.nodes):
        (x, y) = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("        {}       {:7.4f}     {:7.4f}     {:7.4f}"
              .format(i+1, x, y, w))
    print("")
    print("Gauss quadratures for 3D tetrahedra.")
    print("   For 1st degree polynomials:")
    quad = gq.gaussQuadTetrahedron1
    print("      node                co-ordinates               weight")
    for i in range(quad.nodes):
        (x, y, z) = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("       {:2d}       {:7.4f}     {:7.4f}     {:7.4f}     {:7.4f}"
              .format(i+1, x, y, z, w))
    print("   For 3rd degree polynomials:")
    quad = gq.gaussQuadTetrahedron3
    print("      node                co-ordinates               weight")
    for i in range(quad.nodes):
        (x, y, z) = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("       {:2d}       {:7.4f}     {:7.4f}     {:7.4f}     {:7.4f}"
              .format(i+1, x, y, z, w))
    print("   For 5th degree polynomials:")
    quad = gq.gaussQuadTetrahedron5
    print("      node                co-ordinates               weight")
    for i in range(quad.nodes):
        (x, y, z) = quad.coordinates(i+1)
        w = quad.weight(i+1)
        print("       {:2d}       {:7.4f}     {:7.4f}      {:7.4f}    {:7.4f}"
              .format(i+1, x, y, z, w))


run()
