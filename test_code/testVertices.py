#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import cos, sqrt, tan
import numpy as np
from pivotIncomingF import pivot
from vertices import vectorToString, matrixToString, vertex

"""
Created on Mon Jan 21 2019
Updated on Mon May 04 2020

A test file for class vertex in file vertices.py.

author: Prof. Alan Freed
"""


def nominalDiameter():
    # projecting a dodecahedron onto a plane produces 20 triangles so
    alpha = np.pi / 20.0
    # omega is half the inside angle of a regular pentagon, i.e., 54 deg
    omega = 54.0 * np.pi / 180.0
    # phi is the golden ratio which appears in dodecahedra geometry
    phi = (1.0 + sqrt(5.0)) / 2.0
    # septal chord length in a dodecahedron that inscribes an unit sphere
    sqrt3 = sqrt(3.0)
    len0 = 2.0 / (sqrt3 * phi)
    # normalized dodecahedral diameter
    dia0 = tan(omega) * (1.0 + cos(alpha)) * len0
    dia = (1.0 + cos(alpha)) / (sqrt3 * cos(omega))
    print('\nThe diameter of a dodecahedron in its natural co-ordinates is:')
    print('D = {}'.format(dia))
    print('Theoretically, it should be')
    print('D = {}'.format(dia0))


def run():
    # impose a far-field deformation history
    F0 = np.eye(3, dtype=float)
    F1 = np.copy(F0)
    F1[0, 0] += 0.01
    F1[1, 1] -= 0.01
    F1[1, 0] -= 0.01
    F1[2, 0] += 0.01
    F2 = np.copy(F1)
    F2[0, 0] += 0.01
    F2[1, 1] -= 0.01
    F2[0, 1] += 0.02
    F2[2, 0] += 0.01
    F3 = np.copy(F2)
    F3[0, 0] += 0.02
    F3[1, 1] -= 0.02
    F3[0, 2] -= 0.01
    F3[2, 1] += 0.02
    p = pivot(F0)
    p.update(F1)
    p.advance()
    p.update(F2)
    p.advance()
    p.update(F3)

    # get this histories re-indexed deformation gradients
    pF0 = p.pivotedF('ref')
    pF1 = p.pivotedF('prev')
    pF2 = p.pivotedF('curr')
    pF3 = p.pivotedF('next')
    print('\nPivot cases for the imposed deformation were:')
    print('   reference: {}'.format(p.pivotCase('ref')))
    print('   previous:  {}'.format(p.pivotCase('prev')))
    print('   current:   {}'.format(p.pivotCase('curr')))
    print('   next:      {}'.format(p.pivotCase('next')))
    print('\nFor example, given an F_next of\n')
    print(matrixToString(F3))
    print('\nit re-indexes into a pivoted F_next with components of\n')
    print(matrixToString(pF3))

    # vertex 1 located in its natural co-ordinates for this deformation field
    xNC = np.array([1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)])
    x0 = np.matmul(pF0, xNC)
    x1 = np.matmul(pF1, xNC)
    x2 = np.matmul(pF2, xNC)
    x3 = np.matmul(pF3, xNC)

    # create vertex 1
    number = 1
    coordinates = (x0[0], x0[1], x0[2])
    h = 0.001
    v = vertex(number, coordinates, h)
    coordinates = (x1[0], x1[1], x1[2])
    v.update(coordinates)
    v.advance()
    coordinates = (x2[0], x2[1], x2[2])
    v.update(coordinates)
    v.advance()
    coordinates = (x3[0], x3[1], x3[2])
    v.update(coordinates)

    print('\nThe co-ordinate values for this vertex are:')
    print('   reference: ' + v.toString('ref'))
    print('   previous:  ' + v.toString('prev'))
    print('   current:   ' + v.toString('curr'))
    print('   next:      ' + v.toString('next'))

    print('\nwhose displacement vectors in their respective frames are:')
    up = v.displacement(p, 'prev')
    print('   previous:  ' + vectorToString(up))
    uc = v.displacement(p, 'curr')
    print('   current:   ' + vectorToString(uc))
    un = v.displacement(p, 'next')
    print('   next:      ' + vectorToString(un))
    print("that when pushed back into the user's frame of reference become")
    u0 = p.localToGlobalVector(up, 'prev')
    print('   previous:  ' + vectorToString(u0))
    u0 = p.localToGlobalVector(uc, 'curr')
    print('   current:   ' + vectorToString(u0))
    u0 = p.localToGlobalVector(un, 'next')
    print('   next:      ' + vectorToString(u0))

    print('\nwhose velocity vectors in their respective frames are:')
    dup = v.velocity(p, 'prev')
    print('   previous:  ' + vectorToString(dup))
    duc = v.velocity(p, 'curr')
    print('   current:   ' + vectorToString(duc))
    dun = v.velocity(p, 'next')
    print('   next:      ' + vectorToString(dun))
    print("that when pushed back tino the user's frame of reference become")
    du0 = p.localToGlobalVector(dup, 'prev')
    print('   previous:  ' + vectorToString(du0))
    du0 = p.localToGlobalVector(duc, 'curr')
    print('   current:   ' + vectorToString(du0))
    du0 = p.localToGlobalVector(dun, 'next')
    print('   next:      ' + vectorToString(du0))

    print('\nand whose accleration vectors in their respective frames are:')
    d2up = v.acceleration(p, 'prev')
    print('   previous:  ' + vectorToString(d2up))
    d2uc = v.acceleration(p, 'curr')
    print('   current:   ' + vectorToString(d2uc))
    d2un = v.acceleration(p, 'next')
    print('   next:      ' + vectorToString(d2un))
    print("that when pushed back into the user's frame of reference become")
    d2u0 = p.localToGlobalVector(d2up, 'prev')
    print('   previous:  ' + vectorToString(d2u0))
    d2u0 = p.localToGlobalVector(d2uc, 'curr')
    print('   current:   ' + vectorToString(d2u0))
    d2u0 = p.localToGlobalVector(d2un, 'next')
    print('   next:      ' + vectorToString(d2u0))
    print('which are the same, as they ought to be.')

    print('\nTest printing a vertex object with the print(object) command.')
    print(v)
    print("Notice: the current state is printed when using the print command.")


nominalDiameter()
run()
