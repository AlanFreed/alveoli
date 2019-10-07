#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import cos, sqrt, tan
import numpy as np
from vertices import coordinatesToString
from vertices import vertex

"""
Created on Mon Jan 21 2019
Updated on Fri Jul 05 2019

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
    print('\nThe diameter of a normalized dodecahedron is:')
    print('D  = {}'.format(dia))
    print('D0 = {}'.format(dia0))


def run():
    number = 11
    h = 0.1
    x0 = 1.23456789
    y0 = -2.3456789
    z0 = 3.456789
    x1 = 1.34567789
    y1 = -2.23456789
    z1 = 3.3456789
    x2 = 1.456789
    y2 = -2.123456789
    z2 = 3.23456789
    x3 = 1.5678901234
    y3 = -2.012345678
    z3 = 3.123456789
    v = vertex(number, x0, y0, z0, h)
    v.update(x1, y1, z1)
    v.advance()
    v.update(x2, y2, z2)
    v.advance()
    v.update(x3, y3, z3)
    print('\nThe coordinates are:')
    x, y, z = v.coordinates('reference')
    print('Referece: ' + v.toString('r'))
    print('Previous: ' + v.toString('p'))
    print('Current:  ' + v.toString('c'))
    print('Next:     ' + v.toString('n'))
    print('\nThe displacements are:')
    u = v.displacement('prev')
    print('Previous: ' + coordinatesToString(u[0], u[1], u[2]))
    u = v.displacement('curr')
    print('Current:  ' + coordinatesToString(u[0], u[1], u[2]))
    u = v.displacement('next')
    print('Next:     ' + coordinatesToString(u[0], u[1], u[2]))
    print('\nThe velocities are:')
    du = v.velocity('previous')
    print('Previous: ' + coordinatesToString(du[0], du[1], du[2]))
    du = v.velocity('current')
    print('Current:  ' + coordinatesToString(du[0], du[1], du[2]))
    du = v.velocity('next')
    print('Next:     ' + coordinatesToString(du[0], du[1], du[2]))
    print('\nThe acclerations are:')
    d2u = v.acceleration('p')
    print('Previous: ' + coordinatesToString(d2u[0], d2u[1], d2u[2]))
    d2u = v.acceleration('c')
    print('Current:  ' + coordinatesToString(d2u[0], d2u[1], d2u[2]))
    d2u = v.acceleration('n')
    print('Next:     ' + coordinatesToString(d2u[0], d2u[1], d2u[2]))


nominalDiameter()
run()
