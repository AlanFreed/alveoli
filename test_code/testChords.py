#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import chord
import math as m
import numpy as np
from vertices import coordinatesToString
from vertices import vertex

"""
Created on Mon Jan 28 2019
Last Updated:  Feb 10 2020

A test file for class chord in file chords.py.

author: Prof. Alan Freed
"""


def run():
    np.set_printoptions(precision=5)
    print('\nThis test case considers a deforming chord.\n')

    # phi is the golden ratio which appears in dodecahedra geometry
    phi = (1.0 + m.sqrt(5.0)) / 2.0

    # time step size
    h = 0.1
    gaussPts = 3
    sqrt3 = m.sqrt(3.0)
    v1 = vertex(1, 1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3, h)
    v9 = vertex(9, 0.0, phi / sqrt3, 1.0 / (sqrt3 * phi), h)
    c = chord(2, v1, v9, h, gaussPts)

    # print out the material properties of this chord
    print("This chord has the following material properties:")
    print(" chord mass density   = {:8.5e} (gr/cm^3)".format(c.massDensity()))
    print(" chord cross-section  = {:8.5e} (cm^2)".format(c.area('r')))
    print(" collagen has an area = {:8.5e} (cm^2)".format(c.areaCollagen('r')))
    print(" elastin  has an area = {:8.5e} (CM^2)".format(c.areaElastin('r')))
    E1, E2, et = c.matPropCollagen()
    print("collagen material properties are:")
    print("    E1  = {:8.5e} barye".format(E1))
    print("    E2  = {:8.5e} barye".format(E2))
    print("    e_t = {:8.5e}".format(et))
    E1, E2, et = c.matPropElastin()
    print("elastin material properties are:")
    print("    E1  = {:8.5e} barye".format(E1))
    print("    E2  = {:8.5e} barye".format(E2))
    print("    e_t = {:8.5e}".format(et))

    print('reference state:')
    print(c.toString('ref'))
    print('length:   {:8.5e}'.format(c.length('ref')))
    centroid = c.centroid('ref')
    print('centroid: ' +
          coordinatesToString(centroid[0], centroid[1], centroid[2]))
    v1 = c.getVertex(1)
    v9 = c.getVertex(9)

    v1.update(1.0 / sqrt3, 0.925 / sqrt3, 1.1 / sqrt3)
    v1.advance()
    v9.update(-0.1, 0.9 * phi / sqrt3, 0.9 / (sqrt3 * phi))
    v9.advance()
    c.update()
    c.advance()

    v1.update(1.0 / sqrt3, 0.85 / sqrt3, 1.15 / sqrt3)
    v1.advance()
    v9.update(-0.2, 0.8 * phi / sqrt3, 0.8 / (sqrt3 * phi))
    v9.advance()
    c.update()
    c.advance()

    v1.update(1.0 / sqrt3, 0.8 / sqrt3, 1.25 / sqrt3)
    v1.advance()
    v9.update(-0.2, 0.8 * phi / sqrt3, 0.8 / (sqrt3 * phi))
    v9.advance()
    c.update()
    c.advance()

    print('\nprevious state:')
    print(c.toString('prev'))
    print('length:      {:8.5e}'.format(c.length('prev')))
    print('stretch:     {:8.5e}'.format(c.stretch('prev')))
    print('strain:      {:8.5e05}'.format(c.strain('prev')))
    print('strain rate: {:8.5e}'.format(c.dStrain('prev')))
    centroid = c.centroid('prev')
    print('centroid:     ' +
          coordinatesToString(centroid[0], centroid[1], centroid[2]))
    u = c.displacement('prev')
    print('displacement: ' + coordinatesToString(u[0], u[1], u[2]))
    v = c.velocity('prev')
    print('velocity:     ' + coordinatesToString(v[0], v[1], v[2]))
    a = c.acceleration('prev')
    print('acceleration: ' + coordinatesToString(a[0], a[1], a[2]))
    prevP = c.rotation('prev')
    print('rotation:\n' + np.array2string(prevP))
    ident = np.dot(np.transpose(prevP), prevP)
    print('   test for orthongonality: R^t R =\n' + np.array2string(ident))
    print('spin:\n' + np.array2string(c.spin('prev')))

    print('\ncurrent state:')
    print(c.toString('curr'))
    print('length:      {:8.5e}'.format(c.length('curr')))
    print('stretch:     {:8.5e}'.format(c.stretch('curr')))
    print('strain:      {:8.5e}'.format(c.strain('curr')))
    print('strain rate: {:8.5e}'.format(c.dStrain('curr')))
    centroid = c.centroid('curr')
    print('centroid:     ' +
          coordinatesToString(centroid[0], centroid[1], centroid[2]))
    u = c.displacement('curr')
    print('displacement: ' + coordinatesToString(u[0], u[1], u[2]))
    v = c.velocity('curr')
    print('velocity:     ' + coordinatesToString(v[0], v[1], v[2]))
    a = c.acceleration('curr')
    print('acceleration: ' + coordinatesToString(a[0], a[1], a[2]))
    currP = c.rotation('curr')
    print('rotation:\n' + np.array2string(c.rotation('curr')))
    ident = np.dot(np.transpose(currP), currP)
    print('   test for orthongonality: R^t R =\n' + np.array2string(ident))
    print('spin:\n' + np.array2string(c.spin('curr')))

    print('\nnext state:')
    print(c.toString('next'))
    print('length:      {:8.5e}'.format(c.length('next')))
    print('stretch:     {:8.5e}'.format(c.stretch('next')))
    print('strain:      {:8.5e}'.format(c.strain('next')))
    print('strain rate: {:8.5e}'.format(c.dStrain('next')))
    centroid = c.centroid('next')
    print('centroid:     ' +
          coordinatesToString(centroid[0], centroid[1], centroid[2]))
    u = c.displacement('next')
    print('displacement: ' + coordinatesToString(u[0], u[1], u[2]))
    v = c.velocity('next')
    print('velocity:     ' + coordinatesToString(v[0], v[1], v[2]))
    a = c.acceleration('next')
    print('acceleration: ' + coordinatesToString(a[0], a[1], a[2]))
    nextP = c.rotation('next')
    print('rotation:\n' + np.array2string(c.rotation('next')))
    ident = np.dot(np.transpose(nextP), nextP)
    print('   test for orthongonality: R^t R =\n' + np.array2string(ident))
    print('spin:\n' + np.array2string(c.spin('next')))

    print("\nThe kinematic fields associated with this chord are:")
    print("   at Gauss point 1:")
    print("      G = {:8.5e}".format(c.G(1, "curr")))
    print("      F = {:8.5e}".format(c.F(1, "curr")))
    print("      L = {:8.5e}".format(c.L(1, "curr")))
    print("   at Gauss point 2:")
    print("      G = {:8.5e}".format(c.G(2, "curr")))
    print("      F = {:8.5e}".format(c.F(2, "curr")))
    print("      L = {:8.5e}".format(c.L(2, "curr")))
    print("   at Gauss point 3:")
    print("      G = {:8.5e}".format(c.G(3, "curr")))
    print("      F = {:8.5e}".format(c.F(3, "curr")))
    print("      L = {:8.5e}".format(c.L(3, "curr")))

    print("\nThe mass matrix for this chord is:")
    mass = c.massMatrix()
    print(mass)
    print("whose determinant is {:8.5e}".format(np.linalg.det(mass)))
    print("and whose inverse is ")
    print(np.linalg.inv(mass))


    print("\nThe stiffness matrix for this chord is:")
    stiff = c.stiffnessMatrix(1, 1, 1)
    print(stiff)
    

    print("\nThe force vector for this chord is:")
    Force = c.forcingFunction(1, 1)
    print(Force)

    
run()
