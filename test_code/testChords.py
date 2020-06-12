#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import chord
import math as m
import numpy as np
from pivotIncomingF import pivot
from vertices import coordinatesToString
from vertices import vertex

"""
Created on Mon Jan 28 2019
Updated on Mon May 04 2020

A test file for class chord in file chords.py.

author: Prof. Alan Freed
"""


def run():
    print('\nThis test case considers a deforming chord.')
    print('Be patient, the constitutive equations are being integrated.\n')

    # time step size used for numeric integration
    dTime = 1.0e-3

    # degree of polynomial to be integrated exactly (can be 1, 3 or 5)
    degree = 5

    # tolerance for controlling local trunction error during integration
    tol = 0.001

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

    # re-index the co-ordinate systems according to pivot in pivotIncomingF.py
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

    # quantities used when assigning a vertice's natrual co-ordinates
    sqrt3 = m.sqrt(3.0)
    # phi is the golden ratio which appears frequently in dodecahedral geometry
    phi = (1.0 + m.sqrt(5.0)) / 2.0
    # co-ordinates for vertex 1 in its natural co-ordinate frame of reference
    v1_0 = np.array([1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3])
    # co-ordinates for vertex 9 in its natural co-ordinate frame of reference
    v9_0 = np.array([0.0, phi / sqrt3, 1.0 / (sqrt3 * phi)])

    # create the two vertices of a chord and then create the chord
    pv1_0 = np.matmul(pF0, v1_0)
    v1 = vertex(1, (pv1_0[0], pv1_0[1], pv1_0[2]), dTime)
    pv9_0 = np.matmul(pF0, v9_0)
    v9 = vertex(9, (pv9_0[0], pv9_0[1], pv9_0[2]), dTime)
    c = chord(2, v1, v9, dTime, degree, tol)

    # update and advance these three objects for the deformation history given
    pv1_1 = np.matmul(pF1, v1_0)
    v1.update((pv1_1[0], pv1_1[1], pv1_1[2]))
    pv9_1 = np.matmul(pF1, v9_0)
    v9.update((pv9_1[0], pv9_1[1], pv9_1[2]))
    c.update(p)
    v1.advance()
    v9.advance()
    c.advance()
    pv1_2 = np.matmul(pF2, v1_0)
    v1.update((pv1_2[0], pv1_2[1], pv1_2[2]))
    pv9_2 = np.matmul(pF2, v9_0)
    v9.update((pv9_2[0], pv9_2[1], pv9_2[2]))
    c.update(p)
    v1.advance()
    v9.advance()
    c.advance()
    pv1_3 = np.matmul(pF3, v1_0)
    v1.update((pv1_3[0], pv1_3[1], pv1_3[2]))
    pv9_3 = np.matmul(pF3, v9_0)
    v9.update((pv9_3[0], pv9_3[1], pv9_3[2]))
    c.update()

    # get this histories re-indexed deformation gradients
    pF0 = p.pivotedF('ref')
    pF1 = p.pivotedF('prev')
    pF2 = p.pivotedF('curr')
    pF3 = p.pivotedF('next')

    print("The chord, when printed via print(chord), is:")
    print(c)
    print("whose initial and current lengths are {:6.4f} and {:6.4f}."
          .format(c.length('ref'), c.length('curr')))

    # print out the material properties of this chord
    print("\nThis chord has the following material properties:")
    print(" chord mass density   = {:8.5e} (gr/cm^3)".format(c.massDensity()))
    print(" chord cross-section  = {:8.5e} (cm^2)".format(c.area('r')))
    print(" collagen has an area = {:8.5e} (cm^2)".format(c.areaCollagen('r')))
    print(" elastin  has an area = {:8.5e} (cm^2)".format(c.areaElastin('r')))

    print('\nIn the reference state:')
    print(c.toString('ref'))
    print('length:   {:8.5e}'.format(c.length('ref')))
    centroid = c.centroid('ref')
    print('centroid: ' +
          coordinatesToString((centroid[0], centroid[1], centroid[2])))

    print('\nprevious state:')
    print(c.toString('prev'))
    print('length:      {:8.5e}'.format(c.length('prev')))
    print('stretch:     {:8.5e}'.format(c.stretch('prev')))
    print('strain:      {:8.5e}'.format(c.strain('prev')))
    print('strain rate: {:8.5e}'.format(c.dStrain('prev')))
    centroid = c.centroid('prev')
    print('centroid:     ' +
          coordinatesToString((centroid[0], centroid[1], centroid[2])))
    u = c.displacement('prev')
    print('displacement: ' + coordinatesToString((u[0], u[1], u[2])))
    v = c.velocity('prev')
    print('velocity:     ' + coordinatesToString((v[0], v[1], v[2])))
    a = c.acceleration('prev')
    print('acceleration: ' + coordinatesToString((a[0], a[1], a[2])))
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
          coordinatesToString((centroid[0], centroid[1], centroid[2])))
    u = c.displacement('curr')
    print('displacement: ' + coordinatesToString((u[0], u[1], u[2])))
    v = c.velocity('curr')
    print('velocity:     ' + coordinatesToString((v[0], v[1], v[2])))
    a = c.acceleration('curr')
    print('acceleration: ' + coordinatesToString((a[0], a[1], a[2])))
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
          coordinatesToString((centroid[0], centroid[1], centroid[2])))
    u = c.displacement('next')
    print('displacement: ' + coordinatesToString((u[0], u[1], u[2])))
    v = c.velocity('next')
    print('velocity:     ' + coordinatesToString((v[0], v[1], v[2])))
    a = c.acceleration('next')
    print('acceleration: ' + coordinatesToString((a[0], a[1], a[2])))
    nextP = c.rotation('next')
    print('rotation:\n' + np.array2string(c.rotation('next')))
    ident = np.dot(np.transpose(nextP), nextP)
    print('   test for orthongonality: R^t R =\n' + np.array2string(ident))
    print('spin:\n' + np.array2string(c.spin('next')))

    print("\nThe kinematic fields associated with this chord are:")
    print("   at Gauss point 1:")
    atGaussPt = 1
    print("      G = ")
    G = c.G(atGaussPt, 'curr')
    print(G)
    print("      F = ")
    F = c.F(atGaussPt, 'curr')
    print(F)
    print("      L = ")
    L = c.L(atGaussPt, 'curr')
    print(L)
    print("   at Gauss point 2:")
    if degree == 3 or degree == 5:
        atGaussPt = 2
        print("      G = ")
        G = c.G(atGaussPt, 'curr')
        print(G)
        print("      F = ")
        F = c.F(atGaussPt, 'curr')
        print(F)
        print("      L = ")
        L = c.L(atGaussPt, 'curr')
        print(L)
    if degree == 5:
        print("   at Gauss point 3:")
        atGaussPt = 3
        print("      G = ")
        G = c.G(atGaussPt, 'curr')
        print(G)
        print("      F = ")
        F = c.F(atGaussPt, 'curr')
        print(F)
        print("      L = ")
        L = c.L(atGaussPt, 'curr')
        print(L)

    mass = c.massMatrix()
    print("\nThe mass matrix for this chord is:")
    print(mass)
    print("whose determinant is {:8.5e}".format(np.linalg.det(mass)))
    print("and whose inverse is ")
    print(np.linalg.inv(mass))

    """
    M = 1
    se = 1
    sc = 1

    print("\nThe stiffness matrix for this chord is:")
    stiff = c.stiffnessMatrix(M, se, sc)
    print(stiff)

    print("\nThe force vector for this chord is:")
    force = c.forcingFunction(se, sc)
    print(force)
    """


run()
