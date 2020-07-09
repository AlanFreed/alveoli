#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pivotIncomingF import Pivot

"""
Created on Fri Apr 17 2020
Updated on Mon Jul 06 2020

A test file for class pivot in file pivotIncomingF.py.

author: Prof. Alan Freed
"""


def run():
    F0 = np.eye(3, dtype=float)
    F0[0, 2] = 1.0
    p = Pivot(F0)
    print('\nFor F0 =')
    print(F0)
    print('the pivoted F belongs to case {} and has components'
          .format(p.pivotCase('ref')))
    print(p.pivotedF("ref"))
    print('whose orthogonal matrix is')
    print(p.pivotMatrix('ref'))
    F1 = np.eye(3, dtype=float)
    F1[1, 2] = 1.0
    p.update(F1)
    print('\nFor F1 =')
    print(F1)
    print('its pivoted F belongs to case {} and has components'
          .format(p.pivotCase('next')))
    print(p.pivotedF("next"))
    print('whose orthogonal matrix is')
    print(p.pivotMatrix('next'))
    F2 = np.eye(3, dtype=float)
    F2[0, 1] = 1.0
    p.update(F2)
    p.advance()
    print('\nFor F2 =')
    print(F2)
    print('its pivoted F belongs to case {} and has components'
          .format(p.pivotCase('next')))
    print(p.pivotedF("next"))
    print('whose orthogonal matrix is')
    print(p.pivotMatrix('next'))
    F3 = np.eye(3, dtype=float)
    F3[2, 0] = 1.0
    p.update(F3)
    p.advance()
    print('\nFor F3 =')
    print(F3)
    print('its pivoted F belongs to case {} and has components'
          .format(p.pivotCase('next')))
    print(p.pivotedF("next"))
    print('whose orthogonal matrix is')
    print(p.pivotMatrix('next'))
    F4 = np.eye(3, dtype=float)
    F4[1, 0] = 1.0
    p.update(F4)
    p.advance()
    print('\nFor F4 =')
    print(F4)
    print('its pivoted F belongs to case {} and has components'
          .format(p.pivotCase('next')))
    print(p.pivotedF("next"))
    print('whose orthogonal matrix is')
    print(p.pivotMatrix('next'))
    F5 = np.eye(3, dtype=float)
    F5[2, 1] = 1.0
    p.update(F5)
    p.advance()
    print('\nFor F5 =')
    print(F5)
    print('its pivoted F belongs to case {} and has components'
          .format(p.pivotCase('next')))
    print(p.pivotedF("next"))
    print('whose orthogonal matrix is')
    print(p.pivotMatrix('next'))
    F6 = np.eye(3, dtype=float)
    F6[0, 1] = 1.0
    F6[1, 0] = 1.0
    p.update(F6)
    p.advance()
    print('\nFor F6 =')
    print(F6)
    print('its pivoted F belongs to case {} and has components'
          .format(p.pivotCase('next')))
    print(p.pivotedF("next"))
    print('whose orthogonal matrix is')
    print(p.pivotMatrix('next'))
    F7 = np.eye(3, dtype=float)
    F7[1, 0] = 1.0
    F7[2, 0] = 1.0
    F7[0, 1] = 1.0
    p.update(F7)
    p.advance()
    print('\nFor F7 =')
    print(F7)
    print('its pivoted F belongs to case {} and has components'
          .format(p.pivotCase('next')))
    print(p.pivotedF("next"))
    print('whose orthogonal matrix is')
    print(p.pivotMatrix('next'))

    print('\nTest vector and tensor transformations')
    print('given the orthogonal matrix')
    print(p.pivotMatrix('prev'))
    print('\nTesting vectors that were created as arrays:\n')
    vec1 = np.zeros(3, dtype=float)
    vec1[0] = 1.0
    vec1[1] = 2.0
    vec1[2] = 3.0
    vec = p.globalToLocalVector(vec1, 'prev')
    print('For array')
    print(vec1)
    print("it maps from global to local as")
    print(vec)
    print('with its reverse map giving')
    vec = p.localToGlobalVector(vec, 'prev')
    print(vec)
    print('\nTesting vectors that were created as a colmun matrix:\n')
    vec2 = np.zeros((3, 1), dtype=float)
    vec2[0, 0] = 1.0
    vec2[1, 0] = 2.0
    vec2[2, 0] = 3.0
    vec = p.globalToLocalVector(vec2, 'prev')
    print('For column vector')
    print(vec2)
    print("it maps from global to local as")
    print(vec)
    print('with its reverse map giving')
    vec = p.localToGlobalVector(vec, 'prev')
    print(vec)
    print('\nTesting matrices:\n')
    mtx1 = np.zeros((3, 3), dtype=float)
    mtx1[0, 0] = 1.0
    mtx1[0, 1] = 2.0
    mtx1[0, 2] = 3.0
    mtx1[1, 0] = 4.0
    mtx1[1, 1] = 5.0
    mtx1[1, 2] = 6.0
    mtx1[2, 0] = 7.0
    mtx1[2, 1] = 8.0
    mtx1[2, 2] = 9.0
    mtx = p.globalToLocalTensor(mtx1, 'prev')
    print('For matrix')
    print(mtx1)
    print('it maps from global to local as')
    print(mtx)
    print('with a reverse mapping of')
    mtx = p.localToGlobalTensor(mtx, 'prev')
    print(mtx)


run()
