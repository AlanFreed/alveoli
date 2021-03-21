#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math as m
import numpy as np
from shapeFnPentagons import ShapeFunction

"""
Created on Sat Apr 27 2019
Updated on Tue Jul 07 2020

A test file for class shapeFunction in file shapeFnPentagons.py.

author: Prof. Alan Freed
"""


def run():
    np.set_printoptions(precision=5)
    print('\nThis test case considers shape functions.')

    # shape functions at the centroid

    v0Xi = m.cos(0.0)
    v0Eta = m.sin(0.0)
    v0 = (v0Xi, v0Eta)
    sfCen = ShapeFunction(v0)

    # shape functions at the five vertices of a pentagon

    v1Xi = m.cos(m.pi / 2.0)
    v1Eta = m.sin(m.pi / 2.0)
    v1 = (v1Xi, v1Eta)
    sfV1 = ShapeFunction(v1)

    v2Xi = m.cos(9.0 * m.pi / 10.0)
    v2Eta = m.sin(9.0 * m.pi / 10.0)
    v2 = (v2Xi, v2Eta)
    sfV2 = ShapeFunction(v2)

    v3Xi = m.cos(13.0 * m.pi / 10.0)
    v3Eta = m.sin(13.0 * m.pi / 10.0)
    v3 = (v3Xi, v3Eta)
    sfV3 = ShapeFunction(v3)

    v4Xi = m.cos(17.0 * m.pi / 10.0)
    v4Eta = m.sin(17.0 * m.pi / 10.0)
    v4 = (v4Xi, v4Eta)
    sfV4 = ShapeFunction(v4)

    v5Xi = m.cos(1.0 * m.pi / 10.0)
    v5Eta = m.sin(1.0 * m.pi / 10.0)
    v5 = (v5Xi, v5Eta)
    sfV5 = ShapeFunction(v5)

    print('\nShape functions for Vertex 1 = ({:8.6e}, {:8.6e}):'
          .format(v1Xi, v1Eta))
    print('   N1 = {:8.6e}'.format(sfV1.N1))
    print('   N2 = {:8.6e}'.format(sfV1.N2))
    print('   N3 = {:8.6e}'.format(sfV1.N3))
    print('   N4 = {:8.6e}'.format(sfV1.N4))
    print('   N5 = {:8.6e}'.format(sfV1.N5))

    print('\nShape functions for Vertex 2 = ({:8.6e}, {:8.6e}):'
          .format(v2Xi, v2Eta))
    print('   N1 = {:8.6e}'.format(sfV2.N1))
    print('   N2 = {:8.6e}'.format(sfV2.N2))
    print('   N3 = {:8.6e}'.format(sfV2.N3))
    print('   N4 = {:8.6e}'.format(sfV2.N4))
    print('   N5 = {:8.6e}'.format(sfV2.N5))

    print('\nShape functions for Vertex 3 = ({:8.6e}, {:8.6e}):'
          .format(v3Xi, v3Eta))
    print('   N1 = {:8.6e}'.format(sfV3.N1))
    print('   N2 = {:8.6e}'.format(sfV3.N2))
    print('   N3 = {:8.6e}'.format(sfV3.N3))
    print('   N4 = {:8.6e}'.format(sfV3.N4))
    print('   N5 = {:8.6e}'.format(sfV3.N5))

    print('\nShape functions for Vertex 4 = ({:8.6e}, {:8.6e}):'
          .format(v4Xi, v4Eta))
    print('   N1 = {:8.6e}'.format(sfV4.N1))
    print('   N2 = {:8.6e}'.format(sfV4.N2))
    print('   N3 = {:8.6e}'.format(sfV4.N3))
    print('   N4 = {:8.6e}'.format(sfV4.N4))
    print('   N5 = {:8.6e}'.format(sfV4.N5))

    print('\nShape functions for Vertex 5 = ({:8.6e}, {:8.6e}):'
          .format(v5Xi, v5Eta))
    print('   N1 = {:8.6e}'.format(sfV5.N1))
    print('   N2 = {:8.6e}'.format(sfV5.N2))
    print('   N3 = {:8.6e}'.format(sfV5.N3))
    print('   N4 = {:8.6e}'.format(sfV5.N4))
    print('   N5 = {:8.6e}'.format(sfV5.N5))

    print('\nTest partition of unity.')

    addition = sfCen.N1 + sfCen.N2 + sfCen.N3 + sfCen.N4 + sfCen.N5
    print('\n   At centroid, sum of shape functions = {:8.6e}'
          .format(addition))

    addition = sfV1.N1 + sfV1.N2 + sfV1.N3 + sfV1.N4 + sfV1.N5
    print('   For node 1,  sum of shape functions = {:8.6e}'.format(addition))
    addition = sfV2.N1 + sfV2.N2 + sfV2.N3 + sfV2.N4 + sfV2.N5
    print('   For node 2,  sum of shape functions = {:8.6e}'.format(addition))
    addition = sfV3.N1 + sfV3.N2 + sfV3.N3 + sfV3.N4 + sfV3.N5
    print('   For node 3,  sum of shape functions = {:8.6e}'.format(addition))
    addition = sfV4.N1 + sfV4.N2 + sfV4.N3 + sfV4.N4 + sfV4.N5
    print('   For node 4,  sum of shape functions = {:8.6e}'.format(addition))
    addition = sfV5.N1 + sfV5.N2 + sfV5.N3 + sfV5.N4 + sfV5.N5
    print('   For node 5,  sum of shape functions = {:8.6e}'.format(addition))

    print('\nTest for interpolating nodal data.\n')

    print('   N1 * eta1 should be 1; it is {:8.6e}.'.format(sfV1.N1 * v1Eta))
    print('   N2 * eta1 should be 0; it is {:8.6e}.'.format(sfV1.N2 * v1Eta))
    print('   N3 * eta1 should be 0; it is {:8.6e}.'.format(sfV1.N3 * v1Eta))
    print('   N4 * eta1 should be 0; it is {:8.6e}.'.format(sfV1.N4 * v1Eta))
    print('   N5 * eta1 should be 0; it is {:8.6e}.'.format(sfV1.N5 * v1Eta))

    print('\nTest interpolation.')

    print('\nInterpolate on xi values:')
    xi = sfV1.interpolate(v1Xi, v2Xi, v3Xi, v4Xi, v5Xi)
    print('  At vertex 1: should be {:8.6e}; it is {:8.6e}'.format(v1Xi, xi))
    xi = sfV2.interpolate(v1Xi, v2Xi, v3Xi, v4Xi, v5Xi)
    print('  At vertex 2: should be {:8.6e}; it is {:8.6e}'.format(v2Xi, xi))
    xi = sfV3.interpolate(v1Xi, v2Xi, v3Xi, v4Xi, v5Xi)
    print('  At vertex 3: should be {:8.6e}; it is {:8.6e}'.format(v3Xi, xi))
    xi = sfV4.interpolate(v1Xi, v2Xi, v3Xi, v4Xi, v5Xi)
    print('  At vertex 4: should be {:8.6e}; it is {:8.6e}'.format(v4Xi, xi))
    xi = sfV5.interpolate(v1Xi, v2Xi, v3Xi, v4Xi, v5Xi)
    print('  At vertex 5: should be {:8.6e}; it is {:8.6e}'.format(v5Xi, xi))

    print('\nInterpolate on eta values:')
    eta = sfV1.interpolate(v1Eta, v2Eta, v3Eta, v4Eta, v5Eta)
    print('  At vertex 1: should be {:8.6e}; it is {:8.6e}'.format(v1Eta, eta))
    eta = sfV2.interpolate(v1Eta, v2Eta, v3Eta, v4Eta, v5Eta)
    print('  At vertex 2: should be {:8.6e}; it is {:8.6e}'.format(v2Eta, eta))
    eta = sfV3.interpolate(v1Eta, v2Eta, v3Eta, v4Eta, v5Eta)
    print('  At vertex 3: should be {:8.6e}; it is {:8.6e}'.format(v3Eta, eta))
    eta = sfV4.interpolate(v1Eta, v2Eta, v3Eta, v4Eta, v5Eta)
    print('  At vertex 4: should be {:8.6e}; it is {:8.6e}'.format(v4Eta, eta))
    eta = sfV5.interpolate(v1Eta, v2Eta, v3Eta, v4Eta, v5Eta)
    print('  At vertex 5: should be {:8.6e}; it is {:8.6e}'.format(v5Eta, eta))

    print('\nTest the determinant of the Jacobian.\n')
    det = sfV1.jacobianDeterminant(v1, v2, v3, v4, v5)
    print('The determinant of the Jacobian of a regular pentagon')
    print('should be 1; it is {:8.6e}.'.format(det))


run()
