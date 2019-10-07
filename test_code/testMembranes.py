#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from membranes import membrane

"""
Created on Sat Apr 30 2019
Last Updated:  Jul 05 2019

A test file for class membrane in file membranes.py.

author: Prof. Alan Freed
"""


def run():
    np.set_printoptions(precision=5)
    h = 0.1

    # consider uniform dilation
    print('The case of uniform dilation.\n')
    m = membrane(h)
    F1 = np.array([[1.1, 0.0],
                   [0.0, 1.1]])
    F2 = np.array([[1.15, 0.0],
                   [0.0, 1.15]])
    F3 = np.array([[1.175, 0.0],
                   [0.0, 1.175]])
    m.update(F1)
    m.advance()
    m.update(F2)
    m.advance()
    m.update(F3)

    # print out the fields for the current state
    print('The re-indexing matrix is:')
    print(m.Q('current'))
    print('\nThe rotation matrix is:')
    print(m.R('current'))
    print('\nThe Laplace stretch is:')
    print(m.U('current'))
    print('\nThe inverse Laplace strecth is:')
    print(m.UInv('current'))
    print('\nThe Laplace stretch rate is:')
    print(m.dU('current'))
    print('\nThe inverse Laplace stretch rate is:')
    print(m.dUInv('current'))
    print('\nThe spin is:')
    print(m.spin('current'))

    print('\nThe extensive thermodynamic variables are:')
    print('   dilation = {:8.6F}'.format(m.dilation('current')))
    print('   squeeze  = {:8.6F}'.format(m.squeeze('current')))
    print('   shear    = {:8.6F}'.format(m.shear('current')))

    print('\nThe extensive thermodynamic rates are:')
    print('   dDilation = {:8.6F}'.format(m.dDilation('current')))
    print('   dSqueeze  = {:8.6F}'.format(m.dSqueeze('current')))
    print('   dShear    = {:8.6F}'.format(m.dShear('current')))

    # consider squeeze (pure shear)
    print('\nThe case of pure shear.\n')
    m = membrane(h)
    F1 = np.array([[1.1, 0.0],
                   [0.0, 1.0 / 1.1]])
    F2 = np.array([[1.15, 0.0],
                   [0.0, 1.0 / 1.15]])
    F3 = np.array([[1.175, 0.0],
                   [0.0, 1.0 / 1.175]])
    m.update(F1)
    m.advance()
    m.update(F2)
    m.advance()
    m.update(F3)

    # print out the fields for the current state
    print('The re-indexing matrix is:')
    print(m.Q('current'))
    print('\nThe rotation matrix is:')
    print(m.R('current'))
    print('\nThe Laplace stretch is:')
    print(m.U('current'))
    print('\nThe inverse Laplace strecth is:')
    print(m.UInv('current'))
    print('\nThe Laplace stretch rate is:')
    print(m.dU('current'))
    print('\nThe inverse Laplace stretch rate is:')
    print(m.dUInv('current'))
    print('\nThe spin is:')
    print(m.spin('current'))

    print('\nThe extensive thermodynamic variables are:')
    print('   dilation = {:8.6F}'.format(m.dilation('current')))
    print('   squeeze  = {:8.6F}'.format(m.squeeze('current')))
    print('   shear    = {:8.6F}'.format(m.shear('current')))

    print('\nThe extensive thermodynamic rates are:')
    print('   dDilation = {:8.6F}'.format(m.dDilation('current')))
    print('   dSqueeze  = {:8.6F}'.format(m.dSqueeze('current')))
    print('   dShear    = {:8.6F}'.format(m.dShear('current')))

    # consider simple shear
    print('\nThe case of simple shear without re-indexing.\n')
    m = membrane(h)
    F1 = np.array([[1.0, 0.1],
                   [0.0, 1.0]])
    F2 = np.array([[1.0, 0.15],
                   [0.0, 1.0]])
    F3 = np.array([[1.1, 0.175],
                   [0.0, 1.0]])
    m.update(F1)
    m.advance()
    m.update(F2)
    m.advance()
    m.update(F3)

    # print out the fields for the current state
    print('The re-indexing matrix is:')
    print(m.Q('current'))
    print('\nThe rotation matrix is:')
    print(m.R('current'))
    print('\nThe Laplace stretch is:')
    print(m.U('current'))
    print('\nThe inverse Laplace strecth is:')
    print(m.UInv('current'))
    print('\nThe Laplace stretch rate is:')
    print(m.dU('current'))
    print('\nThe inverse Laplace stretch rate is:')
    print(m.dUInv('current'))
    print('\nThe spin is:')
    print(m.spin('current'))

    print('\nThe extensive thermodynamic variables are:')
    print('   dilation = {:8.6F}'.format(m.dilation('current')))
    print('   squeeze  = {:8.6F}'.format(m.squeeze('current')))
    print('   shear    = {:8.6F}'.format(m.shear('current')))

    print('\nThe extensive thermodynamic rates are:')
    print('   dDilation = {:8.6F}'.format(m.dDilation('current')))
    print('   dSqueeze  = {:8.6F}'.format(m.dSqueeze('current')))
    print('   dShear    = {:8.6F}'.format(m.dShear('current')))

    # consider simple shear
    print('\nThe case of simple shear with re-indexing.\n')
    m = membrane(h)
    F1 = np.array([[1.0, 0.0],
                   [0.1, 1.0]])
    F2 = np.array([[1.0, 0.0],
                   [0.15, 1.0]])
    F3 = np.array([[1.1, 0.0],
                   [0.175, 1.0]])
    m.update(F1)
    m.advance()
    m.update(F2)
    m.advance()
    m.update(F3)

    # print out the fields for the current state
    print('The re-indexing matrix is:')
    print(m.Q('current'))
    print('\nThe rotation matrix is:')
    print(m.R('current'))
    print('\nThe Laplace stretch is:')
    print(m.U('current'))
    print('\nThe inverse Laplace strecth is:')
    print(m.UInv('current'))
    print('\nThe Laplace stretch rate is:')
    print(m.dU('current'))
    print('\nThe inverse Laplace stretch rate is:')
    print(m.dUInv('current'))
    print('\nThe spin is:')
    print(m.spin('current'))

    print('\nThe extensive thermodynamic variables are:')
    print('   dilation = {:8.6F}'.format(m.dilation('current')))
    print('   squeeze  = {:8.6F}'.format(m.squeeze('current')))
    print('   shear    = {:8.6F}'.format(m.shear('current')))

    print('\nThe extensive thermodynamic rates are:')
    print('   dDilation = {:8.6F}'.format(m.dDilation('current')))
    print('   dSqueeze  = {:8.6F}'.format(m.dSqueeze('current')))
    print('   dShear    = {:8.6F}'.format(m.dShear('current')))


run()
