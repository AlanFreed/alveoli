#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dodecahedra import dodecahedron
import math
import numpy as np
from pivotIncomingF import Pivot


"""
Created on Wed Jan 23 2019
Updated on Fri Oct 11 2019

A test file for class dodecahedron in file dodecahedra.py.

author: Prof. Alan Freed
"""


def pause():
    input("Press the <ENTER> key to continue...")


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

    # re-index the co-ordinate systems according to pivot in pivotIncomingF.py
    pi = Pivot(F0)
    pi.update(F1)
    pi.advance()
    pi.update(F2)
    pi.advance()
    pi.update(F3)
    # get this histories re-indexed deformation gradients
    piF0 = pi.pivotedF('ref')
    piF1 = pi.pivotedF('prev')
    piF2 = pi.pivotedF('curr')
    piF3 = pi.pivotedF('next')
    
    
    
    d = dodecahedron(piF0)
    # verify the dodecahedron nodal/chordal assignment by printing them out
    print('\nA normalized dodecahedron has vertices:\n')
    print(d.verticesToString('ref'))
    pause()
    print('\nIt has chords:\n')
    print(d.chordsToString('ref'))
    pause()
    print('\nIt has pentagons:\n')
    print(d.pentagonsToString('ref'))
    pause()
    print('\nAnd it has tetrahedra:\n')
    print(d.tetrahedraToString('ref'))
    pause()
    # verify the chordal lengths
    print('\n')
    for i in range(1, 31):
        c = d.getChord(i)
        if i < 10:
            print('Length of chord 0{} is {:8.6F}'.format(i, c.length()))
        else:
            print('Length of chord {} is {:8.6F}'.format(i, c.length()))
    # verify the pentagonal areas
    print('')
    for i in range(1, 13):
        p = d.getPentagon(i)
        if i < 10:
            print('Area of pentagon 0{} is {:8.6F}'.format(i, p.area('ref')))
        else:
            print('Area of pentagon {} is {:8.6F}'.format(i, p.area('ref')))
    # verify the computations for volume
    d.update(piF3)
    d.advance(pi)
    # physical properties of a dodecahedron
    omega = 54 * math.pi / 180.0
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    # volume of a regular dodecahedron
    vol = (40.0 / (3.0 * math.sqrt(3.0) * phi**3) * (math.tan(omega))**2 *
           math.sin(omega))
    print('\nVolume of the dodecahedron is {:8.6F} and should be {:8.6F}'
          .format(d.volume('ref'), vol))
    print('V0 / Vn is {:8.6F} '
          .format(d.V0V('n')))
    print('V0 / Vc is {:8.6F} '
          .format(d.V0V('c')))
    print('V0 / Vp is {:8.6F} '
          .format(d.V0V('p')))
    print('V0 / Vr is {:8.6F} '
          .format(d.V0V('r')))
run()
