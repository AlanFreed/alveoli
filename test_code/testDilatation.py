#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dodecahedra import dodecahedron
import numpy as np
# for creating graphics
from matplotlib import pyplot as plt
from matplotlib import rc
from pylab import rcParams
from pivotIncomingF import Pivot


"""
Created on Mon Feb 04 2019
Updated on Thr Oct 28 2020

Creates figures that examine the chordal and pentagonal responses during a
dilatation of a regular dodecahedron into a deformed regular dodecahedron.

author: Prof. Alan Freed
"""


def run():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

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

    piF0 = pi.pivotedF('ref')
    
    # basic properties for creating the figures

    steps = 150
    maxDilatation = 1.75

    # dilatation

    strain = np.zeros((steps, 30), dtype=float)
    dilation = np.zeros((steps, 12), dtype=float)
    dilatation = np.zeros(steps, dtype=float)
    d = dodecahedron(piF0)
    for i in range(steps):
        d.advance(pi)
        for j in range(1, 31):
            c = d.getChord(j)
            strain[i, j-1] = c.strain('curr')
        for j in range(1, 13):
            p = d.getPentagon(j)
            dilation[i, j-1] = p.arealStrain('curr')
        for j in range(3):
            piF0[j, j] += maxDilatation / steps
        dilatation[i] = d.volumetricStrain('curr')
        d.update(piF0)
    strain1 = np.zeros(steps, dtype=float)
    strain1[:] = strain[:, 0]
    dilation1 = np.zeros(steps, dtype=float)
    dilation1[:] = dilation[:, 0]

    # create the response plots: Figure 1 is for dilatation

    plt.figure(1)
    rcParams['figure.figsize'] = 12, 5
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=11)

    ax1 = plt.subplot(1, 2, 1)
    # add the curves
    line1, = ax1.plot(dilatation, strain1, 'k-', linewidth=2)
    plt.title("Dodecahedral", fontsize=20)
    plt.xlabel(r'Far Field  $\Xi = \ln \sqrt[3]{V \! / V_0}$', fontsize=16)
    plt.ylabel(r'Chordal Strain: $e = \ln (L/L_0)$', fontsize=16)

    ax2 = plt.subplot(1, 2, 2)
    # add the curves
    line1, = ax2.plot(dilatation, dilation1, 'k-', linewidth=2)
    plt.title("Dilatation", fontsize=20)
    plt.xlabel(r'Far Field  $\Xi = \ln \sqrt[3]{V \! / V_0}$', fontsize=16)
    plt.ylabel(r'Pentagonal Dilation: $\xi = \ln \sqrt{A/A_0}$',
               fontsize=16)
    plt.savefig('dilatation.jpg')

    plt.show()


run()









