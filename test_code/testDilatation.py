#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dodecahedra import dodecahedron
import numpy as np
# for creating graphics
from matplotlib import pyplot as plt
from matplotlib import rc
from pylab import rcParams

"""
Created on Mon Feb 04 2019
Updated on Thr Apr 07 2020

Creates figures that examine the chordal and pentagonal responses during a
dilatation of a regular dodecahedron into a deformed regular dodecahedron.

author: Prof. Alan Freed
"""


def run():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # basic properties for creating the figures

    steps = 150
    gaussPts = 1
    maxDilatation = 1.75

    # dilatation

    strain = np.zeros((steps, 30), dtype=float)
    dilation = np.zeros((steps, 12), dtype=float)
    dilatation = np.zeros(steps, dtype=float)
    F = np.eye(3, dtype=float)
    d = dodecahedron(gaussPts, gaussPts, gaussPts, F)
    for i in range(steps):
        d.advance()
        for j in range(1, 31):
            c = d.getChord(j)
            strain[i, j-1] = c.strain('curr')
        for j in range(1, 13):
            p = d.getPentagon(j)
            dilation[i, j-1] = p.arealStrain('curr')
        for j in range(3):
            F[j, j] += maxDilatation / steps
        dilatation[i] = d.volumetricStrain('curr')
        d.update(F)
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
