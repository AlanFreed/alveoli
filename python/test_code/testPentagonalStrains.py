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
Updated on Fri Oct 28 2020

Creates figures that examine the geometric pentagonal response vs. the thermo-
dynamic pentagonal response during a dilatation of a regular dodecahedron into
a deformed regular dodecahedron.

author: Prof. Alan Freed
"""


def run():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # basic properties for creating the figures

    steps = 150
    gaussPts = 5
    maxDilatation = 1.75

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
    
    # far-field deformation is a dilatation

    dilatation = np.zeros(steps, dtype=float)
    dilation = np.zeros((steps, 12), dtype=float)
    delta = np.zeros((steps, 12), dtype=float)
    epsilon = np.zeros((steps, 12), dtype=float)
    gamma = np.zeros((steps, 12), dtype=float)
    d = dodecahedron(piF0)
    for i in range(steps):
        for j in range(1, 13):
            p = d.getPentagon(j)
            # geometric strain
            dilation[i, j-1] = p.arealStrain('curr')
            # thermodynamic strains
            delta[i, j-1] = p.dilation(gaussPts, 'curr')
            epsilon[i, j-1] = p.squeeze(gaussPts, 'curr')
            gamma[i, j-1] = p.shear(gaussPts, 'curr')
        # the far-field imposed deformation
        for j in range(3):
            piF0[j, j] += maxDilatation / steps
        dilatation[i] = d.volumetricStrain('curr')
        d.update(piF0)
        d.advance(pi)
    # create the arrays for plotting
    dilation1 = np.zeros(steps, dtype=float)
    dilation1[:] = dilation[:, 0]
    delta1 = np.zeros(steps, dtype=float)
    delta1[:] = delta[:, 0]
    epsilon1 = np.zeros(steps, dtype=float)
    epsilon1[:] = epsilon[:, 0]
    gamma1 = np.zeros(steps, dtype=float)
    gamma1[:] = gamma[:, 0]

    # create the response plots

    plt.figure(1)
    rcParams['figure.figsize'] = 12, 5
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=11)

    ax1 = plt.subplot(1, 2, 1)
    # add the curves
    line1, = ax1.plot(dilatation, dilation1, 'k-', linewidth=2)
    plt.title("Geometric Strain", fontsize=20)
    plt.xlabel(r'Far Field  $\Xi = \ln \sqrt[3]{V \! / V_0}$', fontsize=16)
    plt.ylabel(r'Pentagonal Dilation: $\ln \sqrt{A/A_0}$',
               fontsize=16)

    ax2 = plt.subplot(1, 2, 2)
    # add the curves
    line1, = ax2.plot(dilatation, delta1, 'g-', linewidth=2)
    line2, = ax2.plot(dilatation, epsilon1, 'r-', linewidth=2)
    line3, = ax2.plot(dilatation, gamma1, 'b--', linewidth=2,
                      dashes=(5, 5))   # length of 5, spacing of 5
    plt.title("Thermodynamic Strains", fontsize=20)
    plt.legend([line1, line2, line3],
               [r"$\xi$, dilation", r"$\epsilon$, squeeze",
                r"$\gamma$, shear"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field  $\Xi = \ln \sqrt[3]{V \! / V_0}$', fontsize=16)
    plt.ylabel(r'Pentagonal Strains: $\xi$, $\epsilon$, $\gamma$',
               fontsize=16)

    plt.savefig('dilatationGeoVsThermo.jpg')

    plt.show()


run()
