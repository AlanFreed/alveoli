#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dodecahedra import dodecahedron
from math import log
import numpy as np
# for creating graphics
from matplotlib import pyplot as plt
from matplotlib import rc
from pylab import rcParams
from pivotIncomingF import Pivot

"""
Created on Mon Feb 04 2019
Updated on Fri Oct 28 2020

Creates figures that examine the thermodynamic strain response of the pentagons
on a dodecahedron that is subjected to squeeze.

author: Prof. Alan Freed
"""


def run():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # basic properties for creating the figures

    steps = 150
    gaussPt = 5
    maxSqueeze = 3.5

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
 
    # a far-field deformation of pure shear

    pureShear = np.zeros(steps, dtype=float)
    delta = np.zeros((steps, 12), dtype=float)
    epsilon = np.zeros((steps, 12), dtype=float)
    gamma = np.zeros((steps, 12), dtype=float)
    d = dodecahedron(piF0)
    for i in range(steps):
        for j in range(1, 13):
            p = d.getPentagon(j)
            # thermodynamic strains
            delta[i, j-1] = p.dilation(gaussPt, 'curr')
            epsilon[i, j-1] = p.squeeze(gaussPt, 'curr')
            gamma[i, j-1] = p.shear(gaussPt, 'curr')
        # far-field pure shear
        piF0[1, 1] += maxSqueeze / steps
        piF0[2, 2] = 1.0 / piF0[1, 1]
        pureShear[i] = log((piF0[1, 1] / piF0[2, 2])**(1.0/3.0))
        d.update(piF0)
        d.advance(pi)
    # create the plotting arrays
    delta1 = np.zeros(steps, dtype=float)
    delta1[:] = delta[:, 0]
    delta2 = np.zeros(steps, dtype=float)
    delta2[:] = delta[:, 1]
    delta3 = np.zeros(steps, dtype=float)
    delta3[:] = delta[:, 4]

    epsilon1 = np.zeros(steps, dtype=float)
    epsilon1[:] = epsilon[:, 0]
    epsilon2 = np.zeros(steps, dtype=float)
    epsilon2[:] = epsilon[:, 1]
    epsilon3 = np.zeros(steps, dtype=float)
    epsilon3[:] = epsilon[:, 4]

    gamma1 = np.zeros(steps, dtype=float)
    gamma1[:] = gamma[:, 0]
    gamma2 = np.zeros(steps, dtype=float)
    gamma2[:] = gamma[:, 1]
    gamma3 = np.zeros(steps, dtype=float)
    gamma3[:] = gamma[:, 4]

    # create the response plots: Figure 1 is for dilatation

    plt.figure(1)
    rcParams['figure.figsize'] = 18, 5
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=11)

    ax1 = plt.subplot(1, 3, 1)
    # add the curves
    line1, = ax1.plot(pureShear, delta1, 'k-', linewidth=2)
    line2, = ax1.plot(pureShear, delta2, 'b-', linewidth=2)
    line3, = ax1.plot(pureShear, delta3, 'r-', linewidth=2)

    plt.title("Dodecahedral", fontsize=20)
    plt.legend([line1, line2, line3],
               ["pentagon 1", "pentagon 2", "pentagon 5"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far-Field $\ln \sqrt{b/c}$', fontsize=16)
    plt.ylabel(r'At Centroid $\xi = \ln \sqrt{uv}$', fontsize=16)

    ax2 = plt.subplot(1, 3, 2)
    # add the curves
    line1, = ax2.plot(pureShear, epsilon1, 'k-', linewidth=2)
    line2, = ax2.plot(pureShear, epsilon2, 'b-', linewidth=2)
    line3, = ax2.plot(pureShear, epsilon3, 'r-', linewidth=2)

    plt.title("Pure Shear", fontsize=20)
    plt.legend([line1, line2, line3],
               ["pentagon 1", "pentagon 2", "pentagon 5"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far-Field $\ln \sqrt{b/c}$', fontsize=16)
    plt.ylabel(r'At Centroid $\epsilon = \ln \sqrt{u/v}$',
               fontsize=16)

    ax3 = plt.subplot(1, 3, 3)
    # add the curves
    line1, = ax3.plot(pureShear, gamma1, 'k-', linewidth=2)
    line2, = ax3.plot(pureShear, gamma2, 'b-', linewidth=2)
    line3, = ax3.plot(pureShear, gamma3, 'r-', linewidth=2)

    plt.title(r"Directions 2 \& 3", fontsize=20)
    plt.legend([line1, line2, line3],
               ["pentagon 1", "pentagon 2", "pentagon 5"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far-Field $\ln \sqrt{b/c}$', fontsize=16)
    plt.ylabel(r'At Centroid $\gamma = g$',
               fontsize=16)

    plt.savefig('pentagonalPureShear23.jpg')

    plt.show()


run()
