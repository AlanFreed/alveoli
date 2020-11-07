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

Creates figures that examine the thermodynamic strain response of the pentagons
on a dodecahedron that is subjected to simple shear.

@author: Prof. Alan Freed
"""


def run():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # basic properties for creating the figures


    steps = 150
    gaussPt = 5
    maxShear = 1.0

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
    
    
    
    # a far-field deformation of simple shear

    shear = np.zeros(steps, dtype=float)
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
        # far-field simple shear motion
        piF0[0, 2] += maxShear / steps
        shear[i] = piF0[0, 2]
        d.update(piF0)
        d.advance(pi)
    # create the plotting arrays
    delta1 = np.zeros(steps, dtype=float)
    delta1[:] = delta[:, 0]
    delta2 = np.zeros(steps, dtype=float)
    delta2[:] = delta[:, 1]
    delta3 = np.zeros(steps, dtype=float)
    delta3[:] = delta[:, 4]
    delta4 = np.zeros(steps, dtype=float)
    delta4[:] = delta[:, 5]

    epsilon1 = np.zeros(steps, dtype=float)
    epsilon1[:] = epsilon[:, 0]
    epsilon2 = np.zeros(steps, dtype=float)
    epsilon2[:] = epsilon[:, 1]
    epsilon3 = np.zeros(steps, dtype=float)
    epsilon3[:] = epsilon[:, 4]
    epsilon4 = np.zeros(steps, dtype=float)
    epsilon4[:] = epsilon[:, 5]

    epsilon5 = np.zeros(steps, dtype=float)
    epsilon5[:] = epsilon[:, 11]

    epsilon6 = np.zeros(steps, dtype=float)
    epsilon6[:] = epsilon[:, 7]
    epsilon7 = np.zeros(steps, dtype=float)
    epsilon7[:] = epsilon[:, 8]
    epsilon8 = np.zeros(steps, dtype=float)
    epsilon8[:] = epsilon[:, 10]

    gamma1 = np.zeros(steps, dtype=float)
    gamma1[:] = gamma[:, 0]
    gamma2 = np.zeros(steps, dtype=float)
    gamma2[:] = gamma[:, 1]
    gamma3 = np.zeros(steps, dtype=float)
    gamma3[:] = gamma[:, 4]
    gamma4 = np.zeros(steps, dtype=float)
    gamma4[:] = gamma[:, 8]
    gamma5 = np.zeros(steps, dtype=float)
    gamma5[:] = gamma[:, 10]

    # create the response plots: Figure 1 is for dilatation

    plt.figure(1)
    rcParams['figure.figsize'] = 18, 5
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=11)

    ax1 = plt.subplot(1, 3, 1)
    # add the curves
    line1, = ax1.plot(shear, delta1, 'k-', linewidth=2)
    line2, = ax1.plot(shear, delta2, 'b-', linewidth=2)
    line3, = ax1.plot(shear, delta3, 'r-', linewidth=2)
    line4, = ax1.plot(shear, delta4, 'g-', linewidth=2)

    plt.title("Dodecahedral", fontsize=20)
    plt.legend([line1, line2, line3, line4],
               ["pentagon 1", "pentagon 2", "pentagon 5", "pentagon 6"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far-Field $\gamma_{13}$', fontsize=16)
    plt.ylabel(r'At Centroid $\xi = \ln \sqrt{uv}$', fontsize=16)

    ax2 = plt.subplot(1, 3, 2)
    # add the curves
    line1, = ax2.plot(shear, epsilon1, 'k-', linewidth=2)
    line2, = ax2.plot(shear, epsilon2, 'b-', linewidth=2)
    line3, = ax2.plot(shear, epsilon3, 'r-', linewidth=2)
    line4, = ax2.plot(shear, epsilon4, 'g-', linewidth=2)

    plt.title("Simple Shear", fontsize=20)
    plt.legend([line1, line2, line3, line4],
               ["pentagon 1", "pentagon 2", "pentagon 5", "pentagon 6"],
               bbox_to_anchor=(0.025, 0.4), loc=2, fontsize=14)
    plt.xlabel(r'Far-Field $\gamma_{13}$', fontsize=16)
    plt.ylabel(r'At Centroid $\epsilon = \ln \sqrt{u/v}$', fontsize=16)

    ax3 = plt.subplot(1, 3, 3)
    # add the curves
    line1, = ax3.plot(shear, gamma1, 'k-', linewidth=2)
    line2, = ax3.plot(shear, gamma2, 'b-', linewidth=2)
    line3, = ax3.plot(shear, gamma3, 'r-', linewidth=2)
    line4, = ax3.plot(shear, gamma4, 'g-', linewidth=2)
    line5, = ax3.plot(shear, gamma5, 'm-', linewidth=2)

    plt.title(r"in the 1-3 Plane", fontsize=20)
    plt.legend([line1, line2, line3, line4, line5],
               ["pentagon 1", "pentagon 2", "pentagon 5", "pentagon 9",
                "pentagon 11"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far-Field $\gamma_{13}$', fontsize=16)
    plt.ylabel(r'At Centroid $\gamma = g$',
               fontsize=16)

    plt.savefig('pentagonalSimpleShear13.jpg')

    plt.show()


run()
