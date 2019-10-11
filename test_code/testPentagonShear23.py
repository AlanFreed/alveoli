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
Updated on Fri Oct 11 2019

Creates figures that examine the thermodynamic strain response of the pentagons
on a dodecahedron that is subjected to simple shear.

author: Prof. Alan Freed
"""


def run():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # basic properties for creating the figures

    steps = 150
    gaussPt = 1
    gaussPts = 1
    maxShear = 1.0

    # a far-field deformation of simple shear

    shear = np.zeros(steps, dtype=float)
    delta = np.zeros((steps, 12), dtype=float)
    epsilon = np.zeros((steps, 12), dtype=float)
    gamma = np.zeros((steps, 12), dtype=float)
    F = np.eye(3, dtype=float)
    d = dodecahedron(gaussPts, gaussPts, gaussPts, F)
    for i in range(steps):
        for j in range(1, 13):
            p = d.getPentagon(j)
            # thermodynamic strains
            delta[i, j-1] = p.dilation(gaussPt, 'curr')
            epsilon[i, j-1] = p.squeeze(gaussPt, 'curr')
            gamma[i, j-1] = p.shear(gaussPt, 'curr')
        # far-field simple shear
        F[1, 2] += maxShear / steps
        shear[i] = F[1, 2]
        d.update(F)
        d.advance()
    # create the plotting arrays
    delta1 = np.zeros(steps, dtype=float)
    delta1[:] = delta[:, 0]
    delta2 = np.zeros(steps, dtype=float)
    delta2[:] = delta[:, 1]
    delta3 = np.zeros(steps, dtype=float)
    delta3[:] = delta[:, 3]
    delta4 = np.zeros(steps, dtype=float)
    delta4[:] = delta[:, 4]

    epsilon1 = np.zeros(steps, dtype=float)
    epsilon1[:] = epsilon[:, 0]
    epsilon2 = np.zeros(steps, dtype=float)
    epsilon2[:] = epsilon[:, 1]
    epsilon3 = np.zeros(steps, dtype=float)
    epsilon3[:] = epsilon[:, 3]
    epsilon4 = np.zeros(steps, dtype=float)
    epsilon4[:] = epsilon[:, 4]

    gamma1 = np.zeros(steps, dtype=float)
    gamma1[:] = gamma[:, 0]
    gamma2 = np.zeros(steps, dtype=float)
    gamma2[:] = gamma[:, 1]
    gamma3 = np.zeros(steps, dtype=float)
    gamma3[:] = gamma[:, 2]
    gamma4 = np.zeros(steps, dtype=float)
    gamma4[:] = gamma[:, 4]
    gamma5 = np.zeros(steps, dtype=float)
    gamma5[:] = gamma[:, 6]

    gamma6 = np.zeros(steps, dtype=float)
    gamma6[:] = gamma[:, 11]

    gamma7 = np.zeros(steps, dtype=float)
    gamma7[:] = gamma[:, 8]
    gamma8 = np.zeros(steps, dtype=float)
    gamma8[:] = gamma[:, 9]

    # create the response plots

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
               ["pentagon 1", "pentagon 2", "pentagon 4", "pentagon 5"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far-Field $\gamma_{23}$', fontsize=16)
    plt.ylabel(r'At Centroid $\xi = \ln \sqrt{uv}$', fontsize=16)

    ax2 = plt.subplot(1, 3, 2)
    # add the curves
    line1, = ax2.plot(shear, epsilon1, 'k-', linewidth=2)
    line2, = ax2.plot(shear, epsilon2, 'b-', linewidth=2)
    line3, = ax2.plot(shear, epsilon3, 'r-', linewidth=2)
    line4, = ax2.plot(shear, epsilon4, 'g-', linewidth=2)

    plt.title("Simple Shear", fontsize=20)
    plt.legend([line1, line2, line3, line4],
               ["pentagon 1", "pentagon 2", "pentagon 4", "pentagon 5"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far-Field $\gamma_{23}$', fontsize=16)
    plt.ylabel(r'At Centroid $\epsilon = \ln \sqrt{u/v}$', fontsize=16)

    ax3 = plt.subplot(1, 3, 3)
    # add the curves
    line1, = ax3.plot(shear, gamma1, 'k-', linewidth=2)
    line2, = ax3.plot(shear, gamma2, 'b-', linewidth=2)
    line3, = ax3.plot(shear, gamma3, 'r-', linewidth=2)
    line4, = ax3.plot(shear, gamma4, 'g-', linewidth=2)
    line5, = ax3.plot(shear, gamma5, 'm-', linewidth=2)

    plt.title("in the 2-3 Plane", fontsize=20)
    plt.legend([line1, line2, line3, line4, line5],
               ["pentagon 1", "pentagon 2", "pentagon 3", "pentagon 5",
                "pentagon 7"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far-Field $\gamma_{23}$', fontsize=16)
    plt.ylabel(r'At Centroid $\gamma = g$', fontsize=16)

    plt.savefig('pentagonalSimpleShear23.jpg')

    plt.show()


run()
