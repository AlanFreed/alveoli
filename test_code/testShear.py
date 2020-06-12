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

Creates figures that examines the cordal and pentagonal responses of a regular
dodecahedron deformed into an irregular dodecahedron subjected to simple shear.

author: Prof. Alan Freed
"""


def run():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # basic properties for creating the figures

    steps = 150
    gaussPts = 1
    maxShear = 1.0

    # shear 1-2

    strain = np.zeros((steps, 30), dtype=float)
    dilation = np.zeros((steps, 12), dtype=float)
    shear = np.zeros(steps, dtype=float)
    F = np.eye(3, dtype=float)
    d = dodecahedron(gaussPts, gaussPts, gaussPts, F)
    for i in range(steps):
        for j in range(1, 31):
            c = d.getChord(j)
            strain[i, j-1] = c.strain('curr')
        for j in range(1, 13):
            p = d.getPentagon(j)
            dilation[i, j-1] = p.arealStrain('curr')
        F[0, 1] += maxShear / steps
        shear[i] = F[0, 1]
        d.update(F)
        d.advance()
    strain1 = np.zeros(steps, dtype=float)
    strain1[:] = strain[:, 0]
    strain2 = np.zeros(steps, dtype=float)
    strain2[:] = strain[:, 1]
    strain3 = np.zeros(steps, dtype=float)
    strain3[:] = strain[:, 3]
    strain4 = np.zeros(steps, dtype=float)
    strain4[:] = strain[:, 5]
    strain5 = np.zeros(steps, dtype=float)
    strain5[:] = strain[:, 7]
    strain6 = np.zeros(steps, dtype=float)
    strain6[:] = strain[:, 9]
    strain7 = np.zeros(steps, dtype=float)
    strain7[:] = strain[:, 12]
    strain8 = np.zeros(steps, dtype=float)
    strain8[:] = strain[:, 17]

    dilation1 = np.zeros(steps, dtype=float)
    dilation1[:] = dilation[:, 0]
    dilation2 = np.zeros(steps, dtype=float)
    dilation2[:] = dilation[:, 1]
    dilation3 = np.zeros(steps, dtype=float)
    dilation3[:] = dilation[:, 2]
    dilation4 = np.zeros(steps, dtype=float)
    dilation4[:] = dilation[:, 4]

    # create the response plots: Figure 1 for shear in 12 plane

    plt.figure(1)
    rcParams['figure.figsize'] = 12, 5
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    ax1 = plt.subplot(1, 2, 1)
    # add the curves
    line1, = ax1.plot(shear, strain1, 'k-', linewidth=2)
    line2, = ax1.plot(shear, strain2, 'b-', linewidth=2)
    line3, = ax1.plot(shear, strain3, 'r-', linewidth=2)
    line4, = ax1.plot(shear, strain4, 'g-', linewidth=2)
    line5, = ax1.plot(shear, strain5, 'm-', linewidth=2)
    line6, = ax1.plot(shear, strain6, 'y-', linewidth=2)
    line7, = ax1.plot(shear, strain7, 'k--', linewidth=2)
    line8, = ax1.plot(shear, strain8, 'b--', linewidth=2)

    plt.title("Dodecahedral", fontsize=20)
    plt.legend([line1, line2, line3, line4, line5, line6, line7, line8],
               ["chord 1", "chord 2", "chord 4", "chord 6", "chord 8",
                "chord 10", "chord 12", "chord 18"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\gamma_{12}$', fontsize=16)
    plt.ylabel(r'Chordal Strain: $e=\ln (L/L_0)$', fontsize=16)

    ax2 = plt.subplot(1, 2, 2)
    # add the curves
    line1, = ax2.plot(shear, dilation1, 'k-', linewidth=2)
    line2, = ax2.plot(shear, dilation2, 'b-', linewidth=2)
    line3, = ax2.plot(shear, dilation3, 'r-', linewidth=2)
    line4, = ax2.plot(shear, dilation4, 'g-', linewidth=2)

    plt.title("Simple Shear: 12 Plane", fontsize=20)
    plt.legend([line1, line2, line3, line4],
               ["pentagon 1", "pentagon 2", "pentagon 3", "pentagon 5"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\gamma_{12}$', fontsize=16)
    plt.ylabel(r'Pentagonal Dilation: $\xi = \ln \sqrt{A/A_0}$',
               fontsize=16)
    plt.savefig('shear12.jpg')

    plt.show()

    # shear 1-3

    strain = np.zeros((steps, 30), dtype=float)
    dilation = np.zeros((steps, 12), dtype=float)
    shear = np.zeros(steps, dtype=float)
    F = np.eye(3, dtype=float)
    d = dodecahedron(gaussPts, gaussPts, gaussPts, F)
    for i in range(steps):
        for j in range(1, 31):
            c = d.getChord(j)
            strain[i, j-1] = c.strain('curr')
        for j in range(1, 13):
            p = d.getPentagon(j)
            dilation[i, j-1] = p.arealStrain('curr')
        F[0, 2] += maxShear / steps
        shear[i] = F[0, 2]
        d.update(F)
        d.advance()
    strain1 = np.zeros(steps, dtype=float)
    strain1[:] = strain[:, 0]
    strain2 = np.zeros(steps, dtype=float)
    strain2[:] = strain[:, 1]
    strain3 = np.zeros(steps, dtype=float)
    strain3[:] = strain[:, 2]
    strain4 = np.zeros(steps, dtype=float)
    strain4[:] = strain[:, 5]
    strain5 = np.zeros(steps, dtype=float)
    strain5[:] = strain[:, 6]
    strain6 = np.zeros(steps, dtype=float)
    strain6[:] = strain[:, 9]
    strain7 = np.zeros(steps, dtype=float)
    strain7[:] = strain[:, 10]
    strain8 = np.zeros(steps, dtype=float)
    strain8[:] = strain[:, 11]

    dilation1 = np.zeros(steps, dtype=float)
    dilation1[:] = dilation[:, 0]
    dilation2 = np.zeros(steps, dtype=float)
    dilation2[:] = dilation[:, 1]
    dilation3 = np.zeros(steps, dtype=float)
    dilation3[:] = dilation[:, 4]
    dilation4 = np.zeros(steps, dtype=float)
    dilation4[:] = dilation[:, 5]

    # create the response plots: Figure 2 for shear in 13 plane

    plt.figure(2)
    rcParams['figure.figsize'] = 12, 5
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    ax3 = plt.subplot(1, 2, 1)
    # add the curves
    line1, = ax3.plot(shear, strain1, 'k-', linewidth=2)
    line2, = ax3.plot(shear, strain2, 'b-', linewidth=2)
    line3, = ax3.plot(shear, strain3, 'r-', linewidth=2)
    line4, = ax3.plot(shear, strain4, 'g-', linewidth=2)
    line5, = ax3.plot(shear, strain5, 'm-', linewidth=2)
    line6, = ax3.plot(shear, strain6, 'y-', linewidth=2)
    line7, = ax3.plot(shear, strain7, 'k--', linewidth=2)
    line8, = ax3.plot(shear, strain8, 'b--', linewidth=2)

    plt.title("Dodecahedral", fontsize=20)
    plt.legend([line1, line2, line3, line4, line5, line6, line7, line8],
               ["chord 1", "chord 2", "chord 3", "chord 6", "chord 7",
                "chord 10", "chord 11", "chord 12"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\gamma_{13}$', fontsize=16)
    plt.ylabel(r'Chordal Strain: $e=\ln (L/L_0)$', fontsize=16)

    ax4 = plt.subplot(1, 2, 2)
    # add the curves
    line1, = ax4.plot(shear, dilation1, 'k-', linewidth=2)
    line2, = ax4.plot(shear, dilation2, 'b-', linewidth=2)
    line3, = ax4.plot(shear, dilation3, 'r-', linewidth=2)
    line4, = ax4.plot(shear, dilation4, 'g-', linewidth=2)

    plt.title("Simple Shear: 13 Plane", fontsize=20)
    plt.legend([line1, line2, line3, line4],
               ["pentagon 1", "pentagon 2", "pentagon 5", "pentagon 6"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\gamma_{13}$', fontsize=16)
    plt.ylabel(r'Pentagonal Dilation: $\xi = \ln \sqrt{A/A_0}$',
               fontsize=16)
    plt.savefig('shear13.jpg')

    plt.show()

    # shear 2-3

    strain = np.zeros((steps, 30), dtype=float)
    dilation = np.zeros((steps, 12), dtype=float)
    shear = np.zeros(steps, dtype=float)
    F = np.eye(3, dtype=float)
    d = dodecahedron(gaussPts, gaussPts, gaussPts, F)
    for i in range(steps):
        for j in range(1, 31):
            c = d.getChord(j)
            strain[i, j-1] = c.strain('curr')
        for j in range(1, 13):
            p = d.getPentagon(j)
            dilation[i, j-1] = p.arealStrain('curr')
        F[1, 2] += maxShear / steps
        shear[i] = F[1, 2]
        d.update(F)
        d.advance()
    strain1 = np.zeros(steps, dtype=float)
    strain1[:] = strain[:, 0]
    strain2 = np.zeros(steps, dtype=float)
    strain2[:] = strain[:, 1]
    strain3 = np.zeros(steps, dtype=float)
    strain3[:] = strain[:, 2]
    strain4 = np.zeros(steps, dtype=float)
    strain4[:] = strain[:, 5]
    strain5 = np.zeros(steps, dtype=float)
    strain5[:] = strain[:, 7]
    strain6 = np.zeros(steps, dtype=float)
    strain6[:] = strain[:, 9]
    strain7 = np.zeros(steps, dtype=float)
    strain7[:] = strain[:, 10]
    strain8 = np.zeros(steps, dtype=float)
    strain8[:] = strain[:, 12]

    dilation1 = np.zeros(steps, dtype=float)
    dilation1[:] = dilation[:, 0]
    dilation2 = np.zeros(steps, dtype=float)
    dilation2[:] = dilation[:, 1]
    dilation3 = np.zeros(steps, dtype=float)
    dilation3[:] = dilation[:, 3]
    dilation4 = np.zeros(steps, dtype=float)
    dilation4[:] = dilation[:, 4]

    # create the response plots: Figure 3 for shear in 23 plane

    plt.figure(3)
    rcParams['figure.figsize'] = 12, 5
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    ax5 = plt.subplot(1, 2, 1)
    # add the curves
    line1, = ax5.plot(shear, strain1, 'k-', linewidth=2)
    line2, = ax5.plot(shear, strain2, 'b-', linewidth=2)
    line3, = ax5.plot(shear, strain3, 'r-', linewidth=2)
    line4, = ax5.plot(shear, strain4, 'g-', linewidth=2)
    line5, = ax5.plot(shear, strain5, 'm-', linewidth=2)
    line6, = ax5.plot(shear, strain6, 'y-', linewidth=2)
    line7, = ax5.plot(shear, strain7, 'k--', linewidth=2)
    line8, = ax5.plot(shear, strain8, 'b--', linewidth=2)

    plt.title("Dodecahedral", fontsize=20)
    plt.legend([line1, line2, line3, line4, line5, line6, line7, line8],
               ["chord 1", "chord 2", "chord 3", "chord 6", "chord 7",
                "chord 10", "chord 11", "chord 13"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\gamma_{23}$', fontsize=16)
    plt.ylabel(r'Chordal Strain: $e=\ln (L/L_0)$', fontsize=16)

    ax6 = plt.subplot(1, 2, 2)
    # add the curves
    line1, = ax6.plot(shear, dilation1, 'k-', linewidth=2)
    line2, = ax6.plot(shear, dilation2, 'b-', linewidth=2)
    line3, = ax6.plot(shear, dilation3, 'r-', linewidth=2)
    line4, = ax6.plot(shear, dilation4, 'g-', linewidth=2)

    plt.title("Simple Shear: 23 Plane", fontsize=20)
    plt.legend([line1, line2, line3, line4],
               ["pentagon 1", "pentagon 2", "pentagon 4", "pentagon 5"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\gamma_{23}$', fontsize=16)
    plt.ylabel(r'Pentagonal Dilation: $\xi = \ln \sqrt{A/A_0}$',
               fontsize=16)
    plt.savefig('shear23.jpg')

    plt.show()


run()
