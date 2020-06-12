#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 04 2019
Updated on Thr Apr 07 2020

Creates figures that examines the chordal and pentagonal responses of a regular
dodecahedron deformed into an irregular dodecahedron subjected to a pure shear.

author: Prof. Alan Freed
"""

from dodecahedra import dodecahedron
import math as m
import numpy as np
# for creating graphics
from matplotlib import pyplot as plt
from matplotlib import rc
from pylab import rcParams


def run():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # basic properties for creating the figures

    steps = 150
    gaussPts = 1
    maxSqueeze = 3.5

    # squeeze: 1-2

    strain = np.zeros((steps, 30), dtype=float)
    dilation = np.zeros((steps, 12), dtype=float)
    squeeze = np.zeros(steps, dtype=float)
    F = np.eye(3, dtype=float)
    d = dodecahedron(gaussPts, gaussPts, gaussPts, F)
    for i in range(steps):
        for j in range(1, 31):
            c = d.getChord(j)
            strain[i, j-1] = c.strain('curr')
        for j in range(1, 13):
            p = d.getPentagon(j)
            dilation[i, j-1] = p.arealStrain('curr')
        F[0, 0] += maxSqueeze / steps
        F[1, 1] = 1.0 / F[0, 0]
        squeeze[i] = m.log((F[0, 0] / F[1, 1])**(1.0/3.0))
        d.update(F)
        d.advance()
    strain1 = np.zeros(steps, dtype=float)
    strain1[:] = strain[:, 0]
    strain2 = np.zeros(steps, dtype=float)
    strain2[:] = strain[:, 1]
    strain3 = np.zeros(steps, dtype=float)
    strain3[:] = strain[:, 5]
    strain4 = np.zeros(steps, dtype=float)
    strain4[:] = strain[:, 9]
    strain5 = np.zeros(steps, dtype=float)
    strain5[:] = strain[:, 10]
    strain6 = np.zeros(steps, dtype=float)
    strain6[:] = strain[:, 17]

    dilation1 = np.zeros(steps, dtype=float)
    dilation1[:] = dilation[:, 0]
    dilation2 = np.zeros(steps, dtype=float)
    dilation2[:] = dilation[:, 1]
    dilation3 = np.zeros(steps, dtype=float)
    dilation3[:] = dilation[:, 4]

    # create the response plots: Figure 1 for squeeze in 1-2 directions

    plt.figure(1)
    rcParams['figure.figsize'] = 12, 5
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=11)

    ax21 = plt.subplot(1, 2, 1)
    # add the curves
    line1, = ax21.plot(squeeze, strain1, 'k-', linewidth=2)
    line2, = ax21.plot(squeeze, strain2, 'b-', linewidth=2)
    line3, = ax21.plot(squeeze, strain3, 'r-', linewidth=2)
    line4, = ax21.plot(squeeze, strain4, 'g-', linewidth=2)
    line5, = ax21.plot(squeeze, strain5, 'm-', linewidth=2)
    line6, = ax21.plot(squeeze, strain6, 'y-', linewidth=2)

    plt.title("Dodecahedral", fontsize=20)
    plt.legend([line1, line2, line3, line4, line5, line6],
               ["chord 1", "chord 2", "chord 6", "chord 10", "chord 11",
                "chord 18"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\ln \sqrt[3]{a/b}$', fontsize=16)
    plt.ylabel(r'Chordal Strain: $e=\ln (L/L_0)$', fontsize=16)

    ax22 = plt.subplot(1, 2, 2)
    # add the curves
    line1, = ax22.plot(squeeze, dilation1, 'k-', linewidth=2)
    line2, = ax22.plot(squeeze, dilation2, 'b-', linewidth=2)
    line3, = ax22.plot(squeeze, dilation3, 'r-', linewidth=2)

    plt.title(r"Pure Shear: 1 \& 2 Directions", fontsize=20)
    plt.legend([line1, line2, line3],
               ["pentagon 1", "pentagon 2", "pentagon 5"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\ln \sqrt[3]{a/b}$', fontsize=16)
    plt.ylabel(r'Pentagonal Dilation: $\xi = \ln \sqrt{A/A_0}$',
               fontsize=16)
    plt.savefig('squeeze12.jpg')

    plt.show()

    # squeeze: 1-3

    strain = np.zeros((steps, 30), dtype=float)
    dilation = np.zeros((steps, 12), dtype=float)
    squeeze = np.zeros(steps, dtype=float)
    F = np.eye(3, dtype=float)
    d = dodecahedron(gaussPts, gaussPts, gaussPts, F)
    for i in range(steps):
        for j in range(1, 31):
            c = d.getChord(j)
            strain[i, j-1] = c.strain('curr')
        for j in range(1, 13):
            p = d.getPentagon(j)
            dilation[i, j-1] = p.arealStrain('curr')
        F[2, 2] += maxSqueeze / steps
        F[0, 0] = 1.0 / F[2, 2]
        squeeze[i] = m.log((F[2, 2] / F[0, 0])**(1.0/3.0))
        d.update(F)
        d.advance()
    strain1 = np.zeros(steps, dtype=float)
    strain1[:] = strain[:, 0]
    strain2 = np.zeros(steps, dtype=float)
    strain2[:] = strain[:, 1]
    strain3 = np.zeros(steps, dtype=float)
    strain3[:] = strain[:, 5]
    strain4 = np.zeros(steps, dtype=float)
    strain4[:] = strain[:, 9]
    strain5 = np.zeros(steps, dtype=float)
    strain5[:] = strain[:, 10]
    strain6 = np.zeros(steps, dtype=float)
    strain6[:] = strain[:, 17]

    dilation1 = np.zeros(steps, dtype=float)
    dilation1[:] = dilation[:, 0]
    dilation2 = np.zeros(steps, dtype=float)
    dilation2[:] = dilation[:, 1]
    dilation3 = np.zeros(steps, dtype=float)
    dilation3[:] = dilation[:, 4]

    # create the response plots: Figure 2 for squeeze in 1-3 directions

    plt.figure(2)
    rcParams['figure.figsize'] = 12, 5
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=11)

    ax21 = plt.subplot(1, 2, 1)
    # add the curves
    line1, = ax21.plot(squeeze, strain1, 'k-', linewidth=2)
    line2, = ax21.plot(squeeze, strain2, 'b-', linewidth=2)
    line3, = ax21.plot(squeeze, strain3, 'r-', linewidth=2)
    line4, = ax21.plot(squeeze, strain4, 'g-', linewidth=2)
    line5, = ax21.plot(squeeze, strain5, 'm-', linewidth=2)
    line6, = ax21.plot(squeeze, strain6, 'y-', linewidth=2)

    plt.title("Dodecahedral", fontsize=20)
    plt.legend([line1, line2, line3, line4, line5, line6],
               ["chord 1", "chord 2", "chord 6", "chord 10", "chord 11",
                "chord 18"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\ln \sqrt[3]{c/a}$', fontsize=16)
    plt.ylabel(r'Chordal Strain: $e=\ln (L/L_0)$', fontsize=16)

    ax22 = plt.subplot(1, 2, 2)
    # add the curves
    line1, = ax22.plot(squeeze, dilation1, 'k-', linewidth=2)
    line2, = ax22.plot(squeeze, dilation2, 'b-', linewidth=2)
    line3, = ax22.plot(squeeze, dilation3, 'r-', linewidth=2)

    plt.title(r"Pure Shear: 1 \& 3 Directions", fontsize=20)
    plt.legend([line1, line2, line3],
               ["pentagon 1", "pentagon 2", "pentagon 5"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\ln \sqrt[3]{c/a}$', fontsize=16)
    plt.ylabel(r'Pentagonal Dilation: $\xi = \ln \sqrt{A/A_0}$',
               fontsize=16)
    plt.savefig('squeeze13.jpg')

    plt.show()

    # squeeze: 2-3

    strain = np.zeros((steps, 30), dtype=float)
    dilation = np.zeros((steps, 12), dtype=float)
    squeeze = np.zeros(steps, dtype=float)
    F = np.eye(3, dtype=float)
    d = dodecahedron(gaussPts, gaussPts, gaussPts, F)
    for i in range(steps):
        for j in range(1, 31):
            c = d.getChord(j)
            strain[i, j-1] = c.strain('curr')
        for j in range(1, 13):
            p = d.getPentagon(j)
            dilation[i, j-1] = p.arealStrain('curr')
        F[1, 1] += maxSqueeze / steps
        F[2, 2] = 1.0 / F[1, 1]
        squeeze[i] = m.log((F[1, 1] / F[2, 2])**(1.0/3.0))
        d.update(F)
        d.advance()
    strain1 = np.zeros(steps, dtype=float)
    strain1[:] = strain[:, 0]
    strain2 = np.zeros(steps, dtype=float)
    strain2[:] = strain[:, 1]
    strain3 = np.zeros(steps, dtype=float)
    strain3[:] = strain[:, 5]
    strain4 = np.zeros(steps, dtype=float)
    strain4[:] = strain[:, 9]
    strain5 = np.zeros(steps, dtype=float)
    strain5[:] = strain[:, 10]
    strain6 = np.zeros(steps, dtype=float)
    strain6[:] = strain[:, 17]

    dilation1 = np.zeros(steps, dtype=float)
    dilation1[:] = dilation[:, 0]
    dilation2 = np.zeros(steps, dtype=float)
    dilation2[:] = dilation[:, 1]
    dilation3 = np.zeros(steps, dtype=float)
    dilation3[:] = dilation[:, 4]

    # create the response plots: Figure 3 for squeeze in 2-3 directions

    plt.figure(3)
    rcParams['figure.figsize'] = 12, 5
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=11)

    ax31 = plt.subplot(1, 2, 1)
    # add the curves
    line1, = ax31.plot(squeeze, strain1, 'k-', linewidth=2)
    line2, = ax31.plot(squeeze, strain2, 'b-', linewidth=2)
    line3, = ax31.plot(squeeze, strain3, 'r-', linewidth=2)
    line4, = ax31.plot(squeeze, strain4, 'g-', linewidth=2)
    line5, = ax31.plot(squeeze, strain5, 'm-', linewidth=2)
    line6, = ax31.plot(squeeze, strain6, 'y-', linewidth=2)

    plt.title("Dodecahedral", fontsize=20)
    plt.legend([line1, line2, line3, line4, line5, line6],
               ["chord 1", "chord 2", "chord 6", "chord 10", "chord 11",
                "chord 18"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\ln \sqrt[3]{b/c}$', fontsize=16)
    plt.ylabel(r'Chordal Strain: $e=\ln (L/L_0)$', fontsize=16)

    ax32 = plt.subplot(1, 2, 2)
    # add the curves
    line1, = ax32.plot(squeeze, dilation1, 'k-', linewidth=2)
    line2, = ax32.plot(squeeze, dilation2, 'b-', linewidth=2)
    line3, = ax32.plot(squeeze, dilation3, 'r-', linewidth=2)

    plt.title(r"Pure Shear: 2 \& 3 Directions", fontsize=20)
    plt.legend([line1, line2, line3],
               ["pentagon 1", "pentagon 2", "pentagon 5"],
               bbox_to_anchor=(0.025, 0.975), loc=2, fontsize=14)
    plt.xlabel(r'Far Field $\ln \sqrt[3]{b/c}$', fontsize=16)
    plt.ylabel(r'Pentagonal Dilation: $\xi = \ln \sqrt{A/A_0}$',
               fontsize=16)
    plt.savefig('squeeze23.jpg')

    plt.show()


run()
