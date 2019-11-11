#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ceChords as ce
import math as m
# for creating graphics
from matplotlib import pyplot as plt
from matplotlib import rc
import materialProperties as mp
import numpy as np
from pylab import rcParams

"""
Created on Thr Nov 07 2019
Updated on Thr Nov 07 2019

A test file for the chordal constitutive response in file ceChords.py.

author: Prof. Alan Freed
"""


def runCollagen():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    curves = 30
    points = 75
    strain = np.zeros((points, curves), dtype=float)
    stress = np.zeros((points, curves), dtype=float)

    alpha = m.pi * 18.0 / 180.0
    omega = m.pi * 54.0 / 180.0
    lenOverDia = 1.0 / (m.tan(omega) * (1.0 + m.cos(alpha)))

    for j in range(curves):
        lenF = lenOverDia * mp.alveolarDiameter()
        chord = ce.ceChord(lenF)
        dStrain = chord.strainMax / points
        for i in range(1, points):
            strain[i, j] = strain[i-1, j] + dStrain
            stress[i, j] = chord.collagenStress(strain[i, j])

    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=11)

    plt.figure(1)
    rcParams['figure.figsize'] = 8, 5

    # add the curves
    ax = plt.subplot(1, 1, 1)
    line1, = ax.plot(strain[:, 0], stress[:, 0], 'k-', linewidth=2)
    line2, = ax.plot(strain[:, 1], stress[:, 1], 'k-', linewidth=2)
    line3, = ax.plot(strain[:, 2], stress[:, 2], 'k-', linewidth=2)
    line4, = ax.plot(strain[:, 3], stress[:, 3], 'k-', linewidth=2)
    line5, = ax.plot(strain[:, 4], stress[:, 4], 'k-', linewidth=2)
    line6, = ax.plot(strain[:, 5], stress[:, 5], 'k-', linewidth=2)
    line7, = ax.plot(strain[:, 6], stress[:, 6], 'k-', linewidth=2)
    line8, = ax.plot(strain[:, 7], stress[:, 7], 'k-', linewidth=2)
    line9, = ax.plot(strain[:, 8], stress[:, 8], 'k-', linewidth=2)
    line10, = ax.plot(strain[:, 9], stress[:, 9], 'k-', linewidth=2)
    line11, = ax.plot(strain[:, 10], stress[:, 10], 'k-', linewidth=2)
    line12, = ax.plot(strain[:, 11], stress[:, 11], 'k-', linewidth=2)
    line13, = ax.plot(strain[:, 12], stress[:, 12], 'k-', linewidth=2)
    line14, = ax.plot(strain[:, 13], stress[:, 13], 'k-', linewidth=2)
    line15, = ax.plot(strain[:, 14], stress[:, 14], 'k-', linewidth=2)
    line16, = ax.plot(strain[:, 15], stress[:, 15], 'k-', linewidth=2)
    line17, = ax.plot(strain[:, 16], stress[:, 16], 'k-', linewidth=2)
    line18, = ax.plot(strain[:, 17], stress[:, 17], 'k-', linewidth=2)
    line19, = ax.plot(strain[:, 18], stress[:, 18], 'k-', linewidth=2)
    line20, = ax.plot(strain[:, 19], stress[:, 19], 'k-', linewidth=2)
    line21, = ax.plot(strain[:, 20], stress[:, 20], 'k-', linewidth=2)
    line22, = ax.plot(strain[:, 21], stress[:, 21], 'k-', linewidth=2)
    line23, = ax.plot(strain[:, 22], stress[:, 22], 'k-', linewidth=2)
    line24, = ax.plot(strain[:, 23], stress[:, 23], 'k-', linewidth=2)
    line25, = ax.plot(strain[:, 24], stress[:, 24], 'k-', linewidth=2)
    line16, = ax.plot(strain[:, 25], stress[:, 25], 'k-', linewidth=2)
    line17, = ax.plot(strain[:, 26], stress[:, 26], 'k-', linewidth=2)
    line18, = ax.plot(strain[:, 27], stress[:, 27], 'k-', linewidth=2)
    line19, = ax.plot(strain[:, 28], stress[:, 28], 'k-', linewidth=2)
    line20, = ax.plot(strain[:, 29], stress[:, 29], 'k-', linewidth=2)
    plt.title("30 Collagen Stress/Strain Curves", fontsize=20)
    plt.xlabel(r'strain  $\epsilon$', fontsize=16)
    plt.ylabel(r'stress  $\sigma$  (barye)', fontsize=16)

    plt.savefig('collagenStressStrain.jpg')

    plt.show()


def runElastin():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    curves = 30
    points = 75
    strain = np.zeros((points, curves), dtype=float)
    stress = np.zeros((points, curves), dtype=float)

    alpha = m.pi * 18.0 / 180.0
    omega = m.pi * 54.0 / 180.0
    lenOverDia = 1.0 / (m.tan(omega) * (1.0 + m.cos(alpha)))

    for j in range(curves):
        lenF = lenOverDia * mp.alveolarDiameter()
        chord = ce.ceChord(lenF)
        dStrain = chord.strainMax / points
        for i in range(1, points):
            strain[i, j] = strain[i-1, j] + dStrain
            stress[i, j] = chord.elastinStress(strain[i, j])

    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=11)

    plt.figure(1)
    rcParams['figure.figsize'] = 8, 5

    # add the curves
    ax = plt.subplot(1, 1, 1)
    line1, = ax.plot(strain[:, 0], stress[:, 0], 'k-', linewidth=2)
    line2, = ax.plot(strain[:, 1], stress[:, 1], 'k-', linewidth=2)
    line3, = ax.plot(strain[:, 2], stress[:, 2], 'k-', linewidth=2)
    line4, = ax.plot(strain[:, 3], stress[:, 3], 'k-', linewidth=2)
    line5, = ax.plot(strain[:, 4], stress[:, 4], 'k-', linewidth=2)
    line6, = ax.plot(strain[:, 5], stress[:, 5], 'k-', linewidth=2)
    line7, = ax.plot(strain[:, 6], stress[:, 6], 'k-', linewidth=2)
    line8, = ax.plot(strain[:, 7], stress[:, 7], 'k-', linewidth=2)
    line9, = ax.plot(strain[:, 8], stress[:, 8], 'k-', linewidth=2)
    line10, = ax.plot(strain[:, 9], stress[:, 9], 'k-', linewidth=2)
    line11, = ax.plot(strain[:, 10], stress[:, 10], 'k-', linewidth=2)
    line12, = ax.plot(strain[:, 11], stress[:, 11], 'k-', linewidth=2)
    line13, = ax.plot(strain[:, 12], stress[:, 12], 'k-', linewidth=2)
    line14, = ax.plot(strain[:, 13], stress[:, 13], 'k-', linewidth=2)
    line15, = ax.plot(strain[:, 14], stress[:, 14], 'k-', linewidth=2)
    line16, = ax.plot(strain[:, 15], stress[:, 15], 'k-', linewidth=2)
    line17, = ax.plot(strain[:, 16], stress[:, 16], 'k-', linewidth=2)
    line18, = ax.plot(strain[:, 17], stress[:, 17], 'k-', linewidth=2)
    line19, = ax.plot(strain[:, 18], stress[:, 18], 'k-', linewidth=2)
    line20, = ax.plot(strain[:, 19], stress[:, 19], 'k-', linewidth=2)
    line21, = ax.plot(strain[:, 20], stress[:, 20], 'k-', linewidth=2)
    line22, = ax.plot(strain[:, 21], stress[:, 21], 'k-', linewidth=2)
    line23, = ax.plot(strain[:, 22], stress[:, 22], 'k-', linewidth=2)
    line24, = ax.plot(strain[:, 23], stress[:, 23], 'k-', linewidth=2)
    line25, = ax.plot(strain[:, 24], stress[:, 24], 'k-', linewidth=2)
    line16, = ax.plot(strain[:, 25], stress[:, 25], 'k-', linewidth=2)
    line17, = ax.plot(strain[:, 26], stress[:, 26], 'k-', linewidth=2)
    line18, = ax.plot(strain[:, 27], stress[:, 27], 'k-', linewidth=2)
    line19, = ax.plot(strain[:, 28], stress[:, 28], 'k-', linewidth=2)
    line20, = ax.plot(strain[:, 29], stress[:, 29], 'k-', linewidth=2)
    plt.title("30 Elastin Stress/Strain Curves", fontsize=20)
    plt.xlabel(r'strain  $\epsilon$', fontsize=16)
    plt.ylabel(r'stress  $\sigma$  (barye)', fontsize=16)

    plt.savefig('elastinStressStrain.jpg')

    plt.show()


runCollagen()
runElastin()
