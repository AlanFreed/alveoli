#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ceChords import controlFiber, septalChord
import math
from peceHE import pece
# for creating graphics
from matplotlib import pyplot as plt
from matplotlib import ticker
import materialProperties as mp
import numpy as np
from pylab import rcParams

"""
Created on Thr Nov 07 2019
Updated on Thr May 28 2020

A test file for the chordal constitutive response in file ceChords.py.

author: Prof. Alan Freed
"""

# assign the number of CE iterations in PE(CE)^m, with m = 0 -> PE method
m = 2
# assign the number of integration steps to be applied
N = 75
# assign the time variables used
t0 = 0.0
tN = 1.0
dt = (tN - t0) / N
# assign the temperature variables used
T0 = 37.0
TN = 37.0
# assign the number of curves to plot
curves = 30
# assign number of variables
ctrlVars = 2
respVars = 4
# assign the initial conditions
eta0 = mp.etaCollagen()
stress0 = 0.0
# parameters used to determine chordal length
alpha = math.pi * 18.0 / 180.0
omega = math.pi * 54.0 / 180.0
lenOverDia = 1.0 / (math.tan(omega) * (1.0 + math.cos(alpha)))


def run():
    # create the data arrays used for plotting
    force = np.zeros((N+1, curves), dtype=float)
    strain = np.zeros((N+1, curves), dtype=float)
    entropy = np.zeros((N+1, curves), dtype=float)

    # initialize a counter
    ruptured = 0
    for j in range(curves):
        # create and initialize the control vector
        L0 = lenOverDia * mp.alveolarDiameter()
        LN = L0 * math.exp(0.1)
        ctrlVec = np.zeros((ctrlVars,), dtype=float)
        ctrlVec[0] = T0
        ctrlVec[1] = L0
        # create the control object
        ctrl = controlFiber(ctrlVec, dt)
        # create and initialize the response vector
        respVec = np.zeros((respVars,), dtype=float)
        respVec[0] = mp.etaCollagen()
        respVec[1] = 0.0
        respVec[2] = mp.etaElastin()
        respVec[3] = 0.0
        # create the response object
        resp = septalChord(ctrlVec, respVec)
        # provide the initial conditions
        force[0, j] = resp.force()
        strain[0, j] = resp.strain()
        entropy[0, j] = resp.entropy()
        # create the integrator
        solver = pece(ctrl, resp, m)
        # integrate the constitutive equation for this boundary value problem
        for i in range(1, N+1):
            # ctrlVec[0] = T0 + (TN - T0) * math.sin(i * math.pi / (2.0 * N))
            # ctrlVec[1] = L0 + (LN - L0) * math.sin(i * math.pi / (2.0 * N))
            ctrlVec[0] = T0 + (TN - T0) * i / N
            ctrlVec[1] = L0 + (LN - L0) * i / N
            solver.integrate(ctrlVec)
            solver.advance()
            force[i, j] = resp.force()
            strain[i, j] = resp.strain()
            entropy[i, j] = resp.entropy()
        if resp.isRuptured() is True:
            ruptured += 1

    print("Out of {} septal chords, ".format(curves) +
          "{} fibers ruptured.".format(ruptured))

    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    plt.figure(1)
    rcParams['figure.figsize'] = 14, 5

    # add the curves for collagen stress/strain plot
    ax1 = plt.subplot(1, 2, 1)
    # ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax1.plot(strain[:, 0], force[:, 0], 'k-', linewidth=2)
    line2, = ax1.plot(strain[:, 1], force[:, 1], 'k-', linewidth=2)
    line3, = ax1.plot(strain[:, 2], force[:, 2], 'k-', linewidth=2)
    line4, = ax1.plot(strain[:, 3], force[:, 3], 'k-', linewidth=2)
    line5, = ax1.plot(strain[:, 4], force[:, 4], 'k-', linewidth=2)
    line6, = ax1.plot(strain[:, 5], force[:, 5], 'k-', linewidth=2)
    line7, = ax1.plot(strain[:, 6], force[:, 6], 'k-', linewidth=2)
    line8, = ax1.plot(strain[:, 7], force[:, 7], 'k-', linewidth=2)
    line9, = ax1.plot(strain[:, 8], force[:, 8], 'k-', linewidth=2)
    line10, = ax1.plot(strain[:, 9], force[:, 9], 'k-', linewidth=2)
    line11, = ax1.plot(strain[:, 10], force[:, 10], 'k-', linewidth=2)
    line12, = ax1.plot(strain[:, 11], force[:, 11], 'k-', linewidth=2)
    line13, = ax1.plot(strain[:, 12], force[:, 12], 'k-', linewidth=2)
    line14, = ax1.plot(strain[:, 13], force[:, 13], 'k-', linewidth=2)
    line15, = ax1.plot(strain[:, 14], force[:, 14], 'k-', linewidth=2)
    line16, = ax1.plot(strain[:, 15], force[:, 15], 'k-', linewidth=2)
    line17, = ax1.plot(strain[:, 16], force[:, 16], 'k-', linewidth=2)
    line18, = ax1.plot(strain[:, 17], force[:, 17], 'k-', linewidth=2)
    line19, = ax1.plot(strain[:, 18], force[:, 18], 'k-', linewidth=2)
    line20, = ax1.plot(strain[:, 19], force[:, 19], 'k-', linewidth=2)
    line21, = ax1.plot(strain[:, 20], force[:, 20], 'k-', linewidth=2)
    line22, = ax1.plot(strain[:, 21], force[:, 21], 'k-', linewidth=2)
    line23, = ax1.plot(strain[:, 22], force[:, 22], 'k-', linewidth=2)
    line24, = ax1.plot(strain[:, 23], force[:, 23], 'k-', linewidth=2)
    line25, = ax1.plot(strain[:, 24], force[:, 24], 'k-', linewidth=2)
    line26, = ax1.plot(strain[:, 25], force[:, 25], 'k-', linewidth=2)
    line27, = ax1.plot(strain[:, 26], force[:, 26], 'k-', linewidth=2)
    line28, = ax1.plot(strain[:, 27], force[:, 27], 'k-', linewidth=2)
    line29, = ax1.plot(strain[:, 28], force[:, 28], 'k-', linewidth=2)
    line30, = ax1.plot(strain[:, 29], force[:, 29], 'k-', linewidth=2)
    plt.title("30 Force/Strain Curves", fontsize=20)
    plt.xlabel(r'strain  $e$', fontsize=18)
    plt.ylabel(r'force  $F$  (dynes)', fontsize=18)

    # add the curves
    ax2 = plt.subplot(1, 2, 2)
    # ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax2.plot(strain[:, 0], entropy[:, 0], 'k-', linewidth=2)
    line2, = ax2.plot(strain[:, 1], entropy[:, 1], 'k-', linewidth=2)
    line3, = ax2.plot(strain[:, 2], entropy[:, 2], 'k-', linewidth=2)
    line4, = ax2.plot(strain[:, 3], entropy[:, 3], 'k-', linewidth=2)
    line5, = ax2.plot(strain[:, 4], entropy[:, 4], 'k-', linewidth=2)
    line6, = ax2.plot(strain[:, 5], entropy[:, 5], 'k-', linewidth=2)
    line7, = ax2.plot(strain[:, 6], entropy[:, 6], 'k-', linewidth=2)
    line8, = ax2.plot(strain[:, 7], entropy[:, 7], 'k-', linewidth=2)
    line9, = ax2.plot(strain[:, 8], entropy[:, 8], 'k-', linewidth=2)
    line10, = ax2.plot(strain[:, 9], entropy[:, 9], 'k-', linewidth=2)
    line11, = ax2.plot(strain[:, 10], entropy[:, 10], 'k-', linewidth=2)
    line12, = ax2.plot(strain[:, 11], entropy[:, 11], 'k-', linewidth=2)
    line13, = ax2.plot(strain[:, 12], entropy[:, 12], 'k-', linewidth=2)
    line14, = ax2.plot(strain[:, 13], entropy[:, 13], 'k-', linewidth=2)
    line15, = ax2.plot(strain[:, 14], entropy[:, 14], 'k-', linewidth=2)
    line16, = ax2.plot(strain[:, 15], entropy[:, 15], 'k-', linewidth=2)
    line17, = ax2.plot(strain[:, 16], entropy[:, 16], 'k-', linewidth=2)
    line18, = ax2.plot(strain[:, 17], entropy[:, 17], 'k-', linewidth=2)
    line19, = ax2.plot(strain[:, 18], entropy[:, 18], 'k-', linewidth=2)
    line20, = ax2.plot(strain[:, 19], entropy[:, 19], 'k-', linewidth=2)
    line21, = ax2.plot(strain[:, 20], entropy[:, 20], 'k-', linewidth=2)
    line22, = ax2.plot(strain[:, 21], entropy[:, 21], 'k-', linewidth=2)
    line23, = ax2.plot(strain[:, 22], entropy[:, 22], 'k-', linewidth=2)
    line24, = ax2.plot(strain[:, 23], entropy[:, 23], 'k-', linewidth=2)
    line25, = ax2.plot(strain[:, 24], entropy[:, 24], 'k-', linewidth=2)
    line26, = ax2.plot(strain[:, 25], entropy[:, 25], 'k-', linewidth=2)
    line27, = ax2.plot(strain[:, 26], entropy[:, 26], 'k-', linewidth=2)
    line28, = ax2.plot(strain[:, 27], entropy[:, 27], 'k-', linewidth=2)
    line29, = ax2.plot(strain[:, 28], entropy[:, 28], 'k-', linewidth=2)
    line30, = ax2.plot(strain[:, 29], entropy[:, 29], 'k-', linewidth=2)
    plt.title("30 Entropy/Strain Curves", fontsize=20)
    plt.xlabel(r'strain  $e$', fontsize=16)
    plt.ylabel(r'entropy  $S$  (erg/K)', fontsize=16)

    plt.savefig('septalChords.jpg')

    plt.show()


run()
