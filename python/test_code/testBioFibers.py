#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ceChords import BioFiber, ControlFiber
import math
from peceHE import PECE
# for creating graphics
from matplotlib import pyplot as plt
from matplotlib import ticker
import materialProperties as mp
import numpy as np
from pylab import rcParams

"""
Created on Wed May 27 2020
Updated on Fri Nov 01 2020

A test file for the BioFiber constitutive response in file ceChords.py.

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
# assign the length variables used
L0 = 1.0
LN = L0 * math.exp(0.05)
# assign the number of curves to plot
curves = 30
# assign number of variables
ctrlVars = 2
respVars = 2
# assign the initial conditions
eta0 = mp.etaCollagen()
stress0 = 0.0


def run():
    # create the data arrays used for plotting
    error = np.zeros((N+1, curves), dtype=float)
    stress = np.zeros((N+1, curves), dtype=float)
    strain = np.zeros((N+1, curves), dtype=float)
    entropy = np.zeros((N+1, curves), dtype=float)
    relStress = np.zeros((N+1, curves), dtype=float)
    relEntropy = np.zeros((N+1, curves), dtype=float)

    # initialize a counter
    ruptured = 0
    for j in range(curves):
        # provide the initial conditions
        stress[0, j] = stress0
        entropy[0, j] = eta0
        relStress[0, j] = 0.0
        relEntropy[0, j] = 0.0
        # create the control vector and object
        eVec0 = np.zeros((ctrlVars,), dtype=float)
        xVec0 = np.zeros((ctrlVars,), dtype=float)
        xVec0[0] = T0
        xVec0[1] = L0
        ctrl = ControlFiber(eVec0, xVec0, dt)
        # create the response vector and object
        rho = mp.rhoCollagen()
        Cp = mp.CpCollagen()
        alpha = mp.alphaCollagen()
        E1, E2, e_t, e_f, s_0 = mp.collagenFiber()
        yVec0 = np.zeros((respVars,), dtype=float)
        yVec0[0] = mp.etaCollagen()
        yVec0[1] = s_0
        
        resp = BioFiber(yVec0, rho, Cp, alpha, E1, E2, e_t, e_f)
        # create the integrator
        solver = PECE(ctrl, resp, m)
        # establish the initial conditios
        e0 = solver.getE()
        y0 = solver.getY()
        ymy0 = solver.getYminusY0()
        strain[0, j] = e0[1]
        entropy[0, j] = y0[0]
        stress[0, j] = y0[1]
        relEntropy[0, j] = ymy0[0]
        relStress[0, j] = ymy0[1]
        # integrate the constitutive equation for this boundary value problem
        xVec = np.zeros((ctrlVars,), dtype=float)
        for i in range(1, N+1):
            # ctrlVec[0] = T0 + (TN - T0) * math.sin(i * math.pi / (2.0 * N))
            # ctrlVec[1] = L0 + (LN - L0) * math.sin(i * math.pi / (2.0 * N))
            xVec[0] = T0 + (TN - T0) * i / N
            xVec[1] = L0 + (LN - L0) * i / N
            solver.integrate(xVec)
            solver.advance()
            e = solver.getE()
            y = solver.getY()
            ymy0 = solver.getYminusY0()
            err = solver.getError()
            strain[i, j] = e[1]
            entropy[i, j] = y[0]
            stress[i, j] = y[1]
            relEntropy[i, j] = ymy0[0]
            relStress[i, j] = ymy0[1]
            error[i, j] = err
            if i == 1:
                error[0, j] = err
        didRupture = resp.isRuptured()
        if didRupture[0]:
            ruptured += 1
        if j == curves - 1:
            print('y0     - ' + y0.__str__())
            print('y      = ' + y.__str__())
            print('y - y0 = ' + ymy0.__str__())

    print("\nOut of {}, ".format(curves) +
          "{} fibers ruptured.\n".format(ruptured))

    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    plt.figure(3)
    rcParams['figure.figsize'] = 7, 5
    plt.show()

    plt.figure(1)
    rcParams['figure.figsize'] = 7, 5

    # add the curves
    ax = plt.subplot(1, 1, 1)
    ax.set_yscale('log')
    line1, = ax.plot(strain[:, 0], error[:, 0], 'k-', linewidth=2)
    line2, = ax.plot(strain[:, 1], error[:, 1], 'k-', linewidth=2)
    line3, = ax.plot(strain[:, 2], error[:, 2], 'k-', linewidth=2)
    line4, = ax.plot(strain[:, 3], error[:, 3], 'k-', linewidth=2)
    line5, = ax.plot(strain[:, 4], error[:, 4], 'k-', linewidth=2)
    line6, = ax.plot(strain[:, 5], error[:, 5], 'k-', linewidth=2)
    line7, = ax.plot(strain[:, 6], error[:, 6], 'k-', linewidth=2)
    line8, = ax.plot(strain[:, 7], error[:, 7], 'k-', linewidth=2)
    line9, = ax.plot(strain[:, 8], error[:, 8], 'k-', linewidth=2)
    line10, = ax.plot(strain[:, 9], error[:, 9], 'k-', linewidth=2)
    line11, = ax.plot(strain[:, 10], error[:, 10], 'k-', linewidth=2)
    line12, = ax.plot(strain[:, 11], error[:, 11], 'k-', linewidth=2)
    line13, = ax.plot(strain[:, 12], error[:, 12], 'k-', linewidth=2)
    line14, = ax.plot(strain[:, 13], error[:, 13], 'k-', linewidth=2)
    line15, = ax.plot(strain[:, 14], error[:, 14], 'k-', linewidth=2)
    line16, = ax.plot(strain[:, 15], error[:, 15], 'k-', linewidth=2)
    line17, = ax.plot(strain[:, 16], error[:, 16], 'k-', linewidth=2)
    line18, = ax.plot(strain[:, 17], error[:, 17], 'k-', linewidth=2)
    line19, = ax.plot(strain[:, 18], error[:, 18], 'k-', linewidth=2)
    line20, = ax.plot(strain[:, 19], error[:, 19], 'k-', linewidth=2)
    line21, = ax.plot(strain[:, 20], error[:, 20], 'k-', linewidth=2)
    line22, = ax.plot(strain[:, 21], error[:, 21], 'k-', linewidth=2)
    line23, = ax.plot(strain[:, 22], error[:, 22], 'k-', linewidth=2)
    line24, = ax.plot(strain[:, 23], error[:, 23], 'k-', linewidth=2)
    line25, = ax.plot(strain[:, 24], error[:, 24], 'k-', linewidth=2)
    line26, = ax.plot(strain[:, 25], error[:, 25], 'k-', linewidth=2)
    line27, = ax.plot(strain[:, 26], error[:, 26], 'k-', linewidth=2)
    line28, = ax.plot(strain[:, 27], error[:, 27], 'k-', linewidth=2)
    line29, = ax.plot(strain[:, 28], error[:, 28], 'k-', linewidth=2)
    line30, = ax.plot(strain[:, 29], error[:, 29], 'k-', linewidth=2)
    plt.title("Numeric Errors of Integration", fontsize=20)
    plt.xlabel(r'strain  $e$', fontsize=16)
    plt.ylabel('local truncation error', fontsize=16)

    plt.savefig('bioFiberErrors.jpg')
    plt.show()

    plt.figure(4)
    rcParams['figure.figsize'] = 16, 16
    plt.show()

    plt.figure(2)
    rcParams['figure.figsize'] = 16, 16

    # add the curves for collagen absolute-stress/strain plot
    ax1 = plt.subplot(2, 2, 1)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax1.plot(strain[:, 0], stress[:, 0], 'k-', linewidth=2)
    line2, = ax1.plot(strain[:, 1], stress[:, 1], 'k-', linewidth=2)
    line3, = ax1.plot(strain[:, 2], stress[:, 2], 'k-', linewidth=2)
    line4, = ax1.plot(strain[:, 3], stress[:, 3], 'k-', linewidth=2)
    line5, = ax1.plot(strain[:, 4], stress[:, 4], 'k-', linewidth=2)
    line6, = ax1.plot(strain[:, 5], stress[:, 5], 'k-', linewidth=2)
    line7, = ax1.plot(strain[:, 6], stress[:, 6], 'k-', linewidth=2)
    line8, = ax1.plot(strain[:, 7], stress[:, 7], 'k-', linewidth=2)
    line9, = ax1.plot(strain[:, 8], stress[:, 8], 'k-', linewidth=2)
    line10, = ax1.plot(strain[:, 9], stress[:, 9], 'k-', linewidth=2)
    line11, = ax1.plot(strain[:, 10], stress[:, 10], 'k-', linewidth=2)
    line12, = ax1.plot(strain[:, 11], stress[:, 11], 'k-', linewidth=2)
    line13, = ax1.plot(strain[:, 12], stress[:, 12], 'k-', linewidth=2)
    line14, = ax1.plot(strain[:, 13], stress[:, 13], 'k-', linewidth=2)
    line15, = ax1.plot(strain[:, 14], stress[:, 14], 'k-', linewidth=2)
    line16, = ax1.plot(strain[:, 15], stress[:, 15], 'k-', linewidth=2)
    line17, = ax1.plot(strain[:, 16], stress[:, 16], 'k-', linewidth=2)
    line18, = ax1.plot(strain[:, 17], stress[:, 17], 'k-', linewidth=2)
    line19, = ax1.plot(strain[:, 18], stress[:, 18], 'k-', linewidth=2)
    line20, = ax1.plot(strain[:, 19], stress[:, 19], 'k-', linewidth=2)
    line21, = ax1.plot(strain[:, 20], stress[:, 20], 'k-', linewidth=2)
    line22, = ax1.plot(strain[:, 21], stress[:, 21], 'k-', linewidth=2)
    line23, = ax1.plot(strain[:, 22], stress[:, 22], 'k-', linewidth=2)
    line24, = ax1.plot(strain[:, 23], stress[:, 23], 'k-', linewidth=2)
    line25, = ax1.plot(strain[:, 24], stress[:, 24], 'k-', linewidth=2)
    line26, = ax1.plot(strain[:, 25], stress[:, 25], 'k-', linewidth=2)
    line27, = ax1.plot(strain[:, 26], stress[:, 26], 'k-', linewidth=2)
    line28, = ax1.plot(strain[:, 27], stress[:, 27], 'k-', linewidth=2)
    line29, = ax1.plot(strain[:, 28], stress[:, 28], 'k-', linewidth=2)
    line30, = ax1.plot(strain[:, 29], stress[:, 29], 'k-', linewidth=2)
    plt.title("30 Absolute Stress/Strain Curves", fontsize=20)
    plt.xlabel(r'strain  $e$', fontsize=18)
    plt.ylabel(r'absolute stress  $s$  (barye)', fontsize=18)

    # add the curves for collagen absolute-entropy/strain plot
    ax2 = plt.subplot(2, 2, 2)
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
    plt.title("30 Absolute Entropy/Strain Curves", fontsize=20)
    plt.xlabel(r'strain  $e$', fontsize=16)
    plt.ylabel(r'absolute entropy density  $\eta$  (erg/g.K)', fontsize=16)

    # add the curves for collagen relative-stress/strain plot
    ax3 = plt.subplot(2, 2, 3)
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax3.plot(strain[:, 0], relStress[:, 0], 'k-', linewidth=2)
    line2, = ax3.plot(strain[:, 1], relStress[:, 1], 'k-', linewidth=2)
    line3, = ax3.plot(strain[:, 2], relStress[:, 2], 'k-', linewidth=2)
    line4, = ax3.plot(strain[:, 3], relStress[:, 3], 'k-', linewidth=2)
    line5, = ax3.plot(strain[:, 4], relStress[:, 4], 'k-', linewidth=2)
    line6, = ax3.plot(strain[:, 5], relStress[:, 5], 'k-', linewidth=2)
    line7, = ax3.plot(strain[:, 6], relStress[:, 6], 'k-', linewidth=2)
    line8, = ax3.plot(strain[:, 7], relStress[:, 7], 'k-', linewidth=2)
    line9, = ax3.plot(strain[:, 8], relStress[:, 8], 'k-', linewidth=2)
    line10, = ax3.plot(strain[:, 9], relStress[:, 9], 'k-', linewidth=2)
    line11, = ax3.plot(strain[:, 10], relStress[:, 10], 'k-', linewidth=2)
    line12, = ax3.plot(strain[:, 11], relStress[:, 11], 'k-', linewidth=2)
    line13, = ax3.plot(strain[:, 12], relStress[:, 12], 'k-', linewidth=2)
    line14, = ax3.plot(strain[:, 13], relStress[:, 13], 'k-', linewidth=2)
    line15, = ax3.plot(strain[:, 14], relStress[:, 14], 'k-', linewidth=2)
    line16, = ax3.plot(strain[:, 15], relStress[:, 15], 'k-', linewidth=2)
    line17, = ax3.plot(strain[:, 16], relStress[:, 16], 'k-', linewidth=2)
    line18, = ax3.plot(strain[:, 17], relStress[:, 17], 'k-', linewidth=2)
    line19, = ax3.plot(strain[:, 18], relStress[:, 18], 'k-', linewidth=2)
    line20, = ax3.plot(strain[:, 19], relStress[:, 19], 'k-', linewidth=2)
    line21, = ax3.plot(strain[:, 20], relStress[:, 20], 'k-', linewidth=2)
    line22, = ax3.plot(strain[:, 21], relStress[:, 21], 'k-', linewidth=2)
    line23, = ax3.plot(strain[:, 22], relStress[:, 22], 'k-', linewidth=2)
    line24, = ax3.plot(strain[:, 23], relStress[:, 23], 'k-', linewidth=2)
    line25, = ax3.plot(strain[:, 24], relStress[:, 24], 'k-', linewidth=2)
    line26, = ax3.plot(strain[:, 25], relStress[:, 25], 'k-', linewidth=2)
    line27, = ax3.plot(strain[:, 26], relStress[:, 26], 'k-', linewidth=2)
    line28, = ax3.plot(strain[:, 27], relStress[:, 27], 'k-', linewidth=2)
    line29, = ax3.plot(strain[:, 28], relStress[:, 28], 'k-', linewidth=2)
    line30, = ax3.plot(strain[:, 29], relStress[:, 29], 'k-', linewidth=2)
    plt.title("30 Relative Stress/Strain Curves", fontsize=20)
    plt.xlabel(r'strain  $e$', fontsize=18)
    plt.ylabel(r'relative stress  $s - s_0$  (barye)', fontsize=18)

    # add the curves for collagen relative-entropy/strain plot
    ax4 = plt.subplot(2, 2, 4)
    # ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax4.plot(strain[:, 0], relEntropy[:, 0], 'k-', linewidth=2)
    line2, = ax4.plot(strain[:, 1], relEntropy[:, 1], 'k-', linewidth=2)
    line3, = ax4.plot(strain[:, 2], relEntropy[:, 2], 'k-', linewidth=2)
    line4, = ax4.plot(strain[:, 3], relEntropy[:, 3], 'k-', linewidth=2)
    line5, = ax4.plot(strain[:, 4], relEntropy[:, 4], 'k-', linewidth=2)
    line6, = ax4.plot(strain[:, 5], relEntropy[:, 5], 'k-', linewidth=2)
    line7, = ax4.plot(strain[:, 6], relEntropy[:, 6], 'k-', linewidth=2)
    line8, = ax4.plot(strain[:, 7], relEntropy[:, 7], 'k-', linewidth=2)
    line9, = ax4.plot(strain[:, 8], relEntropy[:, 8], 'k-', linewidth=2)
    line10, = ax4.plot(strain[:, 9], relEntropy[:, 9], 'k-', linewidth=2)
    line11, = ax4.plot(strain[:, 10], relEntropy[:, 10], 'k-', linewidth=2)
    line12, = ax4.plot(strain[:, 11], relEntropy[:, 11], 'k-', linewidth=2)
    line13, = ax4.plot(strain[:, 12], relEntropy[:, 12], 'k-', linewidth=2)
    line14, = ax4.plot(strain[:, 13], relEntropy[:, 13], 'k-', linewidth=2)
    line15, = ax4.plot(strain[:, 14], relEntropy[:, 14], 'k-', linewidth=2)
    line16, = ax4.plot(strain[:, 15], relEntropy[:, 15], 'k-', linewidth=2)
    line17, = ax4.plot(strain[:, 16], relEntropy[:, 16], 'k-', linewidth=2)
    line18, = ax4.plot(strain[:, 17], relEntropy[:, 17], 'k-', linewidth=2)
    line19, = ax4.plot(strain[:, 18], relEntropy[:, 18], 'k-', linewidth=2)
    line20, = ax4.plot(strain[:, 19], relEntropy[:, 19], 'k-', linewidth=2)
    line21, = ax4.plot(strain[:, 20], relEntropy[:, 20], 'k-', linewidth=2)
    line22, = ax4.plot(strain[:, 21], relEntropy[:, 21], 'k-', linewidth=2)
    line23, = ax4.plot(strain[:, 22], relEntropy[:, 22], 'k-', linewidth=2)
    line24, = ax4.plot(strain[:, 23], relEntropy[:, 23], 'k-', linewidth=2)
    line25, = ax4.plot(strain[:, 24], relEntropy[:, 24], 'k-', linewidth=2)
    line26, = ax4.plot(strain[:, 25], relEntropy[:, 25], 'k-', linewidth=2)
    line27, = ax4.plot(strain[:, 26], relEntropy[:, 26], 'k-', linewidth=2)
    line28, = ax4.plot(strain[:, 27], relEntropy[:, 27], 'k-', linewidth=2)
    line29, = ax4.plot(strain[:, 28], relEntropy[:, 28], 'k-', linewidth=2)
    line30, = ax4.plot(strain[:, 29], relEntropy[:, 29], 'k-', linewidth=2)
    plt.title("30 Relative Entropy/Strain Curves", fontsize=20)
    plt.xlabel(r'strain  $e$', fontsize=16)
    plt.ylabel(r'relative entropy density  $\eta - \eta_0$  (erg/g.K)',
               fontsize=16)

    plt.savefig('bioFibers5.jpg')

    plt.show()


run()
