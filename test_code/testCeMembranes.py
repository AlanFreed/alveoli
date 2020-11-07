#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ceMembranes import controlMembrane, ceMembrane
import math
from peceHE import PECE
# for creating graphics
from matplotlib import pyplot as plt
from matplotlib import ticker
# import materialProperties as mp
import numpy as np
from pylab import rcParams

"""
Created on Fri May 29 2020
Updated on Wed Jun 24 2020

A test file for the membrane constitutive response in file ceMembranes.py.

author: Prof. Alan Freed
"""

# assign the number of CE iterations in PE(CE)^m, with m = 0 -> PE method
m = 2
# assign the number of integration steps to be applied per experiment
N = 75
# assign the time variables used
t0 = 0.0
tN = 1.0
dt = (tN - t0) / N
# assign the temperature variables used
T0 = 37.0
TN = 37.0
# assign the number of curves to plot per experiment and number of experiments
curves = 30
experiments = 5
# assign number of variables
ctrlVars = 4
respVars = 4

def run():
    # create the data arrays used for plotting
    # stress/strain pair for uniform dilation
    xi = np.zeros((N+1, experiments, curves), dtype=float)
    pi = np.zeros((N+1, experiments, curves), dtype=float)
    # stress/strain pair for non-uniform squeeze
    epsilon = np.zeros((N+1, experiments, curves), dtype=float)
    sigma = np.zeros((N+1, experiments, curves), dtype=float)
    # stress/strain pair for non-uniform shear
    gamma = np.zeros((N+1, experiments, curves), dtype=float)
    tau = np.zeros((N+1, experiments, curves), dtype=float)

    # initialize a counter
    ruptured = np.zeros((experiments,), dtype=int)

    # uniform dilation experiment

    experiment = 0
    a0 = 1.0
    b0 = 1.0
    g0 = 0.0
    aN = a0 * math.exp(0.25)
    bN = b0 * math.exp(0.25)
    gN = g0
    for j in range(curves):
        ctrl = None
        resp = None
        solver = None
        # create and initialize the two control vectors
        eVec0 = np.zeros((ctrlVars,), dtype=float)
        xVec0 = np.zeros((ctrlVars,), dtype=float)
        xVec0[0] = T0
        xVec0[1] = a0
        xVec0[2] = b0
        xVec0[3] = g0
        # create the control object
        ctrl = controlMembrane(eVec0, xVec0, dt)
        # create the response object with random thickness assignment
        resp = ceMembrane()
        # create the integrator
        solver = PECE(ctrl, resp, m)
        # assign initial conditions for the plotting arrays
        e0 = solver.getE()
        y0 = solver.getYminusY0()
        xi[0, experiment, j] = e0[1]
        pi[0, experiment, j] = y0[1]
        epsilon[0, experiment, j] = e0[2]
        sigma[0, experiment, j] = y0[2]
        gamma[0, experiment, j] = e0[3]
        tau[0, experiment, j] = y0[3]
        # integrate the constitutive equation for this boundary value problem
        xVec = np.zeros((ctrlVars,), dtype=float)
        for i in range(1, N+1):
            xVec[0] = T0 + (TN - T0) * i / N
            xVec[1] = a0 + (aN - a0) * i / N
            xVec[2] = b0 + (bN - b0) * i / N
            xVec[3] = g0 + (gN - g0) * i / N
            solver.integrate(xVec)
            solver.advance()
            e = solver.getE()
            xi[i, experiment, j] = e[1]
            epsilon[i, experiment, j] = e[2]
            gamma[i, experiment, j] = e[3]
            y = solver.getYminusY0()
            pi[i, experiment, j] = y[1]
            sigma[i, experiment, j] = y[2]
            tau[i, experiment, j] = y[3]
        (hasRuptured,) = resp.isRuptured()
        if hasRuptured:
            ruptured[experiment] += 1

    # pure shear: experiments 1 & 2

    experiment = 1
    lamda0 = 1.0     # initial stretch of pure shear
    lamdaN = 1.107   # final stretch for pure shear => gamma_max = 0.2
    for j in range(curves):
        ctrl = None
        resp = None
        solver = None
        # create and initialize the two control vectors
        eVec0 = np.zeros((ctrlVars,), dtype=float)
        xVec0 = np.zeros((ctrlVars,), dtype=float)
        xVec0[0] = T0
        xVec0[1] = a0
        xVec0[2] = b0
        xVec0[3] = g0
        # create the control objects
        ctrlT = controlMembrane(eVec0, xVec0, dt)   # tension
        ctrlC = controlMembrane(eVec0, xVec0, dt)   # compression
        # create the response objects with random thickness assignments
        respT = ceMembrane()                        # tension
        respC = ceMembrane()                        # compression
        # create the integrators
        solverT = PECE(ctrlT, respT, m)             # tension
        solverC = PECE(ctrlC, respC, m)             # compression
        # assign initial conditions for the plotting arrays
        # tension: a > b
        e0 = solverT.getE()
        y0 = solverT.getYminusY0()
        xi[0, experiment, j] = e0[1]
        pi[0, experiment, j] = y0[1]
        epsilon[0, experiment, j] = e0[2]
        sigma[0, experiment, j] = y0[2]
        gamma[0, experiment, j] = e0[3]
        tau[0, experiment, j] = y0[3]
        # compression: b > a
        e0 = solverC.getE()
        y0 = solverC.getYminusY0()
        xi[0, experiment+1, j] = e0[1]
        pi[0, experiment+1, j] = y0[1]
        epsilon[0, experiment+1, j] = e0[2]
        sigma[0, experiment+1, j] = y0[2]
        gamma[0, experiment+1, j] = e0[3]
        tau[0, experiment+1, j] = y0[3]
        # integrate the constitutive equation for this boundary value problem
        xVec = np.zeros((ctrlVars,), dtype=float)
        for i in range(1, N+1):
            # elongation, case tension: a > 1, b < 1
            xVec[0] = T0 + (TN - T0) * i / N
            lamda = lamda0 + (lamdaN - lamda0) * i / N
            xVec[1] = math.sqrt(lamda**2 + 1.0 / lamda**2) / math.sqrt(2.0)
            xVec[2] = 1.0 / xVec[1]
            xVec[3] = ((lamda**2 - 1.0 / lamda**2) /
                       (lamda**2 + 1.0 / lamda**2))
            solverT.integrate(xVec)
            solverT.advance()
            e = solverT.getE()
            xi[i, experiment, j] = e[1]
            epsilon[i, experiment, j] = e[2]
            gamma[i, experiment, j] = e[3]
            y = solverT.getYminusY0()
            pi[i, experiment, j] = y[1]
            sigma[i, experiment, j] = y[2]
            tau[i, experiment, j] = y[3]
            # elongation case compression: a < 1, b > 1
            xVec[3] = ((1.0 / lamda**2 - lamda**2) /
                       (lamda**2 + 1.0 / lamda**2))
            solverC.integrate(xVec)
            solverC.advance()
            e = solverC.getE()
            xi[i, experiment+1, j] = e[1]
            epsilon[i, experiment+1, j] = e[2]
            gamma[i, experiment+1, j] = e[3]
            y = solverC.getYminusY0()
            pi[i, experiment+1, j] = y[1]
            sigma[i, experiment+1, j] = y[2]
            tau[i, experiment+1, j] = y[3]
        (hasRupturedT,) = respT.isRuptured()
        (hasRupturedC,) = respC.isRuptured()
        if hasRupturedT:
            ruptured[experiment] += 1
        if hasRupturedC:
            ruptured[experiment+1] += 1

    # non-uniform shear: experiments 3 & 4

    experiment = 3
    a0 = 1.0
    b0 = 1.0
    g0 = 0.0
    aN = 1.0
    bN = 1.0
    gN = 0.2
    for j in range(curves):
        ctrl = None
        resp = None
        solver = None
        # create and initialize the two control vectors
        eVec0 = np.zeros((ctrlVars,), dtype=float)
        xVec0 = np.zeros((ctrlVars,), dtype=float)
        xVec0[0] = T0
        xVec0[1] = a0
        xVec0[2] = b0
        xVec0[3] = g0
        # create the control objects
        ctrlT = controlMembrane(eVec0, xVec0, dt)   # tension
        ctrlC = controlMembrane(eVec0, xVec0, dt)   # compression
        # create the response objects with random thickness assignments
        respT = ceMembrane()                        # tension
        respC = ceMembrane()                        # compression
        # create the integrators
        solverT = PECE(ctrlT, respT, m)             # tension
        solverC = PECE(ctrlC, respC, m)             # compression
        # assign initial conditions for the plotting arrays
        # tension: gamma > 0
        e0 = solverT.getE()
        y0 = solverT.getYminusY0()
        xi[0, experiment, j] = e0[1]
        pi[0, experiment, j] = y0[1]
        epsilon[0, experiment, j] = e0[2]
        sigma[0, experiment, j] = y0[2]
        gamma[0, experiment, j] = e0[3]
        tau[0, experiment, j] = y0[3]
        # compression: b > a
        e0 = solverC.getE()
        y0 = solverC.getYminusY0()
        xi[0, experiment+1, j] = e0[1]
        pi[0, experiment+1, j] = y0[1]
        epsilon[0, experiment+1, j] = e0[2]
        sigma[0, experiment+1, j] = y0[2]
        gamma[0, experiment+1, j] = e0[3]
        tau[0, experiment+1, j] = y0[3]
        # integrate the constitutive equation for this boundary value problem
        xVec = np.zeros((ctrlVars,), dtype=float)
        for i in range(1, N+1):
            # elongation, case tension: a > 1, b < 1
            xVec[0] = T0 + (TN - T0) * i / N
            xVec[1] = a0 + (aN - a0) * i / N
            xVec[2] = b0 + (bN - b0) * i / N
            xVec[3] = g0 + (gN - g0) * i / N
            solverT.integrate(xVec)
            solverT.advance()
            e = solverT.getE()
            xi[i, experiment, j] = e[1]
            epsilon[i, experiment, j] = e[2]
            gamma[i, experiment, j] = e[3]
            y = solverT.getYminusY0()
            pi[i, experiment, j] = y[1]
            sigma[i, experiment, j] = y[2]
            tau[i, experiment, j] = y[3]
            # elongation case compression: a < 1, b > 1
            xVec[3] = -xVec[3]
            solverC.integrate(xVec)
            xVec[3] = -xVec[3]
            solverC.advance()
            e = solverC.getE()
            xi[i, experiment+1, j] = e[1]
            epsilon[i, experiment+1, j] = e[2]
            gamma[i, experiment+1, j] = e[3]
            y = solverC.getYminusY0()
            pi[i, experiment+1, j] = y[1]
            sigma[i, experiment+1, j] = y[2]
            tau[i, experiment+1, j] = y[3]
        (hasRupturedT,) = respT.isRuptured()
        (hasRupturedC,) = respC.isRuptured()
        if hasRupturedT:
            ruptured[experiment] += 1
        if hasRupturedC:
            ruptured[experiment+1] += 1

    print("Out of {} septal membranes tested per condition, ".format(curves) +
          "\n   {} membranes ruptured during dilation,".format(ruptured[0]) +
          "\n   {}, {} ".format(ruptured[1], ruptured[2]) +
          " membranes ruptured during pure shear (tension vs. compression)," +
          "\n   {}, {} ".format(ruptured[3], ruptured[4]) +
          " membranes ruptured during simple shear (tension vs. compression).")

    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    plt.figure(1)
    rcParams['figure.figsize'] = 24, 22

    # the dilation experiment

    ax1 = plt.subplot(3, 3, 1)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax1.plot(xi[:, 0, 0], pi[:, 0, 0], 'k-', linewidth=2)
    line2, = ax1.plot(xi[:, 0, 1], pi[:, 0, 1], 'k-', linewidth=2)
    line3, = ax1.plot(xi[:, 0, 2], pi[:, 0, 2], 'k-', linewidth=2)
    line4, = ax1.plot(xi[:, 0, 3], pi[:, 0, 3], 'k-', linewidth=2)
    line5, = ax1.plot(xi[:, 0, 4], pi[:, 0, 4], 'k-', linewidth=2)
    line6, = ax1.plot(xi[:, 0, 5], pi[:, 0, 5], 'k-', linewidth=2)
    line7, = ax1.plot(xi[:, 0, 6], pi[:, 0, 6], 'k-', linewidth=2)
    line8, = ax1.plot(xi[:, 0, 7], pi[:, 0, 7], 'k-', linewidth=2)
    line9, = ax1.plot(xi[:, 0, 8], pi[:, 0, 8], 'k-', linewidth=2)
    line10, = ax1.plot(xi[:, 0, 9], pi[:, 0, 9], 'k-', linewidth=2)
    line11, = ax1.plot(xi[:, 0, 10], pi[:, 0, 10], 'k-', linewidth=2)
    line12, = ax1.plot(xi[:, 0, 11], pi[:, 0, 11], 'k-', linewidth=2)
    line13, = ax1.plot(xi[:, 0, 12], pi[:, 0, 12], 'k-', linewidth=2)
    line14, = ax1.plot(xi[:, 0, 13], pi[:, 0, 13], 'k-', linewidth=2)
    line15, = ax1.plot(xi[:, 0, 14], pi[:, 0, 14], 'k-', linewidth=2)
    line16, = ax1.plot(xi[:, 0, 15], pi[:, 0, 15], 'k-', linewidth=2)
    line17, = ax1.plot(xi[:, 0, 16], pi[:, 0, 16], 'k-', linewidth=2)
    line18, = ax1.plot(xi[:, 0, 17], pi[:, 0, 17], 'k-', linewidth=2)
    line19, = ax1.plot(xi[:, 0, 18], pi[:, 0, 18], 'k-', linewidth=2)
    line20, = ax1.plot(xi[:, 0, 19], pi[:, 0, 19], 'k-', linewidth=2)
    line21, = ax1.plot(xi[:, 0, 20], pi[:, 0, 20], 'k-', linewidth=2)
    line22, = ax1.plot(xi[:, 0, 21], pi[:, 0, 21], 'k-', linewidth=2)
    line23, = ax1.plot(xi[:, 0, 22], pi[:, 0, 22], 'k-', linewidth=2)
    line24, = ax1.plot(xi[:, 0, 23], pi[:, 0, 23], 'k-', linewidth=2)
    line25, = ax1.plot(xi[:, 0, 24], pi[:, 0, 24], 'k-', linewidth=2)
    line26, = ax1.plot(xi[:, 0, 25], pi[:, 0, 25], 'k-', linewidth=2)
    line27, = ax1.plot(xi[:, 0, 26], pi[:, 0, 26], 'k-', linewidth=2)
    line28, = ax1.plot(xi[:, 0, 27], pi[:, 0, 27], 'k-', linewidth=2)
    line29, = ax1.plot(xi[:, 0, 28], pi[:, 0, 28], 'k-', linewidth=2)
    line30, = ax1.plot(xi[:, 0, 29], pi[:, 0, 29], 'k-', linewidth=2)
    plt.title("Thirty", fontsize=20)
    plt.xlabel(r'dilation:  $\xi$', fontsize=18)
    plt.ylabel(r'surface tension:  $s^{\pi} - s_0^{\pi}$  (dynes)',
               fontsize=18)

    # add the curves
    ax2 = plt.subplot(3, 3, 2)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax2.plot(epsilon[:, 0, 0], sigma[:, 0, 0], 'k-', linewidth=2)
    line2, = ax2.plot(epsilon[:, 0, 1], sigma[:, 0, 1], 'k-', linewidth=2)
    line3, = ax2.plot(epsilon[:, 0, 2], sigma[:, 0, 2], 'k-', linewidth=2)
    line4, = ax2.plot(epsilon[:, 0, 3], sigma[:, 0, 3], 'k-', linewidth=2)
    line5, = ax2.plot(epsilon[:, 0, 4], sigma[:, 0, 4], 'k-', linewidth=2)
    line6, = ax2.plot(epsilon[:, 0, 5], sigma[:, 0, 5], 'k-', linewidth=2)
    line7, = ax2.plot(epsilon[:, 0, 6], sigma[:, 0, 6], 'k-', linewidth=2)
    line8, = ax2.plot(epsilon[:, 0, 7], sigma[:, 0, 7], 'k-', linewidth=2)
    line9, = ax2.plot(epsilon[:, 0, 8], sigma[:, 0, 8], 'k-', linewidth=2)
    line10, = ax2.plot(epsilon[:, 0, 9], sigma[:, 0, 9], 'k-', linewidth=2)
    line11, = ax2.plot(epsilon[:, 0, 10], sigma[:, 0, 10], 'k-', linewidth=2)
    line12, = ax2.plot(epsilon[:, 0, 11], sigma[:, 0, 11], 'k-', linewidth=2)
    line13, = ax2.plot(epsilon[:, 0, 12], sigma[:, 0, 12], 'k-', linewidth=2)
    line14, = ax2.plot(epsilon[:, 0, 13], sigma[:, 0, 13], 'k-', linewidth=2)
    line15, = ax2.plot(epsilon[:, 0, 14], sigma[:, 0, 14], 'k-', linewidth=2)
    line16, = ax2.plot(epsilon[:, 0, 15], sigma[:, 0, 15], 'k-', linewidth=2)
    line17, = ax2.plot(epsilon[:, 0, 16], sigma[:, 0, 16], 'k-', linewidth=2)
    line18, = ax2.plot(epsilon[:, 0, 17], sigma[:, 0, 17], 'k-', linewidth=2)
    line19, = ax2.plot(epsilon[:, 0, 18], sigma[:, 0, 18], 'k-', linewidth=2)
    line20, = ax2.plot(epsilon[:, 0, 19], sigma[:, 0, 19], 'k-', linewidth=2)
    line21, = ax2.plot(epsilon[:, 0, 20], sigma[:, 0, 20], 'k-', linewidth=2)
    line22, = ax2.plot(epsilon[:, 0, 21], sigma[:, 0, 21], 'k-', linewidth=2)
    line23, = ax2.plot(epsilon[:, 0, 22], sigma[:, 0, 22], 'k-', linewidth=2)
    line24, = ax2.plot(epsilon[:, 0, 23], sigma[:, 0, 23], 'k-', linewidth=2)
    line25, = ax2.plot(epsilon[:, 0, 24], sigma[:, 0, 24], 'k-', linewidth=2)
    line26, = ax2.plot(epsilon[:, 0, 25], sigma[:, 0, 25], 'k-', linewidth=2)
    line27, = ax2.plot(epsilon[:, 0, 26], sigma[:, 0, 26], 'k-', linewidth=2)
    line28, = ax2.plot(epsilon[:, 0, 27], sigma[:, 0, 27], 'k-', linewidth=2)
    line29, = ax2.plot(epsilon[:, 0, 28], sigma[:, 0, 28], 'k-', linewidth=2)
    line30, = ax2.plot(epsilon[:, 0, 29], sigma[:, 0, 29], 'k-', linewidth=2)
    plt.title("Dilation", fontsize=20)
    plt.xlabel(r'squeeze strain:  $\epsilon$', fontsize=16)
    plt.ylabel(r'squeeze stress:  $s^{\sigma} - s_0^{\sigma}$  (dynes)',
               fontsize=16)

    # add the curves
    ax3 = plt.subplot(3, 3, 3)
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax3.plot(gamma[:, 0, 0], tau[:, 0, 0], 'k-', linewidth=2)
    line2, = ax3.plot(gamma[:, 0, 1], tau[:, 0, 1], 'k-', linewidth=2)
    line3, = ax3.plot(gamma[:, 0, 2], tau[:, 0, 2], 'k-', linewidth=2)
    line4, = ax3.plot(gamma[:, 0, 3], tau[:, 0, 3], 'k-', linewidth=2)
    line5, = ax3.plot(gamma[:, 0, 4], tau[:, 0, 4], 'k-', linewidth=2)
    line6, = ax3.plot(gamma[:, 0, 5], tau[:, 0, 5], 'k-', linewidth=2)
    line7, = ax3.plot(gamma[:, 0, 6], tau[:, 0, 6], 'k-', linewidth=2)
    line8, = ax3.plot(gamma[:, 0, 7], tau[:, 0, 7], 'k-', linewidth=2)
    line9, = ax3.plot(gamma[:, 0, 8], tau[:, 0, 8], 'k-', linewidth=2)
    line10, = ax3.plot(gamma[:, 0, 9], tau[:, 0, 9], 'k-', linewidth=2)
    line11, = ax3.plot(gamma[:, 0, 10], tau[:, 0, 10], 'k-', linewidth=2)
    line12, = ax3.plot(gamma[:, 0, 11], tau[:, 0, 11], 'k-', linewidth=2)
    line13, = ax3.plot(gamma[:, 0, 12], tau[:, 0, 12], 'k-', linewidth=2)
    line14, = ax3.plot(gamma[:, 0, 13], tau[:, 0, 13], 'k-', linewidth=2)
    line15, = ax3.plot(gamma[:, 0, 14], tau[:, 0, 14], 'k-', linewidth=2)
    line16, = ax3.plot(gamma[:, 0, 15], tau[:, 0, 15], 'k-', linewidth=2)
    line17, = ax3.plot(gamma[:, 0, 16], tau[:, 0, 16], 'k-', linewidth=2)
    line18, = ax3.plot(gamma[:, 0, 17], tau[:, 0, 17], 'k-', linewidth=2)
    line19, = ax3.plot(gamma[:, 0, 18], tau[:, 0, 18], 'k-', linewidth=2)
    line20, = ax3.plot(gamma[:, 0, 19], tau[:, 0, 19], 'k-', linewidth=2)
    line21, = ax3.plot(gamma[:, 0, 20], tau[:, 0, 20], 'k-', linewidth=2)
    line22, = ax3.plot(gamma[:, 0, 21], tau[:, 0, 21], 'k-', linewidth=2)
    line23, = ax3.plot(gamma[:, 0, 22], tau[:, 0, 22], 'k-', linewidth=2)
    line24, = ax3.plot(gamma[:, 0, 23], tau[:, 0, 23], 'k-', linewidth=2)
    line25, = ax3.plot(gamma[:, 0, 24], tau[:, 0, 24], 'k-', linewidth=2)
    line26, = ax3.plot(gamma[:, 0, 25], tau[:, 0, 25], 'k-', linewidth=2)
    line27, = ax3.plot(gamma[:, 0, 26], tau[:, 0, 26], 'k-', linewidth=2)
    line28, = ax3.plot(gamma[:, 0, 27], tau[:, 0, 27], 'k-', linewidth=2)
    line29, = ax3.plot(gamma[:, 0, 28], tau[:, 0, 28], 'k-', linewidth=2)
    line30, = ax3.plot(gamma[:, 0, 29], tau[:, 0, 29], 'k-', linewidth=2)
    plt.title("Experiments", fontsize=20)
    plt.xlabel(r'shear strain:  $\gamma$', fontsize=16)
    plt.ylabel(r'shear stress:  $s^{\tau} - s_0^{\tau}$  (dynes)', fontsize=16)

    # the pure shear experiment

    ax4 = plt.subplot(3, 3, 4)
    ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax4.plot(xi[:, 1, 0], pi[:, 1, 0], 'k-', linewidth=2)
    line2, = ax4.plot(xi[:, 1, 1], pi[:, 1, 1], 'k-', linewidth=2)
    line3, = ax4.plot(xi[:, 1, 2], pi[:, 1, 2], 'k-', linewidth=2)
    line4, = ax4.plot(xi[:, 1, 3], pi[:, 1, 3], 'k-', linewidth=2)
    line5, = ax4.plot(xi[:, 1, 4], pi[:, 1, 4], 'k-', linewidth=2)
    line6, = ax4.plot(xi[:, 1, 5], pi[:, 1, 5], 'k-', linewidth=2)
    line7, = ax4.plot(xi[:, 1, 6], pi[:, 1, 6], 'k-', linewidth=2)
    line8, = ax4.plot(xi[:, 1, 7], pi[:, 1, 7], 'k-', linewidth=2)
    line9, = ax4.plot(xi[:, 1, 8], pi[:, 1, 8], 'k-', linewidth=2)
    line10, = ax4.plot(xi[:, 1, 9], pi[:, 1, 9], 'k-', linewidth=2)
    line11, = ax4.plot(xi[:, 1, 10], pi[:, 1, 10], 'k-', linewidth=2)
    line12, = ax4.plot(xi[:, 1, 11], pi[:, 1, 11], 'k-', linewidth=2)
    line13, = ax4.plot(xi[:, 1, 12], pi[:, 1, 12], 'k-', linewidth=2)
    line14, = ax4.plot(xi[:, 1, 13], pi[:, 1, 13], 'k-', linewidth=2)
    line15, = ax4.plot(xi[:, 1, 14], pi[:, 1, 14], 'k-', linewidth=2)
    line16, = ax4.plot(xi[:, 1, 15], pi[:, 1, 15], 'k-', linewidth=2)
    line17, = ax4.plot(xi[:, 1, 16], pi[:, 1, 16], 'k-', linewidth=2)
    line18, = ax4.plot(xi[:, 1, 17], pi[:, 1, 17], 'k-', linewidth=2)
    line19, = ax4.plot(xi[:, 1, 18], pi[:, 1, 18], 'k-', linewidth=2)
    line20, = ax4.plot(xi[:, 1, 19], pi[:, 1, 19], 'k-', linewidth=2)
    line21, = ax4.plot(xi[:, 1, 20], pi[:, 1, 20], 'k-', linewidth=2)
    line22, = ax4.plot(xi[:, 1, 21], pi[:, 1, 21], 'k-', linewidth=2)
    line23, = ax4.plot(xi[:, 1, 22], pi[:, 1, 22], 'k-', linewidth=2)
    line24, = ax4.plot(xi[:, 1, 23], pi[:, 1, 23], 'k-', linewidth=2)
    line25, = ax4.plot(xi[:, 1, 24], pi[:, 1, 24], 'k-', linewidth=2)
    line26, = ax4.plot(xi[:, 1, 25], pi[:, 1, 25], 'k-', linewidth=2)
    line27, = ax4.plot(xi[:, 1, 26], pi[:, 1, 26], 'k-', linewidth=2)
    line28, = ax4.plot(xi[:, 1, 27], pi[:, 1, 27], 'k-', linewidth=2)
    line29, = ax4.plot(xi[:, 1, 28], pi[:, 1, 28], 'k-', linewidth=2)
    line30, = ax4.plot(xi[:, 1, 29], pi[:, 1, 29], 'k-', linewidth=2)
    line31, = ax4.plot(xi[:, 2, 0], pi[:, 2, 0], 'k-', linewidth=2)
    line32, = ax4.plot(xi[:, 2, 1], pi[:, 2, 1], 'k-', linewidth=2)
    line33, = ax4.plot(xi[:, 2, 2], pi[:, 2, 2], 'k-', linewidth=2)
    line34, = ax4.plot(xi[:, 2, 3], pi[:, 2, 3], 'k-', linewidth=2)
    line35, = ax4.plot(xi[:, 2, 4], pi[:, 2, 4], 'k-', linewidth=2)
    line36, = ax4.plot(xi[:, 2, 5], pi[:, 2, 5], 'k-', linewidth=2)
    line37, = ax4.plot(xi[:, 2, 6], pi[:, 2, 6], 'k-', linewidth=2)
    line38, = ax4.plot(xi[:, 2, 7], pi[:, 2, 7], 'k-', linewidth=2)
    line39, = ax4.plot(xi[:, 2, 8], pi[:, 2, 8], 'k-', linewidth=2)
    line40, = ax4.plot(xi[:, 2, 9], pi[:, 2, 9], 'k-', linewidth=2)
    line41, = ax4.plot(xi[:, 2, 10], pi[:, 2, 10], 'k-', linewidth=2)
    line42, = ax4.plot(xi[:, 2, 11], pi[:, 2, 11], 'k-', linewidth=2)
    line43, = ax4.plot(xi[:, 2, 12], pi[:, 2, 12], 'k-', linewidth=2)
    line44, = ax4.plot(xi[:, 2, 13], pi[:, 2, 13], 'k-', linewidth=2)
    line45, = ax4.plot(xi[:, 2, 14], pi[:, 2, 14], 'k-', linewidth=2)
    line46, = ax4.plot(xi[:, 2, 15], pi[:, 2, 15], 'k-', linewidth=2)
    line47, = ax4.plot(xi[:, 2, 16], pi[:, 2, 16], 'k-', linewidth=2)
    line48, = ax4.plot(xi[:, 2, 17], pi[:, 2, 17], 'k-', linewidth=2)
    line49, = ax4.plot(xi[:, 2, 18], pi[:, 2, 18], 'k-', linewidth=2)
    line50, = ax4.plot(xi[:, 2, 19], pi[:, 2, 19], 'k-', linewidth=2)
    line51, = ax4.plot(xi[:, 2, 20], pi[:, 2, 20], 'k-', linewidth=2)
    line52, = ax4.plot(xi[:, 2, 21], pi[:, 2, 21], 'k-', linewidth=2)
    line53, = ax4.plot(xi[:, 2, 22], pi[:, 2, 22], 'k-', linewidth=2)
    line54, = ax4.plot(xi[:, 2, 23], pi[:, 2, 23], 'k-', linewidth=2)
    line55, = ax4.plot(xi[:, 2, 24], pi[:, 2, 24], 'k-', linewidth=2)
    line56, = ax4.plot(xi[:, 2, 25], pi[:, 2, 25], 'k-', linewidth=2)
    line57, = ax4.plot(xi[:, 2, 26], pi[:, 2, 26], 'k-', linewidth=2)
    line58, = ax4.plot(xi[:, 2, 27], pi[:, 2, 27], 'k-', linewidth=2)
    line59, = ax4.plot(xi[:, 2, 28], pi[:, 2, 28], 'k-', linewidth=2)
    line60, = ax4.plot(xi[:, 2, 29], pi[:, 2, 29], 'k-', linewidth=2)
    plt.title("Thirty", fontsize=20)
    plt.xlabel(r'dilation:  $\xi$', fontsize=18)
    plt.ylabel(r'surface tension:  $s^{\pi} - s_0^{\pi}$  (dynes)',
               fontsize=18)

    # add the curves
    ax5 = plt.subplot(3, 3, 5)
    ax5.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax5.plot(epsilon[:, 1, 0], sigma[:, 1, 0], 'k-', linewidth=2)
    line2, = ax5.plot(epsilon[:, 1, 1], sigma[:, 1, 1], 'k-', linewidth=2)
    line3, = ax5.plot(epsilon[:, 1, 2], sigma[:, 1, 2], 'k-', linewidth=2)
    line4, = ax5.plot(epsilon[:, 1, 3], sigma[:, 1, 3], 'k-', linewidth=2)
    line5, = ax5.plot(epsilon[:, 1, 4], sigma[:, 1, 4], 'k-', linewidth=2)
    line6, = ax5.plot(epsilon[:, 1, 5], sigma[:, 1, 5], 'k-', linewidth=2)
    line7, = ax5.plot(epsilon[:, 1, 6], sigma[:, 1, 6], 'k-', linewidth=2)
    line8, = ax5.plot(epsilon[:, 1, 7], sigma[:, 1, 7], 'k-', linewidth=2)
    line9, = ax5.plot(epsilon[:, 1, 8], sigma[:, 1, 8], 'k-', linewidth=2)
    line10, = ax5.plot(epsilon[:, 1, 9], sigma[:, 1, 9], 'k-', linewidth=2)
    line11, = ax5.plot(epsilon[:, 1, 10], sigma[:, 1, 10], 'k-', linewidth=2)
    line12, = ax5.plot(epsilon[:, 1, 11], sigma[:, 1, 11], 'k-', linewidth=2)
    line13, = ax5.plot(epsilon[:, 1, 12], sigma[:, 1, 12], 'k-', linewidth=2)
    line14, = ax5.plot(epsilon[:, 1, 13], sigma[:, 1, 13], 'k-', linewidth=2)
    line15, = ax5.plot(epsilon[:, 1, 14], sigma[:, 1, 14], 'k-', linewidth=2)
    line16, = ax5.plot(epsilon[:, 1, 15], sigma[:, 1, 15], 'k-', linewidth=2)
    line17, = ax5.plot(epsilon[:, 1, 16], sigma[:, 1, 16], 'k-', linewidth=2)
    line18, = ax5.plot(epsilon[:, 1, 17], sigma[:, 1, 17], 'k-', linewidth=2)
    line19, = ax5.plot(epsilon[:, 1, 18], sigma[:, 1, 18], 'k-', linewidth=2)
    line20, = ax5.plot(epsilon[:, 1, 19], sigma[:, 1, 19], 'k-', linewidth=2)
    line21, = ax5.plot(epsilon[:, 1, 20], sigma[:, 1, 20], 'k-', linewidth=2)
    line22, = ax5.plot(epsilon[:, 1, 21], sigma[:, 1, 21], 'k-', linewidth=2)
    line23, = ax5.plot(epsilon[:, 1, 22], sigma[:, 1, 22], 'k-', linewidth=2)
    line24, = ax5.plot(epsilon[:, 1, 23], sigma[:, 1, 23], 'k-', linewidth=2)
    line25, = ax5.plot(epsilon[:, 1, 24], sigma[:, 1, 24], 'k-', linewidth=2)
    line26, = ax5.plot(epsilon[:, 1, 25], sigma[:, 1, 25], 'k-', linewidth=2)
    line27, = ax5.plot(epsilon[:, 1, 26], sigma[:, 1, 26], 'k-', linewidth=2)
    line28, = ax5.plot(epsilon[:, 1, 27], sigma[:, 1, 27], 'k-', linewidth=2)
    line29, = ax5.plot(epsilon[:, 1, 28], sigma[:, 1, 28], 'k-', linewidth=2)
    line30, = ax5.plot(epsilon[:, 1, 29], sigma[:, 1, 29], 'k-', linewidth=2)
    line31, = ax5.plot(epsilon[:, 2, 0], sigma[:, 2, 0], 'k-', linewidth=2)
    line32, = ax5.plot(epsilon[:, 2, 1], sigma[:, 2, 1], 'k-', linewidth=2)
    line33, = ax5.plot(epsilon[:, 2, 2], sigma[:, 2, 2], 'k-', linewidth=2)
    line34, = ax5.plot(epsilon[:, 2, 3], sigma[:, 2, 3], 'k-', linewidth=2)
    line35, = ax5.plot(epsilon[:, 2, 4], sigma[:, 2, 4], 'k-', linewidth=2)
    line36, = ax5.plot(epsilon[:, 2, 5], sigma[:, 2, 5], 'k-', linewidth=2)
    line37, = ax5.plot(epsilon[:, 2, 6], sigma[:, 2, 6], 'k-', linewidth=2)
    line38, = ax5.plot(epsilon[:, 2, 7], sigma[:, 2, 7], 'k-', linewidth=2)
    line39, = ax5.plot(epsilon[:, 2, 8], sigma[:, 2, 8], 'k-', linewidth=2)
    line40, = ax5.plot(epsilon[:, 2, 9], sigma[:, 2, 9], 'k-', linewidth=2)
    line41, = ax5.plot(epsilon[:, 2, 10], sigma[:, 2, 10], 'k-', linewidth=2)
    line42, = ax5.plot(epsilon[:, 2, 11], sigma[:, 2, 11], 'k-', linewidth=2)
    line43, = ax5.plot(epsilon[:, 2, 12], sigma[:, 2, 12], 'k-', linewidth=2)
    line44, = ax5.plot(epsilon[:, 2, 13], sigma[:, 2, 13], 'k-', linewidth=2)
    line45, = ax5.plot(epsilon[:, 2, 14], sigma[:, 2, 14], 'k-', linewidth=2)
    line46, = ax5.plot(epsilon[:, 2, 15], sigma[:, 2, 15], 'k-', linewidth=2)
    line47, = ax5.plot(epsilon[:, 2, 16], sigma[:, 2, 16], 'k-', linewidth=2)
    line48, = ax5.plot(epsilon[:, 2, 17], sigma[:, 2, 17], 'k-', linewidth=2)
    line49, = ax5.plot(epsilon[:, 2, 18], sigma[:, 2, 18], 'k-', linewidth=2)
    line50, = ax5.plot(epsilon[:, 2, 19], sigma[:, 2, 19], 'k-', linewidth=2)
    line51, = ax5.plot(epsilon[:, 2, 20], sigma[:, 2, 20], 'k-', linewidth=2)
    line52, = ax5.plot(epsilon[:, 2, 21], sigma[:, 2, 21], 'k-', linewidth=2)
    line53, = ax5.plot(epsilon[:, 2, 22], sigma[:, 2, 22], 'k-', linewidth=2)
    line54, = ax5.plot(epsilon[:, 2, 23], sigma[:, 2, 23], 'k-', linewidth=2)
    line55, = ax5.plot(epsilon[:, 2, 24], sigma[:, 2, 24], 'k-', linewidth=2)
    line56, = ax5.plot(epsilon[:, 2, 25], sigma[:, 2, 25], 'k-', linewidth=2)
    line57, = ax5.plot(epsilon[:, 2, 26], sigma[:, 2, 26], 'k-', linewidth=2)
    line58, = ax5.plot(epsilon[:, 2, 27], sigma[:, 2, 27], 'k-', linewidth=2)
    line59, = ax5.plot(epsilon[:, 2, 28], sigma[:, 2, 28], 'k-', linewidth=2)
    line60, = ax5.plot(epsilon[:, 2, 29], sigma[:, 2, 29], 'k-', linewidth=2)
    plt.title("Pure Shear", fontsize=20)
    plt.xlabel(r'squeeze strain:  $\epsilon$', fontsize=16)
    plt.ylabel(r'squeeze stress:  $s^{\sigma} - s_0^{\sigma}$  (dynes)',
               fontsize=16)

    # add the curves
    ax6 = plt.subplot(3, 3, 6)
    ax6.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax6.plot(gamma[:, 1, 0], tau[:, 1, 0], 'k-', linewidth=2)
    line2, = ax6.plot(gamma[:, 1, 1], tau[:, 1, 1], 'k-', linewidth=2)
    line3, = ax6.plot(gamma[:, 1, 2], tau[:, 1, 2], 'k-', linewidth=2)
    line4, = ax6.plot(gamma[:, 1, 3], tau[:, 1, 3], 'k-', linewidth=2)
    line5, = ax6.plot(gamma[:, 1, 4], tau[:, 1, 4], 'k-', linewidth=2)
    line6, = ax6.plot(gamma[:, 1, 5], tau[:, 1, 5], 'k-', linewidth=2)
    line7, = ax6.plot(gamma[:, 1, 6], tau[:, 1, 6], 'k-', linewidth=2)
    line8, = ax6.plot(gamma[:, 1, 7], tau[:, 1, 7], 'k-', linewidth=2)
    line9, = ax6.plot(gamma[:, 1, 8], tau[:, 1, 8], 'k-', linewidth=2)
    line10, = ax6.plot(gamma[:, 1, 9], tau[:, 1, 9], 'k-', linewidth=2)
    line11, = ax6.plot(gamma[:, 1, 10], tau[:, 1, 10], 'k-', linewidth=2)
    line12, = ax6.plot(gamma[:, 1, 11], tau[:, 1, 11], 'k-', linewidth=2)
    line13, = ax6.plot(gamma[:, 1, 12], tau[:, 1, 12], 'k-', linewidth=2)
    line14, = ax6.plot(gamma[:, 1, 13], tau[:, 1, 13], 'k-', linewidth=2)
    line15, = ax6.plot(gamma[:, 1, 14], tau[:, 1, 14], 'k-', linewidth=2)
    line16, = ax6.plot(gamma[:, 1, 15], tau[:, 1, 15], 'k-', linewidth=2)
    line17, = ax6.plot(gamma[:, 1, 16], tau[:, 1, 16], 'k-', linewidth=2)
    line18, = ax6.plot(gamma[:, 1, 17], tau[:, 1, 17], 'k-', linewidth=2)
    line19, = ax6.plot(gamma[:, 1, 18], tau[:, 1, 18], 'k-', linewidth=2)
    line20, = ax6.plot(gamma[:, 1, 19], tau[:, 1, 19], 'k-', linewidth=2)
    line21, = ax6.plot(gamma[:, 1, 20], tau[:, 1, 20], 'k-', linewidth=2)
    line22, = ax6.plot(gamma[:, 1, 21], tau[:, 1, 21], 'k-', linewidth=2)
    line23, = ax6.plot(gamma[:, 1, 22], tau[:, 1, 22], 'k-', linewidth=2)
    line24, = ax6.plot(gamma[:, 1, 23], tau[:, 1, 23], 'k-', linewidth=2)
    line25, = ax6.plot(gamma[:, 1, 24], tau[:, 1, 24], 'k-', linewidth=2)
    line26, = ax6.plot(gamma[:, 1, 25], tau[:, 1, 25], 'k-', linewidth=2)
    line27, = ax6.plot(gamma[:, 1, 26], tau[:, 1, 26], 'k-', linewidth=2)
    line28, = ax6.plot(gamma[:, 1, 27], tau[:, 1, 27], 'k-', linewidth=2)
    line29, = ax6.plot(gamma[:, 1, 28], tau[:, 1, 28], 'k-', linewidth=2)
    line30, = ax6.plot(gamma[:, 1, 29], tau[:, 1, 29], 'k-', linewidth=2)
    line31, = ax6.plot(gamma[:, 2, 0], tau[:, 2, 0], 'k-', linewidth=2)
    line32, = ax6.plot(gamma[:, 2, 1], tau[:, 2, 1], 'k-', linewidth=2)
    line33, = ax6.plot(gamma[:, 2, 2], tau[:, 2, 2], 'k-', linewidth=2)
    line34, = ax6.plot(gamma[:, 2, 3], tau[:, 2, 3], 'k-', linewidth=2)
    line35, = ax6.plot(gamma[:, 2, 4], tau[:, 2, 4], 'k-', linewidth=2)
    line36, = ax6.plot(gamma[:, 2, 5], tau[:, 2, 5], 'k-', linewidth=2)
    line37, = ax6.plot(gamma[:, 2, 6], tau[:, 2, 6], 'k-', linewidth=2)
    line38, = ax6.plot(gamma[:, 2, 7], tau[:, 2, 7], 'k-', linewidth=2)
    line39, = ax6.plot(gamma[:, 2, 8], tau[:, 2, 8], 'k-', linewidth=2)
    line40, = ax6.plot(gamma[:, 2, 9], tau[:, 2, 9], 'k-', linewidth=2)
    line41, = ax6.plot(gamma[:, 2, 10], tau[:, 2, 10], 'k-', linewidth=2)
    line42, = ax6.plot(gamma[:, 2, 11], tau[:, 2, 11], 'k-', linewidth=2)
    line43, = ax6.plot(gamma[:, 2, 12], tau[:, 2, 12], 'k-', linewidth=2)
    line44, = ax6.plot(gamma[:, 2, 13], tau[:, 2, 13], 'k-', linewidth=2)
    line45, = ax6.plot(gamma[:, 2, 14], tau[:, 2, 14], 'k-', linewidth=2)
    line46, = ax6.plot(gamma[:, 2, 15], tau[:, 2, 15], 'k-', linewidth=2)
    line47, = ax6.plot(gamma[:, 2, 16], tau[:, 2, 16], 'k-', linewidth=2)
    line48, = ax6.plot(gamma[:, 2, 17], tau[:, 2, 17], 'k-', linewidth=2)
    line49, = ax6.plot(gamma[:, 2, 18], tau[:, 2, 18], 'k-', linewidth=2)
    line50, = ax6.plot(gamma[:, 2, 19], tau[:, 2, 19], 'k-', linewidth=2)
    line51, = ax6.plot(gamma[:, 2, 20], tau[:, 2, 20], 'k-', linewidth=2)
    line52, = ax6.plot(gamma[:, 2, 21], tau[:, 2, 21], 'k-', linewidth=2)
    line53, = ax6.plot(gamma[:, 2, 22], tau[:, 2, 22], 'k-', linewidth=2)
    line54, = ax6.plot(gamma[:, 2, 23], tau[:, 2, 23], 'k-', linewidth=2)
    line55, = ax6.plot(gamma[:, 2, 24], tau[:, 2, 24], 'k-', linewidth=2)
    line56, = ax6.plot(gamma[:, 2, 25], tau[:, 2, 25], 'k-', linewidth=2)
    line57, = ax6.plot(gamma[:, 2, 26], tau[:, 2, 26], 'k-', linewidth=2)
    line58, = ax6.plot(gamma[:, 2, 27], tau[:, 2, 27], 'k-', linewidth=2)
    line59, = ax6.plot(gamma[:, 2, 28], tau[:, 2, 28], 'k-', linewidth=2)
    line60, = ax6.plot(gamma[:, 2, 29], tau[:, 2, 29], 'k-', linewidth=2)
    plt.title("Experiments", fontsize=20)
    plt.xlabel(r'shear strain:  $\gamma$', fontsize=16)
    plt.ylabel(r'shear stress:  $s^{\tau} - s_0^{\tau}$  (dynes)', fontsize=16)

    # the simple shear experiment

    ax7 = plt.subplot(3, 3, 7)
    ax7.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax7.plot(xi[:, 3, 0], pi[:, 3, 0], 'k-', linewidth=2)
    line2, = ax7.plot(xi[:, 3, 1], pi[:, 3, 1], 'k-', linewidth=2)
    line3, = ax7.plot(xi[:, 3, 2], pi[:, 3, 2], 'k-', linewidth=2)
    line4, = ax7.plot(xi[:, 3, 3], pi[:, 3, 3], 'k-', linewidth=2)
    line5, = ax7.plot(xi[:, 3, 4], pi[:, 3, 4], 'k-', linewidth=2)
    line6, = ax7.plot(xi[:, 3, 5], pi[:, 3, 5], 'k-', linewidth=2)
    line7, = ax7.plot(xi[:, 3, 6], pi[:, 3, 6], 'k-', linewidth=2)
    line8, = ax7.plot(xi[:, 3, 7], pi[:, 3, 7], 'k-', linewidth=2)
    line9, = ax7.plot(xi[:, 3, 8], pi[:, 3, 8], 'k-', linewidth=2)
    line10, = ax7.plot(xi[:, 3, 9], pi[:, 3, 9], 'k-', linewidth=2)
    line11, = ax7.plot(xi[:, 3, 10], pi[:, 3, 10], 'k-', linewidth=2)
    line12, = ax7.plot(xi[:, 3, 11], pi[:, 3, 11], 'k-', linewidth=2)
    line13, = ax7.plot(xi[:, 3, 12], pi[:, 3, 12], 'k-', linewidth=2)
    line14, = ax7.plot(xi[:, 3, 13], pi[:, 3, 13], 'k-', linewidth=2)
    line15, = ax7.plot(xi[:, 3, 14], pi[:, 3, 14], 'k-', linewidth=2)
    line16, = ax7.plot(xi[:, 3, 15], pi[:, 3, 15], 'k-', linewidth=2)
    line17, = ax7.plot(xi[:, 3, 16], pi[:, 3, 16], 'k-', linewidth=2)
    line18, = ax7.plot(xi[:, 3, 17], pi[:, 3, 17], 'k-', linewidth=2)
    line19, = ax7.plot(xi[:, 3, 18], pi[:, 3, 18], 'k-', linewidth=2)
    line20, = ax7.plot(xi[:, 3, 19], pi[:, 3, 19], 'k-', linewidth=2)
    line21, = ax7.plot(xi[:, 3, 20], pi[:, 3, 20], 'k-', linewidth=2)
    line22, = ax7.plot(xi[:, 3, 21], pi[:, 3, 21], 'k-', linewidth=2)
    line23, = ax7.plot(xi[:, 3, 22], pi[:, 3, 22], 'k-', linewidth=2)
    line24, = ax7.plot(xi[:, 3, 23], pi[:, 3, 23], 'k-', linewidth=2)
    line25, = ax7.plot(xi[:, 3, 24], pi[:, 3, 24], 'k-', linewidth=2)
    line26, = ax7.plot(xi[:, 3, 25], pi[:, 3, 25], 'k-', linewidth=2)
    line27, = ax7.plot(xi[:, 3, 26], pi[:, 3, 26], 'k-', linewidth=2)
    line28, = ax7.plot(xi[:, 3, 27], pi[:, 3, 27], 'k-', linewidth=2)
    line29, = ax7.plot(xi[:, 3, 28], pi[:, 3, 28], 'k-', linewidth=2)
    line30, = ax7.plot(xi[:, 3, 29], pi[:, 3, 29], 'k-', linewidth=2)
    line31, = ax7.plot(xi[:, 4, 0], pi[:, 4, 0], 'k-', linewidth=2)
    line32, = ax7.plot(xi[:, 4, 1], pi[:, 4, 1], 'k-', linewidth=2)
    line33, = ax7.plot(xi[:, 4, 2], pi[:, 4, 2], 'k-', linewidth=2)
    line34, = ax7.plot(xi[:, 4, 3], pi[:, 4, 3], 'k-', linewidth=2)
    line35, = ax7.plot(xi[:, 4, 4], pi[:, 4, 4], 'k-', linewidth=2)
    line36, = ax7.plot(xi[:, 4, 5], pi[:, 4, 5], 'k-', linewidth=2)
    line37, = ax7.plot(xi[:, 4, 6], pi[:, 4, 6], 'k-', linewidth=2)
    line38, = ax7.plot(xi[:, 4, 7], pi[:, 4, 7], 'k-', linewidth=2)
    line39, = ax7.plot(xi[:, 4, 8], pi[:, 4, 8], 'k-', linewidth=2)
    line40, = ax7.plot(xi[:, 4, 9], pi[:, 4, 9], 'k-', linewidth=2)
    line41, = ax7.plot(xi[:, 4, 10], pi[:, 4, 10], 'k-', linewidth=2)
    line42, = ax7.plot(xi[:, 4, 11], pi[:, 4, 11], 'k-', linewidth=2)
    line43, = ax7.plot(xi[:, 4, 12], pi[:, 4, 12], 'k-', linewidth=2)
    line44, = ax7.plot(xi[:, 4, 13], pi[:, 4, 13], 'k-', linewidth=2)
    line45, = ax7.plot(xi[:, 4, 14], pi[:, 4, 14], 'k-', linewidth=2)
    line46, = ax7.plot(xi[:, 4, 15], pi[:, 4, 15], 'k-', linewidth=2)
    line47, = ax7.plot(xi[:, 4, 16], pi[:, 4, 16], 'k-', linewidth=2)
    line48, = ax7.plot(xi[:, 4, 17], pi[:, 4, 17], 'k-', linewidth=2)
    lin419, = ax7.plot(xi[:, 4, 18], pi[:, 4, 18], 'k-', linewidth=2)
    line50, = ax7.plot(xi[:, 4, 19], pi[:, 4, 19], 'k-', linewidth=2)
    line51, = ax7.plot(xi[:, 4, 20], pi[:, 4, 20], 'k-', linewidth=2)
    line52, = ax7.plot(xi[:, 4, 21], pi[:, 4, 21], 'k-', linewidth=2)
    line53, = ax7.plot(xi[:, 4, 22], pi[:, 4, 22], 'k-', linewidth=2)
    line54, = ax7.plot(xi[:, 4, 23], pi[:, 4, 23], 'k-', linewidth=2)
    line55, = ax7.plot(xi[:, 4, 24], pi[:, 4, 24], 'k-', linewidth=2)
    line56, = ax7.plot(xi[:, 4, 25], pi[:, 4, 25], 'k-', linewidth=2)
    line57, = ax7.plot(xi[:, 4, 26], pi[:, 4, 26], 'k-', linewidth=2)
    line58, = ax7.plot(xi[:, 4, 27], pi[:, 4, 27], 'k-', linewidth=2)
    line59, = ax7.plot(xi[:, 4, 28], pi[:, 4, 28], 'k-', linewidth=2)
    line60, = ax7.plot(xi[:, 4, 29], pi[:, 4, 29], 'k-', linewidth=2)
    plt.title("Thirty", fontsize=20)
    plt.xlabel(r'dilation:  $\xi$', fontsize=18)
    plt.ylabel(r'surface tension:  $s^{\pi} - s_0^{\pi}$  (dynes)',
               fontsize=18)

    # add the curves
    ax8 = plt.subplot(3, 3, 8)
    ax8.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax8.plot(epsilon[:, 3, 0], sigma[:, 3, 0], 'k-', linewidth=2)
    line2, = ax8.plot(epsilon[:, 3, 1], sigma[:, 3, 1], 'k-', linewidth=2)
    line3, = ax8.plot(epsilon[:, 3, 2], sigma[:, 3, 2], 'k-', linewidth=2)
    line4, = ax8.plot(epsilon[:, 3, 3], sigma[:, 3, 3], 'k-', linewidth=2)
    line5, = ax8.plot(epsilon[:, 3, 4], sigma[:, 3, 4], 'k-', linewidth=2)
    line6, = ax8.plot(epsilon[:, 3, 5], sigma[:, 3, 5], 'k-', linewidth=2)
    line7, = ax8.plot(epsilon[:, 3, 6], sigma[:, 3, 6], 'k-', linewidth=2)
    line8, = ax8.plot(epsilon[:, 3, 7], sigma[:, 3, 7], 'k-', linewidth=2)
    line9, = ax8.plot(epsilon[:, 3, 8], sigma[:, 3, 8], 'k-', linewidth=2)
    line10, = ax8.plot(epsilon[:, 3, 9], sigma[:, 3, 9], 'k-', linewidth=2)
    line11, = ax8.plot(epsilon[:, 3, 10], sigma[:, 3, 10], 'k-', linewidth=2)
    line12, = ax8.plot(epsilon[:, 3, 11], sigma[:, 3, 11], 'k-', linewidth=2)
    line13, = ax8.plot(epsilon[:, 3, 12], sigma[:, 3, 12], 'k-', linewidth=2)
    line14, = ax8.plot(epsilon[:, 3, 13], sigma[:, 3, 13], 'k-', linewidth=2)
    line15, = ax8.plot(epsilon[:, 3, 14], sigma[:, 3, 14], 'k-', linewidth=2)
    line16, = ax8.plot(epsilon[:, 3, 15], sigma[:, 3, 15], 'k-', linewidth=2)
    line17, = ax8.plot(epsilon[:, 3, 16], sigma[:, 3, 16], 'k-', linewidth=2)
    line18, = ax8.plot(epsilon[:, 3, 17], sigma[:, 3, 17], 'k-', linewidth=2)
    line19, = ax8.plot(epsilon[:, 3, 18], sigma[:, 3, 18], 'k-', linewidth=2)
    line20, = ax8.plot(epsilon[:, 3, 19], sigma[:, 3, 19], 'k-', linewidth=2)
    line21, = ax8.plot(epsilon[:, 3, 20], sigma[:, 3, 20], 'k-', linewidth=2)
    line22, = ax8.plot(epsilon[:, 3, 21], sigma[:, 3, 21], 'k-', linewidth=2)
    line23, = ax8.plot(epsilon[:, 3, 22], sigma[:, 3, 22], 'k-', linewidth=2)
    line24, = ax8.plot(epsilon[:, 3, 23], sigma[:, 3, 23], 'k-', linewidth=2)
    line25, = ax8.plot(epsilon[:, 3, 24], sigma[:, 3, 24], 'k-', linewidth=2)
    line26, = ax8.plot(epsilon[:, 3, 25], sigma[:, 3, 25], 'k-', linewidth=2)
    line27, = ax8.plot(epsilon[:, 3, 26], sigma[:, 3, 26], 'k-', linewidth=2)
    line28, = ax8.plot(epsilon[:, 3, 27], sigma[:, 3, 27], 'k-', linewidth=2)
    line29, = ax8.plot(epsilon[:, 3, 28], sigma[:, 3, 28], 'k-', linewidth=2)
    line30, = ax8.plot(epsilon[:, 3, 29], sigma[:, 3, 29], 'k-', linewidth=2)
    line31, = ax8.plot(epsilon[:, 4, 0], sigma[:, 4, 0], 'k-', linewidth=2)
    line32, = ax8.plot(epsilon[:, 4, 1], sigma[:, 4, 1], 'k-', linewidth=2)
    line33, = ax8.plot(epsilon[:, 4, 2], sigma[:, 4, 2], 'k-', linewidth=2)
    line34, = ax8.plot(epsilon[:, 4, 3], sigma[:, 4, 3], 'k-', linewidth=2)
    line35, = ax8.plot(epsilon[:, 4, 4], sigma[:, 4, 4], 'k-', linewidth=2)
    line36, = ax8.plot(epsilon[:, 4, 5], sigma[:, 4, 5], 'k-', linewidth=2)
    line37, = ax8.plot(epsilon[:, 4, 6], sigma[:, 4, 6], 'k-', linewidth=2)
    line38, = ax8.plot(epsilon[:, 4, 7], sigma[:, 4, 7], 'k-', linewidth=2)
    line39, = ax8.plot(epsilon[:, 4, 8], sigma[:, 4, 8], 'k-', linewidth=2)
    line40, = ax8.plot(epsilon[:, 4, 9], sigma[:, 4, 9], 'k-', linewidth=2)
    line41, = ax8.plot(epsilon[:, 4, 10], sigma[:, 4, 10], 'k-', linewidth=2)
    line42, = ax8.plot(epsilon[:, 4, 11], sigma[:, 4, 11], 'k-', linewidth=2)
    line43, = ax8.plot(epsilon[:, 4, 12], sigma[:, 4, 12], 'k-', linewidth=2)
    line44, = ax8.plot(epsilon[:, 4, 13], sigma[:, 4, 13], 'k-', linewidth=2)
    line45, = ax8.plot(epsilon[:, 4, 14], sigma[:, 4, 14], 'k-', linewidth=2)
    line46, = ax8.plot(epsilon[:, 4, 15], sigma[:, 4, 15], 'k-', linewidth=2)
    line47, = ax8.plot(epsilon[:, 4, 16], sigma[:, 4, 16], 'k-', linewidth=2)
    line48, = ax8.plot(epsilon[:, 4, 17], sigma[:, 4, 17], 'k-', linewidth=2)
    line49, = ax8.plot(epsilon[:, 4, 18], sigma[:, 4, 18], 'k-', linewidth=2)
    line50, = ax8.plot(epsilon[:, 4, 19], sigma[:, 4, 19], 'k-', linewidth=2)
    line51, = ax8.plot(epsilon[:, 4, 20], sigma[:, 4, 20], 'k-', linewidth=2)
    line52, = ax8.plot(epsilon[:, 4, 21], sigma[:, 4, 21], 'k-', linewidth=2)
    line53, = ax8.plot(epsilon[:, 4, 22], sigma[:, 4, 22], 'k-', linewidth=2)
    line54, = ax8.plot(epsilon[:, 4, 23], sigma[:, 4, 23], 'k-', linewidth=2)
    line55, = ax8.plot(epsilon[:, 4, 24], sigma[:, 4, 24], 'k-', linewidth=2)
    line56, = ax8.plot(epsilon[:, 4, 25], sigma[:, 4, 25], 'k-', linewidth=2)
    line57, = ax8.plot(epsilon[:, 4, 26], sigma[:, 4, 26], 'k-', linewidth=2)
    line58, = ax8.plot(epsilon[:, 4, 27], sigma[:, 4, 27], 'k-', linewidth=2)
    line59, = ax8.plot(epsilon[:, 4, 28], sigma[:, 4, 28], 'k-', linewidth=2)
    line60, = ax8.plot(epsilon[:, 4, 29], sigma[:, 4, 29], 'k-', linewidth=2)
    plt.title("Simple Shear", fontsize=20)
    plt.xlabel(r'squeeze strain:  $\epsilon$', fontsize=16)
    plt.ylabel(r'squeeze stress:  $s^{\sigma} - s_0^{\sigma}$  (dynes)',
               fontsize=16)

    # add the curves
    ax9 = plt.subplot(3, 3, 9)
    ax9.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax9.plot(gamma[:, 3, 0], tau[:, 3, 0], 'k-', linewidth=2)
    line2, = ax9.plot(gamma[:, 3, 1], tau[:, 3, 1], 'k-', linewidth=2)
    line3, = ax9.plot(gamma[:, 3, 2], tau[:, 3, 2], 'k-', linewidth=2)
    line4, = ax9.plot(gamma[:, 3, 3], tau[:, 3, 3], 'k-', linewidth=2)
    line5, = ax9.plot(gamma[:, 3, 4], tau[:, 3, 4], 'k-', linewidth=2)
    line6, = ax9.plot(gamma[:, 3, 5], tau[:, 3, 5], 'k-', linewidth=2)
    line7, = ax9.plot(gamma[:, 3, 6], tau[:, 3, 6], 'k-', linewidth=2)
    line8, = ax9.plot(gamma[:, 3, 7], tau[:, 3, 7], 'k-', linewidth=2)
    line9, = ax9.plot(gamma[:, 3, 8], tau[:, 3, 8], 'k-', linewidth=2)
    line10, = ax9.plot(gamma[:, 3, 9], tau[:, 3, 9], 'k-', linewidth=2)
    line11, = ax9.plot(gamma[:, 3, 10], tau[:, 3, 10], 'k-', linewidth=2)
    line12, = ax9.plot(gamma[:, 3, 11], tau[:, 3, 11], 'k-', linewidth=2)
    line13, = ax9.plot(gamma[:, 3, 12], tau[:, 3, 12], 'k-', linewidth=2)
    line14, = ax9.plot(gamma[:, 3, 13], tau[:, 3, 13], 'k-', linewidth=2)
    line15, = ax9.plot(gamma[:, 3, 14], tau[:, 3, 14], 'k-', linewidth=2)
    line16, = ax9.plot(gamma[:, 3, 15], tau[:, 3, 15], 'k-', linewidth=2)
    line17, = ax9.plot(gamma[:, 3, 16], tau[:, 3, 16], 'k-', linewidth=2)
    line18, = ax9.plot(gamma[:, 3, 17], tau[:, 3, 17], 'k-', linewidth=2)
    line19, = ax9.plot(gamma[:, 3, 18], tau[:, 3, 18], 'k-', linewidth=2)
    line20, = ax9.plot(gamma[:, 3, 19], tau[:, 3, 19], 'k-', linewidth=2)
    line21, = ax9.plot(gamma[:, 3, 20], tau[:, 3, 20], 'k-', linewidth=2)
    line22, = ax9.plot(gamma[:, 3, 21], tau[:, 3, 21], 'k-', linewidth=2)
    line23, = ax9.plot(gamma[:, 3, 22], tau[:, 3, 22], 'k-', linewidth=2)
    line24, = ax9.plot(gamma[:, 3, 23], tau[:, 3, 23], 'k-', linewidth=2)
    line25, = ax9.plot(gamma[:, 3, 24], tau[:, 3, 24], 'k-', linewidth=2)
    line26, = ax9.plot(gamma[:, 3, 25], tau[:, 3, 25], 'k-', linewidth=2)
    line27, = ax9.plot(gamma[:, 3, 26], tau[:, 3, 26], 'k-', linewidth=2)
    line28, = ax9.plot(gamma[:, 3, 27], tau[:, 3, 27], 'k-', linewidth=2)
    line29, = ax9.plot(gamma[:, 3, 28], tau[:, 3, 28], 'k-', linewidth=2)
    line30, = ax9.plot(gamma[:, 3, 29], tau[:, 3, 29], 'k-', linewidth=2)
    line31, = ax9.plot(gamma[:, 4, 0], tau[:, 4, 0], 'k-', linewidth=2)
    line32, = ax9.plot(gamma[:, 4, 1], tau[:, 4, 1], 'k-', linewidth=2)
    line33, = ax9.plot(gamma[:, 4, 2], tau[:, 4, 2], 'k-', linewidth=2)
    line34, = ax9.plot(gamma[:, 4, 3], tau[:, 4, 3], 'k-', linewidth=2)
    line35, = ax9.plot(gamma[:, 4, 4], tau[:, 4, 4], 'k-', linewidth=2)
    line36, = ax9.plot(gamma[:, 4, 5], tau[:, 4, 5], 'k-', linewidth=2)
    line37, = ax9.plot(gamma[:, 4, 6], tau[:, 4, 6], 'k-', linewidth=2)
    line38, = ax9.plot(gamma[:, 4, 7], tau[:, 4, 7], 'k-', linewidth=2)
    line39, = ax9.plot(gamma[:, 4, 8], tau[:, 4, 8], 'k-', linewidth=2)
    line40, = ax9.plot(gamma[:, 4, 9], tau[:, 4, 9], 'k-', linewidth=2)
    line41, = ax9.plot(gamma[:, 4, 10], tau[:, 4, 10], 'k-', linewidth=2)
    line42, = ax9.plot(gamma[:, 4, 11], tau[:, 4, 11], 'k-', linewidth=2)
    line43, = ax9.plot(gamma[:, 4, 12], tau[:, 4, 12], 'k-', linewidth=2)
    line44, = ax9.plot(gamma[:, 4, 13], tau[:, 4, 13], 'k-', linewidth=2)
    line45, = ax9.plot(gamma[:, 4, 14], tau[:, 4, 14], 'k-', linewidth=2)
    line46, = ax9.plot(gamma[:, 4, 15], tau[:, 4, 15], 'k-', linewidth=2)
    line47, = ax9.plot(gamma[:, 4, 16], tau[:, 4, 16], 'k-', linewidth=2)
    line48, = ax9.plot(gamma[:, 4, 17], tau[:, 4, 17], 'k-', linewidth=2)
    line49, = ax9.plot(gamma[:, 4, 18], tau[:, 4, 18], 'k-', linewidth=2)
    line50, = ax9.plot(gamma[:, 4, 19], tau[:, 4, 19], 'k-', linewidth=2)
    line51, = ax9.plot(gamma[:, 4, 20], tau[:, 4, 20], 'k-', linewidth=2)
    line52, = ax9.plot(gamma[:, 4, 21], tau[:, 4, 21], 'k-', linewidth=2)
    line53, = ax9.plot(gamma[:, 4, 22], tau[:, 4, 22], 'k-', linewidth=2)
    line54, = ax9.plot(gamma[:, 4, 23], tau[:, 4, 23], 'k-', linewidth=2)
    line55, = ax9.plot(gamma[:, 4, 24], tau[:, 4, 24], 'k-', linewidth=2)
    line56, = ax9.plot(gamma[:, 4, 25], tau[:, 4, 25], 'k-', linewidth=2)
    line57, = ax9.plot(gamma[:, 4, 26], tau[:, 4, 26], 'k-', linewidth=2)
    line58, = ax9.plot(gamma[:, 4, 27], tau[:, 4, 27], 'k-', linewidth=2)
    line59, = ax9.plot(gamma[:, 4, 28], tau[:, 4, 28], 'k-', linewidth=2)
    line60, = ax9.plot(gamma[:, 4, 29], tau[:, 4, 29], 'k-', linewidth=2)
    plt.title("Experiments", fontsize=20)
    plt.xlabel(r'shear strain:  $\gamma$', fontsize=16)
    plt.ylabel(r'shear stress:  $s^{\tau} - s_0^{\tau}$  (dynes)', fontsize=16)

    plt.savefig('septalMembranes.jpg')

    plt.show()


run()
