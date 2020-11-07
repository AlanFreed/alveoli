#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import log
import numpy as np
from peceHE import Control, Response, PECE
# for creating graphics
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from pylab import rcParams

"""
Created on Thr May 14 2020
Updated on Mon Nov 02 2020

A test file for class ceFiber in file ceChords.py, and also, by import, a test
file for class pece in file peceCE.py.

author: Prof. Alan Freed
"""

# assign the number of CE iterations in PE(CE)^m, with m = 0 -> PE method
m = 1
# assign the number of integration steps to be applied
N = 75
# assign the time variables used
t0 = 0.0
tN = 1.0
dt = (tN - t0) / N

# create a control object


class tensionTest(Control):

    def __init__(self, eVec0, xVec0, dt):
        # Call the constructor of the base type.
        super(tensionTest, self).__init__(eVec0, xVec0, dt)
        # This creates the counter  self.step  which may be useful.
        # Add any other information for your inhereted type, as required.
        self.T0 = 37.0    # initial temperature
        self.TN = 37.0    # final temperature
        self.L0 = 1.0     # initial length
        self.LN = 1.42    # final length
        return  # a new instance of type fiberExtension

    def x(self, t):
        ctrlVars = 2
        xVec = np.zeros((ctrlVars,), dtype=float)
        # You will need to add your application's control functions here,
        # i.e., you will need to populate xVec before returning.
        # The temperature at time t in centigrade
        xVec[0] = self.T0 + (t + t0) * (self.TN - self.T0) / (tN - t0)
        # The strain, i.e., log of stretch, at time t
        Lt = self.L0 + (t + t0) * (self.LN - self.L0) / (tN - t0)
        xVec[1] = log(Lt / self.L0)
        return xVec

    def dxdt(self):
        # Call the base implementation of this method to create dxdtVec.
        dxdtVec = super().dxdt()
        # The returned dxdtVec is computed via finite difference formulae;
        # specifically,
        #   if t = t0              use first-order forward difference formula
        #   if t = t1 or restart   use first-order backward difference formula
        #   otherwise              use second-order backward difference formula
        # Overwrite dxdtVec if this is not appropriate for your application.
        return dxdtVec

    def advance(self):
        # Call the base implementation of this method
        super().advance()
        # Called internally by the pece integrator.  Do not call it yourself.
        # Update your object's data structure, if required
        return  # nothing


def run():
    # create object for fiber control
    ctrlVars = 2
    # create and initialize the two control vectors
    eVec0 = np.zeros((ctrlVars,), dtype=float)
    xVec0 = np.zeros((ctrlVars,), dtype=float)
    
    tt = tensionTest(eVec0, xVec0, dt)
    
    # establish the initial conditions
    eta0 = 3.7E7
    stress0 = 0.0
    yVec0 = np.array([eta0, stress0])    
    # model = ce(eVec0, xVec0, y0, rho, C, alpha, E1, E2, e_t, e_max)
    
    x0 = tt.x(t0)
    T0 = x0[0]
    strain0 = x0[1]

    # create the control object
    ctrl = Control(eVec0, xVec0, dt)
    
    # create the response object with random thickness assignment
    resp = Response(eVec0, xVec0, yVec0)
        
    # create the integrator use to solve this constitutive model

    x = tt.x
    solver = PECE(ctrl, resp)

    # create and initialize the arrays for graphing
    stress = np.zeros((N+1,), dtype=float)
    stress[0] = stress0
    strain = np.zeros((N+1,), dtype=float)
    strain[0] = strain0
    temp = np.zeros((N+1,), dtype=float)
    temp[0] = T0
    entropy = np.zeros((N+1,), dtype=float)
    entropy[0] = eta0
    time = np.zeros((N+1,), dtype=float)
    time[0] = t0
    error = np.zeros((N+1), dtype=float)

    # integrate the constitutive equation
    t = t0
    for i in range(1, N+1):
        t += dt
        solver.integrate(xVec0)
        solver.advance()
        time[i] = t
        x = solver.getX()
        strain[i] = x[1]
        y = solver.getY()
        entropy[i] = y[0]
        stress[i] = y[1]
        error[i] = solver.getError()
        if i == 1:
            error[0] = error[1]

    # create the figure

    plt.figure(1)
    rcParams["figure.figsize"] = 24, 5
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    ax1 = plt.subplot(1, 3, 1)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
    line1, = ax1.plot(strain, stress, 'k-', linewidth=2)

    plt.title("Typical", fontsize=20)
    plt.xlabel(r'Strain, $e = \ln (L/L_0)$', fontsize=18)
    plt.ylabel('Stress,  barye', fontsize=18)

    ax2 = plt.subplot(1, 3, 2)
    # ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.5e'))
    line1, = ax2.plot(strain, entropy, 'k-', linewidth=2)

    plt.title("Biologic", fontsize=20)
    plt.xlabel(r'Strain, $e = \ln (L/L_0)$', fontsize=18)
    plt.ylabel('Entropy, erg/g.K', fontsize=18)

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_yscale('log')
    line1, = ax3.plot(strain, error, 'k-', linewidth=2)

    plt.title("Fiber", fontsize=20)
    plt.xlabel(r'Strain, $e = \ln (L/L_0)$', fontsize=18)
    plt.ylabel('Local Truncation Error', fontsize=18)

    plt.savefig('biologicFiber.jpg')
    plt.show()


run()
