#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    author:  Prof. Alan Freed, Texas A&M University
    date:    June 17, 2017
    update:  April 20, 2020

    This problem is known as the Brusselator:

        dy1/dt = A - (B + 1) y1 + y1^2 y2
        dy2/dt = B y1 - y1^2 y2

    Solutions will have a limit cycle when A = 1 and B = 3.  For x0 = 0, use
    initial conditions of (0.1, 0.1), (1.5, 3), (3, 1), (3.25, 2.5) running
    out to xN = 20.

    Solutions will be stiff when A = 1 and B = 100.  Here one can use the same
    initial conditions, but it is useful to set xN = 0.1.

    Reference:
    A. D. Freed and I. Iskovitz, "Development and Applications of a Rosenbrock
    Integrator," NASA TM 4709, 1996.
"""

import numpy as np
from peceVtoX import pece
# for making plots
from matplotlib import pyplot as plt
from matplotlib import rc


def limitCycle(x, y):
    y1 = 1. - 4. * y[0] + y[0] * y[0] * y[1]
    y2 = 3. * y[0] - y[0] * y[0] * y[1]
    ode = np.array([y1, y2])
    return ode


def stiff(x, y):
    y1 = 1. - 101. * y[0] + y[0] * y[0] * y[1]
    y2 = 100. * y[0] - y[0] * y[0] * y[1]
    ode = np.array([y1, y2])
    return ode


def test():
    print()
    print("Using 200 points for drawing curves:")
    print()
    # limit cycle analysis:
    ode = limitCycle
    h = 0.1
    tol = 0.0001
    x0 = 0.

    # first initial condition:
    y0 = np.array([0.1, 0.1])
    solver = pece(ode, x0, y0, h, tol)
    results = []
    results.append((x0, y0[0], y0[1], tol))
    for i in range(200):
        solver.integrate()
        solver.advance()
        yi = solver.getX()
        xi = solver.getT()
        erri = solver.getError()
        results.append((xi, yi[0], yi[1], erri))
    xxIC1 = np.array([z[0] for z in results])
    y1IC1 = np.array([z[1] for z in results])
    y2IC1 = np.array([z[2] for z in results])
    erIC1 = np.array([z[3] for z in results])
    print("The Brusselator, with ICs ({}, {}) ran with statistics:"
          .format(y0[0], y0[1]))
    n, n_d, n_h, n_r = solver.getStatistics()
    print("   {} steps with {} restarts".format(n, n_r))
    print("   of which {} steps were doubled".format(n_d))
    print("   and {} steps were halved.".format(n_h))
    print()

    # second initial condition:
    y0 = np.array([1.5, 3.0])
    solver = pece(ode, x0, y0, h, tol)
    results = []
    results.append((x0, y0[0], y0[1], tol))
    for i in range(200):
        solver.integrate()
        solver.advance()
        xi = solver.getT()
        yi = solver.getX()
        erri = solver.getError()
        results.append((xi, yi[0], yi[1], erri))
    xxIC2 = np.array([z[0] for z in results])
    y1IC2 = np.array([z[1] for z in results])
    y2IC2 = np.array([z[2] for z in results])
    erIC2 = np.array([z[3] for z in results])
    print("The Brusselator, with ICs ({}, {}) ran with statistics:"
          .format(y0[0], y0[1]))
    n, n_d, n_h, n_r = solver.getStatistics()
    print("   {} steps with {} restarts".format(n, n_r))
    print("   of which {} steps were doubled".format(n_d))
    print("   and {} steps were halved.".format(n_h))
    print()

    # third initial condition:
    y0 = np.array([2.0, 0.5])
    solver = pece(ode, x0, y0, h, tol)
    results = []
    results.append((x0, y0[0], y0[1], tol))
    for i in range(200):
        solver.integrate()
        solver.advance()
        xi = solver.getT()
        yi = solver.getX()
        erri = solver.getError()
        results.append((xi, yi[0], yi[1], erri))
    xxIC3 = np.array([z[0] for z in results])
    y1IC3 = np.array([z[1] for z in results])
    y2IC3 = np.array([z[2] for z in results])
    erIC3 = np.array([z[3] for z in results])
    print("The Brusselator, with ICs ({}, {}) ran with statistics:"
          .format(y0[0], y0[1]))
    n, n_d, n_h, n_r = solver.getStatistics()
    print("   {} steps with {} restarts".format(n, n_r))
    print("   of which {} steps were doubled".format(n_d))
    print("   and {} steps were halved.".format(n_h))
    print()

    # fourth initial condition:
    y0 = np.array([3.25, 2.5])
    solver = pece(ode, x0, y0, h, tol)
    results = []
    results.append((x0, y0[0], y0[1], tol))
    for i in range(200):
        solver.integrate()
        solver.advance()
        xi = solver.getT()
        yi = solver.getX()
        erri = solver.getError()
        results.append((xi, yi[0], yi[1], erri))
    xxIC4 = np.array([z[0] for z in results])
    y1IC4 = np.array([z[1] for z in results])
    y2IC4 = np.array([z[2] for z in results])
    erIC4 = np.array([z[3] for z in results])
    print("The Brusselator, with ICs ({}, {}) ran with statistics:"
          .format(y0[0], y0[1]))
    n, n_d, n_h, n_r = solver.getStatistics()
    print("   {} steps with {} restarts".format(n, n_r))
    print("   of which {} steps were doubled".format(n_d))
    print("   and {} steps were halved.".format(n_h))
    print()

    # font.family        : serif
    # font.serif         : Times, Palatino, New Century Schoolbook, Bookman,
    #                      Computer Modern Roman
    # font.sans-serif    : Helvetica, Avant Garde, Computer Modern Sans serif
    # font.cursive       : Zapf Chancery
    # font.monospace     : Courier, Computer Modern Typewriter

    # for Helvetica and other sans-serif fonts use, e.g.:
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # for Palatino and other serif fonts use, e.g.:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    # etc.

    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # create the response plots

    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(y1IC1, y2IC1, 'k-', linewidth=2)
    line2, = ax.plot(y1IC2, y2IC2, 'r-', linewidth=2)
    line3, = ax.plot(y1IC3, y2IC3, 'b-', linewidth=2)
    line4, = ax.plot(y1IC4, y2IC4, 'g-', linewidth=2)
    line5, = ax.plot(1, 3, 'ko')

    plt.title("Brusselator with Limit Cycle", fontsize=20)
    plt.xlabel('$y_1$', fontsize=16)
    plt.ylabel('$y_2$', fontsize=16)
    plt.legend([line1, line2, line3, line4, line5],
               [r"$\mathbf{y}_0 = (0.1, 0.1)$",
                r"$\mathbf{y}_0 = (1.5, 3.0)$",
                r"$\mathbf{y}_0 = (2.0, 0.5)$",
                r"$\mathbf{y}_0 = (3.25, 2.5)$",
                "steady state"],
               bbox_to_anchor=(0.6, 0.95), loc=2, borderaxespad=0.)
    plt.savefig('limitCycle')
    plt.show()

    # create the error plots

    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    ax.set_yscale('log')
    line1, = ax.plot(xxIC1, erIC1, 'k-', linewidth=2)
    line2, = ax.plot(xxIC2, erIC2, 'r-', linewidth=2)
    line3, = ax.plot(xxIC3, erIC3, 'b-', linewidth=2)
    line4, = ax.plot(xxIC4, erIC4, 'g-', linewidth=2)

    plt.title("Brusselator with Limit Cycle", fontsize=20)
    plt.xlabel('$t$ (seconds)', fontsize=16)
    plt.ylabel('local truncation error', fontsize=16)
    plt.savefig('limitCycleError')
    plt.show()

    # stiff analysis:

    print()
    print("Using 100 points for drawing curves:")
    print()

    ode = stiff
    h = 0.001
    tol = 0.0001
    x0 = 0.

    # first initial condition:
    y0 = np.array([0.1, 0.1])
    solver = pece(ode, x0, y0, h, tol)
    results = []
    results.append((x0, y0[0], y0[1], tol))
    for i in range(100):
        solver.integrate()
        solver.advance()
        xi = solver.getT()
        yi = solver.getX()
        erri = solver.getError()
        results.append((xi, yi[0], yi[1], erri))
    xxIC1 = np.array([z[0] for z in results])
    y1IC1 = np.array([z[1]/y0[0] for z in results])
    y2IC1 = np.array([z[2]/y0[1] for z in results])
    erIC1 = np.array([z[3] for z in results])
    print("Stiff Brusselator, with ICs ({}, {}) ran with statistics:"
          .format(y0[0], y0[1]))
    n, n_d, n_h, n_r = solver.getStatistics()
    print("   {} steps with {} restarts".format(n, n_r))
    print("   of which {} steps were doubled".format(n_d))
    print("   and {} steps were halved.".format(n_h))
    print()

    # second initial condition:
    y0 = np.array([1.5, 3.0])
    solver = pece(ode, x0, y0, h, tol)
    results = []
    results.append((x0, y0[0], y0[1], tol))
    for i in range(100):
        solver.integrate()
        solver.advance()
        xi = solver.getT()
        yi = solver.getX()
        erri = solver.getError()
        results.append((xi, yi[0], yi[1], erri))
    xxIC2 = np.array([z[0] for z in results])
    y1IC2 = np.array([z[1]/y0[0] for z in results])
    y2IC2 = np.array([z[2]/y0[1] for z in results])
    erIC2 = np.array([z[3] for z in results])
    print("Stiff Brusselator, with ICs ({}, {}) ran with statistics:"
          .format(y0[0], y0[1]))
    n, n_d, n_h, n_r = solver.getStatistics()
    print("   {} steps with {} restarts".format(n, n_r))
    print("   of which {} steps were doubled".format(n_d))
    print("   and {} steps were halved.".format(n_h))
    print()

    # third initial condition:
    y0 = np.array([2.0, 0.5])
    solver = pece(ode, x0, y0, h, tol)
    results = []
    results.append((x0, y0[0], y0[1], tol))
    for i in range(100):
        solver.integrate()
        solver.advance()
        xi = solver.getT()
        yi = solver.getX()
        erri = solver.getError()
        results.append((xi, yi[0], yi[1], erri))
    xxIC3 = np.array([z[0] for z in results])
    y1IC3 = np.array([z[1]/y0[0] for z in results])
    y2IC3 = np.array([z[2]/y0[1] for z in results])
    erIC3 = np.array([z[3] for z in results])
    print("Stiff Brusselator, with ICs ({}, {}) ran with statistics:"
          .format(y0[0], y0[1]))
    n, n_d, n_h, n_r = solver.getStatistics()
    print("   {} steps with {} restarts".format(n, n_r))
    print("   of which {} steps were doubled".format(n_d))
    print("   and {} steps were halved.".format(n_h))
    print()

    # fourth initial condition:
    y0 = np.array([3.25, 2.5])
    solver = pece(ode, x0, y0, h, tol)
    results = []
    results.append((x0, y0[0], y0[1], tol))
    for i in range(100):
        solver.integrate()
        solver.advance()
        xi = solver.getT()
        yi = solver.getX()
        erri = solver.getError()
        results.append((xi, yi[0], yi[1], erri))
    xxIC4 = np.array([z[0] for z in results])
    y1IC4 = np.array([z[1]/y0[0] for z in results])
    y2IC4 = np.array([z[2]/y0[1] for z in results])
    erIC4 = np.array([z[3] for z in results])
    print("Stiff Brusselator, with ICs ({}, {}) ran with statistics:"
          .format(y0[0], y0[1]))
    n, n_d, n_h, n_r = solver.getStatistics()
    print("   {} steps with {} restarts".format(n, n_r))
    print("   of which {} steps were doubled".format(n_d))
    print("   and {} steps were halved.".format(n_h))
    print()

    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    # create the response plots

    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(xxIC1, y1IC1, 'k-', linewidth=2)
    line2, = ax.plot(xxIC2, y1IC2, 'r-', linewidth=2)
    line3, = ax.plot(xxIC3, y1IC3, 'b-', linewidth=2)
    line4, = ax.plot(xxIC4, y1IC4, 'g-', linewidth=2)

    plt.title("Stiff Brusselator, $y_1$", fontsize=20)
    plt.xlabel('$t$ (seconds)', fontsize=16)
    plt.ylabel('$y_1(t) / y_1(0)$', fontsize=16)
    plt.legend([line1, line2, line3, line4],
               [r"$\mathbf{y}_0 = (0.1, 0.1)$",
                r"$\mathbf{y}_0 = (1.5, 3.0)$",
                r"$\mathbf{y}_0 = (2.0, 0.5)$",
                r"$\mathbf{y}_0 = (3.25, 2.5)$"],
               bbox_to_anchor=(0.6, 0.95), loc=2, borderaxespad=0.)
    plt.savefig('stiffY1')
    plt.show()

    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(xxIC1, y2IC1, 'k-', linewidth=2)
    line2, = ax.plot(xxIC2, y2IC2, 'r-', linewidth=2)
    line3, = ax.plot(xxIC3, y2IC3, 'b-', linewidth=2)
    line4, = ax.plot(xxIC4, y2IC4, 'g-', linewidth=2)

    plt.title("Stiff Brusselator, $y_2$", fontsize=20)
    plt.xlabel('$t$ (seconds)', fontsize=16)
    plt.ylabel('$y_2(t) / y_2(0)$', fontsize=16)
    plt.legend([line1, line2, line3, line4],
               [r"$\mathbf{y}_0 = (0.1, 0.1)$",
                r"$\mathbf{y}_0 = (1.5, 3.0)$",
                r"$\mathbf{y}_0 = (2.0, 0.5)$",
                r"$\mathbf{y}_0 = (3.25, 2.5)$"],
               bbox_to_anchor=(0.6, 0.85), loc=2, borderaxespad=0.)
    plt.savefig('stiffY2')
    plt.show()

    # create the error plots

    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    ax.set_yscale('log')
    line1, = ax.plot(xxIC1, erIC1, 'k-', linewidth=2)
    line2, = ax.plot(xxIC2, erIC2, 'r-', linewidth=2)
    line3, = ax.plot(xxIC3, erIC3, 'b-', linewidth=2)
    line4, = ax.plot(xxIC4, erIC4, 'g-', linewidth=2)

    plt.title("Stiff Brusselator", fontsize=20)
    plt.xlabel('$t$ (seconds)', fontsize=16)
    plt.ylabel('local truncation error', fontsize=16)
    plt.legend([line1, line2, line3, line4],
               [r"$\mathbf{y}_0 = (0.1, 0.1)$",
                r"$\mathbf{y}_0 = (1.5, 3.0)$",
                r"$\mathbf{y}_0 = (2.0, 0.5)$",
                r"$\mathbf{y}_0 = (3.25, 2.5)$"],
               bbox_to_anchor=(0.6, 0.95), loc=2, borderaxespad=0.)
    plt.savefig('stiffError')
    plt.show()


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
test()
