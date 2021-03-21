#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 03 2019
Updated on Tue Jul 07 2020

Creates a 3D plot of the Wachspress shape function for a pentagon.

author: afreed
"""

import math as m
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
from shapeFnPentagons import ShapeFunction


def run():
    # construct the data arrays

    dim = 276
    x = np.zeros(dim, dtype=float)
    y = np.zeros(dim, dtype=float)
    z = np.zeros(dim, dtype=float)

    # establish the coordinate data
    step = -1
    for i in range(10, 0, -1):
        # vertices spiralling inward
        x1 = (i / 10) * m.cos(m.pi / 2.0)
        y1 = (i / 10) * m.sin(m.pi / 2.0)
        x2 = (i / 10) * m.cos(9.0 * m.pi / 10.0)
        y2 = (i / 10) * m.sin(9.0 * m.pi / 10.0)
        x3 = (i / 10) * m.cos(13.0 * m.pi / 10.0)
        y3 = (i / 10) * m.sin(13.0 * m.pi / 10.0)
        x4 = (i / 10) * m.cos(17.0 * m.pi / 10.0)
        y4 = (i / 10) * m.sin(17.0 * m.pi / 10.0)
        x5 = (i / 10) * m.cos(1.0 * m.pi / 10.0)
        y5 = (i / 10) * m.sin(1.0 * m.pi / 10.0)
        # chord from vertex 1 to vertex 2
        for j in range(i):
            step += 1
            x[step] = x1 + (j / i) * (x2 - x1)
            y[step] = y1 + (j / i) * (y2 - y1)
        # chord from vertex 2 to vertex 3
        for j in range(i):
            step += 1
            x[step] = x2 + (j / i) * (x3 - x2)
            y[step] = y2 + (j / i) * (y3 - y2)
        # chord from vertex 3 to vertex 4
        for j in range(i):
            step += 1
            x[step] = x3 + (j / i) * (x4 - x3)
            y[step] = y3 + (j / i) * (y4 - y3)
        # chord from vertex 4 to vertex 5
        for j in range(i):
            step += 1
            x[step] = x4 + (j / i) * (x5 - x4)
            y[step] = y4 + (j / i) * (y5 - y4)
        # chord from vertex 5 to vertex 1
        for j in range(i):
            step += 1
            x[step] = x5 + (j / i) * (x1 - x5)
            y[step] = y5 + (j / i) * (y1 - y5)
    # add in the origin
    step += 1
    x[step] = 0.0
    y[step] = 0.0

    # evaluate the step functions
    for i in range(step+1):
        coordinates = (x[i], y[i])
        sf = ShapeFunction(coordinates)
        z[i] = sf.N1

    # plot the surface of a shape function

    # create a Delaunay triangularized surface plot

    fig = plt.figure()
    rcParams['figure.figsize'] = 12, 8
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=11)

    # ax1 = Axes3D(fig)
    ax1 = plt.axes(projection='3d')

    ax1.set_xlabel('$x$', fontsize=17)
    ax1.set_ylabel('$y$', fontsize=17)
    ax1.set_zlabel('$N_1$', fontsize=17)

    ax1.set_xlim3d(-1.0, 1.0)
    ax1.set_ylim3d(-0.8, 1.0)
    ax1.set_zlim3d(0.0, 1.0)

    # Rotate it
    # ax1.view_init(20, -30)
    ax1.view_init(20, -10)

    surf = ax1.plot_trisurf(x, y, z, cmap=plt.cm.jet, linewidth=0.2,
                            antialiased=True)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig('shapeFunction.jpg')

    plt.show()


run()
