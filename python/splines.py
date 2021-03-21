#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from crout import LU, LUsolve
import numpy as np

"""
Module splines.py provides a cubic spline for interpolating data.

Copyright (c) 2019 Alan D. Freed

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Module metadata
__version__ = "1.0.0"
__date__ = "09-24-2019"
__update__ = "09-24-2019"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
This module was taken from the author's course in numerical methods.

Given x and y data (viz., xData and yData) where  x  represents the independent
variable and  y  represents the dependent variable, this module finds the
coefficients (a, b, c, d) belonging to a cubic interpolation of these data,
where

    y(x) = a + b*x + c*x^2 + d*x^3

    y'(x) = b + 2*c*x + 3*d*x^2

procedures

    a, b, c, d = getCoef(xData, yData)
        Supplies the coefficients for a cubic polynomial splined over the data.
            xData   the x data array to be splined
            yData   the y data array to be splined
        returns
            a       constant coefficients
            b       linear coefficients
            c       quadratic coefficients
            d       cubic coefficients

    y = Y(a, b, c, d, xData, x)
        Given an independent variable x, it returns the dependent variable y.
            a       constant coefficients
            b       linear coefficients
            c       quadratic coefficients
            d       cubic coefficients
            xData   the x data array to be splined
            x       the value for which y(x) is sought
        returns
            y       the interpolated valued for the dependent variable

    dy = dYdX(b, c, d, xData, x)
        Given an independent variable x, it returns the slope dy/dx.
            b       linear coefficients
            c       quadratic coefficients
            d       cubic coefficients
            xData   the x data array to be splined
            x       the value for which y(x) is sought
        returns
            dy      the interpolated valued for the derivative  dy(x)/dx
"""


def _findSegment(xData, x):
    # finds the left index/node for the segment containing datum  x
    iLeft = 0
    iRight = len(xData) - 1
    if x < xData[iLeft] or x > xData[iRight]:
        raise RuntimeError('x = {} lies outside the data range of [{}, {}].'
                           .format(x, xData[iLeft], xData[iRight]))
    searching = True
    while searching:
        if (iRight - iLeft) <= 1:
            searching = False
        else:
            i = (iLeft + iRight) // 2
            if x < xData[i]:
                iRight = i
            else:
                iLeft = i
    return iLeft


def getCoef(xData, yData):
    if len(xData) == len(yData):
        n = len(xData) - 1
    else:
        raise RuntimeError('Vectors xData and yData must have the same size.')

    # create the arrays to be used
    a = np.zeros(n, dtype=float)
    b = np.zeros(n, dtype=float)
    c = np.zeros(n, dtype=float)
    d = np.zeros(n, dtype=float)
    rhs = np.zeros(4*n, dtype=float)
    mtx = np.zeros((4*n, 4*n), dtype=float)

    # assign values to the matrix
    # construct the first two rows
    x0 = xData[0]
    mtx[0, 0] = 1.0
    mtx[0, 1] = x0
    mtx[0, 2] = x0 * x0
    mtx[0, 3] = x0**3
    mtx[1, 1] = 1.0
    mtx[1, 2] = 2.0 * x0
    mtx[1, 3] = 3.0 * x0 * x0
    # construct the repeating blocks
    for i in range(1, n):
        k = 4 * i
        km1 = k - 1
        km2 = k - 2
        km3 = k - 3
        km4 = k - 4
        kp1 = k + 1
        kp2 = k + 2
        kp3 = k + 3
        xi = xData[i]
        mtx[km2, km4] = 1.0
        mtx[km2, km3] = xi
        mtx[km2, km2] = xi * xi
        mtx[km2, km1] = xi**3
        mtx[km1, km2] = -2.0
        mtx[km1, km1] = -6.0 * xi
        mtx[km1, kp2] = 2.0
        mtx[km1, kp3] = 6.0 * xi
        mtx[k, k] = 1.0
        mtx[k, kp1] = xi
        mtx[k, kp2] = xi * xi
        mtx[k, kp3] = xi**3
        mtx[kp1, km3] = -1.0
        mtx[kp1, km2] = -2.0 * xi
        mtx[kp1, km1] = -3.0 * xi * xi
        mtx[kp1, kp1] = 1.0
        mtx[kp1, kp2] = 2.0 * xi
        mtx[kp1, kp3] = 3.0 * xi * xi
    # construct the last two rows
    k = 4 * n - 1
    km1 = k - 1
    km2 = k - 2
    km3 = k - 3
    xn = xData[n]
    mtx[km1, km3] = 1.0
    mtx[km1, km2] = xn
    mtx[km1, km1] = xn * xn
    mtx[km1, k] = xn**3
    mtx[k, km2] = 1.0
    mtx[k, km1] = 2.0 * xn
    mtx[k, k] = 3.0 * xn * xn

    # assign values to the right-hand side vector
    # construct the first two rows
    rhs[0] = yData[0]
    rhs[1] = (yData[1] - yData[0]) / (xData[1] - xData[0])
    # construct the repeating blocks
    for i in range(1, n):
        rhs[4*i-2] = yData[i]
        rhs[4*i] = yData[i]
    # construct the last two rows
    rhs[4*n-2] = yData[n]
    rhs[4*n-1] = (yData[n] - yData[n-1]) / (xData[n] - xData[n-1])

    # call LU to solve
    L, U = LU(mtx)
    x = LUsolve(L, U, rhs)

    # populate the coefficient vectors
    for i in range(n):
        k = 4 * i
        a[i] = x[k]
        b[i] = x[k+1]
        c[i] = x[k+2]
        d[i] = x[k+3]

    return a, b, c, d


def Y(a, b, c, d, xData, x):
    i = _findSegment(xData, x)
    y = a[i] + b[i] * x + c[i] * x**2 + d[i] * x**3
    return y


def dYdX(b, c, d, xData, x):
    i = _findSegment(xData, x)
    dydx = b[i] + 2.0 * c[i] * x + 3.0 * d[i] * x**2
    return dydx
