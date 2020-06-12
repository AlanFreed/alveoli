#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import sqrt
from numpy import sign
import sys

"""
Module ridder.py provides a root finding algorithm.

Copyright (c) 2020 Alan D. Freed

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
__date__ = "07-06-2018"
__update__ = "05-20-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
Function 'findRoot' uses Ridder's method to determine a root of a function,
i.e., find x such that
   0 = f(x)
to a specified error tolerance.  The default error tolerance is 1.0e-9.

    root = findRoot(xL, xU, f)

If you want to assign a different tolerance use

    root = findRoot(xL, xU, f, tol)

where

    xL   is the lower boundary of the search window
    xU   is the upper boundary of the search window
    f    is the function whose root is being sought
    tol  is the error tolerance in input variable x
"""

maxIter = 30


def findRoot(xL, xU, f, tol=1.0e-9):
    # analyze the original interval
    xl = xL         # x at the left end of the interval
    fl = f(xl)
    if fl == 0.0:
        return xl
    xr = xU         # x at the right end of an interval
    fr = f(xr)
    if fr == 0.0:
        return xr
    if sign(fl) == sign(fr):
        print('Error: root is not bracketed')
        sys.exit

    # perform Ridder's algorithm for root finding
    xiOld = (xl + xr) / 2.0
    for i in range(maxIter):
        xm = (xl + xr) / 2.0         # x at the midpoint
        fm = f(xm)
        s = sqrt(fm**2 - fl*fr)
        if s == 0.0:
            return None
        dx = (xm - xl) * fm / s
        if (fl - fr) < 0.0:
            dx = -dx
        xi = xm + dx              # estimate for the root
        fxi = f(xi)
        # test for convergence
        if abs(xi - xiOld) < tol*max(abs(xi), 1.0):
            return xi
        xiOld = xi
        # rebracket the root as tightly as possible
        if sign(fm) == sign(fxi):
            if sign(fl) != sign(fxi):
                xr = xi
                fr = fxi
            else:
                xl = xi
                fl = fxi
        else:
            xl = xm
            fl = fm
            xr = xi
            fr = fxi
    print('Warning: iterations exceeded maxIter in findRoot.')
    return xi


"""
Changes made in version "1.0.0":

This is the initial version of this code, which I took from my lecture notes.
"""
