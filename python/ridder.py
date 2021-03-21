#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import sqrt
from numpy import sign

"""
Module ridder.py provides a root finding algorithm.

Copyright (c) 2020 Alan D. Freed


This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
Copyright (c) 2020 Alan D. Freed
"""

# Module metadata
__version__ = "1.0.0"
__date__ = "07-06-2018"
__update__ = "07-06-2020"
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

MAX_ITERATIONS = 30


def findRoot(xL, xU, f, tol=1.0e-9):
    # analyze the original interval
    x_left = xL          # x at the left end of the interval
    f_left = f(x_left)
    if f_left == 0.0 or f_left == -0.0:
        return x_left
    x_right = xU         # x at the right end of an interval
    f_right = f(x_right)
    if f_right == 0.0 or f_right == -0.0:
        return x_right
    if sign(f_left) == sign(f_right):
        raise RuntimeError('Error: root is not bracketed')

    # perform Ridder's algorithm for root finding
    xi_old = (x_left + x_right) / 2.0               # previous midpoint
    for i in range(MAX_ITERATIONS):
        x_middle = (x_left + x_right) / 2.0         # x at the midpoint
        f_middle = f(x_middle)
        s = sqrt(f_middle**2 - f_left*f_right)
        if s == 0.0:
            return None
        dx = (x_middle - x_left) * f_middle / s
        if (f_left - f_right) < -0.0:
            dx = -dx
        xi = x_middle + dx                          # estimate for the root
        f_xi = f(xi)
        # test for convergence
        if abs(xi-xi_old) < tol*max(abs(xi), 1.0):
            return xi
        xi_old = xi
        # rebracket the root as tightly as possible
        if sign(f_middle) == sign(f_xi):
            if sign(f_left) != sign(f_xi):
                x_right = xi
                f_right = f_xi
            else:
                x_left = xi
                f_left = f_xi
        else:
            x_left = x_middle
            f_left = f_middle
            x_right = xi
            f_right = f_xi
    print('\nWarning: iterations exceeded MAX_ITERATIONS in findRoot.\n')
    return xi


"""
Changes made in version "1.0.0":

This is the initial version of this code, which I took from my lecture notes.
"""
