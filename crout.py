#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
Module crout.py provides Crout's LU decomposition of a matrix and a solver.

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

It performs an LU decomposition of matrix A and provides the necessary solver
to solve a linear system of equations  Ax = b.  Matrix A is converted into a
lower-triangular matrix  L  and an upper-triangular matrix  U  using Crout's
algorithm.  No pivoting is needed for our application.

L, U = LU(A, tol=1.0e-12)
        A   is a square matrix
        tol is the tolerence used to determine singularity
    returns
        L   is a lower-triangular matrix
        U   is a unit upper-triangular matrix

x = LUsolve(L, U, b, tol=1.0e-12)
        L   is a lower-triangular matrix
        U   is a unit upper-triangular matrix
        b   is the right hand side
        tol is the tolerence used to determine singularity
    returns
        x   is the solution vector for  Ax = b  or  LUx = b
"""


def LU(A, tol=1.0e-12):
    (n, m) = np.shape(A)
    if not n == m:
        raise RuntimeError('Matrix A must be square.')
    L = np.zeros((n, n), dtype=float)
    U = np.identity(n, dtype=float)

    # construct the first column of L and first row of U
    L[0:n, 0] = A[0:n, 0]
    if abs(L[0, 0]) > tol:
        U[0, 1:n] = A[0, 1:n] / L[0, 0]
    else:
        raise RuntimeError('Matrix A was singular.')

    # construct the remaining rows and columns of L and U
    for k in range(1, n):
        for i in range(k, n):
            L[i, k] = A[i, k] - np.dot(L[i, 0:k], U[0:k, k])
        for j in range(k+1, n):
            if abs(L[k, k]) > tol:
                U[k, j] = (A[k, j] - np.dot(L[k, 0:k], U[0:k, j])) / L[k, k]
            else:
                raise RuntimeError('Matrix A was singular.')
    return L, U


def _Lsolve(L, b, tol=1.0e-12):
    n = len(b)
    for k in range(n):
        if abs(L[k, k]) < tol:
            raise RuntimeError('Matrix L was singular.')

    # solve [L]{x} = {b} for {x} via forward substitution
    x = np.copy(b)
    x[0] = x[0] / L[0, 0]
    for k in range(1, n):
        x[k] = (x[k] - np.dot(L[k, 0:k], x[0:k])) / L[k, k]
    return x


def _Usolve(U, b, tol=1.0e-12):
    n = len(b)
    for k in range(n):
        if abs(U[k, k]) < tol:
            raise RuntimeError('Matrix U was singular.')

    # solve [U]{x} = {b} for {x} via backward substitution
    x = np.copy(b)
    x[n-1] = x[n-1] / U[n-1, n-1]
    for k in range(n-2, -1, -1):
        x[k] = (x[k] - np.dot(U[k, k+1:n], x[k+1:n])) / U[k, k]
    return x


def LUsolve(L, U, b, tol=1.0e-12):
    # solve [L]{y} = {b} for {y}
    y = _Lsolve(L, b, tol)
    # solve [U]{x} = {y} for {x}
    x = _Usolve(U, y, tol)
    return x
