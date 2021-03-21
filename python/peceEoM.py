#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA

"""
Module peceEoM.py provides a PECE solver for solving Equations of Motion.

Copyright (c) 2017-2020 Alan D. Freed

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Module metadata
__version__ = "1.0.0"
__date__ = "07-16-2017"
__update__ = "07-06-2020"
__author__ = "Alan D. Freed"
__author_email__ = "adfreed@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


This module solves the following system of ODEs
    dv/dt = a(t,u,v)
subject to initial conditions  u0  and  v0  at time  t0  where  t  is a real
or float and where  u,  v  and  a  are vectors; specifically, they are
instances of np.ndarray.  Here  a  denotes acceleration,  v  denotes velocity,
and  u  denotes displacement, in a generalized sense, with  u  and  v
following from integration
    du/dt = v  and  dv/dt = a
which has the structure of a classic dynamics problem.

A predict/evaluate/correct/evaluate (PECE) solver has been implemented.
The method is second-order accurate in velocities and third-order accurate in
displacements.
    The predictors are:
        uPred = (1 / 3) (4 u_n - u_nm1)
              + (h / 6) (3 v_n + v_nm1)
              + (h^2 / 36) (31 a_n - a_nm1)
        vPred = (1 / 3) (4 v_n - v_nm1)
              + (2 h / 3) (2 a_n - a_nm1)
    at which point the acceleration is evaluated via a supplied function
        aPred = a(t_n + h, uPred, vPred)
    The correctors are:
        uCorr = (1 / 3) (4 u_n - u_nm1)
              + (h / 24) (vPred + 14 v_n + v_nm1)
              + (h^2 / 72) (10 aPred + 51 a_n - a_nm1)
        vCorr = (1 / 3) (4 v_n - v_nm1)
              + (2 h / 3) aPred
    with acceleration then being re-evaluated as
        aCorr = a(t_n + h, uCorr, vCorr)
The step size is taken to be uniform over the integration, which itself may be
under the control of some external global driver.

Results are available at the end of the most recent step.  The solver itself
does not return results.  They are gotten through methods of the object.

The above integrator is not self starting, so a one-step method is used to
take the first integration step:
    the predictor is:
        uPred = u0 + h v0 + (h^2 / 2) a0
        vPred = v0 + h a0
    the corrector is:
        uCorr = u0 + (h / 2) (vPred + v0) - (h^2 / 12) (aPred - a0)
        vCorr = v0 + (h / 2) (aPred + a0)
with evaluations for acceleration occurring after each PC integration.

object constructor

    E.g.:  solver = PECE(u0, v0, t0, dt, aFn, m=1)
        u0  is the array of initial displacements, 1 of 2 initial conditions
        v0  is the array of initial velocities,    2 of 2 initial conditions
        t0  is the initial time
        dt  is the time step size to be applied uniformly over time
        aFn is a callable function for acceleration, viz.,  a = aFn(t, u, v)
        m   is the number of CE interations, i.e. PE(CE)^m, m in [0, 10]
    Creates an object to integrate second-order ODEs using a PE(CE)^m method.

integrate(restart=False)
    E.g.:  solver.integrate(restart)
        restart is to be set to True whenever there's a discontinuity in the
                forcing function, whereat the one-step method is applied
    Integrates for displacement and velocity over its current interval.  This
    procedure may be re-called multiple times between advancements of the
    solution along its path.

advance()
    E.g.:  solver.advance()
    Commit the solution and prepare the data base for advancing to the next
    step.  solver.advance() is to be called after the global solver has
    converged, and before solver.integrate() is to be called once again to move
    a solution along its path.

The following methods return values determined at the end of the most recently
integrated step.

getT()
    E.g.: t = getT()
    Returns a float for current time (the independent variable)

getU()
    E.g.: u = getU()
    Returns a vector for displacement (one of two response variables)

getV()
    E.g.: v = getV()
    Returns a vector for velocity (the second response variable)

getA()
    E.g.: a = getA()
    Returns a vector for acceleration (the control function)

getError()
    E.g.: error = getError()
    Returns an estimate for the current local truncation error

Reference:
    A. D. Freed, "A Technical Note: Two-Step PECE Methods for Approximating
    Solutions To First- and Second-Order ODEs", arXiv 1707.02125, 2017.
"""


class PECE(object):

    def __init__(self, u0, v0, t0, dt, aFn, p, pi, m=1):
        # verify the inputs

        # create the response array for displacements
        if isinstance(u0, np.ndarray):
            (dim,) = np.shape(u0)
            self.dim = dim
            # uPrev and uCurr are for previous and current nodes of integration
            # uPrevR & uCurrR enable reintegrating over the current step
            self.uPrev = np.zeros((dim,), dtype=float)
            self.uCurr = np.zeros((dim,), dtype=float)
            self.uPrevR = np.zeros((dim,), dtype=float)
            self.uCurrR = np.zeros((dim,), dtype=float)
            self.uCurrR[:] = u0[:]
        else:
            raise RuntimeError("Argument u0 must be a NumPy array.")

        # create the response array for velocities
        if isinstance(v0, np.ndarray):
            (dim,) = np.shape(v0)
            if dim != self.dim:
                raise RuntimeError("Vectors u0 and v0 have different sizes.")
            # vPrev and vCurr are for previous and current nodes of integration
            # vPrevR & vCurrR enable reintegrating over the current step
            self.vPrev = np.zeros((dim,), dtype=float)
            self.vCurr = np.zeros((dim,), dtype=float)
            self.vPrevR = np.zeros((dim,), dtype=float)
            self.vCurrR = np.zeros((dim,), dtype=float)
            self.vCurrR[:] = v0[:]
        else:
            raise RuntimeError("Argument v0 must be a NumPy array.")

        # assign time and its differential rate of change at the initial node
        if (isinstance(t0, float) and isinstance(dt, float) and
           dt > 100.0 * np.finfo(float).eps):
            self.dt = dt
            self.t = t0
            self.tR = t0
        else:
            raise RuntimeError("Arguments t0 and dt must be floats, " +
                               "and dt must be > 100 machine epsilon.")

        self.a = aFn
        # assert that aFn is a function that can be called
        if callable(aFn):
            self.a = aFn
        else:
            raise RuntimeError("Argument aFn must be a callable function.")
        a0 = aFn(t0, u0, v0, p, pi)
        if isinstance(a0, np.ndarray):
            (dim,) = np.shape(a0)
            if dim != self.dim:
                raise RuntimeError("Function aFn(t, u, v) must return a " +
                                   "vector of length {},\n".format(self.dim) +
                                   "instead it returned a vector of " +
                                   "length {}.".format(dim))
        else:
            raise RuntimeError("Function aFn(t, u, v) must return a NumPy " +
                               "array.")
        self.aPrev = np.zeros((dim,), dtype=float)
        self.aCurr = np.zeros((dim,), dtype=float)
        self.aPrevR = np.zeros((dim,), dtype=float)
        self.aCurrR = np.zeros((dim,), dtype=float)
        self.aCurrR[:] = a0[:]

        # limit the range for m in our implementation of PE(CE)^m
        if isinstance(m, int):
            if m < 0:
                self.m = 0
            elif m > 10:
                self.m = 10
            else:
                self.m = m
        else:
            raise RuntimeError("Argument m must be an integer.")

        # set a flag to select first-order method when starting an integration
        self.step = 1

        return  # a new integrator object

    def integrate(self, p, pi, restart=False):
        # assign the response variables and their rates (enables reintegration)
        self.t = self.tR
        self.uPrev[:] = self.uPrevR[:]
        self.uCurr[:] = self.uCurrR[:]
        self.vPrev[:] = self.vPrevR[:]
        self.vCurr[:] = self.vCurrR[:]
        self.aPrev[:] = self.aPrevR[:]
        self.aCurr[:] = self.aCurrR[:]

        if self.step == 1 or restart:
            # start or restart an integration with a one-step method
            t1 = self.t + self.dt
            # predict
            u1 = np.add(self.uCurr,
                        np.add(np.multiply(self.dt, self.vCurr),
                               np.multiply(self.dt**2 / 2.0, self.aCurr)))
            v1 = np.add(self.vCurr, np.multiply(self.dt, self.aCurr))
            uP = np.copy(u1)
            # evaluate
            a1 = self.a(t1, u1, v1, p, pi)
            for m in range(self.m):
                # correct
                u1 = np.add(self.uCurr,
                            np.subtract(np.multiply(self.dt / 2.0,
                                                    np.add(v1, self.vCurr)),
                                        np.multiply(self.dt**2 / 12.0,
                                                    np.subtract(a1,
                                                                self.aCurr))))
                v1 = np.add(self.vCurr,
                            np.multiply(self.dt / 2.0, np.add(a1, self.aCurr)))
                # re-evaluate
                a1 = self.a(t1, u1, v1, p, pi)
            # store the displacment vector for computing truncation error
            if self.m != 0:
                uC = np.copy(u1)
            else:
                uC = np.add(self.uCurr,
                            np.subtract(np.multiply(self.dt / 2.0,
                                                    np.add(v1, self.vCurr)),
                                        np.multiply(self.dt**2 / 12.0,
                                                    np.subtract(a1,
                                                                self.aCurr))))
        else:
            # continue integration with Freed's two-step method
            tN = self.t + self.dt
            # predict
            uNm1 = np.multiply(1.0 / 3.0,
                               np.subtract(np.multiply(4.0, self.uCurr),
                                           self.uPrev))
            duNm1 = np.multiply(self.dt / 6.0,
                                np.add(np.multiply(3.0, self.vCurr),
                                       self.vPrev))
            d2uNm1 = np.multiply(self.dt**2 / 36.0,
                                 np.subtract(np.multiply(31., self.aCurr),
                                             self.aPrev))
            uN = np.add(uNm1, np.add(duNm1, d2uNm1))
            vNm1 = np.multiply(1.0 / 3.0,
                               np.subtract(np.multiply(4.0, self.vCurr),
                                           self.vPrev))
            dvNm1 = np.multiply(2.0 * self.dt / 3.0,
                                np.subtract(np.multiply(2.0, self.aCurr),
                                            self.aPrev))
            vN = np.add(vNm1, dvNm1)
            uP = np.copy(uN)
            # evaluate
            aN = self.a(tN, uN, vN, p, pi)
            for m in range(self.m):
                # correct
                duN = np.multiply(self.dt / 24.0,
                                  np.add(vN, np.add(np.multiply(14.0,
                                                                self.vCurr),
                                                    self.vPrev)))
                d2uN = np.multiply(self.dt**2 / 72.0,
                                   np.add(np.multiply(10.0, aN),
                                          np.subtract(np.multiply(51.0,
                                                                  self.aCurr),
                                                      self.aPrev)))
                uN = np.add(uNm1, np.add(duN, d2uN))
                dvN = np.multiply(2.0 * self.dt / 3.0, aN)
                vN = np.add(vNm1, dvN)
                # re-evaluate
                aN = self.a(tN, uN, vN, p, pi)
            if self.m != 0:
                uC = np.copy(uN)
            else:
                duN = np.multiply(self.dt / 24.0,
                                  np.add(vN, np.add(np.multiply(14.0,
                                                                self.vCurr),
                                                    self.vPrev)))
                d2uN = np.multiply(self.dt**2 / 72.0,
                                   np.add(np.multiply(10.0, aN),
                                          np.subtract(np.multiply(51.0,
                                                                  self.aCurr),
                                                      self.aPrev)))
                uC = np.add(uNm1, np.add(duN, d2uN))

        # update the integrator data base
        self.t += self.dt
        self.uPrev[:] = self.uCurr[:]
        self.vPrev[:] = self.vCurr[:]
        self.aPrev[:] = self.aCurr[:]
        if self.step == 1 or restart:
            self.uCurr[:] = u1[:]
            self.vCurr[:] = v1[:]
            self.aCurr[:] = a1[:]
        else:
            self.uCurr[:] = uN[:]
            self.vCurr[:] = vN[:]
            self.aCurr[:] = aN[:]

        # compute the error in displacement
        magP = LA.norm(uP)
        magC = LA.norm(uC)
        self.error = abs(magC - magP) / max(1.0, magC)

        return  # nothing

    def advance(self):
        # advance the integrator's data
        self.step += 1
        self.tR = self.t
        self.uPrevR[:] = self.uPrev[:]
        self.vPrevR[:] = self.vPrev[:]
        self.aPrevR[:] = self.aPrev[:]
        self.uCurrR[:] = self.uCurr[:]
        self.vCurrR[:] = self.vCurr[:]
        self.aCurrR[:] = self.aCurr[:]
        return  # nothing

    # get the time
    def getT(self):
        return self.t

    # get the displacement (first solution)
    def getU(self):
        uVec = np.copy(self.uCurr)
        return uVec

    # get the velocity (second solution)
    def getV(self):
        vVec = np.copy(self.vCurr)
        return vVec

    # get the acceleration
    def getA(self):
        aVec = np.copy(self.aCurr)
        return aVec

    # get the local truncation error
    def getError(self):
        return self.error


"""
Changes made in version "1.0.0":

This is the initial version of Freed's PECE integrator for solving equations of
motion, e.g.,  M u" + C u' + K u = f(t)  where u is a vector of displacements
with u' and u" denoting its first and second derivatives in time, while f is a
forcing function of time t.  In this version, M is a mass matrix, C is a
damping matrix, and K is a stiffness matrix.

This code is a rework of Freed's original Python code found in:  peceAtoVandX.
"""
