#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA

"""
Module peceAtoVandX.py provides a PECE solver for second-order ODEs.

Copyright (c) 2017 Alan D. Freed

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
__version__ = "1.3.0"
__date__ = "07-16-2017"
__update__ = "09-24-2019"
__author__ = "Alan D. Freed"
__author_email__ = "adfreed@tamu.edu"

"""
This module solves the following system of ODEs
    dv/dt = a(t,x,v)    subject to initial conditions x0 and v0 at time  t0
where  t  is a real or float and where  x,  v  and  a  are vectors; they are
instances of np.ndarray.  Here  a  denotes acceleration,  v  denotes velocity,
and  x  denotes displacement, in a generalized sense, with  x  and  v
following from integration
    dx/dt = v  and  dv/dt = a
e.g., this has the structure of a classic dynamics problem.

A pair of predict/evaluate/correct/evaluate (PECE) schemes have been
implemented and solved sequentially. The method is third-order accurate.
    The predictors are:
        xPred = (1 / 3) (4 x_n - x_nm1)
              + (h / 6) (3 v_n + v_nm1)
              + (h^2 / 36) (31 a_n - a_nm1)
        vPred = (1 / 3) (4 v_n - v_nm1)
              + (2 h / 3) (2 a_n - a_nm1)
    at which point the acceleration is evaluated via a supplied function:
        aPred = a(t_n + h, xPred, vPred)
    The correctors are:
        xCorr = (1 / 3) (4 x_n - x_nm1)
              + (h / 24) (vPred + 14 v_n + v_nm1)
              + (h^2 / 72) (10 aPred + 51 a_n - a_nm1)
        vCorr = (1 / 3) (4 v_n - v_nm1)
              + (2 h / 3) aPred
    and acceleration is re-evaluated: aCorr = a(t_n + h, xCorr, vCorr).
The local solution advances along a sub-grid with a local step size  hL  that
is finer than the global step size  hG  in which it is embedded.  The global
step size is taken to be uniform over the integration, which itself may be
under the control of some external global driver.

Results are available at the end of the most recent global step.  The solver
itself does not return results.

The above integrator is not self starting, so a one-step method is used to
take the first integration step:
    the predictor is:
        xPred = x0 + h v0 + (h^2 / 2) a0
        vPred = v0 + h a0
    the corrector is:
        xCorr = x0 + (h / 2) (vPred + v0) - (h^2 / 12) (aPred - a0)
        vCorr = v0 + (h / 2) (aPred + a0)
with evaluations for acceleration occurring after each PC integration.

A PI controller dynamically adjusts the local step size for maximum speed,
while maintaining a user specified error tolerance.  Jumps in the local step
size are by increments of two, up to half the global step size.  Reductions in
local step size are by halves, with re-integrations taking place for those
steps whose error exceeds a specified error tolerance.

The local truncation error comes from taking a difference between the
predicted values and the corrected values; it is:
    xError = ||xCorr - xPred||
    if ||xCorr|| < 1  then  an absolute measure of error is used
        error_n = xError
    else  a relative measure of error is used
        error_n = xError / ||xCorr||

A PI controller adjusts the local step size according to the algorithm:
    if error_n <= tol and error_nm1 <= tol then
        use the PI controller
        coef = (tol / error_n)**(gainI / 4) * (error_nm1 / tol)^(gainP / 4)
    else
        use the I controller
        coef = (tol / error_n)**(1 / 3)
    if coef > 2 then
        if steps2go is even and it is greater than 3 then
            double the step size for the next integration step
        else if coef >= 1 then
            maintain the current step size
        else if coef < 1 and error_n <= tol then
            halve the step size for the next integration step
        else
            halve the step size
        repeat the integration step
    advance to the next step
    error_nm1 = error_n

The PI controller has parameters with typical values of:
    tol = 0.0001  tolerance permitted in the local truncation error
    gainI = 0.7   gain for the   integral   aspect of the controller
    gainP = 0.4   gain for the proportional aspect of the controller

Step halving is accomplished through a cubic Hermite interpolant:
        x_nmhalf = (x_n + x_nm1) / 2 - (hL / 8) * (v_n - v_nm1)
                 + O(h^4) + O(h^{p+1})
where the interpolation has accuracy O(h^4) while the solutions  x_nm1,  x_n,
v_nm1  and  v_n  have accuracies O(h^{p+1}) = O(h^4), so this applies in our
case.

Starting up the integrator requires some special attention.

To estimate the initial size for the first time step for use, compute
    h0 = ||x0|| / ||v0||
    t1 = t0 + h0
    x1 = x0 + h0 v0 + (h0^2 / 2) a0
    v1 = v0 + h0 a0
and then refine this estimate according to
    h1 = 2 | (||x1|| - ||x0||) / (||v0|| + ||v1||) |
with the steps-to-go required to advance to the first global node being
    S = max(2, round(hG / h1))
from which the initial local step size is obtained via
    hL = hG / S
It is not uncommon for the PI controller to adjust this estimate early on
before the algorithm begins to perform in optimal run mode.  Nevertheless,
this startup algorithm has proven to be more reliable than relying on a user
to supply a reasonable estimate for the initial local step size.

Methods

constructor
    E.g.:  solver = pece(aFn, t0, x0, v0, h, tol)
    Creates the object and initializes it.  Variable  aFn  is the function that
    describes acceleration: a = aFn(t, x, v).  Variable  t0  is the initial
    time, typically  t0 = 0.  Variable  x0  is the initial condition for
    displacement, while  v0  is the initial condition for velocity.  Variable
    h  is the size of the global time step.  And variable  tol  is the
    tolerance that is to bound the local truncation error from above.

integrate()
    E.g.:  solver.integrate()
    Integrates displacement and velocity to the end of the current global step.
    This procedure may be recalled multiple times between advancements of a
    solution.

advance()
    E.g.:  solver.advance()
    Commit the solution and prepare the data base for advancing to the next
    global step.  'advance' is to be called after the global solver has
    converged, and before 'integrate' is called for advancing the solution
    across the next global step.

The following methods return values determined at the end of the most recently
integrated global step.

getStatistics()
    E.g.:  n, n_doubled, n_halved, n_reruns = solver.getStatistics()

getT()
    E.g.: t = getT()
    Returns time (the independent variable).

getX()
    E.g.: x = getX()
    Returns displacement (one of two dependent variables)

getV()
    E.g.: v = getV()
    Returns velocity (the other dependent variable)

getA()
    E.g.: a = getA()
    Returns acceleration (the control function)

getError()
    E.g.: error = getError()
    Returns the local truncation error

There are occasions when a global step does not land at a time where a solution
is desired.  A means for a dense output is required.  A cubic Hermite
interpolator is employed to meet this objective.  Specifically

interpolateForX(atT)
    E.g.: xAtT = solver.interpolateForX(atT)
    Returns an estimate of the solution at any location, viz., 'atT', that lies
    within interval [t_n-1, t_n], i.e., a global step back.

interpolateForV(atT)
    E.g.: vAtT = solver.interpolateForV(atT)
    Returns an estimate for the velocity at any location, viz., 'atT', that
    lies within interval [t_n-1, t_n], i.e., a global step back.

References:
    G. S\"oderlind, "Automatic control and adaptive time-stepping", Numerical
    Algorithms, Vol. 31 (2002), 281-310.

    A. D. Freed, "A Technical Note: Two-Step PECE Methods for Approximating
    Solutions to First- and Second-Order ODEs." Texas A&M University, 2017.
"""


class pece(object):

    def __init__(self, aFn, t0, x0, v0, h, tol=0.0001):
        self.committed = False

        self.a = aFn

        if isinstance(t0, float):
            self.t_nm1 = t0
        else:
            raise RuntimeError("Variable 't0' must be a float.")

        if isinstance(x0, np.ndarray):
            self.x_nm1 = x0
            (rows,) = x0.shape
        else:
            raise RuntimeError("Initial condition 'x0' must be a numpy.array.")

        if isinstance(v0, np.ndarray):
            if v0.shape == (rows,):
                self.v_nm1 = v0
            else:
                raise RuntimeError("Initial conditions 'x0' and 'v0' " +
                                   "have different lengths.")
        else:
            raise RuntimeError("Initial condition 'v0' must be a numpy.array.")

        a0 = aFn(t0, x0, v0)
        if (a0.shape == (rows,)):
            self.a_nm1 = a0
        else:
            raise RuntimeError("Function 'aFn' and variables 'x0' and 'v0' " +
                               "are incompatible.")

        if isinstance(h, float) and (h > 0.0):
            self.hG = h
        else:
            raise RuntimeError("Stepsize 'h' must be positive float.")

        if isinstance(tol, float) and (tol >= 0.00000001) and (tol <= 0.1):
            self.errTol = tol
        else:
            self.errTol = 0.0001
            print("A default value for error tolerance was set at 0.0001.")

        # acquire a first estimate for the initial step size
        magx0 = LA.norm(x0)
        magv0 = LA.norm(v0)
        if magv0 > 0.0:
            h0 = magx0 / magv0
            if h0 > h / 10.0:
                h0 = h / 10.0
            if h0 < h / 100.0:
                h0 = h / 100.0
        else:
            h0 = h / 100.0

        # predict
        t1 = t0 + h0
        xp = np.add(x0, np.add(np.multiply(h0, v0),
                               np.multiply(h0 * h0 / 2.0, a0)))
        vp = np.add(v0, np.multiply(h0, a0))
        # evaluate
        ap = aFn(t1, xp, vp)
        # correct
        x1 = np.add(x0, np.subtract(np.multiply(h0 / 2.0, np.add(vp, v0)),
                                    np.multiply(h0 * h0 / 12.0,
                                                np.subtract(ap, a0))))
        v1 = np.add(v0, np.multiply(h0 / 2., np.add(ap, a0)))

        # get an improved estimate for initial step size
        magx1 = LA.norm(x1)
        magv1 = LA.norm(v1)
        if (magv0 + magv1) > 0.0:
            h1 = 2. * abs((magx1 - magx0) / (magv0 + magv1))
            if h1 < h / 1000.0:
                h1 = h / 1000.0
        else:
            h1 = h / 1000.0

        # compute number of steps using h1 that it will take to get to hG
        self.steps2go = int(round(h / h1))
        if self.steps2go < 2:
            self.steps2go = 2
        # determine the initial local step size
        self.hL = self.hG / self.steps2go

        # attributes that are assigned default values
        self.error_nm1 = 1.0  # forces conventional controller to be used first
        self.gainI = 0.7      # gain on the I in the PI controller
        self.gainP = 0.4      # gain on the P in the PI controller
        self.maxSteps = 100   # maximum number of iterations allowed per step

        # counters that need to be initialized
        self.n = 0            # step count at the current node
        self.nDoubled = 0     # number of steps where step size was doubled
        self.nHalved = 0      # number of steps where step size was halved
        self.nReruns = 0      # number of steps where integration was redone

        return  # nothing

    # take the first step of integration
    def takeFirstStep(self):
        t0 = self.t_nm1
        x0 = self.x_nm1
        v0 = self.v_nm1
        a0 = self.a_nm1

        # ensure that the solution satisfies the required error tolerance
        error = 1
        step = 0
        while (error > self.errTol) and (step < self.maxSteps):
            step = step + 1
            if step == self.maxSteps:
                raise RuntimeError("First step did not converge in {} steps."
                                   .format(self.maxSteps))

            # use a one-step method to take the first step of integration
            h1 = self.hL
            t1 = self.t_nm1 + self.hL
            # predict
            xp = np.add(x0, np.add(np.multiply(h1, v0),
                                   np.multiply(h1 * h1 / 2., a0)))
            vp = np.add(v0, np.multiply(h1, a0))
            # evaluate
            ap = self.a(t1, xp, vp)
            # correct
            x1 = np.add(x0, np.subtract(np.multiply(h1 / 2.0, np.add(vp, v0)),
                                        np.multiply(h1 * h1 / 12.0,
                                                    np.subtract(ap, a0))))
            v1 = np.add(v0, np.multiply(h1 / 2.0, np.add(ap, a0)))
            # re-evaluate
            a1 = self.a(t1, x1, v1)

            # compute estimates for the local truncation errors
            error = (h1 / 2.0) * LA.norm(np.subtract(np.subtract(vp, v0),
                                         np.multiply(h1 / 6.0, np.add(ap,
                                                     np.multiply(5.0, a0)))))
            magErr = LA.norm(x1)
            if magErr > 1.0:
                error = error / magErr

            # if error is too large, repeat the step using a smaller step size
            if error > self.errTol:
                self.hL = self.hL / 2.0
                self.steps2go = 2 * self.steps2go

            # end while loop

        self.error_n = error
        self.t_n = t1
        self.x_n = x1
        self.v_n = v1
        self.a_n = a1

        # Assign values so global step can be rerun using this initial state.
        # Does not need to be restarted from step 0, but from step 1, so this
        # procedure does not have to be recalled on first global-step repeats.
        self.steps2go_curr = self.steps2go - 1
        self.hL_curr = h1

        self.n_curr = 1
        self.nDoubled_curr = 0
        self.nHalved_curr = 0
        self.nReruns_curr = 0

        self.error_prev = 1.0
        self.error_curr = error

        self.t_last = t0
        self.t_prev = t0
        self.t_curr = t1
        self.x_last = x0
        self.x_prev = x0
        self.x_curr = x1
        self.v_last = v0
        self.v_prev = v0
        self.v_curr = v1
        self.a_last = a0
        self.a_prev = a0
        self.a_curr = a1

        # the first integration step has taken place, update the data base

        self.n = 1
        self.steps2go = self.steps2go - 1

        return  # nothing

    def integrate(self):
        self.advanced = False
        # prepare the data structure to take this next step of integration
        if self.n == 0:
            self.takeFirstStep()
        else:
            # load the data structure from its most recently advanced solution
            self.steps2go = self.steps2go_curr
            self.hL = self.hL_curr

            self.n = self.n_curr
            self.nDoubled = self.nDoubled_curr
            self.nHalved = self.nHalved_curr
            self.nReruns = self.nReruns_curr

            self.error_nm1 = self.error_prev
            self.error_n = self.error_curr

            self.t_nm1 = self.t_prev
            self.t_n = self.t_curr
            self.x_nm1 = self.x_prev
            self.x_n = self.x_curr
            self.v_nm1 = self.v_prev
            self.v_n = self.v_curr
            self.a_nm1 = self.a_prev
            self.a_n = self.a_curr

        # integrate over the global step
        while self.steps2go > 0:
            h = self.hL
            t = self.t_n + self.hL

            # integrate with the predictors
            #    xPred = (1 / 3) (4 x_n - x_nm1)
            #          + (h / 6) (3 v_n + v_nm1)
            #          + (h^2 / 36) (31 a_n - a_nm1)
            #    vPred = (1/3) (4 v_n - v_nm1)
            #          + (2h/3) (2 a_n - a_nm1)
            x = np.multiply(1.0 / 3.0,
                            np.subtract(np.multiply(4.0, self.x_n),
                                        self.x_nm1))
            dxp = np.multiply(1.0 / 6.0,
                              np.add(np.multiply(3.0, self.v_n), self.v_nm1))
            d2xp = np.multiply(1.0 / 36.0,
                               np.subtract(np.multiply(31., self.a_n),
                                           self.a_nm1))
            xp = np.add(x, np.multiply(h, np.add(dxp, np.multiply(h, d2xp))))

            v = np.multiply(1.0 / 3.0,
                            np.subtract(np.multiply(4.0, self.v_n),
                                        self.v_nm1))
            dvp = np.multiply(2.0 / 3.0,
                              np.subtract(np.multiply(2.0, self.a_n),
                                          self.a_nm1))
            vp = np.add(v, np.multiply(h, dvp))
            # evaluate
            ap = self.a(t, xp, vp)
            # integrate with the correctors
            #    xCorr = (1 / 3) (4 x_n - x_nm1)
            #          + (h / 24) (vPred + 14 v_n + v_nm1)
            #          + (h^2 / 72) (10 aPred + 51 a_n - a_nm1)
            #    vCorr = (1/3) (4 v_n - v_nm1)
            #          + (2h/3) ap
            dxc = np.multiply(1.0 / 24.0, np.add(vp,
                              np.add(np.multiply(14.0, self.v_n), self.v_nm1)))
            d2xc = np.multiply(1.0 / 72.0, np.add(np.multiply(10.0, ap),
                               np.subtract(np.multiply(51.0, self.a_n),
                                           self.a_nm1)))
            xc = np.add(x, np.multiply(h, np.add(dxc, np.multiply(h, d2xc))))

            dvc = np.multiply(2.0 / 3.0, ap)
            vc = np.add(v, np.multiply(h, dvc))
            # re-evaluate
            ac = self.a(t, xc, vc)

            # update the local truncation errors
            self.error_nm1 = self.error_n
            self.error_n = h * LA.norm(np.add(np.subtract(dxc, dxp),
                                       np.multiply(h, np.subtract(d2xc,
                                                                  d2xp))))
            magErr = LA.norm(xc)
            if magErr > 1.0:
                self.error_n = self.error_n / magErr

            # run PI controller to manage step size for maintaining accuracy
            if self.error_n < self.errTol:
                if self.error_n < np.finfo(float).eps:
                    self.error_n = np.finfo(float).eps
                # integration step had an admissible local truncation error
                self.n = self.n + 1

                # run the PI controller
                if self.error_nm1 <= self.errTol:
                    # prior integration step completed successfully, use PI
                    coef = ((self.errTol / self.error_n)**(self.gainI / 4.0) *
                            (self.error_nm1 / self.errTol)**(self.gainP / 4.0))
                else:
                    # prior integration step was rerun, use I controller
                    coef = (self.errTol / self.error_n)**(1.0 / 3.0)

                # update the data structure to prepare for the next step
                if ((coef > 2.0) and (self.steps2go > 2) and
                   (self.steps2go % 2 == 0)):
                    # the following line of code IS NECESSARY
                    t = t - self.hL
                    # double the current step size and advance
                    self.hL = 2.0 * self.hL
                    self.nDoubled = self.nDoubled + 1
                    self.steps2go = int(self.steps2go // 2)
                    # variables at step n-1 do not get updated in this case

                elif (coef >= 1.0):
                    self.steps2go = self.steps2go - 1
                    # maintain the current step size and advance
                    self.t_nm1 = self.t_n
                    self.x_nm1 = self.x_n
                    self.v_nm1 = self.v_n
                    self.a_nm1 = self.a_n

                else:
                    self.steps2go = self.steps2go - 1
                    # halve the current step size and advance
                    self.hL = self.hL / 2.0
                    self.nHalved = self.nHalved + 1
                    self.steps2go = 2 * self.steps2go
                    # use Hermite interpolation to get values at the midpoint
                    self.t_nm1 = t - self.hL
                    self.x_nm1 = np.subtract(
                                    np.multiply(0.5,
                                                np.add(xc, self.x_n)),
                                    np.multiply(self.hL / 4.0,
                                                np.subtract(vc, self.v_n)))
                    self.v_nm1 = np.subtract(
                                    np.multiply(0.5,
                                                np.add(vc, self.v_n)),
                                    np.multiply(self.hL / 4.0,
                                                np.subtract(ac, self.a_n)))
                    self.a_nm1 = self.a(self.t_nm1, self.x_nm1, self.v_nm1)

                # attribute updates common to all successful steps
                self.t_n = t
                self.x_n = xc
                self.v_n = vc
                self.a_n = ac

            else:
                # local truncation error too large, integration must be redone
                self.hL = self.hL / 2.0
                self.nReruns = self.nReruns + 1
                self.steps2go = 2 * self.steps2go

                # use Hermite interpolation to get values at the midpoint
                self.t_nm1 = self.t_n - self.hL
                self.x_nm1 = np.subtract(
                                np.multiply(0.5,
                                            np.add(self.x_n, self.x_nm1)),
                                np.multiply(self.hL / 4.0,
                                            np.subtract(self.v_n, self.v_nm1)))
                self.v_nm1 = np.subtract(
                                np.multiply(0.5,
                                            np.add(self.v_n, self.v_nm1)),
                                np.multiply(self.hL / 4.0,
                                            np.subtract(self.a_n, self.a_nm1)))
                self.a_nm1 = self.a(self.t_nm1, self.x_nm1, self.v_nm1)

                # time, displacement and velocity do not advance: reintegrate
                self.error_n = 1.0  # forces classic controller for next step

            # end while loop

        return  # nothing

    def advance(self):
        # update data for the local step to prepare for the next global step
        self.committed = True

        # curr denotes the end of both the global and local steps
        # prev denotes the end of the prior local step
        # last denotes the end of the prior global step
        self.t_last = self.t_curr
        self.t_curr = self.t_n
        self.x_last = self.x_curr
        self.x_curr = self.x_n
        self.v_last = self.v_curr
        self.v_curr = self.v_n
        self.a_last = self.a_curr
        self.a_curr = self.a_n

        # account for discontinuity in local step size across two global steps
        hPrev = self.hL
        self.steps2go = int(round(self.hG / self.hL))
        hNext = self.hG / self.steps2go
        self.hL = hNext

        if not (((hPrev / hNext) > (1.0 - 100.0 * np.finfo(float).eps)) and
                ((hPrev / hNext) < (1.0 + 100.0 * np.finfo(float).eps))):

            t_nm1 = self.t_n - hNext
            x_nm1 = self.interpolateForX(t_nm1)
            v_nm1 = self.interpolateForV(t_nm1)
            a_nm1 = self.a(t_nm1, x_nm1, v_nm1)

            self.t_nm1 = t_nm1
            self.x_nm1 = x_nm1
            self.v_nm1 = v_nm1
            self.a_nm1 = a_nm1

        # update the data structure to enable reruns of the global step
        self.t_prev = self.t_nm1
        self.x_prev = self.x_nm1
        self.v_prev = self.v_nm1
        self.a_prev = self.a_nm1

        self.steps2go_curr = self.steps2go
        self.hL_curr = self.hL

        self.n_curr = self.n
        self.nDoubled_curr = self.nDoubled
        self.nHalved_curr = self.nHalved
        self.nReruns_curr = self.nReruns

        self.error_prev = self.error_nm1
        self.error_curr = self.error_n

        return  # nothing

    def getStatistics(self):
        if not self.committed:
            raise RuntimeError("Call getStatistics after a commit and " +
                               "before the next integrate.")
        n = self.n_curr
        nd = self.nDoubled_curr
        nh = self.nHalved_curr
        nr = self.nReruns_curr
        return n, nd, nh, nr

    # get the time at the end of the global step
    def getT(self):
        if not self.committed:
            raise RuntimeError("Call getT after a commit and " +
                               "before the next integrate.")
        t = self.t_curr
        return t

    # get the displacement (or solution) at the end of the global step
    def getX(self):
        if not self.committed:
            raise RuntimeError("Call getX after a commit and " +
                               "before the next integrate.")
        x = self.x_curr
        return x

    # get the velocity (or ode) at the end of the global step
    def getV(self):
        if not self.committed:
            raise RuntimeError("Call getV after a commit and " +
                               "before the next integrate.")
        v = self.v_curr
        return v

    # get the acceleration at the end of the global step
    def getA(self):
        if not self.committed:
            raise RuntimeError("Call getA after a commit and " +
                               "before the next integrate.")
        a = self.a_curr
        return a

    # get the local truncation error
    def getError(self):
        if not self.committed:
            raise RuntimeError("Call getError after a commit and " +
                               "before the next integrate.")
        e = self.error_curr
        return e

    # return an interpolation of the solution at the specified time
    def interpolateForX(self, atT):
        # cubic Hermite interpolation over the most recent global step
        if not self.committed:
            raise RuntimeError("Call interpolate after a commit and " +
                               "before the next integrate.")
        if not ((atT > (1.0 - 100.0 * np.finfo(float).eps) * self.t_last) and
                (atT < (1.0 + 100.0 * np.finfo(float).eps) * self.t_curr)):
            raise RuntimeError("Requested an extrapolation, " +
                               "not an interpolation.")
        h = self.t_curr - self.t_last
        theta = (atT - self.t_last) / h
        oneMtheta = 1.0 - theta
        oneMtwoTheta = 1.0 - 2.0 * theta
        x1 = np.add(np.multiply(theta, self.x_curr),
                    np.multiply(oneMtheta, self.x_last))
        x2 = np.multiply(oneMtwoTheta, np.subtract(self.x_curr, self.x_last))
        x3 = np.subtract(np.multiply(h * theta, self.v_curr),
                         np.multiply(h * oneMtheta, self.v_last))
        xAtT = np.subtract(x1, np.multiply(theta * oneMtheta, np.add(x2, x3)))
        return xAtT

    # return an interpolation for the velocity at the specified time
    def interpolateForV(self, atT):
        # cubic Hermite interpolation over the most recent global step
        if not self.committed:
            raise RuntimeError("Call interpolate after a commit and " +
                               "before the next integrate.")
        if not ((atT > (1.0 - 100.0 * np.finfo(float).eps) * self.t_last) and
                (atT < (1.0 + 100.0 * np.finfo(float).eps) * self.t_curr)):
            raise RuntimeError("Requested an extrapolation, " +
                               "not an interpolation.")
        h = self.t_curr - self.t_last
        theta = (atT - self.t_last) / h
        oneMtheta = 1.0 - theta
        oneMtwoTheta = 1.0 - 2.0 * theta
        v1 = np.add(np.multiply(theta, self.v_curr),
                    np.multiply(oneMtheta, self.v_last))
        v2 = np.multiply(oneMtwoTheta, np.subtract(self.v_curr, self.v_last))
        v3 = np.subtract(np.multiply(h * theta, self.a_curr),
                         np.multiply(h * oneMtheta, self.a_last))
        vAtT = np.subtract(v1, np.multiply(theta * oneMtheta, np.add(v2, v3)))
        return vAtT
