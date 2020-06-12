#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA

"""
Module peceHE.py provides a PECE solver for Hypo-Elastic like constitutive
equations plus classes for the control and response functions required by them.

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
__date__ = "07-16-2017"
__update__ = "05-29-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


This module provides three classes:
    pece        implements a PECE solver for first-order differential equations
    control     provides the user with an interface for the control variables
    response    provides the user with an interface for the response variables

The PECE solver solves the following system of ODEs typical of hypo-elasticity
    dy/dt = dy/dx dx/dt
or, more specifically, for our application
    dy/dt = dy/de de/dx dx/dt  or  dy/dt = f(x, y) g(x) dx/dt
where
    f(x, y) = dy/de
        a matrix function establishing the constitutive response for a system
    g(x) = de/dx
        a matrix function converting physical into thermodynamic control rates
subject to an initial condition of
    y_0  at time  t_0
where
    x   is a vector of physical control variables, e.g., stretches
    e   is a vector of thermodynamic control variables, e.g., strains
    y   is a vector of thermodynamic response variables, e.g., stresses
    t   is a floating point number that denotes time
All vectors and matrices are instances of NumPy.ndarray with float elements.

Two Adams-Moulton (predictor/corrector) methods are implemented: a one-step
method for starting an integration, and a two-step method for continuing it.

In the descriptions for the integrator described below, denote
    x_p, y_p, dx_p/dt, dy_p/dt  associate with the previous integration node
    x_c, y_c, dx_c/dt, dy_c/dt  associate with the current integration node
    x_n, y_n, dx_n/dt, dy_n/dt  associate with the next integration node
    yPred and dyPred/dt         represent predicted values for y_n and dy_n/dt

The integrator given below is not self starting, so a one-step method is used
to take the first step of integration, and whenever the integrator is to be
restarted because of a discontinuity in the control variables; specifically,
    Predict:
        yPred = y_0 + dy_0/dt dt
    Evaluate:
        dyPred/dt = f(x_1, yPred) g(x_1) dx_1/dt
    Correct:
        y_1 = y_0 + (1/2)(dyPred/dt + dy_0/dt) dt
    re-Evaluate:
        dy_1/dt = f(x_1, y_1) g(x_1) dx_1/dt
where
    dy_0/dt = f(x_0, y_0) g(x_0) dx_0/dt
with evaluations for functions f and g occurring after each integration.
This is the Adams-Moulton method of Heun for integrating a first-order ODE.

A second-order accurate PECE method is used.  It is described by the equations:
    Predict:
        yPred = (1/3)(4 y_c - y_p) + (2/3)(2 dy_c/dt - dy_p/dt) dt
    Evaluate:
        dyPred/dt = f(x_n, yPred) g(x_n) dx_n/dt
    Correct:
        y_n = (1/3)(4 y_c - y_p) + (2/3) dyPred/dt dt
    re-Evaluate:
        dy_n/dt = f(x_n, y_n) g(x_n) dx_n/dt
This is the Adams-Moulton method of Freed for integrating a first-order ODE.


object constructor

    E.g.:  solver = pece(ctrl, resp, m=1)
        ctrl    an instance of type 'control' (see below):
        resp    an instance of type 'response' (see below):
        m       the number of CE iterations, i.e., PE(CE)^m, m in [1, 5]
    Creates an object that integrates hypo-elastic like constitutive equations
    using a PE(CE)^m method.

methods

integrate(ctrlVec, restart=False)
    E.g.:  solver.integrate(restart)
        ctrlVec is a vector containing control variables for the next node
        restart is to be set to True whenever there's a discontinuity in one of
                the control variables; Heun's one-step method will be called
    Integrates an ODE over its current interval, viz., from the current node
    to the next node.  This procedure may be re-called multiple times between
    advancements of a solution.

advance()
    E.g.:  solver.advance()
    Commits a solution, thereby preparing the data base for advancing the
    integrator to the next step.  solver.advance() is to be called after the
    global FE solver has converged, and before solver.integrate() is called
    once again to travel a solution along its path.

The following methods return values determined at the end of the current
integration step.  They are to be called after solver.advance and before the
next call to solver.integrate.

getT()
    E.g.: t = solver.getT()
    Returns a float for time

getX()
    E.g.: x = solver.getX()
    Returns a vector containing the physical control variables

getY()
    E.g.: y = solver.getY()
    Returns a vector containing response variables

getError()
    E.g.: error = solver.getError()
    Returns an estimate for the local truncation error


A base class has been created for the control variables that is to be extended
in one's application.  Here the methods return values that associate with the
next node.  The interface for this class is:


object constructor

    E.g.: ctrl = control(ctrlVec0, dt)
        ctrlVec0    a vector of control variables at the reference node, x0
        dt          size of the time step to be used for integration

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    node            an integer specifying the current node of integration,
                    which is reset to 0 whenever the integrator is restarted
    dt              a floating point number specifying the time-step size
    xR              a vector holding control variables for the reference node
    xP              a vector holding control variables for the previous node
    xC              a vector holding control variables for the current node
    xN              a vector holding control variables for the next node

methods

update(ctrlVec)
    E.g., ctrl.update(ctrlVec)
        ctrlVec     a vector of control variables for the next node
    ctrl.update may be called multiple times before freezing its values with a
    call to ctrl.advance.

advance()
    E.g., ctrl.advance()
    Updates the object's data structure in preparation for the next integration
    step.  This method is called internally by the pece object and should not
    be called by the user.

dedx()
    E.g., dedxMtx = ctrl.dedx()
        dedxMtx     a matrix containing the mapping of physical control rates
                    into thermodynamic control rates.
    This transformation associates with the next node.  It is a phantom method
    that must be overwritten.

dxdt(restart)
    E.g., dxdtVec = ctrl.dxdt(restart=False)
        dxdtVec     is a vector containing the rate-of-change in control
        restart     set to True whenever there is a discontinuity in control
    This base method implements finite difference formulae to approximate this
    derivative.  A first-order difference formula is used for the reference and
    first nodes, plus whenever a restart is mandated.  A second-order backward
    difference formula is used for all other nodes.  All derivatives associate
    with the next node.  If these approximations are not appropriate for your
    application, then you will need to overwrite them.


Another base class has been created for the response variables that is also to
be extended in one's application.  Here all methods return fields associated
with the next node.  The interface for this class is:

object constructor

    E.g.: resp = response(ctrlVec0, respVec0)
        ctrlVec0    a vector of control variables at the reference node, xR
        respVec0    a vector of response variables at the reference node, yR

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    responses       an integer specifying the number of response variables
    xR              a vector containing the initial condition for control
    yR              a vector containing the initial condition for response

methods

tanMod(ctrlVec, respVec)
    E.g., dyde = resp.tanMod(ctrlVec, respVec)
        dyde        a matrix of tangent moduli, i.e., constitutive equation
        ctrlVec     the vector of physical control variables, i.e., x
        respVec     the vector of response variables, viz, y
    The constitutive equation considered is hypo-elastic like; specifically,
        dy/dt = dy/de de/dx dx/dt
    wherein
        dy/dt       a vector of response rates
        dy/de       a matrix of constitutive tangent moduli
        de/dx       a matrix that transforms physical to thermodynamic rates
        dx/dt       a vector of physical control rates
    This function returns a matrix dy/de.

isRuptured()
    E.g., ruptured = resp.isRuptured()
        ruptured    True if the material has ruptured; False otherwise

rupturedRespVec(ctrlVec, respVec)
    E.g., yVec = resp.rupturedRespVec(ctrlVec, respVec)
        yVec        the response vector after rupture
        ctrlVec     the vector of control variables, i.e, x
        respVec     the vector of response variables before rupture
    The supplied respVec contains the response prior to rupture, while the
    returned yVec contains the response variables after rupture.

rupturedTanMod(ctrlVec, respVec)
    E.g., dyde = resp.rupturedTanMod(ctrlVec, respVec)
        dyde        a matrix of tangent moduli, i.e., constitutive equation
        ctrlVec     the vector of control variables, i.e., x
        respVec     the vector of response variables, i.e., y
    This method returns the tangent matrix dy/de for ruptured states.


Templates for using the above control and response objects are provided below.


class myCtrl(control):

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    node            an integer specifying the current node of integration,
                    which is reset to 0 whenever the integrator is restarted
    dt              a floating point number specifying the time-step size
    xR              a vector holding control variables for the reference node
    xP              a vector holding control variables for the previous node
    xC              a vector holding control variables for the current node
    xN              a vector holding control variables for the next node

methods

    def __init__(self, ctrlVec0, dt):
        # Call the constructor of the base type.
        super().__init__(ctrlVec0, dt)
        # This creates the counter self.node, the number of control variables
        # as self.controls, and the time-step size of integration dt, all of
        # which are useful fields when extending this class.
        # Verify and initialize other data, as required.
        return  # a new instance of type myCtrl

    def update(self, ctrlVec):
        # Call the base implementation of this method to insert this vector
        # into the class' data structure.
        super().update(ctrlVec)
        # Update any other fields pertinent to the user's implementation.
        return  # nothing

    def advance(self):
        # Call the base implementation of this method to advance its data.
        # This moves current data to their previous fields, and then the next
        # data gets moved to their current fields.
        super().advance()
        # Advance any additional data introduced by the user.
        # This method is called internally by the pece integrator and should
        # not be called by the user.
        return  # nothing

    def dedx(self):
        # Call the base implementation of this method to create dedxMtx.
        dedxMtx = super().dedx()
        # This method must be overwritten for one's application.
        return dedxMtx

    def dxdt(self, restart=False):
        # Call the base implementation of this method to create dxdtVec.
        dxdtVec = super().dxdt(restart)
        # The returned dxdtVec is computed via finite difference formulae;
        # specifically,
        #   if self.node is 0, 1   use first-order difference formula
        #   if restart is True     use first-order difference formula
        #   otherwise              use second-order backward difference formula
        # Overwrite dxdtVec if this is not appropriate for your application.
        return dxdtVec


class myResp(response):

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    responses       an integer specifying the number of response variables
    xR              a vector containing the initial condition for control
    yR              a vector containing the initial condition for response

methods

    def __init__(self, ctrlVec0, respVec0):
        # call the base type to verify the inputs and to create variables
        #    self.controls  the number of control variables
        #    self.responses the number of response variables
        #    self.xR        the initial condition for the control variables
        #    self.yR        the initial condition for the response variables
        super().__init__(ctrlVec0, respVec0)
        # verify and initialize other data, as required.
        return  # a new instance of type myResp

    def tanMod(self, ctrlVec, respVec):
        # call the base type to verify the inputs and to create matrix dyde
        dyde = super().tanMod(ctrlVec, respVec)
        # populate the entries of dyde for the tangent moduli below
        return dyde

    def isRuptured(self):
        # no super call is required here, ruptured is to have a boolean value
        return self.ruptured

    def rupturedRespVec(self, ctrlVec, respVec):
        # call the base type to verify the input and to create vector rVec
        rVec = super().rupturedRespVec(ctrlVec, respVec)
        # populate the entries for the ruptured response in rVec below
        # this will result in a discontinuity in the response fields
        return rVec

    def rupturedTanMod(self, ctrlVec, respVec):
        # call the base type to verify the inputs and to create matrix dyde
        dyde = super().rupturedTanMod(ctrlVec, respVec)
        # populate the entries of dyde for the ruptured tangent moduli below
        return dyde

Reference:
    A. D. Freed, "A Technical Note: Two-Step PECE Methods for Approximating
    Solutions To First- and Second-Order ODEs", arXiv 1707.02125, 2017.
"""


class control(object):

    def __init__(self, ctrlVec0, dt):
        if isinstance(ctrlVec0, np.ndarray):
            (controls,) = np.shape(ctrlVec0)
            self.controls = controls
        else:
            raise RuntimeError("Argument ctrlVec0 must by a NumPy array.")
        if isinstance(dt, float) and dt > np.finfo(float).eps:
            self.dt = dt
        else:
            raise RuntimeError("Argument dt must be greater than 100 times " +
                               "machine epsilon.")
        # create the object's remaining internal data structure
        self.node = 0                                        # integration node
        self.xR = np.zeros((controls,), dtype=float)         # reference state
        self.xP = np.zeros((controls,), dtype=float)         # previous state
        self.xC = np.zeros((controls,), dtype=float)         # current state
        self.xN = np.zeros((controls,), dtype=float)         # next state
        # initialize these vector fields
        self.xR[:] = ctrlVec0[:]
        self.xP[:] = ctrlVec0[:]
        self.xC[:] = ctrlVec0[:]
        self.xN[:] = ctrlVec0[:]
        return  # a new instance of a control object

    def update(self, ctrlVec):
        if isinstance(ctrlVec, np.ndarray):
            (controls,) = np.shape(ctrlVec)
            if self.controls != controls:
                raise RuntimeError("Argument ctrlVec must have a length of " +
                                   "{}, but it had ".format(self.controls) +
                                   "a length of {}.".format(controls))
        else:
            raise RuntimeError("Argument ctrlVec must by a NumPy array.")
        self.xN[:] = ctrlVec[:]
        return  # nothing

    def advance(self):
        self.node += 1
        self.xP[:] = self.xC[:]
        self.xC[:] = self.xN[:]
        return  # nothing

    def dedx(self):
        de = np.eye(self.controls, dtype=float)
        return de

    def dxdt(self, restart=False):
        dx = np.zeros((self.controls,), dtype=float)
        if restart is True:
            self.node = 0
        if self.node <= 1:
            # use the first-order forward/backward difference formula
            dx[:] = (self.xN[:] - self.xC[:]) / self.dt
        else:
            # use the second-order backward difference formula
            dx[:] = ((3.0 * self.xN[:] - 4.0 * self.xC[:] + self.xP[:]) /
                     (2.0 * self.dt))
        return dx


class response(object):

    def __init__(self, ctrlVec0, respVec0):
        # verify inputs
        if isinstance(ctrlVec0, np.ndarray):
            (controls,) = np.shape(ctrlVec0)
            self.controls = controls
            self.xR = np.zeros((controls,), dtype=float)
            self.xR[:] = ctrlVec0[:]
        else:
            raise RuntimeError("Argument ctrlVec0 must by a NumPy array.")
        if isinstance(respVec0, np.ndarray):
            (responses,) = np.shape(respVec0)
            self.responses = responses
            self.yR = np.zeros((responses,), dtype=float)
            self.yR[:] = respVec0[:]
        else:
            raise RuntimeError("Argument respVec0 must by a NumPy array.")
        return  # a new instance of this base type

    def tanMod(self, ctrlVec, respVec):
        # verify inputs
        if isinstance(ctrlVec, np.ndarray):
            (controls,) = np.shape(ctrlVec)
            if controls != self.controls:
                raise RuntimeError("The ctrlVec sent had a length of " +
                                   "{}, but it must have ".format(controls) +
                                   "a length of {}.".format(self.controls))
        else:
            raise RuntimeError("Argument ctrlVec must be a NumPy array.")
        if isinstance(respVec, np.ndarray):
            (responses,) = np.shape(respVec)
            if responses != self.responses:
                raise RuntimeError("The respVec sent had a length of " +
                                   "{}, but it must have ".format(responses) +
                                   "a length of {}.".format(self.responses))
        else:
            raise RuntimeError("Argument respVec must be a NumPy array.")
        # create an empty matrix for inserting the tangent moduli
        dyde = np.zeros((self.responses, self.controls), dtype=float)
        return dyde

    def isRuptured(self):
        # the default case
        ruptured = False
        return ruptured

    def rupturedRespVec(self, ctrlVec, respVec):
        # verify inputs
        if isinstance(ctrlVec, np.ndarray):
            (controls,) = np.shape(ctrlVec)
            if controls != self.controls:
                raise RuntimeError("The ctrlVec sent had a length of " +
                                   "{}, but it must have ".format(controls) +
                                   "a length of {}.".format(self.controls))
        else:
            raise RuntimeError("Argument ctrlVec must be a NumPy array.")
        if isinstance(respVec, np.ndarray):
            (responses,) = np.shape(respVec)
            if responses != self.responses:
                raise RuntimeError("The respVec sent had a length of " +
                                   "{}, but it must have ".format(responses) +
                                   "a length of {}.".format(self.responses))
        else:
            raise RuntimeError("Argument respVec must be a NumPy array.")
        # create an empty vector for inserting the ruptured response variables
        yVec = np.zeros((self.responses,), dtype=float)
        return yVec

    def rupturedTanMod(self, ctrlVec, respVec):
        # verify inputs
        if isinstance(ctrlVec, np.ndarray):
            (controls,) = np.shape(ctrlVec)
            if controls != self.controls:
                raise RuntimeError("The ctrlVec sent had a length of " +
                                   "{}, but it must have ".format(controls) +
                                   "a length of {}.".format(self.controls))
        else:
            raise RuntimeError("Argument ctrlVec must be a NumPy array.")
        if isinstance(respVec, np.ndarray):
            (responses,) = np.shape(respVec)
            if responses != self.responses:
                raise RuntimeError("The respVec sent had a length of " +
                                   "{}, but it must have ".format(responses) +
                                   "a length of {}.".format(self.responses))
        else:
            raise RuntimeError("Argument respVec must be a NumPy array.")
        # create an empty matrix for inserting the ruptured tangent moduli
        dyde = np.zeros((self.responses, self.controls), dtype=float)
        return dyde


class pece(object):

    def __init__(self, ctrl, resp, m=1):
        # verify the inputs
        # assert that ctrl is an instance of type control
        if isinstance(ctrl, control):
            self.ctrl = ctrl
        else:
            raise RuntimeError("Argument ctrl must be an object that " +
                               "inherits class control.")
        self.t = 0.0
        self.tR = 0.0
        self.dt = ctrl.dt
        # assert that resp is an instance of type response
        if isinstance(resp, response):
            self.resp = resp
        else:
            raise RuntimeError("Argument resp must be an object that " +
                               "inherits class response.")
        if ctrl.controls != resp.controls:
            raise RuntimeError("The number of control variables is not the " +
                               "same for the supplied ctrl and resp objects.")
        # limit the range for m in the implementation of our PE(CE)^m
        if isinstance(m, int):
            if m < 1:
                self.m = 1
            elif m > 5:
                self.m = 5
            else:
                self.m = m
        else:
            raise RuntimeError("Argument m must be an integer within [1, 5].")
        # set the flags for handling rupture events
        self.isRuptured = False
        self.wasRuptured = False
        # set a flag for selecting Heun's method when starting or restarting
        self.step = 1
        # create the response arrays for the previous, current and next nodes
        self.yP = np.zeros((resp.responses,), dtype=float)
        self.yC = np.zeros((resp.responses,), dtype=float)
        self.yN = np.zeros((resp.responses,), dtype=float)
        self.dydtP = np.zeros((resp.responses,), dtype=float)
        self.dydtC = np.zeros((resp.responses,), dtype=float)
        self.dydtN = np.zeros((resp.responses,), dtype=float)
        # the following arrays enable an integration step to be re-integrated
        self.yPR = np.zeros((resp.responses,), dtype=float)
        self.yCR = np.zeros((resp.responses,), dtype=float)
        self.dydtPR = np.zeros((resp.responses,), dtype=float)
        self.dydtCR = np.zeros((resp.responses,), dtype=float)
        # set the local truncation error to its truncated minimal value
        self.error = 1.0e-10
        return  # a new integrator object

    def integrate(self, ctrlVec, restart=False):
        self.ctrl.update(ctrlVec)
        # obtain a tangent modulus for the reference state
        if self.step == 1:
            dyde0 = self.resp.tanMod(self.ctrl.xR, self.resp.yR)
            dedx0 = self.ctrl.dedx()
            dxdt0 = self.ctrl.dxdt()
            dydt0 = np.matmul(dyde0, np.matmul(dedx0, dxdt0))
            self.yCR[:] = self.resp.yR[:]
            self.dydtCR[:] = dydt0[:]
        # assign response variables and their rates (enables a re-integration)
        self.yP[:] = self.yPR[:]
        self.yC[:] = self.yCR[:]
        self.dydtP[:] = self.dydtPR[:]
        self.dydtC[:] = self.dydtCR[:]
        # perform one step of integration
        if self.step == 1 or restart is True:
            # start or restart the integrator using Heun's one-step method
            x1 = ctrlVec
            dedx1 = self.ctrl.dedx()
            dxdt1 = self.ctrl.dxdt(restart)
            dedt1 = np.matmul(dedx1, dxdt1)
            # predict
            y1 = np.add(self.yC, np.multiply(self.dt, self.dydtC))
            yP = np.copy(y1)
            # evaluate
            if self.isRuptured is False:
                dyde1 = self.resp.tanMod(x1, y1)
            else:
                dyde1 = self.resp.rupturedTanMod(x1, y1)
            dydt1 = np.matmul(dyde1, dedt1)
            # iterate over the CE pair m times
            for m in range(self.m):
                # correct
                y1 = np.add(self.yC, np.multiply(self.dt / 2.0,
                                                 np.add(dydt1, self.dydtC)))
                # re-evaluate
                if self.isRuptured is False:
                    dyde1 = self.resp.tanMod(x1, y1)
                else:
                    dyde1 = self.resp.rupturedTanMod(x1, y1)
                dydt1 = np.matmul(dyde1, dedt1)
            yC = np.copy(y1)
        else:
            # continue integration using Freed's two-step method
            xN = ctrlVec
            dedxN = self.ctrl.dedx()
            dxdtN = self.ctrl.dxdt()
            dedtN = np.matmul(dedxN, dxdtN)
            # predict
            yNm1 = np.multiply(1.0 / 3.0,
                               np.subtract(np.multiply(4.0, self.yC), self.yP))
            dydt = np.subtract(np.multiply(2.0, self.dydtC), self.dydtP)
            yN = np.add(yNm1, np.multiply(2.0 * self.dt / 3.0, dydt))
            yP = np.copy(yN)
            # evaluate
            if self.isRuptured is False:
                dydeN = self.resp.tanMod(xN, yN)
            else:
                dydeN = self.resp.rupturedTanMod(xN, yN)
            dydtN = np.matmul(dydeN, dedtN)
            # iterate over the CE pair m times
            for m in range(self.m):
                # correct: this is Gear's BDF2 formula
                yN = np.add(yNm1, np.multiply(2.0 * self.dt / 3.0, dydtN))
                # re-evaluate
                if self.isRuptured is False:
                    dydeN = self.resp.tanMod(xN, yN)
                else:
                    dydeN = self.resp.rupturedTanMod(xN, yN)
                dydtN = np.matmul(dydeN, dedtN)
            yC = np.copy(yN)
        # update the integrator's data base
        if self.step == 1 or restart is True:
            self.yN[:] = y1[:]
            self.dydtN[:] = dydt1[:]
        else:
            self.yN[:] = yN[:]
            self.dydtN[:] = dydtN[:]
        # compute the error
        magP = LA.norm(yP)
        magC = LA.norm(yC)
        self.error = abs(magC - magP) / max(1.0, magC)
        if self.error < 1.0e-10:
            self.error = 1.0e-10
        # test for and, if necessary, handle a rupture event
        self.isRuptured = self.resp.isRuptured()
        if self.isRuptured is True and self.wasRuptured is False:
            # 'break' continuity in the response and re-run the integrator
            self.wasRuptured = True
            if self.step == 1 or restart is True:
                yRuptured = self.resp.rupturedRespVec(x1, y1)
                dydeRuptured = self.resp.rupturedTanMod(x1, yRuptured)
                dydtRuptured = np.matmul(dydeRuptured, dedt1)
            else:
                yRuptured = self.resp.rupturedRespVec(xN, yN)
                dydeRuptured = self.resp.rupturedTanMod(xN, yRuptured)
                dydtRuptured = np.matmul(dydeRuptured, dedtN)
            self.yCR[:] = yRuptured[:]
            self.dydtCR[:] = dydtRuptured[:]
            restart = True
            self.integrate(ctrlVec, restart)
            restart = False
        return  # nothing

    def advance(self):
        self.ctrl.advance()
        # advance the integrator's data structures
        self.step += 1
        self.t += self.dt
        self.yPR[:] = self.yC[:]
        self.yCR[:] = self.yN[:]
        self.dydtPR[:] = self.dydtC[:]
        self.dydtCR[:] = self.dydtN[:]
        return  # nothing

    # The following procedures are to be called after advance has been called.

    # get the time
    def getT(self):
        # The data structure has already been advanced, hence the -dt.
        time = self.t - self.dt
        return time

    # get the control variables
    def getX(self):
        x = np.zeros((self.ctrl.controls,), dtype=float)
        x[:] = self.ctrl.xN[:]
        return x

    # get the response variables
    def getY(self):
        y = np.zeros((self.resp.responses,), dtype=float)
        y[:] = self.yN[:]
        return y

    # get the local truncation error
    def getError(self):
        return self.error


"""
Changes made in version "1.0.0":

This is the initial version of Freed's PECE integrator for solving equations
of the form  dy/dt = f(x, y) dx/dt  where y is a vector of response variables
and x is a vector of control variables that are explicit functions of time t.

This code is a rework of Freed's original Python code found in:  peceVtoX.py
"""
