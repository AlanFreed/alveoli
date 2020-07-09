#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA

"""
Module peceHE.py provides a PECE solver for Hypo-Elastic like constitutive
equations plus classes for the control and response functions required by them.

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
__author_email__ = "afreed@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


This module provides three classes:
    PECE        implements a PECE solver for first-order differential equations
    Control     provides the user with an interface for the control  variables
    Response    provides the user with an interface for the response variables


The PECE solver solves the following system of ODEs typical of hypo-elasticity
    dy/dt = dy/de de/dt
or, more specifically, for our application
    dy/dt = dy/de de/dx dx/dt  or  dy/dt = f(e, y) de/dx dx/dt
where
    f(e, y) = dy/de
        a matrix function establishing the constitutive response for a system
    de/dx
        a matrix function that converts physical into thermodynamic ctrl rates
    dx/dt
        a vector establishing the physical control rates
subject to an initial condition of
    y_0  at time  t_0
where
    x   is a vector of physical control variables, e.g., stretches
    e   is a vector of thermodynamic control variables, e.g., strains
    y   is a vector of thermodynamic response variables, e.g., stresses
    t   is a floating point number that denotes time with t_0 = 0.0
All vectors and matrices are instances of NumPy.ndarray with float elements.

Two Adams-Moulton (predictor/corrector) methods are implemented: a one-step
method for starting an integration, and a two-step method for continuing it.

In the descriptions for the integrator described below, denote
    e_p, x_p, y_p    associate with the previous node of integration
    e_c, x_c, y_c    associate with the current node of integration
    e_n, x_n, y_n    associate with the next node of integration

The integrator given below is not self starting, so a one-step method is used
to take the first step of integration, and whenever the integrator is to be
restarted because of a discontinuity in the control variables; specifically,
    Predict:  (forward Euler)
        yPred = y_0 + dy_0/dt dt
    Evaluate:
        dyPred/dt = f(e_1, yPred) de_1/dx dx_1/dt
    Correct:  (trapazoidal rule)
        y_1 = y_0 + (1/2)(dyPred/dt + dy_0/dt) dt
    re-Evaluate:
        dy_1/dt = f(e_1, y_1) de_1/dx dx_1/dt
where
    dy_0/dt = f(e_0, y_0) de_0/dx dx_0/dt
This is the Adams-Moulton method of Heun for integrating a first-order ODE.

A second-order accurate PECE method is used.  It is described by the equations:
    Predict:  (Freed's predictor)
        yPred = (1/3)(4 y_c - y_p) + (2/3)(2 dy_c/dt - dy_p/dt) dt
    Evaluate:
        dyPred/dt = f(e_n, yPred) de_n/dx dx_n/dt
    Correct:  (BDF2 method of Gear)
        y_n = (1/3)(4 y_c - y_p) + (2/3) dyPred/dt dt
    re-Evaluate:
        dy_n/dt = f(e_n, y_n) de_n/dx dx_n/dt
This is the Adams-Moulton method of Freed for integrating a first-order ODE.


object constructor

    E.g.:  solver = PECE(ctrl, resp, m=1)
        ctrl    an object extending class 'Control'  (see below):
        resp    an object extending class 'Response' (see below):
        m       the number of CE iterations, i.e., PE(CE)^m, m is in [1, 5]
    Creates an object that integrates hypo-elastic like constitutive equations
    using a PE(CE)^m method.

variables: treat these as read-only

    step        specifies the integration step that integrator is currently at

methods

integrate(xVec, restart=False)
    E.g.:  solver.integrate(xVec, restart)
        xVec    is a vector containing the control variables for the next node
        restart is to be set to True whenever there is a discontinuity in one
                of the control variables; Heun's one-step method is used then
    Integrates an ODE over its current interval, viz., from the current node
    to the next node, also using information from the previous node.  This
    procedure may be re-called multiple times before advancing a solution.

advance()
    E.g.:  solver.advance()
    Commits a solution, thereby preparing the integrator's data base for
    advancing to the next step.  solver.advance() is to be called after the
    global FE solver has converged, and before solver.integrate() is called
    once again to propogate a solution along its path.

The following methods return values determined at the end of the current
integration step.  They are to be called after solver.advance has been called,
and before the next call to solver.integrate is to be issued.

getT()
    E.g.:  t = solver.getT()
    Returns a float for time

getE()
    E.g.:  e = solver.getE()
    Returns a vector containing the thermodynamic control variables (strains)

getX()
    E.g.:  x = solver.getX()
    Returns a vector containing the physical control variables (stretches)

getY()
    E.g.:  y = solver.getY()
    Returns a vector containing the thermodynamic response variables (stresses)

getEminusE0()
    E.g.:  e = solver.getEminusE0()
    Returns a vector containing a difference in the thermodynamic control
    variables relative to their reference state (strains)

getXminusX0()
    E.g.:  x = solver.getXminusX0()
    Returns a vector containing a difference in the physical control variables
    relative to their reference state (stretches)

getYminusY0()
    E.g.:  y = solver.getYminusY0()
    Returns a vector containing a difference in the thermodynamic response
    variables relative to their reference state (stresses)

getError()
    E.g.:  error = solver.getError()
    Returns an estimate for the local truncation error of integration


A base class has been created for the control variables that is to be extended
in one's application.  Here the methods return values that associate with the
next node.  The interface for this class is:


object constructor

    E.g.: ctrl = Control(eVec0, xVec0, dt)
        eVec0       initial conditions for the thermodynamic control variables
        xVec0       initial conditions for the physical control variables
        dt          size of the time step to be used throughout for integration

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    node            an integer specifying the current node of integration,
                    which is reset to 0 whenever the integrator is restarted
    dt              a floating point number specifying the time-step size
    eR              a vector of initial conditions for thermodynamic controls
    eP              a vector of thermodynamic control vars at the previous node
    eC              a vector of thermodynamic control vars at the current node
    eN              a vector of thermodynamic control vars at the next node
    xR              a vector of initial conditions for the physical controls
    xP              a vector of physical control variables at the previous node
    xC              a vector of physical control variables at the current node
    xN              a vector of physical control variables at the next node
Fields eP, eC, eN are obtained by integrating de/dx dx/dt using the correctors
in the PECE method used in objects of class 'pece'.

methods

update(xVec, restart=False)
    E.g.:  ctrl.update(xVec, restart)
        xVec        a vector of physical control variables for the next node
        restart     whenever restart is True, the trapezoidal method is used;
                    otherwise, use Gear's BDF2 method for integration
    ctrl.update may be called multiple times before freezing its values with a
    call to ctrl.advance.

advance()
    E.g.:  ctrl.advance()
    Updates the object's data structure in preparation for the next integration
    step.  This method is called internally by the PECE object and must not be
    called by the user.

dedx()
    E.g.:  dedxMtx = ctrl.dedx()
        dedxMtx     a matrix containing the mapping of physical control rates
                    into thermodynamic control rates.
    This transformation associates with the next node.  It is created as an
    identity matrix in the base class that is to be overwritten.

dxdt()
    E.g.:  dxdtVec = ctrl.dxdt()
        dxdtVec     is a vector containing the rate-of-change in control
    This base method implements finite difference formulae to approximate this
    derivative. A first-order difference formula is used for the reference and
    first nodes, plus the first two nodes after a restart has been mandated.
    A second-order backward difference formula is used for all other nodes.
    All derivatives associate with the next node.  If these approximations are
    not appropriate for your application, then you will need to overwrite them.


Another base class has been created for the response variables that is also to
be extended in one's application.  Here all methods return fields associated
with the next node.  The interface for this class is:


object constructor

    E.g.:  resp = Response(yVec0)
        yVec0       initial conditions for the response variables

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    responses       an integer specifying the number of response variables
    yR              a vector containing the initial condition for response

A mixture of n constituents will have responses of n * controls.

methods

secantModulus(eVec, xVec, yVec)
    E.g.:  Es = ce.secantModulus(eVec, xVec, yVec)
        Es          a matrix of secant moduli, i.e., a constitutive equation
        eVec        a vector of thermodynamic control variables  (strains)
        xVec        a vector of physical control variables       (stretches)
        yVec        a vector of thermodynamic response variables (stresses)
    The secant modulus is not used by the PECE, but is needed by the FE solver.
    This constitutive expression is hyper-elastic.  It returns Es in: y = Es*e.

tangentModulus(eVec, xVec, yVec)
    E.g.:  Et = resp.tangentModulus(eVec, xVec, yVec)
        dyde        a matrix of tangent moduli, i.e., a constitutive equation
        eVec        a vector of thermodynamic control variables  (strains)
        xVec        a vector of physical control variables       (stretches)
        yVec        a vector of thermodynamic response variables (stresses)
    The constitutive equation considered here is hypo-elastic; specifically,
        dy/dt = dy/de de/dx dx/dt  where  de/dt = de/dx dx/dt comes from the
                                          the associated control object.
    wherein
        dy/dt       a vector of response rates
        dy/de       a matrix of constitutive tangent moduli, i.e., Et
    This function returns a matrix Et = dy/de.

isRuptured()
    E.g.:  (ruptured) = resp.isRuptured()
        (ruptured)  is tuple of boolean results specifying if rupture occurred
    There is a entry for each constituent in a material that is a mixture.
    This method is called by the PECE object.

rupturedResponse(eVec, xVec, yBeforeVec)
    E.g.:  yAfterVec = resp.rupturedResponse(eVec, xVec, yBeforeVec)
        eVec        vector of thermodynamic control variables at rupture
        xVec        vector of physical control variables at rupture
        yBeforeVec  vector of response variables just before rupture occurs
    returns
        yAfterVec   vector of response variables just after a rupture event
    Calling this method, which is done internally by the PECE integrator,
    allows for a discontinuity in the field of thermodynamic responses.


Templates for using the above control and response objects are provided below.


class MyCtrl(Control):

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    node            an integer specifying the current node of integration,
                    which is reset to 0 whenever the integrator is restarted
    dt              a floating point number specifying the time-step size
    eR              a vector of initial conditions for thermodynamic controls
    eP              a vector of thermodynamic control vars at the previous node
    eC              a vector of thermodynamic control vars at the current node
    eN              a vector of thermodynamic control vars at the next node
    xR              a vector of initial conditions for the physical  controls
    xP              a vector of physical control variables at the previous node
    xC              a vector of physical control variables at the current node
    xN              a vector of physical control variables at the next node

methods

    def __init__(self, eVec0, xVec0, dt):
        # Call the constructor of the base type to create and initialize the
        # exported variables.
        super().__init__(eVec0, xVec0, dt)
        # Create and initialize any additional fields introduced by the user.
        return  # a new instance of type MyCtrl

    def update(self, xVec, restart=False):
        # Call the base implementation of this method to insert this physical
        # control variable into the data structure of this object, and then to
        # integrate the thermodynamic control variables, eVec, for this update.
        super().update(xVec, restart)
        # Update any additional fields introduced by the user.
        return  # nothing

    def advance(self):
        # Call the base implementation of this method to advance its data
        # structure by copying the current data into their previous fields,
        # and then copying the next data into their current fields.
        super().advance()
        # Advance any additional data introduced by the user.
        # This method is called internally by the pece integrator and must not
        # be called by the user.
        return  # nothing

    def dedx(self):
        # Call the base implementation of this method to create matrix dedxMtx.
        dedxMtx = super().dedx()
        # Assign elements to this matrix per the user's application.
        return dedxMtx

    def dxdt(self):
        # Call the base implementation of this method to create vector dxdtVec.
        dxdtVec = super().dxdt()
        # The returned dxdtVec is computed via finite difference formulae;
        # specifically,
        #   if self.node is 0, 1   use first-order difference formula
        #   if restart is True     use first-order difference formula
        #   otherwise              use second-order backward difference formula
        # Overwrite dxdtVec if this is not appropriate for your application.
        return dxdtVec


class MyResp(Response):

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    responses       an integer specifying the number of response variables
    eR              a vector of thermodynamic controls in the reference state
    xR              a vector of physical controls in the reference state
    yR              a vector containing the initial condition for the response

methods

    def __init__(self, yVec0):
        # A call to the base constructor creates and initializes the exported
        # variables.
        super().__init__(yVec0)
        # Create and initialize any additional fields introduced by the user.
        return  # a new instance of type MyResp

    # secant modulus is not used by the PECE, but is needed by the FE solver
    def secantModulus(self, eVec, xVec, yVec):
        # call the base type to verify the inputs and to create the matrix E
        Es = super().secantModulus(eVec, xVec, yVec)
        # y = Es * e
        #    e   is a vector of thermodynamic control variables  (strains)
        #    x   is a vector of physical control variables       (stretches)
        #    y   is a vector of thermodynamic response variables (stresses)
        # populate the entries of E for the user's secant moduli below
        return Es

    def tangentModulus(self, eVec, xVec, yVec):
        # call the base type to verify the inputs and to create matrix dyde
        Et = super().tangentModulus(eVec, xVec, yVec)
        # dy = Et * de  where  Et = dy/de  and  de = de/dx dx/dt
        #    e   is a vector of thermodynamic control variables  (strains)
        #    x   is a vector of physical control variables       (stretches)
        #    y   is a vector of thermodynamic response variables (stresses)
        # populate the entries of dyde for the user's tangent moduli below
        return Et

    def isRuptured(self):
        # no super call is required here
        return self.ruptured

    def rupturedResponse(self, eVec, xVec yBeforeVec):
        # call the base type to verify the input and to create vector yAfterVec
        yAfterVec = super().rupturedResponse(eVec, xVec, yBeforeVec)
        # populate the entries for the ruptured response in yAfterVec below
        # this will result in a discontinuity in the response fields
        return yAfterVec

Reference:
    A. D. Freed, "A Technical Note: Two-Step PECE Methods for Approximating
    Solutions To First- and Second-Order ODEs", arXiv 1707.02125, 2017.
"""


class Control(object):

    def __init__(self, eVec0, xVec0, dt):
        if isinstance(eVec0, np.ndarray):
            (controls,) = np.shape(eVec0)
            self.controls = controls
        else:
            raise RuntimeError("Argument eVec0 must by a NumPy array.")
        if isinstance(xVec0, np.ndarray):
            (controls,) = np.shape(xVec0)
            if controls != self.controls:
                raise RuntimeError("Vectors eVec0 and xVec0 must have the " +
                                   "same dimension.")
        else:
            raise RuntimeError("Argument xVec0 must by a NumPy array.")
        if isinstance(dt, float) and dt > 100.0 * np.finfo(float).eps:
            self.dt = dt
        else:
            raise RuntimeError("Argument dt must be greater than 100 times " +
                               "machine epsilon, i.e., dt > 2.2E-14.")
        # create the object's remaining internal data structure
        # thermodynamic control variables
        self.eR = np.zeros((self.controls,), dtype=float)    # reference state
        self.eP = np.zeros((self.controls,), dtype=float)    # previous state
        self.eC = np.zeros((self.controls,), dtype=float)    # current state
        self.eN = np.zeros((self.controls,), dtype=float)    # next state
        # physical control variables
        self.xR = np.zeros((self.controls,), dtype=float)    # reference state
        self.xP = np.zeros((self.controls,), dtype=float)    # previous state
        self.xC = np.zeros((self.controls,), dtype=float)    # current state
        self.xN = np.zeros((self.controls,), dtype=float)    # next state
        # initialize these vector fields
        # thermodynamic control variables
        self.eR[:] = eVec0[:]
        self.eP[:] = eVec0[:]
        self.eC[:] = eVec0[:]
        self.eN[:] = eVec0[:]
        # physical control variables
        self.xR[:] = xVec0[:]
        self.xP[:] = xVec0[:]
        self.xC[:] = xVec0[:]
        self.xN[:] = xVec0[:]
        # create the additional fields needed for numeric integration
        self.node = 0
        self.dedtC = np.matmul(self.dedx(), self.dxdt())
        self.dedtN = np.zeros((self.controls,), dtype=float)
        return  # a new instance of a control object

    def update(self, xVec, restart=False):
        # verify the input
        if isinstance(xVec, np.ndarray):
            (controls,) = np.shape(xVec)
            if self.controls != controls:
                raise RuntimeError("Argument xVec must have a length of " +
                                   "{}, but it had ".format(self.controls) +
                                   "a length of {}.".format(controls))
        else:
            raise RuntimeError("Argument xVec must by a NumPy array.")
        if restart:
            self.node = 0
        # update the physical control variables
        self.xN[:] = xVec[:]
        # integrate to update the thermodynamic control variables
        dedt = np.matmul(self.dedx(), self.dxdt())
        self.dedtN[:] = dedt[:]
        if self.node < 2:
            # start or restart the integrator using the trapezoidal rule
            self.eN[:] = self.eC[:] + ((self.dt / 2.0) *
                                       (self.dedtN[:] + self.dedtC[:]))
        else:
            # continue integration with Gear's BDF2 formula
            self.eN[:] = ((1.0 / 3.0) * (4.0 * self.eC[:] - self.eP[:]) +
                          ((2.0 * self.dt) / 3.0) * self.dedtN[:])
        return  # nothing

    def advance(self):
        self.eP[:] = self.eC[:]
        self.eC[:] = self.eN[:]
        self.xP[:] = self.xC[:]
        self.xC[:] = self.xN[:]
        self.dedtC[:] = self.dedtN[:]
        self.node += 1
        return  # nothing

    def dedx(self):
        de = np.eye(self.controls, dtype=float)
        return de

    def dxdt(self):
        dx = np.zeros((self.controls,), dtype=float)
        if self.node < 2:
            # use the first-order forward/backward difference formula
            dx[:] = (self.xN[:] - self.xC[:]) / self.dt
        else:
            # use the second-order backward difference formula
            dx[:] = ((3.0 * self.xN[:] - 4.0 * self.xC[:] + self.xP[:]) /
                     (2.0 * self.dt))
        return dx


class Response(object):

    def __init__(self, yVec0):
        # verify input
        if isinstance(yVec0, np.ndarray):
            (responses,) = np.shape(yVec0)
            self.responses = responses
            self.yR = np.zeros((self.responses,), dtype=float)
            self.yR[:] = yVec0[:]
        else:
            raise RuntimeError("Initial condition yVec0 must by a NumPy " +
                               "array.")
        self.firstCall = True
        return  # a new instance of this base type

    def secantModulus(self, eVec, xVec, yVec):
        # verify inputs
        if isinstance(eVec, np.ndarray):
            (controls,) = np.shape(eVec)
            if self.firstCall:
                self.controls = controls
                if self.responses % self.controls != 0:
                    raise RuntimeError("The number of response variables " +
                                       "must be an integer mulitplier to " +
                                       "the number of control variables.")
                self.eR = np.zeros((self.controls,), dtype=float)
                self.eR[:] = eVec[:]
            else:
                if controls != self.controls:
                    raise RuntimeError("The eVec sent had a length of " +
                                       "{}, but it must ".format(controls) +
                                       "have length {}.".format(self.controls))
        else:
            raise RuntimeError("Argument eVec must be a NumPy array.")
        if isinstance(xVec, np.ndarray):
            (controls,) = np.shape(xVec)
            if self.firstCall:
                self.xR = np.zeros((self.controls,), dtype=float)
                self.xR[:] = xVec[:]
            if controls != self.controls:
                raise RuntimeError("The xVec sent had a length of " +
                                   "{}, but it must have ".format(controls) +
                                   "a length of {}.".format(self.controls))
        else:
            raise RuntimeError("Argument xVec must be a NumPy array.")
        if isinstance(yVec, np.ndarray):
            (responses,) = np.shape(yVec)
            if responses != self.responses:
                raise RuntimeError("The yVec sent had a length of " +
                                   "{}, but it must have ".format(responses) +
                                   "a length of {}.".format(self.responses))
        else:
            raise RuntimeError("Argument yVec must be a NumPy array.")
        # create an empty matrix for inserting the secant moduli into
        Es = np.zeros((self.responses, self.controls), dtype=float)
        # update the first call flag
        if self.firstCall:
            self.firstCall = False
        return Es

    def tangentModulus(self, eVec, xVec, yVec):
        if isinstance(eVec, np.ndarray):
            (controls,) = np.shape(eVec)
            if self.firstCall:
                self.controls = controls
                if self.responses % self.controls != 0:
                    raise RuntimeError("The number of response variables " +
                                       "must be an integer mulitplier to " +
                                       "the number of control variables.")
                self.eR = np.zeros((self.controls,), dtype=float)
                self.eR[:] = eVec[:]
            else:
                if controls != self.controls:
                    raise RuntimeError("The eVec sent had a length of " +
                                       "{}, but it must ".format(controls) +
                                       "have length {}.".format(self.controls))
        else:
            raise RuntimeError("Argument eVec must be a NumPy array.")
        if isinstance(xVec, np.ndarray):
            (controls,) = np.shape(xVec)
            if self.firstCall:
                self.xR = np.zeros((self.controls,), dtype=float)
                self.xR[:] = xVec[:]
            if controls != self.controls:
                raise RuntimeError("The xVec sent had a length of " +
                                   "{}, but it must have ".format(controls) +
                                   "a length of {}.".format(self.controls))
        else:
            raise RuntimeError("Argument xVec must be a NumPy array.")
        if isinstance(yVec, np.ndarray):
            (responses,) = np.shape(yVec)
            if responses != self.responses:
                raise RuntimeError("The yVec sent had a length of " +
                                   "{}, but it must have ".format(responses) +
                                   "a length of {}.".format(self.responses))
        else:
            raise RuntimeError("Argument yVec must be a NumPy array.")
        # create an empty matrix for inserting the tangent moduli into
        Et = np.zeros((self.responses, self.controls), dtype=float)
        # update the first call flag
        if self.firstCall:
            self.firstCall = False
        return Et

    def isRuptured(self):
        constituents = self.responses // self.controls
        ruptured = (False,)
        for i in range(1, constituents):
            ruptured = ruptured + (False,)
        return ruptured

    def rupturedResponse(self, eVec, xVec, yBeforeVec):
        # verify inputs
        if isinstance(eVec, np.ndarray):
            (controls,) = np.shape(eVec)
            if controls != self.controls:
                raise RuntimeError("The eVec sent had a length of " +
                                   "{}, but it must have ".format(controls) +
                                   "a length of {}.".format(self.controls))
        else:
            raise RuntimeError("Argument eVec must be a NumPy array.")
        if isinstance(xVec, np.ndarray):
            (controls,) = np.shape(xVec)
            if controls != self.controls:
                raise RuntimeError("The xVec sent had a length of " +
                                   "{}, but it must have ".format(controls) +
                                   "a length of {}.".format(self.controls))
        else:
            raise RuntimeError("Argument xVec must be a NumPy array.")
        if isinstance(yBeforeVec, np.ndarray):
            (responses,) = np.shape(yBeforeVec)
            if responses != self.responses:
                raise RuntimeError("The yBeforeVec sent had a length of " +
                                   "{}, but it must have ".format(responses) +
                                   "a length of {}.".format(self.responses))
        else:
            raise RuntimeError("Argument yBeforeVec must be a NumPy array.")
        # create an empty vector for inserting the ruptured response variables
        yAfterVec = np.zeros((self.responses,), dtype=float)
        return yAfterVec


class PECE(object):

    def __init__(self, ctrl, resp, m=1):
        # verify the inputs
        # assert that ctrl is an object that extends class control
        if isinstance(ctrl, Control):
            self.ctrl = ctrl
        else:
            raise RuntimeError("Argument ctrl must be an object that " +
                               "inherits class Control.")
        self.t = 0.0
        self.tR = 0.0
        self.dt = ctrl.dt
        # assert that resp is an object that extends class response
        if isinstance(resp, Response):
            self.resp = resp
        else:
            raise RuntimeError("Argument resp must be an object that " +
                               "inherits class Response.")
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
        self.isRuptured = (False,)
        self.wasRuptured = (False,)
        constituents = self.resp.responses // self.ctrl.controls
        for i in range(1, constituents):
            self.isRuptured = self.isRuptured + (False,)
            self.wasRuptured = self.wasRuptured + (False,)
        # set a flag for selecting Heun's method when starting or restarting
        self.step = 1
        # create the response arrays for the previous, current and next nodes
        self.yP = np.zeros((self.resp.responses,), dtype=float)
        self.yC = np.zeros((self.resp.responses,), dtype=float)
        self.yN = np.zeros((self.resp.responses,), dtype=float)
        self.dydtP = np.zeros((self.resp.responses,), dtype=float)
        self.dydtC = np.zeros((self.resp.responses,), dtype=float)
        self.dydtN = np.zeros((self.resp.responses,), dtype=float)
        # the following arrays enable an integration step to be re-integrated
        self.yPR = np.zeros((self.resp.responses,), dtype=float)
        self.yCR = np.zeros((self.resp.responses,), dtype=float)
        self.dydtPR = np.zeros((self.resp.responses,), dtype=float)
        self.dydtCR = np.zeros((self.resp.responses,), dtype=float)
        # set the local truncation error to its truncated minimal value
        self.error = 1.0e-10
        return  # a new integrator object

    def integrate(self, xVec, restart=False):
        self.ctrl.update(xVec, restart)
        # obtain a tangent modulus for the reference state
        if self.step == 1:
            dyde0 = self.resp.tangentModulus(self.ctrl.eR, self.ctrl.xR,
                                             self.resp.yR)
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
        if self.step == 1 or restart:
            # start or restart the integrator using Heun's one-step method
            dedx1 = self.ctrl.dedx()
            dxdt1 = self.ctrl.dxdt()
            dedt1 = np.matmul(dedx1, dxdt1)
            # predict: forward Euler
            y1 = np.add(self.yC, np.multiply(self.dt, self.dydtC))
            yP = np.copy(y1)
            # evaluate
            dyde1 = self.resp.tangentModulus(self.ctrl.eN, self.ctrl.xN, y1)
            dydt1 = np.matmul(dyde1, dedt1)
            # iterate over the CE pair m times
            for m in range(self.m):
                # correct: trapezoidal rule
                y1 = np.add(self.yC, np.multiply(self.dt / 2.0,
                                                 np.add(dydt1, self.dydtC)))
                # re-evaluate
                dyde1 = self.resp.tangentModulus(self.ctrl.eN,
                                                 self.ctrl.xN, y1)
                dydt1 = np.matmul(dyde1, dedt1)
            yC = np.copy(y1)
        else:
            # continue integration using Freed's two-step method
            dedxN = self.ctrl.dedx()
            dxdtN = self.ctrl.dxdt()
            dedtN = np.matmul(dedxN, dxdtN)
            # predict: Freed's explicit two-step method
            yNm1 = np.multiply(1.0 / 3.0,
                               np.subtract(np.multiply(4.0, self.yC), self.yP))
            dydt = np.subtract(np.multiply(2.0, self.dydtC), self.dydtP)
            yN = np.add(yNm1, np.multiply(2.0 * self.dt / 3.0, dydt))
            yP = np.copy(yN)
            # evaluate
            dydeN = self.resp.tangentModulus(self.ctrl.eN, self.ctrl.xN, yN)
            dydtN = np.matmul(dydeN, dedtN)
            # iterate over the CE pair m times
            for m in range(self.m):
                # correct: this is Gear's BDF2 formula
                yN = np.add(yNm1, np.multiply(2.0 * self.dt / 3.0, dydtN))
                # re-evaluate
                dydeN = self.resp.tangentModulus(self.ctrl.eN,
                                                 self.ctrl.xN, yN)
                dydtN = np.matmul(dydeN, dedtN)
            yC = np.copy(yN)
        # update the integrator's data base
        if self.step == 1 or restart:
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
        # test for and handle, if necessary, a rupture event
        self.isRuptured = self.resp.isRuptured()
        if self.isRuptured != self.wasRuptured:
            # 'break' continuity in the response and re-run the integrator
            self.wasRuptured = self.isRuptured
            if self.step == 1 or restart:
                yRuptured = self.resp.rupturedResponse(self.ctrl.eN,
                                                       self.ctrl.xN, y1)
                dydeRuptured = self.resp.tangentModulus(self.ctrl.eN,
                                                        self.ctrl.xN,
                                                        yRuptured)
                dydtRuptured = np.matmul(dydeRuptured, dedt1)
            else:
                yRuptured = self.resp.rupturedResponse(self.ctrl.eN,
                                                       self.ctrl.xN, yN)
                dydeRuptured = self.resp.tangentModulus(self.ctrl.eN,
                                                        self.ctrl.xN,
                                                        yRuptured)
                dydtRuptured = np.matmul(dydeRuptured, dedtN)
            self.yCR[:] = yRuptured[:]
            self.dydtCR[:] = dydtRuptured[:]
            # reintegrate the step using post-rupture material properties
            restart = True
            self.integrate(xVec, restart)
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

    # get the absolute thermodynamic control variables (strains)
    def getE(self):
        e = np.zeros((self.ctrl.controls,), dtype=float)
        if self.step > 1:
            e[:] = self.ctrl.eN[:]
        else:
            e[:] = self.ctrl.eR[:]
        return e

    # get the absolute physical control variables (stretches)
    def getX(self):
        x = np.zeros((self.ctrl.controls,), dtype=float)
        if self.step > 1:
            x[:] = self.ctrl.xN[:]
        else:
            x[:] = self.ctrl.xR[:]
        return x

    # get the absolute thermodynamic response variables (stresses)
    def getY(self):
        y = np.zeros((self.resp.responses,), dtype=float)
        if self.step > 1:
            y[:] = self.yN[:]
        else:
            y[:] = self.resp.yR[:]
        return y

    # get the relativee thermodynamic control variables (strains)
    def getEminusE0(self):
        e = np.zeros((self.ctrl.controls,), dtype=float)
        if self.step > 1:
            e[:] = self.ctrl.eN[:] - self.ctrl.eR[:]
        return e

    # get the relative physical control variables (stretches)
    def getXminusX0(self):
        x = np.zeros((self.ctrl.controls,), dtype=float)
        if self.step > 1:
            x[:] = self.ctrl.xN[:] - self.ctrl.xR[:]
        return x

    # get the relative thermodynamic response variables (stresses)
    def getYminusY0(self):
        y = np.zeros((self.resp.responses,), dtype=float)
        if self.step > 1:
            y[:] = self.yN[:] - self.resp.yR[:]
        return y

    # get the local truncation error of numeric integration
    def getError(self):
        return self.error


"""
Changes made in version "1.0.0":

This is the initial version of Freed's PECE integrator for solving equations
of the form  dy/dt = f(e, y) de/dx dx/dt  where y is a vector of response
variables, viz., stresses, e is a vector of strains, and x is a vector of
stretches, which are explicit functions of time t.

This code is a rework of Freed's original Python code found in:  peceVtoX.py
as developed in the cited reference above.
"""
