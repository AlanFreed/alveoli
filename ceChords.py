#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import materialProperties as mp
import math as m
import numpy as np
from peceHE import Control, Response

"""
Module ceChords.py provides a constitutive description for alveolar chords.

Copyright (c) 2019-2020 Alan D. Freed

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
__date__ = "09-24-2019"
__update__ = "07-17-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

r"""
A listing of changes made wrt version release can be found at the end of file.


This module describes septal chords.  From histological studies, septal chords
are found to be comprised of both collagen and elastin fibers that align in
parallel with one another, with minimal chemical bonding between them.
Consequently, they are modeled here as two elastic rods exposed to the same
temperature and strain but carrying different states of entropy and stress.
Their geometric properties, and some of their constitutive parameters, too, are
described by probability distributions exported from materialProperties.py.
Given these properties, one can create objects describing realistic septal
chords, and thereby, realistic alveoli.  Septal chords form a network of
fibers which circumscribe the septa that collectively envelop an alveolar sac.

This module exports three classes:
    ControlFiber  provides a user interface to control fiber deformation
    BioFiber:     provides the response of a Freed-Rajagopal biologic fiber
    SeptalChord:  provides a user interface for the response of septal chords
These objects are to used by the PECE solver of peceHE.py to update the
material response of septal chords.

The CGS system of physical units adopted:
    length          centimeters (cm)
    mass            grams       (g)
    time            seconds     (s)
    temperature     centigrade  (C)
where
    force           dynes       [g.cm/s^2]      1 Newton = 10^5 dyne
    pressure        barye       [dyne/cm^2]     1 Pascal = 10 barye
    energy          erg         [dyne.cm]       1 Joule  = 10^7 ergs

All fields in this module are evaluated at the end of the current interval of
integration, i.e., at the 'next' node of integration.


class ControlFiber:  It implements and extends class 'Control'


For 1D fibers, the physical control vectors have constituents:
    xVec[0]  contains fiber temperature T (in centigrade)
    xVec[1]  contains fiber length L (in cm)
while the thermodynamic control vectors have constituents:
    eVec[0]  contains ln(T/T_0) where T_0 is a reference temperature
    eVec[1]  contains ln(L/L_0) where L_0 is a reference length
which produce a thermodynamic response vector with constituents:
    yVec[0]  contains the entropy density
    yVec[1]  contains the true stress

constructor

    E.g.:  ctrl = ControlFiber(eVec0, xVec0, dt)
        eVec0       initial conditions for the thermodynamic control variables
        xVec0       initial conditions for the physical control variables
        dt          size of the time step to be used throughout for integration

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    node            an integer specifying the current knot of integration,
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

inherited methods

update(xVec, restart=False)
    E.g.:  ctrl.update(xVec, restart)
        xVec        a vector of physical control variables for the next node
        restart     whenever restart is True, the trapezoidal method is used;
                    otherwise, Gear's BDF2 method is used for integration
    ctrl.update may be called multiple times before freezing its values with a
    call to ctrl.advance.  This becomes important in a finite element setting.

advance()
    E.g.:  ctrl.advance()
    Updates an object's data structure in preparation for the next integration
    step.  It moves current data into their previous fields, and then it moves
    next data into their current fields.  This method is called internally by
    the PECE object and should not be called by the user.

dedx()
    E.g.:  dedxMtx = ctrl.dedx()
        dedxMtx     a matrix containing the mapping of physical control rates
                    into their thermodynamic control rates.
    This transformation associates with the next node.  It is created as an
    identity matrix in the base class whose components are overwritten here.

dxdt()
    E.g.:  dxdtVec = ctrl.dxdt()
        dxdtVec     is a vector containing the time rate-of-change in control
    This base method implements finite difference formulae to approximate this
    derivative. A first-order difference formula is used for the reference and
    first nodes, plus the first two nodes after a restart has been mandated.
    A second-order backward difference formula is used for all other nodes.
    All derivatives associate with the next node.


class BioFiber:  It implements and extends class 'Response'.


Creates objects that implement the Freed-Rajagopal biologic fiber model.
For this model
    eVec[0]  contains fiber temperature strain:  ln(T/T_0)
    eVec[1]  contains fiber mechanical  strain:  ln(L/L_0)
and
    xVec[0]  contains fiber temperature          (in centigrade)
    xVec[1]  contains fiber length               (in cm)
while
    yVec[0]  contains fiber entropy density      (in erg/g.K)
    yVec[1]  contains fiber stress               (in dyne/cm^2 or barye)

constructor

    E.g.:  ce = BioFiber(yVec0, rho, Cp, alpha, E1, E2, e_t, e_f=float("inf"))
        yVec0       initial conditions for the response variables
        rho         density of mass for the fiber
        Cp          density of specific heat at constant pressure for the fiber
        alpha       lineal thermal strain coefficient for the fiber
        E1          compliant modulus, i.e., at zero stress and zero strain
        E2          stiff modulus, i.e., elastic modulus at terminal strain
        e_t         transition strain between compliant and stiff behaviors
        e_f         rupture strain, where s_f = E2 * e_f is the rupture stress.
                    If the default is accepted then it is assigned a rupture
                    strength of 1/100th of its theoretical strength.

variables: treat these as read-only

    responses       an integer specifying the number of response variables
    yR              a vector containing the initial condition for response

inherited methods

secantModulus(eVec, xVec, yVec)
    E.g.:  E = ce.secantModulus(eVec, xVec, yVec)
        Es          a matrix of secant moduli, i.e., the constitutive matrix
        eVec        a vector of thermodynamic control variables  (strains)
        xVec        a vector of physical control variables       (stretches)
        yVec        a vector of thermodynamic response variables (stresses)
    Solves:
    / eta - eta_0 \ - /   C        alpha Es / rho theta \ / ln(theta/theta_0) \
    \   s - s_0   / - \ -alpha Es           Es          / \     ln(L/L_0)     /
    This constitutive expression is hyper-elastic:  s = s_0 + Es * ln(L/L_0).

tangentModulus(eVec, xVec, yVec)
    E.g.:  Et = ce.tangentModulus(eVec, xVec, yVec)
        dyde        a matrix of tangent moduli, i.e., a constitutive equation
        eVec        a vector of thermodynamic control variables  (strains)
        xVec        a vector of physical control variables       (stretches)
        yVec        a vector of thermodynamic response variables (stresses)
    The constitutive equation considered here is hypo-elastic; specifically,
        dy/dt = dy/de de/dt  where  Et = dy/de  and  de/dt = de/dx dx/dt
    wherein
        dy/dt       a vector of thermodynamic response rates
        dy/de       a matrix of tangent moduli (the constitutive equation)
        de/dt       is supplied by objects from class controlFiber
    Solves:
    / dEta \ - /   C        alpha Et / rho theta \ / dTheta / Theta \
    \  ds  / - \ -alpha Et           Et          / \     dL / L     /
    Modulus E differs between the secant and tangent moduli implementations.

isRuptured()
    E.g.:  ruptured = ce.isRuptured()
        ruptured    is a boolean result specifying if a fiber has ruptured

rupturedResponse(eVec, xVec, yBeforeVec)
    E.g., yAfterVec = ce.rupturedResponse(eVec, xVec, yBeforeVec)
        eVec        vector of thermodynamic control variables at rupture
        xVec        vector of physical control variables at rupture
        yBeforeVec  vector of response variables just before rupture occurs
    returns
        yAfterVec   vector of response variables just after a rupture event
    Calling this method, which is done internally by the 'pece' integrator,
    allows for a discontinuity in the field of thermodynamic responses.


class SeptalChord, which also implements and extends class 'Response'


Creates objects that establish a thermoelastic constitutive equation for
septal chords comprised of collagen and elastin fibers laid up in parallel.
For this model
    eVec[0]  contains fiber temperature strain:       ln(T/T_0)
    eVec[1]  contains fiber mechanical  strain:       ln(L/L_0)
while
    xVec[0]  contains fiber temperature               (in centigrade)
    xVec[1]  contains fiber length                    (in cm)
and
    yVec[0]  contains collagen fiber entropy density  (in erg/g.K)
    yVec[1]  contains collagen fiber stress           (in dyne/cm^2 or barye)
    yVec[2]  contains elastin  fiber entropy density  (in erg/g.K)
    yVec[3]  contains elastin  fiber stress           (in dyne/cm^2 or barye)

constructor

    E.g.:  ce = SeptalChord(diaCollagen=None, diaElastin=None)
        diaCollagen     the reference diameter of the collagen fiber
        diaElastin      the reference diameter of the elastin fiber
    If the diameters take on their default values of None, then they are
    assigned via their respective probability distributions.

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    responses       an integer specifying the number of response variables
    eR              a vector containing the initial thermodynamic controls
    xR              a vector containing the initial physical controls
    yR              a vector containing the initial conditions for responses

inherited methods

secantModulus(eVec, xVec, yVec)
    E.g.:  Es = ce.secantModulus(eVec, xVec, yVec)
        Es          a matrix of secant moduli, i.e., the constitutive matrix
        eVec        a vector of thermodynamic control variables  (strains)
        xVec        a vector of physical control variables       (stretches)
        yVec        a vector of thermodynamic response variables (stresses)
    Solves:
    / eta - eta_0 \ - /   C        alpha Es / rho theta \ / ln(theta/theta_0) \
    \   s - s_0   / - \ -alpha Es           Es          / \     ln(L/L_0)     /
    This constitutive expression is hyper-elastic:  s = s_0 + Es * ln(L/L0).

tangentModulus(eVec, xVec, yVec)
    E.g.:  Et = ce.tangentModulus(eVec, xVec, yVec)
        Et = dy/de  a matrix of tangent moduli, i.e., a constitutive equation
        eVec        a vector of thermodynamic control variables  (strains)
        xVec        a vector of physical control variables       (stretches)
        yVec        a vector of thermodynamic response variables (stresses)
    The constitutive equation considered here is hypo-elastic; specifically,
        dy/dt = dy/de de/dt  where  Et = dy/de  and  de/dt = de/dx dx/dt
    wherein
        dy/dt       a vector of thermodynamic response rates
        dy/de       a matrix of tangent moduli (the constitutive equation)
    Solves:
    / dEta \ - /   C        alpha Et / rho theta \ / dT / T \
    \  dS  / - \ -alpha Et           Et          / \ dL / L /
    Modulus E differs between the secant and tangent moduli implementations.

isRuptured()
    E.g.:  ruptured = ce.isRuptured()
        ruptured    is a boolean result specifying if the material has ruptured

rupturedResponse(eVec, xVec, yBeforeVec)
    E.g., yAfterVec = ce.rupturedResponse(eVec, xVec, yBeforeVec)
        eVec        vector of thermodynamic control variables at rupture
        xVec        vector of physical control variables at rupture
        yBeforeVec  vector of response variables just before rupture occurs
    returns
        yAfterVec   vector of response variables just after a rupture event
    Calling this method, which is done internally by the 'PECE' integrator,
    allows for a discontinuity in the field of thermodynamic responses.

additional methods

bioFiberCollagen()
    E.g.:  fiberC = ce.bioFiberCollagen()
        fiberC     an instance of BioFiber representing a collagen fiber

bioFiberElastin()
    E.g.:  fiberE = ce.bioFiberElastin()
        fiberE     an instance of BioFiber representing an elastin fiber

massDensity()
    E.g.:  rho = ce.massDensity()
        rho         the mass density of the septal chord

length()
    E.g.:  len_ = ce.length()
        len_        length of the septal chord at the next node of integration

areaCollagen()
    E.g.:  a_c = ce.areaCollagen()
        a_c         the current cross-sectional area of the collagen fiber

areaElastin()
    E.g.:  a_e = ce.areaElastin()
        a_e         the current cross-sectional area of the elastin fiber

area()
    E.g.:  a = ce.area()
        a           the collective areas of both fibers in the septal chord

volumeCollagen()
    E.g.:  vol = ce.volumeCollagen()
        vol         the volume of the collagen fiber in the chord

volumeElastin()
    E.g.:  vol = ce.volumeElastin()
        vol         the volume of the elastin fiber in the chord

volume()
    E.g.:  vol = ce.volume()
        vol         the collective volume of both fibers in the septal chord

# absolute measures

temperature()
    E.g.:  temp = ce.temperature()
        temp        the temperature in degrees Centigrade

entropy()
    E.g.:  s = ce.entropy()
        s           the total entropy of the septal chord (not entropy density)

strain()
    E.g.:  e = ce.strain()
        e           the natural or logarithmic strain of the chord

stress()
    E.g.:  s = ce.stress()
        s           the nominal stress carried by the septal chord

force()
    E.g.:  f = ce.force()
        f           the total force carried by the septal chord

# relative measures, i.e., current minus initial

relativeTemperature()
    E.g.:  temp = ce.relativeTemperature()
        temp        the relative temperature of the septal chord

relativeEntropy()
    E.g.:  s = ce.relativeEntropy()
        s           the relative entropy of the septal chord

relativeStrain()
    E.g.:  e = ce.relativeStrain()
        e           the relative natural or logarithmic strain of the chord

relativeStress()
    E.g.:  s = ce.relativeStress()
        s           the relative nominal stress carried by the septal chord

relativeForce()
    E.g.:  f = ce.relativeForce()
        f           the relative total force carried by the septal chord

Reference:
    Freed, A. D. and Rajagopal, K. R., “A Promising Approach for Modeling
    Biological Fibers,” ACTA Mechanica, 227 (2016), 1609-1619.
    DOI: 10.1007/s00707-016-1583-8.  Errata: DOI: 10.1007/s00707-018-2183-6
"""


class ControlFiber(Control):

    # variables inherited from the base type: treat these as read-only:
    #   controls    an integer specifying the number of control variables
    #   node        an integer specifying the current integration knot,
    #               which is reset to 0 whenever the integrator is restarted
    #   dt          a floating point number specifying the time-step size
    #   eR          a vector of initial conditions for thermodynamic controls
    #   eP          a vector of thermodynamic control vars at the previous node
    #   eC          a vector of thermodynamic control vars at the current node
    #   eN          a vector of thermodynamic control vars at the next node
    #   xR          a vector holding control variables for the reference node
    #   xP          a vector holding control variables for the previous node
    #   xC          a vector holding control variables for the current node
    #   xN          a vector holding control variables for the next node
    # control vector arguments have interpretations of:
    #   eVec[0]     contains the fiber temperature strain:  ln(T/T_0)
    #   eVec[1]     contains the fiber mechanical  strain:  ln(L/L_0)
    # while
    #   xVec[0]     contains the fiber temperature (in centigrade)
    #   xVec[1]     contains the fiber length      (in cm)

    # constructor

    def __init__(self, eVec0, xVec0, dt):
        # Call the constructor of the base type to create and initialize the
        # exported variables.
        super().__init__(eVec0, xVec0, dt)
        # Create and initialize any additional fields introduced by the user.
        if self.controls != 2:
            raise RuntimeError("There are only 2 control variables for 1D "
                               + "fibers: temperature and length.")
        return  # a new instance of type controlFiber

    # inherited methods

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
        # This method is called internally by the PECE integrator and must not
        # be called by the user.
        return  # nothing

    def dedx(self):
        # Call the base implementation of this method to create matrix dedxMtx.
        dedxMtx = super().dedx()
        # Assign elements to this matrix per the user's application.
        dedxMtx[0, 0] = 1.0 / (273.0 + self.xN[0])   # 1/temperature (in K)
        dedxMtx[1, 1] = 1.0 / self.xN[1]             # 1/length
        return dedxMtx

    def dxdt(self):
        # Call the base implementation of this method to create vector dxdtVec.
        dxdtVec = super().dxdt()
        # The returned dxdtVec is computed via finite difference formulae;
        # specifically,
        #   if self.node is 0, 1   use first-order difference formula
        #   if restart is True     use first-order difference formula
        #   otherwise              use second-order backward difference formula
        return dxdtVec


# constitutive class for biologic fibers


class BioFiber(Response):
    # Implements the Freed-Rajagopal model for biologic fibers where
    # for this model
    #     eVec[0]  contains fiber temperature strain:  ln(T/T_0)
    #     eVec[1]  contains fiber mechanical  strain:  ln(L/L_0)
    # while
    #     xVec[0]  contains fiber temperature          (in centigrade)
    #     xVec[1]  contains fiber length               (in cm)
    # and
    #     yVec[0]  contains fiber entropy density      (in erg/g.K)
    #     yVec[1]  contains fiber stress               (in barye)

    # constructor

    def __init__(self, yVec0, rho, Cp, alpha, E1, E2, e_t, e_f=float("inf")):
        # A call to the base constructor creates and initializes the exported
        # variables.
        super().__init__(yVec0)
        # Create and initialize any additional fields introduced by the user.
        if self.responses != 2:
            raise RuntimeError("A biologic fiber has two response variables.")
        # verify and initialize the remaining data
        if rho > np.finfo(float).eps:
            self.rho = rho
        else:
            raise RuntimeError("Fiber mass density must be positive.")
        if Cp > np.finfo(float).eps:
            self.Cp = Cp
        else:
            raise RuntimeError("Fiber specific heat must be positive.")
        if alpha > np.finfo(float).eps:
            self.alpha = alpha
        else:
            raise RuntimeError("Fiber thermal strain coefficient "
                               + "must be positive.")
        if E1 > np.finfo(float).eps:
            self.E1 = E1
        else:
            raise RuntimeError('Initial fiber modulus E1 must be positive.')
        if E2 > self.E1:
            self.E2 = E2
        else:
            raise RuntimeError('Terminal fiber modulus E2 must be greater '
                               + 'than the initial fiber modulus E1.')
        if e_t > np.finfo(float).eps:
            self.e_t = e_t
        else:
            raise RuntimeError('Limiting fiber reconfiguration strain e_t '
                               + 'must be positive.')
        # one one-hundredth of the theoretical upper bound on fiber strength
        body_temp = 310.0  # Kelvin
        e_f_max = 0.01 * rho * Cp * body_temp / (alpha**2 * E2)
        # establish the strain at fracture
        if e_f > e_f_max:
            self.e_f = e_f_max
        elif e_f > np.finfo(float).eps:
            self.e_f = e_f
        else:
            raise RuntimeError('Fiber failure strain e_f must be positive.')
        self.ruptured = False
        return  # a new instance of a fiber object

    # local methods

    def _secCompliance(self, stress):
        if stress > self.e_f * self.E2:
            self.ruptured = True
        if self.ruptured:
            # a small but positive modulus helps to maintain numeric stability
            c = 1.0 / (100.0 * np.finfo(float).eps)
            return c
        stress0 = self.yR[1]
        if stress <= abs(stress0) * (1.0 + 1000.0 * np.finfo(float).eps):
            # same elastic compliance as at zero strain
            c = (self.E1 + self.E2) / (self.E1 * self.E2)
        else:
            # Freed-Rajagopal elastic fiber model in hyper-elastic form
            stress_t = self.E1 * self.e_t
            c = ((self.e_t / (stress - stress0))
                 * (1.0 - m.sqrt(stress_t)
                    / m.sqrt(stress_t + 2.0 * (stress - stress0)))
                 + 1.0 / self.E2)
        return c

    def _tanCompliance(self, stress, mechanicalStrain, thermalStrain):
        if stress > self.e_f * self.E2:
            self.ruptured = True
        if self.ruptured:
            # a small but positive modulus helps to maintain numeric stability
            c = 1.0 / (100.0 * np.finfo(float).eps)
            return c
        stress0 = self.yR[1]
        if stress <= abs(stress0):
            # same elastic compliance as at zero strain
            c = (self.E1 + self.E2) / (self.E1 * self.E2)
        else:
            # Freed-Rajagopal elastic fiber model in hypo-elastic form
            e1 = mechanicalStrain - (self.alpha * thermalStrain
                                     + (stress - stress0) / self.E2)
            if e1 < self.e_t:
                c = ((self.e_t - e1)
                     / (self.E1 * self.e_t + 2.0 * (stress - stress0))
                     + 1.0 / self.E2)
            else:
                c = 1.0 / self.E2
        return c

    # inherited methods

    def secantModulus(self, eVec, xVec, yVec):
        # call the base type to verify the inputs and to create the matrix E
        Es = super().secantModulus(eVec, xVec, yVec)
        # y - y0 = E * e
        #    e   is a vector of thermodynamic control variables  (strains)
        #    x   is a vector of physical control variables       (stretches)
        #    y   is a vector of thermodynamic response variables (stresses)
        # populate the entries of E for the user's secant moduli below
        temperature = xVec[0]       # temperature                (in C)
        stress0 = self.yR[1]        # initial or residual stress (in barye)
        stress = yVec[1]            # stress                     (in barye)
        E = 1.0 / self._secCompliance(stress)
        rhoT = self.rho * (273.0 + temperature)
        Cs = self.alpha * (stress - stress0) / rhoT
        Ce = self.alpha**2 * E / rhoT
        # compute the tangent modulus
        Es[0, 0] = self.Cp - Cs - Ce
        Es[0, 1] = Ce / self.alpha
        Es[1, 0] = -self.alpha * E
        Es[1, 1] = E
        return Es

    def tangentModulus(self, eVec, xVec, yVec):
        # call the base type to verify the inputs and to create matrix dyde
        Et = super().tangentModulus(eVec, xVec, yVec)
        # dy = dyde * de  where  de = de/dx dx/dt
        #    e   is a vector of thermodynamic control variables  (strains)
        #    x   is a vector of physical control variables       (stretches)
        #    y   is a vector of thermodynamic response variables (stresses)
        # populate the entries of dyde for the user's tangent moduli below
        temperature = xVec[0]       # temperature    (in centigrade)
        thermalStrain = eVec[0]     # ln(T/T_0)      (dimensionless)
        mechanicalStrain = eVec[1]  # ln(L/L_0)      (dimensionless)
        stress0 = self.yR[1]        # initial stress (in barye)
        stress = yVec[1]            # stress         (in barye)
        E = 1.0 / self._tanCompliance(stress, mechanicalStrain, thermalStrain)
        rhoT = self.rho * (273.0 + temperature)
        Cs = self.alpha * (stress - stress0) / rhoT
        Ce = self.alpha**2 * E / rhoT
        # compute the tangent modulus
        Et[0, 0] = self.Cp - Cs - Ce
        Et[0, 1] = Ce / self.alpha
        Et[1, 0] = -self.alpha * E
        Et[1, 1] = E
        return Et

    def isRuptured(self):
        if not self.ruptured:
            hasRuptured = super().isRuptured()
        else:
            hasRuptured = (True,)
        return hasRuptured

    def rupturedResponse(self, eVec, xVec, yBeforeVec):
        # call the base type to verify the input and to create vector rVec
        yAfterVec = super().rupturedResponse(eVec, xVec, yBeforeVec)
        # populate the entries for the ruptured response in rVec below
        # this will result in discontinuities in the response fields
        TN = xVec[0]                 # temperature at next node (centigrade)
        # compute the strains at rupture
        lnTonT0 = eVec[0]            # thermal strain     ln(T/T_0)
        epsilon = eVec[1]            # mechanical strain  ln(L/L_0)
        # a small but positive modulus helps to maintain numeric stability
        E = 100.0 * np.finfo(float).eps
        # provides for a discontinuity in the response vector
        yAfterVec[0] = (self.yR[0] + (self.Cp - 4.0 * self.alpha**2 * E
                                      / (self.rho * (273.0 + TN))) * lnTonT0)
        yAfterVec[1] = E * epsilon
        return yAfterVec


# constitutive class for alveolar chords


class SeptalChord(Response):
    # implements the Freed-Rajagopal model for septal chords, which are
    # comprised of collagen and elastin fibers loaded in parallel, where
    # for this model
    #     eVec[0]  contains fiber temperature strain:       ln(T/T_0)
    #     eVec[1]  contains fiber mechanical  strain:       ln(L/L_0)
    # while
    #     xVec[0]  contains fiber temperature               (in centigrade)
    #     xVec[1]  contains fiber length                    (in cm)
    # and
    #     yVec[0]  contains collagen fiber entropy density  (in erg/g.K)
    #     yVec[1]  contains collagen fiber stress           (in barye)
    #     yVec[2]  contains elastin  fiber entropy density  (in erg/g.K)
    #     yVec[3]  contains elastin  fiber stress           (in barye)

    # constructor

    def __init__(self, diaCollagen=None, diaElastin=None):
        fiberResponses = 2
        yFiber0 = np.zeros((fiberResponses,), dtype=float)
        chordResponses = 2 * fiberResponses
        yChord0 = np.zeros((chordResponses,), dtype=float)
        # create the collagen fiber
        rho_c = mp.rhoCollagen()
        Cp_c = mp.CpCollagen()
        alpha_c = mp.alphaCollagen()
        # the mechanical properties below come from probability distributions
        E1_c, E2_c, et_c, ef_c, s0_c = mp.collagenFiber()
        yFiber0[0] = mp.etaCollagen()
        yFiber0[1] = s0_c
        self.fiberC = BioFiber(yFiber0, rho_c, Cp_c, alpha_c,
                               E1_c, E2_c, et_c, ef_c)
        yChord0[0] = yFiber0[0]        # initial collagen entropy density
        yChord0[1] = yFiber0[1]        # initial collagen pre-stress
        # create the elastin fiber
        rho_e = mp.rhoElastin()
        Cp_e = mp.CpElastin()
        alpha_e = mp.alphaElastin()
        # the mechanical properties below come from probability distributions
        E1_e, E2_e, et_e, s0_e = mp.elastinFiber()
        yFiber0[0] = mp.etaElastin()
        yFiber0[1] = s0_e
        self.fiberE = BioFiber(yFiber0, rho_e, Cp_e, alpha_e, E1_e, E2_e, et_e)
        yChord0[2] = yFiber0[0]         # initial elastin engropy density
        yChord0[3] = yFiber0[1]         # initial elastin pre-stress
        # call the base type constructor to create its data structure
        super().__init__(yChord0)
        # verify and initialize the remaining data
        if diaCollagen is None:
            diaCollagen = mp.fiberDiameterCollagen()
        elif diaCollagen < 0.000005 or diaCollagen > 0.0005:
            raise RuntimeError("Diameter of the collagen fiber must be "
                               + "within the range of 0.05 to 5 microns.")
        else:
            pass
        if diaElastin is None:
            diaElastin = mp.fiberDiameterElastin()
        elif diaElastin < 0.000005 or diaElastin > 0.0005:
            raise RuntimeError("Diameter of the elastin fiber must be "
                               + "within the range of 0.05 to 5 microns.")
        else:
            pass
        self.A0_c = m.pi * diaCollagen**2 / 4.0
        self.A0_e = m.pi * diaElastin**2 / 4.0
        # set a hook for initializing the control fields in the data structure
        # extract the response variables for export
        self.eta0_c = mp.etaCollagen()
        self.stress0_c = s0_c
        self.eta0_e = mp.etaElastin()
        self.stress0_e = s0_e
        self.eta_c = mp.etaCollagen()
        self.stress_c = s0_c
        self.eta_e = mp.etaElastin()
        self.stress_e = s0_e
        return  # a new instance of type ceChord

    # inherited methods

    def secantModulus(self, eVec, xVec, yVec):
        # extract initial control variables for export: must be before super
        if self.firstCall:
            self.strn0 = eVec[1]
            self.temp0 = xVec[0]
            self.len_0 = xVec[1]
        # verify inputs and create the matrix for the secant modulus
        Es = super().secantModulus(eVec, xVec, yVec)
        # assemble the secant moduli
        fiberResp = np.zeros((2,), dtype=float)
        for i in range(2):
            fiberResp[i] = yVec[i]
        EsC = self.fiberC.secantModulus(eVec, xVec, fiberResp)
        for i in range(2):
            fiberResp[i] = yVec[2 + i]
        EsE = self.fiberE.secantModulus(eVec, xVec, fiberResp)
        for i in range(2):
            Es[i, :] = EsC[i, :]
            Es[2 + i, :] = EsE[i, :]
        # extract the controlled variables for export
        self.strn = eVec[1]
        self.temp = xVec[0]
        self.len_ = xVec[1]
        # extract the response variables for export
        self.eta_c = yVec[0]
        self.stress_c = yVec[1]
        self.eta_e = yVec[2]
        self.stress_e = yVec[3]
        return Es

    def tangentModulus(self, eVec, xVec, yVec):
        # extract initial control variables for export: must be before super
        if self.firstCall:
            self.strn0 = eVec[1]
            self.temp0 = xVec[0]
            self.len_0 = xVec[1]
        # verify inputs and create the matrix for the secant modulus
        Et = super().tangentModulus(eVec, xVec, yVec)
        # assemble the tangent moduli
        fiberResp = np.zeros((2,), dtype=float)
        for i in range(2):
            fiberResp[i] = yVec[i]
        EtC = self.fiberC.tangentModulus(eVec, xVec, fiberResp)
        for i in range(2):
            fiberResp[i] = yVec[2 + i]
        EtE = self.fiberE.tangentModulus(eVec, xVec, fiberResp)
        for i in range(2):
            Et[i, :] = EtC[i, :]
            Et[2 + i, :] = EtE[i, :]
        # extract the controlled variables for export
        self.strn = eVec[1]
        self.temp = xVec[0]
        self.len_ = xVec[1]
        # extract the response variables for export
        self.eta_c = yVec[0]
        self.stress_c = yVec[1]
        self.eta_e = yVec[2]
        self.stress_e = yVec[3]
        return Et

    def isRuptured(self):
        ruptured_c = self.fiberC.isRuptured()
        ruptured_e = self.fiberE.isRuptured()
        ruptured = ruptured_c + ruptured_e        # concatination of two tuples
        return ruptured

    def rupturedResponse(self, eVec, xVec, yBeforeVec):
        # call the base type to verify the input and to create vector yAfterVec
        yAfterVec = super().rupturedResponse(eVec, xVec, yBeforeVec)
        # populate the entries for the ruptured response in yAfterVec below
        # this will result in a discontinuity in the response fields
        yVec = np.zeros((2,), dtype=float)
        # for collagen
        (ruptured,) = self.fiberC.isRuptured()
        if ruptured:
            yVec[0] = yBeforeVec[0]
            yVec[1] = yBeforeVec[1]
            rupC = self.fiberC.rupturedResponse(eVec, xVec, yVec)
            yAfterVec[0] = rupC[0]
            yAfterVec[1] = rupC[1]
        else:
            yAfterVec[0] = yBeforeVec[0]
            yAfterVec[1] = yBeforeVec[1]
        # for elastin
        (ruptured,) = self.fiberE.isRuptured()
        if ruptured:
            yVec[0] = yBeforeVec[2]
            yVec[1] = yBeforeVec[3]
            rupE = self.fiberE.rupturedResponse(eVec, xVec, yVec)
            yAfterVec[2] = rupE[0]
            yAfterVec[3] = rupE[1]
        else:
            yAfterVec[2] = yBeforeVec[2]
            yAfterVec[3] = yBeforeVec[3]
        # extract the controlled variables for export
        self.strn = eVec[1]
        self.temp = xVec[0]
        self.len_ = xVec[1]
        # extract the response variables for export
        self.eta_c = yAfterVec[0]
        self.stress_c = yAfterVec[1]
        self.eta_e = yAfterVec[2]
        self.stress_e = yAfterVec[3]
        return yAfterVec

    # additional methods

    # These methods associate with the end of the current integration interval
    # in other words, at the 'next' knot/node of integration

    # the response object for collagen
    def bioFiberCollagen(self):
        return self.fiberC

    # the response object for elastin
    def bioFiberElastin(self):
        return self.fiberE

    def massDensity(self):
        mass_c = mp.rhoCollagen() * self.A0_c * self.len_0
        mass_e = mp.rhoElastin() * self.A0_e * self.len_0
        massDensity = ((mass_c + mass_e) / self.volume())
        return massDensity  # in g/cm^3

    def length(self):
        if self.firstCall:
            L = self.len_0
        else:
            L = self.len_
        return L

    def areaCollagen(self):
        if self.firstCall:
            a = self.A0_c
        else:
            (ruptured,) = self.fiberC.isRuptured()
            if ruptured:
                a = self.A0_c
            else:
                a = self.A0_c * self.len_0 / self.len_
        return a

    def areaElastin(self):
        if self.firstCall:
            a = self.A0_e
        else:
            (ruptured,) = self.fiberE.isRuptured()
            if ruptured:
                a = self.A0_e
            else:
                a = self.A0_e * self.len_0 / self.len_
        return a

    def area(self):
        a = self.areaCollagen() + self.areaElastin()
        return a

    def volumeCollagen(self):
        vol = self.A0_c * self.len_0
        return vol

    def volumeElastin(self):
        vol = self.A0_e * self.len_0
        return vol

    def volume(self):
        vol = (self.A0_c + self.A0_e) * self.len_0
        return vol

    # absolute measures

    def temperature(self):
        body_temp = 37.0  # in degrees centigrade
        if self.firstCall:
            theta = body_temp
        else:
            theta = self.temp
        return theta

    def entropy(self):
        if self.firstCall:
            S = (self.eta0_c * mp.rhoCollagen() * self.A0_c * self.len_0
                 + self.eta0_e * mp.rhoElastin() * self.A0_e * self.len_0)
        else:
            S = (self.eta_c * mp.rhoCollagen() * self.A0_c * self.len_0
                 + self.eta_e * mp.rhoElastin() * self.A0_e * self.len_0)
        return S

    def strain(self):
        if self.firstCall:
            e = 0.0
        else:
            e = self.strn
        return e

    def stress(self):
        sigma = self.force() / self.area()
        return sigma

    def force(self):
        if self.firstCall:
            f = self.stress0_c * self.A0_c + self.stress0_e * self.A0_e
        else:
            f = (self.stress_c * self.areaCollagen()
                 + self.stress_e * self.areaElastin())
        return f

    # relative measures, i.e., current minus initial values for the fields

    def relativeTemperature(self):
        if self.firstCall:
            delTemp = 0.0
        else:
            delTemp = self.temp - self.temp0
        return delTemp

    def relativeEntropy(self):
        if self.firstCall:
            delEta = 0.0
        else:
            delEta = (((self.eta_c - self.eta0_c)
                       * mp.rhoCollagen() * self.A0_c * self.len_0)
                      + ((self.eta_e - self.eta0_e)
                         * mp.rhoElastin() * self.A0_e * self.len_0))
        return delEta

    def relativeStrain(self):
        if self.firstCall:
            delStrain = 0.0
        else:
            delStrain = self.strn - self.strn0
        return delStrain

    def relativeStress(self):
        sigma = self.relativeForce() / self.area()
        return sigma

    def relativeForce(self):
        if self.firstCall:
            delF = 0.0
        else:
            f0 = self.stress0_c * self.A0_c + self.stress0_e * self.A0_e
            f = (self.stress_c * self.areaCollagen()
                 + self.stress_e * self.areaElastin())
            delF = f - f0
        return delF


"""
Changes made in version "1.0.0":

This is the initial version.
"""
 