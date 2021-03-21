#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 05:24:07 2020

@author: al
"""

import meanPropertMicron as mp
import math as m
import numpy as np
from peceHE import Control, Response

"""
Module ceMembranes.py provides a constitutive description for alveolar septa.

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
"""

# Module metadata
__version__ = "1.0.0"
__date__ = "05-28-2020"
__update__ = "11-15-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


This module describes septal membranes.  From histological studies, septa
are found to be comprised of both collagen and elastin fibers, randomly and
somewhat uniformly oriented, with minimal chemical bonding between them.
Also present are cells and extracellular protiens that are collectively known
as the ground substance, along with interstitual fluids and blood.  Unlike
septal chords, where we know enough to implement a mixture theory, septal
membranes are less well known from a mechanics perspective, and as such, are
modeled as an homogeneous membrane whose constitutive parameters are described
by probability distributions that are exported from materialProperties.py.
Given these material properties, one can create objects describing realistic
septal membranes.  This module exports two classes:
    controlMembrane: manages the control variables for a biologic membrane, and
    ceMembrane:      manages the response variables of a biologic membrane.

The CGS system of physical units adopted:
    length          centimeters   [cm]
    mass            grams         [g]
    time            seconds       [s]
    temperature     centigrade    [C]
where
    force           dynes         [g.cm/s^2]      1 Newton = 10^5 dyne
    pressure        barye         [dyne/cm^2]     1 Pascal = 10 barye
    energy          erg           [dyne.cm]       1 Joule  = 10^7 ergs


class controlMembrane:  It implements and extends class 'control'


For 2D membranes, the physical control vector has components of:
    xVec[0]  temperature                'T'                   (centigrade)
    xVec[1]  elongation in 1 direction  'a'                   (dimensionless)
    xVec[2]  elongation in 2 direction  'b'                   (dimensionless)
    xVec[3]  magnitude of shear         'g'                   (dimensionless)
while the thermodynamic control vector has strain components of:
    eVec[0]  thermal strain:            ln(T/T_0)             (dimensionless)
    eVec[1]  dilation:                  ln(a/a_0 * b/b_0)     (dimensionless)
    eVec[2]  squeeze:                   ln(a/a_0 * b_0/b)     (dimensionless)
    eVec[3]  shear:                     g - g_0               (dimensionless)
where a, b and g come from the QR decomposition of a deformation gradient,
which needs to be pivoted prior to its decomposition into a, b and g.

constructor

    E.g.: ctrl = controlMembrane(eVec0, xVec0, dt)
        eVec0       a vector of thermodynamic control variables at reference
        xVec0       a vector of physical control variables at the reference
        dt          size of the time step to be used for numeric integration

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

inherited methods

update(xVec, restart=False)
    E.g.:  ctrl.update(xVec, restart)
        xVec        a vector of physical control variables for the next node
        restart     whenever restart is True, the trapezoidal method is used;
                    otherwise, Gear's BDF2 method is used for integration
    ctrl.update may be called multiple times before freezing its values with a
    call to ctrl.advance.  This is important in a finite element application.

advance()
    E.g., ctrl.advance()
    Updates the object's data structure in preparation for the next integration
    step.  It moves current data into their previous fields, and then it moves
    next data into their current fields.  This method is called internally by
    the pece object in peceHE.py and should not be called by the user.

dedx()
    E.g.:  dedxMtx = ctrl.dedx()
        dedxMtx     a matrix containing the mapping of physical control rates
                    into their thermodynamic control rates.
    This transformation associates with the next node.  It is created as an
    identity matrix in the base class whose components are overwritten here.

dxdt()
    E.g.:  dxdtVec = ctrl.dxdt()
        dxdtVec     is a vector containing a rate-of-change in the controls
    This base method implements finite difference formulae to approximate this
    derivative. A first-order difference formula is used for the reference and
    first nodes, plus the first two nodes after a restart has been mandated.
    A second-order backward difference formula is used for all other nodes.
    All derivatives associate with the next node.  These rates are overwritten
    in this implementation for the shear term, but not the others.


class ceMembrane:  It implements and extends class 'response'.


Creates objects that implement a biologic membrane model.
For this model, the physical control vector has components of:
    xVec[0]  temperature                'T'                   (centigrade)
    xVec[1]  elongation in 1 direction  'a'                   (dimensionless)
    xVec[2]  elongation in 2 direction  'b'                   (dimensionless)
    xVec[3]  magnitude of shear         'g'                   (dimensionless)
while the thermodynamic control vector has strain components of:
    eVec[0]  thermal strain:            ln(T/T_0)             (dimensionless)
    eVec[1]  dilation:                  ln(a/a_0 * b/b_0)     (dimensionless)
    eVec[2]  squeeze:                   ln(a/a_0 * b_0/b)     (dimensionless)
    eVec[3]  shear:                     g - g_0               (dimensionless)
and the thermodynamic response vector has componenents of:
    yVec[0]  entropy density            'eta'                 (erg/g.K)
    yVec[1]  surface tension            'pi'                  (barye)
    yVec[2]  normal stress difference   'sigma'               (barye)
    yVec[3]  shear stress               'tau'                 (barye)

constructor

    E.g.: ce = ceMembrane(thickness=None)
        thickness   width of the membrane
    If the thickness takes on its default value of None, then it is assigned
    via its respective probability distribution; otherwise, it must be between
    2 and 7.5 microns in size for our application.

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    responses       an integer specifying the number of response variables
    eR              a vector containing the initial thermodynamic controls
    xR              a vector containing the initial physical controls
    yR              a vector containing the initial conditions for responses
    eN              a vector of thermodynamic controls at the next node
    xN              a vector of physical controls at the next node
    yN              a vector of thermodynamic responses at the next node

inherited methods

secantModulus(eVec, xVec, yVec)
    E.g.:  E = ce.secantModulus(eVec, xVec, yVec)
        Es          a matrix of secant moduli, i.e., the constitutive matrix
        eVec        a vector of thermodynamic control variables  (strains)
        xVec        a vector of physical control variables       (stretches)
        yVec        a vector of thermodynamic response variables (stresses)
    Solves hyper-elastic equation of form:  s - s_0 = Es * e.

secMod(eVec, xVec, yVec)
        eVec        a vector of thermodynamic control variables  (strains)
        xVec        a vector of physical control variables       (stretches)
        yVec        a vector of thermodynamic response variables (stresses)

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
        de/dt       is supplied by objects from class controlMembrane
    Solves a hypo-elastic equation of the form: ds = Et * de.

tanMod(eVec, xVec, yVec)
        eVec        a vector of thermodynamic control variables  (strains)
        xVec        a vector of physical control variables       (stretches)
        yVec        a vector of thermodynamic response variables (stresses)

isRuptured()
    E.g.:  ruptured = ce.isRuptured()
        ruptured    a tuple of boolean results specifying if a specific
                    constituent has failed.  For a membrane, there is no
                    mixture theory applied so the tuple has length of one.

rupturedResponse(eVec, xVec, yBeforeVec)
    E.g., yAfterVec = ce.rupturedResponse(eVec, xVec, yBeforeVec)
        eVec        vector of thermodynamic control variables at rupture
        xVec        vector of physical control variables at rupture
        yBeforeVec  vector of response variables just before rupture occurs
    returns
        yAfterVec   vector of response variables just after a rupture event
    Calling this method, which is done internally by the 'pece' integrator,
    allows for a discontinuity in the field of thermodynamic responses.  Only
    the dilational response is considered capable of rupture in this modeling.

additional methods

massDensity()
    E.g., rho = ce.massDensity()
        rho         returns the mass density of the septal membrane

thickness()
    E.g., w = ce.thickness()
        w           returns the width or thickness of the membrane

stretch()
    E.g., U = ce.stretch()
        U           returns Laplace stretch in co-ordinate frame of membrane

stretchInv()
    E.g., Uinv = ce.stretchInv()
        Uinv        returns the inverse of Laplace stretch

# absolute measures

temperature()
    E.g., temp = ce.temperature()
        temp        returns the temperature of the septal membrane

entropyDensity()
    E.g., eta = ce.entropyDensity()
        eta         returns the entropy density of the septal membrane

stressMtx()
    E.g., S = ce.stressMtx()
        S           returns stress matrix in the co-ordinate frame of the 
        membrane

intensiveStressVec()
    E.g., T = ce.intensiveStressVec()
        T           returns stress vector T conjugate to strain vector E
        
# relative measures

relativeTemperature()
    E.g., temp = ce.relativeTemperature()
        temp        returns the relative temperature of the septal membrane

relativeEntropyDensity()
    E.g., eta = ce.relativeEntropyDensity()
        eta         returns the relative entropy density of the septal membrane

relativeStress()
    E.g., S = ce.relativeStress()
        S           returns relative stress in the membrane's co-ordinate frame

References:
    1) Freed, A. D. and Rajagopal, K. R., “A Promising Approach for Modeling
       Biological Fibers,” ACTA Mechanica, 227 (2016), 1609-1619.
       DOI: 10.1007/s00707-016-1583-8.  Errata: DOI: 10.1007/s00707-018-2183-6
    2) Freed, A. D., Erel, V. and Moreno, M. R., “Conjugate Stress/Strain Base
       Pairs for the Analysis of Planar Biologic Tissues”, Journal of Mechanics
       of Materials and Structures, 12 (2017), 219-247.
       DOI: 10.2140/jomms.2017.12.219
"""


class controlMembrane(Control):

    # control vector arguments have interpretations of:
    # variables inherited from the base type: treat these as read-only:
    #   controls    an integer specifying the number of control variables
    #   node        an integer specifying the current node of integration,
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
    # For this model, the physical control vector has components of:
    #     xVec[0]  temperature                   'T'           (centigrade)
    #     xVec[1]  elongation in 1 direction     'a'           (dimensionless)
    #     xVec[2]  elongation in 2 direction     'b'           (dimensionless)
    #     xVec[3]  shear magnitude               'g'           (dimensionless)
    # while the thermodynamic control vector has strain components of:
    #     eVec[0]  thermal strain:       ln(T/T_0)             (dimensionless)
    #     eVec[1]  dilation:             ln(a/a_0 * b/b_0)     (dimensionless)
    #     eVec[2]  squeeze:              ln(a/a_0 * b_0/b)     (dimensionless)
    #     eVec[3]  shear:                g - g_0               (dimensionless)

    def __init__(self, eVec0, xVec0, dt):
        # Call the constructor of the base type to create and initialize the
        # exported variables.
        super().__init__(eVec0, xVec0, dt)
        # Create and initialize any additional fields introduced by the user.
        if self.controls != 4:
            raise RuntimeError("There are 4 control variables for a 2D " +
                               "membrane: temperature, two elongation " +
                               "ratios, and a shear.")
        return  # a new instance of type controlMembrane

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
        # Because the matrix created by the super call is an identity matrix,
        # only a few of the cells need to be overwritten
        T = 273.0 + self.xN[0]   # convert Centigrade into Kelvin
        a = self.xN[1]
        b = self.xN[2]
        dedxMtx[0, 0] = 1.0 / T
        dedxMtx[1, 1] = 1.0 / (2.0 * a)
        dedxMtx[1, 2] = 1.0 / (2.0 * b)
        dedxMtx[2, 1] = 1.0 / (2.0 * a)
        dedxMtx[2, 2] = -1.0 / (2.0 * b)
        return dedxMtx

    def dxdt(self):
        # Call the base implementation of this method to create vector dxdtVec.
        dxdtVec = super().dxdt()
        # The returned dxdtVec is computed via finite difference formulae;
        # specifically,
        #   if self.node is 0, 1   use first-order difference formula
        #   if restart is True     use first-order difference formula
        #   otherwise              use second-order backward difference formula
        # This is correct for all but the shear term, which is redefined below.
        aN = self.xN[1]
        aC = self.xC[1]
        if self.node == 0:
            dxdtVec[3] *= aN / aC
        elif self.node == 1:
            dxdtVec[3] *= aC / aN
        else:
            aP = self.xP[1]
            gN = self.xN[3]
            gC = self.xC[3]
            gP = self.xP[3]
            dxdtVec[3] = (2.0 * (aC / aN) * (gN - gC) / self.dt -
                          (aP / aN) * (gN - gP) / (2.0 * self.dt))
        return dxdtVec


# constitutive class for biologic membranes

class ceMembrane(Response):
    # implements the Freed-Rajagopal model for biologic fibers as a membrane
    # model, akin to that of Freed, Erel and Moreno, where
    #   xVec[0]  temperature                  'T'            (centigrade)
    #   xVec[1]  elongation in 1 direction    'a'            (dimensionless)
    #   xVec[2]  elongation in 2 direction    'b'            (dimensionless)
    #   xVec[3]  magnitude of shear           'g'            (dimensionless)
    # while the thermodynamic control vector has strain-like components of:
    #   eVec[0]  thermal strain:    ln(T/T_0)                (dimensionless)
    #   eVec[1]  dilation:          ln(sqrt(a/a_0 * b/b_0))  (dimensionless)
    #   eVec[2]  squeeze:           ln(sqrt(a/a_0 * b_0/b))  (dimensionless)
    #   eVec[3]  shear:             g - g_0                  (dimensionless)
    # and the thermodynamic response vector has stress-like components of:
    #   yVec[0]  entropy density              'eta'          (erg/g.K)
    #   yVec[1]  surface tension              'pi'           (barye)
    #   yVec[2]  normal stress difference     'sigma'        (barye)
    #   yVec[3]  shear stress                 'tau'          (barye)

    def __init__(self, thickness=None):
        # dimension the problem
        controls = 4
        responses = 4
        # get the material properties
        M1, M2, e_1, N1, N2, e_2, G1, G2, e_3, e_f, pi_0 = mp.septalMembrane()
        # assign these material properties to the object
        self.rho = mp.rhoSepta()
        self.Cp = mp.CpSepta()
        self.alpha = mp.alphaSepta()
        self.M1 = M1
        self.M2 = M2
        self.xi_t = e_1
        self.N1 = N1
        self.N2 = N2
        self.epsilon_t = e_2
        self.G1 = G1
        self.G2 = G2
        self.gamma_t = e_3
        self.xi_f = e_f
        # establish the initial conditions for the thermodynamic responses
        yVec0 = np.zeros((responses,), dtype=float)
        yVec0[0] = mp.etaSepta()   # initial entropy density
        yVec0[1] = pi_0            # initial surface tension
        yVec0[2] = 0.0             # initial normal stress difference
        yVec0[3] = 0.0             # initial shear stress

        # create and initialize the two control vectors
        # eVec0 = np.zeros((controls,), dtype=float)
        # xVec0 = np.zeros((controls,), dtype=float)

        # now call the base type to create the exported response fields
        super().__init__(yVec0)
        # establish the geometric property of thickness
        if thickness is None:
            self.thickness = mp.septalWidth()
        elif thickness >= 2.0 and thickness <= 7.5:
            self.thickness = thickness
        else:
            raise RuntimeError("Thickness of a septal membrane must lie " +
                               "within the range of 2 to 7.5 microns.")
        # set default value for membrane rupture
        self.ruptured = False
        # create vectors for the next node used in the output methods
        self.eN = np.zeros((controls,), dtype=float)
        self.xN = np.zeros((controls,), dtype=float)
        self.yN = np.zeros((responses,), dtype=float)
        return  # a new instance of this constitutive membrane object

    def _M_secCompliance(self, s_pi):
        # a membrane can only rupture under an excessive surface tension
        if s_pi > self.xi_f * self.M2:
            self.ruptured = True
        # construct the secant compliance governing the dilation response
        s0_pi = self.yR[1]
        if self.ruptured:
            # a small but positive modulus helps to maintain numeric stability
            M = 100.0 * np.finfo(float).eps
            c = 1.0 / M
        elif s_pi < abs(s0_pi) * (1.0 + 1000.0 * np.finfo(float).eps):
            # same elastic compliance as at zero strain
            c = (self.M1 + self.M2) / (self.M1 * self.M2)
        else:
            # bio membrane model for uniform response
            c = ((4.0 * self.xi_t) / (s_pi - s0_pi) *
                 (1.0 - m.sqrt(self.M1 * self.xi_t) /
                  m.sqrt(self.M1 * self.xi_t + (s_pi - s0_pi) / 2.0)) +
                 1.0 / self.M2)
        return c

    def _N_secCompliance(self, s_sigma):
        # construct the secant compliance governing the squeeze response
        if self.ruptured:
            # a small but positive modulus helps to maintain numeric stability
            N = 100.0 * np.finfo(float).eps
            c = 1.0 / N
        elif abs(s_sigma) < (1000.0 * np.finfo(float).eps):
            # same elastic compliance as at zero strain
            c = (self.N1 + self.N2) / (self.N1 * self.N2)
        else:
            c = ((2.0 * self.epsilon_t) / abs(s_sigma) *
                 (1.0 - m.sqrt(self.N1 * self.epsilon_t) /
                  m.sqrt(self.N1 * self.epsilon_t + abs(s_sigma))) +
                 1.0 / self.N2)
        return c

    def _G_secCompliance(self, s_tau):
        # construct the secant compliance governing the shear response
        if self.ruptured:
            # a small but positive modulus helps to maintain numeric stability
            G = 100.0 * np.finfo(float).eps
            c = 1.0 / G
        elif abs(s_tau) < (1000.0 * np.finfo(float).eps):
            # same elastic compliance as at zero strain
            c = (self.G1 + self.G2) / (self.G1 * self.G2)
        else:
            c = (self.gamma_t / abs(s_tau) *
                 (1.0 - m.sqrt(self.G1 * self.gamma_t) /
                  m.sqrt(self.G1 * self.gamma_t + 2.0 * abs(s_tau))) +
                 1.0 / self.G2)
        return c

    def secantModulus(self, eVec, xVec, yVec):
        # call the base type to verify the inputs and to create matrix ceMtx
        Es = super().secantModulus(eVec, xVec, yVec)
        # assemble the secant moduli
        # y = y0 + Es * e
        #    e   is a vector of thermodynamic control variables  (strains)
        #    x   is a vector of physical control variables       (stretches)
        #    y   is a vector of thermodynamic response variables (stresses)
        # populate the entries of Es for the user's secant moduli below
        temperature = xVec[0]       # temperature                (C)
        s_pi0 = self.yR[1]          # initial surface tension    (barye)
        s_pi = yVec[1]              # surface tension            (barye)
        s_sigma = yVec[2]           # normal stress difference   (barye)
        s_tau = yVec[3]             # shear stress               (barye)
        M = 1.0 / self._M_secCompliance(s_pi)
        N = 1.0 / self._N_secCompliance(s_sigma)
        G = 1.0 / self._G_secCompliance(s_tau)
        rhoT = self.rho * (273.0 + temperature)
        Cs = self.alpha * (s_pi - s_pi0) / rhoT
        Ce = 4.0 * self.alpha**2 * M / rhoT
        # compute the tangent modulus
        Es[0, 0] = self.Cp - Cs - Ce
        Es[0, 1] = Ce / self.alpha
        Es[1, 0] = -4.0 * self.alpha * M
        Es[1, 1] = 4.0 * M
        Es[2, 2] = 2.0 * N
        Es[3, 3] = G
        # update the exported vector fields
        self.eN[:] = eVec[:]
        self.xN[:] = xVec[:]
        self.yN[:] = yVec[:]
        return Es
    
    def secMod(self, eVec, xVec, yVec):    
        Es = self.secantModulus(eVec, xVec, yVec)
        Ms = np.zeros((3, 3), dtype=float)
        Ms[0, 0] = Es[1, 1]
        Ms[1, 1] = Es[2, 2] 
        Ms[2, 2] = Es[3, 3]
        return Ms    

    def _M_tanCompliance(self, temp, xi, s_pi):
        # a membrane can only rupture under an excessive surface tension
        if s_pi > self.xi_f * self.M2:
            self.ruptured = True
        # construct the tangent compliance governing the dilation response
        s0_pi = self.yR[1]
        if self.ruptured:
            # a small but positive modulus helps to maintain numeric stability
            M = 100.0 * np.finfo(float).eps
            c = 1.0 / M
        elif s_pi < abs(s0_pi):
            # same elastic compliance as at zero strain
            c = (self.M1 + self.M2) / (self.M1 * self.M2)
        else:
            # bio membrane model for uniform response
            temp0 = self.xR[0]
            lnTonT0 = m.log((273.0 + temp) / (273.0 + temp0))
            xi1 = xi - self.alpha * lnTonT0 - (s_pi - s0_pi) / (4.0 * self.M2)
            if xi1 < self.xi_t:
                c = ((self.xi_t - xi1) /
                     (self.M1 * self.xi_t + (s_pi - s0_pi) / 2.0) +
                     1.0 / self.M2)
            else:
                c = 1.0 / self.M2
        return c

    def _N_tanCompliance(self, epsilon, s_sigma):
        # construct the tangent compliance that governs the squeeze response
        if self.ruptured:
            # a small but positive modulus helps to maintain numeric stability
            N = 100.0 * np.finfo(float).eps
            c = 1.0 / N
        else:
            epsilon1 = epsilon - s_sigma / (2.0 * self.N2)
            if epsilon1 > np.finfo(float).eps:
                if epsilon1 < self.epsilon_t:
                    c = ((self.epsilon_t - epsilon1) /
                         (self.N1 * self.epsilon_t + s_sigma) + 1.0 / self.N2)
                else:
                    c = 1.0 / self.N2
            elif epsilon1 < -np.finfo(float).eps:
                if epsilon1 > -self.epsilon_t:
                    c = ((-self.epsilon_t - epsilon1) /
                         (-self.N1 * self.epsilon_t + s_sigma) + 1.0 / self.N2)
                else:
                    c = 1.0 / self.N2
            else:
                c = (self.N1 + self.N2) / (self.N1 * self.N2)
        return c

    def _G_tanCompliance(self, gamma, s_tau):
        # construct the tangent compliance that governs the shear response
        if self.ruptured is True:
            # a small but positive modulus helps to maintain numeric stability
            G = 100.0 * np.finfo(float).eps
            c = 1.0 / G
        else:
            gamma1 = gamma - s_tau / self.G2
            if gamma1 > np.finfo(float).eps:
                if gamma1 < self.gamma_t:
                    c = ((self.gamma_t - gamma1) /
                         (self.G1 * self.gamma_t + 2.0 * s_tau) +
                         1.0 / self.G2)
                else:
                    c = 1.0 / self.G2
            elif gamma1 < -np.finfo(float).eps:
                if gamma1 > -self.gamma_t:
                    c = ((-self.gamma_t - gamma1) /
                         (-self.G1 * self.gamma_t + 2.0 * s_tau) +
                         1.0 / self.G2)
                else:
                    c = 1.0 / self.G2
            else:
                c = (self.G1 + self.G2) / (self.G1 * self.G2)
        return c

    def tangentModulus(self, eVec, xVec, yVec):
        # call the base type to verify the inputs and to create matrix ceMtx
        Et = super().tangentModulus(eVec, xVec, yVec)
        # assemble the tangent moduli
        # dy = Et * de
        #    e   a vector of thermodynamic control variables  (strains)
        #    x   a vector of physical control variables       (stretches)
        #    y   a vector of thermodynamic response variables (stresses)
        # populate the entries of Et for the user's tangent moduli below
        temperature = xVec[0]    # temperature                (C)
        xi = eVec[1]             # dilation                   (dimensionless)
        epsilon = eVec[2]        # squeeze                    (dimensionless)
        gamma = eVec[3]          # shear                      (dimensionless)
        s0_pi = self.yR[1]       # initial or residual stress (barye)
        s_pi = yVec[1]           # surface tension            (barye)
        s_sigma = yVec[2]        # normal stress difference   (barye)
        s_tau = yVec[3]          # shear stress               (barye)
        M = 1.0 / self._M_tanCompliance(temperature, xi, s_pi)
        N = 1.0 / self._N_tanCompliance(epsilon, s_sigma)
        G = 1.0 / self._G_tanCompliance(gamma, s_tau)
        rhoT = self.rho * (273.0 + temperature)
        Cs = self.alpha * (s_pi - s0_pi) / rhoT
        Ce = 4.0 * self.alpha**2 * M / rhoT
        # compute the tangent modulus
        Et[0, 0] = self.Cp - Cs - Ce
        Et[0, 1] = Ce / self.alpha
        Et[1, 0] = -4.0 * self.alpha * M
        Et[1, 1] = 4.0 * M
        Et[2, 2] = 2.0 * N
        Et[3, 3] = G
        # update the exported fields
        self.eN[:] = eVec[:]
        self.xN[:] = xVec[:]
        self.yN[:] = yVec[:]
        return Et

    def tanMod(self, eVec, xVec, yVec):    
        Et = self.tangentModulus(eVec, xVec, yVec)
        Mt = np.zeros((3, 3), dtype=float)
        Mt[0, 0] = Et[1, 1]
        Mt[1, 1] = Et[2, 2] 
        Mt[2, 2] = Et[3, 3]
        return Mt  
    
    def isRuptured(self):
        if self.firstCall:
            hasRuptured = (False,)
        elif not self.ruptured:
            hasRuptured = super().isRuptured()
        else:
            hasRuptured = (True,)
        return hasRuptured

    def rupturedRespVec(self, eVec, xVec, yBeforeVec):
        # call the base type to verify the input and to create vector rVec
        yAfterVec = super().rupturedRespVec(eVec, xVec, yBeforeVec)
        # populate the entries for the ruptured response in rVec below
        # this will result in discontinuities in the response fields
        TN = xVec[0]              # temperature at next node (centigrade)
        # compute the strains at rupture
        lnTonT0 = eVec[0]         # thermal strain
        xi = eVec[1]              # dilation
        epsilon = eVec[2]         # squeeze
        gamma = eVec[3]           # shear
        # a small but positive modulus helps to maintain numeric stability
        E = 100.0 * np.finfo(float).eps
        # provides for a discontinuity in the response vector
        yAfterVec[0] = (self.yR[0] + (self.Cp - 4.0 * self.alpha**2 * E /
                                      (self.rho * (273.0 + TN))) * lnTonT0)
        yAfterVec[1] = 4.0 * E * xi
        yAfterVec[2] = 2.0 * E * epsilon
        yAfterVec[3] = E * gamma
        return yAfterVec

    # additional methods

    def massDensity(self):
        return self.rho

    def thickness(self):
        if self.firstCall:
            width = self.thickness
        else:
            a0 = self.xR[1]
            b0 = self.xR[2]
            a = self.xN[1]
            b = self.xN[2]
            width = self.thickness * a0 * b0 / (a * b)
        return width

    def stretch(self):
        U = np.zeros((2, 2), dtype=float)
        if self.firstCall:
            a = self.xR[1]
            b = self.xR[2]
            g = self.xR[3]
        else:
            a = self.xN[1]
            b = self.xN[2]
            g = self.xN[3]
        U[0, 0] = a
        U[0, 1] = a * g
        U[1, 1] = b
        return U

    def stretchInv(self):
        Uinv = np.zeros((2, 2), dtype=float)
        if self.firstCall:
            a = self.xR[1]
            b = self.xR[2]
            g = self.xR[3]
        else:
            a = self.xN[1]
            b = self.xN[2]
            g = self.xN[3]
        Uinv[0, 0] = 1.0 / a
        Uinv[0, 1] = -g / b
        Uinv[1, 1] = 1.0 / b
        return Uinv

    # absolute measures

    def temperature(self):
        bodyTemp = 37.0
        if self.firstCall:
            theta = bodyTemp
        else:
            theta = self.xN[0]
        return theta

    def entropyDensity(self):
        if self.firtsCall:
            eta = mp.etaSepta()
        else:
            eta = self.yN[0]
        return eta

    def stressMtx(self):
        if self.firstCall:
            a = self.xR[1]
            b = self.xR[2]
            g = self.xR[3]
            pi = self.yR[1]
            sigma = self.yR[2]
            tau = self.yR[3]
        else:
            a = self.xN[1]
            b = self.xN[2]
            g = self.xN[3]
            pi = self.yN[1]
            sigma = self.yN[2]
            tau = self.yN[3]
        s = np.zeros((2, 2), dtype=float)
        # # Stresses are established in a physical frame of reference.
        # s00 = (pi + sigma) / 2.0
        # s01 = (b * tau) / a
        # s11 = (pi - sigma) / 2.0
        
        # # Stresses are established in a physical frame of reference.
        # s[0, 0] = s00 / a**2 - 2*g * s01 / (a * b) + (g / a)**2 * s11
        # s[0, 1] = s01 / (a * b) - g * s11 / (b**2)
        # s[1, 0] = s[0, 1]
        # s[1, 1] = s11 / (b**2)        
        
        
        # Stresses are established in a physical frame of reference.
        s[0, 0] = (pi + sigma) / 2.0
        s[0, 1] = (b * tau) / a
        s[1, 0] = s[0, 1]
        s[1, 1] = (pi - sigma) / 2.0 
        
        return s

    def intensiveStressVec(self):
        if self.firstCall:
            pi = self.yR[1]
            sigma = self.yR[2]
            tau = self.yR[3]
        else:
            pi = self.yN[1]
            sigma = self.yN[2]
            tau = self.yN[3]
        T = np.zeros((3,1), dtype=float)
        T[0, 0] = pi 
        T[1, 0] = sigma
        T[2, 0] = tau
        return T
    
    # relative measures

    def relativeTemperature(self):
        if self.firstCall:
            theta = 0.0
        else:
            theta = self.xN[0] - self.xR[0]
        return theta

    def relativeEntropyDensity(self):
        if self.firstCall:
            S = 0.0
        else:
            S = self.yN[0] - self.yR[0]
        return S

    def relativeStress(self):
        s = np.zeros((2, 2), dtype=float)
        if not self.firstCall:
            a = self.xN[1]
            b = self.xN[2]
            pi = self.yN[1] - self.yR[1]   # only surface tension has a prestress
            sigma = self.yN[2]
            tau = self.yN[3]
            s[0, 0] = (pi + sigma) / 2.0
            s[0, 1] = b * tau / a
            s[1, 0] = s[0, 1]
            s[1, 1] = (pi - sigma) / 2.0
        return s


"""
Changes made in version "1.0.0":

This is the initial version.
"""
