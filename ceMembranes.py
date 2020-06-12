#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 05:24:07 2020

@author: al
"""

import materialProperties as mp
import math as m
import numpy as np
from peceHE import control, response

"""
Module ceMembranes.py provides a constitutive description for alveolar septa.

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
__date__ = "05-28-2020"
__update__ = "05-29-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


This module describes septal membranes.  From histological studies, septa
are found to be comprised of both collagen and elastin fibers, randomly and
somewhat uniformly oriented, with minimal chemical bonding between them.
And also cells and extracellular protiens collectively known as the ground
substance along with interstitual fluids and blood.  Unlike septal chords,
where we know enough to implement a mixture theory, septal membranes are less
well known from a mechanics perspective, and as such, are modeled as an
homogeneous membrane whose constitutive parameters described by probability
distributions that are exported from materialProperties.py.  Given these
material properties, one can create objects describing realistic septa.
This module exports two classes:
    controlMembrane: manages the control variables for a biologic membrane, and
    septalMembrane:  manages the response variables of a biologic membrane.

The CGS system of physical units adopted:
    length          centimeters [cm]
    mass            grams       [g]
    time            seconds     [s]
    temperature     centigrade  [C]
where
    force           dynes       [g.cm/s^2]      1 Newton = 10^5 dyne
    pressure        barye       [dyne/cm^2]     1 Pascal = 10 barye
    energy          erg         [dyne.cm]       1 Joule  = 10^7 ergs


class controlMembrane:  It implements and extends class 'control'


For 2D membranes, the control vectors have components:
    ctrlVec[0]  membrane temperature     'theta' (centigrade)
    ctrlVec[1]  membrane elongation      'a'     (dimensionless)
    ctrlVec[2]  membrane elongation      'b'     (dimensionless)
    ctrlVec[3]  membrane shear magnitude 'g'     (dimensionless)
where a, b and g come from the QR decomposition of a deformation gradient,
which needs to be pivoted prior to its decomposition into a, b and g.

constructor

    E.g.: ctrl = controlMembrane(ctrlVec0, dt)
        ctrlVec0    a vector of control variables at the reference node, xR
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

inherited methods

update(ctrlVec)
    E.g., ctrl.update(ctrlVec)
        ctrlVec     a vector of control variables for the next node
    This method may be called multiple times before freezing its values with a
    call to ctrl.advance.

advance()
    E.g., ctrl.advance()
    Updates the object's data structure in preparation for the next integration
    step.  It moves current data into their previous fields, and then it moves
    next data into their current fields.  This method is called internally by
    the pece object in peceHE.py and should not be called by the user.

dedx()
    E.g., dedxMtx = ctrl.dedx()
        dedxMtx     a matrix containing the mapping the user's physical control
                    rates into their thermodynamic equivalent control rates.
    This transformation associates with the next node.  The base method is a
    phantom method that is overwritten here for membranes.

dxdt(restart)
    E.g., dxdtVec = ctrl.dxdt(restart=False)
        dxdtVec     is a vector returning the rate-of-change in the controls
        restart     set to True whenever there is a discontinuity in control
    This base method implements finite difference formulae to approximate this
    derivative.  A first-order difference formula is used for the initial and
    first nodes, plus whenever a restart is mandated.  A second-order backward
    difference formula is used for all other nodes.  All derivatives associate
    with the next node.  These rates are overwritten for the shear term.


class bioMembrane:  It implements and extends class 'response'.


Creates objects that implement a biologic membrane model.
For this model
    ctrlVec[0]  membrane temperature               'theta'  (centigrade)
    ctrlVec[1]  membrane elongation in 1 direction  'a'      (dimensionless)
    ctrlVec[2]  membrane elongation in 2 direction  'b'      (dimensionless)
    ctrlVec[3]  membrane in-plane shear magnitude   'g'      (dimensionless)
and
    respVec[0]  membrane entropy density            'eta'    (erg/g.K)
    respVec[1]  membrane surface tension            'pi'     (barye)
    respVec[2]  membrane normal stress difference   'sigma'  (barye)
    respVec[3]  membrane shear stress               'tau'    (barye)

constructor

    E.g.: ce = bioMembrane(ctrlVec0, respVec0, rho, Cp, alpha, M1, M2, e_Mt,
                           e_max, N1, N2, e_Nt, G1, G2, e_Gt, thickness=None)
        ctrlVec0    control variables in a reference state
        respVec0    response variables in a reference state: initial conditions
        rho         mass density
        Cp          specific heat at constant pressure
        alpha       coefficient of axial thermal expansion
        M1          compliant areal modulus
        M2          terminal areal modulus
        e_Mt        transition (compliant to stiff) areal strain
        e_max       sets rupture, where s_r = e_max * M2 is the rupture stress
        N1          compliant squeeze modulus
        N2          terminal squeeze modulus
        e_Nt        transition (compliant to stiff) squeeze strain
        G1          compliant shear modulus
        G2          termainal shear modulus
        e_Gt        transition (compliant to stiff) shear strain
        thickness   width of the membrane

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    responses       an integer specifying the number of response variables
    xR              a vector containing the initial condition for control
    yR              a vector containing the initial condition for response

inherited methods

tanMod(ctrlVec, respVec)
    E.g., dyde = ce.tanMod(ctrlVec, respVec)
        dyde        the matrix of tangent moduli dy/de (constitutive equation)
    The constitutive equation considered is hypo-elastic like; specifically,
        dy/dt = dy/de de/dx dx/dt
    wherein
        dy/dt       a vector of thermodynamic response rates
        dy/de       a matrix of tangent moduli (the constitutive equation)
        de/dx       a matrix that converts physical into thermodynamic rates
        dx/dt       a vector of physical control rates

isRuptured()
    E.g., ruptured = ce.isRuptured()
        ruptured    is True if the membrane has ruptured; False otherwise

rupturedRespVec(ctrlVec, respVec)
    E.g., rVec = ce.rupturedRespVec(ctrlVec)
        rVec        returns the response vector after rupture: y after rupture
        ctrlVec     the vector of physical control variables
        respVec     vector of response variables just before membrane rupture

rupturedTanMod(ctrlVec, respVec)
    E.g., dyde = ce.rupturedTanMod(ctrlVec, respVec)
        dyde        returns the matrix of tangent moduli after membrane rupture
        ctrlVec     the vector of physical control variables
        respVec     the vector of response variables after membrane rupture

additional methods

massDensity()
    E.g., rho = ce.massDensity()
        rho         returns the mass density of the septal membrane

temperature()
    E.g., temp = ce.temperature()
        temp        returns the temperature of the septal membrane

stretch()
    E.g., U = ce.stretch()
        U           returns Laplace stretch in co-ordinate frame of membrane

stretchInv()
    E.g., Uinv = ce.stretchInv()
        Uinv        returns the inverse of Laplace stretch

entropyDensity()
    E.g., eta = ce.entropyDensity()
        eta         returns the entropy density of the septal membrane

stress()
    E.g., S = ce.stress()
        S           returns stress in the co-ordinate frame of the membrane

thickness()
    E.g., w = ce.thickness()
        w           returns the width or thickness of the membrane
"""


class controlMembrane(control):

    # variables inherited from the base type: treat these as read-only:
    #   controls    an integer specifying the number of control variables
    #   node        an integer specifying the current node of integration,
    #               which is reset to 0 whenever the integrator is restarted
    #   dt          a floating point number specifying the time-step size
    #   xR          a vector holding control variables for the reference node
    #   xP          a vector holding control variables for the previous node
    #   xC          a vector holding control variables for the current node
    #   xN          a vector holding control variables for the next node
    # control vector arguments have interpretations of:
    #   ctrlVec[0]  membrane temperature      'theta'  (centigrade)
    #   ctrlVec[1]  membrane elongation       'a'      (dimensionless)
    #   ctrlVec[2]  membrane elongation       'b'      (dimensionless)
    #   ctrlVec[3]  membrane shear magnitude  'g'      (dimensionless)

    def __init__(self, ctrlVec0, dt):
        # Call the constructor of the base type.
        super().__init__(ctrlVec0, dt)
        # Verify and initialize other data, as required.
        if self.controls != 4:
            raise RuntimeError("There are 4 control variables for 2D " +
                               "membranes: temperature, two elongation " +
                               "ratios, and a shear.")
        return  # a new instance of type controlMembrane

    def update(self, ctrlVec):
        # Call the base implementation of this method to insert this vector
        # into the class' data structure.
        super().update(ctrlVec)
        return  # nothing

    def advance(self):
        # Call the base implementation of this method to advance its data.
        # This moves current data to their previous fields, and then it moves
        # next data to their current fields.
        super().advance()
        # This method is called internally by the pece integrator and should
        # not be called by the user.
        return  # nothing

    def dedx(self):
        # Call the base implementation of this method to create dedxMtx.
        dedxMtx = super().dedx()
        # Because the matrix created by the super call is an identity matrix,
        # only a few of the cells need to be overwritten
        a = self.xN[1]
        b = self.xN[2]
        dedxMtx[1, 1] = 1.0 / (2.0 * a)
        dedxMtx[1, 2] = 1.0 / (2.0 * b)
        dedxMtx[2, 1] = 1.0 / (2.0 * a)
        dedxMtx[2, 2] = -1.0 / (2.0 * b)
        return dedxMtx

    def dxdt(self, restart=False):
        # Call the base implementation of this method to create dxdtVec.
        dxdtVec = super().dxdt(restart)
        # The returned dxdtVec is computed via finite difference formulae;
        # specifically,
        #   if self.node is 0, 1   use first-order difference formula
        #   if restart is True     use first-order difference formula
        #   otherwise              use second-order backward difference formula
        # This is correct for all but the shear term, which is different.
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

class bioMembrane(response):
    # implements the Freed-Rajagopal model for biologic fibers where
    #   ctrlVec[0]  membrane temperature 'theta' (centigrade)
    #   ctrlVec[1]  membrane stretch 'a'
    #   ctrlVec[2]  membrane stretch 'b'
    #   ctrlVec[3]  membrane shear 'g'
    #   respVec[0]  membrane entropy density 'rho' (erg/g.K)
    #   respVec[1]  membrane surface tension stress 'pi' (barye)
    #   respVec[2]  membrane normal stress difference 'sigma' (barye)
    #   respVec[3]  membrane shear stress 'tau' (barye)

    def __init__(self, ctrlVec0, respVec0, rho, Cp, alpha, M1, M2, e_Mt,
                 e_max, N1, N2, e_Nt, G1, G2, e_Gt, thickness=None):
        # call the base type to verify the inputs and to create variables
        #    self.controls  the number of control variables
        #    self.responses the number of response variables
        super().__init__(ctrlVec0, respVec0)
        # verify inputs for the constructor of the base class
        if self.controls != 4:
            raise RuntimeError("A biologic membrane has 4 control variables.")
        if self.responses != 4:
            raise RuntimeError("A biologic membrane has 4 response variables.")
        if ctrlVec0[0] < 33.0 or ctrlVec0[0] > 41.0:
            raise RuntimeError("The initial temperature must be within the " +
                               "range of 33 to 41 degrees Centigrade.")
        # initialize the remaining data
        self.rho = rho
        self.Cp = Cp
        self.alpha = alpha
        self.M1 = M1
        self.M2 = M2
        self.e_Mt = e_Mt
        self.e_max = e_max
        self.N1 = N1
        self.N2 = N2
        self.e_Nt = e_Nt
        self.G1 = G1
        self.G2 = G2
        self.e_Gt = e_Gt
        if thickness is None or thickness < 0.000025 or thickness > 0.000065:
            self.thickness = mp.septalWidth()
        else:
            self.thickness = thickness
        self.ruptured = False
        return  # a new instance of a constitutive membrane object

    def M_compliance(self, temp, xi, s_pi):
        if s_pi > self.e_max * self.M2:
            self.ruptured = True
        if self.ruptured is True:
            # a small but positive modulus helps to maintain numeric stability
            M = 100.0 * np.finfo(float).eps
            c = 1.0 / M
        elif s_pi <= 0.0:
            # same elastic compliance as at zero stress and zero strain
            c = (self.M1 + self.M2) / (self.M1 * self.M2)
        else:
            # bio membrane model for uniform response
            dT = temp - self.xR[0]
            xi1 = xi - self.alpha * dT - s_pi / (4.0 * self.M2)
            if xi1 < self.e_Mt:
                c = ((self.e_Mt - xi1) /
                     (self.M1 * self.e_Mt + s_pi / 2.0) + 1.0 / self.M2)
            else:
                c = 1.0 / self.M2
        return c

    def N_compliance(self, epsilon, s_sigma):
        # bio membrane model for non-uniform squeeze response
        if self.ruptured is True:
            # a small but positive modulus helps to maintain numeric stability
            N = 100.0 * np.finfo(float).eps
            c = 1.0 / N
        else:
            epsilon1 = epsilon - s_sigma / (2.0 * self.N2)
            if epsilon1 > 0.0:
                if epsilon1 < self.e_Nt:
                    c = ((self.e_Nt - epsilon1) /
                         (self.N1 * self.e_Nt + s_sigma) + 1.0 / self.N2)
                else:
                    c = 1.0 / self.N2
            elif epsilon1 < -0.0:
                if epsilon1 > -self.e_Nt:
                    c = ((-self.e_Nt - epsilon1) /
                         (-self.N1 * self.e_Nt + s_sigma) + 1.0 / self.N2)
                else:
                    c = 1.0 / self.N2
            else:
                c = (self.N1 + self.N2) / (self.N1 * self.N2)
        return c

    def G_compliance(self, gamma, s_tau):
        # bio membrane model for non-uniform shear response
        if self.ruptured is True:
            # a small but positive modulus helps to maintain numeric stability
            G = 100.0 * np.finfo(float).eps
            c = 1.0 / G
        else:
            gamma1 = gamma - s_tau / self.G2
            if gamma1 > 0.0:
                if gamma1 < self.e_Gt:
                    c = ((self.e_Gt - gamma1) /
                         (self.G1 * self.e_Gt + 2.0 * s_tau) + 1.0 / self.G2)
                else:
                    c = 1.0 / self.G2
            elif gamma1 < -0.0:
                if gamma1 > -self.e_Gt:
                    c = ((-self.e_Gt - gamma1) /
                         (-self.G1 * self.e_Gt + 2.0 * s_tau) + 1.0 / self.G2)
                else:
                    c = 1.0 / self.G2
            else:
                c = (self.G1 + self.G2) / (self.G1 * self.G2)
        return c

    def tanMod(self, ctrlVec, respVec):
        # call the base type to verify the inputs and to create matrix ceMtx
        dyde = super().tanMod(ctrlVec, respVec)
        # populate the entries of dyde for the tangent moduli below
        # T0 = self.xR[0]         # is introduced and used in M_compliance
        a0 = self.xR[1]           # initial elongation ratio denoted as a
        b0 = self.xR[2]           # initial elongation ratio denoted as b
        g0 = self.xR[3]           # initial magnitude of shear denoted as g
        TN = ctrlVec[0]           # temperature at next node (centigrade)
        aN = ctrlVec[1]           # elongation ratio a at the next node
        bN = ctrlVec[2]           # elongation ratio b at the next node
        gN = ctrlVec[3]           # magintude of shear g at the next node
        # entropy = respVec[0]    # is not needed in this model
        s_pi = respVec[1]         # uniform areal stress at next node (barye)
        s_sigma = respVec[2]      # non-uniform squeeze stress (barye)
        s_tau = respVec[3]        # non-uniform shear stress (barye)
        xi = m.log(m.sqrt((aN * bN) / (a0 * b0)))       # dilation
        epsilon = m.log(m.sqrt((aN * b0) / (a0 * bN)))  # squeeze
        gamma = gN - g0                                 # shear
        M = 1.0 / self.M_compliance(TN, xi, s_pi)
        N = 1.0 / self.N_compliance(epsilon, s_sigma)
        G = 1.0 / self.G_compliance(gamma, s_tau)
        # compute the tangent modulus
        dyde[0, 0] = (self.Cp / (TN + 273.0) -
                      4.0 * self.alpha**2 * M / self.rho)
        dyde[0, 1] = 4.0 * self.alpha * M / self.rho
        dyde[1, 0] = -4.0 * self.alpha * M
        dyde[1, 1] = 4.0 * M
        dyde[2, 2] = 2.0 * N
        dyde[3, 3] = G
        return dyde

    def isRuptured(self):
        # no super call is required here, ruptured is to have a boolean value
        return self.ruptured

    def rupturedRespVec(self, ctrlVec, respVec):
        # call the base type to verify the input and to create vector rVec
        rVec = super().rupturedRespVec(ctrlVec, respVec)
        # populate the entries for the ruptured response in rVec below
        # this will result in discontinuities in the response fields
        T0 = self.xR[0]           # initial temperature (centigrade)
        a0 = self.xR[1]           # initial elongation ratio denoted as a
        b0 = self.xR[2]           # initial elongation ratio denoted as b
        g0 = self.xR[3]           # initial magnitude of shear denoted as g
        TN = ctrlVec[0]           # temperature at next node (centigrade)
        aN = ctrlVec[1]           # elongation ratio a at the next node
        bN = ctrlVec[2]           # elongation ratio b at the next node
        gN = ctrlVec[3]           # magintude of shear g at the next node
        # compute the strains at rupture
        xi = m.log(m.sqrt((aN * bN) / (a0 * b0)))       # dilation
        epsilon = m.log(m.sqrt((aN * b0) / (a0 * bN)))  # squeeze
        gamma = gN - g0                                 # shear
        # a small but positive modulus helps to maintain numeric stability
        E = 100.0 * np.finfo(float).eps
        # provides for a discontinuity in the response vector
        rVec = np.zeros((4,), dtype=float)
        rVec[0] = (self.yR[0] + 4.0 * self.alpha * E *
                   (xi - self.alpha * (TN - T0)) / self.rho)
        rVec[1] = 4.0 * E * xi
        rVec[2] = 2.0 * E * epsilon
        rVec[3] = E * gamma
        return rVec

    def rupturedTanMod(self, ctrlVec, respVec):
        # No need for a super call in this case, just simply call
        return self.tanMod(ctrlVec, respVec)

    def massDensity(self):
        return self.rho

    def temperature(self):
        return self.xN[0]

    def stretch(self):
        U = np.zeros((2, 2), dtype=float)
        a = self.xN[1]
        b = self.xN[2]
        g = self.xN[3]
        U[0, 0] = a
        U[0, 1] = a * g
        U[1, 1] = b
        return U

    def stretchInv(self):
        Uinv = np.zeros((2, 2), dtype=float)
        a = self.xN[1]
        b = self.xN[2]
        g = self.xN[3]
        Uinv[0, 0] = 1.0 / a
        Uinv[0, 1] = -g / b
        Uinv[1, 1] = 1.0 / b
        return Uinv

    def entropyDensity(self):
        return self.yN[0]

    def stress(self):
        a = self.xN[1]
        b = self.xN[2]
        pi = self.yN[1]
        sigma = self.yN[2]
        tau = self.yN[3]
        s = np.zeros((2, 2), dtype=float)
        s[0, 0] = (pi + sigma) / 2.0
        s[0, 1] = b * tau / a
        s[1, 0] = s[0, 1]
        s[1, 1] = (pi - sigma) / 2.0
        return s

    def thickness(self):
        return self.thickness


"""
Changes made in version "1.0.0":

This is the initial version.
"""
