#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import materialProperties as mp
import math as m
import numpy as np
from peceHE import control, response

"""
Module ceChords.py provides a constitutive description for alveolar chords.

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
__date__ = "09-24-2019"
__update__ = "05-28-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


This module describes septal chords.  From histological studies, septal chords
are found to be comprised of both collagen and elastin fibers that align in
parallel with one another, with minimal chemical bonding between them.
Consequently, they are modeled here as two elastic rods exposed to the same
temperature and strain but carrying different states of entropy and stress.
Their geometric properties, and some of their constitutive parameters, are
described by probability distributions exported from materialProperties.py.
Given these properties, one can create objects describing realistic septal
chords, and thereby, realistic alveoli.  This module exports two classes:
    bioFiber:     provides response of a Freed-Rajagopal biologic fiber, and
    septalChord:  provides response of a septal chord.
Septal chords form a network of fibers that circumscribe the septa that
collectively envelop an alveolar sac.

The CGS system of physical units adopted:
    length          centimeters (cm)
    mass            grams       (g)
    time            seconds     (s)
    temperature     centigrade  (C)
where
    force           dynes       [g.cm/s^2]      1 Newton = 10^5 dyne
    pressure        barye       [dyne/cm^2]     1 Pascal = 10 barye
    energy          erg         [dyne.cm]       1 Joule  = 10^7 ergs


class controlFiber:  It implements and extends class 'control'


For 1D fibers, the control vectors have components:
    ctrlVec[0]  contains fiber temperature (in centigrade)
    ctrlVec[1]  contains fiber length (in cm)

constructor

    E.g.: ctrl = controlFiber(ctrlVec0, dt)
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

methods

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
    the pece object and should not be called by the user.

dedx()
    E.g., dedxMtx = ctrl.dedx()
        dedxMtx     a matrix containing the mapping of physical control rates
                    into their thermodynamic control rates.
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


class bioFiber:  It implements and extends class 'response'.


Creates objects that implement the Freed-Rajagopal biologic fiber model.
For this model
    ctrlVec[0]  contains fiber temperature (in centigrade)
    ctrlVec[1]  contains fiber length (in cm)
and
    respVec[0]  contains fiber entropy density (in erg/g.K)
    respVec[1]  contains fiber stress (in dyne/cm^2 or barye)

constructor

    E.g.: ce = bioFiber(ctrlVec0, respVec0, rho, Cp, alpha, E1, E2, e_t, e_max)
        ctrlVec0    contains the control variables in the reference state
        respVec0    contains the initial response variables: initial conditions
        rho         mass density for the fiber
        Cp          specific heat at constant pressure for the fiber
        alpha       coefficient of thermal expansion for the fiber
        E1          compliant modulus, i.e., at zero stress and zero strain
        E2          stiff modulus, i.e., elastic modulus at terminal strain
        e_t         transition strain between compliant and stiff behaviors
        e_max       sets rupture, where s_r = e_max * E2 is the rupture stress

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    responses       an integer specifying the number of response variables
    xR              a vector containing the initial condition for control
    yR              a vector containing the initial condition for response

methods

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
        ruptured    is True if the fiber has ruptured; False otherwise

rupturedRespVec(ctrlVec, respVec)
    E.g., rVec = ce.rupturedRespVec(ctrlVec)
        rVec        the response vector after rupture, viz., y after rupture
        ctrlVec     the vector of physical control variables
        respVec     the vector of response variables just before fiber rupture

rupturedTanMod(ctrlVec, respVec)
    E.g., dyde = ce.rupturedTanMod(ctrlVec, respVec)
        dyde        the matrix of tangent moduli after fiber rupture
        ctrlVec     the vector of physical control variables
        respVec     the vector of response variables after fiber rupture


class septalChord, which also implements and extends class 'response'


Creates objects that implement the Freed-Rajagopal thermoelastic fiber of class
bioFiber for septal chords that are made up of collagen and elastin fibers.
For this model
    ctrlVec[0]  contains fiber temperature (in centigrade)
    ctrlVec[1]  contains fiber length (in cm)
and
    respVec[0]  contains collagen fiber entropy density (in erg/g.K)
    respVec[1]  contains collagen fiber stress (in dyne/cm^2 or barye)
    respVec[2]  contains elastin fiber entropy density (in erg/g.K)
    respVec[3]  contains elastin fiber stress (in dyne/cm^2 or barye)

constructor

    E.g.: ce = ceFiber(ctrlVec0, respVec0, diaCollagen=None, diaElastin=None)
        ctrlVec         the vector of physical control variables, i.e., x
        respVec         the vector of response variables, viz, y
        diaCollagen     the reference diameter of the collagen fiber
        diaElastin      the reference diameter of the elastin fiber
    If the diameters take on their default values of None, then they are
    assigned via their respective probability distributions.

variables: treat these as read-only

    controls        an integer specifying the number of control variables
    responses       an integer specifying the number of response variables
    xR              a vector containing the initial condition for control
    yR              a vector containing the initial condition for response

methods

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
        ruptured    is True if a fiber has ruptured; False otherwise

rupturedRespVec(ctrlVec, respVec)
    E.g., rVec = ce.rupturedRespVec(ctrlVec)
        rVec        the response vector after rupture, viz., y after rupture
        ctrlVec     the vector of physical control variables
        respVec     the vector of response variables just before fiber rupture

rupturedTanMod(ctrlVec, respVec)
    E.g., dyde = ce.rupturedTanMod(ctrlVec, respVec)
        dyde        the matrix of tangent moduli after fiber rupture
        ctrlVec     the vector of physical control variables
        respVec     the vector of response variables after fiber rupture

additional methods

bioFiberCollagen()
    E.g., fiber_c = ce.bioFiberCollagen()
        fiber_c     an instance of bioFiber representing a collagen fiber

bioFiberElastin()
    E.g., fiber_e = ce.bioFiberElastin()
        fiber_e     an instance of bioFiber representing an elastin fiber

areaCollagen()
    E.g., a_c = ce.areaCollagen()
        a_c         the current cross-sectional area of the collagen fiber

areaElastin()
    E.g., a_e = ce.areaElastin()
        a_e         the current cross-sectional area of the elastin fiber

massDensity()
    E.g., rho = ce.massDensity()
        rho         the mass density of the septal chord

volume()
    E.g., vol = ce.volume()
        vol         the volume of the septal chord

temperature()
    E.g., temp = ce.temperature()
        temp        the temperature in degrees Centigrade

strain()
    E.g., e = ce.strain()
        e           the natural or logarithmic strain

force()
    E.g., f = ce.force()
        f           the total force carried by the septal chord

entropy()
    E.g., s = ce.entropy()
        s           the total entropy of the septal chord (not entropy density)


Reference:
    Freed, A. D. and Rajagopal, K. R., “A Promising Approach for Modeling
    Biological Fibers,” ACTA Mechanica, 227 (2016), 1609-1619.
    DOI: 10.1007/s00707-016-1583-8.  Errata: DOI: 10.1007/s00707-018-2183-6
"""


class controlFiber(control):

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
    #   ctrlVec[0]  contains the fiber temperature (in centigrade)
    #   ctrlVec[1]  contains the fiber length (in cm)

    def __init__(self, ctrlVec0, dt):
        # Call the constructor of the base type.
        super().__init__(ctrlVec0, dt)
        # Verify and initialize other data, as required.
        if self.controls != 2:
            raise RuntimeError("There are only 2 control variables for 1D " +
                               "fibers: temperature and length.")
        return  # a new instance of type control1D

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
        # Because the matrix created by the super call is an identity matrix
        dedxMtx[1, 1] = 1.0 / self.xN[1]
        return dedxMtx

    def dxdt(self, restart=False):
        # Call the base implementation of this method to create dxdtVec.
        dxdtVec = super().dxdt(restart)
        # The returned dxdtVec is computed via finite difference formulae;
        # specifically,
        #   if self.node is 0, 1   use first-order difference formula
        #   if restart is True     use first-order difference formula
        #   otherwise              use second-order backward difference formula
        return dxdtVec


# constitutive class for biologic fibers


class bioFiber(response):
    # implements the Freed-Rajagopal model for biologic fibers where
    #   ctrlVec[0]  contains fiber temperature (in centigrade)
    #   ctrlVec[1]  contains fiber length (in cm)
    #   respVec[0]  contains fiber entropy density (in erg/g.K)
    #   respVec[1]  contains fiber stress (in dyne/cm^2 or barye)

    def __init__(self, ctrlVec0, respVec0, rho, Cp, alpha, E1, E2, e_t, e_max):
        # call the base type to verify the inputs and to create variables
        #    self.controls  the number of control variables
        #    self.responses the number of response variables
        super().__init__(ctrlVec0, respVec0)
        # verify inputs for the constructor of the base class
        if self.controls != 2:
            raise RuntimeError("A biologic fiber has two control variables.")
        if self.responses != 2:
            raise RuntimeError("A biologic fiber has two response variables.")
        if ctrlVec0[0] < 33.0 or ctrlVec0[0] > 41.0:
            raise RuntimeError("The initial temperature must be within the " +
                               "range of 33 to 41 degrees Centigrade.")
        if ctrlVec0[1] < 0.0000001:
            raise RuntimeError('The initial fiber length must be greater ' +
                               'than a nanometer.')
        self.ctrlVec0 = np.zeros((self.controls,), dtype=float)
        self.ctrlVec0[:] = ctrlVec0[:]
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
            raise RuntimeError("Fiber thermal expansion coefficient " +
                               "must be positive.")
        if E1 > np.finfo(float).eps:
            self.E1 = E1
        else:
            raise RuntimeError('Initial fiber modulus E1 must be positive.')
        if E2 > self.E1:
            self.E2 = E2
        else:
            raise RuntimeError('Terminal fiber modulus E2 must be greater ' +
                               'than E1.')
        if e_t > np.finfo(float).eps:
            self.e_t = e_t
        else:
            raise RuntimeError('Limiting fiber strain e_t must be positive.')
        if e_max > np.finfo(float).eps:
            self.e_max = e_max
        else:
            raise RuntimeError('Maximum fiber strain e_max must be positive.')
        self.ruptured = False
        return  # a new instance of a fiber object

    def _compliance(self, deltaTemp, strain, stress):
        if stress > self.e_max * self.E2:
            self.ruptured = True
        if self.ruptured is True:
            # a small but positive modulus helps to maintain numeric stability
            E = 100.0 * np.finfo(float).eps
            c = 1.0 / E
        elif stress <= 0.0:
            # same elastic compliance as at zero stress and zero strain
            c = (self.E1 + self.E2) / (self.E1 * self.E2)
        else:
            # Freed-Rajagopal elastic fiber model
            e1 = strain - self.alpha * deltaTemp - stress / self.E2
            if e1 < self.e_t:
                c = ((self.e_t - e1) / (self.E1 * self.e_t + 2.0 * stress) +
                     1.0 / self.E2)
            else:
                c = 1.0 / self.E2
        return c

    def tanMod(self, ctrlVec, respVec):
        # call the base type to verify the inputs and to create matrix ceMtx
        dyde = super().tanMod(ctrlVec, respVec)
        # populate the entries of dyde for the tangent moduli below
        temp0 = self.ctrlVec0[0]    # initial temperature (in centigrade)
        len0 = self.ctrlVec0[1]     # initial length (in cm)
        tempN = ctrlVec[0]          # temperature at next node (in centigrade)
        lenN = ctrlVec[1]           # length at the next node (in cm)
        # entropy = respVec[0]      # is not needed in this model
        stress = respVec[1]         # stress at the next node (in barye)
        deltaTemp = tempN - temp0
        strain = m.log(lenN / len0)
        E = 1.0 / self._compliance(deltaTemp, strain, stress)
        # compute the tangent modulus
        dyde[0, 0] = self.Cp / (tempN + 273.0) - self.alpha**2 * E / self.rho
        dyde[0, 1] = self.alpha * E / self.rho
        dyde[1, 0] = -self.alpha * E
        dyde[1, 1] = E
        return dyde

    def isRuptured(self):
        # no super call is required here, ruptured is to have a boolean value
        return self.ruptured

    def rupturedRespVec(self, ctrlVec, respVec):
        # call the base type to verify the input and to create vector rVec
        rVec = super().rupturedRespVec(ctrlVec, respVec)
        # populate the entries for the ruptured response in rVec below
        # this will result in discontinuities in the response fields
        temp0 = self.xR[0]      # initial temperature (in centigrade)
        len0 = self.xR[1]       # initial length (in cm)
        tempN = ctrlVec[0]      # temperature at the next node (in centigrade)
        lenN = ctrlVec[1]       # length at the next node (in cm)
        # entropy = respVec[0]  # is not needed in this model
        # stress = respVec[1]   # is not needed in this model
        deltaTemp = tempN - temp0
        strain = m.log(lenN / len0)
        # a small but positive modulus helps to maintain numeric stability
        E = 100.0 * np.finfo(float).eps
        # provides for a discontinuity in the response vector
        rVec = np.zeros((2,), dtype=float)
        rVec[0] = (self.yR[0] + self.alpha * E *
                   (strain - self.alpha * deltaTemp) / self.rho)
        rVec[1] = E * strain
        return rVec

    def rupturedTanMod(self, ctrlVec, respVec):
        # call the base type to verify the inputs and to create matrix ceMtx
        dyde = super().rupturedTanMod(ctrlVec, respVec)
        # populate the entries of ceMtx for the ruptured tangent moduli below
        # temp0 = self.xR[0]    # is not needed in this model
        # len0 = self.xR[1]     # is not needed in this model
        tempN = ctrlVec[0]      # temperature at the next node (in centigrade)
        # lenN = ctrlVec[1]       # length at the next node (in cm)
        # entropy = respVec[0]  # is not needed in this model
        # stress = respVec[1]   # is not needed in this model
        # a small but positive modulus helps to maintain numeric stability
        E = 100.0 * np.finfo(float).eps
        # compute the tangent modulus is
        dyde = np.zeros((2, 2), dtype=float)
        dyde[0, 0] = self.C / (tempN + 273.0) - self.alpha**2 * E / self.rho
        dyde[0, 1] = self.alpha * E / self.rho
        dyde[1, 0] = -self.alpha * E
        dyde[1, 1] = E
        return dyde


# constitutive class for alveolar chords


class septalChord(response):
    # implements the Freed-Rajagopal model for septal chords, which are
    # comprised of collagen and elastin fibers loaded in parallel, where:
    #   ctrlVec[0]  contains fiber temperature (in centigrade)
    #   ctrlVec[1]  contains fiber length (in cm)
    #   respVec[0]  contains collagen fiber entropy density (in erg/g.K)
    #   respVec[1]  contains collagen fiber stress (in dyne/cm^2 or barye)
    #   respVec[0]  contains elastin fiber entropy density (in erg/g.K)
    #   respVec[1]  contains elastin fiber stress (in dyne/cm^2 or barye)

    def __init__(self, ctrlVec0, respVec0, diaCollagen=None, diaElastin=None):
        # call the base type to verify the inputs and to create variables
        super().__init__(ctrlVec0, respVec0)
        # verify inputs for the constructor of the base class
        if self.controls != 2:
            raise RuntimeError("A septal chord has two control variables.")
        if self.responses != 4:
            raise RuntimeError("A septal chord has four response variables.")
        if ctrlVec0[0] < 33.0 or ctrlVec0[0] > 41.0:
            raise RuntimeError("The initial temperature must be within the " +
                               "range of 33 to 41 degrees Centigrade.")
        if ctrlVec0[1] < 0.001 or ctrlVec0[1] > 0.02:
            raise RuntimeError('The initial fiber length must be within ' +
                               'the range of 10 to 200 microns.')
        # verify and initialize the remaining data
        if diaCollagen is None:
            diaCollagen = mp.fiberDiameterCollagen()
        elif diaCollagen < 0.000005 or diaCollagen > 0.0005:
            raise RuntimeError("Diameter of the collagen fiber must be " +
                               "within the range of 0.05 to 5 microns.")
        else:
            pass
        if diaElastin is None:
            diaElastin = mp.fiberDiameterElastin()
        elif diaElastin < 0.000005 or diaElastin > 0.0005:
            raise RuntimeError("Diameter of the elastin fiber must be " +
                               "within the range of 0.05 to 5 microns.")
        else:
            pass
        self.A0_c = m.pi * diaCollagen**2 / 4.0
        self.A0_e = m.pi * diaElastin**2 / 4.0
        self.temperature = ctrlVec0[0]
        self.length = ctrlVec0[1]
        self.len0 = ctrlVec0[1]
        # create constitutive objects for the collagen and elastin fibers
        # elastic properties are assigned via their probability distributions
        collagenRespVec0 = np.zeros((2,), dtype=float)
        collagenRespVec0[0] = respVec0[0]
        collagenRespVec0[1] = respVec0[1]
        rho_c = mp.rhoCollagen()
        C_c = mp.CpCollagen()
        alpha_c = mp.alphaCollagen()
        E1_c, E2_c, et_c, emax_c = mp.collagenFiber()
        self.fiberC = bioFiber(ctrlVec0, collagenRespVec0,
                               rho_c, C_c, alpha_c, E1_c, E2_c, et_c, emax_c)
        elastinRespVec0 = np.zeros((2,), dtype=float)
        elastinRespVec0[0] = respVec0[2]
        elastinRespVec0[1] = respVec0[3]
        rho_e = mp.rhoElastin()
        C_e = mp.CpElastin()
        alpha_e = mp.alphaElastin()
        E1_e, E2_e, et_e, emax_e = mp.elastinFiber()
        self.fiberE = bioFiber(ctrlVec0, elastinRespVec0,
                               rho_e, C_e, alpha_e, E1_e, E2_e, et_e, emax_e)
        # provide initial conditions for the various responses
        self.etaC = respVec0[0]
        self.stressC = respVec0[1]
        self.etaE = respVec0[2]
        self.stressE = respVec0[3]
        return  # a new instance of type ceChord

    def tanMod(self, ctrlVec, respVec):
        # call the base type to verify the inputs and to create matrix ceMtx
        ceMtx = super().tanMod(ctrlVec, respVec)
        # extract the controlled variables for export
        self.temperature = ctrlVec[0]
        self.length = ctrlVec[1]
        # extract the response variables for export
        self.etaC = respVec[0]
        self.stressC = respVec[1]
        self.etaE = respVec[2]
        self.stressE = respVec[3]
        # assemble the tangent moduli
        fiberResp = np.zeros((2,), dtype=float)
        for i in range(2):
            fiberResp[i] = respVec[i]
        ceMtxC = self.fiberC.tanMod(ctrlVec, fiberResp)
        for i in range(2):
            fiberResp[i] = respVec[2 + i]
        ceMtxE = self.fiberE.tanMod(ctrlVec, fiberResp)
        for i in range(2):
            ceMtx[i, :] = ceMtxC[i, :]
            ceMtx[2 + i, :] = ceMtxE[i, :]
        return ceMtx

    def isRuptured(self):
        # no super call is required here
        if (self.fiberC.isRuptured() is True or
           self.fiberE.isRuptured() is True):
            return True
        else:
            return False

    def rupturedRespVec(self, ctrlVec, respVec):
        # call the base type to verify the input and to create vector rVec
        rVec = super().rupturedRespVec(ctrlVec, respVec)
        # populate the entries for the ruptured response in rVec below
        # this will result in discontinuities in the response fields
        # extract the controlled variables for export
        self.temperature = ctrlVec[0]
        self.length = ctrlVec[1]
        # assemble the response vector
        fiberResp = np.zeros((2,), dtype=float)
        fiberResp[0] = respVec[0]
        fiberResp[1] = respVec[1]
        if self.fiberC.isRuptured() is True:
            respVecC = self.fiberC.rupturedRespVec(ctrlVec, fiberResp)
        else:
            respVecC = np.zeros((2,), dtype=float)
            respVecC[:] = fiberResp[:]
        fiberResp[0] = respVec[2]
        fiberResp[1] = respVec[3]
        if self.fiberE.isRuptured() is True:
            respVecE = self.fiberE.rupturedRespVec(ctrlVec, fiberResp)
        else:
            respVecE = np.zeros((2,), dtype=float)
            respVecE[:] = fiberResp[:]
        for i in range(2):
            rVec[i] = respVecC[i]
            rVec[i + 2] = respVecE[i]
        # extract the response variables for export
        self.etaC = respVecC[0]
        self.stressC = respVecC[1]
        self.etaE = respVecE[0]
        self.stressE = respVecE[1]
        return rVec

    def rupturedTanMod(self, ctrlVec, respVec):
        # call the base type to verify the inputs and to create matrix ceMtx
        ceMtx = super().rupturedTanMod(ctrlVec, respVec)
        # extract the controlled variables for export
        self.temperature = ctrlVec[0]
        self.length = ctrlVec[1]
        # extract the response variables for export
        self.etaC = respVec[0]
        self.stressC = respVec[1]
        self.etaE = respVec[2]
        self.stressE = respVec[3]
        # assemble the ruptured tangent moduli
        fiberResp = np.zeros((2,), dtype=float)
        fiberResp[0] = respVec[0]
        fiberResp[1] = respVec[1]
        if self.fiberC.isRuptured() is True:
            tanModC = self.fiberC.rupturedTanMod(ctrlVec, fiberResp)
        else:
            tanModC = self.fiberC.tanMod(ctrlVec, fiberResp)
        fiberResp[0] = respVec[2]
        fiberResp[1] = respVec[3]
        if self.fiberE.isRuptured() is True:
            tanModE = self.fiberE.rupturedTanMod(ctrlVec, fiberResp)
        else:
            tanModE = self.fiberE.tanMod(ctrlVec, fiberResp)
        for i in range(2):
            ceMtx[i, :] = tanModC[i, :]
            ceMtx[i + 2, :] = tanModE[i, :]
        return ceMtx

    def bioFiberCollagen(self):
        return self.fiber_c

    def bioFiberElastin(self):
        return self.fiber_e

    def areaCollagen(self):
        a = self.A0_c * self.len0 / self.length
        return a

    def areaElastin(self):
        a = self.A0_e * self.len0 / self.length
        return a

    def massDensity(self):
        mass_c = mp.rhoCollagen() * self.A0_c * self.len0
        mass_e = mp.rhoElastin() * self.A0_e * self.len0
        massDensity = ((mass_c + mass_e) / self.volume())
        return massDensity  # in g/cm^3

    def volume(self):
        vol = (self.A0_c + self.A0_e) * self.len0
        return vol

    def temperature(self):
        return self.temperature

    def strain(self):
        return m.log(self.length / self.len0)

    def force(self):
        f = (self.stressC * self.areaCollagen() +
             self.stressE * self.areaElastin())
        return f

    def entropy(self):
        s = (self.etaC * mp.rhoCollagen() * self.A0_c * self.len0 +
             self.etaE * mp.rhoElastin() * self.A0_e * self.len0)
        return s


"""
Changes made in version "1.0.0":

This is the initial version.
"""
