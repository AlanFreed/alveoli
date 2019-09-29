#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math as m
import numpy as np
from peceVtoX import pece
import random
import splines

"""
Module ceChords.py provides a constitutive equation for alveolar chords.

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
__version__ = "1.3.0"
__date__ = "09-24-2019"
__update__ = "09-24-2019"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
This module describes septal chords.  From histological studies, these septal
chords are found to be comprised of both collagen and elastin fibers that
align in parallel with one another with minimal chemical bonding between them.
Consequently, they are modeled here as two elastic rods exposed to the same
strain but carrying different states of stress.  Statistics for their geometric
properties have been determined by Sobin, Fung and Tremer, Journal of Applied
Physiology, Vol. 64, 1988.  Three procedures can be called to get random
variables that describe these geometric features.  Given these geometric
properties, one can then create an object for describing a septal chord.
This module exports the class ceChord that evaluates the constitutive response
for these chordal fibers that circumscribe the septa within an alveolar sac.

Several methods have a string argument that is denoted as 'state', which can
take on any of the following values:
    'c', 'curr', 'current'       gets the value for  a  current configuration
    'n', 'next'                  gets the value for  a  next configuration
    'p', 'prev', 'previous'      gets the value for  a  previous configuration
    'r', 'ref', 'reference'      gets the value for the reference configuration
Any method that accepts a 'state' as an argument can only be called after the
solution has been advanced, and before its next call to update for integration.

CGS units are used here
    length          centimeters
    mass            grams
    time            seconds
    temperature     centigrade
    force           dynes   [gr.cm/s^2]     1 Newton = 10^5 dyne
    pressure        barye   [dyne/cm^2]     1 Pascal = 10 barye
    energy          erg     [dyne.cm]       1 Joule  = 10^7 ergs

procedures

The following proceedures provide random variables for the geometric features
of an alveolar chord, viz.: its length, the diameter of its collegan fiber,
and the diameter of its elastin fiber.  These statistics came from reference:
Sobin, Fung and Tremer, Journal of Applied Physiology, Vol. 64, 1988.

    l = randomLength()
        Returns a random length 'l' for a septal fiber in centimeters.
        Associates with humans, age 15-35, whose lungs were fixed at 4 cm H20.
        (data are from: Sobin, Fung and Tremer, 1988)

    d = randomDiameterCollagen()
        Returns random diameter 'd' for a collegen septal fiber in centimeters.
        Associates with humans, age 15-35, whose lungs were fixed at 4 cm H20.
        (data are from: Sobin, Fung and Tremer, 1988)

    d = randomDiameterElastin()
        Returns random diameter 'd' for an elastin septal fiber in centimeters.
        Associates with humans, age 15-35, whose lungs were fixed at 4 cm H2O.
        (data are from: Sobin, Fung and Tremer, 1988)


class ceFiber
    Objects of this class are called internally.

constructor

    ce = ceFiber(E1, E2, e_t)
            E1      the compliant modulus, i.e., at zero stress and zero strain
            E2      the stiff modulus, i.e., the terminal elastic modulus
            e_t     the transition strain between compliant and stiff response
        Creates an object that is the Freed-Rajagopal elastic fiber model.

methods

    C = ce.compliance(strain, stress)
        Returns the tangent compliance at specified state of stress and strain.

    E = ce.modulus(strain, stress)
        Returns the tangent modulus at specified state of stress and strain.


class ceChord


constructor

    c = ceChord(diaC, diaE, lenF)
            diaC    reference diameter in centimeters for the collagen fiber
            diaE    reference diameter in centimeters for the elastin fiber
            lenF    reference  length  in centimeters for these two fibers
        Creates an object used for solving stress and entropy in septal chords.
        These chords are modeled using the Freed-Rajagopal elastic fiber model.

        The thermal-physical properties of this model include:
            rho     mass density                                   [gr/cm^3]
            eta0    entropy at atmospheric pressure                [erg/cm^3.K]
            alpha   coefficient of thermal expansion               [1/C]
        which is defined according to
            alpha = (1/L) dL/dT
        while the fiber model has additional parameters that include:
            e_t     transition strain
            E_1     tangent modulus at zero stress and strain      [barye]
            E_2     tangent modulus at terminal stress and strain  [barye]

        The constructor solves the constitutive equation over the maximum range
        of response and then creates a cubic spline for efficient interpolation
        of this response by the various methods of the object.  Integration is
        done using a two-step PECE method from peceVtoX.

methods

    Constitutive Solver

    c.update(lenF)
            lenF    length of the chord in centimeters at the next step
        Updates state 'next' given the new fiber length 'lenF'.

    c.advance()
        Advances a solution along its path to the next step.  It does this by
        copying current values into previous values, and then next values into
        current values.

    Chordal Properties of an Alveolus
        Values are per alveolus, i.e., chordal value / 3, because there are
        three alveoli sharing in these properties per alveolar chord.

    m = c.chordalMass()
        Returns the mass of this septal chord in: grams.

    f = c.chordalForce(state)
        Returns the force carried by the septal chord in configuration 'state'.
        This is the total force carried by the chord.  It is three times the
        force associated with the alveolus in question, as an alveolar chord
        is shared by three alveoli.  Units are in: dynes.

    eta = c.chordalEntropy(state)
        Returns the entropy of this septal chord in configuration 'state'.
        Units are in: erg/K.

    Fiber Properties
        Values are per fiber.  These values are not averaged over 3 alveoli.

    epsilon = c.chordalStrain(state)
        Returns the true strain of this chord in configuration 'state'.

    f = c.collagenForce(state)
        Returns the force carried by the collagen fiber in this septal chord
        in configuration 'state'.  Units are in: dynes.

    f = c.elastinForce(state)
        Returns the force carried by the elastin fiber in this septal chord
        in configureation 'state'.  Units are in: dynes.

    stress = c.collagenStress(state)
        Returns the true stress carried by the collagen fiber in this septal
        chord in configuration 'state'.  Units are in: barye.

    stress = c.elastinStress(state)
        Returns the true stress carried by the elastin fiber in this septal
        chord in configuration 'state'.  Units are in: barye.

    eta = c.collagenEntropy(state)
        Returns the entropy of the collagen fiber from this septal chord in
        configuration 'state'.  Units are in: erg/K.

    eta = c.elastinEntropy(state)
        Returns the entropy of the elastin fiber from this septal chord in
        configuration 'state'.  Units are in: erg/K.
"""

# procedures for chords


def randomChordLength():
    # fiber length was found to distribute normally
    mu = 155.5      # mean length in microns
    sigma = 62.8    # standard deviation
    length = random.gauss(mu, sigma)
    # convert microns to centimeters
    length = length / 10000.0
    # length must be greater than 10 microns
    if length < 0.0001:
        length = randomChordLength()
    return length  # in centimeters


def randomDiameterCollagen():
    # square root of fiber diameter was found to distribute normally
    mu = 0.952      # mean of the square root of fiber diameter in microns
    sigma = 0.242   # standard deviation
    sqrtDia = random.gauss(mu, sigma)
    dia = sqrtDia**2
    # convert microns to centimeters
    dia = dia / 10000.0
    # diameter must be greater than 100 nanometers
    if dia < 0.000001:
        dia = randomDiameterCollagen()
    return dia  # in centimeters


def randomDiameterElastin():
    # square root of fiber diameter was found to distribute normally
    mu = 0.957      # mean of the square root of fiber diameter in microns
    sigma = 0.239   # standard deviation
    sqrtDia = random.gauss(mu, sigma)
    dia = sqrtDia**2
    # convert microns to centimeters
    dia = dia / 10000.0
    # diameter must be greater than 100 nanometers
    if dia < 0.000001:
        dia = randomDiameterElastin()
    return dia  # in centimeters


# constitutive model of Freed & Rajagopal for biologic fibers


class ceFiber(object):

    def __init__(self, E1, E2, e_t):
        if float(E1) > 0.0:
            self.E1 = float(E1)
        else:
            raise RuntimeError('Error: modulus E1 must be positive.')
        if float(E2) > self.E1:
            self.E2 = float(E2)
        else:
            raise RuntimeError('Error: modulus E2 must be greater than E1.')
        if float(e_t) > 0.0:
            self.e_t = float(e_t)
        else:
            raise RuntimeError('Error: limiting strain e_t must be positive.')

    def compliance(self, strain, stress):
        if stress <= 0.0:
            c = (self.E1 + self.E2) / (self.E1 * self.E2)
        else:
            # no thermal strain effect considered in this application
            c = ((self.e_t + stress / self.E2 - strain) /
                 (self.E1 * self.e_t + 2.0 * stress) + 1.0 / self.E2)
        return c

    def modulus(self, strain, stress):
        m = 1.0 / self.compliance(strain, stress)
        return m


# constitutive class for chords


class ceChord(object):

    # constructor

    def __init__(self, diaC, diaE, lenF):
        # The diameters for collagen and elastin fibers and their length in an
        # unloaded reference state.  Their dimensions are to be in centimeters.

        self.committed = True

        # verify the inputs

        if diaC > 0.0:
            self.A0_c = m.pi * diaC**2 / 4.0
        else:
            raise RuntimeError(
                      'Initial collagen fiber diameter diaC must be positive.')
        if diaE > 0.0:
            self.A0_e = m.pi * diaE**2 / 4.0
        else:
            raise RuntimeError(
                       'Initial elastin fiber diameter diaE must be positive.')
        if lenF > 0.0:
            self.len_0 = lenF
        else:
            raise RuntimeError('Initial fiber length lenF must be positive.')

        # physical properties of the collagen and elastin fibers in a chord

        # material properties: collagen fibers
        self.alpha_c = 1.8E-4   # thermal expansion             [1/C]
        self.eta_c_0 = 3.7E4    # entropy                       [erg/gr.K]
        self.et_c = 0.09        # transition strain             [cm/cm]
        self.E1_c = 5.0E5       # initial tangent modulus       [barye]
        self.E2_c = 3.0E7       # terminal tangent modulus      [barye]
        self.rho_c = 1.34       # mass density                  [gr/cm^3]

        # material properties: elastin fibers
        self.alpha_e = 3.2E-4   # thermal expansion             [1/C]
        self.eta_e_0 = 3.4E4    # entropy                       [erg/gr.K]
        self.et_e = 0.4         # transition strain             [cm/cm]
        self.E1_e = 2.3E6       # initial tangent modulus       [barye]
        self.E2_e = 1.0E7       # terminal tangent modulus      [barye]
        self.rho_e = 1.31       # mass density                  [gr/cm^3]

        # create constitutive equations for the collagen and elastin fibers

        self.fiber_c = ceFiber(self.E1_c, self.E2_c, self.et_c)
        self.fiber_e = ceFiber(self.E1_e, self.E2_e, self.et_e)

        # maximum strains associate with stresses that are 0.1 * E2
        self.strainMax_c = self.et_c + 0.1
        self.strainMax_e = self.et_e + 0.1

        # arrays to hold stress-strain response data over range +/- strainMax
        nodalPts = 101  # must be odd
        self.strain_c = np.zeros(nodalPts, dtype=float)
        self.strain_e = np.zeros(nodalPts, dtype=float)
        self.stress_c = np.zeros(nodalPts, dtype=float)
        self.stress_e = np.zeros(nodalPts, dtype=float)

        # fiber stress-strain responses
        dStrain_c = self.strainMax_c / (nodalPts // 2)
        dStrain_e = self.strainMax_e / (nodalPts // 2)
        strain0 = 0.0
        stress0 = 0.0
        # stress-strain response in compression is compliant Hookean
        E_c = self.fiber_c.modulus(strain0, stress0)
        E_e = self.fiber_e.modulus(strain0, stress0)
        for i in range(nodalPts//2-1, -1, -1):
            self.strain_c[i] = self.strain_c[i+1] - dStrain_c
            self.stress_c[i] = E_c * self.strain_c[i]
            self.strain_e[i] = self.strain_e[i+1] - dStrain_e
            self.stress_e[i] = E_e * self.strain_e[i]
        # stress-strain response in tension is Freed-Rajagopal
        pece_c = pece(self.fiber_c.modulus, strain0, stress0, dStrain_c)
        pece_e = pece(self.fiber_e.modulus, strain0, stress0, dStrain_e)
        for i in range(nodalPts//2+1, nodalPts):
            self.strain_c[i] = self.strain_c[i-1] + dStrain_c
            pece_c.integrate()
            pece_c.advance()
            self.stress_c[i] = pece_c.getX()
            self.strain_e[i] = self.strain_e[i-1] + dStrain_e
            pece_e.integrate()
            pece_e.advance()
            self.stress_e[i] = pece_e.getX()

        # fit stress-strain data with a cubic spline for stress evaluations
        self.a_c, self.b_c, self.c_c, self.d_c = splines.getCoef(self.strain_c,
                                                                 self.stress_c)
        self.a_e, self.b_e, self.c_e, self.d_e = splines.getCoef(self.strain_e,
                                                                 self.stress_e)

        # the fields needed to supply the requested chordal properties
        self.len_prev = self.len_0
        self.len_curr = self.len_0
        self.len_next = self.len_0

        self.entropy_c_0 = self.rho_c * self.len_0 * self.A0_c * self.eta_c_0
        self.eta_c_prev = self.entropy_c_0
        self.eta_c_curr = self.entropy_c_0
        self.eta_c_next = self.entropy_c_0

        self.entropy_e_0 = self.rho_e * self.len_0 * self.A0_e * self.eta_e_0
        self.eta_e_prev = self.entropy_e_0
        self.eta_e_curr = self.entropy_e_0
        self.eta_e_next = self.entropy_e_0

        self.stress_c_prev = 0.0
        self.stress_c_curr = 0.0
        self.stress_c_next = 0.0

        self.stress_e_prev = 0.0
        self.stress_e_curr = 0.0
        self.stress_e_next = 0.0

        return  # a new instance of type chord

    # methods pertaining to the constitutive solver

    def update(self, lenF):
        self.committed = False
        self.len_next = lenF
        stretch = self.len_next / self.len_0
        strain = m.log(stretch)
        self.stress_c_next = splines.Y(self.a_c, self.b_c, self.c_c, self.d_c,
                                       self.strain_c, strain)
        self.stress_e_next = splines.Y(self.a_e, self.b_e, self.c_e, self.d_e,
                                       self.strain_e, strain)
        force_c = self.stress_c_next * self.A0_c / stretch
        volume_c = self.A0_c * self.len_0
        self.eta_c_next = self.entropy_c_0 + self.alpha_c * force_c / volume_c
        force_e = self.stress_e_next * self.A0_e / stretch
        volume_e = self.A0_e * self.len_0
        self.eta_e_next = self.entropy_e_0 + self.alpha_e * force_e / volume_e

        return  # nothing

    def advance(self):
        self.committed = True

        # map current fields into previous fields
        self.len_prev = self.len_curr
        self.eta_c_prev = self.eta_c_curr
        self.eta_e_prev = self.eta_e_curr
        self.stress_c_prev = self.stress_c_curr
        self.stress_e_prev = self.stress_e_curr

        # map next fields into current fields
        self.len_curr = self.len_next
        self.eta_c_curr = self.eta_c_next
        self.eta_e_curr = self.eta_e_next
        self.stress_c_curr = self.stress_c_next
        self.stress_e_curr = self.stress_e_next

        return  # nothing

    # chordal properties

    def chordalMass(self):
        # mass of the septal chord
        mass_c = self.A0_c * self.len_0 * self.rho_c
        mass_e = self.A0_e * self.len_0 * self.rho_e
        # divide total chordal mass by 3 to get chordal mass for the alveolus
        mass = (mass_c + mass_e) / 3.0
        return mass  # in gr

    def chordalForce(self, state):
        # divide total chordal force by 3 to get chordal force for the alveolus
        f = (self.collagenForce(state) + self.elastinForce(state)) / 3.0
        return f  # in dynes

    def chordalEntropy(self, state):
        # divide total chordal entropy by 3 to get chordal entropy for alveolus
        entropy = (self.collagenEntropy(state) +
                   self.elastinEntropy(state)) / 3.0
        return entropy  # in erg/cm^3.K

    # collagen and elastin fiber properties

    def chordalStrain(self, state):
        if not self.committed:
            raise RuntimeError("Call chordalStrain after an advance and " +
                               "before the next update.")
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return m.log(self.len_curr / self.len_0)
            elif state == 'n' or state == 'next':
                return m.log(self.len_next / self.len_0)
            elif state == 'p' or state == 'prev' or state == 'previous':
                return m.log(self.len_prev / self.len_0)
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("Error: unknown state " +
                                   "{} in call to ceChord.chordalStrain."
                                   .format(state))
        else:
            raise RuntimeError("Error: unknown state " +
                               "{} in call to ceChord.chordalStrain."
                               .format(str(state)))

    def collagenForce(self, state):
        if not self.committed:
            raise RuntimeError("Call collagenForce after an advance and " +
                               "before the next update.")
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                stretch = self.len_curr / self.len_0
                area = self.A0_c / stretch
                return self.stress_c_curr * area
            elif state == 'n' or state == 'next':
                stretch = self.len_next / self.len_0
                area = self.A0_c / stretch
                return self.stress_c_next * area
            elif state == 'p' or state == 'prev' or state == 'previous':
                stretch = self.len_prev / self.len_0
                area = self.A0_c / stretch
                return self.stress_c_prev * area
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("Error: unknown state " +
                                   "{} in call to ceChord.collagenForce."
                                   .format(state))
        else:
            raise RuntimeError("Error: unknown state " +
                               "{} in call to ceChord.collagenForce."
                               .format(str(state)))

    def elastinForce(self, state):
        if not self.committed:
            raise RuntimeError("Call elastinForce after an advance and " +
                               "before the next update.")
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                stretch = self.len_curr / self.len_0
                area = self.A0_e / stretch
                return self.stress_e_curr * area
            elif state == 'n' or state == 'next':
                stretch = self.len_next / self.len_0
                area = self.A0_e / stretch
                return self.stress_e_next * area
            elif state == 'p' or state == 'prev' or state == 'previous':
                stretch = self.len_prev / self.len_0
                area = self.A0_e / stretch
                return self.stress_e_prev * area
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("Error: unknown state " +
                                   "{} in call to ceChord.elastinForce."
                                   .format(state))
        else:
            raise RuntimeError("Error: unknown state " +
                               "{} in call to ceChord.elastinForce."
                               .format(str(state)))

    def collagenStress(self, state):
        if not self.committed:
            raise RuntimeError("Call collagenStress after an advance and " +
                               "before the next update.")
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self.stress_c_curr
            elif state == 'n' or state == 'next':
                return self.stress_c_next
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self.stress_c_prev
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("Error: unknown state " +
                                   "{} in call to ceChord.collagenStress."
                                   .format(state))
        else:
            raise RuntimeError("Error: unknown state " +
                               "{} in call to ceChord.collagenStress."
                               .format(str(state)))

    def elastinStress(self, state):
        if not self.committed:
            raise RuntimeError("Call elastinStress after an advance and " +
                               "before the next update.")
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self.stress_e_curr
            elif state == 'n' or state == 'next':
                return self.stress_e_next
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self.stress_e_prev
            elif state == 'r' or state == 'ref' or state == 'reference':
                return 0.0
            else:
                raise RuntimeError("Error: unknown state " +
                                   "{} in call to ceChord.elastinStress."
                                   .format(state))
        else:
            raise RuntimeError("Error: unknown state " +
                               "{} in call to ceChord.elastinStress."
                               .format(str(state)))

    def collagenEntropy(self, state):
        if not self.committed:
            raise RuntimeError("Call collagenEntropy after an advance and " +
                               "before the next update.")
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self.eta_c_curr
            elif state == 'n' or state == 'next':
                return self.eta_c_next
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self.eta_c_prev
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self.rho_c * self.eta_c_0
            else:
                raise RuntimeError("Error: unknown state " +
                                   "{} in call to ceChord.collagenEntropy."
                                   .format(state))
        else:
            raise RuntimeError("Error: unknown state " +
                               "{} in call to ceChord.collagenEntropy."
                               .format(str(state)))

    def elastinEntropy(self, state):
        if not self.committed:
            raise RuntimeError("Call elastinEntropy after an advance and " +
                               "before the next update.")
        if isinstance(state, str):
            if state == 'c' or state == 'curr' or state == 'current':
                return self.eta_e_curr
            elif state == 'n' or state == 'next':
                return self.eta_e_next
            elif state == 'p' or state == 'prev' or state == 'previous':
                return self.eta_e_prev
            elif state == 'r' or state == 'ref' or state == 'reference':
                return self.rho_e * self.eta_e_0
            else:
                raise RuntimeError("Error: unknown state " +
                                   "{} in call to ceChord.elastinEntropy."
                                   .format(state))
        else:
            raise RuntimeError("Error: unknown state " +
                               "{} in call to ceChord.elastinEntropy."
                               .format(str(state)))
