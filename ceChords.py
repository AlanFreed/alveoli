#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import materialProperties as mp
import math as m
import numpy as np
from peceVtoX import pece
import splines

"""
Module ceChords.py provides a constitutive description for alveolar chords.

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
__update__ = "11-07-2019"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
This module describes septal chords.  From histological studies, these septal
chords are found to be comprised of both collagen and elastin fibers that
align in parallel with one another with minimal chemical bonding between them.
Consequently, they are modeled here as two elastic rods exposed to the same
strain but carrying different states of stress.  Their geometric properties
are described by statistical distributions exported from materialProperties.py.
Given these properties, one can create an object describing a septal chord.
This module exports the class ceChord that evaluates a constitutive response
for these chordal fibers which circumscribe septa surrounding an alveolar sac.

The physical units adopted here are CGS:
    length          centimeters
    mass            grams
    time            seconds
    temperature     centigrade
    force           dynes   [gr.cm/s^2]     1 Newton = 10^5 dyne
    pressure        barye   [dyne/cm^2]     1 Pascal = 10 barye
    energy          erg     [dyne.cm]       1 Joule  = 10^7 ergs

class ceFiber

    Creates an object that implements the Freed-Rajagopal elastic fiber model.
    Objects of this class are created and used internally.

constructor

    ce = ceFiber(E1, E2, e_t)
            E1      the compliant modulus, i.e., at zero stress and zero strain
            E2      the stiff modulus, i.e., elastic modulus at terminal strain
            e_t     the transition strain between compliant and stiff response

methods

    C = ce.compliance(strain, stress)
        Returns the tangent compliance at specified state of stress and strain.

    E = ce.modulus(strain, stress)
        Returns the tangent modulus at a specified state of stress and strain.

class ceChord

    Creates an object used for determining the constitutive response of septal
    chords. They are represented using the Freed-Rajagopal elastic fiber model.
    When not specifically assigned, the fiber diameters are extracted from
    their statistical distributions exported by materialProperties.py.

constructors
    There are two options for calling the class constructor, they are:

    c = ceChord(lenF)
            lenF        reference length in centimeters for the two fibers

    c = ceChord(lenF, diaC, diaE, nodalPts)
            lenF        reference  length  in centimeters for these two fibers
            diaC        reference diameter in centimeters for collagen fiber
                        setting to 0.0 will result in a statistical assignment
            diaE        reference diameter in centimeters for elastin fiber
                        setting to 0.0 will result in a statistical assignment
            nodalPts    number of nodes used to spline the stress/strain curve
                        default is 51: 25 in tension, 25 in compression

    This constructor solves the constitutive equation over a maximum range of
    response and then creates a cubic spline for efficient interpolation of
    this response used by the various methods exported by the object.  Numeric
    integration is done using a two-step PECE method exported from peceVtoX.py.

methods

Values are per alveolus, so mass, force and entropy are each divided by 3, with
the other 2/3rds associating with neighboring alveoli.  Stress and strain are
not divided by 3.

    Chordal Properties Belonging to an Alveolus

    m = c.chordalMass()
        Returns the mass of a septal chord divided by 3 thereby returning the
        mass of this chord for the alveolus in question.  Units are in: grams.

    epsilon = c.chordalStrain(length)
        Returns the true strain of this chord given a specified chordal lentgh.

    eta = c.chordalEntropy(strain)
        Returns the entropy of a septal chord at a specified state of strain
        divided by 3 thereby returning the entropy of this chord for the
        alveolus in quesiton.  Units are in: erg/K; it is not a density.

    f = c.chordalForce(strain)
        Returns the force carried by a septal chord at a specified strain
        divided by 3 thereby returning the force carried by this chord for the
        alveolus in question.  Units are in: dynes.

    sigma = c.chordalStress(strain)
        Returns the stress carried by a septal chord at a specified strain.
        Units are in: barye.

    Individual Fiber Properties

    eta = c.collagenEntropy(strain)
        Returns the alveolar entropy of the collagen fiber in this septal chord
        at a specified strain.  Units are in: erg/K; it is not a density.

    eta = c.elastinEntropy(strain)
        Returns the alveolar entropy of the elastin fiber in this septal chord
        at a specified strain.  Units are in: erg/K; it is not a density.

    f = c.collagenForce(strain)
        Returns the alveolar force carried by the collagen fiber in this septal
        chord at a specified strain.  Units are in: dynes.

    f = c.elastinForce(strain)
        Returns the alveolar force carried by the elastin fiber in this septal
        chord at a specified strain.  Units are in: dynes.

    sigma = c.collagenStress(strain)
        Returns the true stress carried by the collagen fiber in this septal
        chord at a specified strain.  Units are in: barye.

    sigma = c.elastinStress(strain)
        Returns the true stress carried by the elastin fiber in this septal
        chord at a specified strain.  Units are in: barye.
"""

# constitutive model based upon the Freed-Rajagopal model for biologic fibers


class ceFiber(object):

    def __init__(self, E1, E2, e_t):
        if E1 > np.finfo(float).eps:
            self.E1 = E1
        else:
            raise RuntimeError('Initial modulus E1 must be positive.')
        if E2 > self.E1:
            self.E2 = E2
        else:
            raise RuntimeError('Terminal modulus E2 must be greater than E1.')
        if e_t > np.finfo(float).eps:
            self.e_t = e_t
        else:
            raise RuntimeError('Limiting strain e_t must be positive.')

    def compliance(self, strain, stress):
        if stress <= 0.0:
            c = (self.E1 + self.E2) / (self.E1 * self.E2)
        else:
            # there is no thermal strain effect considered in this application
            c = ((self.e_t + stress / self.E2 - strain) /
                 (self.E1 * self.e_t + 2.0 * stress) + 1.0 / self.E2)
        return np.array([c])

    def modulus(self, strain, stress):
        c = self.compliance(strain, stress)
        return np.array([1.0 / c[0]])


# constitutive class for chords


class ceChord(object):

    # constructor

    def __init__(self, lenF, diaC=None, diaE=None, nodalPts=51):

        # verify the inputs

        lenF = abs(float(lenF))
        if lenF < np.finfo(float).eps:
            raise RuntimeError('Initial fiber length lenF must be positive.')
        if (diaC is None) or (float(diaC) < np.finfo(float).eps):
            diaC = mp.fiberDiameterCollagen()
        else:
            diaC = float(diaC)
        if (diaE is None) or (float(diaE) < np.finfo(float).eps):
            diaE = mp.fiberDiameterElastin()
        else:
            diaE = float(diaE)
        nodalPts = int(abs(nodalPts))
        if nodalPts < 25:
            nodalPts = 25
        else:
            nodalPts = 2 * (nodalPts // 2) + 1

        # establish the geometric quantities

        self.A0_c = m.pi * diaC**2 / 4.0
        self.A0_e = m.pi * diaE**2 / 4.0
        self.len_0 = lenF

        # create constitutive equations for the collagen and elastin fibers
        # all properties are assigned via their statistical distributions

        E1_c, E2_c, et_c = mp.collagenFiber()
        fiber_c = ceFiber(E1_c, E2_c, et_c)
        E1_e, E2_e, et_e = mp.elastinFiber()
        fiber_e = ceFiber(E1_e, E2_e, et_e)

        # maximum strains associate with stresses that are 0.1 * E2
        strainMax_c = et_c + 0.1
        strainMax_e = et_e + 0.1
        self.strainMax = min(et_c, et_e) + 0.1

        # arrays to hold stress-strain response data over range +/- strainMax
        self.strain_c = np.zeros(nodalPts, dtype=float)
        self.strain_e = np.zeros(nodalPts, dtype=float)
        stress_c = np.zeros(nodalPts, dtype=float)
        stress_e = np.zeros(nodalPts, dtype=float)

        # fiber stress-strain responses
        dStrain_c = strainMax_c / (nodalPts // 2)
        dStrain_e = strainMax_e / (nodalPts // 2)
        strain0 = 0.0
        stress0 = np.zeros(1, dtype=float)

        # stress-strain response in compression is compliant Hookean
        E_c = fiber_c.modulus(strain0, stress0)
        E_e = fiber_e.modulus(strain0, stress0)
        for i in range(nodalPts//2-1, -1, -1):
            self.strain_c[i] = self.strain_c[i+1] - dStrain_c
            stress_c[i] = E_c * self.strain_c[i]
            self.strain_e[i] = self.strain_e[i+1] - dStrain_e
            stress_e[i] = E_e * self.strain_e[i]

        # integrate stress-strain response in tension via Freed-Rajagopal model
        tolerance = 0.001
        pece_c = pece(fiber_c.modulus, strain0, stress0, dStrain_c, tolerance)
        pece_e = pece(fiber_e.modulus, strain0, stress0, dStrain_e, tolerance)
        for i in range(nodalPts//2+1, nodalPts):
            self.strain_c[i] = self.strain_c[i-1] + dStrain_c
            pece_c.integrate()
            pece_c.advance()
            stress = pece_c.getX()
            stress_c[i] = stress[0]
            self.strain_e[i] = self.strain_e[i-1] + dStrain_e
            pece_e.integrate()
            pece_e.advance()
            stress = pece_e.getX()
            stress_e[i] = stress[0]

        # fit stress-strain data with a cubic spline for stress evaluations
        self.a_c, self.b_c, self.c_c, self.d_c = splines.getCoef(self.strain_c,
                                                                 stress_c)
        self.a_e, self.b_e, self.c_e, self.d_e = splines.getCoef(self.strain_e,
                                                                 stress_e)

        return  # a new instance of type ceChord

    # chordal properties

    def chordalMass(self):
        # mass of the septal chord
        mass_c = self.A0_c * self.len_0 * mp.rhoCollagen()
        mass_e = self.A0_e * self.len_0 * mp.rhoElastin()
        # divide total chordal mass by 3 to get chordal mass for the alveolus
        mass = (mass_c + mass_e) / 3.0
        return mass  # in gr

    def chordalStrain(self, length):
        strain = m.log(length / self.len_0)
        if (strain < -self.strainMax) or (strain > self.strainMax):
            raise RuntimeError(
                         "Chordal strain exceeds its admissible strain range.")
        return strain

    def chordalEntropy(self, strain):
        entropy = self.collagenEntropy(strain) + self.elastinEntropy(strain)
        return entropy  # in erg/K

    def chordalForce(self, strain):
        f = self.collagenForce(strain) + self.elastinForce(strain)
        return f  # in dynes

    def chordalStress(self, strain):
        # use the rule of mixtures to determine chordal stress
        sigma = ((self.A0_c * self.collagenStress(strain) +
                  self.A0_e * self.elastinStress(strain)) /
                 (self.A0_c + self.A0_e))
        return sigma  # in barye

    # collagen and elastin fiber properties

    def collagenEntropy(self, strain):
        # divide the entropy by 3 as each chord is shared between 3 alveoli
        eta = (self.A0_c * self.len_0 *  # assumes volume is preserved
               (mp.rhoCollagen() * mp.etaCollagen() +
                mp.alphaCollagen() * self.collagenStress(strain)) / 3.0)
        return eta  # in erg/K

    def elastinEntropy(self, strain):
        # divide the entropy by 3 as each chord is shared between 3 alveoli
        eta = (self.A0_e * self.len_0 *  # assumes volume is preserved
               (mp.rhoElastin() * mp.etaElastin() +
                mp.alphaElastin() * self.elastinStress(strain)) / 3.0)
        return eta  # in erg/K

    def collagenForce(self, strain):
        stress = self.collagenStress(strain)
        area = self.A0_c / m.exp(strain)  # assumes volume is preserved
        # divide the force by 3 as each chord is shared between 3 alveoli
        f = stress * area / 3.0
        return f  # in dynes

    def elastinForce(self, strain):
        stress = self.elastinStress(strain)
        area = self.A0_e / m.exp(strain)  # assumes volume is preserved
        # divide the force by 3 as each chord is shared between 3 alveoli
        f = stress * area / 3.0
        return f  # in dynes

    def collagenStress(self, strain):
        stress = splines.Y(self.a_c, self.b_c, self.c_c, self.d_c,
                           self.strain_c, strain)
        return stress  # in barye

    def elastinStress(self, strain):
        stress = splines.Y(self.a_e, self.b_e, self.c_e, self.d_e,
                           self.strain_e, strain)
        return stress  # in barye
