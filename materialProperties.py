#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

"""
Module materialProperties.py provides properties for the alveolar constituents.

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
__date__ = "10-05-2019"
__update__ = "10-06-2019"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
For those properties that have units, CGS units are used here:
    length          centimeters
    mass            grams
    time            seconds
    temperature     centigrade
    force           dynes   [gr.cm/s^2]     1 Newton = 10^5 dyne
    pressure        barye   [dyne/cm^2]     1 Pascal = 10 barye
    energy          erg     [dyne.cm]       1 Joule  = 10^7 ergs

procedures

Mass densities (gr/cm^3) for the various alveolar constituents are:

    rho = rhoAir()

    rho = rhoBlood()

    rho = rhoCollagen()

    rho = rhoElastin()

    rho = rhoH2O()

    rho = rhoSepta()

The following procedures supply random values describing alveolar geometry.

    l = alveolarDia()
        Determines a random length l for the septal fiber in centimeters.  From
        this information, a mean alveolar diameter is determined and returned.
        Associates with humans, age 15-35, whose lungs were fixed at 4 cm H20.
        Data are from: Sobin, Fung & Tremer, 1988.  The values are extrapolated
        back to zero gauge pressure, which are the returned values.

    d = chordDiaCollagen()
        Returns a random diameter d for a collegen septal fiber in centimeters.
        Associates with humans, age 15-35, whose lungs were fixed at 4 cm H20.
        Data are from: Sobin, Fung & Tremer, 1988.

    d = chordDiaElastin()
        Returns a random diameter d for an elastin septal fiber in centimeters.
        Associates with humans, age 15-35, whose lungs were fixed at 4 cm H2O.
        Data are from: Sobin, Fung & Tremer, 1988.

    w = septalWidth()
        Returns a random thickness for the septal membrane.  No data are known
        from which to establish these statistics, so they are assumed.

The following procedures supply random values for the constitutive parameters.
No data are available to establish the statistical qualities of these model
parameters so they are all asigned based upon our judgment.

    E1, E2, e_t = collagenChord()
        Returns random elastic moduli for the compliant, E1, and stiff, E2,
        fiber responses with a transtion between them occuring at strain e_t.

    E1, E2, e_t = elastinChord()
        Returns random elastic moduli for the compliant, E1, and stiff, E2,
        fiber responses with a transtion between them occuring at strain e_t.

    M1, M2, e_Mt, N1, N2, e_Nt, G1, G2, e_Gt = septalMembrane()
        Returns the elastic moduli for dilation (M1, M2, e_Mt), for squeeze
        (N1, N2, e_Nt) and for shear (G1, G2, e_Gt) such that Poisson's ratio
        is fixed at a half; consequently, (N1, N2, e_Nt) = (M1, M2, e_Mt)/3;
        however, the shear moduli are significantly less, on the order of a
        thousand times less.  This is an important characteristic of tissues.
        The first moduli in these sets describe the compliant response, the
        second moduli in these sets describe the stiff response, and the third
        set of values represent their strains of transition.

These parameters associate with the implicit elastic model of Freed & Rajagopal
"""


def rhoAir():
    rho = 1.16E-3
    return rho  # in grams per centimeter cubed


def rhoBlood():
    rho = 1.04
    return rho  # in grams per centimeter cubed


def rhoCollagen():
    rho = 1.34
    return rho  # in grams per centimeter cubed


def rhoElastin():
    rho = 1.31
    return rho  # in grams per centimeter cubed


def rhoH2O():
    rho = 1.0
    return rho  # in grams per centimeter cubed


def rhoSepta():
    # these volume fractions are our best estimates
    rho = (0.1 * (rhoCollagen() + rhoElastin()) +
           0.5 * rhoH2O() + 0.3 * rhoBlood())
    return rho  # in grams per centimeter cubed


def alveolarDia():
    # Statistics from: Sobin, Fung and Tremer, J. Appl. Phys., Vol. 64, 1988.
    # alveolar diameter was found to distribute normally
    mu = 177.0      # mean diameter in microns: extrapolated to zero pressure
    sigma = 84.0    # standard deviation:       extrapolated to zero pressure
    dia = random.gauss(mu, sigma)
    # bracket the permissible variability
    while (dia < mu - 1.5 * sigma) or (dia > mu + 4.0 * sigma):
        dia = random.gauss(mu, sigma)
    # convert microns to centimeters
    dia = dia / 10000.0
    return dia  # in centimeters


def chordDiaCollagen():
    # Statistics from: Sobin, Fung and Tremer, J. Appl. Phys., Vol. 64, 1988.
    # the square root of fiber diameter was found to distribute normally
    mu = 0.952      # mean of the square root of fiber diameter in microns
    sigma = 0.242   # standard deviation
    sqrtDia = random.gauss(mu, sigma)
    # bracket the permissible variability
    while (sqrtDia < mu - 3.0 * sigma) or (sqrtDia > mu + 4.0 * sigma):
        sqrtDia = random.gauss(mu, sigma)
    dia = sqrtDia**2
    # convert microns to centimeters
    dia = dia / 10000.0
    return dia  # in centimeters


def chordDiaElastin():
    # Statistics from: Sobin, Fung and Tremer, J. Appl. Phys., Vol. 64, 1988.
    # the square root of fiber diameter was found to distribute normally
    mu = 0.957      # mean of the square root of fiber diameter in microns
    sigma = 0.239   # standard deviation
    sqrtDia = random.gauss(mu, sigma)
    # bracket the permissible variability
    while (sqrtDia < mu - 3.0 * sigma) or (sqrtDia > mu + 4.0 * sigma):
        sqrtDia = random.gauss(mu, sigma)
    dia = sqrtDia**2
    # convert microns to centimeters
    dia = dia / 10000.0
    return dia  # in centimeters


def septalWidth():
    # no statistics available, based on judgment
    mu = 4.5       # the mean thickness is taken to be 4.5 microns
    sigma = 0.5    # the standard deviation is taken to be a half of a micron
    width = random.gauss(mu, sigma)
    # bracket the permissible variability
    while (width < mu - 4.0 * sigma) or (width > mu + 4.0 * sigma):
        width = random.gauss(mu, sigma)
    # convert microns to centimeters
    width = width / 10000.0
    return width  # in centimeters


def collagenChord():
    # no statistics available, based on judgment
    # the compliant modulus
    mu1 = 5.0E4     # the mean compliant modulus in barye
    sigma1 = 2.0E4  # the standard deviation
    E1 = random.gauss(mu1, sigma1)
    # bracket the permissible variability
    while (E1 < mu1 - 2.0 * sigma1) or (E1 > mu1 + 4.0 * sigma1):
        E1 = random.gauss(mu1, sigma1)
    # the stiffness modulus
    mu2 = 5.0E7     # the mean stiff modulus in barye
    sigma2 = 1.5E7  # the standard deviation
    E2 = random.gauss(mu2, sigma2)
    # bracket the permissible variability
    while (E1 >= E2) or (E2 < mu2 - 3.0 * sigma2) or (E2 > mu2 + 4.0 * sigma2):
        E2 = random.gauss(mu2, sigma2)
    # the transition strain
    muT = 0.09       # the mean transition strain at 30% total lung capacity
    sigmaT = 0.03    # standard deviation
    e_t = random.gauss(muT, sigmaT)
    # bracket the permissible variability
    while (e_t < muT - 2.0 * sigmaT) or (e_t > muT + 4.0 * sigmaT):
        e_t = random.gauss(muT, sigmaT)
    return E1, E2, e_t


def elastinChord():
    # no statistics available, based on judgment
    # the compliant modulus
    mu1 = 2.3E6     # the mean compliant modulus in barye
    sigma1 = 1.0E6  # the standard deviation
    E1 = random.gauss(mu1, sigma1)
    # bracket the permissible variability
    while (E1 < mu1 - 2.0 * sigma1) or (E1 > mu1 + 4.0 * sigma1):
        E1 = random.gauss(mu1, sigma1)
    # the stiffness modulus
    mu2 = 1.0E7     # the mean stiff modulus in barye
    sigma2 = 3.0E6  # the standard deviation
    E2 = random.gauss(mu2, sigma2)
    # bracket the permissible variability
    while (E1 >= E2) or (E2 < mu2 - 3.0 * sigma2) or (E2 > mu2 + 4.0 * sigma2):
        E2 = random.gauss(mu2, sigma2)
    # the transition strain
    muT = 0.4        # the mean transition strain
    sigmaT = 0.1    # standard deviation
    e_t = random.gauss(muT, sigmaT)
    # bracket the permissible variability
    while (e_t < muT - 3.0 * sigmaT) or (e_t > muT + 4.0 * sigmaT):
        e_t = random.gauss(muT, sigmaT)
    return E1, E2, e_t


def septalMembrane():
    # no statistics available, based on judgment
    # We use the data fit of the model for viseral plueral from:
    # Freed, Erel, and Moreno, JoMMS, Vol. 12, 2017.
    # The visceral plueral is about 50 +/- 30 microns thick, which is about
    # 100 times thicker than a typical basement membrane in a septal plane.

    # the dilation response

    # the compliant modulus
    mu1 = 1.0E4     # the mean compliant modulus in barye
    sigma1 = 3.0E3  # the standard deviation
    M1 = random.gauss(mu1, sigma1)
    # bracket the permissible variability
    while (M1 < mu1 - 3.0 * sigma1) or (M1 > mu1 + 4.0 * sigma1):
        M1 = random.gauss(mu1, sigma1)
    # the stiffness modulus
    mu2 = 4.0E7     # the mean stiff modulus in barye
    sigma2 = 1.0E7  # the standard deviation
    M2 = random.gauss(mu2, sigma2)
    # bracket the permissible variability
    while (M1 >= M2) or (M2 < mu2 - 3.0 * sigma2) or (M2 > mu2 + 4.0 * sigma2):
        M2 = random.gauss(mu2, sigma2)
    # the transition strain
    muT = 0.2       # the mean transition strain
    sigmaT = 0.05   # standard deviation
    e_Mt = random.gauss(muT, sigmaT)
    # bracket the permissible variability
    while (e_Mt < muT - 3.0 * sigmaT) or (e_Mt > muT + 4.0 * sigmaT):
        e_Mt = random.gauss(muT, sigmaT)

    # the squeeze response

    # for Poisons ratio to be a half it follows that
    N1 = M1 / 3.0
    N2 = M2 / 3.0
    e_Nt = e_Mt / 3.0

    # the shear response

    # assumed to be approximately 1/1000th that of the dilation response
    G1 = M1 / 1000.0
    G2 = M2 / 1000.0
    e_Gt = e_Mt

    return M1, M2, e_Mt, N1, N2, e_Nt, G1, G2, e_Gt


# set a truly random seed for the random number generator
random.seed()
