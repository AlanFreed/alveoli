#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

"""
Module materialProperties.py provides properties for the alveolar constituents.

Copyright (c) 2019-2020 Alan D. Freed

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
__date__ = "10-05-2019"
__update__ = "06-24-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"

"""
A listing of changes made wrt version release can be found at the end of file.


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

Entropy densities (erg/gr.K) for the various alveolar constituents are:

    eta = etaAir()

    eta = etaBlood()

    eta = etaCollagen()

    eta = etaElastin()

    eta = etaH2O()

    eta = etaSepta()

Specific heats at constant pressure (erg/gr.K) for the alveolar consitituents:

    cp = CpAir()

    cp = CpBlood()

    cp = CpCollagen()

    cp = CpElastin()

    cp = CpH2O()

    cp = CpSepta()

lineal thermal strain coefficients (dimensionless) for aveolar constituents:

    alpha = alphaAir()

    alpha = alphaBlood()

    alpha = alphaCollagen()

    alpha = alphaElastin()

    alpha = alphaH2O()

    alpha = alphaSepta()

The following procedures supply random values describing geometries in alveoli.

    l = alveolarDiameter()
        Determines an alveolar diameter from its probability distribution.
        Associates with humans, age 15-35, whose lungs were fixed at 4 cm H20.
        Data are from: Sobin, Fung & Tremer, 1988.  The returned values have
        been extrapolated back to zero gauge pressure.

    d = fiberDiameterCollagen()
        Returns the diameter (in centimeters) of a collagen fiber determined
        from its probability distribution.  Associates with humans, age 15-35,
        whose lungs were fixed at 4 cm H20.  Data are from: Sobin, Fung and
        Tremer, 1988.  The returned values have been extrapolated back to zero
        gauge pressure.

    d = fiberDiameterElastin()
        Returns the diameter (in centimeters) of an elastin fiber determined
        from its probability distribution.  Associates with humans, age 15-35,
        whose lungs were fixed at 4 cm H20.  Data are from: Sobin, Fung and
        Tremer, 1988.  The returned values have been extrapolated back to zero
        gauge pressure.

    w = septalWidth()
        Returns a random thickness for a septal membrane.  No data are known
        from which to establish these statistics, so they are assumed.

The following procedures supply random values for the constitutive parameters.
No data are available to establish the statistical qualities of these model
parameters so they have all been asigned based upon our judgment.

    E1, E2, e_t, e_f, s_0 = collagenFiber()
        Returns random elastic moduli for the compliant, E1, and stiff, E2,
        fiber responses with a transtion between them occuring at strain e_t
        that is the limiting strain of molecular reconfiguration, plus it
        returns the strain at failure e_f and the initial pre-stress s_0.

    E1, E2, e_t, s_0 = elastinFiber()
        Returns random elastic moduli for the compliant, E1, and stiff, E2,
        fiber responses with a transtion between them occuring at strain e_t
        that is the limiting strain of molecular reconfiguration, and the
        initial or fiber pre-stress s_0.  (The elastin fiber is assumed not
        to rupture.)

    M1, M2, xi_t, N1, N2, epsilon_t, G1, G2, gamma_t, xi_f, pi_0
    = septalMembrane()
        Returns the elastic moduli for dilation (M1, M2, xi_t), for squeeze
        (N1, N2, epsilon_t), and for shear (G1, G2, gamma_t) so that Poisson's
        ratio is fixed at a half; however, the shear moduli are significantly
        less, on the order of a thousand times less than those for dilation
        and squeeze.  This is an important characteristic of tissues.  Also
        included is the strain at fracture, viz., xi_f = pi_f / M2, and the
        pre or residual surface tension pi_0 at zero dilation, i.e., @ xi = 0.

These parameters associate with an implicit elastic model of Freed & Rajagopal.

Reference:
    Freed, A. D. and Rajagopal, K. R., “A Promising Approach for Modeling
    Biological Fibers,” ACTA Mechanica, 227 (2016), 1609-1619.
    DOI: 10.1007/s00707-016-1583-8.  Errata: DOI: 10.1007/s00707-018-2183-6
"""

# volume fractions for constituents in septa, which are best estimates

vfCollagen = 0.05
vfElastin = 0.05
vfH2O = 0.6
vfBlood = 0.3


# mass densities are in grams per centimeter cubed

def rhoAir():
    rho = 1.16E-3
    return rho


def rhoBlood():
    rho = 1.04
    return rho


def rhoCollagen():
    rho = 1.34
    return rho


def rhoElastin():
    rho = 1.31
    return rho


def rhoH2O():
    rho = 1.0
    return rho


def rhoSepta():
    rho = (vfCollagen * rhoCollagen() + vfElastin * rhoElastin() +
           vfH2O * rhoH2O() + vfBlood * rhoBlood())
    return rho


# entropy densities are in ergs per gram Kelvin

def etaAir():
    eta = 3.796E7
    return eta


def etaBlood():
    eta = 3.0E7  # this is a guess - not able to find a value for this
    return eta


def etaCollagen():
    eta = 3.7E7
    return eta


def etaElastin():
    eta = 3.4E7
    return eta


def etaH2O():
    eta = 3.883E7
    return eta


def etaSepta():
    eta = (vfCollagen * etaCollagen() + vfElastin * etaElastin() +
           vfH2O * etaH2O() + vfBlood * etaBlood())
    return eta


# specific heats at constant pressure are in ergs per gram Kelvin

def CpAir():
    cp = 1.006E7
    return cp


def CpBlood():
    cp = 9.4E6
    return cp


def CpCollagen():
    cp = 1.7E7
    return cp


def CpElastin():
    cp = 4.2E7
    return cp


def CpH2O():
    cp = 4.187E7
    return cp


def CpSepta():
    cp = (vfCollagen * CpCollagen() + vfElastin * CpElastin() +
          vfH2O * CpH2O() + vfBlood * CpBlood())
    return cp


# coefficients  for  lineal thermal expansion (in reciprocal Kelvin)
# are converted into lineal thermal strain coefficients (dimensionless) via

def _convertAlpha(alpha):
    # convert alpha for thermal strain:  alpha (T - T0)
    # into an alpha for thermal strain:  alpha ln(T/T0)
    newAlpha = 310.0 * alpha   # T0 is body temperature
    return newAlpha


# linear thermal expansion coefficients

def alphaAir():
    alpha = 1.0 / 310.0  # = 1 / body temperature (in K)
    return _convertAlpha(alpha)


def alphaBlood():
    alpha = 2.5E-4
    return _convertAlpha(alpha)


def alphaCollagen():
    alpha = 1.8E-4
    return _convertAlpha(alpha)


def alphaElastin():
    alpha = 3.2E-4
    return _convertAlpha(alpha)


def alphaH2O():
    alpha = 2.9E-4
    return _convertAlpha(alpha)


def alphaSepta():
    alpha = (vfCollagen * alphaCollagen() + vfElastin * alphaElastin() +
             vfH2O * alphaH2O() + vfBlood * alphaBlood())
    return alpha


# statistical data


def alveolarDiameter():
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


def fiberDiameterCollagen():
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


def fiberDiameterElastin():
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


def collagenFiber():
    # no statistics available, based on judgment
    # the compliant modulus
    mu1 = 5.0E5     # the mean compliant modulus in barye
    sigma1 = 1.0E5  # the standard deviation
    E1 = random.gauss(mu1, sigma1)
    # bracket the permissible variability
    while (E1 < mu1 - 4.0 * sigma1) or (E1 > mu1 + 4.0 * sigma1):
        E1 = random.gauss(mu1, sigma1)
    # the stiffness modulus
    mu2 = 5.0E7     # the mean stiff modulus in barye
    sigma2 = 5.0E6  # the standard deviation
    E2 = random.gauss(mu2, sigma2)
    # bracket the permissible variability
    while (E1 >= E2) or (E2 < mu2 - 5.0 * sigma2) or (E2 > mu2 + 5.0 * sigma2):
        E2 = random.gauss(mu2, sigma2)
    # transition strain, i.e., limiting strain of molecular reconfiguration
    muT = 0.09       # the mean transition strain at 30% total lung capacity
    sigmaT = 0.018   # standard deviation
    e_t = random.gauss(muT, sigmaT)
    # bracket the permissible variability
    while (e_t < muT - 4.0 * sigmaT) or (e_t > muT + 4.0 * sigmaT):
        e_t = random.gauss(muT, sigmaT)
    # the strain at fracture, i.e., s_f = E2 * e_f
    muT = 0.25       # the mean rupture strain
    sigmaT = 0.025   # standard deviation
    e_f = random.gauss(muT, sigmaT)
    # bracket the permissible variability for fiber failure
    while (e_f < muT - 5.0 * sigmaT) or (e_f > muT + 5.0 * sigmaT):
        e_f = random.gauss(muT, sigmaT)
    # set the initial or fiber prestress
    s_0 = E1 * e_t / 2.0
    return E1, E2, e_t, e_f, s_0


def elastinFiber():
    # no statistics available, based on judgment
    # the compliant modulus
    mu1 = 2.3E6     # the mean compliant modulus in barye
    sigma1 = 3.0E5  # the standard deviation
    E1 = random.gauss(mu1, sigma1)
    # bracket the permissible variability
    while (E1 < mu1 - 5.0 * sigma1) or (E1 > mu1 + 5.0 * sigma1):
        E1 = random.gauss(mu1, sigma1)
    # the stiffness modulus
    mu2 = 1.0E7     # the mean stiff modulus in barye
    sigma2 = 1.0E6  # the standard deviation
    E2 = random.gauss(mu2, sigma2)
    # bracket the permissible variability
    while (E1 >= E2) or (E2 < mu2 - 5.0 * sigma2) or (E2 > mu2 + 5.0 * sigma2):
        E2 = random.gauss(mu2, sigma2)
    # the transition strain
    muT = 0.4        # the mean transition strain
    sigmaT = 0.08    # standard deviation
    e_t = random.gauss(muT, sigmaT)
    # bracket the permissible variability
    while (e_t < muT - 4.0 * sigmaT) or (e_t > muT + 4.0 * sigmaT):
        e_t = random.gauss(muT, sigmaT)
    # set the initial or fiber prestress
    s_0 = E1 * e_t / 2.0
    return E1, E2, e_t, s_0


def septalMembrane():
    # no statistics available, based on judgment

    # the dilation response

    # the compliant modulus
    mu1 = 1.0E4     # the mean compliant modulus in barye
    sigma1 = 1.0E3  # the standard deviation
    M1 = random.gauss(mu1, sigma1)
    # bracket the permissible variability
    while (M1 < mu1 - 4.0 * sigma1) or (M1 > mu1 + 4.0 * sigma1):
        M1 = random.gauss(mu1, sigma1)
    # the stiffness modulus
    mu2 = 3.0E6     # the mean stiff modulus in barye
    sigma2 = 1.0E5  # the standard deviation
    M2 = random.gauss(mu2, sigma2)
    # bracket the permissible variability
    while (M1 >= M2) or (M2 < mu2 - 4.0 * sigma2) or (M2 > mu2 + 4.0 * sigma2):
        M2 = random.gauss(mu2, sigma2)
    # the transition strain
    muT = 0.24      # the mean transition strain
    sigmaT = 0.025  # standard deviation
    xi_t = random.gauss(muT, sigmaT)
    # bracket the permissible variability
    while (xi_t < muT - 4.0 * sigmaT) or (xi_t > muT + 4.0 * sigmaT):
        xi_t = random.gauss(muT, sigmaT)

    # the squeeze response

    N1 = 2.0 * M1 / 3.0
    N2 = 5.0 * M2 / 4.0
    epsilon_t = xi_t / 4.0

    # the shear response

    # assumed to be approximately 1/1000th that of the dilation response
    G1 = M1 / 25.0
    G2 = M2 / 25.0
    gamma_t = 3.0 * xi_t / 2.0

    # the maximum strain at rupture: pi_f = M2 * xi_f
    xi_f = 0.2

    # the membrane pre-stressing of the surface tension
    pi_0 = M1 * xi_t / 2.0

    return M1, M2, xi_t, N1, N2, epsilon_t, G1, G2, gamma_t, xi_f, pi_0


# set a truly random seed for the random number generator
random.seed()

"""
Changes made in version "1.0.0":

This is the initial version of materialProperties.py.
"""
