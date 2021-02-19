#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
__date__ = "11-10-2019"
__update__ = "11-15-2020"
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
        Determines an alveolar diameter.
        Associates with humans, age 15-35, whose lungs were fixed at 4 cm H20.
        Data are from: Sobin, Fung & Tremer, 1988.  The returned values have
        been extrapolated back to zero gauge pressure.

    d = fiberDiameterCollagen()
        Returns the diameter (in centimeters) of a collagen fiber determined.  
        Associates with humans, age 15-35,
        whose lungs were fixed at 4 cm H20.  Data are from: Sobin, Fung and
        Tremer, 1988.  The returned values have been extrapolated back to zero
        gauge pressure.

    d = fiberDiameterElastin()
        Returns the diameter (in centimeters) of an elastin fiber determined.  
        Associates with humans, age 15-35,
        whose lungs were fixed at 4 cm H20.  Data are from: Sobin, Fung and
        Tremer, 1988.  The returned values have been extrapolated back to zero
        gauge pressure.

    w = septalWidth()
        Returns a mean thickness for a septal membrane.  No data are known
        from which to establish these statistics, so they are assumed.

The following procedures supply random values for the constitutive parameters.
No data are available to establish the statistical qualities of these model
parameters so they have all been asigned based upon our judgment.

    E1, E2, e_t, e_f, s_0 = collagenFiber()
        Returns the mean elastic moduli for the compliant, E1, and stiff, E2,
        fiber responses with a transtion between them occuring at strain e_t
        that is the limiting strain of molecular reconfiguration, plus it
        returns the strain at failure e_f and the initial pre-stress s_0.

    E1, E2, e_t, s_0 = elastinFiber()
        Returns the mean elastic moduli for the compliant, E1, and stiff, E2,
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


# mass densities are in grams per micron cubed

def rhoAir():
    rho = 1.16E-15
    return rho


def rhoBlood():
    rho = 1.04E-12
    return rho


def rhoCollagen():
    rho = 1.34E-12
    return rho


def rhoElastin():
    rho = 1.31E-12
    return rho


def rhoH2O():
    rho = 1.0E-12
    return rho


def rhoSepta():
    rho = (vfCollagen * rhoCollagen() + vfElastin * rhoElastin() +
           vfH2O * rhoH2O() + vfBlood * rhoBlood())
    return rho


# entropy densities are in ergs per gram Kelvin

def etaAir():
    eta = 3.796E11
    return eta


def etaBlood():
    eta = 3.0E11  # this is a guess - not able to find a value for this
    return eta


def etaCollagen():
    eta = 3.7E11
    return eta


def etaElastin():
    eta = 3.4E11
    return eta


def etaH2O():
    eta = 3.883E11
    return eta


def etaSepta():
    eta = (vfCollagen * etaCollagen() + vfElastin * etaElastin() +
           vfH2O * etaH2O() + vfBlood * etaBlood())
    return eta


# specific heats at constant pressure are in ergs per gram Kelvin

def CpAir():
    cp = 1.006E11
    return cp


def CpBlood():
    cp = 9.4E10
    return cp


def CpCollagen():
    cp = 1.7E11
    return cp


def CpElastin():
    cp = 4.2E11
    return cp


def CpH2O():
    cp = 4.187E11
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
    dia = mu
    # # convert microns to centimeters
    # dia = dia / 10000.0
    return dia  # in micron


def fiberDiameterCollagen():
    # Statistics from: Sobin, Fung and Tremer, J. Appl. Phys., Vol. 64, 1988.
    # the square root of fiber diameter was found to distribute normally
    mu = 0.952      # mean of the square root of fiber diameter in microns
    dia = mu
    # # convert microns to centimeters
    # dia = dia / 10000.0
    return dia  # in micron


def fiberDiameterElastin():
    # Statistics from: Sobin, Fung and Tremer, J. Appl. Phys., Vol. 64, 1988.
    # the square root of fiber diameter was found to distribute normally
    mu = 0.957      # mean of the square root of fiber diameter in microns
    dia = mu
    # # convert microns to centimeters
    # dia = dia / 10000.0
    return dia  # in micron


def septalWidth():
    # no statistics available, based on judgment
    mu = 4.5       # the mean thickness is taken to be 4.5 microns
    width = mu
    # # convert microns to centimeters
    # width = width / 10000.0
    return width  # in microns


def collagenFiber():
    # no statistics available, based on judgment
    # the compliant modulus
    mu1 = 5.0E1     # the mean compliant modulus in barye
    E1 = mu1
    # the stiffness modulus
    mu2 = 5.0E3     # the mean stiff modulus in barye
    E2 = mu2
    # transition strain, i.e., limiting strain of molecular reconfiguration
    muT = 0.09       # the mean transition strain at 30% total lung capacity
    e_t = muT
    # the strain at fracture, i.e., s_f = E2 * e_f
    muT = 0.25       # the mean rupture strain
    e_f = muT
    # set the initial or fiber prestress
    s_0 = E1 * e_t / 2.0
    return E1, E2, e_t, e_f, s_0


def elastinFiber():
    # no statistics available, based on judgment
    # the compliant modulus
    mu1 = 2.3E2     # the mean compliant modulus in barye
    E1 = mu1
    # the stiffness modulus
    mu2 = 1.0E3     # the mean stiff modulus in barye
    E2 = mu2
    # the transition strain
    muT = 0.4        # the mean transition strain
    e_t = muT
    # set the initial or fiber prestress
    s_0 = E1 * e_t / 2.0
    return E1, E2, e_t, s_0


def septalMembrane():
    # no statistics available, based on judgment

    # the dilation response

    # the compliant modulus
    mu1 = 1.0     # the mean compliant modulus in barye (1.0E4), in g/(micron.s^2) (1.0)
    M1 = mu1
    # the stiffness modulus
    mu2 = 300.0     # the mean stiff modulus in barye (3.0E6), in g/(micron.s^2) (300.0) 
    M2 = mu2
    # the transition strain
    muT = 0.24      # the mean transition strain
    xi_t = muT

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


def septalSac():
    # no statistics available, based on judgment
    
    #  The atmospheric pressure at sea level (1 bar or 10E4 Pa or 10E5 barye),
    p_0 = 10E1              # pressure                   (barye)
    # V0 / V
    v0v = 1.002616
    return p_0, v0v
    

