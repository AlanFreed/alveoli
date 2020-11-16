

import meanProperties as mp
import numpy as np
from peceHE import Control, Response

"""
Module ceSac.py provides a constitutive description for alveolar Sac.

Copyright (c) 2021 Shahla Zamani

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
__date__ = "11-07-2020"
__update__ = "11-16-2020"
__author__ = "Shahla Zamani"
__author_email__ = "zamani.shahla@tamu.edu"

"""

This module describes septal sac, and exports two classes:
    controlsac: manages the control variables for a biologic sac, and
    ceSac:      manages the response variables of a biologic sac.

The CGS system of physical units adopted:
    length          centimeters   [cm]
    mass            grams         [g]
    time            seconds       [s]
    temperature     centigrade    [C]
where
    force           dynes         [g.cm/s^2]      1 Newton = 10^5 dyne
    pressure        barye         [dyne/cm^2]     1 Pascal = 10 barye
    energy          erg           [dyne.cm]       1 Joule  = 10^7 ergs


class controlSac:  It implements and extends class 'control'


For 3D sac, the physical control vector has components of:
    #     eVec[0]  thermal strain:   ln(T/T_0)                  (dimensionless)
    #     eVec[1]  dilation:         ln(a/a_0 * b/b_0 * c/c_0)  (dimensionless)
    #     eVec[2]  squeeze:          ln(a/a_0 * b_0/b)          (dimensionless)
    #     eVec[2]  squeeze:          ln(b/b_0 * c_0/c)          (dimensionless)
    #     eVec[3]  shear:            alp - alp_0                (dimensionless)
    #     eVec[3]  shear:            bet - bet_0                (dimensionless)
    #     eVec[3]  shear:            gam - gam_0                (dimensionless)
    
    # and the thermodynamic response vector has componenents of:
    #     yVec[0]  entropy density           'eta'                    (erg/g.K)
    #     yVec[1]  pressure                  'pi'                     (barye)
    #     yVec[2]  normal stress difference  'sigma1'                 (barye)
    #     yVec[3]  normal stress difference  'sigma2'                 (barye)
    #     yVec[4]  shear stress              'tau1'                   (barye)
    #     yVec[5]  shear stress              'tau2'                   (barye)
    #     yVec[6]  shear stress              'tau3'                   (barye)
    
where a, b, c, g1, g2 and g3 come from the QR decomposition of a deformation gradient,
which needs to be pivoted prior to its decomposition into a, b, c, g1, g2 and g3.

constructor

    E.g.: ctrl = controlSac(eVec0, xVec0, dt)
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


class ceSac:  It implements and extends class 'response'.


    #     xVec[0]  temperature                         'T'      (centigrade)
    #     xVec[1]  elongation in 1 direction           'a'      (dimensionless)
    #     xVec[2]  elongation in 2 direction           'b'      (dimensionless)
    #     xVec[3]  elongation in 3 direction           'c'      (dimensionless)
    #     xVec[4]  magnitude of shear in the 23 plane  'alp'    (dimensionless)
    #     xVec[5]  magnitude of shear in the 13 plane  'bet'    (dimensionless)
    #     xVec[6]  magnitude of shear in the 12 plane  'gam'  
    # while the thermodynamic control vector has strain components of:
    #     eVec[0]  thermal strain:   ln(T/T_0)                  (dimensionless)
    #     eVec[1]  dilation:         ln(a/a_0 * b/b_0 * c/c_0)  (dimensionless)
    #     eVec[2]  squeeze:          ln(a/a_0 * b_0/b)          (dimensionless)
    #     eVec[2]  squeeze:          ln(b/b_0 * c_0/c)          (dimensionless)
    #     eVec[3]  shear:            alp - alp_0                (dimensionless)
    #     eVec[3]  shear:            bet - bet_0                (dimensionless)
    #     eVec[3]  shear:            gam - gam_0                (dimensionless)
    # and the thermodynamic response vector has componenents of:
    #     yVec[0]  entropy density           'eta'                    (erg/g.K)
    #     yVec[1]  pressure                  'pi'                     (barye)
    #     yVec[2]  normal stress difference  'sigma1'                 (barye)
    #     yVec[3]  normal stress difference  'sigma2'                 (barye)
    #     yVec[4]  shear stress              'tau1'                   (barye)
    #     yVec[5]  shear stress              'tau2'                   (barye)
    #     yVec[6]  shear stress              'tau3'                   (barye)


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
        de/dt       is supplied by objects from class controlSac
    Solves a hypo-elastic equation of the form: ds = Et * de.

tanMod(eVec, xVec, yVec)
        eVec        a vector of thermodynamic control variables  (strains)
        xVec        a vector of physical control variables       (stretches)
        yVec        a vector of thermodynamic response variables (stresses)

isRuptured()
    E.g.:  ruptured = ce.isRuptured()
        ruptured    a tuple of boolean results specifying if a specific
                    constituent has failed.  For a volume, there is no
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
        rho         returns the mass density of the septal Sac
        
stretch()
    E.g., U = ce.stretch()
        U           returns Laplace stretch in co-ordinate frame of volume

stretchInv()
    E.g., Uinv = ce.stretchInv()
        Uinv        returns the inverse of Laplace stretch

# absolute measures

temperature()
    E.g., temp = ce.temperature()
        temp        returns the temperature of the septal Sac

entropyDensity()
    E.g., eta = ce.entropyDensity()
        eta         returns the entropy density of the septal Sac

stressMtx()
    E.g., S = ce.stress()
        S           returns stress matrix

intensiveStressVec()
    E.g., T = ce.intensiveStressVec()
        T           returns stress vector T conjugate to strain vector E
        
# relative measures

relativeTemperature()
    E.g., temp = ce.relativeTemperature()
        temp        returns the relative temperature of the septal Sac

relativeEntropyDensity()
    E.g., eta = ce.relativeEntropyDensity()
        eta         returns the relative entropy density of the septal Sac

relativeStress()
    E.g., S = ce.relativeStress()
        S           returns relative stress 

References:
    1) Freed, A. D. and Rajagopal, K. R., “A Promising Approach for Modeling
       Biological Fibers,” ACTA Mechanica, 227 (2016), 1609-1619.
       DOI: 10.1007/s00707-016-1583-8.  Errata: DOI: 10.1007/s00707-018-2183-6
    2) Freed, A. D., Erel, V. and Moreno, M. R., “Conjugate Stress/Strain Base
       Pairs for the Analysis of Planar Biologic Tissues”, Journal of Mechanics
       of Materials and Structures, 12 (2017), 219-247.
       DOI: 10.2140/jomms.2017.12.219
"""


class controlSac(Control):

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
    #     xVec[0]  temperature                         'T'      (centigrade)
    #     xVec[1]  elongation in 1 direction           'a'      (dimensionless)
    #     xVec[2]  elongation in 2 direction           'b'      (dimensionless)
    #     xVec[3]  elongation in 3 direction           'c'      (dimensionless)
    #     xVec[4]  magnitude of shear in the 23 plane  'alp'    (dimensionless)
    #     xVec[5]  magnitude of shear in the 13 plane  'bet'    (dimensionless)
    #     xVec[6]  magnitude of shear in the 12 plane  'gam'  
    # while the thermodynamic control vector has strain components of:
    #     eVec[0]  thermal strain:   ln(T/T_0)                  (dimensionless)
    #     eVec[1]  dilation:         ln(a/a_0 * b/b_0 * c/c_0)  (dimensionless)
    #     eVec[2]  squeeze:          ln(a/a_0 * b_0/b)          (dimensionless)
    #     eVec[2]  squeeze:          ln(b/b_0 * c_0/c)          (dimensionless)
    #     eVec[3]  shear:            alp - alp_0                (dimensionless)
    #     eVec[3]  shear:            bet - bet_0                (dimensionless)
    #     eVec[3]  shear:            gam - gam_0                (dimensionless)
    def __init__(self, eVec0, xVec0, dt):
        # Call the constructor of the base type to create and initialize the
        # exported variables.
        super().__init__(eVec0, xVec0, dt)
        # Create and initialize any additional fields introduced by the user.
        if self.controls != 7:
            raise RuntimeError("There are 7 control variables for a 3D " +
                               "volume: temperature, three elongation " +
                               "ratios, and three shear.")
        return  # a new instance of type controlSac

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
        c = self.xN[3]
        dedxMtx[0, 0] = 1.0 / T
        dedxMtx[1, 1] = 1.0 / (3.0 * a)
        dedxMtx[1, 2] = 1.0 / (3.0 * b)
        dedxMtx[1, 3] = 1.0 / (3.0 * c)
        dedxMtx[2, 1] = 1.0 / (3.0 * a)
        dedxMtx[2, 2] = -1.0 / (3.0 * b)
        dedxMtx[3, 1] = 1.0 / (3.0 * b)
        dedxMtx[3, 2] = -1.0 / (3.0 * c)
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
        bN = self.xN[2]
        bC = self.xC[2]
        if self.node == 0:
            dxdtVec[4] *= bN / bC
            dxdtVec[5] *= aN / aC
            dxdtVec[6] *= aN / aC
        elif self.node == 1:
            dxdtVec[4] *= bC / bN
            dxdtVec[5] *= aC / aN
            dxdtVec[6] *= aC / aN
        else:
            aP = self.xP[1]
            bP = self.xP[2]
            alpN = self.xN[4]
            alpC = self.xC[4]
            alpP = self.xP[4]
            betN = self.xN[5]
            betC = self.xC[5]
            betP = self.xP[5]
            gamN = self.xN[6]
            gamC = self.xC[6]
            gamP = self.xP[6]
            dxdtVec[4] = (2.0 * (bC / bN) * (alpN - alpC) / self.dt -
                          (bP / bN) * (alpN - alpP) / (2.0 * self.dt))
            dxdtVec[5] = (2.0 * (aC / aN) * (betN - betC) / self.dt -
                          (aP / aN) * (betN - betP) / (2.0 * self.dt))
            dxdtVec[6] = (2.0 * (aC / aN) * (gamN - gamC) / self.dt -
                          (aP / aN) * (gamN - gamP) / (2.0 * self.dt))
        return dxdtVec


# constitutive class for biologic Sac

class ceSac(Response):
    #     xVec[0]  temperature                         'T'      (centigrade)
    #     xVec[1]  elongation in 1 direction           'a'      (dimensionless)
    #     xVec[2]  elongation in 2 direction           'b'      (dimensionless)
    #     xVec[3]  elongation in 3 direction           'c'      (dimensionless)
    #     xVec[4]  magnitude of shear in the 23 plane  'alp'    (dimensionless)
    #     xVec[5]  magnitude of shear in the 13 plane  'bet'    (dimensionless)
    #     xVec[6]  magnitude of shear in the 12 plane  'gam'  
    # while the thermodynamic control vector has strain components of:
    #     eVec[0]  thermal strain:   ln(T/T_0)                  (dimensionless)
    #     eVec[1]  dilation:         ln(a/a_0 * b/b_0 * c/c_0)  (dimensionless)
    #     eVec[2]  squeeze:          ln(a/a_0 * b_0/b)          (dimensionless)
    #     eVec[2]  squeeze:          ln(b/b_0 * c_0/c)          (dimensionless)
    #     eVec[3]  shear:            alp - alp_0                (dimensionless)
    #     eVec[3]  shear:            bet - bet_0                (dimensionless)
    #     eVec[3]  shear:            gam - gam_0                (dimensionless)
    # and the thermodynamic response vector has componenents of:
    #     yVec[0]  entropy density           'eta'                    (erg/g.K)
    #     yVec[1]  pressure                  'pi'                     (barye)
    #     yVec[2]  normal stress difference  'sigma1'                 (barye)
    #     yVec[3]  normal stress difference  'sigma2'                 (barye)
    #     yVec[4]  shear stress              'tau1'                   (barye)
    #     yVec[5]  shear stress              'tau2'                   (barye)
    #     yVec[6]  shear stress              'tau3'                   (barye)


    def __init__(self):
        # dimension the problem
        self.controls = 7
        self.responses = 7
        # get the material properties
        p_0, v0v = mp.septalSac()
        # assign these material properties to the object
        self.rho = mp.rhoAir()
        self.Cp = mp.CpAir()
        self.alpha = mp.alphaAir()
        self.v0v = v0v

        # establish the initial conditions for the thermodynamic responses
        yVec0 = np.zeros((self.responses,), dtype=float)
        yVec0[0] = mp.etaAir()    # initial entropy density
        yVec0[1] = -3 * p_0       # initial pressure                   'pi'                                  (barye)
        yVec0[2] = 0              # initial normal stress difference   'sigma1'                              (barye)
        yVec0[3] = 0              # initial normal stress difference   'sigma2'                              (barye)
        yVec0[4] = 0              # initial shear stress               'tau1'                                (barye)
        yVec0[5] = 0              # initial shear stress               'tau2'                                (barye)
        yVec0[6] = 0              # initial shear stress               'tau3' 

        # create and initialize the two control vectors
        eVec0 = np.zeros((self.controls,), dtype=float)
        xVec0 = np.zeros((self.controls,), dtype=float)

        # now call the base type to create the exported response fields
        super().__init__(eVec0, xVec0, yVec0)
        # set default value for voluse rupture
        self.ruptured = False
        # create vectors for the next node used in the output methods
        self.eN = np.zeros((self.controls,), dtype=float)
        self.xN = np.zeros((self.controls,), dtype=float)
        self.yN = np.zeros((self.responses,), dtype=float)
        return  # a new instance of this constitutive volume object

    def _K_secantModulus(self):
        
        p_0 = self.yR[1]
        # temperature
        T_0 = 37.0
        T = 273.0 + self.xN[0]   # convert Centigrade into Kelvin
        # bulk modulus
        kt = p_0 * (T / T_0) * self.v0v
        # The maximum bulk modulus
        kmax = 3.9E5
        # a volume can rupture under an excessive bulk modulus
        if kt > kmax:
            self.ruptured = True
        return kt
    
    def secMod(self, eVec, xVec, yVec):
        # call the base type to verify the inputs and to create matrix ceMtx
        Ms = super().secantModulus(eVec, xVec, yVec)
        k = self._K_secantModulus()
        Ms = np.zeros((self.responses - 1, self.controls - 1), dtype=float)
        Ms[0, 0] = 9 * k
        
        # update the exported vector fields
        self.eN[:] = eVec[:]
        self.xN[:] = xVec[:]
        self.yN[:] = yVec[:]
        return Ms    

    def tanMod(self, eVec, xVec, yVec):
        # call the base type to verify the inputs and to create matrix ceMtx
        Mt = super().secantModulus(eVec, xVec, yVec)
        k = self._K_secantModulus()
        Mt = np.zeros((self.responses - 1, self.controls - 1), dtype=float)
        Mt[0, 0] = 9 * k

        # update the exported vector fields
        self.eN[:] = eVec[:]
        self.xN[:] = xVec[:]
        self.yN[:] = yVec[:]
        return Mt  
    
    def isRuptured(self):
        if self.firstCall:
            hasRuptured = (False,)
        elif not self.ruptured:
            hasRuptured = super().isRuptured()
        else:
            hasRuptured = (True,)
        return hasRuptured

    # additional methods

    def massDensity(self):
        return self.rho

    def stretch(self):
        U = np.zeros((3, 3), dtype=float)
        if self.firstCall:
            a = self.xR[1]
            b = self.xR[2]
            c = self.xR[3]
            alp = self.xR[4]
            bet = self.xR[5]
            gam = self.xR[6]
        else:
            a = self.xN[1]
            b = self.xN[2]
            c = self.xN[3]
            alp = self.xN[4]
            bet = self.xN[5]
            gam = self.xN[6]
        U[0, 0] = a
        U[0, 1] = a * gam
        U[0, 2] = a * bet
        U[1, 1] = b
        U[1, 2] = b * alp
        U[2, 2] = c
        return U

    def stretchInv(self):
        Uinv = np.zeros((2, 2), dtype=float)
        if self.firstCall:
            a = self.xR[1]
            b = self.xR[2]
            c = self.xR[3]
            alp = self.xR[4]
            bet = self.xR[5]
            gam = self.xR[6]
        else:
            a = self.xN[1]
            b = self.xN[2]
            c = self.xN[3]
            alp = self.xN[4]
            bet = self.xN[5]
            gam = self.xN[6]       
        
        Uinv[0, 0] = 1.0 / a
        Uinv[0, 1] = -gam / b
        Uinv[0, 2] = -(bet - alp * gam) / c
        Uinv[1, 1] = 1.0 / b
        Uinv[1, 2] = -alp / c
        Uinv[2, 2] = 1.0 / c
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
            eta = mp.etaAir()
        else:
            eta = self.yN[0]
        return eta

    def stressMtx(self):
        if self.firstCall:
            a = self.xR[1]
            b = self.xR[2]
            c = self.xR[3]
            alp = self.xR[4]
            pi = self.yR[1]
            sigma1 = self.yR[2]
            sigma2 = self.yR[3]
            tau1 = self.yR[4]
            tau2 = self.yR[5]
            tau3 = self.yR[6]
        else:
            a = self.xN[1]
            b = self.xN[2]
            c = self.xN[3]
            alp = self.xN[4]
            pi = self.yN[1]
            sigma1 = self.yN[2]
            sigma2 = self.yN[3]
            tau1 = self.yN[4]
            tau2 = self.yN[5]
            tau3 = self.yN[6]
        s = np.zeros((3, 3), dtype=float)
        s[0, 0] = (2 * sigma1 + sigma2 + pi) / 3
        s[0, 1] = (b * (tau3 + alp * tau2) )/ a
        s[0, 2] = (c * tau2)  / a
        s[1, 0] = s[0, 1]
        s[1, 1] = (-sigma1 + sigma2 + pi) / 3
        s[1, 2] = (c * tau1) / b
        s[2, 0] = s[0, 2]
        s[2, 1] = s[1, 2]
        s[2, 2] = (pi - sigma1 - 2 * sigma2) / 3
        return s

    def intensiveStressVec(self):
        if self.firstCall:
            pi = self.yR[1]
            sigma1 = self.yR[2]
            sigma2 = self.yR[3]
            tau1 = self.yR[4]
            tau2 = self.yR[5]
            tau3 = self.yR[6]
        else:
            pi = self.yN[1]
            sigma1 = self.yN[2]
            sigma2 = self.yN[3]
            tau1 = self.yN[4]
            tau2 = self.yN[5]
            tau3 = self.yN[6]
        T = np.zeros((6, 1), dtype=float)
        T[0, 0] = pi
        T[1, 0] = sigma1
        T[2, 0] = sigma2
        T[3, 0] = tau1
        T[4, 0] = tau2
        T[5, 0] = tau3
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
        s = np.zeros((3, 3), dtype=float)
        if not self.firstCall:
            a = self.xN[1]
            b = self.xN[2]
            c = self.xN[3]
            alp = self.xN[4]
            pi = self.yN[1] - self.yR[1]
            sigma1 = self.yN[2]
            sigma2 = self.yN[3]
            tau1 = self.yN[4]
            tau2 = self.yN[5]
            tau3 = self.yN[6]
        s = np.zeros((3, 3), dtype=float)
        s[0, 0] = (2 * sigma1 + sigma2 + pi) / 3
        s[0, 1] = (b * (tau3 + alp * tau2)) / a
        s[0, 2] = (c * tau2)  / a
        s[1, 0] = s[0, 1]
        s[1, 1] = (-sigma1 + sigma2 + pi) / 3
        s[1, 2] = (c * tau1)/ b
        s[2, 0] = s[0, 2]
        s[2, 1] = s[1, 2]
        s[2, 2] = (pi - sigma1 - 2 * sigma2) / 3
        
        return s


"""
Changes made in version "1.0.0":

This is the initial version.
"""
