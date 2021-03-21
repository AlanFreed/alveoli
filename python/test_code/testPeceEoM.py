#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from peceEoM import PECE
# for making plots
from matplotlib import pyplot as plt
from pylab import rcParams
from pivotIncomingF import Pivot
from dodecahedra import dodecahedron


r"""
    author:  Prof. Alan Freed, Texas A&M University
    date:    Tue Jun 20, 2017
    update:  Mon May 18, 2020

    To illustrate this class of problems, consider a vibration model for a car
    in three degrees of freedom: bounce, pitch and roll, all measured at the
    center of gravity of a car.  This example simulates an FSAE race car.

    x = {b, p, r}^T   where  b = bounce,  p = pitch,  r = roll
    v = {db, dp, dr}^T
    a = {d2b, d2p, d2r}^T  and this is given by the equation:

    a = M^{-1} [fFn(t) - C*v - K*x]

    Bounce is in feet, while pitch and roll are in radians, per FSAE rules.
    Bounce is positive downward (towards the ground).  Pitch is positive when
    the nose is up and the tail is down.  Roll is positive when the driver is
    up and the passenger is down.

    The mass matrix  M  for this 3 degree-of-freedom (DOF) problem is
            / m   0  0  \                        / 1/m  0    0   \
        M = | 0  Jy  0  |    so that    M^{-1} = |  0  1/Jy      |
            \ 0   0  Jx /                        \  0   0   1/Jx /
    where  m  is the mass of the vehicle in slugs, while  Jx  and  Jy  are the
    moments of inertia in units of  ft.lbs/(rad/sec^2)  about the x and y axes,
    per FSAE rules.

    The symmetric damping matrix  C  for this 3 DOF car simulation is
            / c11 c12 c13 \
        C = | c12 c22 c23 |
            \ c13 c23 c33 /
    wherein
        c11 = c1 + c2 + c3 + c4
        c12 = −(c1 + c2) lf + (c3 + c4) lr
        c13 = −(c1 − c2) rf + (c3 − c4) rr
        c22 = (c1 + c2) lf^2 + (c3 + c4) lr^2
        c23 = -(c1 − c2) lf rf + (c3 − c4) lr rr
        c33 = (c1 + c2) rf^2 + (c3 + c4) rr^2
    where  c1  is the damping of the driver front shock absorber,  c2  is the
    damping of the passenger front shock absorber,  c3  is the damping of the
    passenger rear shock absorber,  c4  is the damping of the driver rear
    shock absorber, all of which have units of lb/(ft/sec).  Parameter  lf
    is the distance from the center of gravity (CG) to the front axle,  lr
    is the distance from the CG to the rear axle,  rf  is the radial distance
    from the axial centerline (CL) to the center of the tire patch at the front
    axle, and  rr  is the radial distance from the CL to the center of the tire
    patch at the rear axle, with distances being in feet, per FSAE rules.

    The symmetric stiffness matrix  K  for this 3 DOF car simulation is
            / k11 k12 k13 \
        K = | k12 k22 k23 |
            \ k13 k23 k33 /
    wherein
        k11 = k1 + k2 + k3 + k4
        k12 = −(k1 + k2) lf + (k3 + k4) lr
        k13 = −(k1 − k2) rf + (k3 − k4) rr
        k22 = (k1 + k2) lf^2 + (k3 + k4) lr^2
        k23 = -(k1 − k2) lf rf + (k3 − k4) lr rr
        k33 = (k1 + k2) rf^2 + (k3 + k4) rr^2
    where  k1  is the stiffness of the driver front spring,  k2  is the
    stiffness of the passenger front spring,  k3  is the stiffness of the
    passenger rear spring,  k4  is the stiffness of the driver rear spring,
    all of which have units of lb/ft, per FSAE rules.  The other parameters
    are as defined for the damping matrix.

    The forcing function  fFn  for thie 3 DOF car simulation is
              / f1 \
        fFn = | f2 |
              \ f3 /
    wherein
       f1 = w − c1 Ṙ1 − c2 Ṙ2 − c3 Ṙ3 − c4 Ṙ4
          −  k1 R1 − k2 R2 − k3 R3 − k4 R4
       f2 = (c1 Ṙ1 + c2 Ṙ2 + k1 R1 + k2 R2) lf
          − (c3 Ṙ3 + c4 Ṙ4 + k3 R3 + k4 R4) lr
       f3 = (c1 Ṙ1 − c2 Ṙ2 + k1 R1 − k2 R2) rf
          − (c3 Ṙ3 − c4 Ṙ4 + k3 R3 − k4 R4) rr
    where  w  is the weight of the car in pounds,  R1, R2, R3, R4  are the
    upward displacements of the roadway, which are functions of time, and
    Ṙ1, Ṙ2, Ṙ3, Ṙ4  are their rates of change, which are also functions of
    time.  Units are in ft and ft/sec, respectively.  The other parameters
    are as defined for the damping and stiffness matrices.

    Representative parameters for a typical FSAE race car with driver are:
        m = 14       in slugs
        w = 450      in lbs
        Jx = 20      in ft.lbs/(rad/sec^2)
        Jy = 45      in ft.lbs/(rad/sec^2)
        lf = 3.2     in ft
        lr = 1.8     in ft
        rf = 2.1     in ft
        rr = 2       in ft
        c1 = 10      in lbs/(in/sec)
        c2 = 10      in lbs/(in/sec)
        c3 = 15      in lbs/(in/sec)
        c4 = 15      in lbs/(in/sec)
        k1 = 150     in lbs/in
        k2 = 150     in lbs/in
        k3 = 300     in lbs/in
        k4 = 300     in lbs/in
"""


def bump(height, length, top, x, v):
    # geometric properties of the bump: dimensions are in ft
    height = 1. / 6.   # height of the bump
    length = 2.        # length of the bump
    top = 0.5          # length of flat region on top of bump

    # a haversine bump
    if (x <= 0.) or (x >= length):
        # located either ahead of or behing the bump
        R = 0.
        dR = 0.
    else:
        # locate where your position is on the bump
        if x < (length - top) / 2.:
            phi = 2. * math.pi * x / (length - top)
        elif x < (length + top) / 2.:
            phi = math.pi
        else:
            phi = 2 * math.pi * (x - top) / (length - top)
        R = (1. - math.cos(phi)) * height / 2.
        dR = (math.pi * height / (length - top)) * math.sin(phi) * v

    return R, dR


def trajectory(t):
    # consider constant speed
    mph2fps = 1.467
    speed = 10 * mph2fps

    x = speed * t
    v = speed
    return x, v


def mogul(position, speed):
    # properties of the mogul field
    height = 0.25     # height of each bump in the moguls
    length = 5.       # wavelength of each bump
    top = 2.          # length of flat reagon on top of each bump

    if position >= 0. and position < length:
        location = position
    elif position >= length and position < 2. * length:
        location = position - length
    elif position >= 2. * length and position < 3. * length:
        location = position - 2. * length
    elif position >= 3. * length and position < 4. * length:
        location = position - 3. * length
    elif position >= 4. * length and position < 5. * length:
        location = position - 4. * length
    else:
        location = -1.0
    R, dR = bump(height, length, top, location, speed)
    return R, dR


def roadwayDF(t):
    position, speed = trajectory(t)
    return mogul(position, speed)


def roadwayPF(t):
    offset = 0.5         # distance passenger side trails the driver's side

    position, speed = trajectory(t)
    position = position - offset
    return mogul(position, speed)


def roadwayPR(t):
    offset = 0.5         # distance passenger side trails the driver's side
    wheelbase = 6.       # distance rear axle is behind the front axle

    position, speed = trajectory(t)
    position = position - wheelbase - offset
    return mogul(position, speed)


def roadwayDR(t):
    wheelbase = 6.       # distance rear axle is behind the front axle

    position, speed = trajectory(t)
    position = position - wheelbase
    return mogul(position, speed)


class FSAE(object):

    # asign the parameters that define a car

    m = 14      # mass in slugs
    w = 450     # weight in lbs
    Jx = 20     # moment of inertia resisting roll  in ft.lbs/(rad/sec^2)
    Jy = 45     # moment of inertia resisting pitch in ft.lbs/(rad/sec^2)
    lf = 3.2    # distance from CG to front axle in ft
    lr = 1.8    # distance from CG to rear  axle in ft
    rf = 2.1    # distance from CL to center tire patch at front axle in ft
    rr = 2      # distance from CL to center tire patch at rear  axle in ft
    c1 = 120    # driver front damping from shock absorber in lbs/(ft/sec)
    c2 = 120    # passenger front damping from shock absorber in lbs/(ft/sec)
    c3 = 180    # passenger rear damping from shock absorber in lbs/(ft/sec)
    c4 = 180    # driver rear damping from shock absorber in lbs/(ft/sec)
    k1 = 1800   # driver front spring stiffness in lbs/ft
    k2 = 1800   # passenger front spring stiffness in lbs/ft
    k3 = 3600   # passenger rear spring stiffness in lbs/ft
    k4 = 3600   # driver rear spring stiffness in lbs/ft

    def __init__(self):
        return  # new instance of class FSAE

    def getMInv(self):
        # the inverse of the mass matrix
        MInv = np.asmatrix(np.array([[1./self.m, 0., 0.],
                                     [0., 1./self.Jy, 0.],
                                     [0., 0., 1./self.Jx]]))
        return MInv

    def getC(self):
        # the damping matrix
        c11 = self.c1 + self.c2 + self.c3 + self.c4
        c12 = -(self.c1 + self.c2) * self.lf + (self.c3 + self.c4) * self.lr
        c13 = -(self.c1 - self.c2) * self.rf + (self.c3 - self.c4) * self.rr
        c22 = ((self.c1 + self.c2) * self.lf**2 +
               (self.c3 + self.c4) * self.lr**2)
        c23 = (-(self.c1 - self.c2) * self.lf * self.rf +
               (self.c3 - self.c4) * self.lr * self.rr)
        c33 = ((self.c1 + self.c2) * self.rf**2 +
               (self.c3 + self.c4) * self.rr**2)
        C = np.asmatrix(np.array([[c11, c12, c13],
                                  [c12, c22, c23],
                                  [c13, c23, c33]]))
        return C

    def getK(self):
        # the stiffness matrix
        k11 = self.k1 + self.k2 + self.k3 + self.k4
        k12 = -(self.k1 + self.k2) * self.lf + (self.k3 + self.k4) * self.lr
        k13 = -(self.k1 - self.k2) * self.rf + (self.k3 - self.k4) * self.rr
        k22 = ((self.k1 + self.k2) * self.lf**2 +
               (self.k3 + self.k4) * self.lr**2)
        k23 = (-(self.k1 - self.k2) * self.lf * self.rf +
               (self.k3 - self.k4) * self.lr * self.rr)
        k33 = ((self.k1 + self.k2) * self.rf**2 +
               (self.k3 + self.k4) * self.rr**2)
        K = np.asmatrix(np.array([[k11, k12, k13],
                                  [k12, k22, k23],
                                  [k13, k23, k33]]))
        return K

    def getF(self, t):
        R1, dR1 = roadwayDF(t)
        R2, dR2 = roadwayPF(t)
        R3, dR3 = roadwayPR(t)
        R4, dR4 = roadwayDR(t)
        f1 = (self.w - self.c1 * dR1 - self.c2 * dR2 - self.c3 * dR3 -
              self.c4 * dR4 - self.k1 * R1 - self.k2 * R2 - self.k3 * R3 -
              self.k4 * R4)
        f2 = ((self.c1 * dR1 + self.c2 * dR2 + self.k1 * R1 +
              self.k2 * R2) * self.lf - (self.c3 * dR3 + self.c4 * dR4 +
              self.k3 * R3 + self.k4 * R4) * self.lr)
        f3 = ((self.c1 * dR1 - self.c2 * dR2 + self.k1 * R1 -
              self.k2 * R2) * self.rf - (self.c3 * dR3 - self.c4 * dR4 +
              self.k3 * R3 - self.k4 * R4) * self.rr)
        fFn = np.asmatrix(np.array([[f1],
                                    [f2],
                                    [f3]]))
        return fFn

    def getICs(self):
        f0 = np.asmatrix(np.array([self.w, 0., 0.])).T
        K = self.getK()
        x = K.I * f0
        x0 = np.array([x[0, 0], x[1, 0], x[2, 0]])
        v0 = np.array([0., 0., 0.])
        return x0, v0

    def getA(self, t, X, V, p, pi):
        MInv = self.getMInv()
        C = self.getC()
        V = np.asmatrix(V).T
        Cv = C * V
        K = self.getK()
        X = np.asmatrix(X).T
        Kx = K * X
        f = self.getF(t)
        Amtx = MInv * (np.subtract(f, np.add(Cv, Kx)))
        A = np.array([Amtx[0, 0], Amtx[1, 0], Amtx[2, 0]])
        return A


def test():
    # properties for the integrator
    N = 500        # number of global steps
    T = 2.5        # time at the end of the run/analysis
    h = T / N      # global step size
    
    
    
       # impose a far-field deformation history        
    F0 = np.eye(3, dtype=float)
    F1 = np.copy(F0)
    F2 = np.copy(F1)
    F3 = np.copy(F2)
    
        
    # re-index the co-ordinate systems according to pivot in pivotIncomingF.py
    pi = Pivot(F0)
    pi.update(F1)
    pi.advance()
    pi.update(F2)
    pi.advance()
    pi.update(F3)  
    piF0 = pi.pivotedF('ref')
    
    
    d = dodecahedron(piF0)
    
    # pentagon 4
    p = d.getPentagon(4) 
    

    car = FSAE()
    # establish the initial state
    t0 = 0.
    x0, v0 = car.getICs()
    a0 = car.getA(t0, x0, v0, p, pi)

    print("")
    print("Static deflection is:")
    print("  z     = {:7.4f} inches.".format(12. * x0[0]))
    print("  theta = {:7.4f} degrees.".format((180. / math.pi) * x0[1]))
    print("  phi   = {:7.4f} degrees.".format((180. / math.pi) * x0[2]))

    solver = PECE(x0, v0, t0, h, car.getA, p, pi)

    resultsE = []
    resultsH = []
    resultsP = []
    resultsR = []
    resultsE.append((t0, 0.0))               # time and error
    resultsH.append((0., v0[0], a0[0]))      # heave
    resultsP.append((x0[1], v0[1], a0[1]))   # pitch
    resultsR.append((x0[2], v0[2], a0[2]))   # roll

    for n in range(N):
        solver.integrate(p, pi)
        solver.advance()
        x = solver.getU()
        v = solver.getV()
        a = solver.getA()
        resultsE.append((solver.getT(), solver.getError()))
        resultsH.append((12 * x[0], v[0], a[0]))
        resultsP.append(((180. / math.pi) * x[1], v[1], a[1]))
        resultsR.append(((180. / math.pi) * x[2], v[2], a[2]))

    # create the heave plot
    time = np.array([z[0] for z in resultsE])
    heave = np.array([z[0] for z in resultsH])
    heave[0] = heave[1]

    plt.figure(1)
    rcParams["figure.figsize"] = 21, 5
    ax = plt.subplot(1, 3, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(time, heave, 'k-', linewidth=2)

    plt.title("Heave", fontsize=20)
    plt.xlabel('time,  $t$  (seconds)', fontsize=16)
    plt.ylabel('bounce,  $z$  (inches)', fontsize=16)
    plt.legend([line1],
               ["heave"], fontsize=14,
               bbox_to_anchor=(0.65, 0.2), loc=2, borderaxespad=0.)

    # create the pitch and roll plot
    time = np.array([z[0] for z in resultsE])
    pitch = np.array([z[0] for z in resultsP])
    pitch[0] = pitch[1]
    roll = np.array([z[0] for z in resultsR])
    roll[0] = roll[1]

    ax = plt.subplot(1, 3, 2)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(time, pitch, 'r-', linewidth=2)
    line2, = ax.plot(time, roll, 'b-', linewidth=2)

    plt.title("Pitch and Roll", fontsize=20)
    plt.xlabel('time,  $t$  (seconds)', fontsize=16)
    plt.ylabel('rotations,  $theta$, $phi$  (degrees)', fontsize=16)
    plt.legend([line1, line2],
               ["pitch", "roll"], fontsize=14,
               bbox_to_anchor=(0.7, 0.3), loc=2, borderaxespad=0.)

    # create the error plot
    time = np.array([z[0] for z in resultsE])
    error = np.array([z[1] for z in resultsE])
    error[0] = error[1]

    ax = plt.subplot(1, 3, 3)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    ax.set_yscale('log')
    line1, = ax.plot(time, error, 'k-', linewidth=2)

    plt.title("Local Truncation Error", fontsize=20)
    plt.xlabel('time,  $t$  (seconds)', fontsize=16)
    plt.ylabel(r'error,  $\epsilon$', fontsize=16)
    plt.legend([line1],
               ["error"], fontsize=14,
               bbox_to_anchor=(0.7, 0.9), loc=2, borderaxespad=0.)
    plt.savefig('fsae')
    plt.show()


test()
