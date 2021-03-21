#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chords import Chord
import math as m
import numpy as np
from pentagons import pentagon
from peceAtoVandX import pece
from pivotIncomingF import Pivot
from vertices import Vertex
# for making plots
from matplotlib import pyplot as plt
from matplotlib import rc
from math import log




class lung(object):

    def __init__(self):
        return  # new instance of object fsae

    def getMInv(self, p):
        # the inverse of the mass matrix
        M = p.massMatrix()
        MInv = np.linalg.inv(M)
        return MInv

    def getM(self, p):
        # the inverse of the mass matrix
        M = p.massMatrix()
        return M

    def getC(self, p):
        # the damping matrix
        C = p.tangentStiffnessMtxC()
        return C

    def getK(self, p, pi):
        # the secant stiffness matrix
        K = p.secantStiffnessMtxK(pi)
        return K

    def getF(self, t, p):
        # the force vector
        fFn = p.forcingFunction()
        return fFn

    def getICs(self, p, pi):
        f0 = np.asmatrix(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])).T
        K = self.getK(p, pi)
        x = np.linalg.inv(K) * f0
        x0 = np.array([x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0], x[5, 0],
                       x[6, 0], x[7, 0], x[8, 0], x[9, 0]])
        v0 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        return x0, v0

    def getA(self, t, X, V, p, pi):
        MInv = self.getMInv(p)
        C = self.getC(p)
        V = np.asmatrix(V).T
        Cv = C * V
        K = self.getK(p, pi)
        X = np.asmatrix(X).T
        Kx = K * X
        f = self.getF(t, p)
        Amtx = MInv * (np.subtract(f, np.add(Cv, Kx)))
        A = np.array([Amtx[0, 0], Amtx[1, 0], Amtx[2, 0], Amtx[3, 0],
                      Amtx[4, 0], Amtx[5, 0], Amtx[6, 0], Amtx[7, 0],
                      Amtx[8, 0], Amtx[9, 0]])
        return A


def test():
    # properties for the integrator
    tol = 0.0001   # upper bound on the local truncation error
    N = 180        # number of global steps
    T = 1.0E-6     # time at the end of the run/analysis
    h = T / N      # global step size
    t = 0.
    gaussPts = 5

    sep = lung()

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

    v1_0 = np.array([m.cos(m.pi/2), m.sin(m.pi/2), 0.0])
    v2_0 = np.array([m.cos(9*m.pi/10), m.sin(9*m.pi/10), 0.0])
    v3_0 = np.array([m.cos(13*m.pi/10), m.sin(13*m.pi/10), 0.0])
    v4_0 = np.array([m.cos(17*m.pi/10), m.sin(17*m.pi/10), 0.0])
    v5_0 = np.array([m.cos(21*m.pi/10), m.sin(21*m.pi/10), 0.0])

    pv1_0 = np.matmul(piF0, v1_0)
    pv2_0 = np.matmul(piF0, v2_0)
    pv3_0 = np.matmul(piF0, v3_0)
    pv4_0 = np.matmul(piF0, v4_0)
    pv5_0 = np.matmul(piF0, v5_0)

    # assign the vertices for pentagon 1 in the dodecahedron
    v1 = Vertex(1, (pv1_0[0], pv1_0[1], pv1_0[2]), h)
    v2 = Vertex(2, (pv2_0[0], pv2_0[1], pv2_0[2]), h)
    v3 = Vertex(3, (pv3_0[0], pv3_0[1], pv3_0[2]), h)
    v4 = Vertex(4, (pv4_0[0], pv4_0[1], pv4_0[2]), h)
    v5 = Vertex(5, (pv5_0[0], pv5_0[1], pv5_0[2]), h)

    # assign the cords for a pentagon that inscribes an unit circle
    c1 = Chord(1, v5, v1, h, tol)
    c2 = Chord(2, v1, v2, h, tol)
    c3 = Chord(3, v2, v3, h, tol)
    c4 = Chord(6, v3, v4, h, tol)
    c5 = Chord(7, v4, v5, h, tol)  
    
    # create the pentagon
    p = pentagon(1, c1, c2, c3, c4, c5, h)

    # establish the initial state
    x0, v0 = sep.getICs(p, pi)
    x0[4] = 0.0
    x0[5] = 0.0
    x0[7] = 0.0
    a0 = sep.getA(t, x0, v0, p, pi)

    solver = pece(sep.getA, t, x0, v0, h, p, pi, tol)

    M0 = sep.getM(p)
    K0 = sep.getK(p, pi)
    
    # force
    F0I = np.dot(M0, a0)
    F0C = np.dot(K0, x0)
    F0T = np.dot(M0, a0) + np.dot(K0, x0)  

    resultsX1 = []
    resultsV1 = []
    resultsA1 = []
    resultsFI = []
    resultsFC = []
    resultsFT = []
    resultsT = []
    resultsE1 = []
    resultsPif11 = []

    resultsX1.append((x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7],
                      x0[8], x0[9]))
    resultsV1.append((v0[0], v0[1], v0[2], v0[3], v0[4], v0[5], v0[6], v0[7],
                      v0[8], v0[9]))
    resultsA1.append((a0[0], a0[1], a0[2], a0[3], a0[4], a0[5], a0[6], a0[7],
                      a0[8], a0[9]))
    resultsFI.append((F0I[0], F0I[1], F0I[2], F0I[3], F0I[4], F0I[5], F0I[6],
                      F0I[7], F0I[8], F0I[9]))
    resultsFC.append((F0C[0], F0C[1], F0C[2], F0C[3], F0C[4], F0C[5], F0C[6],
                      F0C[7], F0C[8], F0C[9]))
    resultsFT.append((F0T[0], F0T[1], F0T[2], F0T[3], F0T[4], F0T[5], F0T[6],
                      F0T[7], F0T[8], F0T[9]))

    resultsE1.append((tol))          # error
    resultsT.append((t))           # time
    resultsPif11.append((piF0[1, 1]))     # deformation grdient
    
    resultsSt11 = []
    resultsSt22 = []
  
    dilation = []
    epsilon1 = []
    pureShear = []
    dDilation = []
    dEpsilon = []
    
    
    
    for n in range(1, N):  

        
        # stress at each gauss point
        s1 = p.Ss[1]
        s2 = p.Ss[2]
        s3 = p.Ss[3]
        s4 = p.Ss[4]
        s5 = p.Ss[5]
        
        resultsSt11.append((s1[0, 0], s2[0, 0], s3[0, 0], s4[0, 0], s5[0, 0]))
        resultsSt22.append((s1[1, 1], s2[1, 1], s3[1, 1], s4[1, 1], s5[1, 1]))
        
        
        solver.integrate(p, pi)
        solver.advance(p, pi)
        x = solver.getX()
        x[4] = 0.0
        x[5] = 0.0
        x[7] = 0.0
        v = solver.getV()
        a = solver.getA()
        t = solver.getT()
        e = solver.getError()

        M = sep.getM(p)
        K = sep.getK(p, pi)

        FI = np.dot(M, a)
        FC = np.dot(K, x)
        FT = np.dot(M, a) + np.dot(K, x)

    
        resultsX1.append((x[0], x[1], x[2], x[3], x[4], x[5], x[6],
                          x[7], x[8], x[9]))
        resultsV1.append((v[0], v[1], v[2], v[3], v[4], v[5], v[6],
                          v[7], v[8], v[9]))
        resultsA1.append((a[0], a[1], a[2], a[3], a[4], a[5], a[6],
                          a[7], a[8], a[9]))
        resultsFI.append((FI[0], FI[1], FI[2], FI[3], FI[4], FI[5], FI[6],
                          FI[7], FI[8], FI[9]))
        resultsFC.append((FC[0], FC[1], FC[2], FC[3], FC[4], FC[5], FC[6],
                          FC[7], FC[8], FC[9]))
        resultsFT.append((FT[0], FT[1], FT[2], FT[3], FT[4], FT[5], FT[6],
                          FT[7], FT[8], FT[9]))

        resultsE1.append((e))
        resultsT.append((t))

        # thermodynamic strains
        di = p.dilationSingleComp(gaussPts, 'curr')
        ep = p.squeezeSingleComp(gaussPts, 'curr')    

        # thermodynamic strain rates
        dDi = p.dDilationSingleComp(gaussPts, 'curr')
        dEp = p.dSqueezeSingleComp(gaussPts, 'curr')


        # thermodynamic strains
        dilation.append(di)
        epsilon1.append(ep)  

        # thermodynamic strain rates
        dDilation.append(dDi)
        dEpsilon.append(dEp)
        
        
        # # compression
        # if t < T/3 :
        #     piF0[0, 0] = 1
        #     piF0[1, 1] -= 11.5 * (t / T)**5        
        #     piF0[2, 2] = 1
        # # expansion
        # else:
        #     piF0[0, 0] = 1
        #     piF0[1, 1] += 0.025 * (t / T)**5      
        #     piF0[2, 2] = 1


        # compression
        if t < T/3 :
            piF0[0, 0] = 1
            piF0[1, 1] -= 11.5 * (t / T)**5        
            piF0[2, 2] = 1
        # expansion
        else:
            piF0[0, 0] = 1
            piF0[1, 1] += 0.0022 * (T / t)**2      
            piF0[2, 2] = 1

             
        ps= log((piF0[0, 0] / piF0[1, 1])**(1.0/3.0))
        pureShear.append(ps)

        resultsPif11.append((piF0[1, 1]))

        pv1_n = np.matmul(piF0, v1_0)
        pv2_n = np.matmul(piF0, v2_0)
        pv3_n = np.matmul(piF0, v3_0)
        pv4_n = np.matmul(piF0, v4_0)
        pv5_n = np.matmul(piF0, v5_0)

        # assign the vertices for pentagon 1 in the dodecahedron
        v1 = Vertex(1, (pv1_n[0], pv1_n[1], pv1_n[2]), h)
        v2 = Vertex(2, (pv2_n[0], pv2_n[1], pv2_n[2]), h)
        v3 = Vertex(3, (pv3_n[0], pv3_n[1], pv3_n[2]), h)
        v4 = Vertex(4, (pv4_n[0], pv4_n[1], pv4_n[2]), h)
        v5 = Vertex(5, (pv5_n[0], pv5_n[1], pv5_n[2]), h)

        # assign the cords for a pentagon that inscribes an unit circle
        c1 = Chord(1, v5, v1, h, tol)
        c2 = Chord(2, v1, v2, h, tol)
        c3 = Chord(3, v2, v3, h, tol)
        c4 = Chord(6, v3, v4, h, tol)
        c5 = Chord(7, v4, v5, h, tol)
        
        # create the pentagon
        p = pentagon(1, c1, c2, c3, c4, c5, h)
        p.update()
        p.advance(pi)


    dis1x = np.array([z[:] for z in resultsX1])
    displacement1x = dis1x[:, 0]

    dis1y = np.array([z[:] for z in resultsX1])
    displacement1y = dis1y[:, 1]
    
    vel1x = np.array([z[:] for z in resultsV1])
    velocity1x = vel1x[:, 0]

    vel1y = np.array([z[:] for z in resultsV1])
    velocity1y = vel1y[:, 1]
    
    ac1x = np.array([z[:] for z in resultsA1])
    acceleration1x = ac1x[:, 0]

    ac1y = np.array([z[:] for z in resultsA1])
    acceleration1y = ac1y[:, 1]

    frIx = np.array([z[:] for z in resultsFI])
    forceIx = frIx[:, 0]

    frCx = np.array([z[:] for z in resultsFC])
    forceCx = frCx[:, 0]

    frTx = np.array([z[:] for z in resultsFT])
    forceTx = frTx[:, 0]



    st11G1 = np.array([z[:] for z in resultsSt11])
    stress11G1 = st11G1[:, 0]

    st22G1 = np.array([z[:] for z in resultsSt22])
    stress22G1 = st22G1[:, 0]
    
    
    
    st11G2 = np.array([z[:] for z in resultsSt11])
    stress11G2 = st11G2[:, 1]

    st22G2 = np.array([z[:] for z in resultsSt22])
    stress22G2 = st22G2[:, 1]



    st11G3 = np.array([z[:] for z in resultsSt11])
    stress11G3 = st11G3[:, 2]

    st22G3 = np.array([z[:] for z in resultsSt22])
    stress22G3 = st22G3[:, 2]



    st11G4 = np.array([z[:] for z in resultsSt11])
    stress11G4 = st11G4[:, 3]

    st22G4 = np.array([z[:] for z in resultsSt22])
    stress22G4 = st22G4[:, 3]



    st11G5 = np.array([z[:] for z in resultsSt11])
    stress11G5 = st11G5[:, 4]

    st22G5 = np.array([z[:] for z in resultsSt22])
    stress22G5 = st22G5[:, 4]

    
    
    time = resultsT
    error = resultsE1
    PiF11 = resultsPif11

    print("")
    print("The FSAE race car ran this course with statistics:")
    n, n_d, n_h, n_r = solver.getStatistics()
    print("   {} steps with {} restarts".format(n, n_r))
    print("   of which {} steps were doubled".format(n_d))
    print("   and {} steps were halved.".format(n_h))
    print("")

    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)


    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(time, PiF11, 'b-', linewidth=2)
    plt.xlabel('Time, t (s)', fontsize=16)
    plt.ylabel('F11  ', fontsize=16)
    plt.savefig('F11')
    plt.show()



    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(time, displacement1x, 'k-', linewidth=2)
    line2, = ax.plot(time, displacement1y, 'b-', linewidth=2)

    plt.xlabel('Time, t (s)', fontsize=16)
    plt.ylabel('Displacement, u (cm) ', fontsize=16)
    plt.legend([line1, line2],
                ["ux", "uy"],
                bbox_to_anchor=(0, 1), loc='upper left', fontsize=8)
    plt.savefig('displacement')
    plt.show()



    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(time, velocity1x, 'k-', linewidth=2)
    line2, = ax.plot(time, velocity1y, 'b-', linewidth=2)

    plt.xlabel('Time, t  (s)', fontsize=16)
    plt.ylabel('Velocity, v  $(cm.s^{-1})$ ', fontsize=16)
    plt.legend([line1, line2],
                ["vx", "vy"],
                bbox_to_anchor=(0, 1), loc='upper left', fontsize=8)
    plt.savefig('velocity')
    plt.show()



    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(time, acceleration1x, 'k-', linewidth=2)
    line2, = ax.plot(time, acceleration1y, 'b-', linewidth=2)

    plt.xlabel('Time, t (s)', fontsize=16)
    plt.ylabel('Acceleration, a  $(cm.s^{-2})$', fontsize=16)
    plt.legend([line1, line2],
                ["ax", "ay"],
                bbox_to_anchor=(0, 1), loc='upper left', fontsize=8)
    plt.savefig('acceleration')
    plt.show()



    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(time, forceIx, 'k-', linewidth=2)
    line2, = ax.plot(time, forceCx, 'b-', linewidth=2)
    line3, = ax.plot(time, forceTx, 'r-', linewidth=2)
    plt.legend([line1, line2, line3],
                ["Inertia Force", "Constitutive Force", "Total Force"],
                bbox_to_anchor=(0, 1), loc='upper left', fontsize=8)
    plt.xlabel('Time, t (s)', fontsize=12)
    plt.ylabel('Force, f  (dyne)', fontsize=12)
    plt.savefig('forceTime')
    plt.show()



    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(displacement1x, forceIx, 'k-', linewidth=2)
    line2, = ax.plot(displacement1x, forceCx, 'b-', linewidth=2)
    line3, = ax.plot(displacement1x, forceTx, 'r-', linewidth=2)
    plt.legend([line1, line2, line3],
                ["Inertia Force", "Constitutive Force", "Total Force"],
                bbox_to_anchor=(0, 1), loc='upper left', fontsize=8)
    plt.xlabel('Displacement, u (cm) ', fontsize=12)
    plt.ylabel('Force, f  (dyne)', fontsize=12)
    plt.savefig('forceDisplacement')
    plt.show()



    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(displacement1x, forceIx, 'k-', linewidth=2)
    plt.xlabel('Displacement, u (cm) ', fontsize=12)
    plt.ylabel('Inertia Force, f  (dyne) ', fontsize=12)
    plt.savefig('inertiaForceDisplacement')
    plt.show()


    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(displacement1x, forceCx, 'b-', linewidth=2)
    plt.xlabel('Displacement, u (cm) ', fontsize=12)
    plt.ylabel('Constitutive Force, f  (dyne)', fontsize=12)
    plt.savefig('constitutiveForceDisplacement')
    plt.show()



    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(displacement1x, forceTx, 'r-', linewidth=2)
    plt.xlabel('Displacement, u (cm) ', fontsize=12)
    plt.ylabel('Total Force, f  (dyne)', fontsize=12)
    plt.savefig('totalForceDisplacement')
    plt.show()



    # create the error plot
    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    ax.set_yscale('log')
    line1, = ax.plot(time, error, 'k-', linewidth=2)

    plt.title("Local Truncation Error", fontsize=20)
    plt.xlabel('time, t (s)', fontsize=16)
    plt.ylabel(r'error, e ', fontsize=16)
    plt.legend([line1],
                ["error"], fontsize=14,
                bbox_to_anchor=(0.7, 0.9), loc=2, borderaxespad=0.)
    plt.savefig('UVAerror')
    plt.show()



    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(pureShear, dilation, 'k-', linewidth=2)

    plt.xlabel(r'Far-Field $\ln \sqrt{a/b}$', fontsize=12)
    plt.ylabel(r'$\xi = \ln \sqrt{uv}$', fontsize=12)
    plt.savefig('pureshearDilation')
    plt.show()



    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(pureShear, epsilon1, 'k-', linewidth=2)
    plt.xlabel(r'Far-Field $\ln \sqrt{a/b}$', fontsize=12)
    plt.ylabel(r'$\epsilon = \ln \sqrt{u/v}$', fontsize=12)
    plt.savefig('pureshearEpsilon')
    plt.show()




    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(dilation, stress11G1, 'k-', linewidth=2)
    line2, = ax.plot(dilation, stress11G2, 'b-', linewidth=2)
    line3, = ax.plot(dilation, stress11G3, 'r-', linewidth=2)
    line4, = ax.plot(dilation, stress11G4, 'g-', linewidth=2)
    line5, = ax.plot(dilation, stress11G5, 'm-', linewidth=2)
    plt.legend([line1, line2, line3, line4, line5],
                ["$S_{11}$ at Gauss Point 1", "$S_{11}$ at Gauss Point 2", 
                 "$S_{11}$ at Gauss Point 3", "$S_{11}$ at Gauss Point 4", 
                 "$S_{11}$ at Gauss Point 5"],
                bbox_to_anchor=(0, 1), loc='upper left', fontsize=8)
    plt.xlabel(r'$\xi = \ln \sqrt{uv}$', fontsize=12)
    plt.ylabel('Stress, $S_{11}$  $(dyne.cm^{-2})$', fontsize=12)
    plt.savefig('stressXi')
    plt.show()




    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(epsilon1, stress22G1, 'k-', linewidth=2)
    line2, = ax.plot(epsilon1, stress22G2, 'b-', linewidth=2)
    line3, = ax.plot(epsilon1, stress22G3, 'r-', linewidth=2)
    line4, = ax.plot(epsilon1, stress22G4, 'g-', linewidth=2)
    line5, = ax.plot(epsilon1, stress22G5, 'm-', linewidth=2)
    plt.legend([line1, line2, line3, line4, line5],
                ["$S_{22}$ at Gauss Point 1", "$S_{22}$ at Gauss Point 2", 
                 "$S_{22}$ at Gauss Point 3", "$S_{22}$ at Gauss Point 4", 
                 "$S_{22}$ at Gauss Point 5"],
                bbox_to_anchor=(0, 1), loc='upper left', fontsize=8)
    plt.xlabel(r'$\epsilon = \ln \sqrt{u/v}$', fontsize=12)
    plt.ylabel('Stress, $S_{22}$  $(dyne.cm^{-2})$', fontsize=12)
    plt.savefig('stressEpsilon')
    plt.show()
    
    
    
    # ax = plt.subplot(1, 1, 1)
    # # change fontsize of minor and major tick labels
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.tick_params(axis='both', which='minor', labelsize=10)
    # # add the curves
    # line1, = ax.plot(pureShear, dDilation, 'k-', linewidth=2)
    # plt.xlabel(r'Far-Field $\ln \sqrt{a/b}$', fontsize=12)
    # plt.ylabel(r'$d\xi $', fontsize=12)
    # plt.savefig('puresheardDilation')
    # plt.show()



    # ax = plt.subplot(1, 1, 1)
    # # change fontsize of minor and major tick labels
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.tick_params(axis='both', which='minor', labelsize=10)
    # # add the curves
    # line1, = ax.plot(pureShear, dEpsilon, 'k-', linewidth=2)
    # plt.xlabel(r'Far-Field $\ln \sqrt{a/b}$', fontsize=12)
    # plt.ylabel(r'$d\epsilon $', fontsize=12)

    # plt.savefig('puresheardEpsilon')
    # plt.show()


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
test()

