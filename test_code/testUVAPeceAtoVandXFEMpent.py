#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from peceAtoVandX import pece
from dodecahedra import dodecahedron
from pivotIncomingF import Pivot
# for making plots
from matplotlib import pyplot as plt
from matplotlib import rc
from math import log
# import calfem.utils as cfu



class fsae(object):

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
    
    # def vertex(self, p):
    #     # get the updated coordinates for the vetices of the pentagon
    #     x1 = p._vertex[1].coordinates('next')
    #     x2 = p._vertex[2].coordinates('next')
    #     x3 = p._vertex[3].coordinates('next')
    #     x4 = p._vertex[4].coordinates('next')
    #     x5 = p._vertex[5].coordinates('next')
        
    #     return x1, x2, x3, x4, x5
        
        


def test():
    # properties for the integrator
    tol = 0.0001   # upper bound on the local truncation error
    N = 150        # number of global steps
    T = 1        # time at the end of the run/analysis
    h = T / N      # global step size

    gaussPts = 5
    maxSqueeze = 0.7

    sep = fsae()
    
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
    p = d.getPentagon(1)
    
    
    # establish the initial state
    t0 = 0.
    x0, v0 = sep.getICs(p, pi)
    x0[4] = 0.0
    x0[5] = 0.0
    x0[7] = 0.0
    a0 = sep.getA(t0, x0, v0, p, pi)


    solver = pece(sep.getA, t0, x0, v0, h, p, pi, tol)
    
    M0 = sep.getM(p)
    K0 = sep.getK(p, pi)
      

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
    resultsPif00 = []

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
    
    resultsT.append((t0))           # time
    
    resultsPif00.append((piF0[0, 0]))


    dilation = np.zeros(N+1, dtype=float)
    epsilon = np.zeros(N+1, dtype=float)
    pureShear = np.zeros(N+1, dtype=float)
    dDilation = np.zeros(N+1, dtype=float)
    dEpsilon = np.zeros(N+1, dtype=float)    
    
    
    
    for n in range(N):
        
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
        resultsFI.append((FI[0], FI[1], FI[2], FI[3], FI[4], FI[5], FI[6], FI[7], 
                          FI[8], FI[9]))  
        resultsFC.append((FC[0], FC[1], FC[2], FC[3], FC[4], FC[5], FC[6], FC[7], 
                          FC[8], FC[9]))
        resultsFT.append((FT[0], FT[1], FT[2], FT[3], FT[4], FT[5], FT[6], FT[7], 
                          FT[8], FT[9]))
        resultsE1.append((e)) 
        
        resultsT.append((t))

        # thermodynamic strains
        dilation[n] = p.dilationSingleComp(gaussPts, 'curr')
        epsilon[n] = p.squeezeSingleComp(gaussPts, 'curr')
        
        
        # thermodynamic strain rates
        dDilation[n] = p.dDilationSingleComp(gaussPts, 'curr')
        dEpsilon[n] = p.dSqueezeSingleComp(gaussPts, 'curr')


        if n < N/3 :
            piF0[0, 0] -= maxSqueeze / (N+1)
            piF0[1, 1] = 1        
            piF0[2, 2] = 1
        else:
            piF0[0, 0] += maxSqueeze / (N+1)
            piF0[1, 1] = 1        
            piF0[2, 2] = 1

        pureShear[n] = log((piF0[0, 0] / piF0[1, 1])**(1.0/3.0))
        
        resultsPif00.append((piF0[0, 0]))
     
                
        d.update(piF0)
        d.advance(pi)   


    dis1x = np.array([z[:] for z in resultsX1])
    displacement1x = dis1x[:, 0]  
    # # normalize this vector
    # maxEle = np.amax(displacement1x)
    # displacement1x = displacement1x / maxEle
    
    dis1y = np.array([z[:] for z in resultsX1])
    displacement1y = dis1y[:, 1]     
    # # normalize this vector
    # maxEle = np.amax(displacement1y)
    # displacement1y = displacement1y / maxEle

    frIx = np.array([z[:] for z in resultsFI])
    forceIx = frIx[:, 0]  
    # # normalize this vector
    # maxEle = np.amax(forceIx)
    # forceIx = forceIx / maxEle
    
    
    frCx = np.array([z[:] for z in resultsFC])
    forceCx = frCx[:, 0]  
    # # normalize this vector
    # maxEle = np.amax(forceCx)
    # forceCx = forceCx / maxEle


    frTx = np.array([z[:] for z in resultsFT])
    forceTx = frTx[:, 0]    
    # # normalize this vector
    # maxEle = np.amax(forceTx)
    # forceTx = forceTx / maxEle
    
        
    time = resultsT
    error = resultsE1
    PiF00 = resultsPif00
    
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
    line1, = ax.plot(time, PiF00, 'b-', linewidth=2)
    plt.xlabel('time (ms)', fontsize=16)
    plt.ylabel('PiF00  ', fontsize=16)
    plt.savefig('PiF00')
    plt.show()









    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(time, displacement1x, 'k-', linewidth=2)
    line2, = ax.plot(time, displacement1y, 'b-', linewidth=2)

    plt.xlabel('time (ms)', fontsize=16)
    plt.ylabel('displacement  ', fontsize=16)
    plt.legend([line1, line2],
                ["ux", "uy"],
                bbox_to_anchor=(0, 1), loc='upper left', fontsize=8)
    plt.savefig('displacement')
    plt.show()





    # create the velocity plot
    vel1x = np.array([z[:] for z in resultsV1])
    velocity1x = vel1x[:, 0]   
    # # normalize this vector
    # maxEle = np.amax(velocity1x)
    # velocity1x = velocity1x / maxEle
    
    vel1y = np.array([z[:] for z in resultsV1])
    velocity1y = vel1y[:, 1]    
    # # normalize this vector
    # maxEle = np.amax(velocity1y)
    # velocity1y = velocity1y / maxEle


    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(time, velocity1x, 'k-', linewidth=2)
    line2, = ax.plot(time, velocity1y, 'b-', linewidth=2)

    plt.xlabel('time  (ms)', fontsize=16)
    plt.ylabel('velocity', fontsize=16)
    plt.legend([line1, line2],
                ["vx", "vy"],
                bbox_to_anchor=(0, 1), loc='upper left', fontsize=8)
    plt.savefig('velocity')
    plt.show()





    # create the acceleration plot
    # ac1x = resultsA1
    ac1x = np.array([z[:] for z in resultsA1])
    acceleration1x = ac1x[:, 0]   
    # # normalize this vector
    # maxEle = np.amax(acceleration1x)
    # acceleration1x = acceleration1x / maxEle
    
    
    ac1y = np.array([z[:] for z in resultsA1])
    acceleration1y = ac1y[:, 1]      
    # # normalize this vector
    # maxEle = np.amax(acceleration1y)
    # acceleration1y = acceleration1y / maxEle


    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(time, acceleration1x, 'k-', linewidth=2)
    line2, = ax.plot(time, acceleration1y, 'b-', linewidth=2)

    plt.xlabel('time (ms)', fontsize=16)
    plt.ylabel('acceleration', fontsize=16)
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
    plt.xlabel('time ', fontsize=12)
    plt.ylabel('force', fontsize=12) 
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
    plt.xlabel('displacement ', fontsize=12)
    plt.ylabel('force', fontsize=12) 
    plt.savefig('forceDisplacement')
    plt.show()
    
    
    
    
    

    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(displacement1x, forceIx, 'k-', linewidth=2)   
    plt.xlabel('displacement ', fontsize=12)
    plt.ylabel('Inertia force ', fontsize=12) 
    plt.savefig('inertiaforceDisplacement')
    plt.show()
    
    
    





    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(displacement1x, forceCx, 'b-', linewidth=2)
    plt.xlabel('displacement ', fontsize=12)
    plt.ylabel('Constitutive force', fontsize=12) 
    plt.savefig('constitutiveforceDisplacement')
    plt.show()






    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(displacement1x, forceTx, 'r-', linewidth=2)  
    plt.xlabel('displacement ', fontsize=12)
    plt.ylabel('total force', fontsize=12) 
    plt.savefig('totalforceDisplacement')
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
    plt.xlabel('time (ms)', fontsize=16)
    plt.ylabel(r'error ', fontsize=16)
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
    line1, = ax.plot(pureShear, epsilon, 'k-', linewidth=2)  
    plt.xlabel(r'Far-Field $\ln \sqrt{a/b}$', fontsize=12)
    plt.ylabel(r'$\epsilon = \ln \sqrt{u/v}$', fontsize=12)  
    plt.savefig('pureshearEpsilon')
    plt.show()    
    
    






    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(pureShear, dDilation, 'k-', linewidth=2)  
    plt.xlabel(r'Far-Field $\ln \sqrt{a/b}$', fontsize=12)
    plt.ylabel(r'$d\xi = \ln \sqrt{uv}$', fontsize=12) 
    plt.savefig('puresheardDilation')
    plt.show()   







    ax = plt.subplot(1, 1, 1)
    # change fontsize of minor and major tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    # add the curves
    line1, = ax.plot(pureShear, dEpsilon, 'k-', linewidth=2)  
    plt.xlabel(r'Far-Field $\ln \sqrt{a/b}$', fontsize=12)
    plt.ylabel(r'$d\epsilon = \ln \sqrt{u/v}$', fontsize=12)  

    plt.savefig('puresheardEpsilon')
    plt.show()   


    

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
test()
