#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import materialProperties as mp
# for creating graphics
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np

"""
Created on Sun Oct 06 2019
Updated on Sun Oct 06 2019

A test file for the material property functions in file materialProperties.py.

author: Prof. Alan Freed
"""


def runAlveolar():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    dataPts = 10000

    aDia = []
    for i in range(dataPts):
        aDia.append(mp.alveolarDia())

    n, bins, patches = plt.hist(x=aDia, bins='auto', color='#0504aa',
                                density=False, alpha=0.7, rwidth=0.85)
    maxfreq = n.max()
    plt.grid(axis='y', alpha=0.75)
    plt.ylabel('Counts per 10,000', fontsize=16)
    plt.xlabel('Alveolar Diameter (cm)', fontsize=16)
    plt.ylim(ymax=np.ceil(maxfreq / 100) * 100
             if maxfreq % 100 else maxfreq + 100)
    plt.savefig('alveolarDiaHistogram.jpg')
    plt.show()


def runCollagen():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    dataPts = 10000

    cDia = []
    for i in range(dataPts):
        dia = mp.chordDiaCollagen()
        dia = 10000.0 * dia
        cDia.append(dia)

    n, bins, patches = plt.hist(x=cDia, bins='auto', color='#0504aa',
                                density=False, alpha=0.7, rwidth=0.85)
    maxfreq = n.max()
    plt.grid(axis='y', alpha=0.75)
    plt.ylabel('Counts per 10,000', fontsize=16)
    plt.xlabel(r'Collagen Fiber Diameter ($\mu$m)', fontsize=16)
    plt.ylim(ymax=np.ceil(maxfreq / 100) * 100
             if maxfreq % 100 else maxfreq + 100)
    plt.savefig('collagenFiberDiaHistogram.jpg')
    plt.show()


def runElastin():
    # select the font family and allow TeX commands
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

    dataPts = 10000

    eDia = []
    for i in range(dataPts):
        dia = mp.chordDiaElastin()
        dia = 10000.0 * dia
        eDia.append(dia)

    n, bins, patches = plt.hist(x=eDia, bins='auto', color='#0504aa',
                                density=False, alpha=0.7, rwidth=0.85)
    maxfreq = n.max()
    plt.grid(axis='y', alpha=0.75)
    plt.ylabel('Counts per 10,000', fontsize=16)
    plt.xlabel(r'Elastin Fiber Diameter ($\mu$m)', fontsize=16)
    plt.ylim(ymax=np.ceil(maxfreq / 100) * 100
             if maxfreq % 100 else maxfreq + 100)
    plt.savefig('elastinFiberDiaHistogram.jpg')
    plt.show()


runAlveolar()
runCollagen()
runElastin()
