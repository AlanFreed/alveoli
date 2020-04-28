#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
Module gaussQuadratures.py provides Gaussian quadratures for integration.

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
__version__ = "1.4.0"
__date__ = "04-14-2020"
__update__ = "04-14-2020"
__author__ = "Alan D. Freed, Shahla Zamani"
__author_email__ = "afreed@tamu.edu, Zamani.Shahla@tamu.edu"


"""
A listing of changes made wrt version release can be found at the end of file.


Overview of module gaussQuadratures.py:


Module gaussQuadratures.py exports four classes: gaussQuadChords,
gaussQuadTriangles, gaussQuadPentagons and gaussQuadTetrahedra.  Three
instances for each class are also exported, i.e., one object each that
integrates polynomials exactly which are of degrees 1, 3 and 5.

Each object has three methods associated with it.

object.gaussPts()
    returns the number of Gauss points for this quadrature rule

object.coordinates(atGaussPt)
    returns 'natural' co-ordinates at specified Gauss point as a tuple

object.weight(atGaussPt)
    returns the weight of integration at the specified Gauss point


class

    gaussQuadChords

constructor

    gRod = gaussQuadChords(degree)
        degree   is the order of polynomials that can be integrated exactly

methods

    n = gRod.gaussPts()
    returns
        n   is the total number of Gauss points for this quadrature rule

    (xi) = gRod.coordinates(atGaussPt)
    inputs
        atGaussPt   specifies Gauss point whose co-ordinates are sought
    returns
        (xi)        tuple of 'natural' co-ordinates locating the Gauss point

    w = gRod.weight(atGaussPt)
    inputs
        atGaussPt   specifies Gauss point whose weight is sought
    returns
        w           quadrature weight at the specified Gauss point

objects

    gaussQuadChord1
        an instance of gaussQuadChords that integrates polynomials of degree 1

    gaussQuadChord3
        an instance of gaussQuadChords that integrates polynomials of degree 3

    gaussQuadChord5
        an instance of gaussQuadChords that integrates polynomials of degree 5


class

    gaussQuadTriangles

constructor

    gTri = gaussQuadTriangles(degree)
        degree   is the order of polynomials that can be integrated exactly

methods

    n = gTri.gaussPts()
    returns
        n   is the total number of Gauss points for this quadrature rule

    (xi, eta) = gTri.coordinates(atGaussPt)
    inputs
        atGaussPt   specifies Gauss point whose co-ordinates are sought
    returns
        (xi, eta)   tuple of 'natural' co-ordinates locating the Gauss point

    w = gTri.weight(atGaussPt)
    inputs
        atGaussPt   specifies Gauss point whose weight is sought
    returns
        w           quadrature weight at the specified Gauss point

objects

    gaussQuadTriangle1
        instance of gaussQuadTriangles that integrates polynomials of degree 1

    gaussQuadTriangle3
        instance of gaussQuadTriangles that integrates polynomials of degree 3

    gaussQuadTriangle5
        instance of gaussQuadTriangles that integrates polynomials of degree 5


class

    gaussQuadPentagons

constructor

    gPen = gaussQuadPentagons(degree)
        degree   is the order of polynomials that can be integrated exactly

methods

    n = gPen.gaussPts()
    returns
        n   is the total number of Gauss points for this quadrature rule

    (xi, eta) = gPen.coordinates(atGaussPt)
    inputs
        atGaussPt   specifies Gauss point whose co-ordinates are sought
    returns
        (xi, eta)   tuple of 'natural' co-ordinates locating the Gauss point

    w = gPen.weight(atGaussPt)
    inputs
        atGaussPt   specifies Gauss point whose weight is sought
    returns
        w           quadrature weight at the specified Gauss point

objects

    gaussQuadPentagon1
        instance of gaussQuadPentagons that integrates polynomials of degree 1

    gaussQuadPentagon3
        instance of gaussQuadPentagons that integrates polynomials of degree 3

    gaussQuadPentagon5
        instance of gaussQuadPentagons that integrates polynomials of degree 5


class

    gaussQuadTetrahedra

constructor

    gTet = gaussQuadTetrahedra(degree)
        degree   is the order of polynomials that can be integrated exactly

methods

    n = gTet.gaussPts()
    returns
        n   is the total number of Gauss points for this quadrature rule

    (xi, eta, zeta) = gTet.coordinates(atGaussPt)
    inputs
        atGaussPt   specifies Gauss point whose co-ordinates are sought
    returns
        (xi, eta, zeta)  tuple of 'natural' co-ordinates locating the Gauss pt

    w = gTet.weight(atGaussPt)
    inputs
        atGaussPt   specifies Gauss point whose weight is sought
    returns
        w           quadrature weight at the specified Gauss point

objects

    gaussQuadTetrahedron1
        instance of gaussQuadTetrahedra that integrates polynomials of degree 1

    gaussQuadTetrahedron3
        instance of gaussQuadTetrahedra that integrates polynomials of degree 3

    gaussQuadTetrahedron5
        instance of gaussQuadTetrahedra that integrates polynomials of degree 5
"""


class gaussQuadChords(object):

    def __init__(self, degree):
        if degree == 1:
            self.nodes = 1
            self.xi = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = 0.0
            self.w[0] = 2.0
        elif degree == 3:
            self.nodes = 2
            self.xi = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = -0.577350269189626
            self.xi[1] = 0.577350269189626
            self.w[0] = 1.0
            self.w[1] = 1.0
        elif degree == 5:
            self.nodes = 3
            self.xi = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = -0.774596669241483
            self.xi[1] = 0.0
            self.xi[2] = 0.774596669241483
            self.w[0] = 0.555555555555556
            self.w[1] = 0.888888888888889
            self.w[2] = 0.555555555555556
        else:
            raise RuntimeError("Argument 'degree' was {}.  ".format(degree) +
                               "It must be either 1, 3 or 5.")
        return  # instance of a new object

    def gaussPts(self):
        return self.nodes

    def coordinates(self, atGaussPt):
        if (atGaussPt < 1) or (atGaussPt > self.nodes):
            raise RuntimeError("You requested co-ordinates for Gauss point " +
                               "{}.  Argument atGaussPt ".format(atGaussPt) +
                               "must lie in [1,..,{}].".format(self.nodes))
        xi = self.xi[atGaussPt-1]
        return (xi)

    def weight(self, atGaussPt):
        if (atGaussPt < 1) or (atGaussPt > self.nodes):
            raise RuntimeError("You requested co-ordinates for Gauss point " +
                               "{}.  Argument atGaussPt ".format(atGaussPt) +
                               "must lie in [1,..,{}].".format(self.nodes))
        return self.w[atGaussPt-1]


# objects of type gaussQuadChords

gaussQuadChord1 = gaussQuadChords(1)

gaussQuadChord3 = gaussQuadChords(3)

gaussQuadChord5 = gaussQuadChords(5)


class gaussQuadTriangles(object):

    def __init__(self, degree):
        if degree == 1:
            self.nodes = 1
            self.xi = np.zeros(self.nodes, dtype=float)
            self.eta = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = 0.333333333333333
            self.eta[0] = 0.333333333333333
            self.w[0] = 0.500000000000000
        elif degree == 3:
            self.nodes = 4
            self.xi = np.zeros(self.nodes, dtype=float)
            self.eta = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = 0.333333333333333
            self.xi[1] = 0.200000000000000
            self.xi[2] = 0.200000000000000
            self.xi[3] = 0.600000000000000
            self.eta[0] = 0.333333333333333
            self.eta[1] = 0.600000000000000
            self.eta[2] = 0.200000000000000
            self.eta[3] = 0.200000000000000
            self.w[0] = -0.281250000000000
            self.w[1] = 0.260416666666667
            self.w[2] = 0.260416666666667
            self.w[3] = 0.260416666666667
        elif degree == 5:
            self.nodes = 7
            self.xi = np.zeros(self.nodes, dtype=float)
            self.eta = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = 0.333333333333333
            self.xi[1] = 0.101286507323456
            self.xi[2] = 0.101286507323456
            self.xi[3] = 0.797426985353087
            self.xi[4] = 0.470142064105115
            self.xi[5] = 0.470142064105115
            self.xi[6] = 0.059715871789770
            self.eta[0] = 0.333333333333333
            self.eta[1] = 0.797426985353087
            self.eta[2] = 0.101286507323456
            self.eta[3] = 0.101286507323456
            self.eta[4] = 0.059715871789770
            self.eta[5] = 0.470142064105115
            self.eta[6] = 0.470142064105115
            self.w[0] = 0.112500000000000
            self.w[1] = 0.062969590272413
            self.w[2] = 0.062969590272413
            self.w[3] = 0.062969590272413
            self.w[4] = 0.066197076394253
            self.w[5] = 0.066197076394253
            self.w[6] = 0.066197076394253
        else:
            raise RuntimeError("Argument 'degree' was {}.  ".format(degree) +
                               "It must be either 1, 3 or 5.")
        return  # instance of a new object

    def gaussPts(self):
        return self.nodes

    def coordinates(self, atGaussPt):
        if (atGaussPt < 1) or (atGaussPt > self.nodes):
            raise RuntimeError("You requested co-ordinates for Gauss point " +
                               "{}.  Argument atGaussPt ".format(atGaussPt) +
                               "must lie in [1,..,{}].".format(self.nodes))
        xi = self.xi[atGaussPt-1]
        eta = self.eta[atGaussPt-1]
        return (xi, eta)

    def weight(self, atGaussPt):
        if (atGaussPt < 1) or (atGaussPt > self.nodes):
            raise RuntimeError("You requested co-ordinates for Gauss point " +
                               "{}.  Argument atGaussPt ".format(atGaussPt) +
                               "must lie in [1,..,{}].".format(self.nodes))
        return self.w[atGaussPt-1]


# objects of type gaussQuadTriangles

gaussQuadTriangle1 = gaussQuadTriangles(1)

gaussQuadTriangle3 = gaussQuadTriangles(3)

gaussQuadTriangle5 = gaussQuadTriangles(5)


class gaussQuadPentagons(object):

    def __init__(self, degree):
        if degree == 1:
            self.nodes = 1
            self.xi = np.zeros(self.nodes, dtype=float)
            self.eta = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = 0.0000000000000000
            self.eta[0] = 0.0000000000000000
            self.w[0] = 2.3776412907378837
        elif degree == 3:
            self.nodes = 4
            self.xi = np.zeros(self.nodes, dtype=float)
            self.eta = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = -0.0349156305831802
            self.xi[1] = -0.5951653065516678
            self.xi[2] = 0.0349156305831798
            self.xi[3] = 0.5951653065516677
            self.eta[0] = 0.6469731019095136
            self.eta[1] = -0.0321196846022659
            self.eta[2] = -0.6469731019095134
            self.eta[3] = 0.0321196846022661
            self.w[0] = 0.5449124407446143
            self.w[1] = 0.6439082046243272
            self.w[2] = 0.5449124407446146
            self.w[3] = 0.6439082046243275
        elif degree == 5:
            self.nodes = 7
            self.xi = np.zeros(self.nodes, dtype=float)
            self.eta = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = 0.0000000000000000
            self.xi[1] = -0.1351253857178451
            self.xi[2] = -0.6970858746672087
            self.xi[3] = -0.4651171392611024
            self.xi[4] = 0.2842948078559476
            self.xi[5] = 0.7117958231685716
            self.xi[6] = 0.5337947578638855
            self.eta[0] = -0.0000000000000002
            self.eta[1] = 0.7099621260052327
            self.eta[2] = 0.1907259121533272
            self.eta[3] = -0.5531465782166917
            self.eta[4] = -0.6644407817506509
            self.eta[5] = -0.1251071394727008
            self.eta[6] = 0.4872045224587945
            self.w[0] = 0.6257871064166934
            self.w[1] = 0.3016384608809768
            self.w[2] = 0.3169910433902452
            self.w[3] = 0.3155445150066620
            self.w[4] = 0.2958801959111726
            self.w[5] = 0.2575426306970870
            self.w[6] = 0.2642573384350463
        else:
            raise RuntimeError("Argument 'degree' was {}.  ".format(degree) +
                               "It must be either 1, 3 or 5.")
        return  # instance of a new object

    def gaussPts(self):
        return self.nodes

    def coordinates(self, atGaussPt):
        if (atGaussPt < 1) or (atGaussPt > self.nodes):
            raise RuntimeError("You requested co-ordinates for Gauss point " +
                               "{}.  Argument atGaussPt ".format(atGaussPt) +
                               "must lie in [1,..,{}].".format(self.nodes))
        xi = self.xi[atGaussPt-1]
        eta = self.eta[atGaussPt-1]
        return (xi, eta)

    def weight(self, atGaussPt):
        if (atGaussPt < 1) or (atGaussPt > self.nodes):
            raise RuntimeError("You requested co-ordinates for Gauss point " +
                               "{}.  Argument atGaussPt ".format(atGaussPt) +
                               "must lie in [1,..,{}].".format(self.nodes))
        return self.w[atGaussPt-1]


# objects of type gaussQuadPentagons

gaussQuadPentagon1 = gaussQuadPentagons(1)

gaussQuadPentagon3 = gaussQuadPentagons(3)

gaussQuadPentagon5 = gaussQuadPentagons(5)


class gaussQuadTetrahedra(object):

    def __init__(self, degree):
        if degree == 1:
            self.nodes = 1
            self.xi = np.zeros(self.nodes, dtype=float)
            self.eta = np.zeros(self.nodes, dtype=float)
            self.zeta = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = 0.250000000000000
            self.eta[0] = 0.250000000000000
            self.zeta[0] = 0.250000000000000
            self.w[0] = 0.166666666666667
        elif degree == 3:
            self.nodes = 5
            self.xi = np.zeros(self.nodes, dtype=float)
            self.eta = np.zeros(self.nodes, dtype=float)
            self.zeta = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = 0.250000000000000
            self.xi[1] = 0.500000000000000
            self.xi[2] = 0.166666666666667
            self.xi[3] = 0.166666666666667
            self.xi[4] = 0.166666666666667
            self.eta[0] = 0.250000000000000
            self.eta[1] = 0.166666666666667
            self.eta[2] = 0.500000000000000
            self.eta[3] = 0.166666666666667
            self.eta[4] = 0.166666666666667
            self.zeta[0] = 0.250000000000000
            self.zeta[1] = 0.166666666666667
            self.zeta[2] = 0.166666666666667
            self.zeta[3] = 0.500000000000000
            self.zeta[4] = 0.166666666666667
            self.w[0] = -0.133333333333333
            self.w[1] = 0.075000000000000
            self.w[2] = 0.075000000000000
            self.w[3] = 0.075000000000000
            self.w[4] = 0.075000000000000
        elif degree == 5:
            self.nodes = 15
            self.xi = np.zeros(self.nodes, dtype=float)
            self.eta = np.zeros(self.nodes, dtype=float)
            self.zeta = np.zeros(self.nodes, dtype=float)
            self.w = np.zeros(self.nodes, dtype=float)
            self.xi[0] = 0.250000000000000
            self.xi[1] = 0.000000000000000
            self.xi[2] = 0.333333333333333
            self.xi[3] = 0.333333333333333
            self.xi[4] = 0.333333333333333
            self.xi[5] = 0.727272727272727
            self.xi[6] = 0.090909090909091
            self.xi[7] = 0.090909090909091
            self.xi[8] = 0.090909090909091
            self.xi[9] = 0.066550153573664
            self.xi[10] = 0.433449846426336
            self.xi[11] = 0.433449846426336
            self.xi[12] = 0.433449846426336
            self.xi[13] = 0.066550153573664
            self.xi[14] = 0.066550153573664
            self.eta[0] = 0.250000000000000
            self.eta[1] = 0.333333333333333
            self.eta[2] = 0.000000000000000
            self.eta[3] = 0.333333333333333
            self.eta[4] = 0.333333333333333
            self.eta[5] = 0.090909090909091
            self.eta[6] = 0.727272727272727
            self.eta[7] = 0.090909090909091
            self.eta[8] = 0.090909090909091
            self.eta[9] = 0.433449846426336
            self.eta[10] = 0.066550153573664
            self.eta[11] = 0.433449846426336
            self.eta[12] = 0.066550153573664
            self.eta[13] = 0.433449846426336
            self.eta[14] = 0.066550153573664
            self.zeta[0] = 0.250000000000000
            self.zeta[1] = 0.333333333333333
            self.zeta[2] = 0.333333333333333
            self.zeta[3] = 0.000000000000000
            self.zeta[4] = 0.333333333333333
            self.zeta[5] = 0.090909090909091
            self.zeta[6] = 0.090909090909091
            self.zeta[7] = 0.727272727272727
            self.zeta[8] = 0.090909090909091
            self.zeta[9] = 0.433449846426336
            self.zeta[10] = 0.433449846426336
            self.zeta[11] = 0.066550153573664
            self.zeta[12] = 0.066550153573664
            self.zeta[13] = 0.066550153573664
            self.zeta[14] = 0.433449846426336
            self.w[0] = 0.030283678097089
            self.w[1] = 0.006026785714286
            self.w[2] = 0.006026785714286
            self.w[3] = 0.006026785714286
            self.w[4] = 0.006026785714286
            self.w[5] = 0.011645249086029
            self.w[6] = 0.011645249086029
            self.w[7] = 0.011645249086029
            self.w[8] = 0.011645249086029
            self.w[9] = 0.010949141561386
            self.w[10] = 0.010949141561386
            self.w[11] = 0.010949141561386
            self.w[12] = 0.010949141561386
            self.w[13] = 0.010949141561386
            self.w[14] = 0.010949141561386
        else:
            raise RuntimeError("Argument 'degree' was {}.  ".format(degree) +
                               "It must be either 1, 3 or 5.")
        return  # instance of a new object

    def gaussPts(self):
        return self.nodes

    def coordinates(self, atGaussPt):
        if (atGaussPt < 1) or (atGaussPt > self.nodes):
            raise RuntimeError("You requested co-ordinates for Gauss point " +
                               "{}.  Argument atGaussPt ".format(atGaussPt) +
                               "must lie in [1,..,{}].".format(self.nodes))
        xi = self.xi[atGaussPt-1]
        eta = self.eta[atGaussPt-1]
        zeta = self.zeta[atGaussPt-1]
        return (xi, eta, zeta)

    def weight(self, atGaussPt):
        if (atGaussPt < 1) or (atGaussPt > self.nodes):
            raise RuntimeError("You requested co-ordinates for Gauss point " +
                               "{}.  Argument atGaussPt ".format(atGaussPt) +
                               "must lie in [1,..,{}].".format(self.nodes))
        return self.w[atGaussPt-1]


# objects of type gaussQuadTetrahedra

gaussQuadTetrahedron1 = gaussQuadTetrahedra(1)

gaussQuadTetrahedron3 = gaussQuadTetrahedra(3)

gaussQuadTetrahedron5 = gaussQuadTetrahedra(5)

"""
Changes made in version "1.4.0":

Original version
"""
