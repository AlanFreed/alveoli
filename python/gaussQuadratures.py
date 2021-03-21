#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

"""
Module gaussQuadratures.py provides an abstract interface for classes where
Gaussian quadrature rules are provided for integration over an element.

Copyright (c) 2020 Alan D. Freed

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
__date__ = "07-07-2020"
__update__ = "07-17-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"


r"""
A listing of changes made wrt version release can be found at the end of file.


Overview of module gaussQuadratures.py:


Module gaussQuadratures.py exports the abstract class: GaussQuadrature.


constructor

    gq = GaussQuadrature()
    This constructor must be called via the super command in inherited classes.

abstract properties

    gPts = gq.gaussPoints()
    Implementations specify number of Gauss points for this quadrature rule.

abstract methods

    yG = gq.interpolate()
    Implementations return a sequence (equal in number to the number of inputs)
    for values of a field, say y, interpolated from the nodal points (the
    inputs) in to the Gauss points (the outputs).

    dY = gq.extrapolate()
    Implementations return a sequence (equal in number to the number of inputs)
    for values of a field, say y, extrapolated from the Gauss points (the
    inputs) out to the nodal points (the outputs).

    coord = gq.coordinates(atGaussPt)
    Implementations return the natural co-ordinates at the specified Gauss
    point, which must lie within the range of [1,..,gq.gaussPoints].

    wgt = gq.weight(atGaussPt)
    Implementations return the weight of quadrature at the specified Gauss
    point, which must lie within the range of [1,..,gq.gaussPoints].
"""


class GaussQuadrature(ABC):

    def __init__(self):
        super(GaussQuadrature, self).__init__()
        return  # a new instance for a Gauss quadrature rule

    @property
    @abstractmethod
    def gaussPoints(self):
        # Implementations specify number of Gauss points--a read-only property.
        pass

    @abstractmethod
    def interpolate(self):
        # Implementations interpolate a field whose values are known at all
        # nodal points of an element that are then interpolated in to all
        # Gauss points of the element.
        pass

    @abstractmethod
    def extrapolate(self):
        # Implementations extrapolate a field whose values are known at all
        # Gauss points of an element that are then extrapolated out to all
        # nodal points of the element.
        pass

    @abstractmethod
    def coordinates(self, atGaussPt):
        # Implementations return the natrual co-ordinates at the specified
        # Gauss point, which takes on values of [1, .., self.gaussPoints].
        pass

    @abstractmethod
    def weight(self, atGaussPt):
        # Implementations return the weight of quadrature at the specified
        # Gauss point, which takes on values of [1, .., self.gaussPoints].
        pass


"""
Changes made in version "1.0.0":

Original version
"""
