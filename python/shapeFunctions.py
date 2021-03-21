#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

"""
File shapeFunctions.py provides a base class with abstract methods for
implementing shape functions.

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
__update__ = "07-06-2020"
__author__ = "Alan D. Freed"
__author_email__ = "afreed@tamu.edu"


"""
A listing of changes made wrt version release can be found at the end of file.

This module provides the base type for shape (FE interpolation) functions whose
methods are all abstract and must therefore be implemented when inherited.

class

    ShapeFunction
        Inherits and extends class ABC (Abstract Base Class).

constructor

    sf = ShapeFunction(coordinates)
        coordinates     a tuple containing a set of natural co-ordinates
    returns
        sf              an instance of the abstract base class ShapeFunction
    The constructor of an inherited class must be called via the super command.

abstract methods

    y = sf.interpolate()
    Implementations return an interpolation for a field at the specified co-
    cordinate location assigned by the constructor whose values are known
    (supplied) at the nodal points of an element.

    Jmtx = sf.jacobianMatrix()
    Implementations return the Jacobian matrix at the specified co-ordinate
    location assigned by the constructor given a set of global co-ordinates
    (supplied) that position the element's nodes.

    Jdet = sf.jacobianDeterminant()
    Implementations return the determinant of the Jacobian matrix at the
    specified co-ordinate location assigned by the constructor given a set of
    global co-ordinate (supplied) that position the element's nodes.

    Gmtx = sf.G()
    Implementations return the displacement gradient at the specified co-
    ordinate location assigned by the constructor given a set of global co-
    ordinates (supplied) that position the element's nodes.

    Fmtx = sf.F()
    Implementations return the deformation gradient at the specified co-
    ordinates location assigned by the constructor given a set of global co-
    ordinates (supplied) that position the element's nodes.
"""


class ShapeFunction(ABC):

    def __init__(self, coordinates):
        if isinstance(coordinates, tuple):
            super(ShapeFunction, self).__init__()
        else:
            raise RuntimeError("The coordinates sent to create a shape "
                               + "function must be in the form of a tuple.")
        return  # A new instance for a shape function at specied co-ordinates.

    @abstractmethod
    def interpolate(self):
        # Implementations interpolate a field whose values are known at all
        # nodal points of an element that are then interpolated down to the
        # specified co-ordinates within the element of this shape function.
        pass

    @abstractmethod
    def jacobianMatrix(self):
        # Implementations return the Jacobian matrix at the co-ordinates
        # specified for this shape function.
        pass

    @abstractmethod
    def jacobianDeterminant(self):
        # Implementations return the determinant of its Jacobian matrix.
        pass

    @abstractmethod
    def G(self):
        # Implementations return the displacement gradient at the specified
        # co-ordinates of this shape function.
        pass

    @abstractmethod
    def F(self):
        # Implementations return the deformation gradient at the specified
        # co-ordinates of this shape function.
        pass


"""
Changes made in version "1.0.0":

Original version
"""
