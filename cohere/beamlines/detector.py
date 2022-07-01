# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################
import cohere.utilities.utils as ut
import os

__author__ = "Ross Harder"
__docformat__ = 'restructuredtext en'
__all__ = ['Detector']


class Detector(object):
    """
    class cohere.Detector(self)
    ===========================

    Abstract class representing detector.

    """
    __all__ = ['get_frame',
               'insert_seam',
               'clear_seam',
               'get_pixel'
               ]

    def __init__(self):
        self.name = "default"


    def get_frame(self, filename, roi, Imult):
        """
        Reads raw 2D frame from a file. Concrete function in subclass applies correction for the specific detector. For example it could be darkfield correction or whitefield correction.

        Parameters
        ----------
        filename : str
            data file name
        roi : list
            detector area used to take image. If None the entire detector area will be used.
        Imult : int
            multiplier

        Returns
        -------
        ndarray
            frame after instrument correction

        """
        filename = filename.replace(os.sep, '/')
        self.raw_frame = ut.read_tif(filename)
        return self.raw_frame


    def insert_seam(self, arr, roi=None):
        """
        Corrects the non-continuous areas of detector. Concrete function in subclass inserts rows/columns in frame as instrument correction.

        Parameters
        ----------
        arr : ndarray
            frame to insert the correction
        roi : list
            detector area used to take image. If None the entire detector area will be used.

        Returns
        -------
        ndarray
            frame after instrument correction

        """
        return arr


    def clear_seam(self, arr, roi=None):
        """
        Corrects the non-continuous areas of detector. Concrete function in subclass removes rows/columns in frame as instrument correction.

        Parameters
        ----------
        arr : ndarray
            frame to apply the correction
        roi : list
            detector area used to take image. If None the entire detector area will be used.

        Returns
        -------
        ndarray
            frame after instrument correction

        """
        return arr

    def get_pixel(self):
        """
        Returns detector pixel size.  Concrete function in subclass returns value applicable to the detector.

        Returns
        -------
        tuple
            size of pixel

        """
        pass