# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This module encapsulates diffractometer.
"""

__author__ = "Ross Harder"
__docformat__ = 'restructuredtext en'
__all__ = ['Diffractometer']


class Diffractometer(object):
    """
    class cohere.Diffractometer(self, diff_name)
    ============================================

    Abstract class representing diffractometer. It keeps fields related to the specific diffractometer represented by a subclass.
        diff_name : str
            diffractometer name

    """
    name = None

    def __init__(self, diff_name):
        """
        Constructor.

        Parameters
        ----------
        diff_name : str
            diffractometer name

        """
        self.det_name = diff_name
