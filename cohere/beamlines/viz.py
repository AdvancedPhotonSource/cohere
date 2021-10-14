# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import numpy as np
import os
from tvtk.api import tvtk
import cohere.utilities.viz_util as vut


__author__ = "Ross Harder"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['CXDViz.__init__',
           'CXDViz.visualize',
           'CXDViz.update_dirspace',
           'CXDViz.update_recipspace',
           'CXDViz.clear_direct_arrays',
           'CXDViz.clear_recip_arrays',
           'CXDViz.add_ds_arrays',
           'CXDViz.add_rs_arrays',
           'CXDViz.get_ds_structured_grid',
           'CXDViz.get_rs_structured_grid',
           'CXDViz.write_directspace',
           'CXDViz.write_recipspace']


class CXDViz:
    """
      This class generates files for visualization. It uses tvtk package for that purpose.
    """
    dir_arrs = {}
    recip_arrs = {}

    def __init__(self, crop, geometry):
        """
        The constructor creates objects assisting with visualization.
        Parameters
        ----------
        crop : tuple or list
            list of fractions; the fractions will be applied to each dimension to derive region to visualize
        geometry : tuple of arrays
            arrays containing geometry in reciprocal and direct space
        Returns
        -------
        constructed object
        """
        self.crop = crop
        self.Trecip, self.Tdir = geometry
        self.dirspace_uptodate = 0
        self.recipspace_uptodate = 0


    def visualize(self, image, support, coh, save_dir, is_twin=False):
        """
        Manages visualization process.
        Parameters
        ----------
        image : ndarray
            image array
        support : ndarray
            support array or None
        coh : ndarray
            coherence array or None
        save_dir : str
            a directory to save the results
        Returns
        -------
        nothing
        """
        arrays = {"imAmp": abs(image), "imPh": np.angle(image)}
        self.add_ds_arrays(arrays)
        if is_twin:
            self.write_directspace(os.path.join(save_dir, 'twin_image'))
        else:
            self.write_directspace(os.path.join(save_dir, 'image'))
        self.clear_direct_arrays()
        if support is not None:
            arrays = {"support": support}
            self.add_ds_arrays(arrays)
            if is_twin:
                self.write_directspace(os.path.join(save_dir, 'twin_support'))
            else:
                self.write_directspace(os.path.join(save_dir, 'support'))
            self.clear_direct_arrays()

        if coh is not None:
            coh = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(coh)))
            coh = vut.get_zero_padded_centered(coh, image.shape)
            arrays = {"cohAmp": np.abs(coh), "cohPh": np.angle(coh)}
            self.add_ds_arrays(arrays)
            self.write_directspace(os.path.join(save_dir, 'coherence'))
            self.clear_direct_arrays()


    def update_dirspace(self, shape):
        """
        Updates direct space grid.
        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        Returns
        -------
        nothing
        """
        dims = list(shape)
        self.dxdir = 1.0 / shape[0]
        self.dydir = 1.0 / shape[1]
        self.dzdir = 1.0 / shape[2]

        r = np.mgrid[
            0:dims[0] * self.dxdir:self.dxdir, \
            0:dims[1] * self.dydir:self.dydir, \
            0:dims[2] * self.dzdir:self.dzdir]

        r.shape = 3, dims[0] * dims[1] * dims[2]

        self.dir_coords = np.dot(self.Tdir, r).transpose()

        self.dirspace_uptodate = 1

    def update_recipspace(self, shape):
        """
        Updates reciprocal space grid.
        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        Returns
        -------
        nothing
        """
        dims = list(shape)
        q = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]

        q.shape = 3, dims[0] * dims[1] * dims[2]

        self.recip_coords = np.dot(self.Trecip, q).transpose()
        self.recipspace_uptodate = 1


    def clear_direct_arrays(self):
        self.dir_arrs.clear()


    def clear_recip_arrays(self):
        self.recip_arrs.clear()


    def add_ds_arrays(self, named_arrays, logentry=None):
        names = sorted(list(named_arrays.keys()))
        shape = named_arrays[names[0]].shape
        if not self.are_same_shapes(named_arrays, shape):
            print('arrays in set should have the same shape')
            return
        # find crop beginning and ending
        [(x1, x2), (y1, y2), (z1, z2)] = self.get_crop_points(shape)
        for name in named_arrays.keys():
            self.dir_arrs[name] = named_arrays[name][x1:x2, y1:y2, z1:z2]
        if (not self.dirspace_uptodate):
            self.update_dirspace((x2 - x1, y2 - y1, z2 - z1))


    def are_same_shapes(self, arrays, shape):
        for name in arrays.keys():
            arr_shape = arrays[name].shape
            for i in range(len(shape)):
                if arr_shape[i] != shape[i]:
                    return False
        return True


    def get_crop_points(self, shape):
        # shape and crop should be 3 long
        crop_points = []
        for i in range(len(shape)):
            cropped_size = int(shape[i] * self.crop[i])
            chopped = int((shape[i] - cropped_size) / 2)
            crop_points.append((chopped, chopped + cropped_size))
        return crop_points


    def get_ds_structured_grid(self, **args):
        sg = tvtk.StructuredGrid()
        arr0 = self.dir_arrs[list(self.dir_arrs.keys())[0]]
        dims = list(arr0.shape)
        sg.points = self.dir_coords
        for a in self.dir_arrs.keys():
            arr = tvtk.DoubleArray()
            arr.from_array(self.dir_arrs[a].ravel())
            arr.name = a
            sg.point_data.add_array(arr)

        sg.dimensions = (dims[2], dims[1], dims[0])
        sg.extent = 0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1
        return sg


    def get_rs_structured_grid(self, **args):
        sg = tvtk.StructuredGrid()
        arr0 = self.recip_arrs[list(self.recip_arrs.keys())[0]]
        dims = list(arr0.shape)
        sg.points = self.recip_coords
        for a in self.recip_arrs.keys():
            arr = tvtk.DoubleArray()
            arr.from_array(self.recip_arrs[a].ravel())
            arr.name = a
            sg.point_data.add_array(arr)

        sg.dimensions = (dims[2], dims[1], dims[0])
        sg.extent = 0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1
        return sg


    def write_directspace(self, filename, **args):
        sgwriter = tvtk.XMLStructuredGridWriter()
        # sgwriter.file_type = 'binary'
        if filename.endswith(".vtk"):
            sgwriter.file_name = filename
        else:
            sgwriter.file_name = filename + '.vts'
        sgwriter.set_input_data(self.get_ds_structured_grid())
        sgwriter.write()
        print('saved file', filename)


    def write_recipspace(self, filename, **args):
        sgwriter = tvtk.XMLStructuredGridWriter()
        if filename.endswith(".vtk"):
            sgwriter.file_name = filename
        else:
            sgwriter.file_name = filename + '.vts'
        sgwriter.set_input_data(self.get_rs_structured_grid())
        sgwriter.write()
        print('saved file', filename)
