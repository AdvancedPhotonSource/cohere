import numpy as np
import sys
import cohere.src_py.utilities.utils as ut
from multiprocessing import Pool
from multiprocessing import cpu_count


class PrepData:
    """
    This is a parent class. It checks the presence of configurerion files. If any mandatory file/parameter is
    missing, it will

    """

    def __init__(self, experiment_dir, *args):
        """
        Creates PrepData instance. Sets fields to configuration parameters.

        Parameters
        ----------
        experiment_dir : str
            directory where the files for the experiment processing are created

        Returns
        -------
        PrepData object
        """
        pass


    # Pooling the read and align since that takes a while for each array
    def prep_data(self, dirs, indexes):
        """
        Creates prep_data.tif file in <experiment_dir>/prep directory or multiple prep_data.tif in <experiment_dir>/<scan_<scan_no>>/prep directories.

        Parameters
        ----------
        none

        Returns
        -------
        nothing
        """
        if len(dirs) == 1:
            arr = self.read_scan(dirs[0])
            if arr is not None:
                self.write_prep_arr(arr)
            return

        try:
            separate_scans = self.separate_scans
        except:
            separate_scans = False

        if separate_scans:
            self.dirs = dirs
            self.scans = scans

            with Pool(processes=min(len(dirs), cpu_count())) as pool:
                pool.map_async(self.read_write, range(len(dirs)))
                pool.close()
                pool.join()
        else:
            first_dir = dirs.pop(0)
            refarr = self.read_scan(first_dir)
            if refarr is None:
                return
            sumarr = np.zeros_like(refarr)
            sumarr = sumarr + refarr
            self.fft_refarr = np.fft.fftn(refarr)
            arr_size = sys.getsizeof(refarr)

            # https://www.edureka.co/community/1245/splitting-a-list-into-chunks-in-python
            # Need to further chunck becauase the result queue needs to hold N arrays.
            # if there are a lot of them and they are big, it runs out of ram.
            # since process takes 10-12 arrays, divide nscans/15 (to be safe) and make that many
            # chunks to pool.  Also ask between pools how much ram is avaiable and modify nproc.

            while (len(dirs) > 0):
                nproc = ut.estimate_no_proc(arr_size, 15)
                chunklist = dirs[0:min(len(dirs), nproc)]
                poollist = [dirs.pop(0) for i in range(len(chunklist))]
                with Pool(processes=nproc) as pool:
                    res = pool.map_async(self.read_align, poollist)
                    pool.close()
                    pool.join()
                for arr in res.get():
                    sumarr = sumarr + arr
            self.write_prep_arr(sumarr)


    def read_write(self, index):
        arr = self.read_scan(self.dirs[index])
        self.write_prep_arr(arr, self.scans[index])


    def get_dirs(self, **args):
        pass


    def read_scan(self, dir, **args):
        """
        Reads raw data files from scan directory, applies correction, and returns 3D corrected data for a single scan directory.

        The correction is detector dependent. It can be darkfield and/ot whitefield correction.

        Parameters
        ----------
        dir : str
            directory to read the raw files from

        Returns
        -------
        arr : ndarray
            3D array containing corrected data for one scan.
        """
        pass


    def write_prep_arr(self, arr, index=None):
        pass


    def read_align(self, dir):
        """
        Aligns scan with reference array.  Referrence array is field of this class.

        Parameters
        ----------
        dir : str
            directory to the raw data

        Returns
        -------
        aligned_array : array
            aligned array
        """
        # read
        arr = self.read_scan(dir)
        # align
        return np.abs(ut.shift_to_ref_array(self.fft_refarr, arr))


    def get_detector_name(self):
        return None


    def set_detector(self, det_obj, map):
        pass
