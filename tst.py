import numpy as np
import cohere_core.utilities as ut
import os

#functions
def getinfo(filename):
    batch_arr = ut.read_tif(filename)
    print(batch_arr.shape)

    max_coordinates = list(np.unravel_index(np.argmax(batch_arr), batch_arr.shape))
    # print("Reciprocal Space:")
    print('max z coord value', max_coordinates[2])
          #, batch_arr[max_coordinates[0], max_coordinates[1], max_coordinates[2]])

    print("maximum = ", np.max(batch_arr))
    print("mean = ", np.mean(batch_arr))

    batch_arr[batch_arr == 0] = np.nan
    print("mean value ignoring zeros = ", np.nanmean(batch_arr))

    # zmax = max_coordinates[2]
    # arr_slice = np.squeeze(batch_arr[:,:,zmax])
    # print("Saving reciprocal space slice at zmax to csv")
    # np.savetxt("arr_slice.csv", arr_slice, delimiter=",")


    # arr_slice_trim = np.where(arr_slice > 3)
    # np.savetxt("arr_slice_trim.csv", arr_slice_trim, delimiter=",")
    # print("Saving real space slice at zmax to csv")
    # print("Real Space:")
    # batch_arr = np.abs(np.fft.ifftn(batch_arr))
    # print(batch_arr.shape)
    # max_coordinates = list(np.unravel_index(np.argmax(batch_arr), batch_arr.shape))
    # print('max z coord value', max_coordinates[2])
    #
    # print("maximum = ", np.max(batch_arr))
    # print("mean = ", np.mean(batch_arr))

def get_correlation(data_dir, refarr, request_err):
    if request_err:
        i = 0
        err = 0
        for scan_dir in os.listdir(data_dir):
            if scan_dir.startswith('scan') and \
                not scan_dir.endswith('2825') and \
                not scan_dir.endswith('2837') and \
                not scan_dir.endswith('2840'):
                subdir = data_dir + '/' + scan_dir
                datafile = subdir + '/preprocessed_data/prep_data.tif'
                arr = ut.read_tif(datafile)
                comp = arr == refarr
                if not comp.all():
                    print('datafile loaded: ', scan_dir)
                    err = err + ut.pixel_shift(refarr, arr)
                    i = i + 1
        return err/i
    # else:
    #     i = 0
    #     total_points = 0
    #     for scan_dir in os.listdir(data_dir):
    #         if scan_dir.startswith('scan'):
    #             subdir = data_dir + '/' + scan_dir
    #             datafile = subdir + '/preprocessed_data/prep_data.tif'
    #             arr = ut.read_tif(datafile)
    #             comp = arr == refarr
    #             if not comp.all():
    #                 print('datafile loaded: ', scan_dir)
    #                 fft_refarr = np.fft.fftn(refarr)
    #                 # align
    #                 aligned = np.abs(cohere_scripts.util.util.shift_to_ref_array(fft_refarr, arr))
    #                 total_points = np.count_nonzero(aligned)
    #                 i = i + 1
    #     return total_points/i



# #Testing Script
data_dir = '/home/phoebus3/BFROSIK/paul/cohere/cohere-ui/example_workspace/LauePUP423a_18scans_2825-2876'
for scan_dir in os.listdir(data_dir):
    if scan_dir.startswith('scan'):
        subdir = data_dir + '/' + scan_dir
        datafile = subdir + '/preprocessed_data/prep_data.tif'
        refarr = ut.read_tif(datafile)

        print('getting correlation error for ' + scan_dir)
        err = get_correlation(data_dir, refarr, True)
        print('average overall error for ' + scan_dir + ':')
        print(err)
        print(' ')

        # print('getting aligned total points for ' + scan_dir)
        # err = get_correlation(data_dir, refarr, False)
        # print('average overall aligned total points for ' + scan_dir)
        # print(err)
        # print(' ')
