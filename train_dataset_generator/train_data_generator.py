#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file usage : python train_data_generator.py ../data
"""
Created on Wed Dec 22 14:50:48 2021.

@author: satish
"""

import os
import numpy as np
import glob
import spectral as spy
import spectral.io.envi as envi
import spectral.algorithms as algo
from spectral.algorithms.detectors import MatchedFilter, matched_filter
from joblib import Parallel, delayed

# custom imports
from utils.land_cover_cls import Indexes
from utils.arg_parser import args
from match_filter import segmentation_match_filter, match_filter, segMatchedFilterRect

# from utils.rectify_mfout import Ortho_Correction
from utils.pix_boundary import GetEdgeLocs
from data_tiling import DataTiling

ROOT = "../data/training_data"
DIRECTORY = "../data/aviris_data"


def image_obj(hdr, img):
    """create a object of the image corresponding to certain header."""
    head = envi.read_envi_header(hdr)
    param = envi.gen_params(head)
    param.filename = img  # spectral data file corresponding to .hdr file

    if "bbl" in head:
        head["bbl"] = [float(b) for b in head["bbl"]]
        print(len(head["bbl"]))
        good_bands = np.nonzero(head["bbl"])[0]  # get good bands only for computation
        bad_bands = np.nonzero(np.array(head["bbl"]) == 0)[0]
        print("calculating target signature...")
        # t_mean = target_sign(png_img, img_data_obj, channel, good_bands)

    interleave = head["interleave"]
    if interleave == "bip" or interleave == "BIP":
        print("it is a bip")
        from spectral.io.bipfile import BipFile

        img_obj = BipFile(param, head)

    if interleave == "bil" or interleave == "BIL":
        print("It is a bil file")
        from spectral.io.bilfile import BilFile

        img_obj = BilFile(param, head)

    return img_obj


class DataLoader:
    def __init__(self, data_dir):
        self.file_paths = []
        self.glt_paths = []
        self.data_dir = data_dir
        self.cur_img_path = ""
        self.find_data_paths()

    def find_data_paths(self):
        for root, subdirs, files in os.walk(self.data_dir):
            for _file in files:
                # Depending on what files we need
                if "glt.hdr" in _file:
                    self.glt_paths.append(os.path.join(root, _file))
                    _img_filename = f"{_file.split('glt.hdr')[0]}img.hdr"
                    if not os.path.exists(os.path.join(root, _img_filename)):
                        import pdb

                        pdb.set_trace()
                        print("file not found")
                    else:
                        self.file_paths.append(os.path.join(root, _img_filename))

        if len(self.glt_paths) != len(self.file_paths):
            print("Some files are missing, press c to continue")
            import pdb

            pdb.set_trace()
        print(self.file_paths)

    def load_data(self, file_path):
        img_path = file_path[:-4]
        self.cur_img_path = img_path
        hdr_path = file_path

        print(img_path)
        print(hdr_path)

        self.row_size, self.col_size, self.channel = spy.envi.open(hdr_path).shape
        img_data_obj = image_obj(hdr_path, img_path)

        # reading the whole image at once as we have a 128GB of RAM size
        print("Reading image.. ", file_path)
        big_img_data = img_data_obj.read_subregion((0, self.row_size), (0, self.col_size))
        # import pdb; pdb.set_trace()
        return big_img_data

    def _glt_data(self):
        for glt_path in self.glt_paths:
            img_path = glt_path[:-4]
            hdr_path = glt_path

            # check if current glt file is same as the data file processed
            if self.cur_img_path[:-13] is not img_path[:-13]:
                print("file name mis-match")
                import pdb

                pdb.set_trace()

            glt_data_obj = image_obj(hdr_path, img_path)
            print("Reading image.. ", glt_path)
            glt = glt_data_obj.read_bands([0, 1])

            yield glt_data

    def load_glt_data(self):
        try:
            img_path = f"{self.cur_img_path[:-13]}_rdn_glt"
            hdr_path = f"{img_path}.hdr"
            glt_data_obj = image_obj(hdr_path, img_path)
            print("Reading image.. ", img_path)
            import pdb

            pdb.set_trace()
            glt_data = glt_data_obj.read_bands([0, 1])

            return glt_data
        except:
            print(img_path, "not found")
            import pdb

            pdb.set_trace()

    def normalize(self, img_data, _mask=None):
        np.seterr(divide="ignore", invalid="ignore", over="ignore")
        mask_img = img_data[~_mask, :]
        mask_img -= np.amin(mask_img, axis=0, keepdims=True)
        mask_img /= np.amax(mask_img, axis=0, keepdims=True)
        img_data[~_mask, :] = mask_img

        del mask_img
        return img_data


def main():
    data_loader = DataLoader(DIRECTORY)
    index = Indexes(args)
    locs = GetEdgeLocs()
    tiler = DataTiling(size=256, offset=128)
    pxl_batch_size = args.pxl_batch_size
    num_sensors = args.num_of_sensor

    # load gas signature
    t_sig_path = f"../data/gas_signature/"
    t_mean = np.loadtxt(os.path.join(t_sig_path, os.listdir(t_sig_path)[0]))[:, -1]
    print(t_mean.shape)

    Parallel(n_jobs=1)(
        delayed(computeMF)(file_path, data_loader, index, locs, tiler, pxl_batch_size, num_sensors, t_mean)
        for file_path in data_loader.file_paths
    )


def computeMF(file_path, data_loader, index, locs, tiler, pxl_batch_size, num_sensors, t_mean):
    # checking if file has already been computed
    _temp_filename = file_path.split("/")[-2]
    _temp_mf_filepath = f"{ROOT}/mf_output/{_temp_filename}_img.npy"
    if os.path.exists(_temp_mf_filepath):
        print("skipping file", _temp_filename)
        return

    big_img_data = data_loader.load_data(file_path)[:, :, 6:]  # removing corruputed channels
    # mask the invalid values, aviris invalid val=-50.0
    big_img_mask = np.ma.masked_less(big_img_data[:, :, 10], -49.0).mask
    channel = big_img_data.shape[-1]

    if (channel - t_mean.size) > 0:
        target = np.append(t_mean, np.zeros((channel - t_mean.size)))
    elif (channel - t_mean.size) < 0:
        target = t_mean[0:channel]

    # check for land_cover classification type and visualize based on passed arguments
    _, segmentation, color_img = index.getLandCls(big_img_data, big_img_mask)
    index.visualizer()

    # get boundary pixels locations
    all_contours = locs.getLocs(big_img_mask)

    # comment the standardize function when we have mean and std of the whole data
    big_img_data = data_loader.normalize(big_img_data, _mask=big_img_mask)
    if args.segmentation_mf:
        mf_output = segmentation_match_filter(big_img_data, segmentation, target, pxl_batch_size, all_contours)
    elif args.sensor_noisefree_mf:
        mf_output = segMatchedFilterRect(big_img_data, segmentation, target, num_sensors, all_contours)
    else:
        mf_output = match_filter(big_img_data, target, args.num_columns)

    # create tiles of M/F output and RGB image; and save
    filename = data_loader.cur_img_path.split("/")[-1]
    tiler.dataTiling(mf_output, filename, dir_path=f"{ROOT}/mf_tiles")
    tiler.dataTiling(color_img.data, filename, dir_path=f"{ROOT}/rgb_tiles")
    # taking last 90 channels for tiles(~ 2100-2500nm) highest methane concentration
    tiler.dataTiling(big_img_data[:, :, -90:], filename, dir_path=f"{ROOT}/rdata_tiles")

    # add mask and save output
    filename = data_loader.cur_img_path.split("/")[-1]
    mf_output = np.ma.masked_array(mf_output, mask=big_img_mask)
    mfo_path = f"{ROOT}/mf_output/{filename}.npy"
    # print("Saving output as:", mfo_path)
    np.save(mfo_path, mf_output.data)


if __name__ == "__main__":
    main()
