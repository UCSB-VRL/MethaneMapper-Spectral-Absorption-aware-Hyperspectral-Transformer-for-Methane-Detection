#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 14:50:48 2022.

@author: satish
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Indexes:
    def __init__(self, args):
        self.args = args
        print("Initializing Index computation class")
        self.epsilon = 0.0000001
        self.mask_val = -100.0

    def ndvi(self, img):
        # picking bands with guassain distribution with sigma 3
        # TODO : binomial distribution
        idxs = np.random.normal(15, 3, 10)
        b = img[:, :, np.uint8(idxs)]
        self.B = np.sum(b, axis=2) / len(idxs)

        idxs = np.random.normal(31, 3, 10)
        g = img[:, :, np.uint8(idxs)]
        self.G = np.sum(g, axis=2) / len(idxs)

        idxs = np.random.normal(57, 3, 10)
        r = img[:, :, np.uint8(idxs)]
        self.R = np.sum(r, axis=2) / len(idxs)

        idxs = np.random.normal(88, 3, 10)
        inf = img[:, :, np.uint8(idxs)]
        self.Infra = np.sum(inf, axis=2) / len(idxs)

        # Allow division by zero
        np.seterr(divide="ignore", invalid="ignore")
        """
				#water
				ndvi_water = (G.astype(float) - infra.astype(float))/(infra+G)
				if(np.isnan(ndvi_water).any()):
						ndvi_water = np.ma.masked_invalid(ndvi_water)
				"""

        # vegetation
        ndvi_veg = (self.Infra.astype(float) - self.R.astype(float)) / (self.Infra + self.R + self.epsilon)
        # if(np.isnan(ndvi_veg).any()):
        # 	ndvi_veg = np.ma.masked_invalid(ndvi_veg)

        return ndvi_veg

    def getLandCls(self, img, img_mask):
        img = self.ndvi(img)
        img[img_mask] = self.mask_val
        ndvi_class_bins = [-np.inf, self.mask_val + 1, -0.6, -0.3, -0.08, 0, 0.4, 0.6, 0.8, np.inf]
        ndvi_class = np.digitize(img, ndvi_class_bins)
        cus_clr = ["black", "red", "orange", "salmon", "y", "olive", "yellowgreen", "g", "darkgreen"]
        self.cus_cmap = ListedColormap(cus_clr)
        self.ndvi_class = np.ma.masked_where(np.ma.getmask(img), ndvi_class)

        # generate RGB color image for tiling
        clr_img = self.visualize_rgb(split_flag=True, mask=img_mask)

        return self.cus_cmap, self.ndvi_class, clr_img

    def visualizer(self):
        if self.args.side_by_side:
            self.visualize_rgb(split_flag=False)
            self.visualize_land_cls()
            self.visualize_side_by_side()
        else:
            if self.args.color_img:
                self.visualize_rgb(split_flag=False)
            if self.args.visualize:
                self.visualize_land_cls()

    def visualize_rgb(self, split_flag=False, mask=None):
        # create color image of ground terrain
        B = self.B
        G = self.G
        R = self.R
        B = np.ma.masked_array(B, mask=mask)
        G = np.ma.masked_array(G, mask=mask)
        R = np.ma.masked_array(R, mask=mask)
        b_mn = B.min()
        g_mn = G.min()
        r_mn = R.min()
        B = ((B - b_mn) / (B.max() - b_mn)) * 255
        G = ((G - g_mn) / (B.max() - g_mn)) * 255
        R = ((R - r_mn) / (R.max() - r_mn)) * 255

        clr_img = np.uint8(np.stack((B, G, R), axis=2))
        print("Equalizing rgb histogram")

        for i in range(3):
            clr_img[:, :, i] = cv2.equalizeHist(np.uint8(clr_img[:, :, i]))
        """
				img_yuv = cv2.cvtColor(clr_img, cv2.COLOR_BGR2YUV)

				# equalize the histogram of the Y channel
				img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

				# convert the YUV image back to RGB format
				clr_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
				#clr_img = cv2.equalizeHist(np.uint8(clr_img))"""

        if not split_flag:
            cv2.imwrite("rgb_out.png", clr_img)
            print("RGB image generator")
        else:
            return clr_img

    def visualize_land_cls(self):
        vmin = self.ndvi_class.min()
        vmax = self.ndvi_class.max()

        plt.imsave("out.png", self.ndvi_class, vmin=vmin, vmax=vmax, cmap=self.cus_cmap)

    def visualize_side_by_side(self):
        rgb_im = cv2.imread("rgb_out.png")
        cls_im = cv2.imread("out.png")
        sbs_img = cv2.hconcat([rgb_im, cls_im])
        cv2.imwrite("sbs_out.png", sbs_img)
