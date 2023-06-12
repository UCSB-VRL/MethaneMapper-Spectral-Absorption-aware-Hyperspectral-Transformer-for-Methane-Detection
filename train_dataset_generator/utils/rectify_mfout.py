#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 08 14:15:38 2022
@author: Satish
"""
# doing the ortho-correction on the processed data from matchedFilter

import os
import numpy as np
#import spectral as spy
#import spectral.io.envi as envi
import json
#import shutil
#import statistics

class Ortho_Correction:
	def __init__(self, dir_path):
		print("Initializing ortho-correction class")
		#manual offset file load
		f = open(f'{dir_path}/manual_offset.json')
		try:
			#Read the manually computed offset file
			offset_data = json.load(f)
			self.OFFSET_DICT = offset_data['OFFSET_DICT']
		except:
			print("No manual offset file found")
			pass

	# Use this fucntion in case you have data other than the custom dataset
	def ideal_rectification(self, glt: np.ndarray, img: np.ndarray, b_val=0.0, output=None) -> np.ndarray:
		"""
		does the ortho-correction of the file
		glt: 2L, world-relative coordinates L1: y (rows), L2: x (columns)
		img: 1L, unrectified, output from matched filter
		output: 1L, rectified version of img, with shape: glt.shape
		"""
		if output is None:
			output = np.zeros((glt.shape[0], glt.shape[1]))
		if not np.array_equal(output.shape, [glt.shape[0], glt.shape[1]]):
			print("image dimension of output arrary do not match the GLT file")
		# getting the absolute even if GLT has negative values
		# magnitude
		glt_mag = np.absolute(glt) 
		# GLT value of zero means no data, extract this because python has zero-indexing.
		glt_mask = np.all(glt_mag==0, axis=2)
		output[glt_mask] = b_val
		glt_mag[glt_mag>(img.shape[0]-1)] = 0
		# now check the lookup and fill in the location, -1 to map to zero-indexing
		# output[~glt_mask] = img[glt_mag[~glt_mask, 1] - 1, glt_mag[~glt_mask, 0] - 1]
		output[~glt_mask] = img[glt_mag[~glt_mask, 1]-1, glt_mag[~glt_mask, 0]-1]
		
		return output

	def custom_rectification(self, file_name, glt: np.ndarray, img: np.ndarray, b_val=0.0, output=None) -> np.ndarray:
		"""does the ortho-correction of the file
		glt: 2L, world-relative coordinates L1: y (rows), L2: x (columns)
		img: 1L, unrectified, output from matched filter
		output: 1L, rectified version of img, with shape: glt.shape
		"""

		if output is None:
			output = np.zeros((glt.shape[0], glt.shape[1]))
		if not np.array_equal(output.shape, [glt.shape[0], glt.shape[1]]):
			print("image dimension of output arrary do not match the GLT file")
		
		print(file_name)
		if file_name in self.OFFSET_DICT.keys():
			offset_mul = self.OFFSET_DICT[file_name]
		else:
			return 0
		print(offset_mul)
		off_v = int(offset_mul*1005)
		img_readB = img[off_v:img.shape[0],:]
		img_readA = img[0:off_v,:]
		img_read = np.vstack((img_readB,img_readA))
		if ((glt.shape[0]-img.shape[0])>0):
			print("size mismatch. Fixing it...")
			completion_shape = np.zeros((glt.shape[0]-img.shape[0], img.shape[1]))
			img_read = np.vstack((img_read, completion_shape))
		print(img_read.shape)
		# getting the absolute even if GLT has negative values
		# magnitude
		glt_mag = np.absolute(glt)
		# GLT value of zero means no data, extract this because python has zero-indexing.
		glt_mask = np.all(glt_mag==0, axis=2)
		output[glt_mask] = b_val
		glt_mag[glt_mag>(img.shape[0]-1)] = 0
		# now check the lookup and fill in the location, -1 to map to zero-indexing
		output[~glt_mask] = img_read[glt_mag[~glt_mask,1]-1, glt_mag[~glt_mask,0]-1]
		
		return output

