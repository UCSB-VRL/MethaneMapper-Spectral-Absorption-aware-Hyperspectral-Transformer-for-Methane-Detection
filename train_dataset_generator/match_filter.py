import numpy as np
import spectral.algorithms as algo
from spectral.algorithms.detectors import MatchedFilter, matched_filter
from utils.arg_parser import args


def segMatchedFilterRect(b_img_data, segmentation, target, num_sensors, contours):
	"""
	Takes image, segmentation masks, target signature, pxl_batch and contours
	to compute match filter outputs from pixels belonging to same segmentation class
	class 1 is background (black regions after ortho-correction), ignore this class in computations
	contours contains pixels that belong to each sensor, to get rid of sensor noise
	"""
	num_classes = np.max(segmentation)
	rows, cols, bands =  b_img_data.shape
	print(f"Num clusters:{num_classes}, Rows:{rows}, Cols:{cols}, img_shape:{b_img_data.shape}")
	segmentation_mask = np.zeros((num_classes, rows, cols), dtype=bool)

	for i in range(num_classes):
		segmentation_mask[i] = ~np.ma.make_mask(segmentation -1 -i)
	#print(f"Segmentation mask size: {segmentation_mask.shape}, type: {segmentation_mask.dtype}")

	#initializing with negative value since numpy doesn't save mask
	alpha = np.ones((rows, cols), dtype=float) * (-10.0)

	#removing class 1, black regions after ortho-correction
	num_classes = num_classes - 1
	segmentation_mask = segmentation_mask[1:,:,:]

	#import pdb; pdb.set_trace()
	curr_idx = 0
	all_snsr = len(contours)
	prv_clstr_cov = 0
	for _count in range(num_sensors, all_snsr+num_sensors, num_sensors):
		#Get first num_sensors locations in x,y spatial 
		print(f"computing mf output for sensor set {curr_idx}: {_count}")
		sensor_xy = np.concatenate(contours[curr_idx: min(_count, all_snsr)])
		cls_wise_locs = segmentation_mask[:, sensor_xy[:,1], sensor_xy[:,0]]
		# locations where the class is true
		prev_pix_locs = np.array([[0,0]])
		for _cls in range(num_classes):
			#print(f"Calculating gausian stats, mean, cov of background for cluster {_cls}")
			cls_xy = sensor_xy[cls_wise_locs[_cls, :], :]
			cls_xy = np.concatenate((cls_xy, prev_pix_locs))
			b_img_data[0,0,:] = 0 #init values in case number of pixels in cluster is too low
			clustr_pxls = b_img_data[cls_xy[:,1], cls_xy[:,0], :]
			clustr_pxls = np.expand_dims(clustr_pxls, axis=1)
			#print(f"number of cluster pixels: {clustr_pxls.shape}")
			num_pxls = len(clustr_pxls)
			if(num_pxls==0): continue
			elif(num_pxls>=20):	
				try:
					b_mean_cov_obj = algo.calc_stats(clustr_pxls, mask=None, index=None)
					# Progressively replace clustr_pxls with matched filter outputs
					a = matched_filter(clustr_pxls, target, b_mean_cov_obj)
				except:
					print("singluar matrix, merging data to next iteration")
					prev_pix_locs = cls_xy.copy()
					continue
			else:
				#print(f"Number of pixels is low {num_pxls}, merging data to nearest cluster")
				prev_pix_locs = cls_xy.copy()
				continue

			#print(f"shape of mf output: {a.shape}")
			alpha[cls_xy[:,1], cls_xy[:,0]] = a
		curr_idx = _count
	
	del b_img_data
	return alpha

def segmentation_match_filter(b_img_data, segmentation, target, pxl_batch):
	"""
	Takes image, segmentation masks, target signature, and pxl_batch
	to compute match filter outputs from pixels belonging to same segmentation class
	class 1 is background (black regions after ortho-correction), ignore this class in computations
	"""

	num_classes = np.max(segmentation)
	rows, cols, bands =  b_img_data.shape
	print(f"Num clusters:  {num_classes}, Rows: {rows}, Cols: {cols}, img_shape: {b_img_data.shape}, mask min: {np.min(segmentation)}")
	segmentation_mask = np.zeros((num_classes, rows, cols), dtype=bool)
	
	# Create mask from segmentation. --> should test whether its faster to create mask and fancy indexing or use np.where
	for i in range(num_classes):
		segmentation_mask[i] = ~np.ma.make_mask(segmentation -1 -i)
	print(f"Segmentation mask size: {segmentation_mask.shape}, type: {segmentation_mask.dtype}")
	
	#initializing with negative value since numpy doesn't save mask
	alpha = np.ones((rows, cols), dtype=float) * (-10.0)

	#removing class 1, black regions after ortho-correction
	num_classes = num_classes - 1
	segmentation_mask = segmentation_mask[1:,:,:]

	for i in range(num_classes):
		print(f"Calculating gausian stats, mean, cov of background for cluster {i}")


		if args.segmentation_mf == 'column_wise':
			# Get all pixels from specific segmentation bin in columnwise order
			clustr_pxls = b_img_data.transpose()[:, segmentation_mask[i].transpose()].transpose() 
		elif args.segmentation_mf == 'row_wise':
			# Get all pixels from specific segmentation bin in rowwise order
			clustr_pxls = b_img_data[segmentation_mask[i], :] 

			
		clustr_mf_output = np.zeros(clustr_pxls.shape[0])
		num_pxls, _ = clustr_pxls.shape # How many pixels in this segmentation bin
		print(f"	Number of pixels in cluster {i}: {num_pxls}") 

		clustr_pxls = clustr_pxls.reshape(clustr_pxls.shape[0], 1, clustr_pxls.shape[1])
		curr_idx = 0
		while(curr_idx < num_pxls):
			idx_range = min(pxl_batch, num_pxls - curr_idx) # Get indexible range
			print(f"	Computing stats for cluster {i} indices {curr_idx} to {curr_idx+idx_range}")
			b_mean_cov_obj = algo.calc_stats(clustr_pxls[curr_idx:curr_idx+idx_range, :, :], mask=None, index=None)
			print(f"	b_mean_cov_obj -> type: {type(b_mean_cov_obj)} ")
			print(f"	clustr_mf_output: {clustr_mf_output[curr_idx:curr_idx+idx_range].shape}")
			
			# Progressively replace clustr_pxls with matched filter outputs
			print(f"	clustr_pxls: {clustr_pxls.shape}")
			a = matched_filter(clustr_pxls[curr_idx:curr_idx+idx_range,:,:], target, b_mean_cov_obj)
			print(f"	shape of mf output: {a.shape}, shape of array allocated for output: {clustr_mf_output[curr_idx:curr_idx+idx_range].shape}")
			clustr_mf_output[curr_idx:curr_idx+idx_range] = a.flatten() 
			curr_idx += idx_range

		# Place matched filter output for specific bin at correct pixel locations
		print(f"	Assigning matched filter output of cluster {i} to correct pixel locations\n")
		alpha.transpose()[segmentation_mask[i].transpose()] = clustr_mf_output
			
	del b_img_data
	return alpha


def match_filter(b_img_data, target, num_columns):
	#compute mean and covariance of set of 5 columns and then run matched filter
	rows, cols, bands = b_img_data.shape
	alpha = np.zeros((rows, 0), dtype=float)
	for _cols in range(0, cols, num_columns):
		print("Calculating gausian stats, mean, cov of background")
		col_range = min(_cols+num_columns, cols)
		b_mean_cov_obj = algo.calc_stats(b_img_data[:, _cols:col_range, :], mask=None, index=None)
		print("Calculating stats of matchFilter...")

		'''
			Matched Filter for processing hyperspectral data
			H0 background distribution -> get it from the most of the images so as to
			have a good background distribution
			H1 target distribution -> get it from the multiple image with the source information.

			aplha(x)=\frac{(\mu_t-\mu_b)^T\Sigma^{-1}(x-\mu_b)}{(\mu_t-\mu_b)^T\Sigma^{-1}(\mu_t-\mu_b)}
			OR
			aplha(x) = a_hat = (transpose(x-u) . inv_cov . (t-u) / (transpose(t-u) . inv_cov . (t-u))
		'''

		print("b_img_data", b_img_data[:, _cols:col_range, :].shape, "target_mean : ", \
					target.shape)

		alpha = np.concatenate((alpha, matched_filter(b_img_data[:, _cols:col_range, :], \
								target, b_mean_cov_obj)), axis=1)
		print("Cols computed so far :", alpha.shape)

	del b_img_data
	return alpha
