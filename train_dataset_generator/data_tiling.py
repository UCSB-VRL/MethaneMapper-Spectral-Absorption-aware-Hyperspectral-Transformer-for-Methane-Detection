import os
import numpy as np
import math
import shutil
import cv2

class DataTiling:
	def __init__(self, size=256, offset=128):
		print("Init DataTiling class")
		self.tile_size = (size, size)
		self.offset = (offset, offset)

	def dataTiling(self, img, filename, dir_path, filetype=None):
		#create tiles directory
		_dir_path = f'{dir_path}/{filename}_tiles'
		self.createDirectory(_dir_path)
		
		img_shape = img.shape
		for i in range(int(math.ceil(img_shape[0]/(self.offset[1] * 1.0)))):
			for j in range(int(math.ceil(img_shape[1]/(self.offset[0] * 1.0)))):
				tile = img[self.offset[1]*i : min(self.offset[1]*i + self.tile_size[1], img_shape[0]), \
							self.offset[0]*j : min(self.offset[0]*j + self.tile_size[0], img_shape[1])]


				if filetype is None:
					tile_name = f'{filename}_{i}_{j}.npy'
				elif filetype is 'png':
					tile_name = f'{filename}_{i}_{j}.png'

				if tile_name not in _dir_path:
					if filetype is None:
						np.save(os.path.join(_dir_path, tile_name), tile)
					if filetype is 'png':
						cv2.imwrite(os.path.join(_dir_path, tile_name), tile)

	def createDirectory(self, _dir_path):
		if not(os.path.isdir(_dir_path)):
			os.mkdir(_dir_path)
			print("\n Created", _dir_path)
		elif os.path.isdir(_dir_path):
			print("\n Already exist", _dir_path, "deleting it")
			shutil.rmtree(_dir_path)
			os.mkdir(_dir_path)
			print("\n Created", _dir_path)

def main(ROOT):
	#gt_img_dir = f"{ROOT}/gt_img"
	gt_mask_dir = f"{ROOT}/gt_mask"
	#gt_img_tiles = f"{ROOT}/gt_img_tiles"
	gt_mask_tiles = f"{ROOT}/gt_mask_tiles"

	#all_imgs = os.listdir(gt_img_dir)
	all_masks = os.listdir(gt_mask_dir)
	#all_img_paths = [os.path.join(gt_img_dir, _name) for _name in all_imgs]
	all_mask_paths = [os.path.join(gt_mask_dir, _name) for _name in all_masks]

	#initializing DataTiling class
	DTObj = DataTiling()

	def _callDataTiling(all_paths, dest_dir):
		for _img_path in all_paths:
			if _img_path.split('.')[-1] == 'npy':
				img = np.load(_img_path)
			elif _img_path.split('.')[-1] == 'png':
				img = cv2.imread(_img_path)
			else: continue
			filename = _img_path.split("/")[-1].split(".")[0]
			DTObj.dataTiling(img, filename, dest_dir, filetype='png')

	#create tiles
	#_callDataTiling(all_img_paths, gt_img_tiles)
	_callDataTiling(all_mask_paths, gt_mask_tiles)

if __name__=='__main__':
	#ROOT = "/data/satish/REFINERY_DATA/create_dataset/data2020/training_data2020"
	#ROOT = "/data/satish/REFINERY_DATA/create_dataset/Methane_leakages_at_oil_refineries/data"
	ROOT = "/data/satish/REFINERY_DATA/create_dataset/Methane_leakages_at_oil_refineries/data/test_set"
	main(ROOT)
