# Copyright (c) UCSB

import json
from pathlib import Path

import numpy as np
import torch

from skimage import draw
import datasets.transforms as T
import torch.utils.data as tdata
import glob

class HyperSegment(tdata.Dataset):
	def __init__(self, img_folder, ann_file, stats_file, return_masks):
		self.data_items = LoadItems(img_folder, ann_file, stats_file)
		self._transform_rgb = makeHyperTransform(img_type='rgb')
		self._transform_mf = makeHyperTransform(img_type='mf')
		self._transform_raw = makeHyperTransform(img_type='raw', stats=self.data_items)
		self.prepare = ConvertHyperToMask(return_masks)
		self.target_keys = list(self.data_items.anns.keys())

	def __getitem__(self, idx):
		rgb_path =	self.data_items.rgb_paths[idx]
		mf_path = self.data_items.mf_paths[idx]
		raw_path = self.data_items.raw_paths[idx]
		target = self.data_items.anns[self.target_keys[idx]]
		rgb_img, mf_img, raw_img, target = self.prepare(rgb_path, mf_path, raw_path, target)

		if self._transform_rgb is not None:
			rgb_img, target = self._transform_rgb(rgb_img, target)
			mf_img, _ = self._transform_mf(mf_img)
			raw_img, _ = self._transform_raw(raw_img)

		return {
			"rgb" : rgb_img,
			"mf" : mf_img,
			"raw" : raw_img,
			"target" : target
			}

	def __len__(self):
		return len(self.data_items.rgb_paths)

def makeHyperTransform(img_type, stats=None):
	if img_type == 'rgb':
		normalize = T.Compose([
			T.ScaleRGB(),
			T.ToTensorHyper(),
			T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			#T.RandomHorizontalFlip()
		])
		return T.Compose([normalize])

	if img_type == 'mf':
		normalize = T.Compose([
			T.ToTensorHyper(),
			T.Normalize([0.49631513], [0.08298772]),
			#T.RandomHorizontalFlip()
		])
		return T.Compose([normalize])

	if img_type == 'raw':
		normalize = T.Compose([
			T.RemoveMaskedArea(mask_val=-49.0),
			T.ToTensorHyper(),
			T.Normalize(stats.mean[-90:].tolist(), stats.std[-90:].tolist()),
			#T.RandomHorizontalFlip()
		])
		return T.Compose([normalize])

	'''
	if image_set == 'train':
		return T.Compose([
			normalize,
		])

	if image_set == 'val':
		return T.Compose([
			normalize,
		])

	raise ValueError(f'unknown {image_set}')
	'''

class LoadItems:
	def __init__(self, img_folder, ann_file, stats_file):
		#load dataset
		self.anns, self.imgs = dict(), dict()
		self.img_dir = img_folder

		if not ann_file == None:
			print("loading annotations into memory...")
			dataset = json.load(open(ann_file, 'r'))
			assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
			self.dataset = dataset
			self.createList()
			self.createPaths()

		if not stats_file == None:
			print("loading mean and std for each band...")
			self.mean = np.load(f"{stats_file}/dataset_mean.npy")
			self.std = np.load(f"{stats_file}/dataset_std.npy")

	def createList(self):
		#create list of all images and annotations
		print("creating annotations list")
		anns, img_id, patch_id = dict(), dict(), dict()
		#FIXED : fix the missing annotations because of same file names
		unq_id = 1 #assigning a id to each image file, at train time it's the image id
		if 'annotations' in self.dataset:
			for ann in self.dataset['annotations']:
				anns[f"{ann['patch_name']}"] = {'segmentation' : ann['segmentation'],
													'bbox' : ann['bbox'],
													'category_id' : ann['category_id'],
													'image_id' : unq_id}
				img_id[f"{ann['patch_name']}"] = ann['image_id']
				patch_id[f"{ann['patch_name']}"] = ann['patch_id']
				unq_id += 1

		#create class members
		self.anns = anns
		self.img_id = img_id
		self.patch_id = patch_id

	def createPaths(self):
		#create a list of all rgb, mf, raw images
		print("creating image paths list")
		rgb_paths, mf_paths, raw_paths = [], [], []

		for _ann_key in list(self.anns.keys()):
			_iid = self.img_id[_ann_key]
			_pid = self.patch_id[_ann_key]

			_rgb = glob.glob(f"{self.img_dir}/rgb_tiles/{_iid}_*/*_{_pid}.npy")[0]
			_tmp = _rgb.split("/")
			_mf = f"{self.img_dir}/mf_tiles/{_tmp[-2]}/{_tmp[-1]}"
			_raw = f"{self.img_dir}/rdata_tiles/{_tmp[-2]}/{_tmp[-1]}"
			rgb_paths.append(_rgb), mf_paths.append(_mf), raw_paths.append(_raw)

		assert len(rgb_paths)==len(self.anns), "Number of annotations are different from images"

		#create class members
		self.rgb_paths = rgb_paths
		self.mf_paths = mf_paths
		self.raw_paths = raw_paths

class ConvertHyperToMask(object):
	def __init__(self, return_masks=False):
		self.return_masks = return_masks

	def __call__(self, rgb_path, mf_path, raw_path, target):
		# dim order according to coco : ch, h, w
		rgb_img = np.load(rgb_path).transpose((2,0,1))
		mf_img = np.expand_dims(np.load(mf_path), axis=2).transpose((2,0,1))
		raw_img = np.load(raw_path).transpose((2,0,1))

		_, h, w = rgb_img.shape

		image_id = target["image_id"]
		image_id = torch.tensor([image_id])

		def _plotGTforVerify(mf_path, rgb_path, target):
			import cv2
			import pdb; pdb.set_trace()
			#mf_img = np.load(mf_path)*255
			rgb_img = np.load(rgb_path)
			target_seg = target['segmentation']
			all_segs = []
			for _seg in target_seg:
				_seg = np.array(_seg)
				all_segs.append(np.expand_dims(_seg, axis=1))
			all_segs = tuple(all_segs)
			#mf_img = cv2.cvtColor(np.uint8(mf_img), cv2.COLOR_GRAY2RGB)
			cv2.drawContours(rgb_img, all_segs, -1, (255,0,0), 3)
			cont_img_path = f"/data/satish/REFINERY_DATA/hyper_detr/data/visual_gt/{mf_path.split('/')[-1].split('.')[0]}.png"
			cv2.imwrite(cont_img_path, rgb_img)

		#_plotGTforVerify(mf_path, rgb_path, target)

		boxes = target["bbox"]
		#guard against no boxes via resizing
		boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
		boxes[:, 2:] += boxes[:, :2]
		boxes[:, 0::2].clamp_(min=0, max=w)
		boxes[:, 1::2].clamp_(min=0, max=h)

		classes = target["category_id"]
		classes = torch.tensor(classes, dtype=torch.int64)

		masks = []
		if self.return_masks:
			segmentations = target["segmentation"]
			for _seg in segmentations:
				#convert boundary pixels to binary segmentation mask
				masks.append(draw.polygon2mask((w, h), np.array(_seg)).T) #this function takes image shape in w,h
			masks = np.array(masks)

		keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
		boxes = boxes[keep]
		classes = classes[keep]
		if self.return_masks:
			masks = torch.from_numpy(masks)
			masks = masks[keep]

		target = {}
		target["boxes"] = boxes
		target["labels"] = classes
		target["image_id"] = image_id
		if self.return_masks:
			target["masks"] = masks
			#print("masks", masks.shape, raw_path)

		target["orig_size"] = torch.as_tensor([int(h), int(w)])
		target["size"] = torch.as_tensor([int(h), int(w)])

		return rgb_img, mf_img, raw_img, target

def buildHyperSeg(image_set, args):
	root = Path(args.hyper_path)
	assert root.exists(), f'provided hyperspectral data path {root} does not exist'
	PATHS = {
		"train" : (root / "train", root / "annotations" / "train" / "train_dummy.json" , root / "data_stats"),
		"val": (root / "val", root / "annotations" / "train" / "val_dummy.json" , root / "data_stats")
		}

	img_folder, ann_file, stats_file = PATHS[image_set]
	dataset = HyperSegment(img_folder, ann_file, stats_file, return_masks=args.masks)

	return dataset
