# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.hyper_segm_eval import HyperEvaluator as segm_evaluator
from datasets.hyper_eval import HyperEvaluator as box_evaluator

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
					data_loader: Iterable, optimizer: torch.optim.Optimizer,
					device: torch.device, epoch: int, max_norm: float = 0):

	model.train()
	criterion.train()
	metric_logger = utils.MetricLogger(delimiter="	")
	metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
	metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
	header = 'Epoch: [{}]'.format(epoch)
	print_freq = 10

	for rgb_img, mf_img, raw_img, targets in metric_logger.log_every(data_loader, print_freq, header):
		#import pdb; pdb.set_trace()
		rgb_img = rgb_img.to(device)
		mf_img = mf_img.to(device)
		raw_img = raw_img.to(device)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

		outputs = model(rgb_img, mf_img, raw_img)
		loss_dict = criterion(outputs, targets)
		weight_dict = criterion.weight_dict
		losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

		# reduce losses over all GPUs for logging purposes
		loss_dict_reduced = utils.reduce_dict(loss_dict)
		loss_dict_reduced_unscaled = {f'{k}_unscaled': v
									  for k, v in loss_dict_reduced.items()}
		loss_dict_reduced_scaled = {k: v * weight_dict[k]
									for k, v in loss_dict_reduced.items() if k in weight_dict}
		losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

		loss_value = losses_reduced_scaled.item()

		if not math.isfinite(loss_value):
			print("Loss is {}, stopping training".format(loss_value))
			print(loss_dict_reduced)
			sys.exit(1)

		optimizer.zero_grad()
		losses.backward()
		if max_norm > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
		optimizer.step()

		metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
		metric_logger.update(class_error=loss_dict_reduced['class_error'])
		metric_logger.update(lr=optimizer.param_groups[0]["lr"])
		#break
	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)
	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(dataset_file, model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args):
	model.eval()
	criterion.eval()

	metric_logger = utils.MetricLogger(delimiter="	")
	metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
	header = 'Test:'

	iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
	preds = {}
	gts = {}
	indices = {}
	image_size_list = []

	for rgb_img, mf_img, raw_img, targets in metric_logger.log_every(data_loader, 10, header):
		rgb_img = rgb_img.to(device)
		mf_img = mf_img.to(device)
		raw_img = raw_img.to(device)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

		outputs = model(rgb_img, mf_img, raw_img)
		loss_dict = criterion(outputs, targets)
		weight_dict = criterion.weight_dict

		# reduce losses over all GPUs for logging purposes
		loss_dict_reduced = utils.reduce_dict(loss_dict)
		loss_dict_reduced_scaled = {k: v * weight_dict[k]
									for k, v in loss_dict_reduced.items() if k in weight_dict}
		loss_dict_reduced_unscaled = {f'{k}_unscaled': v
									  for k, v in loss_dict_reduced.items()}
		metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
							 **loss_dict_reduced_scaled,
							 **loss_dict_reduced_unscaled)
		metric_logger.update(class_error=loss_dict_reduced['class_error'])

		orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
		image_size_list.append(orig_target_sizes)
		results = postprocessors['bbox'](outputs, orig_target_sizes)
		if 'segm' in postprocessors.keys():
			target_sizes = torch.stack([t["size"] for t in targets], dim=0)
			results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
		#res = {target['image_id'].item(): output for target, output in zip(targets, results)}
		for target, output in zip(targets, results):
			preds[target['image_id'].item()] = output
			gts[target['image_id'].item()]  = target


		#if len(list(preds.keys())) > 5:
		#	break


	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)

	if args.masks:
		evaluator = segm_evaluator(preds, gts, image_size_list)
	else:
		evaluator = box_evaluator(preds, gts, image_size_list)

	stats = evaluator.evaluate()
	# accumulate predictions from all images
	#stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
	return stats
