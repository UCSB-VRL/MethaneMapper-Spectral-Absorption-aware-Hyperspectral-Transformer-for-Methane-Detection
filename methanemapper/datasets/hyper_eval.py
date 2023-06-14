import numpy as np
from collections import defaultdict
from util import box_ops
import torch

class HyperEvaluator():
	def __init__(self, preds, gts, img_size):
		self.overlap_iou = 0.5
		self.max_plumes = 100
		self.img_size = torch.cat(img_size, dim=0)

		self.fp = []
		self.tp = []
		self.score = []
		self.sum_gts = 0

		self.preds = []
		for img_pred in preds:
			img_pred = {k: v.to('cpu').numpy() for k, v in preds[img_pred].items()}
			bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_pred['boxes'], img_pred['labels'])]
			scores = img_pred['scores']

			label_and_score = [{'category_id': _bboxes['category_id'], 'score': _score}
								for _bboxes, _score in zip(bboxes, scores)]

			#sort all scores in decreasing order
			#label_and_score.sort(key=lambda k: (k.get('score', 0)), reverse=True)
			#label_and_score = label_and_score[:self.max_plumes]

			self.preds.append({
				'predictions': bboxes,
				'label_score': label_and_score,
				'score': scores
			})

		self.gts = []
		for idx, key in enumerate(gts):
			#convert each coordinate from cxcywh to xyxy
			boxes = gts[key]['boxes']
			img_h, img_w = self.img_size[idx]
			scale_fct = torch.stack([img_w, img_h, img_w, img_h])
			gts[key]['boxes'] = box_ops.box_cxcywh_to_xyxy(boxes)
			gts[key]['boxes'] = gts[key]['boxes']*scale_fct[None, :]

			img_gt = {k: v.to('cpu').numpy() for k, v in gts[key].items()}
			bboxes = [{'bbox': bbox, 'category_id': label, 'image_id': key}
						for bbox, label in zip(img_gt['boxes'], img_gt['labels'])]

			labels = [{'category_id': _bboxes['category_id']} for _bboxes in bboxes]

			self.gts.append({
				'annotations': bboxes,
				'label': labels
			})
			self.sum_gts += 1

	def evaluate(self):
		for img_preds, img_gts in zip(self.preds, self.gts):
			pred_bboxes = img_preds['predictions']
			gt_bboxes = img_gts['annotations']
			pred_label_score = img_preds['label_score']
			pred_score = img_preds['score']
			gt_label = img_gts['label']

			if len(gt_bboxes) != 0:
				iou_mat, iou_mat_ov = self.computeIouMat(gt_bboxes, pred_bboxes)
				self.computeFpTp(iou_mat, iou_mat_ov, pred_score, gt_label)
			else:
				for i in range(len(pred_bboxes)):
					self.tp.append(0)
					self.fp.append(1)
					self.score.append(pred_score[i])
		map = self.compute_map()
		return map

	def compute_map(self):
		print('-----------------------------------------------------------------')
		ap = defaultdict(lambda: 0)
		aps = {}

		tp = np.array((self.tp))
		fp = np.array((self.fp))
		if len(tp) == 0:
			ap = 0
		else:
			score = np.array(self.score)
			sort_inds = np.argsort(-score) #sort in descending order
			#TODO : test taking only top 50 indexes
			fp = fp[sort_inds]
			tp = tp[sort_inds]
			fp = np.cumsum(fp)
			tp = np.cumsum(tp)
			rec = tp / self.sum_gts
			prec = tp / (fp + tp)
			ap = self.voc_ap(rec, prec)

		print('------------------------------------------------------------')
		print('mAP all:', ap)
		print('------------------------------------------------------------')

		#aps['AP_{}'.format(ap)] = ap
		aps.update({'mAP_all': ap})

		return aps

	def voc_ap(self, rec, prec):
		ap = 0.
		for t in np.arange(0., 1.1, 0.1):
			if np.sum(rec >= t) == 0:
				p = 0
			else:
				p = np.max(prec[rec >= t])
			ap = ap + p / 11.
		return ap

	def computeFpTp(self, iou_mat, iou_mat_ov, pred_label_score, gt_label):
		scores = pred_label_score
		score_mat = iou_mat.copy()
		score_mat = score_mat*scores

		if np.all(iou_mat==0):
			for i in range(len(scores)):
				self.tp.append(0)
				self.fp.append(1)
				self.score.append(scores[i])
			return

		for _row, score_row in enumerate(score_mat):
			score_mat[_row][np.argmax(score_row)] = 1
		score_mat[score_mat!=1] = 0
		#take sum along rows to get the true positives and false positives
		fptp_mat = np.array(np.sum(score_mat, axis=0), dtype=bool)
		for idx, _entry in enumerate(fptp_mat):
			if _entry == True:
				self.tp.append(1)
				self.fp.append(0)
			elif _entry == False:
				self.tp.append(0)
				self.fp.append(1)
			self.score.append(scores[idx])

	def computeIouMat(self, bbox_list1, bbox_list2):
		iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
		if len(bbox_list1) == 0 or len(bbox_list2) == 0:
			return {}

		for i, bbox1 in enumerate(bbox_list1):
			for j, bbox2 in enumerate(bbox_list2):
				iou_i = self.computeIOU(bbox1, bbox2)
				iou_mat[i, j] = iou_i

		iou_mat_ov=iou_mat.copy()
		iou_mat[iou_mat>=self.overlap_iou] = 1
		iou_mat[iou_mat<self.overlap_iou] = 0

		return iou_mat, iou_mat_ov

	def computeIOU(self, bbox1, bbox2):
		if isinstance(bbox1['category_id'], str):
			bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
		if isinstance(bbox2['category_id'], str):
			bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
		if bbox1['category_id'] == bbox2['category_id']:
			rec1 = bbox1['bbox']
			rec2 = bbox2['bbox']
			# computing area of each rectangles
			S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
			S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

			# computing the sum_area
			sum_area = S_rec1 + S_rec2

			# find the each edge of intersect rectangle
			left_line = max(rec1[1], rec2[1])
			right_line = min(rec1[3], rec2[3])
			top_line = max(rec1[0], rec2[0])
			bottom_line = min(rec1[2], rec2[2])

			# judge if there is an intersect
			if left_line >= right_line or top_line >= bottom_line:
				return 0
			else:
				intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
				return intersect / (sum_area - intersect)
		else:
			return 0
