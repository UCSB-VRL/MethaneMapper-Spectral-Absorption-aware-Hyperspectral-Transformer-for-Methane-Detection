import numpy as np
import torchvision
import torch, torch.nn as nn
import torch.nn.functional as F

from .basemodel import ResnetBackbone, _NestedTensor
from .position_encoding import build_position_encoding
from util.misc import NestedTensor, is_main_process
#from models.detr import MLP

#TODO : add masks to all the outputs.. missing right now. make nested tensor and then return the values

class MLP(nn.Module):
	""" Very simple multi-layer perceptron (also called FFN)"""

	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		self.num_layers = num_layers
		h = [hidden_dim] * (num_layers - 1)
		self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
		return x


class MfExtractor(ResnetBackbone):
	"""ResNet with frozen BatchNorm"""
	def __init__(self, name, in_channels, out_channels, return_interm_layers):
		self.in_channels = in_channels
		self.out_channels = out_channels
		super().__init__(name, in_channels, out_channels, return_interm_layers)
		self.proj = nn.Conv2d(self.out_channels, 100, kernel_size=1, stride=1)

	def forward(self, tensor_list):
		xs = self.body(tensor_list)
		_layer = list(xs.keys())
		xs_out = self.proj(xs[_layer[-1]])

		return xs_out.flatten(2)

class MfProjection(nn.Module):
	"""Feeding matched filter output to transformer decoder as queries"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		in_dim = self.in_channels**2; hid_dim = in_dim//2
		out_dim = self.in_channels*self.out_channels
		self.mf_proj = MLP(in_dim, hid_dim, out_dim, 2)

	def forward(self, tensor_list):
		bs, ch, h, w = tensor_list.shape
		tensor_list = tensor_list.flatten(start_dim=1, end_dim=-1)
		#reshaping to fit with the shape of the query
		out = self.mf_proj(tensor_list).reshape(bs, self.out_channels, self.in_channels)

		return out

class RgbExtractor(ResnetBackbone):
	"""ResNet with frozen BatchNorm"""
	def __init__(self, name, in_channels, out_channels, return_interm_layers):
		self.in_channels = in_channels
		self.out_channels = out_channels
		super().__init__(name, in_channels, out_channels, return_interm_layers)

	def forward(self, tensor_list):
		xs = self.body(tensor_list)

		return xs

class MultiSpecExtractor(ResnetBackbone):
	"""ResNet with frozen BatchNorm"""
	def __init__(self, name, in_channels, out_channels, return_interm_layers):
		self.in_channels = in_channels
		self.out_channels = out_channels
		super().__init__(name, in_channels, out_channels, return_interm_layers)

	def forward(self, tensor_list):
		xs = self.body(tensor_list)

		return xs

class CombineFeatures(nn.Sequential):
	def __init__(self, rgb_model, multispectral_model, position_embedding, mf_model=None):
		super().__init__(rgb_model, mf_model, multispectral_model, position_embedding)

		self.combiner = nn.Sequential(
			nn.Conv2d(1024*2, 1024, kernel_size=(1,1), stride=1, padding=0),
			nn.ReLU(inplace=True)
			)

	def forward(self, rgb_img, mf_img, multispectral_img):
		comb_out = []
		pos = []
		mf_out = []
		rgb_feat = self[0](rgb_img)
		mf_feat = self[1](mf_img)
		multispectral_feat = self[2](multispectral_img)
		comb_feat = torch.cat((rgb_feat['2'], multispectral_feat['2']), dim=1)

		_comb_out = self.combiner(comb_feat)
		_pos, mask = self[3](_comb_out)
		comb_out.append(_NestedTensor(_comb_out, mask=mask))
		pos.append(_pos)
		#keep matched filter features are separate to pass as queries to transformer decoder
		mf_out.append(_NestedTensor(mf_feat, mask=mask))

		return comb_out, pos, mf_out, rgb_feat, multispectral_feat

def build_backbone(args):
	position_embedding = build_position_encoding(args)
	train_backbone = args.lr_backbone > 0
	ril = args.masks
	name = args.backbone
	rgb_model = RgbExtractor(name, in_channels=3, out_channels=1024, return_interm_layers=ril)
	mf_model = MfExtractor(name, in_channels=1, out_channels=1024, return_interm_layers=None)
	multispectral_model = MultiSpecExtractor(name, in_channels=90, out_channels=1024, return_interm_layers=ril)
	comb_model = CombineFeatures(rgb_model, multispectral_model, position_embedding, mf_model=mf_model)
	comb_model.num_channels = rgb_model.out_channels
	return comb_model

if __name__ == '__main__':
	model1 = RgbExtractor(in_channels=3, out_channels=256)
	model2 = MfExtractor(in_channels=256, out_channels=100)
	model3 = MultiSpecExtractor(in_channels=100, out_channels=256)
	comb_model = CombineFeatures(model1.eval(), model2.eval(), model3.eval())

	y = torch.tensor(np.ones((1, 3, 256, 256), dtype=np.float32))
	x1 = torch.randn(1, 3, 256, 256)
	x2 = torch.randn(1, 1, 256, 256)
	x3 = torch.randn(1, 100, 256, 256)
	out = comb_model(x1, x2, x3)

	print(output.shape)
	assert output.shape == (1, 128, 112, 112)
