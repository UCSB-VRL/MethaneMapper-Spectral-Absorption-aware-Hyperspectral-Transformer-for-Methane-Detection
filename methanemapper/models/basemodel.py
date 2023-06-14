import numpy as np
import torchvision
import torch, torch.nn as nn
import torchvision.models as models
from torch import Tensor

from typing import Optional, List

class _NestedTensor(object):
	def __init__(self, tensors, mask: Optional[Tensor]):
		self.tensors = tensors
		self.mask = mask

	def to(self, device):
		# type: (Device) -> NestedTensor # noqa
		cast_tensor = self.tensors.to(device)
		mask = self.mask
		if mask is not None:
			assert mask is not None
			cast_mask = mask.to(device)
		else:
			cast_mask = None
		return NestedTensor(cast_tensor, cast_mask)

	def decompose(self):
		return self.tensors, self.mask

	def __repr__(self):
		return str(self.tensors)

class FrozenBatchNorm2d(nn.Module):
	"""
	BatchNorm2d where the batch statistics and the affine parameters are fixed.

	Copy-paste from torchvision.misc.ops with added eps before rqsrt,
	without which any other models than torchvision.models.resnet[18,34,50,101]
	produce nans.
	"""

	def __init__(self, n):
		super(FrozenBatchNorm2d, self).__init__()
		self.register_buffer("weight", torch.ones(n))
		self.register_buffer("bias", torch.zeros(n))
		self.register_buffer("running_mean", torch.zeros(n))
		self.register_buffer("running_var", torch.ones(n))

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
							  missing_keys, unexpected_keys, error_msgs):
		num_batches_tracked_key = prefix + 'num_batches_tracked'
		if num_batches_tracked_key in state_dict:
			del state_dict[num_batches_tracked_key]

		super(FrozenBatchNorm2d, self)._load_from_state_dict(
			state_dict, prefix, local_metadata, strict,
			missing_keys, unexpected_keys, error_msgs)

	def forward(self, x):
		# move reshapes to the beginning
		# to make it fuser-friendly
		w = self.weight.reshape(1, -1, 1, 1)
		b = self.bias.reshape(1, -1, 1, 1)
		rv = self.running_var.reshape(1, -1, 1, 1)
		rm = self.running_mean.reshape(1, -1, 1, 1)
		eps = 1e-5
		scale = w * (rv + eps).rsqrt()
		bias = b - rm * scale
		return x * scale + bias

class ResnetBackbone(nn.Module):
	"""Pre-trained Resnet50 for feature extraction"""
	def __init__(self, name='resnet50', in_channels=3, out_channels=256, return_interm_layers=None):
		super().__init__()
		backbone = getattr(torchvision.models, name)(
			pretrained=True, norm_layer=FrozenBatchNorm2d)

		backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7),
									stride=(2, 2), padding=(3, 3), bias=False)

		if return_interm_layers:
			#layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
			layers = {"layer1": "0", "layer2": "1", "layer3": "2"}
			self.num_channels = [256, 512, 1024]
		else:
			layers = {"layer3": "2"}
			self.num_channels = [1024]
		self.body = models._utils.IntermediateLayerGetter(backbone, return_layers=layers)
		self.out_channels = out_channels

	"""
	def forward(self, tensor_list):
		xs = self.body(tensor_list)

		out: Dict[str, NestedTensor] = {}
		import pdb; pdb.set_trace()
		for name, x in xs.items():
			m = tensor_list.mask
			assert m is not None
			mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
			out[name] = NestedTensor(x, mask)

		return xs['0']
	"""

if __name__ == '__main__':
	model = ResnetBackbone()
	print("Model Layer: \n", model.body)

	y = torch.tensor(np.ones((1, 3, 224, 224), dtype=np.float32))
	x = torch.randn(1, 3, 224, 224)
	model.eval()
	output = model.forward(x)
	print(output.shape)
	assert output.shape == (1, 128, 112, 112)
