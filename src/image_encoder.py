# -*- coding: utf-8 -*-
"""
Testing image encoder
"""

# requirements -------------------------------------------
#! pip install pytorch_pretrained_vit
#! pip install timm
#! wget https://raw.githubusercontent.com/lukemelas/EfficientNet-PyTorch/master/examples/simple/img.jpg
#! wget https://raw.githubusercontent.com/lukemelas/EfficientNet-PyTorch/master/examples/simple/labels_map.txt
# --------------------------------------------------------

import json
import torch
import timm
from PIL import Image
from torchvision import transforms
from pytorch_pretrained_vit import ViT

DEVICE = "cuda:0" if torch.cuda.is_available() else None
MODELNAME = "B_16_imagenet1k"

def load_pretrained_model(model_name="B_16_imagenet1k"):
	""" Download pretrained ViT model loaded into device """
	model = ViT(model_name, pretrained=True)
	model.to(device)
	return model

def load_pretrained_fromtimm(model_name="vit_base_patch16_224.augreg2_in21k_ft_in1k"):
	""" Download pretrained ViT model from timm HF hub """
	model = timm.create_model(model_name, pretrained=True)
	model.to(device)
	return model

def preprocess_image(img):
	"""
	Resize and load image into device
	"""
	tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
	img = tfms(img).unsqueeze(0)
	return torch.tensor(img, device = DEVICE)
	
def evaluateImage(img, labelclass, noutputs=1):
	"""
	Returns class(es) associated to the most likely category predicted by ViT
	Args:
		labelclass: (list) of label names
		noutputs: (int) number of outputs ranked by confidence
	"""
	model.eval()
	with torch.no_grad():
		logits = model(img).squeeze(0)
	probs = torch.softmax(logits, -1)

	#torch.argmax(probs)
	top = torch.topk(probs, k=noutputs)
	for i in range(noutputs):
		print(f"Predicted class: {labelclass[top.indices[i]]}  Confidence: ({top.values[i]})")
	return top
	
	
if __name__ == "__main__":
	
	# loading model 
	model = load_pretrained_model(MODELNAME)
	
	# normalizing images
	img = preprocess_image(Image.open('img.jpg'))
	
	# load labels
	labels_map = json.load(open('labels_map.txt'))
	labels_map = [labels_map[str(i)] for i in range(1000)]
	
	print(evaluateImage(img))

