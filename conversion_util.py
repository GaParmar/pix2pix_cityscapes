import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import tensor2im
import numpy as np
import pdb
from PIL import Image

## color map
# channel 0 : RGB(128,64,128) - road
# channel 1 : RGB(70,130,180) - sky
# channel 2 : RGB(107,142,35) - vegetation
# channel 3 : RGB(70,70,70)   - buildings
# channel 4 : OTHER           - unlabelled

def nchannel_to_rgb(X):
	img = np.zeros((256,256,3))
	class_map = np.zeros((256,256))
	for x in range(256):
		for y in range(256):
			max_channel = 0
			max_val = float(X[0,0,x,y])
			for i in range(5):
				if float(X[0,i,x,y]) > max_val:
					max_channel = i
					max_val = float(X[0,i,x,y])
			class_map[x,y] = max_channel
			if max_channel == 0:
				img[x,y,0] = 128
				img[x,y,1] = 64
				img[x,y,2] = 128
			if max_channel == 1:
				img[x,y,0] = 70
				img[x,y,1] = 130
				img[x,y,2] = 180
			if max_channel == 2:
				img[x,y,0] = 107
				img[x,y,1] = 142
				img[x,y,2] = 35
			if max_channel == 3:
				img[x,y,0] = 70
				img[x,y,1] = 70
				img[x,y,2] = 70
			if max_channel == 4:
				img[x,y,0] = 0
				img[x,y,1] = 0
				img[x,y,2] = 0
	return img, class_map