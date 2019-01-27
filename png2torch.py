from os import listdir
from os.path import isfile, join
import pdb
import cv2
import numpy as np
import pdb
from PIL import Image
import torch
import torchvision.transforms as transforms

test_root = "./datasets/cityscapes/test/"
train_root = "./datasets/cityscapes/train/"


## color map
# channel 0 : RGB(128,64,128) - road
# channel 1 : RGB(70,130,180) - sky
# channel 2 : RGB(107,142,35) - vegetation
# channel 3 : RGB(70,70,70)   - buildings
# channel 4 : OTHER           - unlabelled


def encode_5_channels(img):
    newx = np.zeros([5, 256, 256])
    for x in range(256):
        for y in range(256):
            [b,g,r] = img[x,y,:]
            if r==128 and g==64 and b==128:
                newx[0,x,y] = 1.0
            elif r==70 and g==130 and b==180:
                newx[1,x,y] = 1.0
            elif r==107 and g==142 and b==35:
                newx[2,x,y] = 1.0
            elif r==70 and g==70 and b==70:
                newx[3,x,y] = 1.0
            else:
                newx[4,x,y] = 1.0
    return newx

for file in listdir(test_root):
    if ".png" in file:
        AB = Image.open(test_root+file).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        # A = A
        B = encode_5_channels(np.array(B))

        B = torch.tensor(B)
        torch.save(B, test_root+file[:-4]+"_seg_.pt")
        A = torch.tensor(np.array(A))
        torch.save(A, test_root+file[:-4]+"_img_.pt")
        print file


for file in listdir(train_root):
    if ".png" in file:
        AB = Image.open(train_root+file).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        # A = A
        B = encode_5_channels(np.array(B))

        B = torch.tensor(B)
        torch.save(B, train_root+file[:-4]+"_seg_.pt")
        A = torch.tensor(np.array(A))
        torch.save(A, train_root+file[:-4]+"_img_.pt")
        print file
