from os import listdir
from os.path import isfile, join
import pdb
import cv2
import numpy as np
import pdb
from PIL import Image
import torch
import torchvision.transforms as transforms

root="./datasets/cityscapes/"
test_root = "./datasets/cityscapes/test/"
train_root = "./datasets/cityscapes/train/"


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


## --> collect a list of filepaths of test images
## --> for each, load it, split it, save to testA and testB
for file in listdir(test_root):
    if ".png" in file:
        AB = Image.open(test_root+file).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        a = np.array(A)
        b = np.array(B)

        Image.fromarray(a.astype('uint8')).save(root+"/testA/"+file)
        Image.fromarray(b.astype('uint8')).save(root+"/testB/"+file)

        torch.save(torch.tensor(a), root+"/testA/"+file[:-4]+"_img_.pt")
        torch.save(torch.tensor(encode_5_channels(b)), root+"/testB/"+file[:-4]+"_seg_.pt")
        print(test_root+file)


for file in listdir(train_root):
    if ".png" in file:
        AB = Image.open(train_root+file).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        a = np.array(A)
        b = np.array(B)

        Image.fromarray(a.astype('uint8')).save(root+"/trainA/"+file)
        Image.fromarray(b.astype('uint8')).save(root+"/trainB/"+file)

        torch.save(torch.tensor(a), root+"/trainA/"+file[:-4]+"_img_.pt")
        torch.save(torch.tensor(encode_5_channels(b)), root+"/trainB/"+file[:-4]+"_seg_.pt")
        print(train_root+file)





## color map
# channel 0 : RGB(128,64,128) - road
# channel 1 : RGB(70,130,180) - sky
# channel 2 : RGB(107,142,35) - vegetation
# channel 3 : RGB(70,70,70)   - buildings
# channel 4 : OTHER           - unlabelled

