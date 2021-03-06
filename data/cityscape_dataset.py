"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import pdb
import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np

label_map = {}
label_map[(  0,  0,  0)] = [0, 0, 0] 
label_map[(111, 74,  0)] = [1, 0, 5] 
label_map[( 81,  0, 81)] = [2, 0, 6] 
label_map[(128, 64,128)] = [3, 0, 7]
label_map[(244, 35,232)] = [4, 0, 8]
label_map[(250,170,160)] = [5, 0, 9]
label_map[(230,150,140)] = [6, 0, 10]
label_map[( 70, 70, 70)] = [7, 0, 11]
label_map[(102,102,156)] = [8, 0, 12]
label_map[(190,153,153)] = [9, 0, 13]
label_map[(180,165,180)] = [10, 0, 14]
label_map[(150,100,100)] = [11, 0, 15]
label_map[(150,120, 90)] = [12, 0, 16]
label_map[(153,153,153)] = [13, 0, 17]
label_map[(250,170, 30)] = [14, 0, 19]
label_map[(220,220,  0)] = [15, 0, 20]
label_map[(107,142, 35)] = [16, 0, 21]
label_map[(152,251,152)] = [17, 0, 22]
label_map[( 70,130,180)] = [18, 0, 23]
label_map[(220, 20, 60)] = [19, 0, 24]
label_map[(255,  0,  0)] = [20, 0, 25]
label_map[(  0,  0,142)] = [21, 0, 26]
label_map[(  0,  0, 70)] = [22, 0, 27]
label_map[(  0, 60,100)] = [23, 0, 28]
label_map[(  0,  0, 90)] = [24, 0, 29]
label_map[(  0,  0,110)] = [25, 0, 30]
label_map[(  0, 80,100)] = [26, 0, 31]
label_map[(  0,  0,230)] = [27, 0, 32]
label_map[(119, 11, 32)] = [28, 0, 33]
label_map[(  0,  0,142)] = [29, 0, -1]





# labels = [
#     #       name                       color
#     Label(  'static'               , (  0,  0,  0), 0 ),
#     Label(  'dynamic'              , (111, 74,  0), 1 ),
#     Label(  'ground'               , ( 81,  0, 81), 2 ),
#     Label(  'road'                 , (128, 64,128), 3 ),
#     Label(  'sidewalk'             , (244, 35,232), 4 ),
#     Label(  'parking'              , (250,170,160), 5 ),
#     Label(  'rail track'           , (230,150,140), 6 ),
#     Label(  'building'             , ( 70, 70, 70), 7 ),
#     Label(  'wall'                 , (102,102,156), 8 ),
#     Label(  'fence'                , (190,153,153), 9 ),
#     Label(  'guard rail'           , (180,165,180), 10 ),
#     Label(  'bridge'               , (150,100,100), 11 ),
#     Label(  'tunnel'               , (150,120, 90), 12 ),
#     Label(  'pole'                 , (153,153,153), 13 ),
#     Label(  'traffic light'        , (250,170, 30), 14 ),
#     Label(  'traffic sign'         , (220,220,  0), 15 ),
#     Label(  'vegetation'           , (107,142, 35), 16 ),
#     Label(  'terrain'              , (152,251,152), 17 ),
#     Label(  'sky'                  , ( 70,130,180), 18 ),
#     Label(  'person'               , (220, 20, 60), 19 ),
#     Label(  'rider'                , (255,  0,  0), 20 ),
#     Label(  'car'                  , (  0,  0,142), 21 ),
#     Label(  'truck'                , (  0,  0, 70), 22 ),
#     Label(  'bus'                  , (  0, 60,100), 23 ),
#     Label(  'caravan'              , (  0,  0, 90), 24 ),
#     Label(  'trailer'              , (  0,  0,110), 25 ),
#     Label(  'train'                , (  0, 80,100), 26 ),
#     Label(  'motorcycle'           , (  0,  0,230), 27 ),
#     Label(  'bicycle'              , (119, 11, 32), 28 ),
#     Label(  'license plate'        , (  0,  0,142), 29 ),
# ]


class CityscapeDataset(BaseDataset):

    def __init__(self, opt):
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # get all image paths
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size)) 
        # crop_size should be smaller than the size of loaded image
        assert(self.opt.load_size >= self.opt.crop_size)   
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc



    def __getitem__(self, index):
        AB_path = self.AB_paths[index]

        # path to image and segmentation files
        img = AB_path[:-4]+"_img_.pt"
        seg = AB_path[:-4]+"_seg_.pt"

        A = torch.load(img).view(3,256,256).type('torch.FloatTensor')
        B = torch.load(seg).type('torch.FloatTensor')

        return {'A': A, 'B': B, 'path': AB_path}



    def __len__(self):
        """Return the total number of images."""
        return len(self.AB_paths)
