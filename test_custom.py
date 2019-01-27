"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    
    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
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
from conversion_util import nchannel_to_rgb


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



label_conv = np.zeros((30))

for key in label_map:
    label_conv[label_map[key][0]] = label_map[key][2]


def encode_output(A):
    A = np.array(A).reshape((30,256,256))
    Ap = np.zeros((256,256))
    for x in range(256):
        for y in range(256):
            max_channel = 0
            max_val = A[0,x,y]
            for i in range(30):
                if A[i,x,y] > max_val:
                    max_val = A[i,x,y]
                    max_channel = i
            Ap[x,y] = label_conv[max_channel]
    return Ap

def encode_3(A):
    A = np.array(A).reshape((3,256,256))
    Ap = np.zeros((256,256,3))
    for x in range(256):
        for y in range(256):
            max_channel = 0
            r,g,b = A[0,x,y], A[1,x,y], A[2,x,y]
            if r == max(r,g,b) and r>0.5:
                Ap[x,y,0] = 50
            if g == max(r,g,b) and g>0.5:
                Ap[x,y,1] = 50
            if b == max(r,g,b) and b>0.5:
                Ap[x,y,2] = 50
    return Ap

def decode_3(A):
    A = np.array(A).reshape((3,256,256))
    Ap = np.zeros((256,256))
    for x in range(256):
        for y in range(256):
            max_channel = 0
            max_val = A[0,x,y]
            for i in range(3):
                if A[i,x,y] > max_val:
                    max_val = A[i,x,y]
                    max_channel = i
            Ap[x,y] = i
    return Ap


## color map
# channel 0 : RGB(128,64,128) - road
# channel 1 : RGB(70,130,180) - sky
# channel 2 : RGB(107,142,35) - vegetation
# channel 3 : RGB(70,70,70)   - buildings
# channel 4 : OTHER           - unlabelled
conv = {
    "0": [128, 64, 128],
    "1": [70,130,180],
    "2": [107,142,35],
    "3": [70,70,70],
    "4": [0,0,0]
}

def computeIOU(X,Y):
    ttl_inter = 0.0
    ttl_union = 0.0

    for ch in range(5):
        inter = 0.0
        union = 0.0
        for x in range(256):
            for y in range(256):
                if X[x,y] == ch and Y[x,y] == ch:
                    inter += 1
                if X[x,y] == ch or Y[x,y] == ch:
                    union += 1
        ttl_inter += inter/3.0
        ttl_union += union/3.0
    return ttl_inter/ttl_union




if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 4   
    opt.batch_size = 1   
    opt.serial_batches = True  
    opt.no_flip = True    
    opt.display_id = -1   
    dataset = create_dataset(opt)  
    model = create_model(opt)      
    model.setup(opt)             
    
    if opt.eval:
        model.eval()
    total = 0.0
    ctr = 0.0
    for i, data in enumerate(dataset):
        model.set_input(data)  
        model.test()           
        visuals = model.get_current_visuals()

        name = data["path"][0].split("/")[-1][:-4]
        # pdb.set_trace()
        x = np.array(data["A"]).astype("uint8").reshape(256,256,3)
        # x = tensor2im(data["A"])
        y, mapy = nchannel_to_rgb(data["B"])
        # y = tensor2im(y)
        # pdb.set_trace()
        yp, mapyp = nchannel_to_rgb(visuals["fake_B"])
        # yp = tensor2im(yp)
        

        combined = np.hstack((x, y, yp))

        y_im = Image.fromarray(y.astype('uint8'))
        yp_im = Image.fromarray(yp.astype('uint8'))
        c_im = Image.fromarray(combined.astype("uint8"))

        # y_label = Image.fromarray(decode_3(data["B"])).convert("L")
        # yp_label = Image.fromarray(decode_3(visuals["fake_B"])).convert("L")

        # Compute IOU
        sc = computeIOU(mapy, mapyp)
        print("IOU ", i, sc)
        total += sc

        y_im.save("./RESULTS/gtFine/"+name+"_gtFine_labelIds.png")
        yp_im.save("./RESULTS/label/"+name+".png")
        c_im.save("./RESULTS/combined/"+name+".png")

        print(i)
        ctr += 1
    
    print("final meanIOU", total, total/ctr)

