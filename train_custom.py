"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = TrainOptions().parse() 
    dataset = create_dataset(opt)  
    dataset_size = len(dataset)   
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)     
    model.setup(opt)               

    total_iters = 0              

    # outer loop for different epochs; 
    # we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    
        # timer for entire epoch
        epoch_start_time = time.time()  
        # timer for data loading per iteration
        iter_data_time = time.time()    
        # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0                  

        for i, data in enumerate(dataset):  # inner loop within one epoch
            
            pdb.set_trace()
            # timer for computation per iteration
            iter_start_time = time.time()  
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # unpack data from dataset and apply preprocessing
            model.set_input(data)     
            # calculate loss functions, get gradients, update network weights    
            model.optimize_parameters()   

            if total_iters % opt.display_freq == 0:   
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()

            if total_iters % opt.print_freq == 0: 
                # get training loss
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print(losses)

            if total_iters % opt.save_latest_freq == 0:   
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0: 
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        # update learning rates at the end of every epoch.
        model.update_learning_rate()                     
