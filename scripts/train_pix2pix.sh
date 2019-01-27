set -ex
python3 train_custom.py --dataroot ./datasets/cityscapes --name cityscapes_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode cityscape --norm batch --pool_size 0 --output_nc 5 --gpu_ids 0,1,2,3,4,5,6,7 --preprocess none --batch_size 1024
