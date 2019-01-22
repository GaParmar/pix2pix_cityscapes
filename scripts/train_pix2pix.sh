set -ex
python3 train_custom.py --dataroot ./datasets/cityscapes --name cityscapes_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode cityscape --norm batch --pool_size 0 --gpu_ids -1 --preprocess none
