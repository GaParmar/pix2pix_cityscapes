set -ex
python3 train_custom.py --dataroot ./datasets/cityscapes --name cityscapes_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --gpu_ids -1
