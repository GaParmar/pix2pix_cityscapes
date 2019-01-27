set -ex
python3 test_custom.py --dataroot ./datasets/cityscapes --name cityscapes_pix2pix --model pix2pix --netG unet_256 --direction AtoB --dataset_mode cityscape --norm batch --gpu_ids -1 --output_nc 5
