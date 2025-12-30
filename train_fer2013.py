import os
# Train SqueezeNet 1.1 on FER2013 dataset
cmd = '/Users/usman/Desktop/BTU/ShuffViT-DFER/.venv/bin/python train_squeezenet_fer2013.py --model Ourmodel --bs 32 --lr 0.0001'
os.system(cmd)
print("Train SqueezeNet 1.1 on FER2013 complete!")
os.system('pause')

