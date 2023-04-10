import os

os.system('''CUDA_VISIBLE_DEVICES=0 python lvae_train.py --baseline --dataset SVHN --encode_z 0 --temperature 2 --eval --threshold 0.3''')