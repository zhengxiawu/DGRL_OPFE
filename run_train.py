import sys,os
from train import Train
from Utils import logging
from Utils.color_lib import RGBmean,RGBstdv
from Utils.Utils import data_dict_reader,mkdir_if_missing
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
Data= 'CUB'
inter_distance = 4
dst = '/home/zhengxiawu/project/DGCRL_pytorch/para/CUB/softmax_orth_0.5/'
phase = 'train'
mkdir_if_missing(dst)
sys.stdout = logging.Logger(os.path.join(dst, 'log.txt'))
data_dict = data_dict_reader(Data,phase)
Train.learn(dst, RGBmean[Data], RGBstdv[Data], data_dict, inter_distance=inter_distance).run()
