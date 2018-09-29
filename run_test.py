from test import Test
from Utils.color_lib import RGBmean,RGBstdv
from Utils.Utils import data_dict_reader,mkdir_if_missing
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
Data= 'CUB'
dst = '/home/zhengxiawu/project/DGCRL_pytorch/para/CUB/softmax_orth_0.4/'
phase = 'test'
mkdir_if_missing(dst)
data_dict = data_dict_reader(Data,phase)
model_path = '/home/zhengxiawu/project/DGCRL_pytorch/para/CUB/softmax_orth_0.4/20_model.pth'
Test.eval(dst, Data, model_path, feature_save_name='selected_conv_feature.npz',salency='scda',phase=phase).extract_feature_operator()
# for i in range(10):
#     threshold = i / 10.
#     print str(threshold)+':'
#     Test.eval(dst, Data, model_path, feature_save_name='', threshold=threshold).run()
# for i in range(50):
#     model_path = dst+str(i*10)+'_model.pth'
#     print model_path
#     if os.path.isfile(model_path):
#         print str(i*10)+':'
#         Test.eval(dst, Data, model_path, feature_save_name='').run()