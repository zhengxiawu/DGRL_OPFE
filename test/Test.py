import os,cv2
import tqdm
import torch
from PIL import Image
import numpy as np
from Utils.Reader import ImageReader
from Utils.color_lib import RGBmean,RGBstdv
from Utils.Utils import data_info,data_dict_reader
import torchvision.transforms as transforms
from sklearn.metrics import pairwise_distances

def channel_check(tensor):
    if tensor.shape[0]==1:
        temp = torch.ones((3,tensor.shape[1],tensor.shape[2]))
        temp[0, :, :] = tensor
        temp[1, :, :] = tensor
        temp[2, :, :] = tensor
        return temp
    else:
        return tensor

def get_result_list(query_sorted_idx,gt_list,ignore_list,top_k):
    return_retrieval_list = []
    count = 0
    while len(return_retrieval_list)<top_k:
        query_idx = query_sorted_idx[count]
        if query_idx in ignore_list:
            pass
        else:
            if query_idx in gt_list:
                return_retrieval_list.append(1)
            else:
                return_retrieval_list.append(0)
        count+=1
    return return_retrieval_list
def recall_at_k(feature,query_id,retrieval_list,top_k):
    distance = pairwise_distances(feature,feature)
    #distance = compute_distances_self(feature),metric='cosine'
    result = 0
    for i in range(len(query_id)):
        query_distance = distance[query_id[i],:]
        gt_list = retrieval_list[i][0]
        ignore_list = retrieval_list[i][1]
        query_sorted_idx = np.argsort(query_distance)
        query_sorted_idx = query_sorted_idx.tolist()
        result_list = get_result_list(query_sorted_idx,gt_list,ignore_list,top_k)
        #print result_list
        result += 1. if sum(result_list)>0 else 0
    result = result/float(len(query_id))
    return result
class eval():
    def __init__(self, dst, Data, model_path,
                 gpuid = 'cuda:2', feature_save_name = 'feature.npy', imgsize = (256,256),
                 salency='oc_mask', scale=128, pool_type='max_avg',threshold = 0.5,
                 test_mode = 'resource',phase = 'extract_conv_feature'
                 ):
        self.dst = dst
        self.Data = Data
        self.data_dict = data_dict_reader(Data,'test')
        self.gpuid = gpuid
        self.feature_save_name = feature_save_name

        self.imgsize = imgsize
        self.RGBmean = RGBmean[Data]
        self.RGBstdv = RGBstdv[Data]
        self.test_model = test_mode
        self.normalize =transforms.Normalize(mean=self.RGBmean,
                                     std=self.RGBstdv)
        self.classSize = len(self.data_dict)

        self.model_path = model_path
        self.saliency = salency
        self.scale = scale
        self.pool_type = pool_type
        self.threshold = threshold
        self.phase = 'extract_conv_feature'

        self.test_mode = test_mode
        self.save_feature =os.path.join(dst,feature_save_name)

    def run(self):
        self.setsys()
        self.setModel()
        self.extract_feature()
        self.get_recall()
        pass

    def extract_feature_operator(self):
        self.setsys()
        self.setModel()
        self.extract_conv_feature()

    ##################################################
    # step 0: System check and predefine function
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        self.device = torch.device(self.gpuid)
        return True

    ##################################################
    # step 1: set model
    ##################################################
    def setModel(self):
        from models import resnet_50

        self.model = resnet_50.resnet_50(pretrained=False,num_class =self.classSize,
                                         saliency=self.saliency,pool_type = self.pool_type,
                                         is_train = False,scale = self.scale, threshold = self.threshold,
                                         phase = self.phase)
        self.model.cuda()
        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_path).state_dict())
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.9)
        return

    ##################################################
    # step 2: extract feature
    ##################################################
    def extract_feature(self):
        feature = []
        to_tensor = transforms.ToTensor()
        self.dsets = ImageReader(self.data_dict, self.normalize)
        name_list = self.dsets.imgs
        if os.path.isfile(self.save_feature) and len(self.save_feature)>2:
            self.feature = np.load(self.save_feature)
        else:
            if self.test_mode == 'resize':
                for i in tqdm.tqdm(range(len(name_list))):
                    img_path = name_list[i][0]
                    img_list = []
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (self.imgsize))
                    # channel swap
                    img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0].copy()
                    img = np.swapaxes(img, 0, 2)
                    img = np.swapaxes(img, 1, 2)
                    img_list.append(img)
                    image_data = np.asarray(img_list)
                    t_image = torch.from_numpy(image_data)
                    t_image = torch.autograd.Variable(self.normalize(t_image.type(torch.FloatTensor))).cuda()
                    im_feature = self.model(t_image)
                    im_feature = im_feature.cpu().detach().numpy()
                    feature.append(im_feature)
            else:
                for i in tqdm.tqdm(range(len(name_list))):
                    # pytorch method
                    img_path = name_list[i][0]
                    img = Image.open(img_path)
                    w, h = img.size
                    if min(h, w) > 700:
                        size = (int(round(w * (700. / min(h, w)))), int(round(h * (700. / min(h, w)))))
                    else:
                        size = (h, w)
                    scaler = transforms.Resize(size=size)
                    img_tensor = to_tensor(scaler(img))
                    img_tensor = channel_check(img_tensor)
                    t_image = torch.autograd.Variable(self.normalize(img_tensor).unsqueeze(0)).cuda()
                    im_feature = self.model(t_image)
                    im_feature = im_feature.cpu().detach().numpy()
                    feature.append(im_feature)
            feature = np.array(feature)
            self.feature = np.reshape(feature, (feature.shape[0], feature.shape[2]))
            if len(self.save_feature) > 2:
                np.save(self.save_feature,self.feature)

    ##################################################
    # step 2-1: simetimes we need extract convolutional feature
    ##################################################
    def extract_conv_feature(self):
        feature = []
        to_tensor = transforms.ToTensor()
        self.dsets = ImageReader(self.data_dict, self.normalize)
        name_list = self.dsets.imgs
        if os.path.isfile(self.save_feature) and len(self.save_feature)>2:
            self.feature = np.load(self.save_feature)
        else:
            if self.test_mode == 'resize':
                for i in tqdm.tqdm(range(len(name_list))):
                    img_path = name_list[i][0]
                    img_list = []
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (self.imgsize))
                    # channel swap
                    img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0].copy()
                    img = np.swapaxes(img, 0, 2)
                    img = np.swapaxes(img, 1, 2)
                    img_list.append(img)
                    image_data = np.asarray(img_list)
                    t_image = torch.from_numpy(image_data)
                    t_image = torch.autograd.Variable(self.normalize(t_image.type(torch.FloatTensor))).cuda()
                    im_feature = self.model(t_image)
                    im_feature = im_feature.cpu().detach().numpy()
                    feature.append(im_feature)
            else:
                for i in tqdm.tqdm(range(len(name_list))):
                    # pytorch method
                    img_path = name_list[i][0]
                    img = Image.open(img_path)
                    w, h = img.size
                    if min(h, w) > 700:
                        size = (int(round(w * (700. / min(h, w)))), int(round(h * (700. / min(h, w)))))
                    else:
                        size = (h, w)
                    scaler = transforms.Resize(size=size)
                    img_tensor = to_tensor(scaler(img))
                    img_tensor = channel_check(img_tensor)
                    t_image = torch.autograd.Variable(self.normalize(img_tensor).unsqueeze(0)).cuda()
                    im_feature = self.model(t_image)
                    im_feature = im_feature.cpu().detach().numpy()
                    im_feature = np.reshape(im_feature,(im_feature.shape[0]*im_feature.shape[1],
                                                        im_feature.shape[2]*im_feature.shape[3]))
                    im_feature = im_feature.T[~np.all(im_feature.T == 0, axis=1)]
                    if len(feature)==0:
                        feature = im_feature
                        lvecs_idx = np.ones((im_feature.shape[0],))*i
                    else:
                        feature = np.concatenate([feature,im_feature],axis=0)
                        lvecs_idx = np.concatenate([lvecs_idx,np.ones((im_feature.shape[0],))*i])
            #feature = np.array(feature)
            self.feature = feature
            if len(self.save_feature) > 2:
                np.savez(self.save_feature,feature = self.feature, image_id = lvecs_idx)
    ##################################################
    # step 3: recall
    ##################################################
    def get_recall(self):
        data, query_id, retrieval_list = data_info(self.Data,'test')
        top_k = [1,2,4,8,16,32]
        result = []
        for i in top_k:
            result.append(recall_at_k(self.feature,query_id,retrieval_list,i))
        print result