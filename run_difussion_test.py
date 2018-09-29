import numpy as np
from Utils import Utils
from test.diffussion import *
from matplotlib import pyplot as plt
import math

K = 100 # approx 50 mutual nns
QUERYKNN = 10
R = 2000
alpha = 0.9


data = 'CUB'
X = np.load('/home/zhengxiawu/project/DGCRL_pytorch/para/CUB/softmax_orth_0.4/feature.npy')
X = X / 128.
data,query_id,retrieval_list = Utils.data_info(data,'test')
sim  = np.dot(X, X.T)
in_sim = []
out_sim = []
# for i in range(X.shape[0]):
#     qsim = sim[i,:]
#     ranks = np.argsort(-qsim)
#     index = ranks[1] if ranks[0] in retrieval_list[i][1] else ranks[0]
#     if index in retrieval_list[i][0]:
#         in_sim.append(qsim[index])
#     else:
#         out_sim.append(qsim[index])
# bins = np.arange(0.,2.,0.01)
# plt.subplot(121)
# plt.hist(in_sim,bins=bins)
# plt.subplot(122)
# plt.hist(out_sim,bins=bins)
# plt.show()
kernel_sim = sim_kernel(sim)
topk_w_ = topK_W(kernel_sim,K)
plain_recall_num = [0,0,0,0,0,0]
cg_ranks_recall = [0,0,0,0,0,0]
import time
for i in range(X.shape[0]):
    a = time.time()
    qsim = kernel_sim[i,:]
    qsim = np.delete(qsim, (i), axis=0)
    qsim = np.reshape(qsim,(1,int(qsim.shape[0])))
    A = np.delete(sim,(i),axis=0)
    A = np.delete(A,(i),axis=1)
    W = np.delete(topk_w_,(i),axis=0)
    W = np.delete(W,(i),axis=1)
    Wn = normalize_connection_graph(W)
    plain_ranks = np.argsort(-qsim)
    cg_ranks = cg_diffusion(qsim, Wn, alpha)
    #fast_spectral_ranks = fsr_rankR(qsim, Wn, alpha, R)
    this_plain_recall_num = 0
    this_cg_ranks_recall = 0
    for j in range(32):
        idx = math.log(j+1, 2)
        if plain_ranks[0,j]>i:
            plain_ranks[0, j]+=1
        if plain_ranks[0,j] in retrieval_list[i][0]:
            this_plain_recall_num+=1
        if cg_ranks[j,0]>i:
            cg_ranks[j, 0]+=1
        if cg_ranks[j,0] in retrieval_list[i][0]:
            this_cg_ranks_recall += 1
        if idx.is_integer():
            if this_plain_recall_num > 0:
                plain_recall_num[int(idx)]+=1
            if this_cg_ranks_recall > 0:
                cg_ranks_recall[int(idx)]+=1
    print 'plain:'+str(plain_recall_num)
    print 'cg:'+str(cg_ranks_recall)
print np.array(plain_recall_num,dtype='float')/float(X.shape[0])
print np.array(cg_ranks_recall,dtype='float')/float(X.shape[0])
