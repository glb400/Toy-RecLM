
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split


# ----- Old Dataset: Design for LLaMA2's Language Task -----

# class PretrainDataset(Dataset):
#     def __init__(self,data_path_lst,max_length=256,memmap=False):
#         super().__init__()
#         #
#         if memmap:
#             with open(data_path_lst[0],'r') as f:
#                 nbytes = f.seek(0,2)
#                 flen = f.tell() // np.dtype('uint16').itemsize
#             self.data = np.memmap(data_path_lst[0],dtype=np.dtype('uint16'),shape=(flen//max_length,max_length))
#         else:
#             data_lst=[]
#             for data_path in data_path_lst:
#                 with open(data_path,'rb') as f:
#                     data=np.fromfile(f,dtype=np.uint16)
#                     data_lst.append(data)
#             data = np.concatenate(data_lst)
#             data = data[:max_length*int(len(data)/max_length)]
#             #np.random.shuffle(data)
#             self.data = data.reshape(-1,max_length)
#         #
#         print("memmap:{} train data.shape:{}".format(memmap,self.data.shape))
#         print("downloading finished.....")
        
#     def __len__(self):
#         return self.data.shape[0]
#     def __getitem__(self, index: int):
#         #
#         sample = self.data[index]
#         X=np.array(sample[:-1]).astype(np.int64)
#         Y=np.array(sample[1:]).astype(np.int64)
        
#         return torch.from_numpy(X),torch.from_numpy(Y)


# ----- New Dataset: Design for New Model's Recommender System Task -----

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


class PretrainDataset_NEW(Dataset):
    def __init__(self, user_train, usernum, itemnum, maxlen):
        super().__init__()        
        # data_lst=[]
        # for data_path in data_path_lst:
        #     with open(data_path,'rb') as f:
        #         data=np.fromfile(f,dtype=np.uint16)
        #         data_lst.append(data)
        # data = np.concatenate(data_lst)
        # data = data[:max_length*int(len(data)/max_length)]
        # #np.random.shuffle(data)
        # self.data = data.reshape(-1,max_length)

        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen

        print("downloading finished.....")
        
    def __len__(self):
        return len(self.user_train.items())
    
    def __getitem__(self, index: int):
        # sample = self.data[index]
        # X=np.array(sample[:-1]).astype(np.int64)
        # Y=np.array(sample[1:]).astype(np.int64)       
        # return torch.from_numpy(X),torch.from_numpy(Y)
        
        # random sampling
        user = np.random.randint(1, self.usernum + 1)
        while len(self.user_train[user]) <= 1: user = np.random.randint(1, self.usernum + 1)

        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        nxt = self.user_train[user][-1]
        idx = self.maxlen - 1

        ts = set(self.user_train[user])
        for i in reversed(self.user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        user_arr=np.array(user).astype(np.int64)
        seq_arr=np.array(seq).astype(np.int64)
        pos_arr=np.array(pos).astype(np.int64)
        neg_arr=np.array(neg).astype(np.int64)

        return torch.from_numpy(user_arr), torch.from_numpy(seq_arr), torch.from_numpy(pos_arr), torch.from_numpy(neg_arr)
    

if __name__=="__main__":
    pass