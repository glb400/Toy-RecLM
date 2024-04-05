
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split

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
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen

        print("downloading finished.....")
        
    def __len__(self):
        return len(self.user_train.items())
    
    def __getitem__(self, index: int):
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