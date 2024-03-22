import math
import torch
import numpy as np
from config import *

def batch(lens, batch_size, chunk_size, shuffle):
    idx = np.arange(len(lens))
    if shuffle == True:
        np.random.default_rng().shuffle(idx)
        if chunk_size != None:
            chunk_size = chunk_size * batch_size
            chunks_num = math.ceil(len(idx) / chunk_size)
            split_indices = np.cumsum([chunk_size] * chunks_num)
            chunks = np.split(idx, split_indices)[:-1]
            chunks = [c[np.argsort(lens[c])] for c in chunks]
            idx = np.concatenate(chunks)
    batches_num = math.ceil(len(idx) / batch_size)
    split_indices = np.cumsum([batch_size] * batches_num)
    batches = np.split(idx, split_indices)[:-1]
    idx = np.concatenate(batches)
    return batches

def collate(sequences, batch_first=True):
    max_len = len(max(sequences, key=len))
    t = lambda x: torch.tensor(x, dtype=torch.long)
    p = lambda s: [BOS_IDX] + list(s) + [EOS_IDX] + [PAD_IDX] * (max_len-len(s))
    stack, dim = (torch.vstack, 0) if batch_first else (torch.hstack, 1)
    return stack([t(p(s)).unsqueeze(dim) for s in sequences])

class DataLoader(object):
    def __init__(self, data, batch_size, chunk_size, shuffle):
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.lens = np.array([len(x[0]) for x in data])
        self.batches = batch(self.lens, self.batch_size, self.chunk_size, self.shuffle)
        self.cur_batch = 0

    def __next__(self):
        if self.cur_batch < len(self.batches):
            idx = self.batches[self.cur_batch]
            self.cur_batch = self.cur_batch + 1
            sources, targets = zip(*self.data[idx])
            return collate(sources), collate(targets)
        self.cur_batch = 0
        self.batches = batch(self.lens, self.batch_size, self.chunk_size, self.shuffle)
        raise StopIteration
    
    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        return self