"""
	Replay Buffer for Deep Reinforcement Learning

"""

from collections import deque
import random
import numpy as np


class ReplayBuffer():
    def __init__(self, size_buffer, random_seed=8):
        self.__size_bf = size_buffer
        self.__length = 0
        self.__buffer = deque()
        random.seed(random_seed)

    def add(self, state, action, reward, state_next, done):
        exp = (state, action, reward, state_next, done)
        if self.__length < self.__size_bf:
            self.__buffer.append(exp)
            self.__length += 1
        else:
            self.__buffer.popleft()
            self.__buffer.append(exp)

    def add_batch(self, batch_s, batch_a, batch_r, batch_sn, batch_d):
        for i in range(len(batch_s)):
            self.add(batch_s[i], batch_a[i], batch_r[i], batch_sn[i], batch_d[i])

    def __len__(self):
        return self.__length

    def sample_batch(self, size_batch):

        if self.__length < size_batch:
            batch = random.sample(self.__buffer, self.__length)
        else:
            batch = random.sample(self.__buffer, size_batch)

        batch_s = np.array([d[0] for d in batch])
        batch_a = np.array([d[1] for d in batch])
        batch_r = np.array([d[2] for d in batch])
        batch_sn = np.array([d[3] for d in batch])
        batch_d = np.array([d[4] for d in batch])

        return batch_s, batch_a, batch_r, batch_sn, batch_d

    def clear(self):
        self.__buffer.clear()
        self.count = 0
