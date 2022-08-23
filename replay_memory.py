import random
import numpy as np
import collections

class ReplayMemory():
    def __init__(self,max_size):
            "当队列超过最大值时，会自动删除最前面的记录"
            self.buffer=collections.deque(maxlen=max_size)

    def append(self,exp):
        self.buffer.append(exp)

    #对存储的经验分成大小为batch_size的批次
    #再把不同的参数分开存放
    def sample(self,batch_size):
        "本身就是不放回的抽样"
        mini_batch=random.sample(self.buffer,batch_size)
        obs_batch,action_batch,reward_batch,next_obs_batch,done_batch=[],[],[],[],[]

        for experience in mini_batch:
            s,a,r,n_s,done=experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(n_s)
            done_batch.append(done)


        return np.array(obs_batch).astype('float32'), \
                np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'), \
                np.array(next_obs_batch).astype('float32'),np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

# #建立对象
# e=ReplayMemory(max_size=10)
#
# #输入的参数为要加入的数值
# e.append((1,1,100,3))
#
# #sample函数的返回值就是分开的各个样本
# print(e.sample(batch_size=1))
