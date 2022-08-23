import numpy as np
import paddle.fluid as fluid
import parl
from parl import layers
from algorithm import DQN
from model import Model

#定义代理
class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim, e_greed=None, e_greed_decrement=0):

        self.obs_dim=obs_dim
        self.act_dim=act_dim
        #print(act_dim.shape)
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔50个training steps再把model的参数复制到target_model中

        self.e_greed=e_greed
        self.e_greed_decrement=e_greed_decrement
    #初始化计算图表
    def build_program(self):
        self.pred_program=fluid.Program()  #初始化得到预测值的图表，输入值为观察值obs
        self.learn_program=fluid.Program()  #初始化训练的图表

        #给用于预测值的图表定义数据
        with fluid.program_guard(self.pred_program):
            obs=layers.data(name='obs',shape=[self.obs_dim],dtype='float32')  #定义输入
            self.value=self.algorithm.predict(obs)                            #定义输出

        #给用于训练的图表定义数据
        with fluid.program_guard(self.learn_program):
            obs=layers.data(name='obs',shape=[self.obs_dim],dtype='float32')
            "拿这一个动作去训练,这个动作的维度是动作空间的维度"
            action=layers.data(name='act',shape=[1],dtype='int32')
            reward=layers.data(name='reward',shape=[],dtype='float32')
            next_obs=layers.data(name='next_obs',shape=[self.obs_dim],dtype='float32')
            terminal=layers.data(name='terminal',shape=[],dtype='bool')
            self.cost=self.algorithm.learn(obs,action,reward,next_obs,terminal)   #定义输出


    #代理的第二个功能，与环境交互进行动作的选取
    def sample(self,obs):
        sample=np.random.rand()
        if sample<self.e_greed:
            act=np.random.randint(self.act_dim) #挑选一个整数
        else:
            act=self.predict(obs)
        self.e_greed=max(0.01,self.e_greed-self.e_greed_decrement)

        return act

    def predict(self,obs):
        obs=np.expand_dims(obs,axis=0)
        pred_Q =self.fluid_executor.run(
            self.pred_program,
            feed={'obs': np.array(obs.astype('float32'))},
            fetch_list=[self.value])[0]
        pred_Q=np.squeeze(pred_Q,axis=0)
        "返回最大值对应的索引下标，对应动作"
        # act=np.argwhere(pred_Q==np.max(pred_Q))
        act=np.argmax(pred_Q)
        "此时的返回值类型是[[1]]"
        # act=int(np.squeeze(act,axis=0))
        return act

    def learn(self,obs,act,reward,next_obs,terminal):
        #每隔200更新目标网络参数
        if self.global_step % self.update_target_steps==0 and self.global_step!=0:
            self.algorithm.sync_target()
        self.global_step+=1

        act=np.expand_dims(act,-1)
        #构建用于训练的数据种子
        feed={
            'obs':obs.astype('float32'),
            'act':act.astype('int32'),
            'reward':reward,
            'next_obs':next_obs.astype('float32'),
            'terminal':terminal.astype('float32')

        }
        cost=self.fluid_executor.run(
            self.learn_program,feed=feed,fetch_list=[self.cost])[0]
        return cost




