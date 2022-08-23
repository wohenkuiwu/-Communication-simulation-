import paddle.fluid as fluid
import parl
from parl import layers

#网络的作用是输出所用动作对应的Q值
class Model(parl.Model):

    def __init__(self, act_dim):
        super().__init__()
        hid1_size=128
        hid2_size=64
        # hid3_size = 32
        # hid4_size=32
        #定义四层网络
        self.fc1=layers.fc(size=hid1_size,act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)
        # self.fc4 = layers.fc(size=hid4_size, act='relu')
        # self.fc5 =layers.fc(size=act_dim, act=None)


    def value(self,obs):
        h1=self.fc1(obs)
        h2=self.fc2(h1)
        Q=self.fc3(h2)
        # h4= self.fc4(h3)
        # Q=self.fc5(h4)


        return Q

