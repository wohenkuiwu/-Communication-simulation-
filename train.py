import numpy as np
from parl.utils import logger
from model import Model
from algorithm import DQN
from agent import Agent
from replay_memory import ReplayMemory
from env import  Environment
import matplotlib.pyplot as plt
import warnings
from visualdl import LogWriter
import json
warnings.simplefilter("ignore", ResourceWarning)

LEARN_FREQ = 10  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 200000 # replay memory的大小

"batch_size还是要尽可能的大一些，可以打乱数据之间的依赖性，可以让reward表现的更好"
MEMORY_WARMUP_SIZE =5000 # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn

BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.0001  # 学习率

"超参数伽马值越大，越关注未来的回报,loss的起始点越好，但是reward会不好"
GAMMA = 0.9

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def find(number):
    """找到动作列表中出现次数最多的动作"""
    maxnumber=max(number,key=number.count)
    return maxnumber


"存储主模型的经验"
def run_episode(env,agent_1,agent_2,agent_3,agent_4,agent_5,agent_6,agent_7,agent_8,
                rpm_1,rpm_2,rpm_3,rpm_4,rpm_5,rpm_6,rpm_7,rpm_8,episode):
# def run_episode(env, agent_1,rpm_1, rpm_2, rpm_3, rpm_4, rpm_5, rpm_6, rpm_7, rpm_8):

    step=1
    all_reward=0
    obs ,obs_norm= env.reset()
    while step<=50:


        action=[]
        step += 1
        all_reward=0

        # obs_norm=obs_norm.reshape((8,14))
        obs_1_norm = obs_norm[0, :]
        obs_2_norm = obs_norm[1, :]
        obs_3_norm = obs_norm[2, :]
        obs_4_norm = obs_norm[3, :]
        obs_5_norm = obs_norm[4, :]
        obs_6_norm = obs_norm[5, :]
        obs_7_norm = obs_norm[6, :]
        obs_8_norm = obs_norm[7, :]

        # obs_norm=np.array(obs_norm).reshape(-1)

        "第一个智能体交互得到数据"
        obs_1=obs[0,:]
        # action_1=agent_1.sample(obs_1_norm)
        action_1 = agent_1.sample(obs_norm)


        "第二个智能体交互得到数据"
        obs_2=obs[1,:]
        # action_2=agent_2.sample(obs_2_norm)
        action_2 = agent_2.sample(obs_norm)


        "第三个智能体交互得到数据"
        obs_3=obs[2,:]
        # action_3=agent_3.sample(obs_3_norm)
        action_3 = agent_3.sample(obs_norm)


        "第四个智能体交互得到数据"
        obs_4=obs[3,:]
        # action_4=agent_4.sample(obs_4_norm)
        action_4 = agent_4.sample(obs_norm)


        "第五个智能体交互得到数据"
        obs_5=obs[4,:]
        # action_5=agent_5.sample(obs_5_norm)
        action_5 = agent_5.sample(obs_norm)


        "第六个智能体交互得到数据"
        obs_6=obs[5,:]
        # action_6=agent_6.sample(obs_6_norm)
        action_6 = agent_6.sample(obs_norm)


        "第七个智能体交互得到数据"
        obs_7=obs[6,:]
        # action_7=agent_7.sample(obs_7_norm)
        action_7 = agent_7.sample(obs_norm)


        "第八个智能体交互得到数据"
        obs_8=obs[7,:]
        # action_8=agent_8.sample(obs_8_norm)
        action_8 = agent_8.sample(obs_norm)


        action.append(action_1)
        action.append(action_2)
        action.append(action_3)
        action.append(action_4)
        action.append(action_5)
        action.append(action_6)
        action.append(action_7)
        action.append(action_8)

        "所有智能体统一与环境交互"
        # reward,next_obs,next_obs_norm,rrr,fitness,iter_num=env.step(action,obs,episode)
        reward, next_obs, next_obs_norm = env.step(action, obs, episode)


        if step==50:
            done=1
        else:
            done=0
        all_reward=sum(reward)

        "判断对应最好的reward的智能体"
        # bset_agent=reward.index(max(reward))            #返回最大值对应的索引

        next_obs_1_norm = next_obs_norm[0, :]
        next_obs_2_norm= next_obs_norm [1, :]
        next_obs_3_norm = next_obs_norm[2, :]
        next_obs_4_norm= next_obs_norm [3, :]
        next_obs_5_norm = next_obs_norm[4, :]

        next_obs_6_norm = next_obs_norm[5, :]
        next_obs_7_norm = next_obs_norm[6, :]
        next_obs_8_norm = next_obs_norm[7, :]

        # rpm_best.append((obs_norm[bset_agent, :], action[bset_agent], reward[bset_agent], next_obs_norm[bset_agent, :]))

        # next_obs_norm=np.array(next_obs_norm).reshape(-1)
        # rpm_1.append((obs_1_norm, action_1, reward[0], next_obs_1_norm,done))
        # rpm_2.append((obs_2_norm, action_2, reward[1], next_obs_2_norm,done))
        # rpm_3.append((obs_3_norm, action_3, reward[2], next_obs_3_norm,done))
        # rpm_4.append((obs_4_norm, action_4, reward[3], next_obs_4_norm,done))
        # rpm_5.append((obs_5_norm, action_5, reward[4], next_obs_5_norm,done))
        # rpm_6.append((obs_6_norm, action_6, reward[5], next_obs_6_norm,done))
        # rpm_7.append((obs_7_norm, action_7, reward[6], next_obs_7_norm,done))
        # rpm_8.append((obs_8_norm, action_8, reward[7], next_obs_8_norm,done))

        rpm_1.append((obs_norm, action_1, all_reward, next_obs_norm,done))
        rpm_2.append((obs_norm, action_2, all_reward, next_obs_norm,done))
        rpm_3.append((obs_norm, action_3, all_reward, next_obs_norm,done))
        rpm_4.append((obs_norm, action_4, all_reward, next_obs_norm,done))
        rpm_5.append((obs_norm, action_5, all_reward, next_obs_norm,done))
        rpm_6.append((obs_norm, action_6, all_reward, next_obs_norm,done))
        rpm_7.append((obs_norm, action_7, all_reward, next_obs_norm,done))
        rpm_8.append((obs_norm, action_8, all_reward, next_obs_norm,done))

        "每个智能体奖励是全体的奖励"
        # rpm_1.append((obs_1_norm, action_1, reward, next_obs_norm,done))
        # rpm_2.append((obs_2_norm, action_2, reward, next_obs_norm,done))
        # rpm_3.append((obs_3_norm, action_3, reward, next_obs_norm,done))
        # rpm_4.append((obs_4_norm, action_4, reward, next_obs_norm,done))
        # rpm_5.append((obs_5_norm, action_5, reward, next_obs_norm,done))
        # rpm_6.append((obs_6_norm, action_6, reward, next_obs_norm,done))
        # rpm_7.append((obs_7_norm, action_7, reward, next_obs_norm,done))
        # rpm_8.append((obs_8_norm, action_8, reward, next_obs_norm,done))

        obs=next_obs
        obs_norm = next_obs_norm

"主框架的训练"
def run_episode2(env,agent_1,agent_2,agent_3,agent_4,agent_5,agent_6,agent_7,agent_8,
                    rpm_1,rpm_2,rpm_3,rpm_4,rpm_5,rpm_6,rpm_7,rpm_8,episode):
# def run_episode2(env, agent_1, rpm_1, rpm_2, rpm_3, rpm_4, rpm_5, rpm_6, rpm_7, rpm_8):

    total_e_local=0
    total_reward=0
    total_reward_random=0
    fitness_best=0
    iteration_reward=0
    total_reward_0=  0
    total_reward_1 = 0
    total_reward_2 = 0
    total_reward_3 = 0
    total_reward_4 = 0
    total_reward_5 = 0
    total_reward_6 = 0
    total_reward_7 = 0

    obs,obs_norm=env.reset()
    step=1
    total_loss=0
    total_loss_1=0
    total_loss_2 = 0
    total_loss_3 = 0
    total_loss_4 = 0
    total_loss_5 = 0
    total_loss_6 = 0
    total_loss_7 = 0
    total_loss_8 = 0

    e_local_0=0
    e_local_1=0
    e_local_2=0
    e_local_3=0
    e_local_4=0
    e_local_5=0
    e_local_6=0
    e_local_7=0

    e_mec_0 = 0
    e_mec_1 = 0
    e_mec_2 = 0
    e_mec_3 = 0
    e_mec_4 = 0
    e_mec_5 = 0
    e_mec_6 = 0
    e_mec_7 = 0

    while step<=50:

        best_action_1 = []
        best_action_2 = []
        best_action_3 = []
        best_action_4 = []
        best_action_5 = []
        best_action_6 = []
        best_action_7 = []
        best_action_8 = []

        action=[]
        step+=1
        all_reward=0

        if step==50:
            done=1
        else:
            done=0

        f_lcoal=[]
        e_local=[]
        "计算全本地的能耗"
        for i in range(8):
            local=10**9
            f_lcoal.append(local)

        for i in range(8):
            e=-(10**(-27)*(f_lcoal[i]**2)*obs[i][1])
            e_local.append(e)

        obs_1_norm = obs_norm[0, :]
        obs_2_norm = obs_norm[1, :]
        obs_3_norm = obs_norm[2, :]
        obs_4_norm = obs_norm[3, :]
        obs_5_norm = obs_norm[4, :]
        obs_6_norm = obs_norm[5, :]
        obs_7_norm = obs_norm[6, :]
        obs_8_norm = obs_norm[7, :]

        # obs_norm=np.array(obs_norm).reshape(-1)

        "第一个智能体交互得到数据"
        obs_1=obs[0,:]
        # action_1=agent_1.sample(obs_1_norm)
        action_1 = agent_1.sample(obs_norm)

        "第二个智能体交互得到数据"
        obs_2=obs[1,:]
        # action_2=agent_2.sample(obs_2_norm)
        action_2 = agent_2.sample(obs_norm)

        "第三个智能体交互得到数据"
        obs_3=obs[2,:]
        # action_3=agent_3.sample(obs_3_norm)
        action_3 = agent_3.sample(obs_norm)

        "第四个智能体交互得到数据"
        obs_4=obs[3,:]
        # action_4=agent_4.sample(obs_4_norm)
        action_4 = agent_4.sample(obs_norm)

        "第五个智能体交互得到数据"
        obs_5=obs[4,:]
        # action_5=agent_5.sample(obs_5_norm)
        action_5 = agent_5.sample(obs_norm)

        "第六个智能体交互得到数据"
        obs_6=obs[5,:]
        # action_6=agent_6.sample(obs_6_norm)
        action_6 = agent_6.sample(obs_norm)

        "第七个智能体交互得到数据"
        obs_7=obs[6,:]
        # action_7=agent_7.sample(obs_7_norm)
        action_7 = agent_7.sample(obs_norm)

        "第八个智能体交互得到数据"
        obs_8=obs[7,:]
        # action_8=agent_8.sample(obs_8_norm)
        action_8 = agent_8.sample(obs_norm)

        action.append(action_1)
        action.append(action_2)
        action.append(action_3)
        action.append(action_4)
        action.append(action_5)
        action.append(action_6)
        action.append(action_7)
        action.append(action_8)

        "所有智能体统一与环境交互"
        "传出来的奖励是一个列表"
        # reward,next_obs,next_obs_norm,reward_random,fitness,iteration_number=env.step(action,obs,episode)
        reward, next_obs, next_obs_norm = env.step(action, obs, episode)
        all_reward=sum(reward)

        next_obs_1_norm = next_obs_norm[0,: ]
        next_obs_2_norm= next_obs_norm[1, :]
        next_obs_3_norm = next_obs_norm[2, :]
        next_obs_4_norm= next_obs_norm[3, :]
        next_obs_5_norm = next_obs_norm[4, :]
        next_obs_6_norm = next_obs_norm[5, :]
        next_obs_7_norm = next_obs_norm[6, :]
        next_obs_8_norm = next_obs_norm[7, :]

        # "判断对应最好的reward的智能体"
        # bset_agent = reward.index(max(reward))  # 返回最大值对应的索引
        # rpm_best.append((obs_norm[bset_agent, :], action[bset_agent], reward[bset_agent], next_obs_norm[bset_agent, :]))

        # next_obs_norm=np.array(next_obs_norm).reshape(-1)
        # rpm_1.append((obs_1_norm, action_1, reward[0], next_obs_1_norm,done))
        # rpm_2.append((obs_2_norm, action_2, reward[1], next_obs_2_norm,done))
        # rpm_3.append((obs_3_norm, action_3, reward[2], next_obs_3_norm,done))
        # rpm_4.append((obs_4_norm, action_4, reward[3], next_obs_4_norm,done))
        # rpm_5.append((obs_5_norm, action_5, reward[4], next_obs_5_norm,done))
        # rpm_6.append((obs_6_norm, action_6, reward[5], next_obs_6_norm,done))
        # rpm_7.append((obs_7_norm, action_7, reward[6], next_obs_7_norm,done))
        # rpm_8.append((obs_8_norm, action_8, reward[7], next_obs_8_norm,done))

        "经验池中存放的五元组，在队列中视为一条"
        rpm_1.append((obs_norm, action_1, all_reward, next_obs_norm,done))
        rpm_2.append((obs_norm, action_2, all_reward, next_obs_norm,done))
        rpm_3.append((obs_norm, action_3, all_reward, next_obs_norm,done))
        rpm_4.append((obs_norm, action_4, all_reward, next_obs_norm,done))
        rpm_5.append((obs_norm, action_5, all_reward, next_obs_norm,done))
        rpm_6.append((obs_norm, action_6, all_reward, next_obs_norm,done))
        rpm_7.append((obs_norm, action_7, all_reward, next_obs_norm,done))
        rpm_8.append((obs_norm, action_8, all_reward, next_obs_norm,done))


        # rpm_1.append((obs_1_norm, action_1, reward, next_obs_norm,done))
        # rpm_2.append((obs_2_norm, action_2, reward, next_obs_norm,done))
        # rpm_3.append((obs_3_norm, action_3, reward, next_obs_norm,done))
        # rpm_4.append((obs_4_norm, action_4, reward, next_obs_norm,done))
        # rpm_5.append((obs_5_norm, action_5, reward, next_obs_norm,done))
        # rpm_6.append((obs_6_norm, action_6, reward, next_obs_norm,done))
        # rpm_7.append((obs_7_norm, action_7, reward, next_obs_norm,done))
        # rpm_8.append((obs_8_norm, action_8, reward, next_obs_norm,done))

        "训练数据"
        if (len(rpm_1)>MEMORY_WARMUP_SIZE) and(step%LEARN_FREQ==0): #预先存放一些经验

            (batch_obs_1,batch_action_1,batch_reward_1,batch_next_obs_1,batch_done_1)=rpm_1.sample(BATCH_SIZE)
            train_loss_1=agent_1.learn(batch_obs_1,batch_action_1,batch_reward_1,batch_next_obs_1,batch_done_1)

            (batch_obs_2,batch_action_2,batch_reward_2,batch_next_obs_2,batch_done_2)=rpm_2.sample(BATCH_SIZE)
            train_loss_2=agent_2.learn(batch_obs_2,batch_action_2,batch_reward_2,batch_next_obs_2,batch_done_2)

            (batch_obs_3, batch_action_3, batch_reward_3, batch_next_obs_3,batch_done_3) = rpm_3.sample(BATCH_SIZE)
            train_loss_3 = agent_3.learn(batch_obs_3, batch_action_3, batch_reward_3, batch_next_obs_3,batch_done_3)

            (batch_obs_4, batch_action_4, batch_reward_4, batch_next_obs_4,batch_done_4) = rpm_4.sample(BATCH_SIZE)
            train_loss_4 = agent_4.learn (batch_obs_4, batch_action_4, batch_reward_4, batch_next_obs_4,batch_done_4)

            (batch_obs_5, batch_action_5, batch_reward_5, batch_next_obs_5,batch_done_5) = rpm_5.sample(BATCH_SIZE)
            train_loss_5 = agent_5.learn (batch_obs_5, batch_action_5, batch_reward_5, batch_next_obs_5,batch_done_5)

            (batch_obs_6, batch_action_6, batch_reward_6, batch_next_obs_6,batch_done_6) = rpm_6.sample(BATCH_SIZE)
            train_loss_6 = agent_6.learn (batch_obs_6, batch_action_6, batch_reward_6, batch_next_obs_6,batch_done_6)

            (batch_obs_7, batch_action_7, batch_reward_7, batch_next_obs_7,batch_done_7) = rpm_7.sample(BATCH_SIZE)
            train_loss_7 = agent_7.learn (batch_obs_7, batch_action_7, batch_reward_7, batch_next_obs_7,batch_done_7)

            (batch_obs_8, batch_action_8, batch_reward_8, batch_next_obs_8,batch_done_8) = rpm_8.sample(BATCH_SIZE)
            train_loss_8 = agent_8.learn (batch_obs_8, batch_action_8, batch_reward_8, batch_next_obs_8,batch_done_8)

            total_loss_1+=  train_loss_1
            total_loss_2 += train_loss_2
            total_loss_3 += train_loss_3
            total_loss_4 += train_loss_4
            total_loss_5 += train_loss_5
            total_loss_6 += train_loss_6
            total_loss_7 += train_loss_7
            total_loss_8 += train_loss_8
            # total_loss+=(train_loss_1+train_loss_2+train_loss_3+train_loss_4+train_loss_5+train_loss_6+train_loss_7+train_loss_8)/8

        total_e_local+=sum(e_local)
        e_local_0+=e_local[0]
        e_local_1+=e_local[1]
        e_local_2+=e_local[2]
        e_local_3+=e_local[3]
        e_local_4+=e_local[4]
        e_local_5+=e_local[5]
        e_local_6+=e_local[6]
        e_local_7+=e_local[7]

        # e_mec_0+=reward_random[0]
        # e_mec_1+=reward_random[1]
        # e_mec_2+=reward_random[2]
        # e_mec_3+=reward_random[3]
        # e_mec_4+=reward_random[4]
        # e_mec_5+=reward_random[5]
        # e_mec_6+=reward_random[6]
        # e_mec_7+=reward_random[7]

        # total_reward_random+=sum(reward_random)
        total_reward+=sum(reward)
        # fitness_best += sum(fitness)
        # iteration_reward+=sum(iteration_number)
        total_reward_0+=reward[0]
        total_reward_1+=reward[1]
        total_reward_2+=reward[2]
        total_reward_3+=reward[3]
        total_reward_4+=reward[4]
        total_reward_5+=reward[5]
        total_reward_6+=reward[6]
        total_reward_7+=reward[7]

        obs=next_obs
        obs_norm = next_obs_norm

        "判断最好的动作"
        best_action_1.append(action_1)
        best_action_2.append(action_2)
        best_action_3.append(action_3)
        best_action_4.append(action_4)
        best_action_5.append(action_5)
        best_action_6.append(action_6)
        best_action_7.append(action_7)
        best_action_8.append(action_8)


    # return total_reward/step,total_loss_1/5,total_loss_2/5,total_loss_3/5,\
    #        total_loss_4/5,total_loss_5/5,total_loss_6/5,total_loss_7/5,total_loss_8/5,\
    #        total_reward_0/step,total_reward_1/step,total_reward_2/step,total_reward_3/step,\
    #        total_reward_4/step,total_reward_5/step,total_reward_6/step,total_reward_7/step,\
    #        total_reward_random/step,total_e_local/step,e_local_0/step,e_local_1/step,e_local_2/step,\
    #        e_local_3/step,e_local_4/step,e_local_5/step,e_local_6/step,e_local_7/step,\
    #        e_mec_0/step,e_mec_1/step,e_mec_2/step,e_mec_3/step,e_mec_4/step,e_mec_5/step,e_mec_6/step,e_mec_7/step,fitness_best/step,iteration_reward/step
    return total_reward/step,total_loss/5,total_e_local/step,best_action_1,best_action_2,best_action_3,\
           best_action_4,best_action_5,best_action_6,best_action_7,best_action_8

def main():
    #生成对象
    loss_loss_1=[]
    loss_loss_2 = []
    loss_loss_3 = []
    loss_loss_4 = []
    loss_loss_5 = []
    loss_loss_6 = []
    loss_loss_7 = []
    loss_loss_8 = []
    reward_reward_0=[]
    reward_reward_1 = []
    reward_reward_2 = []
    reward_reward_3 = []
    reward_reward_4 = []
    reward_reward_5 = []
    reward_reward_6 = []
    reward_reward_7 = []
    reward_random=[]
    local_0 = []
    local_1 = []
    local_2 = []
    local_3 = []
    local_4 = []
    local_5 = []
    local_6 = []
    local_7 = []

    mec_0 = []
    mec_1 = []
    mec_2 = []
    mec_3 = []
    mec_4 = []
    mec_5 = []
    mec_6 = []
    mec_7 = []

    best_action_1=[]
    best_action_2 = []
    best_action_3 = []
    best_action_4 = []
    best_action_5 = []
    best_action_6 = []
    best_action_7 = []
    best_action_8 = []

    action_dim=13
    obs_shape=112
    # obs_shape = 14

    "每个智能体都有自己交互的经验池"
    rpm_1 = ReplayMemory(MEMORY_SIZE)
    rpm_2 = ReplayMemory(MEMORY_SIZE)
    rpm_3 = ReplayMemory(MEMORY_SIZE)
    rpm_4 = ReplayMemory(MEMORY_SIZE)
    rpm_5 = ReplayMemory(MEMORY_SIZE)
    rpm_6 = ReplayMemory(MEMORY_SIZE)
    rpm_7 = ReplayMemory(MEMORY_SIZE)
    rpm_8 = ReplayMemory(MEMORY_SIZE)
    # rpm_best=ReplayMemory(MEMORY_SIZE)

    "搭建主框架"
    model_1=  Model(act_dim=action_dim)
    model_2 = Model(act_dim=action_dim)
    model_3 = Model(act_dim=action_dim)
    model_4 = Model(act_dim=action_dim)
    model_5 = Model(act_dim=action_dim)
    model_6 = Model(act_dim=action_dim)
    model_7 = Model(act_dim=action_dim)
    model_8 = Model(act_dim=action_dim)

    algorithm_1=  DQN(model_1,act_dim=action_dim,gamma=GAMMA,lr=LEARNING_RATE)
    algorithm_2=  DQN(model_2,act_dim=action_dim,gamma=GAMMA,lr=LEARNING_RATE)
    algorithm_3 = DQN(model_3, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    algorithm_4 = DQN(model_4, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    algorithm_5 = DQN(model_5, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    algorithm_6 = DQN(model_6, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    algorithm_7 = DQN(model_7, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    algorithm_8 = DQN(model_8, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)

    "创建多个智能体，每个智能体对应一个用户，学习自己的动作，更新同一个网络"
    agent_1 = Agent(algorithm_1, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)
    agent_2 = Agent(algorithm_2, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)
    agent_3 = Agent(algorithm_3, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)
    agent_4 = Agent(algorithm_4, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)
    agent_5 = Agent(algorithm_5, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)
    agent_6 = Agent(algorithm_6, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)
    agent_7 = Agent(algorithm_7, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)
    agent_8 = Agent(algorithm_8, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)
    # agent_best =Agent(algorithm, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)

    #预加载主模型数据大小，不达标则一直加载与环境交互得到的数据
    while len(rpm_1)<MEMORY_WARMUP_SIZE:
        env = Environment()
        run_episode(env,agent_1,agent_2,agent_3,agent_4,agent_5,agent_6,agent_7,agent_8,
                    rpm_1,rpm_2,rpm_3,rpm_4,rpm_5,rpm_6,rpm_7,rpm_8,1000)
        # run_episode(env,agent_1,rpm_1,rpm_2,rpm_3,rpm_4,rpm_5,rpm_6,rpm_7,rpm_8)
  
    max_episode=3000

    #开始训练
    episode=1
    total_list_reward=[]
    total_e_local=[]
    total_fitness=[]
    iteration_iteration=[]

    "训练的主循环"
    while episode<max_episode:

        "主模型的训练"
        env = Environment()
        # total_reward,loss_1,loss_2,loss_3,loss_4,loss_5,loss_6,loss_7,loss_8,\
        # reward_0,reward_1,reward_2,reward_3,reward_4,reward_5,reward_6,reward_7,reward_r,\
        # e_local,e_local_0,e_local_1,e_local_2,e_local_3,e_local_4,e_local_5,e_local_6,e_local_7,e_mec_0,e_mec_1,e_mec_2,e_mec_3,e_mec_4,e_mec_5,e_mec_6,e_mec_7\
        #     ,fitness_best,iteration_number\
        #     =run_episode2\
        #     (env,agent_1,agent_2,agent_3,agent_4,agent_5,agent_6,agent_7,agent_8,
        #             rpm_1,rpm_2,rpm_3,rpm_4,rpm_5,rpm_6,rpm_7,rpm_8,episode)
        total_reward,loss_1,e_local,action_1,action_2,action_3,action_4,\
            action_5,action_6,action_7,action_8=run_episode2(env,agent_1,agent_2,agent_3,agent_4,agent_5,agent_6,agent_7,agent_8,
                    rpm_1,rpm_2,rpm_3,rpm_4,rpm_5,rpm_6,rpm_7,rpm_8,episode)

        #判断动作
        best_1=find(action_1)
        best_action_1.append(best_1)
        print(best_1)

        best_2=find(action_2)
        best_action_2.append(best_2)

        best_3=find(action_3)
        best_action_3.append(best_3)

        best_4=find(action_4)
        best_action_4.append(best_4)

        best_5=find(action_5)
        best_action_5.append(best_5)

        best_6=find(action_6)
        best_action_6.append(best_6)

        best_7=find(action_7)
        best_action_7.append(best_7)

        best_8=find(action_8)
        best_action_8.append(best_8)


        episode += 1
        logger.info('episode:{}   '.format(
            episode))

        # best_action_1.append(action_1)
        # best_action_2.append(action_2)
        # best_action_3.append(action_3)
        # best_action_4.append(action_4)
        # best_action_5.append(action_5)
        # best_action_6.append(action_6)
        # best_action_7.append(action_7)
        # best_action_8.append(action_8)


        #所有用户的总回报
        # total_fitness.append(fitness_best)
        # reward_random.append(reward_r)
        total_list_reward.append(total_reward)
        # iteration_iteration.append(iteration_number)
        loss_loss_1.append(loss_1)
        # loss_loss_2.append(loss_2)
        # loss_loss_3.append(loss_3)
        # loss_loss_4.append(loss_4)
        # loss_loss_5.append(loss_5)
        # loss_loss_6.append(loss_6)
        # loss_loss_7.append(loss_7)
        # loss_loss_8.append(loss_8)
        # reward_reward_0.append(reward_0)
        # reward_reward_1.append(reward_1)
        # reward_reward_2.append(reward_2)
        # reward_reward_3.append(reward_3)
        # reward_reward_4.append(reward_4)
        # reward_reward_5.append(reward_5)
        # reward_reward_6.append(reward_6)
        # reward_reward_7.append(reward_7)

        total_e_local.append(e_local)
        # local_0.append(e_local_0)
        # local_1.append(e_local_1)
        # local_2.append(e_local_2)
        # local_3.append(e_local_3)
        # local_4.append(e_local_4)
        # local_5.append(e_local_5)
        # local_6.append(e_local_6)
        # local_7.append(e_local_7)

        # mec_0.append(e_mec_0)
        # mec_1.append(e_mec_1)
        # mec_2.append(e_mec_2)
        # mec_3.append(e_mec_3)
        # mec_4.append(e_mec_4)
        # mec_5.append(e_mec_5)
        # mec_6.append(e_mec_6)
        # mec_7.append(e_mec_7)

    #保存为josn文件
    # file_name='action_1.json'
    # with open(file_name,'w') as f:
    #     json.dump(best_action_1,f)
    #
    # file_name='action_2.json'
    # with open(file_name,'w') as f:
    #     json.dump(best_action_2,f)
    #
    # file_name = 'action_3.json'
    # with open(file_name, 'w') as f:
    #     json.dump(best_action_3, f)
    #
    # file_name='action_4.json'
    # with open(file_name,'w') as f:
    #     json.dump(best_action_4,f)
    #
    # file_name='action_5.json'
    # with open(file_name,'w') as f:
    #     json.dump(best_action_5,f)
    #
    # file_name='action_6.json'
    # with open(file_name,'w') as f:
    #     json.dump(best_action_6,f)
    #
    # file_name='action_7.json'
    # with open(file_name,'w') as f:
    #     json.dump(best_action_7,f)
    #
    # file_name='action_8.json'
    # with open(file_name,'w') as f:
    #     json.dump(best_action_8,f)


    # 初始化一个记录器
    "b不同的书记只能写到不同的文件下面"
    "不同的小文件名会有不同的颜色曲线"
    "命令行运行时，运行大文件"
    with LogWriter(logdir="./log/action") as writer:
        for step in range(max_episode-1):
            # 向记录器添加一个tag为`loss`的数据
            # writer.add_scalar(tag="loss", step=step, value=loss_loss[step])
            writer.add_scalar(tag="action_0", step=step, value=best_action_1[step])
            writer.add_scalar(tag="action_1", step=step, value=best_action_2[step])
            writer.add_scalar(tag="action_2", step=step, value=best_action_3[step])
            writer.add_scalar(tag="action_3", step=step, value=best_action_4[step])
            writer.add_scalar(tag="action_4", step=step, value=best_action_5[step])
            writer.add_scalar(tag="action_5", step=step, value=best_action_6[step])
            writer.add_scalar(tag="action_6", step=step, value=best_action_7[step])
            writer.add_scalar(tag="action_7", step=step, value=best_action_8[step])

    # with LogWriter(logdir="./log/fitness") as writer:
    #     for step in range(max_episode-1):
    #         # 向记录器添加一个tag为`loss`的数据
    #         # writer.add_scalar(tag="loss", step=step, value=loss_loss[step])
    #         writer.add_scalar(tag="total_reward", step=step, value=total_fitness[step])

    # with LogWriter(logdir="./log/iter") as writer:
    #     for step in range(max_episode-1):
    #         # 向记录器添加一个tag为`loss`的数据
    #         # writer.add_scalar(tag="loss", step=step, value=loss_loss[step])
    #         writer.add_scalar(tag="total_reward", step=step, value=iteration_iteration[step])


    with LogWriter(logdir="./log/8_dqn_all_reward") as writer:
        for step in range(max_episode-1):
            # 向记录器添加一个tag为`loss`的数据
            writer.add_scalar(tag="loss_1", step=step, value=loss_loss_1[step])
            # writer.add_scalar(tag="loss_2", step=step, value=loss_loss_2[step])
            # writer.add_scalar(tag="loss_3", step=step, value=loss_loss_3[step])
            # writer.add_scalar(tag="loss_4", step=step, value=loss_loss_4[step])
            # writer.add_scalar(tag="loss_5", step=step, value=loss_loss_5[step])
            # writer.add_scalar(tag="loss_6", step=step, value=loss_loss_6[step])
            # writer.add_scalar(tag="loss_7", step=step, value=loss_loss_7[step])
            # writer.add_scalar(tag="loss_8", step=step, value=loss_loss_8[step])
            writer.add_scalar(tag="total_reward", step=step, value=total_list_reward[step])
            # writer.add_scalar(tag="reward_0", step=step, value=reward_reward_0[step])
            # writer.add_scalar(tag="reward_1", step=step, value=reward_reward_1[step])
            # writer.add_scalar(tag="reward_2", step=step, value=reward_reward_2[step])
            # writer.add_scalar(tag="reward_3", step=step, value=reward_reward_3[step])
            # writer.add_scalar(tag="reward_4", step=step, value=reward_reward_4[step])
            # writer.add_scalar(tag="reward_5", step=step, value=reward_reward_5[step])
            # writer.add_scalar(tag="reward_6", step=step, value=reward_reward_6[step])
            # writer.add_scalar(tag="reward_7", step=step, value=reward_reward_7[step])

    # with LogWriter(logdir="./log/local") as writer:
    #     for step in range(max_episode-1):
    #         # 向记录器添加一个tag为`loss`的数据
    #         #writer.add_scalar(tag="loss", step=step, value=loss_loss[step])
    #         writer.add_scalar(tag="total_reward", step=step, value=total_e_local[step])
    #         # writer.add_scalar(tag="reward_0", step=step, value=local_0[step])
    #         # writer.add_scalar(tag="reward_1", step=step, value=local_1[step])
    #         # writer.add_scalar(tag="reward_2", step=step, value=local_2[step])
    #         # writer.add_scalar(tag="reward_3", step=step, value=local_3[step])
    #         # writer.add_scalar(tag="reward_4", step=step, value=local_4[step])
    #         # writer.add_scalar(tag="reward_5", step=step, value=local_5[step])
    #         # writer.add_scalar(tag="reward_6", step=step, value=local_6[step])
    #         # writer.add_scalar(tag="reward_7", step=step, value=local_7[step])


    # "画图"
    # plt.figure(1)
    # xxx=[i for i in range(max_episode-1)]
    # plt.xlabel("episode")
    # plt.ylabel('Total energy consumption')
    # plt.plot(xxx, total_list_reward)
    # plt.plot(xxx, total_e_local)
    # plt.plot(xxx, reward_random)
    # plt.legend(["Optimization method", "local","offload"])
    #
    # plt.figure(2)
    # plt.xlabel("episode")
    # plt.ylabel('Total energy consumption')
    # plt.plot(xxx, reward_reward_0)
    # plt.plot(xxx, local_0)
    # plt.plot(xxx, mec_0)
    # plt.legend(["Optimization method", "local","offload"])
    #
    # plt.figure(3)
    # plt.xlabel("episode")
    # plt.ylabel('Total energy consumption')
    # plt.plot(xxx, reward_reward_1)
    # plt.plot(xxx, local_1)
    # plt.plot(xxx, mec_1)
    # plt.legend(["Optimization method", "local","offload"])
    #
    # plt.figure(4)
    # plt.xlabel("episode")
    # plt.ylabel('Total energy consumption')
    # plt.plot(xxx, reward_reward_2)
    # plt.plot(xxx, local_2)
    # plt.plot(xxx, mec_2)
    # plt.legend(["Optimization method", "local","offload"])
    #
    # plt.figure(5)
    # plt.xlabel("episode")
    # plt.ylabel('Total energy consumption')
    # plt.plot(xxx, reward_reward_3)
    # plt.plot(xxx, local_3)
    # plt.plot(xxx, mec_3)
    # plt.legend(["Optimization method", "local","offload"])
    #
    # plt.figure(6)
    # plt.xlabel("episode")
    # plt.ylabel('Total energy consumption')
    # plt.plot(xxx, reward_reward_4)
    # plt.plot(xxx, local_4)
    # plt.plot(xxx, mec_4)
    # plt.legend(["Optimization method", "local","offload"])
    #
    # plt.figure(7)
    # plt.xlabel("episode")
    # plt.ylabel('Total energy consumption')
    # plt.plot(xxx, reward_reward_5)
    # plt.plot(xxx, local_5)
    # plt.plot(xxx, mec_5)
    # plt.legend(["Optimization method", "local","offload"])
    #
    # plt.figure(8)
    # plt.xlabel("episode")
    # plt.ylabel('Total energy consumption')
    # plt.plot(xxx, reward_reward_6)
    # plt.plot(xxx, local_6)
    # plt.plot(xxx, mec_6)
    # plt.legend(["Optimization method", "local","offload"])
    #
    # plt.figure(9)
    # plt.xlabel("episode")
    # plt.ylabel('Total energy consumption')
    # plt.plot(xxx, reward_reward_7)
    # plt.plot(xxx, local_7)
    # plt.plot(xxx, mec_7)
    # plt.legend(["Optimization method", "local","offload"])
    #
    # plt.show()

    # "保存模型"
    # save_path = './dqn_model.capt'
    # agent.save(save_path)

if __name__ == '__main__':
    main()
