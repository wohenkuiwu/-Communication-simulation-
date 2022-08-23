import numpy as np

class Environment(object):

    def __init__(self):
        self.w=10**6
        self.f_s_mec=2.5*10**9
        self.f_m_mec=2.5*10**10
        self.noise=9.99999999999996e-11
        # self.noise=3.9810717055349565e-21
        self.p_max=0.19952623149688797
        self.pathloss=3.7
        self.k=10**(-27)
        self.d_max=0.5

        self.c1 = 1.49445
        self.c2 = 1.49445
        self.ws = 0.9
        self.we = 0.4

    #返回状态值，每次调用状态值不一样,不需要指向对象的内存空间
    def reset(self):
        obsvation=[]
        for i in range(8):
            A_in=np.random.uniform(100*1024,200*1024)
            A_com=np.random.uniform(100*1024*150,200*1024*150)

            "用户坐标"
            d_x=np.random.uniform(1,500)
            d_y=np.random.uniform(1,500)
            "生成第一个基站内四条信道的信道增益"
            g_00=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-250)**2)**0.5)**(-self.pathloss))
            g_01=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-250)**2)**0.5)**(-self.pathloss))
            g_02=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-250)**2)**0.5)**(-self.pathloss))
            g_03=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-250)**2)**0.5)**(-self.pathloss))
            "生成第二个基站内四条信道的信道增益"
            g_10=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-375)**2)**0.5)** (-self.pathloss))
            g_11=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-375)**2)**0.5)** (-self.pathloss))
            g_12=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-375)**2)**0.5)** (-self.pathloss))
            g_13=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-375)**2)**0.5)** (-self.pathloss))
            "生成第三个基站内四条信道的信道增益"
            g_20=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-125)**2)**0.5)**(-self.pathloss))
            g_21=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-125)**2)**0.5)**(-self.pathloss))
            g_22=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-125)**2)**0.5)**(-self.pathloss))
            g_23=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-125)**2)**0.5)**(-self.pathloss))

            obsvation.append(A_in)
            obsvation.append(A_com)
            obsvation.append(g_00)
            obsvation.append(g_01)
            obsvation.append(g_02)
            obsvation.append(g_03)

            obsvation.append(g_10)
            obsvation.append(g_11)
            obsvation.append(g_12)
            obsvation.append(g_13)

            obsvation.append(g_20)
            obsvation.append(g_21)
            obsvation.append(g_22)
            obsvation.append(g_23)
        obsvation=np.array(obsvation).reshape((8,14))

        "对数组进行归一化处理"

        "找到每一列的最大值和最小值"
        max_max=[]
        min_min=[]
        max_number=[]
        for i in range(14):
            max_number=[]
            for j in range(8):
                number=obsvation[j][i]
                max_number.append(number)
            value_max=max(max_number)
            value_min=min(max_number)
            max_max.append(value_max)
            min_min.append(value_min)

        obsvation_norm=np.array([0.5 for i in range(112)]).reshape((8,14))

        for i in range(8):
            for j in range(14):
                obsvation_norm[i][j]=(obsvation[i][j]-min_min[j])/(max_max[j]-min_min[j])

        return obsvation,obsvation_norm

    def step(self,action,obs,episode):
        result = np.array(obs).reshape((8, 14))

        A_in_list=[]
        A_com_list=[]

        # for i in range(8):
        #     # A_in=np.random.uniform(100*1024,200*1024)
        #     # A_com=np.random.uniform(100*1024*150,200*1024*150)
        #     # A_in_list.append(A_in)
        #     # A_com_list.append(A_com)

        # result=np.column_stack((A_com_list,result))
        # result = np.column_stack((A_in_list, result))

        "所有用户连接基站数目的列表"
        a=[0 for i in range(8)]
        "所有用户选择信道数目的列表"
        b=[0 for i in range(8)]
        "生成二进制卸载列表"
        xx = [0 for i in range(8)]

        "分解所有动作为连接基站和信道的选择"
        for i in range(8):
            if action[i]%3==0 and action[i]%4==0 and action[i]!=0:
                xx[i]=0
            else:
                xx[i]=1
            a[i]=action[i]%3                      #连接的基站数目
            b[i]=(action[i]//3)%4                 #选择的信道数目

        p=[0.5 for i in range(96)]
        p=np.array(p).reshape((8,12))

        "得到每个用户对每个基站不同信道的最小发射功率"
        for i in range(8):
            for j in range(12):
                if j<=3:
                    # p[i][j]=((2**(result[i][0]/self.w*(self.d_max-(result[i][1]/self.f_m_mec)))-1)*self.noise)/result[i][j+2]
                    p[i][j]=self.p_max
                else:
                    # p[i][j] = ((2 ** (result[i][0] / self.w * (
                    #             self.d_max - (result[i][1] / self.f_s_mec))) - 1) * self.noise )/ result[i][j + 2]
                    p[i][j]=self.p_max

        "计算在MBS卸载时，小小区复用信道产生的干扰"

        mbs_0=[x for x,y in list(enumerate(a)) if y==0 and xx[x]!=0] #在宏基站卸载的用户索引
        sbs_1 = [x for x, y in list(enumerate(a)) if y == 1 and xx[x] != 0]  # 在小区基站1卸载的用户索引
        sbs_2 = [x for x, y in list(enumerate(a)) if y == 2 and xx[x] != 0]  # 在小区基站2卸载的用户索引
        channel_0 = [x for x, y in list(enumerate(b)) if y == 0 and xx[x] != 0 and a[x]!=0]  # 小小区基站选择信道0的用户索引
        channel_1 = [x for x, y in list(enumerate(b)) if y == 1 and xx[x] != 0 and a[x]!=0]  # 小小区基站选择信道1的用户索引
        channel_2 = [x for x, y in list(enumerate(b)) if y == 2 and xx[x] != 0and a[x]!=0]   # 小小区基站选择信道2的用户索引
        channel_3 = [x for x, y in list(enumerate(b)) if y == 3 and xx[x] != 0 and a[x]!=0]  # 小小区基站选择信道3的用户索引

        noise_in_mbs_channel_0=[0 for i in range(8)]
        noise_in_mbs_channel_1=[0 for i in range(8)]
        noise_in_mbs_channel_2=[0 for i in range(8)]
        noise_in_mbs_channel_3=[0 for i in range(8)]

        "计算在mbs卸载使用信道的用户，其它小小区使用相同信道的干扰"
        for i in range(0,len(mbs_0)):
            for j in range(0,len(channel_0),1):
                if channel_0[j] in sbs_1:
                    noise_in_mbs_channel_0[channel_0[j]]+=p[channel_0[j]][b[channel_0[j]]]*result[channel_0[j]][2+b[channel_0[j]]]
                if channel_0[j] in sbs_2:
                    noise_in_mbs_channel_0[channel_0[j]] += p[channel_0[j]][b[channel_0[j]]] * result[channel_0[j]][2 + b[channel_0[j]]]
            for j in range(0,len(channel_1),1):
                if channel_1[j] in sbs_1:
                    noise_in_mbs_channel_1[channel_1[j]]+=p[channel_1[j]][b[channel_1[j]]]*result[channel_1[j]][2+b[channel_1[j]]]
                if channel_1[j] in sbs_2:
                    noise_in_mbs_channel_1[channel_1[j]]+= p[channel_1[j]][b[channel_1[j]]] * result[channel_1[j]][2 + b[channel_1[j]]]
            for j in range(0,len(channel_2),1):
                if channel_2[j] in sbs_1:
                    noise_in_mbs_channel_2[channel_2[j]]+=p[channel_2[j]][b[channel_2[j]]]*result[channel_2[j]][2+b[channel_2[j]]]
                if channel_2[j] in sbs_2:
                    noise_in_mbs_channel_2[channel_2[j]] += p[channel_2[j]][b[channel_2[j]]] * result[channel_2[j]][2 + b[channel_2[j]]]
            for j in range(0,len(channel_3),1):
                if channel_3[j] in sbs_1:
                    noise_in_mbs_channel_3[channel_3[j]]+=p[channel_3[j]][b[channel_3[j]]]*result[channel_3[j]][2+b[channel_3[j]]]
                if channel_3[j] in sbs_2:
                    noise_in_mbs_channel_3[channel_3[j]] += p[channel_3[j]][b[channel_3[j]]] * result[channel_3[j]][2 + b[channel_3[j]]]

        "同基站同信道干扰"
        "SBS_1"
        noise_in_sbs_1 = [0 for i in range(8)]
        G_0=[]
        G_1 = []
        G_2 = []
        G_3 = []
        sbs_11=[x for x,y in list(enumerate(a)) if y==1 and xx[x]!=0]     #在小区基站1卸载的用户索引

        offload_in_sbs1_channel_0=[]
        offload_in_sbs1_channel_1=[]
        offload_in_sbs1_channel_2=[]
        offload_in_sbs1_channel_3=[]

        "得到每个卸载到sbs1的用户的信道选择"
        for i in range(0,len(sbs_11),1):
            if b[sbs_11[i]]==0:
                offload_in_sbs1_channel_0.append(sbs_11[i])
            if b[sbs_11[i]]==1:
                offload_in_sbs1_channel_1.append(sbs_11[i])
            if b[sbs_11[i]]==2:
                offload_in_sbs1_channel_2.append(sbs_11[i])
            if b[sbs_11[i]]==3:
                offload_in_sbs1_channel_3.append(sbs_11[i])

        "得到0信道用户对应的信道增益"
        for i in range(0,len(offload_in_sbs1_channel_0),1):
            a_0=[offload_in_sbs1_channel_0[i],result[offload_in_sbs1_channel_0[i]][6]]
            G_0.append(a_0)

        "得到1信道用户对应的信道增益"
        for i in range(0,len(offload_in_sbs1_channel_1),1):
            a_1=[offload_in_sbs1_channel_1[i],result[offload_in_sbs1_channel_1[i]][7]]
            G_1.append(a_1)

        "得到2信道用户对应的信道增益"
        for i in range(0,len(offload_in_sbs1_channel_2),1):
            a_2=[offload_in_sbs1_channel_2[i],result[offload_in_sbs1_channel_2[i]][8]]
            G_2.append(a_2)

        "得到3信道用户对应的信道增益"
        for i in range(0,len(offload_in_sbs1_channel_3),1):
            a_3=[offload_in_sbs1_channel_3[i],result[offload_in_sbs1_channel_3[i]][9]]
            G_3.append(a_3)

        G_0=np.array(G_0)
        "对数组进行降序排序"
        if len(G_0)<=1:
            G_0 = np.array(G_0)
        else:
            G_0=G_0[np.argsort(-G_0[:,1])]

        G_2=np.array(G_2)
        "对数组进行降序排序"
        if len(G_2)<=1:
            G_2 = np.array(G_2)
        else:
            G_2=G_2[np.argsort(-G_2[:,1])]

        G_1=np.array(G_1)
        "对数组进行降序排序"
        if len(G_1)<=1:
            G_1 = np.array(G_1)
        else:
            G_1=G_1[np.argsort(-G_1[:,1])]

        G_3=np.array(G_3)
        "对数组进行降序排序"
        if len(G_3)<=1:
            G_3 = np.array(G_3)
        else:
            G_3=G_3[np.argsort(-G_3[:,1])]

        while len(G_0)>1:
            for j in range(-1,-(len(G_0[:,0])),-1):    #不包含第一个元素
                noise_in_sbs_1[int(G_0[j][0])]+=p[int(G_0[j][0])][4]*result[int(G_0[j][0])][6]
            G_0=np.delete(G_0,0,axis=0)  #删除第一个用户

        while len(G_1)>1:
            for j in range(-1,-(len(G_1[:,0])),-1):    #不包含第一个元素
                noise_in_sbs_1[int(G_1[j][0])]+=p[int(G_1[j][0])][5]*result[int(G_1[j][0])][7]
            G_1=np.delete(G_1,0,axis=0)  #删除第一个用户

        while len(G_2)>1:
            for j in range(-1,-(len(G_2[:,0])),-1):    #不包含第一个元素
                noise_in_sbs_1[int(G_2[j][0])]+=p[int(G_2[j][0])][6]*result[int(G_2[j][0])][8]
            G_2=np.delete(G_2,0,axis=0)  #删除第一个用户

        while len(G_3)>1:
            for j in range(-1,-(len(G_3[:,0])),-1):    #不包含第一个元素
                noise_in_sbs_1[int(G_3[j][0])]+=p[int(G_3[j][0])][7]*result[int(G_3[j][0])][9]
            G_3=np.delete(G_3,0,axis=0)  #删除第一个用户

        "SBS_2"
        noise_in_sbs_2 = [0 for i in range(8)]
        G_00 = []
        G_11 = []
        G_22 = []
        G_33 = []
        sbs_22=[x for x,y in list(enumerate(a)) if y==2 and xx[x]!=0]     #在小区基站1卸载的用户索引

        offload_in_sbs2_channel_0=[]
        offload_in_sbs2_channel_1=[]
        offload_in_sbs2_channel_2=[]
        offload_in_sbs2_channel_3=[]

        "得到每个卸载到sbs2的用户的信道选择"
        for i in range(0,len(sbs_22),1):
            if b[sbs_22[i]]==0:
                offload_in_sbs2_channel_0.append(sbs_22[i])
            if b[sbs_22[i]]==1:
                offload_in_sbs2_channel_1.append(sbs_22[i])
            if b[sbs_22[i]]==2:
                offload_in_sbs2_channel_2.append(sbs_22[i])
            if b[sbs_22[i]]==3:
                offload_in_sbs2_channel_3.append(sbs_22[i])

        "得到0信道用户对应的信道增益"
        for i in range(0,len(offload_in_sbs2_channel_0),1):
            a_00=[offload_in_sbs2_channel_0[i],result[offload_in_sbs2_channel_0[i]][10]]
            G_00.append(a_00)

        "得到1信道用户对应的信道增益"
        for i in range(0,len(offload_in_sbs2_channel_1),1):
            a_11=[offload_in_sbs2_channel_1[i],result[offload_in_sbs2_channel_1[i]][11]]
            G_11.append(a_11)

        "得到2信道用户对应的信道增益"
        for i in range(0,len(offload_in_sbs2_channel_2),1):
            a_22=[offload_in_sbs2_channel_2[i],result[offload_in_sbs2_channel_2[i]][12]]
            G_22.append(a_22)

        "得到3信道用户对应的信道增益"
        for i in range(0,len(offload_in_sbs2_channel_3),1):
            a_33=[offload_in_sbs2_channel_3[i],result[offload_in_sbs2_channel_3[i]][13]]
            G_33.append(a_33)

        G_00=np.array(G_00)
        "对数组进行降序排序"
        if len(G_00)<=1:
            G_00 = np.array(G_00)
        else:
            G_00=G_00[np.argsort(-G_00[:,1])]

        G_22=np.array(G_22)
        "对数组进行降序排序"
        if len(G_22)<=1:
            G_22 = np.array(G_22)
        else:
            G_22=G_22[np.argsort(-G_22[:,1])]

        G_11=np.array(G_11)
        "对数组进行降序排序"
        if len(G_11)<=1:
            G_11 = np.array(G_11)
        else:
            G_11=G_11[np.argsort(-G_11[:,1])]

        G_33=np.array(G_33)
        "对数组进行降序排序"
        if len(G_33)<=1:
            G_33 = np.array(G_33)
        else:
            G_33=G_33[np.argsort(-G_33[:,1])]

        while len(G_00) > 1:
            for j in range(-1, -(len(G_00[:, 0])), -1):  # 不包含第一个元素
                noise_in_sbs_2[int(G_00[j][0])] += p[int(G_00[j][0])][8] * result[int(G_00[j][0])][10]
            G_00 = np.delete(G_00, 0, axis=0)  # 删除第一个用户

        while len(G_11) > 1:
            for j in range(-1, -(len(G_11[:, 0])), -1):  # 不包含第一个元素
                noise_in_sbs_2[int(G_11[j][0])] += p[int(G_11[j][0])][9] * result[int(G_11[j][0])][11]
            G_11 = np.delete(G_11, 0, axis=0)  # 删除第一个用户

        while len(G_22) > 1:
            for j in range(-1, -(len(G_22[:, 0])), -1):  # 不包含第一个元素
                noise_in_sbs_2[int(G_22[j][0])] += p[int(G_22[j][0])][10] * result[int(G_22[j][0])][12]
            G_22 = np.delete(G_22, 0, axis=0)  # 删除第一个用户

        while len(G_33) > 1:
            for j in range(-1, -(len(G_33[:, 0])), -1):  # 不包含第一个元素
                noise_in_sbs_2[int(G_33[j][0])] += p[int(G_33[j][0])][11] * result[int(G_33[j][0])][13]
            G_33 = np.delete(G_33, 0, axis=0)  # 删除第一个用户


        "卸载到小小区时，复用宏基站信道的跨层干扰"
        "sbs1"
        noise_in_sbs_1_channel_0 = [0 for i in range(8)]
        noise_in_sbs_1_channel_1 = [0 for i in range(8)]
        noise_in_sbs_1_channel_2 = [0 for i in range(8)]
        noise_in_sbs_1_channel_3 = [0 for i in range(8)]

        for i in range(0,len(offload_in_sbs1_channel_0),1):
            noise_in_sbs_1_channel_0[offload_in_sbs1_channel_0[i]]=p[offload_in_sbs1_channel_0[i]][0]\
                                                                   *result[offload_in_sbs1_channel_0[i]][6]
        for i in range(0,len(offload_in_sbs1_channel_1),1):
            noise_in_sbs_1_channel_1[offload_in_sbs1_channel_1[i]]=p[offload_in_sbs1_channel_1[i]][1]\
                                                                   *result[offload_in_sbs1_channel_1[i]][7]
        for i in range(0,len(offload_in_sbs1_channel_2),1):
            noise_in_sbs_1_channel_2[offload_in_sbs1_channel_2[i]]=p[offload_in_sbs1_channel_2[i]][2]\
                                                                   *result[offload_in_sbs1_channel_2[i]][8]
        for i in range(0,len(offload_in_sbs1_channel_3),1):
            noise_in_sbs_1_channel_3[offload_in_sbs1_channel_3[i]]=p[offload_in_sbs1_channel_3[i]][3]\
                                                                   *result[offload_in_sbs1_channel_3[i]][9]

        "sbs2"
        noise_in_sbs_2_channel_0 = [0 for i in range(8)]
        noise_in_sbs_2_channel_1 = [0 for i in range(8)]
        noise_in_sbs_2_channel_2 = [0 for i in range(8)]
        noise_in_sbs_2_channel_3 = [0 for i in range(8)]

        for i in range(0, len(offload_in_sbs2_channel_0), 1):
            noise_in_sbs_2_channel_0[offload_in_sbs2_channel_0[i]] = p[offload_in_sbs2_channel_0[i]][4] \
                                                                     * result[offload_in_sbs2_channel_0[i]][10]
        for i in range(0, len(offload_in_sbs2_channel_1), 1):
            noise_in_sbs_2_channel_1[offload_in_sbs2_channel_1[i]] = p[offload_in_sbs2_channel_1[i]][5] \
                                                                     * result[offload_in_sbs2_channel_1[i]][11]
        for i in range(0, len(offload_in_sbs2_channel_2), 1):
            noise_in_sbs_2_channel_2[offload_in_sbs2_channel_2[i]] = p[offload_in_sbs2_channel_2[i]][6] \
                                                                     * result[offload_in_sbs2_channel_2[i]][12]
        for i in range(0, len(offload_in_sbs2_channel_3), 1):
            noise_in_sbs_2_channel_3[offload_in_sbs2_channel_3[i]] = p[offload_in_sbs2_channel_3[i]][7] \
                                                                     * result[offload_in_sbs2_channel_3[i]][13]

        "计算不同基站，相同信道的干扰"
        "sbs1"
        noise_out_sbs1_channel_0 =[0 for i in range(8)]
        noise_out_sbs1_channel_1 = [0 for i in range(8)]
        noise_out_sbs1_channel_2 = [0 for i in range(8)]
        noise_out_sbs1_channel_3 = [0 for i in range(8)]

        for i in range(0,len(offload_in_sbs2_channel_0),):
            noise_out_sbs1_channel_0[offload_in_sbs2_channel_0[i]]+=p[offload_in_sbs2_channel_0[i]][4]\
                                                                    *result[offload_in_sbs2_channel_0[i]][6]
        for i in range(0,len(offload_in_sbs2_channel_1),):
            noise_out_sbs1_channel_1[offload_in_sbs2_channel_1[i]]+=p[offload_in_sbs2_channel_1[i]][5]\
                                                                    *result[offload_in_sbs2_channel_1[i]][7]
        for i in range(0,len(offload_in_sbs2_channel_2),):
            noise_out_sbs1_channel_2[offload_in_sbs2_channel_2[i]]+=p[offload_in_sbs2_channel_2[i]][6]\
                                                                    *result[offload_in_sbs2_channel_2[i]][8]
        for i in range(0,len(offload_in_sbs2_channel_3),):
            noise_out_sbs1_channel_3[offload_in_sbs2_channel_3[i]]+=p[offload_in_sbs2_channel_3[i]][7]\
                                                                    *result[offload_in_sbs2_channel_3[i]][9]

        "sbs2"
        noise_out_sbs2_channel_0 = [0 for i in range(8)]
        noise_out_sbs2_channel_1 = [0 for i in range(8)]
        noise_out_sbs2_channel_2 = [0 for i in range(8)]
        noise_out_sbs2_channel_3 = [0 for i in range(8)]
        for i in range(0,len(offload_in_sbs1_channel_0),):
            noise_out_sbs2_channel_0[offload_in_sbs1_channel_0[i]]+=p[offload_in_sbs1_channel_0[i]][0]\
                                                                    *result[offload_in_sbs1_channel_0[i]][10]
        for i in range(0,len(offload_in_sbs1_channel_1),):
            noise_out_sbs2_channel_1[offload_in_sbs1_channel_1[i]]+=p[offload_in_sbs1_channel_1[i]][1]\
                                                                    *result[offload_in_sbs1_channel_1[i]][11]
        for i in range(0,len(offload_in_sbs1_channel_2),):
            noise_out_sbs2_channel_2[offload_in_sbs1_channel_2[i]]+=p[offload_in_sbs1_channel_2[i]][2]\
                                                                    *result[offload_in_sbs1_channel_2[i]][12]
        for i in range(0,len(offload_in_sbs1_channel_3),):
            noise_out_sbs2_channel_3[offload_in_sbs1_channel_3[i]]+=p[offload_in_sbs1_channel_3[i]][3]\
                                                                    *result[offload_in_sbs1_channel_3[i]][13]

        "计算每个用户的上传速率"
        R=[0 for i in range(8)]
        r=[0 for i in range(8)]

        for i in range(8):
            if xx[i]==1 and a[i]==0 and b[i]==0:
                r[i]=p[i][0]*result[i][2]/(noise_in_mbs_channel_0[i]+self.noise)
            if xx[i]==1 and a[i]==0 and b[i]==1:
                r[i]=p[i][1]*result[i][3]/(noise_in_mbs_channel_1[i]+self.noise)
            if xx[i]==1 and a[i]==0 and b[i]==2:
                r[i]=p[i][2]*result[i][4]/(noise_in_mbs_channel_2[i]+self.noise)
            if xx[i]==1 and a[i]==0 and b[i]==3:
                r[i]=p[i][3]*result[i][5]/(noise_in_mbs_channel_3[i]+self.noise)
            if xx[i]==1 and a[i]==1 and b[i]==0:
                r[i]=p[i][4]*result[i][6]/(noise_in_sbs_1[i]+noise_in_sbs_1_channel_0[i]+noise_out_sbs1_channel_0[i]+self.noise)
            if xx[i]==1 and a[i]==1 and b[i]==1:
                r[i]=p[i][5]*result[i][7]/(noise_in_sbs_1[i]+noise_in_sbs_1_channel_1[i]+noise_out_sbs1_channel_1[i]+self.noise)
            if xx[i]==1 and a[i]==1 and b[i]==2:
                r[i]=p[i][6]*result[i][8]/(noise_in_sbs_1[i]+noise_in_sbs_1_channel_2[i]+noise_out_sbs1_channel_2[i]+self.noise)
            if xx[i]==1 and a[i]==1 and b[i]==3:
                r[i]=p[i][7]*result[i][9]/(noise_in_sbs_1[i]+noise_in_sbs_1_channel_3[i]+noise_out_sbs1_channel_3[i]+self.noise)
            if xx[i]==1 and a[i]==2 and b[i]==0:
                r[i]=p[i][8]*result[i][10]/(noise_in_sbs_2[i]+noise_in_sbs_2_channel_0[i]+noise_out_sbs2_channel_0[i]+self.noise)
            if xx[i]==1 and a[i]==2 and b[i]==1:
                r[i]=p[i][9]*result[i][11]/(noise_in_sbs_2[i]+noise_in_sbs_2_channel_1[i]+noise_out_sbs2_channel_1[i]+self.noise)
            if xx[i]==1 and a[i]==2 and b[i]==2:
                r[i]=p[i][10]*result[i][12]/(noise_in_sbs_2[i]+noise_in_sbs_2_channel_2[i]+noise_out_sbs2_channel_2[i]+self.noise)
            if xx[i]==1 and a[i]==2 and b[i]==3:
                r[i]=p[i][11]*result[i][13]/(noise_in_sbs_2[i]+noise_in_sbs_2_channel_3[i]+noise_out_sbs2_channel_3[i]+self.noise)

        for i in range(8):
            R[i]=self.w*np.log2(1+r[i])

        "优化每个用户本地的计算资源"
        f_local=[]
        for i in range(8):
            local=result[i][1]/self.d_max
            f_local.append(local)

        "计算本地处理时间和能耗"
        t_local=[0 for i in range(8)]
        e_local = [0 for i in range(8)]
        for i in range(8):
            if xx[i]==0:
                t_local[i]=result[i][1]/f_local[i]
                e_local[i]=self.k*(f_local[i]**2)*result[i][1]

        "计算上传和处理时间"
        t_tran= [0 for i in range(8)]
        t_mec = [0 for i in range(8)]
        e_mec = [0 for i in range(8)]
        for i in range(8):
            if xx[i]==1 and a[i] == 0 and b[i] == 0:
                t_tran[i]=result[i][0]/R[i]
                e_mec[i]=p[i][0]*t_tran[i]
                t_mec[i]=result[i][1]/self.f_m_mec
            if xx[i] == 1 and a[i] == 0 and b[i] == 1:
                t_tran[i] = result[i][0] / R[i]
                e_mec[i] = p[i][1] * t_tran[i]
                t_mec[i] = result[i][1] / self.f_m_mec
            if xx[i]==1 and a[i]==0 and b[i] == 2:
                t_tran[i]=result[i][0]/R[i]
                e_mec[i]=p[i][2]*t_tran[i]
                t_mec[i] = result[i][1] / self.f_m_mec
            if xx[i]==1 and a[i]==0 and b[i]==3:
                t_tran[i]=result[i][0]/R[i]
                e_mec[i]=p[i][3]*t_tran[i]
                t_mec[i] = result[i][1] / self.f_m_mec
            if xx[i]==1 and a[i]==1 and b[i]==0:
                t_tran[i]=result[i][0]/R[i]
                e_mec[i]=p[i][4]*t_tran[i]
                t_mec[i] = result[i][1] / self.f_s_mec
            if xx[i]==1 and a[i]==1 and b[i]==1:
                t_tran[i]=result[i][0]/R[i]
                e_mec[i]=p[i][5]*t_tran[i]
                t_mec[i] = result[i][1] / self.f_s_mec
            if xx[i]==1 and a[i]==1 and b[i]==2:
                t_tran[i]=result[i][0]/R[i]
                e_mec[i]=p[i][6]*t_tran[i]
                t_mec[i] = result[i][1] / self.f_s_mec
            if xx[i]==1 and a[i]==1 and b[i]==3:
                t_tran[i]=result[i][0]/R[i]
                e_mec[i]=p[i][7]*t_tran[i]
                t_mec[i] = result[i][1] / self.f_s_mec
            if xx[i]==1 and a[i]==2 and b[i]==0:
                t_tran[i]=result[i][0]/R[i]
                e_mec[i]=p[i][8]*t_tran[i]
                t_mec[i] = result[i][1] / self.f_s_mec
            if xx[i]==1 and a[i]==2 and b[i]==1:
                t_tran[i]=result[i][0]/R[i]
                e_mec[i]=p[i][9]*t_tran[i]
                t_mec[i] = result[i][1] / self.f_s_mec
            if xx[i]==1 and a[i]==2 and b[i]==2:
                t_tran[i]=result[i][0]/R[i]
                e_mec[i]=p[i][10]*t_tran[i]
                t_mec[i] = result[i][1] / self.f_s_mec
            if xx[i]==1 and a[i]==2 and b[i]==3:
                t_tran[i]=result[i][0]/R[i]
                e_mec[i]=p[i][11]*t_tran[i]
                t_mec[i] = result[i][1] / self.f_s_mec

        "计算所有用户的奖励"
        reward=[0 for i in range(8)]
        "加入最低传输速率的限制,小于这个速率则看为传输失败"
        R_min=[0 for i in range(8)]
        for i in range(8):
            R_min[i]=result[i][0]/(self.d_max-t_mec[i])

        # for i in range(8):

        #     if len(channel_0)>3:                    #设置信道最大连接个数
        #         for i in range(0,len(channel_0),1):
        #             reward[channel_0[i]]=-1
        #     elif len(channel_1)>3:
        #         for i in range(0,len(channel_1),1):
        #             reward[channel_1[i]]=-1
        #     elif len(channel_2)>3:
        #         for i in range(0,len(channel_2),1):
        #             reward[channel_2[i]]=-1
        #     elif len(channel_3)>3:
        #         for i in range(0,len(channel_3),1):
        #             reward[channel_3[i]]=-1
        #     else:
        #         reward[i]=-(e_local[i]+e_mec[i])

        for i in range(8):
            if xx[i]==1 and R[i]<R_min[i]:
                reward[i]=-0.02
            else:
                reward[i]=-(e_local[i]+e_mec[i])

        # g_best=[]
        # "粒子群寻优"
        # for i in range(8):
        #     gbest=np.random.randint(0,13)
        #     g_best.append(gbest)
        #
        # "所有用户连接基站数目的列表"
        # g_a = [0 for i in range(8)]
        # "所有用户选择信道数目的列表"
        # g_b = [0 for i in range(8)]
        # "生成二进制卸载列表"
        # g_xx = [0 for i in range(8)]
        #
        # "分解所有动作为连接基站和信道的选择"
        # for i in range(8):
        #     if g_best[i] % 3 == 0 and g_best[i] % 4 == 0 and g_best[i] != 0:
        #         g_xx[i] = 0
        #     else:
        #         g_xx[i] = 1
        #     g_a[i] = g_best[i] % 3  # 连接的基站数目
        #     g_b[i] = (g_best[i] // 3) % 4  # 选择的信道数目
        #
        # "计算在MBS卸载时，小小区复用信道产生的干扰"
        #
        # g_mbs_0 = [x for x, y in list(enumerate(g_a)) if y == 0 and g_xx[x] != 0]  # 在宏基站卸载的用户索引
        # g_sbs_1 = [x for x, y in list(enumerate(g_a)) if y == 1 and g_xx[x] != 0]  # 在小区基站1卸载的用户索引
        # g_sbs_2 = [x for x, y in list(enumerate(g_a)) if y == 2 and g_xx[x] != 0]  # 在小区基站2卸载的用户索引
        # g_channel_0 = [x for x, y in list(enumerate(g_b)) if y == 0 and g_xx[x] != 0 and g_a[x] != 0]  # 小小区基站选择信道0的用户索引
        # g_channel_1 = [x for x, y in list(enumerate(g_b)) if y == 1 and g_xx[x] != 0 and g_a[x] != 0]  # 小小区基站选择信道1的用户索引
        # g_channel_2 = [x for x, y in list(enumerate(g_b)) if y == 2 and g_xx[x] != 0 and g_a[x] != 0]  # 小小区基站选择信道2的用户索引
        # g_channel_3 = [x for x, y in list(enumerate(g_b)) if y == 3 and g_xx[x] != 0 and g_a[x] != 0]  # 小小区基站选择信道3的用户索引
        #
        # g_noise_in_mbs_channel_0 = [0 for i in range(8)]
        # g_noise_in_mbs_channel_1 = [0 for i in range(8)]
        # g_noise_in_mbs_channel_2 = [0 for i in range(8)]
        # g_noise_in_mbs_channel_3 = [0 for i in range(8)]
        #
        # "计算在mbs卸载使用信道的用户，其它小小区使用相同信道的干扰"
        # for i in range(0, len(g_mbs_0)):
        #     for j in range(0, len(g_channel_0), 1):
        #         if g_channel_0[j] in g_sbs_1:
        #             g_noise_in_mbs_channel_0[g_channel_0[j]] += p[g_channel_0[j]][g_b[g_channel_0[j]]] * result[g_channel_0[j]][
        #                 2 + g_b[g_channel_0[j]]]
        #         if g_channel_0[j] in g_sbs_2:
        #             g_noise_in_mbs_channel_0[g_channel_0[j]] += p[g_channel_0[j]][g_b[g_channel_0[j]]] * result[g_channel_0[j]][
        #                 2 + g_b[g_channel_0[j]]]
        #     for j in range(0, len(g_channel_1), 1):
        #         if g_channel_1[j] in g_sbs_1:
        #             g_noise_in_mbs_channel_1[g_channel_1[j]] += p[g_channel_1[j]][g_b[g_channel_1[j]]] * result[g_channel_1[j]][
        #                 2 + g_b[g_channel_1[j]]]
        #         if g_channel_1[j] in g_sbs_2:
        #             g_noise_in_mbs_channel_1[g_channel_1[j]] += p[g_channel_1[j]][g_b[g_channel_1[j]]] * result[g_channel_1[j]][
        #                 2 + g_b[g_channel_1[j]]]
        #     for j in range(0, len(g_channel_2), 1):
        #         if g_channel_2[j] in g_sbs_1:
        #             g_noise_in_mbs_channel_2[g_channel_2[j]] += p[g_channel_2[j]][g_b[g_channel_2[j]]] * result[g_channel_2[j]][
        #                 2 + g_b[g_channel_2[j]]]
        #         if g_channel_2[j] in g_sbs_2:
        #             g_noise_in_mbs_channel_2[g_channel_2[j]] += p[g_channel_2[j]][g_b[g_channel_2[j]]] * result[g_channel_2[j]][
        #                 2 + g_b[g_channel_2[j]]]
        #     for j in range(0, len(g_channel_3), 1):
        #         if g_channel_3[j] in g_sbs_1:
        #             g_noise_in_mbs_channel_3[g_channel_3[j]] += p[g_channel_3[j]][g_b[g_channel_3[j]]] * result[g_channel_3[j]][
        #                 2 + g_b[g_channel_3[j]]]
        #         if g_channel_3[j] in g_sbs_2:
        #             g_noise_in_mbs_channel_3[g_channel_3[j]] += p[g_channel_3[j]][g_b[g_channel_3[j]]] * result[g_channel_3[j]][
        #                 2 + g_b[g_channel_3[j]]]
        #
        # "同基站同信道干扰"
        # "SBS_1"
        # g_noise_in_sbs_1 = [0 for i in range(8)]
        # g_G_0 = []
        # g_G_1 = []
        # g_G_2 = []
        # g_G_3 = []
        # g_sbs_11 = [x for x, y in list(enumerate(g_a)) if y == 1 and g_xx[x] != 0]  # 在小区基站1卸载的用户索引
        #
        # g_offload_in_sbs1_channel_0 = []
        # g_offload_in_sbs1_channel_1 = []
        # g_offload_in_sbs1_channel_2 = []
        # g_offload_in_sbs1_channel_3 = []
        #
        # "得到每个卸载到sbs1的用户的信道选择"
        # for i in range(0, len(g_sbs_11), 1):
        #     if g_b[g_sbs_11[i]] == 0:
        #         g_offload_in_sbs1_channel_0.append(g_sbs_11[i])
        #     if g_b[g_sbs_11[i]] == 1:
        #         g_offload_in_sbs1_channel_1.append(g_sbs_11[i])
        #     if g_b[g_sbs_11[i]] == 2:
        #         g_offload_in_sbs1_channel_2.append(g_sbs_11[i])
        #     if g_b[g_sbs_11[i]] == 3:
        #         g_offload_in_sbs1_channel_3.append(g_sbs_11[i])
        #
        # "得到0信道用户对应的信道增益"
        # for i in range(0, len(g_offload_in_sbs1_channel_0), 1):
        #     g_a_0 = [g_offload_in_sbs1_channel_0[i], result[g_offload_in_sbs1_channel_0[i]][6]]
        #     g_G_0.append(g_a_0)
        #
        # "得到1信道用户对应的信道增益"
        # for i in range(0, len(g_offload_in_sbs1_channel_1), 1):
        #     g_a_1 = [g_offload_in_sbs1_channel_1[i], result[g_offload_in_sbs1_channel_1[i]][7]]
        #     g_G_1.append(g_a_1)
        #
        # "得到2信道用户对应的信道增益"
        # for i in range(0, len(g_offload_in_sbs1_channel_2), 1):
        #     g_a_2 = [g_offload_in_sbs1_channel_2[i], result[g_offload_in_sbs1_channel_2[i]][8]]
        #     g_G_2.append(g_a_2)
        #
        # "得到3信道用户对应的信道增益"
        # for i in range(0, len(g_offload_in_sbs1_channel_3), 1):
        #     g_a_3 = [g_offload_in_sbs1_channel_3[i], result[g_offload_in_sbs1_channel_3[i]][9]]
        #     g_G_3.append(g_a_3)
        #
        # g_G_0 = np.array(g_G_0)
        # "对数组进行降序排序"
        # if len(g_G_0) <= 1:
        #     g_G_0 = np.array(g_G_0)
        # else:
        #     g_G_0 = g_G_0[np.argsort(-g_G_0[:, 1])]
        #
        # g_G_2 = np.array(g_G_2)
        # "对数组进行降序排序"
        # if len(g_G_2) <= 1:
        #     g_G_2 = np.array(g_G_2)
        # else:
        #     g_G_2 = g_G_2[np.argsort(-g_G_2[:, 1])]
        #
        # g_G_1 = np.array(g_G_1)
        # "对数组进行降序排序"
        # if len(g_G_1) <= 1:
        #     g_G_1 = np.array(g_G_1)
        # else:
        #     g_G_1 = g_G_1[np.argsort(-g_G_1[:, 1])]
        #
        # g_G_3 = np.array(g_G_3)
        # "对数组进行降序排序"
        # if len(g_G_3) <= 1:
        #     g_G_3 = np.array(g_G_3)
        # else:
        #     g_G_3 = g_G_3[np.argsort(-g_G_3[:, 1])]
        #
        # while len(g_G_0) > 1:
        #     for j in range(-1, -(len(g_G_0[:, 0])), -1):  # 不包含第一个元素
        #         g_noise_in_sbs_1[int(g_G_0[j][0])] += p[int(g_G_0[j][0])][4] * result[int(g_G_0[j][0])][6]
        #     g_G_0 = np.delete(g_G_0, 0, axis=0)  # 删除第一个用户
        #
        # while len(g_G_1) > 1:
        #     for j in range(-1, -(len(g_G_1[:, 0])), -1):  # 不包含第一个元素
        #         g_noise_in_sbs_1[int(g_G_1[j][0])] += p[int(g_G_1[j][0])][5] * result[int(g_G_1[j][0])][7]
        #     g_G_1 = np.delete(g_G_1, 0, axis=0)  # 删除第一个用户
        #
        # while len(g_G_2) > 1:
        #     for j in range(-1, -(len(g_G_2[:, 0])), -1):  # 不包含第一个元素
        #         g_noise_in_sbs_1[int(g_G_2[j][0])] += p[int(g_G_2[j][0])][6] * result[int(g_G_2[j][0])][8]
        #     g_G_2 = np.delete(g_G_2, 0, axis=0)  # 删除第一个用户
        #
        # while len(g_G_3) > 1:
        #     for j in range(-1, -(len(g_G_3[:, 0])), -1):  # 不包含第一个元素
        #         g_noise_in_sbs_1[int(g_G_3[j][0])] += p[int(g_G_3[j][0])][7] * result[int(g_G_3[j][0])][9]
        #     g_G_3 = np.delete(g_G_3, 0, axis=0)  # 删除第一个用户
        #
        # "SBS_2"
        # g_noise_in_sbs_2 = [0 for i in range(8)]
        # g_G_00 = []
        # g_G_11 = []
        # g_G_22 = []
        # g_G_33 = []
        # g_sbs_22 = [x for x, y in list(enumerate(g_a)) if y == 2 and g_xx[x] != 0]  # 在小区基站1卸载的用户索引
        #
        # g_offload_in_sbs2_channel_0 = []
        # g_offload_in_sbs2_channel_1 = []
        # g_offload_in_sbs2_channel_2 = []
        # g_offload_in_sbs2_channel_3 = []
        #
        # "得到每个卸载到sbs2的用户的信道选择"
        # for i in range(0, len(g_sbs_22), 1):
        #     if g_b[g_sbs_22[i]] == 0:
        #         g_offload_in_sbs2_channel_0.append(g_sbs_22[i])
        #     if g_b[g_sbs_22[i]] == 1:
        #         g_offload_in_sbs2_channel_1.append(g_sbs_22[i])
        #     if g_b[g_sbs_22[i]] == 2:
        #         g_offload_in_sbs2_channel_2.append(g_sbs_22[i])
        #     if g_b[g_sbs_22[i]] == 3:
        #         g_offload_in_sbs2_channel_3.append(g_sbs_22[i])
        #
        # "得到0信道用户对应的信道增益"
        # for i in range(0, len(g_offload_in_sbs2_channel_0), 1):
        #     g_a_00 = [g_offload_in_sbs2_channel_0[i], result[g_offload_in_sbs2_channel_0[i]][10]]
        #     g_G_00.append(g_a_00)
        #
        # "得到1信道用户对应的信道增益"
        # for i in range(0, len(g_offload_in_sbs2_channel_1), 1):
        #     g_a_11 = [g_offload_in_sbs2_channel_1[i], result[g_offload_in_sbs2_channel_1[i]][11]]
        #     g_G_11.append(g_a_11)
        #
        # "得到2信道用户对应的信道增益"
        # for i in range(0, len(g_offload_in_sbs2_channel_2), 1):
        #     g_a_22 = [g_offload_in_sbs2_channel_2[i], result[g_offload_in_sbs2_channel_2[i]][12]]
        #     g_G_22.append(g_a_22)
        #
        # "得到3信道用户对应的信道增益"
        # for i in range(0, len(g_offload_in_sbs2_channel_3), 1):
        #     g_a_33 = [g_offload_in_sbs2_channel_3[i], result[g_offload_in_sbs2_channel_3[i]][13]]
        #     g_G_33.append(g_a_33)
        #
        # g_G_00 = np.array(g_G_00)
        # "对数组进行降序排序"
        # if len(g_G_00) <= 1:
        #     g_G_00 = np.array(g_G_00)
        # else:
        #     g_G_00 = g_G_00[np.argsort(-g_G_00[:, 1])]
        #
        # g_G_22 = np.array(g_G_22)
        # "对数组进行降序排序"
        # if len(g_G_22) <= 1:
        #     g_G_22 = np.array(g_G_22)
        # else:
        #     g_G_22 = g_G_22[np.argsort(-g_G_22[:, 1])]
        #
        # g_G_11 = np.array(g_G_11)
        # "对数组进行降序排序"
        # if len(g_G_11) <= 1:
        #     g_G_11 = np.array(g_G_11)
        # else:
        #     g_G_11 = g_G_11[np.argsort(-g_G_11[:, 1])]
        #
        # g_G_33 = np.array(g_G_33)
        # "对数组进行降序排序"
        # if len(g_G_33) <= 1:
        #     g_G_33 = np.array(g_G_33)
        # else:
        #     g_G_33 = g_G_33[np.argsort(-g_G_33[:, 1])]
        #
        # while len(g_G_00) > 1:
        #     for j in range(-1, -(len(g_G_00[:, 0])), -1):  # 不包含第一个元素
        #         g_noise_in_sbs_2[int(g_G_00[j][0])] += p[int(g_G_00[j][0])][8] * result[int(g_G_00[j][0])][10]
        #     g_G_00 = np.delete(g_G_00, 0, axis=0)  # 删除第一个用户
        #
        # while len(g_G_11) > 1:
        #     for j in range(-1, -(len(g_G_11[:, 0])), -1):  # 不包含第一个元素
        #         g_noise_in_sbs_2[int(g_G_11[j][0])] += p[int(g_G_11[j][0])][9] * result[int(g_G_11[j][0])][11]
        #     g_G_11 = np.delete(g_G_11, 0, axis=0)  # 删除第一个用户
        #
        # while len(g_G_22) > 1:
        #     for j in range(-1, -(len(g_G_22[:, 0])), -1):  # 不包含第一个元素
        #         g_noise_in_sbs_2[int(g_G_22[j][0])] += p[int(g_G_22[j][0])][10] * result[int(g_G_22[j][0])][12]
        #     g_G_22 = np.delete(g_G_22, 0, axis=0)  # 删除第一个用户
        #
        # while len(g_G_33) > 1:
        #     for j in range(-1, -(len(g_G_33[:, 0])), -1):  # 不包含第一个元素
        #         g_noise_in_sbs_2[int(g_G_33[j][0])] += p[int(g_G_33[j][0])][11] * result[int(g_G_33[j][0])][13]
        #     g_G_33 = np.delete(g_G_33, 0, axis=0)  # 删除第一个用户
        #
        # "卸载到小小区时，复用宏基站信道的跨层干扰"
        # "sbs1"
        # g_noise_in_sbs_1_channel_0 = [0 for i in range(8)]
        # g_noise_in_sbs_1_channel_1 = [0 for i in range(8)]
        # g_noise_in_sbs_1_channel_2 = [0 for i in range(8)]
        # g_noise_in_sbs_1_channel_3 = [0 for i in range(8)]
        #
        # for i in range(0, len(g_offload_in_sbs1_channel_0), 1):
        #     g_noise_in_sbs_1_channel_0[g_offload_in_sbs1_channel_0[i]] = p[g_offload_in_sbs1_channel_0[i]][0] \
        #                                                              * result[g_offload_in_sbs1_channel_0[i]][6]
        # for i in range(0, len(g_offload_in_sbs1_channel_1), 1):
        #     g_noise_in_sbs_1_channel_1[g_offload_in_sbs1_channel_1[i]] = p[g_offload_in_sbs1_channel_1[i]][1] \
        #                                                              * result[g_offload_in_sbs1_channel_1[i]][7]
        # for i in range(0, len(g_offload_in_sbs1_channel_2), 1):
        #     g_noise_in_sbs_1_channel_2[g_offload_in_sbs1_channel_2[i]] = p[g_offload_in_sbs1_channel_2[i]][2] \
        #                                                              * result[g_offload_in_sbs1_channel_2[i]][8]
        # for i in range(0, len(g_offload_in_sbs1_channel_3), 1):
        #     g_noise_in_sbs_1_channel_3[g_offload_in_sbs1_channel_3[i]] = p[g_offload_in_sbs1_channel_3[i]][3] \
        #                                                              * result[g_offload_in_sbs1_channel_3[i]][9]
        #
        # "sbs2"
        # g_noise_in_sbs_2_channel_0 = [0 for i in range(8)]
        # g_noise_in_sbs_2_channel_1 = [0 for i in range(8)]
        # g_noise_in_sbs_2_channel_2 = [0 for i in range(8)]
        # g_noise_in_sbs_2_channel_3 = [0 for i in range(8)]
        #
        # for i in range(0, len(g_offload_in_sbs2_channel_0), 1):
        #     g_noise_in_sbs_2_channel_0[g_offload_in_sbs2_channel_0[i]] = p[g_offload_in_sbs2_channel_0[i]][4] \
        #                                                              * result[g_offload_in_sbs2_channel_0[i]][10]
        # for i in range(0, len(g_offload_in_sbs2_channel_1), 1):
        #     g_noise_in_sbs_2_channel_1[g_offload_in_sbs2_channel_1[i]] = p[g_offload_in_sbs2_channel_1[i]][5] \
        #                                                              * result[g_offload_in_sbs2_channel_1[i]][11]
        # for i in range(0, len(g_offload_in_sbs2_channel_2), 1):
        #     g_noise_in_sbs_2_channel_2[g_offload_in_sbs2_channel_2[i]] = p[g_offload_in_sbs2_channel_2[i]][6] \
        #                                                              * result[g_offload_in_sbs2_channel_2[i]][12]
        # for i in range(0, len(g_offload_in_sbs2_channel_3), 1):
        #     g_noise_in_sbs_2_channel_3[g_offload_in_sbs2_channel_3[i]] = p[g_offload_in_sbs2_channel_3[i]][7] \
        #                                                              * result[g_offload_in_sbs2_channel_3[i]][13]
        #
        # "计算不同基站，相同信道的干扰"
        # "sbs1"
        # g_noise_out_sbs1_channel_0 = [0 for i in range(8)]
        # g_noise_out_sbs1_channel_1 = [0 for i in range(8)]
        # g_noise_out_sbs1_channel_2 = [0 for i in range(8)]
        # g_noise_out_sbs1_channel_3 = [0 for i in range(8)]
        #
        # for i in range(0, len(g_offload_in_sbs2_channel_0), ):
        #     g_noise_out_sbs1_channel_0[g_offload_in_sbs2_channel_0[i]] += p[g_offload_in_sbs2_channel_0[i]][4] \
        #                                                               * result[g_offload_in_sbs2_channel_0[i]][6]
        # for i in range(0, len(g_offload_in_sbs2_channel_1), ):
        #     g_noise_out_sbs1_channel_1[g_offload_in_sbs2_channel_1[i]] += p[g_offload_in_sbs2_channel_1[i]][5] \
        #                                                               * result[g_offload_in_sbs2_channel_1[i]][7]
        # for i in range(0, len(g_offload_in_sbs2_channel_2), ):
        #     g_noise_out_sbs1_channel_2[g_offload_in_sbs2_channel_2[i]] += p[g_offload_in_sbs2_channel_2[i]][6] \
        #                                                               * result[g_offload_in_sbs2_channel_2[i]][8]
        # for i in range(0, len(g_offload_in_sbs2_channel_3), ):
        #     g_noise_out_sbs1_channel_3[g_offload_in_sbs2_channel_3[i]] += p[g_offload_in_sbs2_channel_3[i]][7] \
        #                                                               * result[g_offload_in_sbs2_channel_3[i]][9]
        #
        # "sbs2"
        # g_noise_out_sbs2_channel_0 = [0 for i in range(8)]
        # g_noise_out_sbs2_channel_1 = [0 for i in range(8)]
        # g_noise_out_sbs2_channel_2 = [0 for i in range(8)]
        # g_noise_out_sbs2_channel_3 = [0 for i in range(8)]
        # for i in range(0, len(g_offload_in_sbs1_channel_0), ):
        #     g_noise_out_sbs2_channel_0[g_offload_in_sbs1_channel_0[i]] += p[g_offload_in_sbs1_channel_0[i]][0] \
        #                                                               * result[g_offload_in_sbs1_channel_0[i]][10]
        # for i in range(0, len(g_offload_in_sbs1_channel_1), ):
        #     g_noise_out_sbs2_channel_1[g_offload_in_sbs1_channel_1[i]] += p[g_offload_in_sbs1_channel_1[i]][1] \
        #                                                               * result[g_offload_in_sbs1_channel_1[i]][11]
        # for i in range(0, len(g_offload_in_sbs1_channel_2), ):
        #     g_noise_out_sbs2_channel_2[g_offload_in_sbs1_channel_2[i]] += p[g_offload_in_sbs1_channel_2[i]][2] \
        #                                                               * result[g_offload_in_sbs1_channel_2[i]][12]
        # for i in range(0, len(g_offload_in_sbs1_channel_3), ):
        #     g_noise_out_sbs2_channel_3[g_offload_in_sbs1_channel_3[i]] += p[g_offload_in_sbs1_channel_3[i]][3] \
        #                                                               * result[g_offload_in_sbs1_channel_3[i]][13]
        #
        # "计算每个用户的上传速率"
        # g_R = [0 for i in range(8)]
        # g_r = [0 for i in range(8)]
        #
        # for i in range(8):
        #     if g_xx[i] == 1 and g_a[i] == 0 and g_b[i] == 0:
        #         g_r[i] = p[i][0] * result[i][2] / (g_noise_in_mbs_channel_0[i] + self.noise)
        #     if g_xx[i] == 1 and g_a[i] == 0 and g_b[i] == 1:
        #         g_r[i] = p[i][1] * result[i][3] / (g_noise_in_mbs_channel_1[i] + self.noise)
        #     if g_xx[i] == 1 and g_a[i] == 0 and g_b[i] == 2:
        #         g_r[i] = p[i][2] * result[i][4] / (g_noise_in_mbs_channel_2[i] + self.noise)
        #     if g_xx[i] == 1 and g_a[i] == 0 and g_b[i] == 3:
        #         g_r[i] = p[i][3] * result[i][5] / (g_noise_in_mbs_channel_3[i] + self.noise)
        #     if g_xx[i] == 1 and g_a[i] == 1 and g_b[i] == 0:
        #         g_r[i] = p[i][4] * result[i][6] / (
        #                     g_noise_in_sbs_1[i] + g_noise_in_sbs_1_channel_0[i] + g_noise_out_sbs1_channel_0[i] + self.noise)
        #     if g_xx[i] == 1 and g_a[i] == 1 and g_b[i] == 1:
        #         g_r[i] = p[i][5] * result[i][7] / (
        #                     g_noise_in_sbs_1[i] + g_noise_in_sbs_1_channel_1[i] + g_noise_out_sbs1_channel_1[i] + self.noise)
        #     if g_xx[i] == 1 and g_a[i] == 1 and g_b[i] == 2:
        #         g_r[i] = p[i][6] * result[i][8] / (
        #                     g_noise_in_sbs_1[i] + g_noise_in_sbs_1_channel_2[i] + g_noise_out_sbs1_channel_2[i] + self.noise)
        #     if g_xx[i] == 1 and g_a[i] == 1 and g_b[i] == 3:
        #         g_r[i] = p[i][7] * result[i][9] / (
        #                     g_noise_in_sbs_1[i] + g_noise_in_sbs_1_channel_3[i] + g_noise_out_sbs1_channel_3[i] + self.noise)
        #     if g_xx[i] == 1 and g_a[i] == 2 and g_b[i] == 0:
        #         g_r[i] = p[i][8] * result[i][10] / (
        #                     g_noise_in_sbs_2[i] + g_noise_in_sbs_2_channel_0[i] + g_noise_out_sbs2_channel_0[i] + self.noise)
        #     if g_xx[i] == 1 and g_a[i] == 2 and g_b[i] == 1:
        #         g_r[i] = p[i][9] * result[i][11] / (
        #                     g_noise_in_sbs_2[i] + g_noise_in_sbs_2_channel_1[i] + g_noise_out_sbs2_channel_1[i] + self.noise)
        #     if g_xx[i] == 1 and g_a[i] == 2 and g_b[i] == 2:
        #         g_r[i] = p[i][10] * result[i][12] / (
        #                     g_noise_in_sbs_2[i] + g_noise_in_sbs_2_channel_2[i] + g_noise_out_sbs2_channel_2[i] + self.noise)
        #     if g_xx[i] == 1 and g_a[i] == 2 and g_b[i] == 3:
        #         g_r[i] = p[i][11] * result[i][13] / (
        #                     g_noise_in_sbs_2[i] + g_noise_in_sbs_2_channel_3[i] + g_noise_out_sbs2_channel_3[i] + self.noise)
        #
        # for i in range(8):
        #     g_R[i] = self.w * np.log2(1 + g_r[i])
        #
        # "优化每个用户本地的计算资源"
        # g_f_local = []
        # for i in range(8):
        #     g_local = result[i][1] / self.d_max
        #     g_f_local.append(g_local)
        #
        # "计算本地处理时间和能耗"
        # g_e_local = [0 for i in range(8)]
        # for i in range(8):
        #     if g_xx[i] == 0:
        #         g_e_local[i] = self.k * (g_f_local[i] ** 2) * result[i][1]
        #
        # "计算上传和处理时间"
        # g_t_tran = [0 for i in range(8)]
        # g_t_mec = [0 for i in range(8)]
        # g_e_mec = [0 for i in range(8)]
        # for i in range(8):
        #     if g_xx[i] == 1 and g_a[i] == 0 and g_b[i] == 0:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][0] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_m_mec
        #     if g_xx[i] == 1 and g_a[i] == 0 and g_b[i] == 1:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][1] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_m_mec
        #     if g_xx[i] == 1 and g_a[i] == 0 and g_b[i] == 2:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][2] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_m_mec
        #     if g_xx[i] == 1 and g_a[i] == 0 and g_b[i] == 3:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][3] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_m_mec
        #     if g_xx[i] == 1 and g_a[i] == 1 and g_b[i] == 0:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][4] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_s_mec
        #     if g_xx[i] == 1 and g_a[i] == 1 and g_b[i] == 1:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][5] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_s_mec
        #     if g_xx[i] == 1 and g_a[i] == 1 and g_b[i] == 2:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][6] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_s_mec
        #     if g_xx[i] == 1 and g_a[i] == 1 and g_b[i] == 3:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][7] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_s_mec
        #     if g_xx[i] == 1 and g_a[i] == 2 and g_b[i] == 0:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][8] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_s_mec
        #     if g_xx[i] == 1 and g_a[i] == 2 and g_b[i] == 1:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][9] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_s_mec
        #     if g_xx[i] == 1 and g_a[i] == 2 and g_b[i] == 2:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][10] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_s_mec
        #     if g_xx[i] == 1 and g_a[i] == 2 and g_b[i] == 3:
        #         g_t_tran[i] = result[i][0] / g_R[i]
        #         g_e_mec[i] = p[i][11] * g_t_tran[i]
        #         g_t_mec[i] = result[i][1] / self.f_s_mec
        #
        # "计算所有用户的奖励"
        # g_reward = [0 for i in range(8)]
        # "加入最低传输速率的限制,小于这个速率则看为传输失败"
        # g_R_min = [0 for i in range(8)]
        # for i in range(8):
        #     g_R_min[i] = result[i][0] / (self.d_max - g_t_mec[i])
        #
        # for i in range(8):
        #     if g_xx[i] == 1 and g_R[i] < g_R_min[i]:
        #         g_reward[i] = -0.02
        #     else:
        #         g_reward[i] = -(g_e_local[i] + g_e_mec[i])
        #
        # fitness_best=g_reward
        # vv=[]
        # for i in range(8):
        #     v=np.random.randint(0,13)
        #     vv.append(v)
        # p_best=[]
        # for i in range(8):
        #     pbest=np.random.randint(0,13)
        #     p_best.append(pbest)
        # x_best=p_best
        #
        # "粒子群迭代"
        #
        # for ttt in range(500):
        #     w = self.ws - ((self.ws - self.we) * (ttt+1 / 500))
        #     for i in range(8):
        #         # w = self.ws - ((self.ws - self.we) * (i+1 / ttt+1))
        #         "更新速度"
        #         vv[i]=w*vv[i]+self.c1*np.random.rand()*(p_best[i]-x_best[i])+self.c2*np.random.rand()*(g_best[i]-x_best[i])
        #         "更新位置"
        #         if vv[i]<0:
        #             vv[i]=0
        #         if vv[i]>12:
        #             vv[i]=12
        #         x_best[i]=int(x_best[i]+vv[i])
        #         if x_best[i]>12:
        #             x_best[i]=x_best[i]%10
        #
        #     "计算更新后的能耗"
        #     "所有用户连接基站数目的列表"
        #     x_a = [0 for i in range(8)]
        #     "所有用户选择信道数目的列表"
        #     x_b = [0 for i in range(8)]
        #     "生成二进制卸载列表"
        #     x_xx = [0 for i in range(8)]
        #
        #     "分解所有动作为连接基站和信道的选择"
        #     for i in range(8):
        #         if x_best[i] % 3 == 0 and x_best[i] % 4 == 0 and x_best[i] != 0:
        #             x_xx[i] = 0
        #         else:
        #             x_xx[i] = 1
        #         x_a[i] = x_best[i] % 3  # 连接的基站数目
        #         x_b[i] = (x_best[i] // 3) % 4  # 选择的信道数目
        #
        #     "计算在MBS卸载时，小小区复用信道产生的干扰"
        #
        #     x_mbs_0 = [x for x, y in list(enumerate(x_a)) if y == 0 and x_xx[x] != 0]  # 在宏基站卸载的用户索引
        #     x_sbs_1 = [x for x, y in list(enumerate(x_a)) if y == 1 and x_xx[x] != 0]  # 在小区基站1卸载的用户索引
        #     x_sbs_2 = [x for x, y in list(enumerate(x_a)) if y == 2 and x_xx[x] != 0]  # 在小区基站2卸载的用户索引
        #     x_channel_0 = [x for x, y in list(enumerate(x_b)) if y == 0 and x_xx[x] != 0 and x_a[x] != 0]  # 小小区基站选择信道0的用户索引
        #     x_channel_1 = [x for x, y in list(enumerate(x_b)) if y == 1 and x_xx[x] != 0 and x_a[x] != 0]  # 小小区基站选择信道1的用户索引
        #     x_channel_2 = [x for x, y in list(enumerate(x_b)) if y == 2 and x_xx[x] != 0 and x_a[x] != 0]  # 小小区基站选择信道2的用户索引
        #     x_channel_3 = [x for x, y in list(enumerate(x_b)) if y == 3 and x_xx[x] != 0 and x_a[x] != 0]  # 小小区基站选择信道3的用户索引
        #
        #     x_noise_in_mbs_channel_0 = [0 for i in range(8)]
        #     x_noise_in_mbs_channel_1 = [0 for i in range(8)]
        #     x_noise_in_mbs_channel_2 = [0 for i in range(8)]
        #     x_noise_in_mbs_channel_3 = [0 for i in range(8)]
        #
        #     "计算在mbs卸载使用信道的用户，其它小小区使用相同信道的干扰"
        #     for i in range(0, len(x_mbs_0)):
        #         for j in range(0, len(x_channel_0), 1):
        #             if x_channel_0[j] in x_sbs_1:
        #                 x_noise_in_mbs_channel_0[x_channel_0[j]] += p[x_channel_0[j]][x_b[x_channel_0[j]]] * result[x_channel_0[j]][
        #                     2 + x_b[x_channel_0[j]]]
        #             if x_channel_0[j] in x_sbs_2:
        #                 x_noise_in_mbs_channel_0[x_channel_0[j]] += p[x_channel_0[j]][x_b[x_channel_0[j]]] * result[x_channel_0[j]][
        #                     2 + x_b[x_channel_0[j]]]
        #         for j in range(0, len(x_channel_1), 1):
        #             if x_channel_1[j] in x_sbs_1:
        #                 x_noise_in_mbs_channel_1[x_channel_1[j]] += p[x_channel_1[j]][x_b[x_channel_1[j]]] * result[x_channel_1[j]][
        #                     2 + x_b[x_channel_1[j]]]
        #             if x_channel_1[j] in x_sbs_2:
        #                 x_noise_in_mbs_channel_1[x_channel_1[j]] += p[x_channel_1[j]][x_b[x_channel_1[j]]] * result[x_channel_1[j]][
        #                     2 + x_b[x_channel_1[j]]]
        #         for j in range(0, len(x_channel_2), 1):
        #             if x_channel_2[j] in x_sbs_1:
        #                 x_noise_in_mbs_channel_2[x_channel_2[j]] += p[x_channel_2[j]][x_b[x_channel_2[j]]] * result[x_channel_2[j]][
        #                     2 + x_b[x_channel_2[j]]]
        #             if x_channel_2[j] in x_sbs_2:
        #                 x_noise_in_mbs_channel_2[x_channel_2[j]] += p[x_channel_2[j]][x_b[x_channel_2[j]]] * result[x_channel_2[j]][
        #                     2 + x_b[x_channel_2[j]]]
        #         for j in range(0, len(x_channel_3), 1):
        #             if x_channel_3[j] in x_sbs_1:
        #                 x_noise_in_mbs_channel_3[x_channel_3[j]] += p[x_channel_3[j]][x_b[x_channel_3[j]]] * result[x_channel_3[j]][
        #                     2 + x_b[x_channel_3[j]]]
        #             if x_channel_3[j] in x_sbs_2:
        #                 x_noise_in_mbs_channel_3[x_channel_3[j]] += p[x_channel_3[j]][x_b[x_channel_3[j]]] * result[x_channel_3[j]][
        #                     2 + x_b[x_channel_3[j]]]
        #
        #     "同基站同信道干扰"
        #     "SBS_1"
        #     x_noise_in_sbs_1 = [0 for i in range(8)]
        #     x_G_0 = []
        #     x_G_1 = []
        #     x_G_2 = []
        #     x_G_3 = []
        #     x_sbs_11 = [x for x, y in list(enumerate(x_a)) if y == 1 and x_xx[x] != 0]  # 在小区基站1卸载的用户索引
        #
        #     x_offload_in_sbs1_channel_0 = []
        #     x_offload_in_sbs1_channel_1 = []
        #     x_offload_in_sbs1_channel_2 = []
        #     x_offload_in_sbs1_channel_3 = []
        #
        #     "得到每个卸载到sbs1的用户的信道选择"
        #     for i in range(0, len(x_sbs_11), 1):
        #         if x_b[x_sbs_11[i]] == 0:
        #             x_offload_in_sbs1_channel_0.append(x_sbs_11[i])
        #         if x_b[x_sbs_11[i]] == 1:
        #             x_offload_in_sbs1_channel_1.append(x_sbs_11[i])
        #         if x_b[x_sbs_11[i]] == 2:
        #             x_offload_in_sbs1_channel_2.append(x_sbs_11[i])
        #         if x_b[x_sbs_11[i]] == 3:
        #             x_offload_in_sbs1_channel_3.append(x_sbs_11[i])
        #
        #     "得到0信道用户对应的信道增益"
        #     for i in range(0, len(x_offload_in_sbs1_channel_0), 1):
        #         x_a_0 = [x_offload_in_sbs1_channel_0[i], result[x_offload_in_sbs1_channel_0[i]][6]]
        #         x_G_0.append(x_a_0)
        #
        #     "得到1信道用户对应的信道增益"
        #     for i in range(0, len(x_offload_in_sbs1_channel_1), 1):
        #         x_a_1 = [x_offload_in_sbs1_channel_1[i], result[x_offload_in_sbs1_channel_1[i]][7]]
        #         x_G_1.append(x_a_1)
        #
        #     "得到2信道用户对应的信道增益"
        #     for i in range(0, len(x_offload_in_sbs1_channel_2), 1):
        #         x_a_2 = [x_offload_in_sbs1_channel_2[i], result[x_offload_in_sbs1_channel_2[i]][8]]
        #         x_G_2.append(x_a_2)
        #
        #     "得到3信道用户对应的信道增益"
        #     for i in range(0, len(x_offload_in_sbs1_channel_3), 1):
        #         x_a_3 = [x_offload_in_sbs1_channel_3[i], result[x_offload_in_sbs1_channel_3[i]][9]]
        #         x_G_3.append(x_a_3)
        #
        #     x_G_0 = np.array(x_G_0)
        #     "对数组进行降序排序"
        #     if len(x_G_0) <= 1:
        #         x_G_0 = np.array(x_G_0)
        #     else:
        #         x_G_0 = x_G_0[np.argsort(-x_G_0[:, 1])]
        #
        #     x_G_2 = np.array(x_G_2)
        #     "对数组进行降序排序"
        #     if len(x_G_2) <= 1:
        #         x_G_2 = np.array(x_G_2)
        #     else:
        #         x_G_2 = x_G_2[np.argsort(-x_G_2[:, 1])]
        #
        #     x_G_1 = np.array(x_G_1)
        #     "对数组进行降序排序"
        #     if len(x_G_1) <= 1:
        #         x_G_1 = np.array(x_G_1)
        #     else:
        #         x_G_1 = x_G_1[np.argsort(-x_G_1[:, 1])]
        #
        #     x_G_3 = np.array(x_G_3)
        #     "对数组进行降序排序"
        #     if len(x_G_3) <= 1:
        #         x_G_3 = np.array(x_G_3)
        #     else:
        #         x_G_3 = x_G_3[np.argsort(-x_G_3[:, 1])]
        #
        #     while len(x_G_0) > 1:
        #         for j in range(-1, -(len(x_G_0[:, 0])), -1):  # 不包含第一个元素
        #             x_noise_in_sbs_1[int(x_G_0[j][0])] += p[int(x_G_0[j][0])][4] * result[int(x_G_0[j][0])][6]
        #         x_G_0 = np.delete(x_G_0, 0, axis=0)  # 删除第一个用户
        #
        #     while len(x_G_1) > 1:
        #         for j in range(-1, -(len(x_G_1[:, 0])), -1):  # 不包含第一个元素
        #             x_noise_in_sbs_1[int(x_G_1[j][0])] += p[int(x_G_1[j][0])][5] * result[int(x_G_1[j][0])][7]
        #         x_G_1 = np.delete(x_G_1, 0, axis=0)  # 删除第一个用户
        #
        #     while len(x_G_2) > 1:
        #         for j in range(-1, -(len(x_G_2[:, 0])), -1):  # 不包含第一个元素
        #             x_noise_in_sbs_1[int(x_G_2[j][0])] += p[int(x_G_2[j][0])][6] * result[int(x_G_2[j][0])][8]
        #         x_G_2 = np.delete(x_G_2, 0, axis=0)  # 删除第一个用户
        #
        #     while len(x_G_3) > 1:
        #         for j in range(-1, -(len(x_G_3[:, 0])), -1):  # 不包含第一个元素
        #             x_noise_in_sbs_1[int(x_G_3[j][0])] += p[int(x_G_3[j][0])][7] * result[int(x_G_3[j][0])][9]
        #         x_G_3 = np.delete(x_G_3, 0, axis=0)  # 删除第一个用户
        #
        #     "SBS_2"
        #     x_noise_in_sbs_2 = [0 for i in range(8)]
        #     x_G_00 = []
        #     x_G_11 = []
        #     x_G_22 = []
        #     x_G_33 = []
        #     x_sbs_22 = [x for x, y in list(enumerate(x_a)) if y == 2 and x_xx[x] != 0]  # 在小区基站1卸载的用户索引
        #
        #     x_offload_in_sbs2_channel_0 = []
        #     x_offload_in_sbs2_channel_1 = []
        #     x_offload_in_sbs2_channel_2 = []
        #     x_offload_in_sbs2_channel_3 = []
        #
        #     "得到每个卸载到sbs2的用户的信道选择"
        #     for i in range(0, len(x_sbs_22), 1):
        #         if x_b[x_sbs_22[i]] == 0:
        #             x_offload_in_sbs2_channel_0.append(x_sbs_22[i])
        #         if x_b[x_sbs_22[i]] == 1:
        #             x_offload_in_sbs2_channel_1.append(x_sbs_22[i])
        #         if x_b[x_sbs_22[i]] == 2:
        #             x_offload_in_sbs2_channel_2.append(x_sbs_22[i])
        #         if x_b[x_sbs_22[i]] == 3:
        #             x_offload_in_sbs2_channel_3.append(x_sbs_22[i])
        #
        #     "得到0信道用户对应的信道增益"
        #     for i in range(0, len(x_offload_in_sbs2_channel_0), 1):
        #         x_a_00 = [x_offload_in_sbs2_channel_0[i], result[x_offload_in_sbs2_channel_0[i]][10]]
        #         x_G_00.append(x_a_00)
        #
        #     "得到1信道用户对应的信道增益"
        #     for i in range(0, len(x_offload_in_sbs2_channel_1), 1):
        #         x_a_11 = [x_offload_in_sbs2_channel_1[i], result[x_offload_in_sbs2_channel_1[i]][11]]
        #         x_G_11.append(x_a_11)
        #
        #     "得到2信道用户对应的信道增益"
        #     for i in range(0, len(x_offload_in_sbs2_channel_2), 1):
        #         x_a_22 = [x_offload_in_sbs2_channel_2[i], result[x_offload_in_sbs2_channel_2[i]][12]]
        #         x_G_22.append(x_a_22)
        #
        #     "得到3信道用户对应的信道增益"
        #     for i in range(0, len(x_offload_in_sbs2_channel_3), 1):
        #         x_a_33 = [x_offload_in_sbs2_channel_3[i], result[x_offload_in_sbs2_channel_3[i]][13]]
        #         x_G_33.append(x_a_33)
        #
        #     x_G_00 = np.array(x_G_00)
        #     "对数组进行降序排序"
        #     if len(x_G_00) <= 1:
        #         x_G_00 = np.array(x_G_00)
        #     else:
        #         x_G_00 = x_G_00[np.argsort(-x_G_00[:, 1])]
        #
        #     x_G_22 = np.array(x_G_22)
        #     "对数组进行降序排序"
        #     if len(x_G_22) <= 1:
        #         x_G_22 = np.array(x_G_22)
        #     else:
        #         x_G_22 = x_G_22[np.argsort(-x_G_22[:, 1])]
        #
        #     x_G_11 = np.array(x_G_11)
        #     "对数组进行降序排序"
        #     if len(x_G_11) <= 1:
        #         x_G_11 = np.array(x_G_11)
        #     else:
        #         x_G_11 = x_G_11[np.argsort(-x_G_11[:, 1])]
        #
        #     x_G_33 = np.array(x_G_33)
        #     "对数组进行降序排序"
        #     if len(x_G_33) <= 1:
        #         x_G_33 = np.array(x_G_33)
        #     else:
        #         x_G_33 = x_G_33[np.argsort(-x_G_33[:, 1])]
        #
        #     while len(x_G_00) > 1:
        #         for j in range(-1, -(len(x_G_00[:, 0])), -1):  # 不包含第一个元素
        #             x_noise_in_sbs_2[int(x_G_00[j][0])] += p[int(x_G_00[j][0])][8] * result[int(x_G_00[j][0])][10]
        #         x_G_00 = np.delete(x_G_00, 0, axis=0)  # 删除第一个用户
        #
        #     while len(x_G_11) > 1:
        #         for j in range(-1, -(len(x_G_11[:, 0])), -1):  # 不包含第一个元素
        #             x_noise_in_sbs_2[int(x_G_11[j][0])] += p[int(x_G_11[j][0])][9] * result[int(x_G_11[j][0])][11]
        #         x_G_11 = np.delete(x_G_11, 0, axis=0)  # 删除第一个用户
        #
        #     while len(x_G_22) > 1:
        #         for j in range(-1, -(len(x_G_22[:, 0])), -1):  # 不包含第一个元素
        #             x_noise_in_sbs_2[int(x_G_22[j][0])] += p[int(x_G_22[j][0])][10] * result[int(x_G_22[j][0])][12]
        #         x_G_22 = np.delete(x_G_22, 0, axis=0)  # 删除第一个用户
        #
        #     while len(x_G_33) > 1:
        #         for j in range(-1, -(len(x_G_33[:, 0])), -1):  # 不包含第一个元素
        #             x_noise_in_sbs_2[int(x_G_33[j][0])] += p[int(x_G_33[j][0])][11] * result[int(x_G_33[j][0])][13]
        #         x_G_33 = np.delete(x_G_33, 0, axis=0)  # 删除第一个用户
        #
        #     "卸载到小小区时，复用宏基站信道的跨层干扰"
        #     "sbs1"
        #     x_noise_in_sbs_1_channel_0 = [0 for i in range(8)]
        #     x_noise_in_sbs_1_channel_1 = [0 for i in range(8)]
        #     x_noise_in_sbs_1_channel_2 = [0 for i in range(8)]
        #     x_noise_in_sbs_1_channel_3 = [0 for i in range(8)]
        #
        #     for i in range(0, len(x_offload_in_sbs1_channel_0), 1):
        #         x_noise_in_sbs_1_channel_0[x_offload_in_sbs1_channel_0[i]] = p[x_offload_in_sbs1_channel_0[i]][0] \
        #                                                                  * result[x_offload_in_sbs1_channel_0[i]][6]
        #     for i in range(0, len(x_offload_in_sbs1_channel_1), 1):
        #         x_noise_in_sbs_1_channel_1[x_offload_in_sbs1_channel_1[i]] = p[x_offload_in_sbs1_channel_1[i]][1] \
        #                                                                  * result[x_offload_in_sbs1_channel_1[i]][7]
        #     for i in range(0, len(x_offload_in_sbs1_channel_2), 1):
        #         x_noise_in_sbs_1_channel_2[x_offload_in_sbs1_channel_2[i]] = p[x_offload_in_sbs1_channel_2[i]][2] \
        #                                                                  * result[x_offload_in_sbs1_channel_2[i]][8]
        #     for i in range(0, len(x_offload_in_sbs1_channel_3), 1):
        #         x_noise_in_sbs_1_channel_3[x_offload_in_sbs1_channel_3[i]] = p[x_offload_in_sbs1_channel_3[i]][3] \
        #                                                                  * result[x_offload_in_sbs1_channel_3[i]][9]
        #
        #     "sbs2"
        #     x_noise_in_sbs_2_channel_0 = [0 for i in range(8)]
        #     x_noise_in_sbs_2_channel_1 = [0 for i in range(8)]
        #     x_noise_in_sbs_2_channel_2 = [0 for i in range(8)]
        #     x_noise_in_sbs_2_channel_3 = [0 for i in range(8)]
        #
        #     for i in range(0, len(x_offload_in_sbs2_channel_0), 1):
        #         x_noise_in_sbs_2_channel_0[x_offload_in_sbs2_channel_0[i]] = p[x_offload_in_sbs2_channel_0[i]][4] \
        #                                                                  * result[x_offload_in_sbs2_channel_0[i]][10]
        #     for i in range(0, len(x_offload_in_sbs2_channel_1), 1):
        #         x_noise_in_sbs_2_channel_1[x_offload_in_sbs2_channel_1[i]] = p[x_offload_in_sbs2_channel_1[i]][5] \
        #                                                                  * result[x_offload_in_sbs2_channel_1[i]][11]
        #     for i in range(0, len(x_offload_in_sbs2_channel_2), 1):
        #         x_noise_in_sbs_2_channel_2[x_offload_in_sbs2_channel_2[i]] = p[x_offload_in_sbs2_channel_2[i]][6] \
        #                                                                  * result[x_offload_in_sbs2_channel_2[i]][12]
        #     for i in range(0, len(x_offload_in_sbs2_channel_3), 1):
        #         x_noise_in_sbs_2_channel_3[x_offload_in_sbs2_channel_3[i]] = p[x_offload_in_sbs2_channel_3[i]][7] \
        #                                                                  * result[x_offload_in_sbs2_channel_3[i]][13]
        #
        #     "计算不同基站，相同信道的干扰"
        #     "sbs1"
        #     x_noise_out_sbs1_channel_0 = [0 for i in range(8)]
        #     x_noise_out_sbs1_channel_1 = [0 for i in range(8)]
        #     x_noise_out_sbs1_channel_2 = [0 for i in range(8)]
        #     x_noise_out_sbs1_channel_3 = [0 for i in range(8)]
        #
        #     for i in range(0, len(x_offload_in_sbs2_channel_0), ):
        #         x_noise_out_sbs1_channel_0[x_offload_in_sbs2_channel_0[i]] += p[x_offload_in_sbs2_channel_0[i]][4] \
        #                                                                   * result[x_offload_in_sbs2_channel_0[i]][6]
        #     for i in range(0, len(x_offload_in_sbs2_channel_1), ):
        #         x_noise_out_sbs1_channel_1[x_offload_in_sbs2_channel_1[i]] += p[x_offload_in_sbs2_channel_1[i]][5] \
        #                                                                   * result[x_offload_in_sbs2_channel_1[i]][7]
        #     for i in range(0, len(x_offload_in_sbs2_channel_2), ):
        #         x_noise_out_sbs1_channel_2[x_offload_in_sbs2_channel_2[i]] += p[x_offload_in_sbs2_channel_2[i]][6] \
        #                                                                   * result[x_offload_in_sbs2_channel_2[i]][8]
        #     for i in range(0, len(x_offload_in_sbs2_channel_3), ):
        #         x_noise_out_sbs1_channel_3[x_offload_in_sbs2_channel_3[i]] += p[x_offload_in_sbs2_channel_3[i]][7] \
        #                                                                   * result[x_offload_in_sbs2_channel_3[i]][9]
        #
        #     "sbs2"
        #     x_noise_out_sbs2_channel_0 = [0 for i in range(8)]
        #     x_noise_out_sbs2_channel_1 = [0 for i in range(8)]
        #     x_noise_out_sbs2_channel_2 = [0 for i in range(8)]
        #     x_noise_out_sbs2_channel_3 = [0 for i in range(8)]
        #     for i in range(0, len(x_offload_in_sbs1_channel_0), ):
        #         x_noise_out_sbs2_channel_0[x_offload_in_sbs1_channel_0[i]] += p[x_offload_in_sbs1_channel_0[i]][0] \
        #                                                                   * result[x_offload_in_sbs1_channel_0[i]][10]
        #     for i in range(0, len(x_offload_in_sbs1_channel_1), ):
        #         x_noise_out_sbs2_channel_1[x_offload_in_sbs1_channel_1[i]] += p[x_offload_in_sbs1_channel_1[i]][1] \
        #                                                                   * result[x_offload_in_sbs1_channel_1[i]][11]
        #     for i in range(0, len(x_offload_in_sbs1_channel_2), ):
        #         x_noise_out_sbs2_channel_2[x_offload_in_sbs1_channel_2[i]] += p[x_offload_in_sbs1_channel_2[i]][2] \
        #                                                                   * result[x_offload_in_sbs1_channel_2[i]][12]
        #     for i in range(0, len(x_offload_in_sbs1_channel_3), ):
        #         x_noise_out_sbs2_channel_3[x_offload_in_sbs1_channel_3[i]] += p[x_offload_in_sbs1_channel_3[i]][3] \
        #                                                                   * result[x_offload_in_sbs1_channel_3[i]][13]
        #
        #     "计算每个用户的上传速率"
        #     x_R = [0 for i in range(8)]
        #     x_r = [0 for i in range(8)]
        #
        #     for i in range(8):
        #         if x_xx[i] == 1 and x_a[i] == 0 and x_b[i] == 0:
        #             x_r[i] = p[i][0] * result[i][2] / (x_noise_in_mbs_channel_0[i] + self.noise)
        #         if x_xx[i] == 1 and x_a[i] == 0 and x_b[i] == 1:
        #             x_r[i] = p[i][1] * result[i][3] / (x_noise_in_mbs_channel_1[i] + self.noise)
        #         if x_xx[i] == 1 and x_a[i] == 0 and x_b[i] == 2:
        #             x_r[i] = p[i][2] * result[i][4] / (x_noise_in_mbs_channel_2[i] + self.noise)
        #         if x_xx[i] == 1 and x_a[i] == 0 and x_b[i] == 3:
        #             x_r[i] = p[i][3] * result[i][5] / (x_noise_in_mbs_channel_3[i] + self.noise)
        #         if x_xx[i] == 1 and x_a[i] == 1 and x_b[i] == 0:
        #             x_r[i] = p[i][4] * result[i][6] / (
        #                         x_noise_in_sbs_1[i] + x_noise_in_sbs_1_channel_0[i] + x_noise_out_sbs1_channel_0[i] + self.noise)
        #         if x_xx[i] == 1 and x_a[i] == 1 and x_b[i] == 1:
        #             x_r[i] = p[i][5] * result[i][7] / (
        #                         x_noise_in_sbs_1[i] + x_noise_in_sbs_1_channel_1[i] + x_noise_out_sbs1_channel_1[i] + self.noise)
        #         if x_xx[i] == 1 and x_a[i] == 1 and x_b[i] == 2:
        #             x_r[i] = p[i][6] * result[i][8] / (
        #                         x_noise_in_sbs_1[i] + x_noise_in_sbs_1_channel_2[i] + x_noise_out_sbs1_channel_2[i] + self.noise)
        #         if x_xx[i] == 1 and x_a[i] == 1 and x_b[i] == 3:
        #             x_r[i] = p[i][7] * result[i][9] / (
        #                         x_noise_in_sbs_1[i] + x_noise_in_sbs_1_channel_3[i] + x_noise_out_sbs1_channel_3[i] + self.noise)
        #         if x_xx[i] == 1 and x_a[i] == 2 and x_b[i] == 0:
        #             x_r[i] = p[i][8] * result[i][10] / (
        #                         x_noise_in_sbs_2[i] + x_noise_in_sbs_2_channel_0[i] + x_noise_out_sbs2_channel_0[i] + self.noise)
        #         if x_xx[i] == 1 and x_a[i] == 2 and x_b[i] == 1:
        #             x_r[i] = p[i][9] * result[i][11] / (
        #                         x_noise_in_sbs_2[i] + x_noise_in_sbs_2_channel_1[i] + x_noise_out_sbs2_channel_1[i] + self.noise)
        #         if x_xx[i] == 1 and x_a[i] == 2 and x_b[i] == 2:
        #             x_r[i] = p[i][10] * result[i][12] / (
        #                         x_noise_in_sbs_2[i] + x_noise_in_sbs_2_channel_2[i] + x_noise_out_sbs2_channel_2[i] + self.noise)
        #         if x_xx[i] == 1 and x_a[i] == 2 and x_b[i] == 3:
        #             x_r[i] = p[i][11] * result[i][13] / (
        #                         x_noise_in_sbs_2[i] + x_noise_in_sbs_2_channel_3[i] + x_noise_out_sbs2_channel_3[i] + self.noise)
        #
        #     for i in range(8):
        #         x_R[i] = self.w * np.log2(1 + x_r[i])
        #
        #     "优化每个用户本地的计算资源"
        #     x_f_local = []
        #     for i in range(8):
        #         x_local = result[i][1] / self.d_max
        #         x_f_local.append(x_local)
        #
        #     "计算本地处理时间和能耗"
        #     x_e_local = [0 for i in range(8)]
        #     for i in range(8):
        #         if x_xx[i] == 0:
        #             x_e_local[i] = self.k * (x_f_local[i] ** 2) * result[i][1]
        #
        #     "计算上传和处理时间"
        #     x_t_tran = [0 for i in range(8)]
        #     x_t_mec = [0 for i in range(8)]
        #     x_e_mec = [0 for i in range(8)]
        #     for i in range(8):
        #         if x_xx[i] == 1 and x_a[i] == 0 and x_b[i] == 0:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][0] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_m_mec
        #         if x_xx[i] == 1 and x_a[i] == 0 and x_b[i] == 1:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][1] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_m_mec
        #         if x_xx[i] == 1 and x_a[i] == 0 and x_b[i] == 2:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][2] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_m_mec
        #         if x_xx[i] == 1 and x_a[i] == 0 and x_b[i] == 3:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][3] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_m_mec
        #         if x_xx[i] == 1 and x_a[i] == 1 and x_b[i] == 0:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][4] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_s_mec
        #         if x_xx[i] == 1 and x_a[i] == 1 and x_b[i] == 1:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][5] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_s_mec
        #         if x_xx[i] == 1 and x_a[i] == 1 and x_b[i] == 2:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][6] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_s_mec
        #         if x_xx[i] == 1 and x_a[i] == 1 and x_b[i] == 3:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][7] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_s_mec
        #         if x_xx[i] == 1 and x_a[i] == 2 and x_b[i] == 0:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][8] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_s_mec
        #         if x_xx[i] == 1 and x_a[i] == 2 and x_b[i] == 1:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][9] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_s_mec
        #         if x_xx[i] == 1 and x_a[i] == 2 and x_b[i] == 2:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][10] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_s_mec
        #         if x_xx[i] == 1 and x_a[i] == 2 and x_b[i] == 3:
        #             x_t_tran[i] = result[i][0] / x_R[i]
        #             x_e_mec[i] = p[i][11] * x_t_tran[i]
        #             x_t_mec[i] = result[i][1] / self.f_s_mec
        #
        #     "计算所有用户的奖励"
        #     x_reward = [0 for i in range(8)]
        #     "加入最低传输速率的限制,小于这个速率则看为传输失败"
        #     x_R_min = [0 for i in range(8)]
        #     for i in range(8):
        #         x_R_min[i] = result[i][0] / (self.d_max - x_t_mec[i])
        #
        #     for i in range(8):
        #         if x_xx[i] == 1 and x_R[i] < x_R_min[i]:
        #             x_reward[i] = -0.02
        #         else:
        #             x_reward[i] = -(x_e_local[i] + x_e_mec[i])
        #
        #     "计算适应度值"
        #     fitness=x_reward
        #     "对个体值进行更新"
        #     for i in range(8):
        #         if fitness[i]>fitness_best[i]:
        #             p_best[i]=x_best[i]
        #     "更新全局最优"
        #     for i in range(8):
        #         if fitness[i]>fitness_best[i]:
        #             g_best[i]=x_best[i]
        #             fitness_best[i]=fitness[i]



        # "迭代寻优"
        # user_reward = []
        # for i in range(8):
        #     for jj in range(13):
        #         "所有用户连接基站数目的列表"
        #         i_a = [0 for i in range(8)]
        #         "所有用户选择信道数目的列表"
        #         i_b = [0 for i in range(8)]
        #         "生成二进制卸载列表"
        #         i_xx = [0 for i in range(8)]
        #
        #         "分解所有动作为连接基站和信道的选择"
        #         for i in range(8):
        #             if jj % 3 == 0 and jj % 4 == 0 and jj != 0:
        #                 i_xx[i] = 0
        #             else:
        #                 i_xx[i] = 1
        #             i_a[i] = jj % 3  # 连接的基站数目
        #             i_b[i] = (jj // 3) % 4  # 选择的信道数目
        #
        #         "计算在MBS卸载时，小小区复用信道产生的干扰"
        #
        #         i_mbs_0 = [x for x, y in list(enumerate(i_a)) if y == 0 and xx[x] != 0]  # 在宏基站卸载的用户索引
        #         i_sbs_1 = [x for x, y in list(enumerate(i_a)) if y == 1 and xx[x] != 0]  # 在小区基站1卸载的用户索引
        #         i_sbs_2 = [x for x, y in list(enumerate(i_a)) if y == 2 and xx[x] != 0]  # 在小区基站2卸载的用户索引
        #         i_channel_0 = [x for x, y in list(enumerate(i_b)) if y == 0 and i_xx[x] != 0 and i_a[x] != 0]  # 小小区基站选择信道0的用户索引
        #         i_channel_1 = [x for x, y in list(enumerate(i_b)) if y == 1 and i_xx[x] != 0 and i_a[x] != 0]  # 小小区基站选择信道1的用户索引
        #         i_channel_2 = [x for x, y in list(enumerate(i_b)) if y == 2 and i_xx[x] != 0 and i_a[x] != 0]  # 小小区基站选择信道2的用户索引
        #         i_channel_3 = [x for x, y in list(enumerate(i_b)) if y == 3 and i_xx[x] != 0 and i_a[x] != 0]  # 小小区基站选择信道3的用户索引
        #
        #         i_noise_in_mbs_channel_0 = [0 for i in range(8)]
        #         i_noise_in_mbs_channel_1 = [0 for i in range(8)]
        #         i_noise_in_mbs_channel_2 = [0 for i in range(8)]
        #         i_noise_in_mbs_channel_3 = [0 for i in range(8)]
        #
        #         "计算在mbs卸载使用信道的用户，其它小小区使用相同信道的干扰"
        #         for i in range(0, len(i_mbs_0)):
        #             for j in range(0, len(i_channel_0), 1):
        #                 if i_channel_0[j] in i_sbs_1:
        #                     i_noise_in_mbs_channel_0[i_channel_0[j]] += p[channel_0[j]][i_b[i_channel_0[j]]] * \
        #                                                             result[i_channel_0[j]][2 + i_b[i_channel_0[j]]]
        #                 if i_channel_0[j] in i_sbs_2:
        #                     i_noise_in_mbs_channel_0[i_channel_0[j]] += p[channel_0[j]][i_b[i_channel_0[j]]] * \
        #                                                             result[i_channel_0[j]][2 + i_b[i_channel_0[j]]]
        #             for j in range(0, len(i_channel_1), 1):
        #                 if i_channel_1[j] in i_sbs_1:
        #                     i_noise_in_mbs_channel_1[i_channel_1[j]] += p[channel_1[j]][i_b[i_channel_1[j]]] * \
        #                                                             result[i_channel_1[j]][2 + b[i_channel_1[j]]]
        #                 if i_channel_1[j] in i_sbs_2:
        #                     i_noise_in_mbs_channel_1[i_channel_1[j]] += p[channel_1[j]][i_b[i_channel_1[j]]] * \
        #                                                             result[i_channel_1[j]][2 + i_b[i_channel_1[j]]]
        #             for j in range(0, len(i_channel_2), 1):
        #                 if i_channel_2[j] in i_sbs_1:
        #                     i_noise_in_mbs_channel_2[i_channel_2[j]] += p[channel_2[j]][i_b[i_channel_2[j]]] * \
        #                                                             result[i_channel_2[j]][2 + i_b[i_channel_2[j]]]
        #                 if i_channel_2[j] in i_sbs_2:
        #                     i_noise_in_mbs_channel_2[i_channel_2[j]] += p[channel_2[j]][i_b[i_channel_2[j]]] * \
        #                                                             result[i_channel_2[j]][2 + i_b[i_channel_2[j]]]
        #             for j in range(0, len(i_channel_3), 1):
        #                 if i_channel_3[j] in i_sbs_1:
        #                     i_noise_in_mbs_channel_3[i_channel_3[j]] += p[channel_3[j]][i_b[i_channel_3[j]]] * \
        #                                                             result[i_channel_3[j]][2 + i_b[i_channel_3[j]]]
        #                 if i_channel_3[j] in i_sbs_2:
        #                     i_noise_in_mbs_channel_3[i_channel_3[j]] += p[channel_3[j]][i_b[i_channel_3[j]]] * \
        #                                                             result[i_channel_3[j]][2 + i_b[i_channel_3[j]]]
        #
        #         "同基站同信道干扰"
        #         "SBS_1"
        #         i_noise_in_sbs_1 = [0 for i in range(8)]
        #         i_G_0 = []
        #         i_G_1 = []
        #         i_G_2 = []
        #         i_G_3 = []
        #         i_sbs_11 = [x for x, y in list(enumerate(i_a)) if y == 1 and i_xx[x] != 0]  # 在小区基站1卸载的用户索引
        #
        #         i_offload_in_sbs1_channel_0 = []
        #         i_offload_in_sbs1_channel_1 = []
        #         i_offload_in_sbs1_channel_2 = []
        #         i_offload_in_sbs1_channel_3 = []
        #
        #         "得到每个卸载到sbs1的用户的信道选择"
        #         for i in range(0, len(i_sbs_11), 1):
        #             if i_b[i_sbs_11[i]] == 0:
        #                 i_offload_in_sbs1_channel_0.append(i_sbs_11[i])
        #             if i_b[i_sbs_11[i]] == 1:
        #                 i_offload_in_sbs1_channel_1.append(i_sbs_11[i])
        #             if i_b[i_sbs_11[i]] == 2:
        #                 i_offload_in_sbs1_channel_2.append(i_sbs_11[i])
        #             if i_b[i_sbs_11[i]] == 3:
        #                 i_offload_in_sbs1_channel_3.append(i_sbs_11[i])
        #
        #         "得到0信道用户对应的信道增益"
        #         for i in range(0, len(i_offload_in_sbs1_channel_0), 1):
        #             i_a_0 = [i_offload_in_sbs1_channel_0[i], result[i_offload_in_sbs1_channel_0[i]][6]]
        #             i_G_0.append(i_a_0)
        #
        #         "得到1信道用户对应的信道增益"
        #         for i in range(0, len(i_offload_in_sbs1_channel_1), 1):
        #             i_a_1 = [i_offload_in_sbs1_channel_1[i], result[i_offload_in_sbs1_channel_1[i]][7]]
        #             i_G_1.append(i_a_1)
        #
        #         "得到2信道用户对应的信道增益"
        #         for i in range(0, len(i_offload_in_sbs1_channel_2), 1):
        #             i_a_2 = [i_offload_in_sbs1_channel_2[i], result[i_offload_in_sbs1_channel_2[i]][8]]
        #             i_G_2.append(i_a_2)
        #
        #         "得到3信道用户对应的信道增益"
        #         for i in range(0, len(i_offload_in_sbs1_channel_3), 1):
        #             i_a_3 = [i_offload_in_sbs1_channel_3[i], result[i_offload_in_sbs1_channel_3[i]][9]]
        #             i_G_3.append(i_a_3)
        #
        #         i_G_0 = np.array(i_G_0)
        #         "对数组进行降序排序"
        #         if len(i_G_0) <= 1:
        #             i_G_0 = np.array(i_G_0)
        #         else:
        #             i_G_0 = i_G_0[np.argsort(-i_G_0[:, 1])]
        #
        #         i_G_2 = np.array(i_G_2)
        #         "对数组进行降序排序"
        #         if len(i_G_2) <= 1:
        #             i_G_2 = np.array(i_G_2)
        #         else:
        #             i_G_2 = i_G_2[np.argsort(-i_G_2[:, 1])]
        #
        #         i_G_1 = np.array(i_G_1)
        #         "对数组进行降序排序"
        #         if len(i_G_1) <= 1:
        #             i_G_1 = np.array(i_G_1)
        #         else:
        #             i_G_1 = i_G_1[np.argsort(-i_G_1[:, 1])]
        #
        #         i_G_3 = np.array(i_G_3)
        #         "对数组进行降序排序"
        #         if len(i_G_3) <= 1:
        #             i_G_3 = np.array(i_G_3)
        #         else:
        #             i_G_3 = i_G_3[np.argsort(-i_G_3[:, 1])]
        #
        #         while len(i_G_0) > 1:
        #             for j in range(-1, -(len(i_G_0[:, 0])), -1):  # 不包含第一个元素
        #                 i_noise_in_sbs_1[int(i_G_0[j][0])] += p[int(i_G_0[j][0])][4] * result[int(i_G_0[j][0])][6]
        #             i_G_0 = np.delete(i_G_0, 0, axis=0)  # 删除第一个用户
        #
        #         while len(i_G_1) > 1:
        #             for j in range(-1, -(len(i_G_1[:, 0])), -1):  # 不包含第一个元素
        #                 i_noise_in_sbs_1[int(i_G_1[j][0])] += p[int(i_G_1[j][0])][5] * result[int(i_G_1[j][0])][7]
        #             i_G_1 = np.delete(i_G_1, 0, axis=0)  # 删除第一个用户
        #
        #         while len(i_G_2) > 1:
        #             for j in range(-1, -(len(i_G_2[:, 0])), -1):  # 不包含第一个元素
        #                 i_noise_in_sbs_1[int(i_G_2[j][0])] += p[int(i_G_2[j][0])][6] * result[int(i_G_2[j][0])][8]
        #             i_G_2 = np.delete(i_G_2, 0, axis=0)  # 删除第一个用户
        #
        #         while len(i_G_3) > 1:
        #             for j in range(-1, -(len(i_G_3[:, 0])), -1):  # 不包含第一个元素
        #                 i_noise_in_sbs_1[int(i_G_3[j][0])] += p[int(i_G_3[j][0])][7] * result[int(i_G_3[j][0])][9]
        #             i_G_3 = np.delete(i_G_3, 0, axis=0)  # 删除第一个用户
        #
        #         "SBS_2"
        #         i_noise_in_sbs_2 = [0 for i in range(8)]
        #         i_G_00 = []
        #         i_G_11 = []
        #         i_G_22 = []
        #         i_G_33 = []
        #         i_sbs_22 = [x for x, y in list(enumerate(i_a)) if y == 2 and i_xx[x] != 0]  # 在小区基站1卸载的用户索引
        #
        #         i_offload_in_sbs2_channel_0 = []
        #         i_offload_in_sbs2_channel_1 = []
        #         i_offload_in_sbs2_channel_2 = []
        #         i_offload_in_sbs2_channel_3 = []
        #
        #         "得到每个卸载到sbs2的用户的信道选择"
        #         for i in range(0, len(i_sbs_22), 1):
        #             if i_b[i_sbs_22[i]] == 0:
        #                 i_offload_in_sbs2_channel_0.append(i_sbs_22[i])
        #             if i_b[i_sbs_22[i]] == 1:
        #                 i_offload_in_sbs2_channel_1.append(i_sbs_22[i])
        #             if i_b[i_sbs_22[i]] == 2:
        #                 i_offload_in_sbs2_channel_2.append(i_sbs_22[i])
        #             if i_b[i_sbs_22[i]] == 3:
        #                 i_offload_in_sbs2_channel_3.append(i_sbs_22[i])
        #
        #         "得到0信道用户对应的信道增益"
        #         for i in range(0, len(i_offload_in_sbs2_channel_0), 1):
        #             i_a_00 = [i_offload_in_sbs2_channel_0[i], result[i_offload_in_sbs2_channel_0[i]][10]]
        #             i_G_00.append(i_a_00)
        #
        #         "得到1信道用户对应的信道增益"
        #         for i in range(0, len(i_offload_in_sbs2_channel_1), 1):
        #             i_a_11 = [i_offload_in_sbs2_channel_1[i], result[i_offload_in_sbs2_channel_1[i]][11]]
        #             i_G_11.append(i_a_11)
        #
        #         "得到2信道用户对应的信道增益"
        #         for i in range(0, len(i_offload_in_sbs2_channel_2), 1):
        #             i_a_22 = [i_offload_in_sbs2_channel_2[i], result[i_offload_in_sbs2_channel_2[i]][12]]
        #             i_G_22.append(i_a_22)
        #
        #         "得到3信道用户对应的信道增益"
        #         for i in range(0, len(i_offload_in_sbs2_channel_3), 1):
        #             i_a_33 = [i_offload_in_sbs2_channel_3[i], result[i_offload_in_sbs2_channel_3[i]][13]]
        #             i_G_33.append(i_a_33)
        #
        #         i_G_00 = np.array(i_G_00)
        #         "对数组进行降序排序"
        #         if len(i_G_00) <= 1:
        #             i_G_00 = np.array(i_G_00)
        #         else:
        #             i_G_00 = i_G_00[np.argsort(-i_G_00[:, 1])]
        #
        #         i_G_22 = np.array(i_G_22)
        #         "对数组进行降序排序"
        #         if len(i_G_22) <= 1:
        #             i_G_22 = np.array(i_G_22)
        #         else:
        #             i_G_22 = i_G_22[np.argsort(-i_G_22[:, 1])]
        #
        #         i_G_11 = np.array(i_G_11)
        #         "对数组进行降序排序"
        #         if len(i_G_11) <= 1:
        #             i_G_11 = np.array(i_G_11)
        #         else:
        #             i_G_11 = i_G_11[np.argsort(-i_G_11[:, 1])]
        #
        #         i_G_33 = np.array(i_G_33)
        #         "对数组进行降序排序"
        #         if len(i_G_33) <= 1:
        #             i_G_33 = np.array(i_G_33)
        #         else:
        #             i_G_33 = i_G_33[np.argsort(-i_G_33[:, 1])]
        #
        #         while len(i_G_00) > 1:
        #             for j in range(-1, -(len(i_G_00[:, 0])), -1):  # 不包含第一个元素
        #                 i_noise_in_sbs_2[int(i_G_00[j][0])] += p[int(i_G_00[j][0])][8] * result[int(i_G_00[j][0])][10]
        #             i_G_00 = np.delete(i_G_00, 0, axis=0)  # 删除第一个用户
        #
        #         while len(i_G_11) > 1:
        #             for j in range(-1, -(len(i_G_11[:, 0])), -1):  # 不包含第一个元素
        #                 i_noise_in_sbs_2[int(i_G_11[j][0])] += p[int(i_G_11[j][0])][9] * result[int(i_G_11[j][0])][11]
        #             i_G_11 = np.delete(i_G_11, 0, axis=0)  # 删除第一个用户
        #
        #         while len(i_G_22) > 1:
        #             for j in range(-1, -(len(i_G_22[:, 0])), -1):  # 不包含第一个元素
        #                 i_noise_in_sbs_2[int(i_G_22[j][0])] += p[int(i_G_22[j][0])][10] * result[int(i_G_22[j][0])][12]
        #             i_G_22 = np.delete(i_G_22, 0, axis=0)  # 删除第一个用户
        #
        #         while len(i_G_33) > 1:
        #             for j in range(-1, -(len(i_G_33[:, 0])), -1):  # 不包含第一个元素
        #                 i_noise_in_sbs_2[int(i_G_33[j][0])] += p[int(i_G_33[j][0])][11] * result[int(i_G_33[j][0])][13]
        #             i_G_33 = np.delete(i_G_33, 0, axis=0)  # 删除第一个用户
        #
        #         "卸载到小小区时，复用宏基站信道的跨层干扰"
        #         "sbs1"
        #         i_noise_in_sbs_1_channel_0 = [0 for i in range(8)]
        #         i_noise_in_sbs_1_channel_1 = [0 for i in range(8)]
        #         i_noise_in_sbs_1_channel_2 = [0 for i in range(8)]
        #         i_noise_in_sbs_1_channel_3 = [0 for i in range(8)]
        #
        #         for i in range(0, len(i_offload_in_sbs1_channel_0), 1):
        #             i_noise_in_sbs_1_channel_0[i_offload_in_sbs1_channel_0[i]] = p[i_offload_in_sbs1_channel_0[i]][0] \
        #                                                                      * result[i_offload_in_sbs1_channel_0[i]][6]
        #         for i in range(0, len(i_offload_in_sbs1_channel_1), 1):
        #             i_noise_in_sbs_1_channel_1[i_offload_in_sbs1_channel_1[i]] = p[i_offload_in_sbs1_channel_1[i]][1] \
        #                                                                      * result[i_offload_in_sbs1_channel_1[i]][7]
        #         for i in range(0, len(i_offload_in_sbs1_channel_2), 1):
        #             i_noise_in_sbs_1_channel_2[i_offload_in_sbs1_channel_2[i]] = p[i_offload_in_sbs1_channel_2[i]][2] \
        #                                                                      * result[i_offload_in_sbs1_channel_2[i]][8]
        #         for i in range(0, len(i_offload_in_sbs1_channel_3), 1):
        #             i_noise_in_sbs_1_channel_3[i_offload_in_sbs1_channel_3[i]] = p[i_offload_in_sbs1_channel_3[i]][3] \
        #                                                                      * result[i_offload_in_sbs1_channel_3[i]][9]
        #
        #         "sbs2"
        #         i_noise_in_sbs_2_channel_0 = [0 for i in range(8)]
        #         i_noise_in_sbs_2_channel_1 = [0 for i in range(8)]
        #         i_noise_in_sbs_2_channel_2 = [0 for i in range(8)]
        #         i_noise_in_sbs_2_channel_3 = [0 for i in range(8)]
        #
        #         for i in range(0, len(i_offload_in_sbs2_channel_0), 1):
        #             i_noise_in_sbs_2_channel_0[i_offload_in_sbs2_channel_0[i]] = p[i_offload_in_sbs2_channel_0[i]][4] \
        #                                                                      * result[i_offload_in_sbs2_channel_0[i]][10]
        #         for i in range(0, len(i_offload_in_sbs2_channel_1), 1):
        #             i_noise_in_sbs_2_channel_1[i_offload_in_sbs2_channel_1[i]] = p[i_offload_in_sbs2_channel_1[i]][5] \
        #                                                                      * result[i_offload_in_sbs2_channel_1[i]][11]
        #         for i in range(0, len(i_offload_in_sbs2_channel_2), 1):
        #             i_noise_in_sbs_2_channel_2[i_offload_in_sbs2_channel_2[i]] = p[i_offload_in_sbs2_channel_2[i]][6] \
        #                                                                      * result[i_offload_in_sbs2_channel_2[i]][12]
        #         for i in range(0, len(i_offload_in_sbs2_channel_3), 1):
        #             i_noise_in_sbs_2_channel_3[i_offload_in_sbs2_channel_3[i]] = p[i_offload_in_sbs2_channel_3[i]][7] \
        #                                                                      * result[i_offload_in_sbs2_channel_3[i]][13]
        #
        #         "计算不同基站，相同信道的干扰"
        #         "sbs1"
        #         i_noise_out_sbs1_channel_0 = [0 for i in range(8)]
        #         i_noise_out_sbs1_channel_1 = [0 for i in range(8)]
        #         i_noise_out_sbs1_channel_2 = [0 for i in range(8)]
        #         i_noise_out_sbs1_channel_3 = [0 for i in range(8)]
        #
        #         for i in range(0, len(i_offload_in_sbs2_channel_0), ):
        #             i_noise_out_sbs1_channel_0[i_offload_in_sbs2_channel_0[i]] += p[i_offload_in_sbs2_channel_0[i]][4] \
        #                                                                       * result[i_offload_in_sbs2_channel_0[i]][6]
        #         for i in range(0, len(i_offload_in_sbs2_channel_1), ):
        #             i_noise_out_sbs1_channel_1[i_offload_in_sbs2_channel_1[i]] += p[i_offload_in_sbs2_channel_1[i]][5] \
        #                                                                       * result[i_offload_in_sbs2_channel_1[i]][7]
        #         for i in range(0, len(i_offload_in_sbs2_channel_2), ):
        #             i_noise_out_sbs1_channel_2[i_offload_in_sbs2_channel_2[i]] += p[i_offload_in_sbs2_channel_2[i]][6] \
        #                                                                       * result[i_offload_in_sbs2_channel_2[i]][8]
        #         for i in range(0, len(i_offload_in_sbs2_channel_3), ):
        #             i_noise_out_sbs1_channel_3[i_offload_in_sbs2_channel_3[i]] += p[i_offload_in_sbs2_channel_3[i]][7] \
        #                                                                       * result[i_offload_in_sbs2_channel_3[i]][9]
        #
        #         "sbs2"
        #         i_noise_out_sbs2_channel_0 = [0 for i in range(8)]
        #         i_noise_out_sbs2_channel_1 = [0 for i in range(8)]
        #         i_noise_out_sbs2_channel_2 = [0 for i in range(8)]
        #         i_noise_out_sbs2_channel_3 = [0 for i in range(8)]
        #         for i in range(0, len(i_offload_in_sbs1_channel_0), ):
        #             i_noise_out_sbs2_channel_0[i_offload_in_sbs1_channel_0[i]] += p[i_offload_in_sbs1_channel_0[i]][0] \
        #                                                                       * result[i_offload_in_sbs1_channel_0[i]][10]
        #         for i in range(0, len(i_offload_in_sbs1_channel_1), ):
        #             i_noise_out_sbs2_channel_1[i_offload_in_sbs1_channel_1[i]] += p[i_offload_in_sbs1_channel_1[i]][1] \
        #                                                                       * result[i_offload_in_sbs1_channel_1[i]][11]
        #         for i in range(0, len(i_offload_in_sbs1_channel_2), ):
        #             i_noise_out_sbs2_channel_2[i_offload_in_sbs1_channel_2[i]] += p[i_offload_in_sbs1_channel_2[i]][2] \
        #                                                                       * result[i_offload_in_sbs1_channel_2[i]][12]
        #         for i in range(0, len(i_offload_in_sbs1_channel_3), ):
        #             i_noise_out_sbs2_channel_3[i_offload_in_sbs1_channel_3[i]] += p[i_offload_in_sbs1_channel_3[i]][3] \
        #                                                                       * result[i_offload_in_sbs1_channel_3[i]][13]
        #
        #         "计算每个用户的上传速率"
        #         i_R = [0 for i in range(8)]
        #         i_r = [0 for i in range(8)]
        #
        #         for i in range(8):
        #             if i_xx[i] == 1 and i_a[i] == 0 and i_b[i] == 0:
        #                 i_r[i] = p[i][0] * result[i][2] / (i_noise_in_mbs_channel_0[i] + self.noise)
        #             if i_xx[i] == 1 and i_a[i] == 0 and i_b[i] == 1:
        #                 i_r[i] = p[i][1] * result[i][3] / (i_noise_in_mbs_channel_1[i] + self.noise)
        #             if i_xx[i] == 1 and i_a[i] == 0 and i_b[i] == 2:
        #                 i_r[i] = p[i][2] * result[i][4] / (i_noise_in_mbs_channel_2[i] + self.noise)
        #             if i_xx[i] == 1 and i_a[i] == 0 and i_b[i] == 3:
        #                 i_r[i] = p[i][3] * result[i][5] / (i_noise_in_mbs_channel_3[i] + self.noise)
        #             if i_xx[i] == 1 and i_a[i] == 1 and i_b[i] == 0:
        #                 i_r[i] = p[i][4] * result[i][6] / (
        #                             i_noise_in_sbs_1[i] + i_noise_in_sbs_1_channel_0[i] + i_noise_out_sbs1_channel_0[
        #                         i] + self.noise)
        #             if i_xx[i] == 1 and i_a[i] == 1 and i_b[i] == 1:
        #                 i_r[i] = p[i][5] * result[i][7] / (
        #                             i_noise_in_sbs_1[i] + i_noise_in_sbs_1_channel_1[i] + i_noise_out_sbs1_channel_1[
        #                         i] + self.noise)
        #             if i_xx[i] == 1 and i_a[i] == 1 and i_b[i] == 2:
        #                 i_r[i] = p[i][6] * result[i][8] / (
        #                             i_noise_in_sbs_1[i] + i_noise_in_sbs_1_channel_2[i] + i_noise_out_sbs1_channel_2[
        #                         i] + self.noise)
        #             if i_xx[i] == 1 and i_a[i] == 1 and i_b[i] == 3:
        #                 i_r[i] = p[i][7] * result[i][9] / (
        #                             i_noise_in_sbs_1[i] + i_noise_in_sbs_1_channel_3[i] + i_noise_out_sbs1_channel_3[
        #                         i] + self.noise)
        #             if i_xx[i] == 1 and i_a[i] == 2 and i_b[i] == 0:
        #                 i_r[i] = p[i][8] * result[i][10] / (
        #                             i_noise_in_sbs_2[i] + i_noise_in_sbs_2_channel_0[i] + i_noise_out_sbs2_channel_0[
        #                         i] + self.noise)
        #             if i_xx[i] == 1 and i_a[i] == 2 and i_b[i] == 1:
        #                 i_r[i] = p[i][9] * result[i][11] / (
        #                             i_noise_in_sbs_2[i] + i_noise_in_sbs_2_channel_1[i] + i_noise_out_sbs2_channel_1[
        #                         i] + self.noise)
        #             if i_xx[i] == 1 and i_a[i] == 2 and i_b[i] == 2:
        #                 i_r[i] = p[i][10] * result[i][12] / (
        #                             i_noise_in_sbs_2[i] + i_noise_in_sbs_2_channel_2[i] + i_noise_out_sbs2_channel_2[
        #                         i] + self.noise)
        #             if i_xx[i] == 1 and i_a[i] == 2 and i_b[i] == 3:
        #                 i_r[i] = p[i][11] * result[i][13] / (
        #                             i_noise_in_sbs_2[i] + i_noise_in_sbs_2_channel_3[i] + i_noise_out_sbs2_channel_3[
        #                         i] + self.noise)
        #
        #         for i in range(8):
        #             i_R[i] = self.w * np.log2(1 + i_r[i])
        #
        #         "优化每个用户本地的计算资源"
        #         i_f_local = []
        #         for i in range(8):
        #             i_local = result[i][1] / self.d_max
        #             i_f_local.append(i_local)
        #
        #         "计算本地处理时间和能耗"
        #         i_e_local = [0 for i in range(8)]
        #         for i in range(8):
        #             if i_xx[i] == 0:
        #                 i_e_local[i] = self.k * (i_f_local[i] ** 2) * result[i][1]
        #
        #         "计算上传和处理时间"
        #         i_t_tran = [0 for i in range(8)]
        #         i_t_mec = [0 for i in range(8)]
        #         i_e_mec = [0 for i in range(8)]
        #         for i in range(8):
        #             if i_xx[i] == 1 and i_a[i] == 0 and i_b[i] == 0:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][0] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_m_mec
        #             if i_xx[i] == 1 and i_a[i] == 0 and i_b[i] == 1:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][1] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_m_mec
        #             if i_xx[i] == 1 and i_a[i] == 0 and i_b[i] == 2:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][2] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_m_mec
        #             if i_xx[i] == 1 and i_a[i] == 0 and i_b[i] == 3:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][3] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_m_mec
        #             if i_xx[i] == 1 and i_a[i] == 1 and i_b[i] == 0:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][4] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_s_mec
        #             if i_xx[i] == 1 and i_a[i] == 1 and i_b[i] == 1:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][5] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_s_mec
        #             if i_xx[i] == 1 and i_a[i] == 1 and i_b[i] == 2:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][6] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_s_mec
        #             if i_xx[i] == 1 and i_a[i] == 1 and i_b[i] == 3:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][7] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_s_mec
        #             if i_xx[i] == 1 and i_a[i] == 2 and i_b[i] == 0:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][8] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_s_mec
        #             if i_xx[i] == 1 and i_a[i] == 2 and i_b[i] == 1:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][9] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_s_mec
        #             if i_xx[i] == 1 and i_a[i] == 2 and i_b[i] == 2:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][10] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_s_mec
        #             if i_xx[i] == 1 and i_a[i] == 2 and i_b[i] == 3:
        #                 i_t_tran[i] = result[i][0] / i_R[i]
        #                 i_e_mec[i] = p[i][11] * i_t_tran[i]
        #                 i_t_mec[i] = result[i][1] / self.f_s_mec
        #
        #         "计算所有用户的奖励"
        #         i_reward = [0 for i in range(8)]
        #         "加入最低传输速率的限制,小于这个速率则看为传输失败"
        #         i_R_min = [0 for i in range(8)]
        #         max_reward = []
        #         for i in range(8):
        #             i_R_min[i] = result[i][0] / (self.d_max - i_t_mec[i])
        #
        #         for i in range(8):
        #             if i_xx[i] == 1 and i_R[i] < i_R_min[i]:
        #                 i_reward[i] = -0.02
        #             else:
        #                 i_reward[i] = -(i_e_local[i] + i_e_mec[i])
        #
        #         max_reward.append(sum(i_reward))
        #     "每个用户对应动作最大的奖励"
        #     iter_max=max(max_reward)
        #     "组合每个用户的最大zhi"
        #     user_reward.append(iter_max)



        # "计算全卸载的能耗"
        # a_random=[0 for i in range(8)]     #随机基站的选择
        # b_random=[0 for i in range(8)]     #随机信道的选择
        # for i in range(8):
        #     action_random=np.random.randint(12)
        #     a_random[i]=action_random%3                      #连接的基站数目
        #     b_random[i]=(action_random//3)%4                 #选择的信道数目
        #
        # "计算在MBS卸载时，小小区复用信道产生的干扰"
        #
        # mbs_0_random=[x for x,y in list(enumerate(a_random)) if y==0 ] #在宏基站卸载的用户索引
        # sbs_1_random = [x for x, y in list(enumerate(a_random)) if y == 1 ]  # 在小区基站1卸载的用户索引
        # sbs_2_random = [x for x, y in list(enumerate(a_random)) if y == 2 ]  # 在小区基站2卸载的用户索引
        # channel_0_random = [x for x, y in list(enumerate(b_random)) if y == 0  and a_random[x]!=0]  # 小小区基站选择信道0的用户索引
        # channel_1_random = [x for x, y in list(enumerate(b_random)) if y == 1 and a_random[x]!=0]  # 小小区基站选择信道1的用户索引
        # channel_2_random = [x for x, y in list(enumerate(b_random)) if y == 2 and a_random[x]!=0]   # 小小区基站选择信道2的用户索引
        # channel_3_random = [x for x, y in list(enumerate(b_random)) if y == 3 and a_random[x]!=0]  # 小小区基站选择信道3的用户索引
        #
        # noise_in_mbs_channel_0_random=[0 for i in range(8)]
        # noise_in_mbs_channel_1_random=[0 for i in range(8)]
        # noise_in_mbs_channel_2_random=[0 for i in range(8)]
        # noise_in_mbs_channel_3_random=[0 for i in range(8)]
        #
        # "计算在mbs随机卸载使用信道的用户，其它小小区使用相同信道的干扰"
        # for i in range(0,len(mbs_0_random)):
        #     for j in range(0,len(channel_0_random),1):
        #         if channel_0_random[j] in sbs_1_random:
        #             noise_in_mbs_channel_0_random[channel_0_random[j]]\
        #                 +=p[channel_0_random[j]][b_random[channel_0_random[j]]]*result[channel_0_random[j]][2+b_random[channel_0_random[j]]]
        #
        #         if channel_0_random[j] in sbs_2_random:
        #             noise_in_mbs_channel_0_random[channel_0_random[j]] \
        #                 +=p[channel_0_random[j]][b_random[channel_0_random[j]]]*result[channel_0_random[j]][2+b_random[channel_0_random[j]]]
        #
        #     for j in range(0,len(channel_1_random),1):
        #         if channel_1_random[j] in sbs_1_random:
        #             noise_in_mbs_channel_1_random[channel_1_random[j]]\
        #                 +=p[channel_1_random[j]][b_random[channel_1_random[j]]]*result[channel_1_random[j]][2+b_random[channel_1_random[j]]]
        #
        #         if channel_1_random[j] in sbs_2_random:
        #             noise_in_mbs_channel_1_random[channel_1_random[j]]\
        #                 +=p[channel_1_random[j]][b_random[channel_1_random[j]]]*result[channel_1_random[j]][2+b_random[channel_1_random[j]]]
        #
        #     for j in range(0,len(channel_2_random),1):
        #         if channel_2_random[j] in sbs_1_random:
        #             noise_in_mbs_channel_2_random[channel_2_random[j]]\
        #                 +=p[channel_2_random[j]][b_random[channel_2_random[j]]]*result[channel_2_random[j]][2+b_random[channel_2_random[j]]]
        #
        #         if channel_2_random[j] in sbs_2_random:
        #             noise_in_mbs_channel_2_random[channel_2_random[j]] \
        #                 +=p[channel_2_random[j]][b_random[channel_2_random[j]]]*result[channel_2_random[j]][2+b_random[channel_2_random[j]]]
        #
        #     for j in range(0,len(channel_3_random),1):
        #         if channel_3_random[j] in sbs_1_random:
        #             noise_in_mbs_channel_3_random[channel_3_random[j]]\
        #                 +=p[channel_3_random[j]][b_random[channel_3_random[j]]]*result[channel_3_random[j]][2+b_random[channel_3_random[j]]]
        #
        #         if channel_3_random[j] in sbs_2_random:
        #             noise_in_mbs_channel_3_random[channel_3_random[j]] \
        #                 += p[channel_3_random[j]][b_random[channel_3_random[j]]]*result[channel_3_random[j]][2+b_random[channel_3_random[j]]]
        #
        # "随机卸载同基站同信道干扰"
        # "SBS_1"
        # noise_in_sbs_1_random = [0 for i in range(8)]
        # G_0_random = []
        # G_1_random = []
        # G_2_random = []
        # G_3_random = []
        # sbs_11_random = [x for x, y in list(enumerate(a_random)) if y == 1 ]  # 在小区基站1卸载的用户索引
        #
        # offload_in_sbs1_channel_0_random = []
        # offload_in_sbs1_channel_1_random = []
        # offload_in_sbs1_channel_2_random = []
        # offload_in_sbs1_channel_3_random = []
        #
        # "得到每个卸载到sbs1的用户的信道选择"
        # for i in range(0, len(sbs_11_random), 1):
        #     if b[sbs_11_random[i]] == 0:
        #         offload_in_sbs1_channel_0_random.append(sbs_11_random[i])
        #     if b[sbs_11_random[i]] == 1:
        #         offload_in_sbs1_channel_1_random.append(sbs_11_random[i])
        #     if b[sbs_11_random[i]] == 2:
        #         offload_in_sbs1_channel_2_random.append(sbs_11_random[i])
        #     if b[sbs_11_random[i]] == 3:
        #         offload_in_sbs1_channel_3_random.append(sbs_11_random[i])
        #
        # "得到0信道用户对应的信道增益"
        # for i in range(0, len(offload_in_sbs1_channel_0_random), 1):
        #     a_0_random = [offload_in_sbs1_channel_0_random[i], result[offload_in_sbs1_channel_0_random[i]][6]]
        #     G_0_random.append(a_0_random)
        #
        # "得到1信道用户对应的信道增益"
        # for i in range(0, len(offload_in_sbs1_channel_1_random), 1):
        #     a_1_random = [offload_in_sbs1_channel_1_random[i], result[offload_in_sbs1_channel_1_random[i]][7]]
        #     G_1_random.append(a_1_random)
        #
        # "得到2信道用户对应的信道增益"
        # for i in range(0, len(offload_in_sbs1_channel_2_random), 1):
        #     a_2_random = [offload_in_sbs1_channel_2_random[i], result[offload_in_sbs1_channel_2_random[i]][8]]
        #     G_2_random.append(a_2_random)
        #
        # "得到3信道用户对应的信道增益"
        # for i in range(0, len(offload_in_sbs1_channel_3_random), 1):
        #     a_3_random = [offload_in_sbs1_channel_3_random[i], result[offload_in_sbs1_channel_3_random[i]][9]]
        #     G_3_random.append(a_3_random)
        #
        # G_0_random = np.array(G_0_random)
        # "对数组进行降序排序"
        # if len(G_0_random) <= 1:
        #     G_0_random = np.array(G_0_random)
        # else:
        #     G_0_random = G_0_random[np.argsort(-G_0_random[:, 1])]
        #
        # G_2_random = np.array(G_2_random)
        # "对数组进行降序排序"
        # if len(G_2_random) <= 1:
        #     G_2_random = np.array(G_2_random)
        # else:
        #     G_2_random = G_2_random[np.argsort(-G_2_random[:, 1])]
        #
        # G_1_random = np.array(G_1_random)
        # "对数组进行降序排序"
        # if len(G_1_random) <= 1:
        #     G_1_random = np.array(G_1_random)
        # else:
        #     G_1_random = G_1_random[np.argsort(-G_1_random[:, 1])]
        #
        # G_3_random = np.array(G_3_random)
        # "对数组进行降序排序"
        # if len(G_3_random) <= 1:
        #     G_3_random = np.array(G_3_random)
        # else:
        #     G_3_random = G_3_random[np.argsort(-G_3_random[:, 1])]
        #
        # while len(G_0_random) > 1:
        #     for j in range(-1, -(len(G_0_random[:, 0])), -1):  # 不包含第一个元素
        #         noise_in_sbs_1_random[int(G_0_random[j][0])] += p[int(G_0_random[j][0])][4] * result[int(G_0_random[j][0])][6]
        #     G_0_random = np.delete(G_0_random, 0, axis=0)  # 删除第一个用户
        #
        # while len(G_1_random) > 1:
        #     for j in range(-1, -(len(G_1_random[:, 0])), -1):  # 不包含第一个元素
        #         noise_in_sbs_1_random[int(G_1_random[j][0])] += p[int(G_1_random[j][0])][5] * result[int(G_1_random[j][0])][7]
        #     G_1_random = np.delete(G_1_random, 0, axis=0)  # 删除第一个用户
        #
        # while len(G_2_random) > 1:
        #     for j in range(-1, -(len(G_2_random[:, 0])), -1):  # 不包含第一个元素
        #         noise_in_sbs_1_random[int(G_2_random[j][0])] += p[int(G_2_random[j][0])][6] * result[int(G_2_random[j][0])][8]
        #     G_2_random = np.delete(G_2_random, 0, axis=0)  # 删除第一个用户
        #
        # while len(G_3_random) > 1:
        #     for j in range(-1, -(len(G_3_random[:, 0])), -1):  # 不包含第一个元素
        #         noise_in_sbs_1_random[int(G_3_random[j][0])] += p[int(G_3_random[j][0])][7]*result[int(G_3_random[j][0])][9]
        #     G_3_random = np.delete(G_3_random, 0, axis=0)  # 删除第一个用户
        #
        # "SBS_2"
        # noise_in_sbs_2_random = [0 for i in range(8)]
        # G_00_random = []
        # G_11_random = []
        # G_22_random = []
        # G_33_random = []
        # sbs_22_random = [x for x, y in list(enumerate(a_random)) if y == 2 ]  # 在小区基站1卸载的用户索引
        #
        # offload_in_sbs2_channel_0_random = []
        # offload_in_sbs2_channel_1_random = []
        # offload_in_sbs2_channel_2_random = []
        # offload_in_sbs2_channel_3_random = []
        #
        # "得到每个卸载到sbs2的用户的信道选择"
        # for i in range(0, len(sbs_22_random), 1):
        #     if b[sbs_22_random[i]] == 0:
        #         offload_in_sbs2_channel_0_random.append(sbs_22_random[i])
        #     if b[sbs_22_random[i]] == 1:
        #         offload_in_sbs2_channel_1_random.append(sbs_22_random[i])
        #     if b[sbs_22_random[i]] == 2:
        #         offload_in_sbs2_channel_2_random.append(sbs_22_random[i])
        #     if b[sbs_22_random[i]] == 3:
        #         offload_in_sbs2_channel_3_random.append(sbs_22_random[i])
        #
        # "得到0信道用户对应的信道增益"
        # for i in range(0, len(offload_in_sbs2_channel_0_random), 1):
        #     a_00_random = [offload_in_sbs2_channel_0_random[i], result[offload_in_sbs2_channel_0_random[i]][10]]
        #     G_00_random.append(a_00_random)
        #
        # "得到1信道用户对应的信道增益"
        # for i in range(0, len(offload_in_sbs2_channel_1_random), 1):
        #     a_11_random = [offload_in_sbs2_channel_1_random[i], result[offload_in_sbs2_channel_1_random[i]][11]]
        #     G_11_random.append(a_11_random)
        #
        # "得到2信道用户对应的信道增益"
        # for i in range(0, len(offload_in_sbs2_channel_2_random), 1):
        #     a_22_random = [offload_in_sbs2_channel_2_random[i], result[offload_in_sbs2_channel_2_random[i]][12]]
        #     G_22_random.append(a_22_random)
        #
        # "得到3信道用户对应的信道增益"
        # for i in range(0, len(offload_in_sbs2_channel_3_random), 1):
        #     a_33_random = [offload_in_sbs2_channel_3_random[i], result[offload_in_sbs2_channel_3_random[i]][13]]
        #     G_33_random.append(a_33_random)
        #
        # G_00_random = np.array(G_00_random)
        # "对数组进行降序排序"
        # if len(G_00_random) <= 1:
        #     G_00_random = np.array(G_00_random)
        # else:
        #     G_00_random = G_00_random[np.argsort(-G_00_random[:, 1])]
        #
        # G_22_random = np.array(G_22_random)
        # "对数组进行降序排序"
        # if len(G_22_random) <= 1:
        #     G_22_random = np.array(G_22_random)
        # else:
        #     G_22_random = G_22_random[np.argsort(-G_22_random[:, 1])]
        #
        # G_11_random = np.array(G_11_random)
        # "对数组进行降序排序"
        # if len(G_11_random) <= 1:
        #     G_11_random = np.array(G_11_random)
        # else:
        #     G_11_random = G_11_random[np.argsort(-G_11_random[:, 1])]
        #
        # G_33_random = np.array(G_33_random)
        # "对数组进行降序排序"
        # if len(G_33_random) <= 1:
        #     G_33_random = np.array(G_33_random)
        # else:
        #     G_33_random = G_33_random[np.argsort(-G_33_random[:, 1])]
        #
        # while len(G_00_random) > 1:
        #     for j in range(-1, -(len(G_00_random[:, 0])), -1):  # 不包含第一个元素
        #         noise_in_sbs_2_random[int(G_00_random[j][0])] += p[int(G_00_random[j][0])][8]*result[int(G_00_random[j][0])][10]
        #     G_00_random = np.delete(G_00_random, 0, axis=0)  # 删除第一个用户
        #
        # while len(G_11_random) > 1:
        #     for j in range(-1, -(len(G_11_random[:, 0])), -1):  # 不包含第一个元素
        #         noise_in_sbs_2_random[int(G_11_random[j][0])] += p[int(G_11_random[j][0])][9]*result[int(G_11_random[j][0])][11]
        #     G_11_random = np.delete(G_11_random, 0, axis=0)  # 删除第一个用户
        #
        # while len(G_22_random) > 1:
        #     for j in range(-1, -(len(G_22_random[:, 0])), -1):  # 不包含第一个元素
        #         noise_in_sbs_2_random[int(G_22_random[j][0])] += p[int(G_22_random[j][0])][10]*result[int(G_22_random[j][0])][12]
        #     G_22_random = np.delete(G_22_random, 0, axis=0)  # 删除第一个用户
        #
        # while len(G_33_random) > 1:
        #     for j in range(-1, -(len(G_33_random[:, 0])), -1):  # 不包含第一个元素
        #         noise_in_sbs_2_random[int(G_33_random[j][0])] += p[int(G_33_random[j][0])][11]*result[int(G_33_random[j][0])][13]
        #     G_33_random = np.delete(G_33_random, 0, axis=0)  # 删除第一个用户
        #
        # "随机卸载到小小区时，复用宏基站信道的跨层干扰"
        # "sbs1"
        # noise_in_sbs_1_channel_0_random = [0 for i in range(8)]
        # noise_in_sbs_1_channel_1_random = [0 for i in range(8)]
        # noise_in_sbs_1_channel_2_random = [0 for i in range(8)]
        # noise_in_sbs_1_channel_3_random = [0 for i in range(8)]
        #
        # for i in range(0, len(offload_in_sbs1_channel_0_random), 1):
        #     noise_in_sbs_1_channel_0_random[offload_in_sbs1_channel_0_random[i]] =\
        #         p[offload_in_sbs1_channel_0_random[i]][0] \
        #                                                              * result[offload_in_sbs1_channel_0_random[i]][6]
        # for i in range(0, len(offload_in_sbs1_channel_1_random), 1):
        #     noise_in_sbs_1_channel_1_random[offload_in_sbs1_channel_1_random[i]] = p[offload_in_sbs1_channel_1_random[i]][1] \
        #                                                              * result[offload_in_sbs1_channel_1_random[i]][7]
        #
        # for i in range(0, len(offload_in_sbs1_channel_2_random), 1):
        #     noise_in_sbs_1_channel_2_random[offload_in_sbs1_channel_2_random[i]] = p[offload_in_sbs1_channel_2_random[i]][2] \
        #                                                              * result[offload_in_sbs1_channel_2_random[i]][8]
        # for i in range(0, len(offload_in_sbs1_channel_3_random), 1):
        #     noise_in_sbs_1_channel_3_random[offload_in_sbs1_channel_3_random[i]] = p[offload_in_sbs1_channel_3_random[i]][3] \
        #                                                              * result[offload_in_sbs1_channel_3_random[i]][9]
        #
        # "sbs2"
        # noise_in_sbs_2_channel_0_random = [0 for i in range(8)]
        # noise_in_sbs_2_channel_1_random = [0 for i in range(8)]
        # noise_in_sbs_2_channel_2_random = [0 for i in range(8)]
        # noise_in_sbs_2_channel_3_random = [0 for i in range(8)]
        #
        # for i in range(0, len(offload_in_sbs2_channel_0_random), 1):
        #     noise_in_sbs_2_channel_0_random[offload_in_sbs2_channel_0_random[i]] = p[offload_in_sbs2_channel_0_random[i]][4] \
        #                                                              * result[offload_in_sbs2_channel_0_random[i]][10]
        # for i in range(0, len(offload_in_sbs2_channel_1_random), 1):
        #     noise_in_sbs_2_channel_1_random[offload_in_sbs2_channel_1_random[i]] = p[offload_in_sbs2_channel_1_random[i]][5] \
        #                                                              * result[offload_in_sbs2_channel_1_random[i]][11]
        # for i in range(0, len(offload_in_sbs2_channel_2_random), 1):
        #     noise_in_sbs_2_channel_2_random[offload_in_sbs2_channel_2_random[i]] = p[offload_in_sbs2_channel_2_random[i]][6] \
        #                                                              * result[offload_in_sbs2_channel_2_random[i]][12]
        # for i in range(0, len(offload_in_sbs2_channel_3_random), 1):
        #     noise_in_sbs_2_channel_3_random[offload_in_sbs2_channel_3_random[i]] = p[offload_in_sbs2_channel_3_random[i]][7] \
        #                                                              * result[offload_in_sbs2_channel_3_random[i]][13]
        #
        # "计算随机卸载不同基站，相同信道的干扰"
        # "sbs1"
        # noise_out_sbs1_channel_0_random =[0 for i in range(8)]
        # noise_out_sbs1_channel_1_random = [0 for i in range(8)]
        # noise_out_sbs1_channel_2_random = [0 for i in range(8)]
        # noise_out_sbs1_channel_3_random = [0 for i in range(8)]
        #
        # for i in range(0,len(offload_in_sbs2_channel_0_random),):
        #     noise_out_sbs1_channel_0_random[offload_in_sbs2_channel_0_random[i]]+=p[offload_in_sbs2_channel_0_random[i]][4]\
        #                                                             *result[offload_in_sbs2_channel_0_random[i]][6]
        # for i in range(0,len(offload_in_sbs2_channel_1_random),):
        #     noise_out_sbs1_channel_1_random[offload_in_sbs2_channel_1_random[i]]+=p[offload_in_sbs2_channel_1_random[i]][5]\
        #                                                             *result[offload_in_sbs2_channel_1_random[i]][7]
        # for i in range(0,len(offload_in_sbs2_channel_2_random),):
        #     noise_out_sbs1_channel_2_random[offload_in_sbs2_channel_2_random[i]]+=p[offload_in_sbs2_channel_2_random[i]][6]\
        #                                                             *result[offload_in_sbs2_channel_2_random[i]][8]
        # for i in range(0,len(offload_in_sbs2_channel_3_random),):
        #     noise_out_sbs1_channel_3_random[offload_in_sbs2_channel_3_random[i]]+=p[offload_in_sbs2_channel_3_random[i]][7]\
        #                                                             *result[offload_in_sbs2_channel_3_random[i]][9]
        #
        # "sbs2"
        # noise_out_sbs2_channel_0_random = [0 for i in range(8)]
        # noise_out_sbs2_channel_1_random = [0 for i in range(8)]
        # noise_out_sbs2_channel_2_random = [0 for i in range(8)]
        # noise_out_sbs2_channel_3_random = [0 for i in range(8)]
        #
        # for i in range(0,len(offload_in_sbs1_channel_0_random),):
        #     noise_out_sbs2_channel_0_random[offload_in_sbs1_channel_0_random[i]]+=p[offload_in_sbs1_channel_0_random[i]][0]\
        #                                                             *result[offload_in_sbs1_channel_0_random[i]][10]
        # for i in range(0,len(offload_in_sbs1_channel_1_random),):
        #     noise_out_sbs2_channel_1_random[offload_in_sbs1_channel_1_random[i]]+=p[offload_in_sbs1_channel_1_random[i]][1]\
        #                                                             *result[offload_in_sbs1_channel_1_random[i]][11]
        # for i in range(0,len(offload_in_sbs1_channel_2_random),):
        #     noise_out_sbs2_channel_2_random[offload_in_sbs1_channel_2_random[i]]+=p[offload_in_sbs1_channel_2_random[i]][2]\
        #                                                             *result[offload_in_sbs1_channel_2_random[i]][12]
        # for i in range(0,len(offload_in_sbs1_channel_3_random),):
        #     noise_out_sbs2_channel_3_random[offload_in_sbs1_channel_3_random[i]]+=p[offload_in_sbs1_channel_3_random[i]][3]\
        #                                                             *result[offload_in_sbs1_channel_3_random[i]][13]
        #
        # "计算每个用户的上传速率"
        # R_random=[0 for i in range(8)]
        # r_random=[0 for i in range(8)]
        #
        # for i in range(8):
        #     if  a_random[i]==0 and b_random[i]==0:
        #         r_random[i]=p[i][0]*result[i][2]/(noise_in_mbs_channel_0_random[i]+self.noise)
        #     if  a_random[i]==0 and b_random[i]==1:
        #         r_random[i]=p[i][1]*result[i][3]/(noise_in_mbs_channel_1_random[i]+self.noise)
        #     if  a_random[i]==0 and b_random[i]==2:
        #         r_random[i]=p[i][2]*result[i][4]/(noise_in_mbs_channel_2_random[i]+self.noise)
        #     if  a_random[i]==0 and b_random[i]==3:
        #         r_random[i]=p[i][3]*result[i][5]/(noise_in_mbs_channel_3_random[i]+self.noise)
        #     if a_random[i]==1 and b_random[i]==0:
        #         r_random[i]=p[i][4]*result[i][6]\
        #                     /(noise_in_sbs_1_random[i]+noise_in_sbs_1_channel_0_random[i]+noise_out_sbs1_channel_0_random[i]+self.noise)
        #     if  a_random[i]==1 and b_random[i]==1:
        #         r_random[i]=p[i][5]*result[i][7]\
        #                     /(noise_in_sbs_1_random[i]+noise_in_sbs_1_channel_1_random[i]+noise_out_sbs1_channel_1_random[i]+self.noise)
        #     if  a_random[i]==1 and b_random[i]==2:
        #         r_random[i]=p[i][6]*result[i][8]\
        #                     /(noise_in_sbs_1_random[i]+noise_in_sbs_1_channel_2_random[i]+noise_out_sbs1_channel_2_random[i]+self.noise)
        #     if  a_random[i]==1 and b_random[i]==3:
        #         r_random[i]=p[i][7]*result[i][9]\
        #                     /(noise_in_sbs_1_random[i]+noise_in_sbs_1_channel_3_random[i]+noise_out_sbs1_channel_3_random[i]+self.noise)
        #     if  a_random[i]==2 and b_random[i]==0:
        #         r_random[i]=p[i][8]*result[i][10]\
        #                     /(noise_in_sbs_2_random[i]+noise_in_sbs_2_channel_0_random[i]+noise_out_sbs2_channel_0_random[i]+self.noise)
        #
        #     if  a_random[i]==2 and b_random[i]==1:
        #         r_random[i]=p[i][9]*result[i][11]/(noise_in_sbs_2_random[i]+noise_in_sbs_2_channel_1_random[i]+noise_out_sbs2_channel_1_random[i]+self.noise)
        #
        #
        #     if  a_random[i]==2 and b_random[i]==2:
        #         r_random[i]=p[i][10]*result[i][12]\
        #              /(noise_in_sbs_2_random[i]+noise_in_sbs_2_channel_2_random[i]+noise_out_sbs2_channel_2_random[i]+self.noise)
        #     if  a_random[i]==2 and b_random[i]==3:
        #         r_random[i]=p[i][11]*result[i][13]\
        #              /(noise_in_sbs_2_random[i]+noise_in_sbs_2_channel_3_random[i]+noise_out_sbs2_channel_3_random[i]+self.noise)
        #
        # for i in range(8):
        #     R_random[i]=self.w*np.log2(1+r_random[i])
        #
        # "计算上传和处理时间"
        # t_tran_random= [0 for i in range(8)]
        # t_mec_random = [0 for i in range(8)]
        # e_mec_random = [0 for i in range(8)]
        # for i in range(8):
        #     if  a_random[i] == 0 and b_random[i] == 0:
        #         t_tran_random[i]=result[i][0]/R_random[i]
        #         e_mec_random[i]=p[i][0]*t_tran_random[i]
        #         t_mec_random[i]=result[i][1]/self.f_m_mec
        #     if  a_random[i] == 0 and b_random[i] == 1:
        #         t_tran_random[i] = result[i][0] / R_random[i]
        #         e_mec_random[i] = p[i][1] * t_tran_random[i]
        #         t_mec_random[i] = result[i][1] / self.f_m_mec
        #     if  a_random[i]==0 and b_random[i] == 2:
        #         t_tran_random[i]=result[i][0]/R_random[i]
        #         e_mec_random[i]=p[i][2]*t_tran_random[i]
        #         t_mec_random[i] = result[i][1] / self.f_m_mec
        #     if  a_random[i]==0 and b_random[i]==3:
        #         t_tran_random[i]=result[i][0]/R_random[i]
        #         e_mec_random[i]=p[i][3]*t_tran_random[i]
        #         t_mec_random[i] = result[i][1] / self.f_m_mec
        #     if  a_random[i]==1 and b_random[i]==0:
        #         t_tran_random[i]=result[i][0]/R_random[i]
        #         e_mec_random[i]=p[i][4]*t_tran_random[i]
        #         t_mec_random[i] = result[i][1] / self.f_s_mec
        #     if  a_random[i]==1 and b_random[i]==1:
        #         t_tran_random[i]=result[i][0]/R_random[i]
        #         e_mec_random[i]=p[i][5]*t_tran_random[i]
        #         t_mec_random[i] = result[i][1] / self.f_s_mec
        #     if  a_random[i]==1 and b_random[i]==2:
        #         t_tran_random[i]=result[i][0]/R_random[i]
        #         e_mec_random[i]=p[i][6]*t_tran_random[i]
        #         t_mec_random[i] = result[i][1] / self.f_s_mec
        #     if  a_random[i]==1 and b_random[i]==3:
        #         t_tran_random[i]=result[i][0]/R_random[i]
        #         e_mec_random[i]=p[i][7]*t_tran_random[i]
        #         t_mec_random[i] = result[i][1] / self.f_s_mec
        #     if  a_random[i]==2 and b_random[i]==0:
        #         t_tran_random[i]=result[i][0]/R_random[i]
        #         e_mec_random[i]=p[i][8]*t_tran_random[i]
        #         t_mec_random[i] = result[i][1] / self.f_s_mec
        #     if  a_random[i]==2 and b_random[i]==1:
        #         t_tran_random[i]=result[i][0]/R_random[i]
        #         e_mec_random[i]=p[i][9]*t_tran_random[i]
        #         t_mec_random[i] = result[i][1] / self.f_s_mec
        #     if  a_random[i]==2 and b_random[i]==2:
        #         t_tran_random[i]=result[i][0]/R_random[i]
        #         e_mec_random[i]=p[i][10]*t_tran_random[i]
        #         t_mec_random[i] = result[i][1] / self.f_s_mec
        #     if  a_random[i]==2 and b_random[i]==3:
        #         t_tran_random[i]=result[i][0]/R_random[i]
        #         e_mec_random[i]=p[i][11]*t_tran_random[i]
        #         t_mec_random[i] = result[i][1] / self.f_s_mec
        #
        # e_mec_random = [0 for i in range(8)]
        # "计算所有用户的奖励"
        # reward_random=[0 for i in range(8)]
        # "加入最低传输速率的限制,小于这个速率则看为传输失败"
        # R_min_random=[0 for i in range(8)]
        # for i in range(8):
        #     R_min_random[i]=result[i][0]/(self.d_max-t_mec_random[i])
        #
        # for i in range(8):
        #     if  R_random[i]<R_min_random[i]:
        #         reward_random[i]=-0.02
        #     else:
        #         reward_random[i]=-e_mec_random[i]


        next_obsvation=[]
        for i in range(8):

            A_in=np.random.uniform(100*1024,200*1024)
            A_com=np.random.uniform(100*1024*150,200*1024*150)
            "用户坐标"
            d_x=np.random.uniform(1,500)
            d_y=np.random.uniform(1,500)
            "生成第一个基站内四条信道的信道增益"
            g_00=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-250)**2)**0.5)**(-self.pathloss))
            g_01=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-250)**2)**0.5)**(-self.pathloss))
            g_02=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-250)**2)**0.5)**(-self.pathloss))
            g_03=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-250)**2)**0.5)**(-self.pathloss))
            "生成第二个基站内四条信道的信道增益"
            g_10=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-375)**2)**0.5)** (-self.pathloss))
            g_11=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-375)**2)**0.5)** (-self.pathloss))
            g_12=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-375)**2)**0.5)** (-self.pathloss))
            g_13=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-375)**2)**0.5)** (-self.pathloss))
            "生成第三个基站内四条信道的信道增益"
            g_20=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-125)**2)**0.5)**(-self.pathloss))
            g_21=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-125)**2)**0.5)**(-self.pathloss))
            g_22=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-125)**2)**0.5)**(-self.pathloss))
            g_23=float( ((np.random.normal(loc=0,scale=1,size=1))**2)*(((d_x-250)**2+(d_y-125)**2)**0.5)**(-self.pathloss))

            next_obsvation.append(A_in)
            next_obsvation.append(A_com)
            next_obsvation.append(g_00)
            next_obsvation.append(g_01)
            next_obsvation.append(g_02)
            next_obsvation.append(g_03)

            next_obsvation.append(g_10)
            next_obsvation.append(g_11)
            next_obsvation.append(g_12)
            next_obsvation.append(g_13)

            next_obsvation.append(g_20)
            next_obsvation.append(g_21)
            next_obsvation.append(g_22)
            next_obsvation.append(g_23)
        next_obsvation=np.array(next_obsvation).reshape((8,14))

        "对数组进行归一化处理"

        "找到每一列的最大值和最小值"
        max_max=[]
        min_min=[]
        for i in range(14):
            max_number=[]
            for j in range(8):
                number=next_obsvation[j][i]
                max_number.append(number)
            value_max=max(max_number)
            value_min=min(max_number)
            max_max.append(value_max)
            min_min.append(value_min)

        next_obsvation_norm=np.array([0.5 for i in range(112)]).reshape((8,14))

        for i in range(8):
            for j in range(14):
                next_obsvation_norm[i][j]=(next_obsvation[i][j]-min_min[j])/(max_max[j]-min_min[j])

        # return reward,next_obsvation,next_obsvation_norm,reward_random,fitness_best,user_reward
        return reward, next_obsvation, next_obsvation_norm


# env=Environment()
# #
# # print(env.reset())
# obs=[1.10891204e+05, 2.40098027e+07, 1.52048910e-09, 7.82256237e-10,
#         3.85177787e-10, 3.90133109e-09, 2.41271248e-11, 1.27306827e-10,
#         7.32697675e-11, 2.04319855e-10, 1.72652818e-12, 8.23775762e-09,
#         1.67044699e-10, 2.12657587e-10,
#        1.03441230e+05, 2.60463349e+07, 3.92815649e-09, 3.48103935e-09,
#         1.51680337e-10, 3.68850867e-10, 3.60209011e-12, 4.91612347e-11,
#         2.63194056e-10, 1.93356360e-09, 2.24383345e-08, 1.01774425e-09,
#         1.66751829e-08, 1.42873831e-08,
#        1.96287849e+05, 1.69551358e+07, 1.90908891e-10, 1.83611135e-08,
#         5.97044837e-09, 4.30896050e-09, 6.79627183e-09, 9.66763787e-08,
#         1.42176030e-09, 4.98714620e-09, 3.03431155e-09, 4.16303845e-10,
#         8.03938146e-10, 3.61540797e-10,
#        1.94999001e+05, 1.92147841e+07, 2.21419518e-09, 4.92338454e-11,
#         2.18710403e-14, 4.50739680e-10, 9.47184471e-11, 4.56365899e-12,
#         1.21379419e-10, 5.17602238e-10, 5.90169702e-10, 3.69244751e-08,
#         4.04246097e-09, 1.20275684e-08,
#        1.53898452e+05, 1.95384485e+07, 2.52348345e-08, 1.73908588e-08,
#         1.46946138e-07, 8.12109387e-10, 2.47612143e-09, 4.08802698e-11,
#         2.50994200e-09, 1.31542712e-08, 2.43176777e-10, 2.32605827e-10,
#         7.50975070e-14, 1.45692556e-08,
#        1.36447967e+05, 3.00023110e+07, 6.56886513e-10, 8.31301657e-09,
#         5.33498607e-11, 2.43978982e-11, 4.30702590e-11, 1.44905381e-11,
#         8.28060631e-11, 7.44154392e-11, 6.49384053e-08, 1.53144968e-08,
#         1.40398265e-07, 1.35507483e-08,
#        1.26845415e+05, 1.53808203e+07, 6.98603451e-10, 4.58049756e-09,
#         3.97752034e-09, 1.36691180e-09, 8.97886116e-10, 2.20574880e-09,
#         8.39872158e-10, 3.05226377e-10, 1.36979886e-09, 1.86726442e-09,
#         1.40228912e-08, 2.36675836e-10,
#        1.52826172e+05, 1.96186305e+07, 5.47470073e-11, 1.77371956e-11,
#         1.05361588e-10, 2.24644003e-10, 2.67434944e-10, 1.48747911e-10,
#         2.61966386e-10, 7.19315182e-11, 5.31882482e-09, 2.00508370e-12,
#         9.92691455e-10, 5.35786392e-10]
#
# action=[7 for i in range(8)]
# for i in range(8):
#     action[i]=np.random.randint(0,12)
#
# print(env.step(action,obs,100))

