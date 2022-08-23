from collections import Counter
import random
from visualdl import LogWriter
import numpy as np
import collections
from collections import deque
import matplotlib.pyplot as plt

numbers=("0.5s","0.4s","0.3s")
a=[0.044,0.063,0.078]
b=[0.11,0.16,0.21]
c=[0.029,0.059,0.072]
d=[0.18,0.21,0.24]
e=[0.078,0.095,0.12]

bar_width=0.15
index_a=np.arange(len(numbers))
index_b=index_a+bar_width
index_c=index_b+bar_width
index_d=index_c+bar_width
index_e=index_d+bar_width

plt.bar(index_a,height=a,width=bar_width,label='Optimization',color='b')
plt.bar(index_b,height=b,width=bar_width,label='Particle swarm',color='g')
plt.bar(index_c,height=c,width=bar_width,label='Iteration',color='purple')
plt.bar(index_d,height=d,width=bar_width,label='Full local',color='teal')
plt.bar(index_e,height=e,width=bar_width,label='Full offloading',color='peru')

plt.legend()
plt.xticks(index_a+bar_width/2,numbers)
plt.ylabel("Total energy(J)")
plt.show()



