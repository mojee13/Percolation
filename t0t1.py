#ig

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm
from igraph import *
import pandas as pd
#random.seed(10)
plt.rcParams.update({'font.size': 12})

def er_add(g):
    nodes = np.arange(0,len(g.vs))
    source = random.choice(nodes)
    target = random.choice(nodes)
    while source == target or g.are_connected(source, target):
        target = random.choice(nodes)
    g.add_edge(source, target)
    
def pr_add(g):
    nodes = np.arange(0,len(g.vs))
    l=list(g.connected_components(mode='weak'))
    maxx=len(max(l, key=len))
    re1=random.sample(list(nodes), 2)
    #while g.are_connected(re1[0], re1[1]) : re1=random.sample(list(nodes), 2)
    re2=random.sample(list(nodes), 2)
    #while g.are_connected(re2[0], re2[1]) : re2=random.sample(list(nodes), 2)
    p1=index_c(l,re1[0])*index_c(l,re1[1])
    p2=index_c(l,re2[0])*index_c(l,re2[1])
    if p1>p2:
        g.add_edge(re2[0],re2[1])
    else:
        g.add_edge(re1[0],re1[1])
    return maxx

def bf_add(g):
    nodes = np.arange(0,len(g.vs))
    l=list(g.connected_components(mode='weak'))
    maxx=len(max(l, key=len))
    re1=random.sample(list(nodes), 2)
    while g.are_connected(re1[0], re1[1]) : re1=random.sample(list(nodes), 2)
    re2=random.sample(list(nodes), 2)
    while g.are_connected(re2[0], re2[1]) : re2=random.sample(list(nodes), 2)
    p1=index_c(l,re1[0])*index_c(l,re1[1])
    if p1==1:
        g.add_edge(re1[0],re1[1])
    else:
        g.add_edge(re2[0],re2[1])
    return maxx

def index_c(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return len(myList[i])


# finding delta in er bf pr
random.seed()
def delta(model,n):
    g=Graph(n,directed=False)
    
    #********ER********
    if model=='er':
        t0=0
        t1=0
        gcc =0
        delta=0
        while gcc<np.sqrt(n):
            er_add(g)
            gcc = len(g.clusters().giant().vs)
            t0+=1
            t1+=1
        print('t0=',t0,'n=',n,'gcc=',gcc)
        while gcc<n/2:
            er_add(g)
            gcc = len(g.clusters().giant().vs)
            t1+=1
        print('t1=',t1,'n=',n,'gcc=',gcc)
        delta=t1-t0
        print('delta=',delta,'n=',n,'gcc=',gcc)
    #********BR********
    if model=='bf':
        t0=0
        t1=0
        gcc =0
        delta=0
        while gcc<np.sqrt(n):
            gcc = bf_add(g)
            t0+=1
            t1+=1
        print('t0=',t0,'n=',n,'gcc=',gcc)
        while gcc<n/2:
            gcc = bf_add(g)
            t1+=1
        print('t1=',t1,'n=',n,'gcc=',gcc)
        delta=t1-t0
        print('delta=',delta,'n=',n,'gcc=',gcc)
    #****************br
    if model=='pr':
        t0=0
        t1=0
        gcc =0
        delta=0
        while gcc<np.sqrt(n):
            gcc = pr_add(g)
            t0+=1
            t1+=1
        while gcc<n/2:
           
            gcc =  pr_add(g)
            t1+=1
        delta=t1-t0
    if model=='pr':
        return delta/(n**(2/3))
    else:
        return delta/n


def t(n):
    g=Graph(n,directed=False)
    t0=0
    t1=0
    gcc =0
    delta=0
    while gcc<np.sqrt(n):
        er_add(g)
        gcc = len(g.clusters().giant().vs)
        t0+=1
        t1+=1
    while gcc<n/2:
        er_add(g)
        gcc = len(g.clusters().giant().vs)
        t1+=1
    delta=t1-t0
    return t0/n,t1/n

ts=[]
std_t0=[]
std_t1=[]
ns=[1000,10000,25000,50000,75000,100000]
ens=5
for n in tqdm.tqdm(ns):
    av_t0=[]
    av_t1=[]

    for _ in tqdm.tqdm(range(ens)):
        t0,t1=t(n)
        av_t0.append(t0)
        av_t1.append(t1)
    ts.append([mean(av_t0),mean(av_t1)])
    std_t0.append(np.std(av_t0))
    std_t1.append(np.std(av_t1))
    
    

t0=[lst[0] for lst in ts]
t1=[lst[1] for lst in ts]

    

plt.plot(ns,t0,'o',color='k',label="t0")
plt.errorbar(ns,t0, std_t0,color='k', capsize=10,fmt = 'o')
plt.plot(ns,t1,'o',color='blue',label="t1")
plt.errorbar(ns,t1, std_t1,color='blue', capsize=10,fmt = 'o')

plt.legend()
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.xlabel('n')
plt.ylabel('t/n')

data = {
    'ns': ns,
    't0': t0,
    't1': t1,
    'std_t0': std_t0,
    'std_t1': std_t1
}

df = pd.DataFrame(data)
plt.savefig("t0t1.pdf")
df.to_csv('dataframe_t0t1.csv', index=False)
