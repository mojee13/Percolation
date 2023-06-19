import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm
from igraph import *
import pandas as pd
#random.seed(10)
# finding delta in er bf pr

#ig

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

    
deltas_pr=[]
ns=[100000,150000]
std_pr=[]
ens=10
for n in tqdm.tqdm(ns):
    av=[]
    for _ in tqdm.tqdm(range(ens)):
        av.append(delta('pr',n))
    deltas_pr.append(mean(av))
    std_pr.append(np.std(av))


data = {
    'ns': ns,
    'deltas_pr': deltas_pr,
    'std_pr': std_pr,
}

df = pd.DataFrame(data)
df.to_csv('dataframe_delta_pr_100000.csv', index=False)

plt.scatter(ns,deltas_pr,s=7,marker='+',color='k',label='PR')
plt.errorbar(ns,deltas_pr,std_pr, capsize=10,fmt = 'o')

d = u"\u0394"  # Delta symbol
n_power = r"$n^{\frac{2}{3}}$"  # LaTeX formatting for n^2/3

x_label = f"{d}/{n_power}"

plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.ylabel(x_label)
plt.xlabel('n')
plt.legend()
plt.savefig("delta_pr_100000.pdf")
