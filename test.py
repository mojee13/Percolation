import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm
from igraph import *
#random.seed(10)
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

#ig
#ER
n=1000
g=Graph(n)
lgg_er=[]
r_er=[]
for p in tqdm.tqdm(range(int(1.5*n))):
    er_add(g)
    gcc = len(max(g.connected_components(mode='weak'), key=len))/n
    lgg_er.append(gcc)
    r_er.append(p/n)
#PR

g=Graph(n)
lgg_pr=[]
r_pr=[]
for p in tqdm.tqdm(range(int(1.5*n))):
    gcc =  pr_add(g)/n
    lgg_pr.append(gcc)
    r_pr.append(p/n)
#BF

g=Graph(n)
lgg_bf=[]
r_bf=[]
for p in tqdm.tqdm(range(int(1.5*n))):
    gcc =  bf_add(g)/n
    lgg_bf.append(gcc)
    r_bf.append(p/n)

#plot theEvolution of largest component size in ER,BF and PR 
plt.scatter(r_er,lgg_er,s=7,marker='o',alpha=0.5,color='k',facecolors='none',label='ER')
plt.scatter(r_bf,lgg_bf,s=7,marker='o',alpha=0.5,color='blue',facecolors='none',label='BF(k=1)')
plt.scatter(r_pr,lgg_pr,s=7,marker='o',alpha=0.5,color='red',facecolors='none',label='PR')

plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.ylim(-0.05,1)
plt.xlim(-0.05,1.6)
plt.ylabel('C/n')
plt.xlabel('r')
plt.title('Evolution of largest component size in ER,BF and PR in n=16000')
plt.legend()
plt.savefig("test.pdf")
