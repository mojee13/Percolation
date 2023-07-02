import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from scipy.special import comb
import numpy as np
import pandas as pd
import tqdm


#######################################################
KK=np.logspace(np.log10(1), np.log10(100), num=20)
num_nodes = 10000
iterations= 10000
#####################################################3

def generate_random_graph(degree_sequence):
    # Generate a random graph with the desired degree sequence
    graph = nx.configuration_model(degree_sequence)
    graph = nx.Graph(graph)  # Remove parallel edges and self-loops
    graph.remove_edges_from(nx.selfloop_edges(graph))  # Remove self-loops
    return graph

def calculate_ejk(j, k, p, q,K):
    n= 0.5 * (1 - math.exp(-1/K))
    exponent = math.exp(-(j + k) / K)
    binomial_term = comb(j + k, j) * p**j * q**k + comb(j + k, k) * p**k * q**j
    ejk = n * exponent * binomial_term
    return ejk

def apply_metropolis_dynamics(g, EJK, iterations):
    # Apply Metropolis dynamics to the graph
    num_edges = g.number_of_edges()
    for _ in tqdm.tqdm(range(iterations)):
        # Choose two random edges
        edges = list(g.edges())
        r1=np.random.choice(len(edges))
        r2=np.random.choice(len(edges))
        v1, w1 = edges[r1]
        v2, w2 = edges[r2]

        # Measure  degrees
        j1 = g.degree(v1)-1
        k1 = g.degree(w1)-1
        j2 = g.degree(v2)-1
        k2 = g.degree(w2)-1 

        # Calculate the acceptance probability
        p_change = EJK[j1,j2]*EJK[k1,k2] / EJK[j1,k1]*EJK[j2,k2]
        acceptance_prob = min(1, p_change)
        #print(p_change)
        # Replace the edges with the new ones with the acceptance probability
        if np.random.random() < acceptance_prob:
            #print(p_change)
            g.remove_edges_from([(v1, w1), (v2, w2)])
            g.add_edges_from([(v1, v2), (w1, w2)])

    return g.copy()


G_A=[]
asort_A=[]

for K in KK:
    N=10+2*int(K)
    p = 0.5
    q = 1-p
    EJK=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            EJK[i][j]=calculate_ejk(i+1,j+1,p,q,K)

    num_columns = len(EJK[0])

    column_sums = [0] * num_columns

    for row in EJK:
        for i in range(num_columns):
            column_sums[i] += row[i]
    qk=column_sums

    pk=[]
    for i in range(len(qk)):
        p_k=qk[i]/(i+1)
        pk.append(p_k)

    # Degree distribution probabilities
    degree_distribution = pk/np.sum(pk)

    # Generate degree sequence
    degree_sequence = np.random.choice(range(len(degree_distribution)), size=num_nodes, p=degree_distribution)

    # Print degree sequence
    #print(degree_sequence)
    if np.sum(degree_sequence)%2 ==1 :
        degree_sequence[-1]+=1

    graph = generate_random_graph(degree_sequence)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    #pos = nx.spring_layout(graph)
    #nx.draw(graph,pos,node_size=5,width=.1)
    g = graph.copy()
    updated_graph = apply_metropolis_dynamics(g,EJK, iterations=iterations)
    gcc=len(max(nx.connected_components(updated_graph), key=len))
    G_A.append(gcc/num_nodes)
    asort_A.append(nx.degree_assortativity_coefficient(updated_graph))
    print(K,'A Done')
############################
G_N=[]
asort_N=[]

for K in KK:
    N=10+2*int(K)
    p = 0.1464
    q = 1-p
    EJK=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            EJK[i][j]=calculate_ejk(i+1,j+1,p,q,K)

    num_columns = len(EJK[0])

    column_sums = [0] * num_columns

    for row in EJK:
        for i in range(num_columns):
            column_sums[i] += row[i]
    qk=column_sums

    pk=[]
    for i in range(len(qk)):
        p_k=qk[i]/(i+1)
        pk.append(p_k)

    # Degree distribution probabilities
    degree_distribution = pk/np.sum(pk)

    # Generate degree sequence
    degree_sequence = np.random.choice(range(len(degree_distribution)), size=num_nodes, p=degree_distribution)

    # Print degree sequence
    #print(degree_sequence)
    if np.sum(degree_sequence)%2 ==1 :
        degree_sequence[-1]+=1

    graph = generate_random_graph(degree_sequence)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    #pos = nx.spring_layout(graph)
    #nx.draw(graph,pos,node_size=5,width=.1)
    g = graph.copy()
    updated_graph = apply_metropolis_dynamics(g,EJK, iterations=10000)
    gcc=len(max(nx.connected_components(updated_graph), key=len))
    G_N.append(gcc/num_nodes)
    asort_N.append(nx.degree_assortativity_coefficient(updated_graph))
    print(K,'N Done')
#################################
G_D=[]
asort_D=[]

for K in KK:
    N=10+2*int(K)
    p = 0.05
    q = 1-p
    EJK=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            EJK[i][j]=calculate_ejk(i+1,j+1,p,q,K)

    num_columns = len(EJK[0])

    column_sums = [0] * num_columns

    for row in EJK:
        for i in range(num_columns):
            column_sums[i] += row[i]
    qk=column_sums

    pk=[]
    for i in range(len(qk)):
        p_k=qk[i]/(i+1)
        pk.append(p_k)

    # Degree distribution probabilities
    degree_distribution = pk/np.sum(pk)

    # Generate degree sequence
    degree_sequence = np.random.choice(range(len(degree_distribution)), size=num_nodes, p=degree_distribution)

    # Print degree sequence
    #print(degree_sequence)
    if np.sum(degree_sequence)%2 ==1 :
        degree_sequence[-1]+=1

    graph = generate_random_graph(degree_sequence)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    #pos = nx.spring_layout(graph)
    #nx.draw(graph,pos,node_size=5,width=.1)
    g = graph.copy()
    updated_graph = apply_metropolis_dynamics(g,EJK, iterations=10000)
    gcc=len(max(nx.connected_components(updated_graph), key=len))
    G_D.append(gcc/num_nodes)
    asort_D.append(nx.degree_assortativity_coefficient(updated_graph))
    print(K,'D Done')

########################3
    
plt.figure(figsize=(8, 6))

plt.scatter(KK,G_A,marker='o',color='red',label='assortative')
plt.plot(KK,G_A,linestyle='--',color='red')

plt.plot(KK,G_N,linestyle='--',color='k')
plt.scatter(KK,G_N,marker='x',color='k',label='neutral')

plt.scatter(KK,G_D,marker='+',color='blue',label='disassortative')
plt.plot(KK,G_D,linestyle='--',color='blue')
plt.ylabel('GCC/n')
plt.xlabel('K')
plt.xscale('log')
plt.legend()
plt.savefig('AS_GCC_ROY.pdf')


#####################3

plt.figure(figsize=(8, 6))

plt.scatter(KK,asort_A,marker='o',color='red',label='assortative')
plt.plot(KK,asort_A,linestyle='--',color='red')

plt.plot(KK,asort_N,linestyle='--',color='k')
plt.scatter(KK,asort_N,marker='x',color='k',label='neutral')

plt.scatter(KK,asort_N,marker='+',color='blue',label='disassortative')
plt.plot(KK,asort_N,linestyle='--',color='blue')
plt.xlabel('K')
plt.ylabel('asort')
plt.xscale('log')

plt.legend()

plt.savefig('AS_ROY.pdf')


############
data = {
    'K': KK,
    'GCCA': G_A,
    'GCCN': G_N,
    'GCCD': G_D,
    'AA': asort_A,
    'AN': asort_N,
    'AD': asort_D
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('data_AS_ROY.csv', index=False)
