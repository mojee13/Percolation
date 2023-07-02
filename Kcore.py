#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import igraph as ig
import numpy as np 
from scipy.special import gammainc
from scipy.special import gamma
from matplotlib.ticker import MultipleLocator
import random
from scipy.special import gammainc , gamma
from scipy.optimize import root
import tqdm
import tempfile


# In[2]:


def M_k(k, graph):
    
    n = graph.vcount()
    
    while True:
        
        # Compute the degrees of all nodes in the graph
        degrees = graph.degree()

        # Identify the nodes with degrees less than k
        nodes_to_delete = [node for node, degree in enumerate(degrees) if degree < k]

        # If all nodes have degree >= k, exit the loop
        if not nodes_to_delete:
            break

        # Delete the identified nodes from the graph
        graph.delete_vertices(nodes_to_delete)
        
    return graph


# In[3]:


def rdel(graph, q):
    num_nodes = graph.vcount()
    num_nodes_to_delete = int(num_nodes * q)
    
    nodes_to_delete = random.sample(range(num_nodes), num_nodes_to_delete)
    graph.delete_vertices(nodes_to_delete)
    
    return graph


# In[4]:


# Specify the filename of the saved graph
filenameRL = "router_INET.txt"
# Load the graph from the edgelist file
RL = ig.Graph.Read_Edgelist(filenameRL, directed=False)


# In[5]:


# Specify the filename of the saved graph
filename1 = "graph_c_2_gamma_2.5_inf_1e+07.graphml"
# Load the graph from the edgelist file
scale1 = ig.Graph.Read_Edgelist(filename1, directed=False)
gamma1 = 2.5


# In[6]:


def convert_value(value):
    try:
        return int(float(value))
    except ValueError:
        return 0  # or any other value you prefer for invalid entries

file_path = "text_scale2.txt"

data = np.genfromtxt(file_path, converters={0: convert_value, 1: convert_value})

# Create a temporary file and write the data to it
with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
    np.savetxt(temp_file, data, delimiter=' ', fmt='%d')

    # Read the edgelist from the temporary file
    scale2 = ig.Graph.Read_Edgelist(temp_file.name, directed=False)

# Remove the temporary file
temp_file.close()
gamma2 = 4


# In[7]:


# Specify the filename of the saved graph
filename3 = "graph_c_50_gamma_7.graphml"
# Load the graph from the edgelist file
scale3 = ig.Graph.Read_Edgelist(filename3, directed=False)
gamma3 = 7


# In[ ]:


# Generate a random Erdos-Renyi graph
ss=25
n = 1000000
z10 = 10
z20 = 20
p10 = z10/n
p20 = z20/n
graph10 = ig.Graph.Erdos_Renyi(n, p10)
graph20 = ig.Graph.Erdos_Renyi(n, p20)
fontsize=20

# In[ ]:


GRAPH = [[graph10 , z10 , 'o' , 'ER' , 'white'] , [graph20, z20 , '^'  , 'ER' , 'white'] , [scale1 , gamma1 , 's' , 'SF' , 'black']  , [scale2 , gamma2 , '^' , 'SF' , 'black'] , [scale3 , gamma3 , 'o' , 'SF' , 'black'] , [RL , 1 , '*' , 'RL' , 'black']] 


# In[ ]:


SEED = np.linspace(12345,123456,ss)

# Specify the minimum degree threshold
K = np.arange(2,30)

fig , (ax1,ax2) = plt.subplots(2,1 , figsize = (10,10))
ax2p = ax2.twinx()

for graph in GRAPH:
    
    if graph[3] != 'RL':
        
        mk_average =[]
        sk_average = []
        
        for k in tqdm.tqdm(K):
            
            mk = []
            
            for seed in SEED:
                
                random.seed(seed)
                g = graph[0].copy()
                n = g.vcount()
                g = M_k(k,g)
                a = g.vcount()/n
                mk.append(a)
                
            if all(mk) != 0:
                
                mk_average.append(np.mean(mk))
                
            else:
                
                break
                
        index = np.where(K == k)[0][0]
        sk_average = [mk_average[i] - mk_average[i+1] if i < len(mk_average)-1 else mk_average[i] for i in range(len(mk_average))]
        
        if graph[3]=='ER':
            
            ax1.plot(K[:index], mk_average, marker=graph[2], linestyle='-', markerfacecolor=graph[4],
                     markeredgecolor='black' , color='black' , markersize = 5)
            ax1.text(K[index] , mk_average[-1] , r'$z_1 =$' + f'{graph[1]}' , fontsize = 0.7*fontsize)
            ax2p.plot(K[:index], sk_average, marker=graph[2], linestyle='-', markerfacecolor=graph[4],
                      markeredgecolor='black' , color='black' , markersize = 5)
            ax2p.text(K[index] , sk_average[-1] , r'$z_1 =$' + f'{graph[1]}' , fontsize = 0.7*fontsize)

        elif graph[3]=='SF':
            
            ax1.plot(K[:index], mk_average, marker=graph[2], linestyle='-', markerfacecolor=graph[4], markeredgecolor='black' , color='black' , markersize = 5)
            ax1.text(K[index] , mk_average[-1] , r'$\gamma =$' + f'{graph[1]}' , fontsize = 0.7*fontsize)
            ax2.plot(K[1:index], sk_average[1:], marker=graph[2], linestyle='-', markerfacecolor=graph[4],
                     markeredgecolor='black' , color='black' , markersize = 5)
            ax2.text(K[index] , sk_average[-1] , r'$\gamma =$' + f'{graph[1]}' , fontsize = 0.7*fontsize)

    elif graph[3] == 'RL':
        
        print(graph[0].summary())
        
        mk_average =[]
        sk_average = []
        
        for k in tqdm.tqdm(K[0:10]):
            
            mk = []
            
            for seed in SEED:
                
                random.seed(seed)
                g = graph[0].copy()
                n = g.vcount()
                g = M_k(k,g)
                a = g.vcount()/n
                mk.append(a)
                
            if all(mk) != 0:
                
                mk_average.append(np.mean(mk))
                
            else:
                
                break
                
        index = np.where(K == k)[0][0]
        sk_average = [mk_average[i] - mk_average[i+1] if i < len(mk_average)-1 else mk_average[i] for i in range(len(mk_average))]
                
        ax1.plot(K[:index+1], mk_average, marker=graph[2], linestyle='-',
                 markerfacecolor=graph[4], markeredgecolor='black' , color='black' , markersize = 9)
        ax1.text(K[0]-1 , mk_average[0] , r'$RL$' , fontsize = 0.7*fontsize)
        ax2.plot(K[1:index], sk_average[1:-1], marker=graph[2], linestyle='-',
                 markerfacecolor=graph[4], markeredgecolor='black' , color='black' , markersize = 9)
        ax2.text(K[2] , sk_average[2] , r'RL' , fontsize = 0.7*fontsize)
        
        
ax1.set_xlabel(r'$K$', fontsize=fontsize, color='black', labelpad=5)
ax1.set_ylabel(r'$M(k)$', fontsize=fontsize, color='black', rotation='vertical', labelpad=5)
ax1.set_ylim(-0.1,1.2)
ax1.set_xlim(1,28)
ax1.tick_params(axis='both', which='major', top=False, right=False, labeltop=False, labelright=False , direction='out', labelsize=0.7 * fontsize , length= 8, width=1, pad=5)
ax1.tick_params(axis='both', which='minor', top=False, right=False, labeltop=False, labelright=False , direction='out', labelsize=0.7 * fontsize , length= 3, width=1, pad=5)
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
ax1.set_xticks(np.arange(0, 30, 5))
ax1.set_yticks(np.arange(0, 1.2, 0.5))
ax1.set_xticklabels(['','5', '10' , '15' , '20' , '25'])
ax1.set_yticklabels(['0,0','0,5', '1,0'])
ax1.text(26, 1.1, "(a)", fontsize=fontsize, ha='right', va='top') 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)


ax2p.set_ylim(0.00001, 1)
ax2p.set_yscale("log")
ax2p.tick_params(axis='y', which='major', top=False, labeltop=False, direction='in', labelsize=0.7 * fontsize, length=8, width=1, pad=5)
ax2p.tick_params(axis='y', which='minor', top=False, labeltop=False, labelbottom=False, direction='in', labelsize=0.7 * fontsize, length=3, width=1, pad=5)
ytick_labelsp = [ r'$1 \times 10^{-5}$', r'$1 \times 10^{-4}$', r'$1 \times 10^{-3}$','0.01','0.1', '1']
ytick_positionsp = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
ax2p.set_yticks(ytick_positionsp)
ax2p.set_yticklabels(ytick_labelsp)
ax2p.spines['top'].set_visible(False)


ax2.set_xlabel(r'$K$', fontsize=fontsize, color='black', labelpad=5)
ax2.set_ylabel(r'$S(k)$', fontsize=fontsize, color='black', rotation='vertical', labelpad=5)
ax2.set_ylim(0.0001, 0.5)
ax2.set_xlim(2, 40)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.tick_params(axis='both', which='major', top=False, right=False, labeltop=False, labelright=False, direction='out', labelsize=0.7 * fontsize, length=8, width=1, pad=7)
ax2.tick_params(axis='x', which='minor', top=False, right=False, labeltop=False, bottom=True, labelbottom=True, labelright=False, direction='out', labelsize=0.7 * fontsize, length=3, width=1, pad=7)
ax2.set_xticks([3, 10])
ax2.set_xticklabels(['3', '10'])
ytick_labels = [ r'$1 \times 10^{-4}$', r'$1 \times 10^{-3}$', r'$1 \times 10^{-2}$','0.1', '']
ytick_positions = [ 0.0001, 0.001, 0.01, 0.1, 0.5]
ax2.set_yticks(ytick_positions)
ax2.set_yticklabels(ytick_labels)
ax2.spines['top'].set_visible(False)
ax2.text(0.95, 0.6, "(b)",  transform=ax2.transAxes ,fontsize=fontsize, ha='right', va='top') 
 
plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots
plt.show()

plt.savefig('MS_K.pdf')


# In[ ]:


SEED = np.linspace(12345, 123456, ss)
fontsize = 20

K = [7, 6, 5, 4, 3]
Q = np.linspace(0, 0.9, 200)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax_inset = ax.inset_axes([0.55, 0.55, 0.4, 0.4])


for k in K:
    
    mk_average =[]
    z1_k_average = []
    
    for q in tqdm.tqdm(Q):
        
        z1 = []
        mk = []
        
        for seed in SEED:
        
            random.seed(seed)
            g = graph10.copy()
            g = rdel(g, q)
            g = M_k(k, g)
            z = np.mean(g.degree())
            z1.append(z)
            a = g.vcount()/n
            mk.append(a)
            
        if all(mk) != 0:
            
            mk_average.append(np.mean(mk))
            z1_k_average.append(np.mean(z1))
            
        else:
            
            break

    index = np.where(Q == q)[0][0]
    ax.plot(Q[:index], mk_average, color='black')
    ax_inset.plot(Q[:index], z1_k_average, color='black')

    # Add dashed lines from the last y point to the x-axis
    last_x = Q[index-1]
    last_y = mk_average[-1]
    ax.plot([last_x, last_x], [last_y, 0], color='black', linestyle='dotted')
    
    last_x_inset = Q[index-1]
    last_y_inset = z1_k_average[-1]
    ax_inset.plot([last_x_inset, last_x_inset], [last_y_inset, 0], color='black', linestyle='dotted')


    if k==7:
        
        ax.text(last_x+0.01 , 0.1 ,f'k ={k}', fontsize = 0.7*fontsize)
        ax.text(last_x+0.01 , 0.05 , f'Q ={last_x*100:0.1f}%' , fontsize = 0.7*fontsize)
        
        ax_inset.text(last_x_inset+0.01 , 1 ,f'k = {k}', fontsize = 0.5*fontsize)        
    else:
        
        ax.text(last_x+0.01 , 0.1 ,f'{k}', fontsize = 0.7*fontsize)
        ax.text(last_x+0.01 , 0.05 , f'{last_x*100:0.1f}%' , fontsize = 0.7*fontsize)
        
        ax_inset.text(last_x_inset+0.01 , 1 , f'{k}' , fontsize = 0.5*fontsize)

ax.set_xlabel(r'$Q$', fontsize=fontsize, color='black')
ax.set_ylabel(r'$M(k)$', fontsize=fontsize, color='black', rotation='vertical', labelpad=5)
ax.set_ylim(0,1.19)
ax.set_xlim(0,0.8)
ax.tick_params(axis='both', which='major',direction='out', length=8, width=1, pad=5 , top=False, right=False, labeltop=False, labelright=False ,labelsize=0.7 * fontsize)
ax.tick_params(axis='both', which='minor',direction='out', length=5, width=1, pad=5 , top=False, right=False, labeltop=False, labelright=False ,labelsize=0.7 * fontsize)
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.set_xticks(np.arange(0, 0.801, 0.2))
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.set_xticklabels(['0,0','0,2', '0,4' , '0,6' , '0,8'])
ax.set_yticklabels(['0,0','0,2', '0,4' , '0,6' , '0,8' , '1,0'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax_inset.set_xlabel(r'$Q$', fontsize=0.7*fontsize, color='black')
ax_inset.set_ylabel(r'$Z_1(k)$', fontsize=0.7*fontsize, color='black', rotation='vertical', labelpad=5)
ax_inset.set_ylim(0, 11)
ax_inset.set_xlim(0, 0.8)
ax_inset.tick_params(axis='both', which='major', top=False, right=False, labeltop=False, labelright=False , direction='out', labelsize=0.7 * fontsize , length= 4, width=1, pad=5)
ax_inset.tick_params(axis='both', which='minor', top=False, right=False, labeltop=False, labelright=False , direction='out', labelsize=0.7 * fontsize , length= 2, width=1, pad=5)
ax_inset.xaxis.set_minor_locator(MultipleLocator(0.05))
ax_inset.yaxis.set_minor_locator(MultipleLocator(1))
ax_inset.set_xticks(np.arange(0, 0.75, 0.2))
ax_inset.set_yticks(np.arange(0, 11, 2))
ax_inset.set_xticklabels(['0,0','0,2', '0,4' , '0,6'])
ax_inset.spines['top'].set_visible(False)
ax_inset.spines['right'].set_visible(False)

plt.savefig('ER_Q.pdf')


# In[ ]:


# def equation(R, z1, k, p):
#     phi_k = gammainc(k - 1, z1 * (1 - R)) / gamma(k - 1)
#     return R - (1 - p + p * phi_k)

# # Define the wrapper function
# def equation_wrapper(R):
#     return equation(R, z1, k, p)


# In[ ]:


# z1 = 10
# K = [7,6,5,4,3]
# Q = np.linspace(0, 1, 20)
# fig, ax = plt.subplots(1, 1, figsize=(9, 7))

# for k in K:
# #     print(f'k = {k}')
#     M_k = []
#     for q in Q:
#         p = 1 - q
#         # equation_wrapper and root function definition is not provided, so you need to add them accordingly
#         solution = root(equation_wrapper, x0=0.5)
#         if solution.success:
#             R = solution.x[0]
#         m_k = p * (1 - gammainc(k, z1 * (1 - R)) / gamma(k))
#         M_k.append(m_k)
#     ax.plot(Q , M_k , label=f'k = {k}')

# fontsize = 20
# ax.set_xlabel(r'$Q$', fontsize=fontsize, color='black')
# ax.set_ylabel(r'$M_k$', fontsize=fontsize, color='black', rotation='horizontal', labelpad=10)
# ax.tick_params(axis='both', which='both', top=False, right=False, labeltop=False, labelright=False)
# ax.tick_params(axis='both', which='both', direction='out', labelsize=0.7 * fontsize, width=1, length=6)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.legend()

# # Determine inset plot coordinates
# inset_width = 0.4
# inset_height = 0.3
# inset_x = 0.5
# inset_y = 0.6

# # Create a new set of axes for the inset plot
# ax_inset = fig.add_axes([inset_x, inset_y, inset_width, inset_height])

# # Copy the main plot into the inset plot
# for line in ax.lines:
#     ax_inset.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())

# # Set limits for the inset plot
# ax_inset.set_xlim(0.2, 0.8)
# ax_inset.set_ylim(0.4, 1)

# # Remove top and right axes from the inset plot
# ax_inset.spines['top'].set_visible(False)
# ax_inset.spines['right'].set_visible(False)

# # Add x and y ticks to the inset plot
# ax_inset.set_xticks([0.3, 0.5, 0.7])
# ax_inset.set_yticks([0.5, 0.7, 0.9])

# # Remove ticks and labels from the inset plot's axes
# ax_inset.tick_params(axis='both', which='both', top=False, right=False, labeltop=False, labelright=False)
# ax_inset.set_ylim(0,1)
# ax_inset.set_xlim(0,0.8)

# # Add a border around the inset plot
# ax_inset.spines['bottom'].set_color('black')
# ax_inset.spines['left'].set_color('black')

# # Add labels to the inset plot
# ax_inset.set_xlabel(r'$Q$', fontsize=fontsize, color='black')
# ax_inset.set_ylabel(r'$M_k$', fontsize=fontsize, color='black', rotation='horizontal', labelpad=10)

# # Add a legend to the inset plot
# ax_inset.legend()

# plt.show()


# In[ ]:


# # Generate a random Erdos-Renyi graph
# n = 5000
# z1 = 10
# p = z1/n
# graph = ig.Graph.Erdos_Renyi(n, p)
# SEED = np.linspace(12345,123456,20)
# fontsize = 20

# # Specify the minimum degree threshold
# K = [7,6 ,5, 4 ,3 ]
# # random delete range
# Q = np.linspace(0,0.9 , 500)


# fig , ax = plt.subplots(1,1 , figsize = (10,7))
# for k in K:
#     mk_average =[]
#     for q in tqdm.tqdm(Q):
#         mk = []
#         for seed in SEED:
#             random.seed(seed)
#             g = graph.copy()
#             g = rdel(g,q)
#             g = M_k(k,g)
#             a = g.vcount()/n
#             mk.append(a)
            
#         if all(mk) != 0:
#             mk_average.append(np.mean(mk))
#         else:
#             break
#     index = np.where(Q == q)[0][0]
#     ax.plot(Q[:index], mk_average, color='black')

#     # Add dashed lines from the last y point to the x-axis
#     last_x = Q[index-1]
#     last_y = mk_average[-1]
#     ax.plot([last_x, last_x], [last_y, 0], color='black', linestyle='dashed')
#     if k==7:
#         ax.text(last_x+0.01 , 0.1 , f'k = {k}' , fontsize = 0.7*fontsize)
#         ax.text(last_x+0.01 , 0.05 , f'Q = {last_x*100:0.1f}%' , fontsize = 0.7*fontsize)
#     else:
#         ax.text(last_x+0.01 , 0.1 , f'{k}' , fontsize = 0.7*fontsize)
#         ax.text(last_x+0.01 , 0.05 , f'{last_x*100:0.1f}%' , fontsize = 0.7*fontsize)

# ax.set_xlabel(r'$Q$', fontsize=fontsize, color='black')
# ax.set_ylabel(r'$M(k)$', fontsize=fontsize, color='black', rotation='vertical', labelpad=15)
# ax.set_ylim(0,1.2)
# ax.set_xlim(0,0.8)
# ax.tick_params(axis='both', which='both', top=False, right=False, labeltop=False, labelright=False)
# ax.tick_params(axis='both', which='both', direction='out', labelsize=0.7 * fontsize)
# ax.xaxis.set_tick_params(which='both', direction='out', length=10, width=1, pad=15)
# ax.set_xticks(np.arange(0, 1, 0.2))
# ax.set_yticks(np.arange(0, 1.1, 0.2))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)


# In[ ]:


# n = 1000
# z1 = 10
# p = z1 / n
# graph = ig.Graph.Erdos_Renyi(n, p)
# SEED = np.linspace(12345, 123456, 10)
# fontsize = 20

# K = [7, 6, 5, 4, 3]
# Q = np.linspace(0, 0.9, 200)

# fig, ax = plt.subplots(1, 1, figsize=(10, 7))
# for k in K:
#     z1_k_average = []
#     for q in tqdm.tqdm(Q):
#         z1 = []
#         mk = []
#         for seed in SEED:
#             random.seed(seed)
#             g = graph.copy()
#             g = rdel(g, q)
#             g = M_k(k, g)
#             z = np.mean(g.degree())
#             z1.append(z)
#             a = g.vcount()/n
#             mk.append(a)
            
#         if all(mk) != 0:
#             z1_k_average.append(np.mean(z1))
#         else:
#             break

#     index = len(z1_k_average)
#     ax.plot(Q[:index], z1_k_average, color='black')
#     last_x = Q[index-1]
#     last_y = z1_k_average[-1]
#     ax.plot([last_x, last_x], [last_y, 0], color='black', linestyle='dashed')
    
#     if k == 7:
#         ax.text(last_x + 0.01, 1, f'k = {k}', fontsize=0.7*fontsize)
#         ax.text(last_x + 0.01, 0.5, f'Q = {last_x * 100:0.1f}%', fontsize=0.7*fontsize)
#     else:
#         ax.text(last_x + 0.01, 1, f'{k}', fontsize=0.7*fontsize)
#         ax.text(last_x + 0.01, 0.5, f'{last_x * 100:0.1f}%', fontsize=0.7*fontsize)
        
        
# ax.set_xlabel(r'$Q$', fontsize=fontsize, color='black')
# ax.set_ylabel(r'$Z_1(k)$', fontsize=fontsize, color='black', rotation='vertical', labelpad=15)
# ax.set_ylim(0, 10)
# ax.set_xlim(0, 0.8)

# # ax.tick_params(axis='y', which='major',labelleft=True, top=False, right=False, labeltop=False, labelright=False)
# # ax.tick_params(axis='y', which='major', direction='out', labelsize=0.7 * fontsize)
# # ax.xaxis.set_tick_params(which='both', direction='out', length=10, width=1, pad=15)

# # Manipulate tick parameters for x-axis
# plt.tick_params(axis='both',           # Selecting x-axis ticks
#                 which='major',       # Apply changes to both major and minor ticks
#                 bottom=True,        # Show ticks on the bottom
#                 top=False,           # Show ticks on the top
#                 labelbottom=True,   # Show tick labels on the bottom
#                 labeltop=False,     # Hide tick labels on the top
#                 direction='out',  # Tick direction (inward, outward, or inout)
#                 length=7,           # Length of the tick marks in points
#                 width=2,            # Width of the tick marks in points
#                 color='black',        # Color of the tick marks
#                 pad=2,              # Padding between tick marks and tick labels in points
#                 labelsize=0.7*20,        # Font size of the tick labels
#                 labelcolor='black')  # Color of the tick labels

# # Manipulate tick parameters for x-axis
# plt.tick_params(axis='both',           # Selecting x-axis ticks
#                 which='minor',       # Apply changes to both major and minor ticks
#                 bottom=True,        # Show ticks on the bottom
#                 top=False,           # Show ticks on the top
#                 labelbottom=True,   # Show tick labels on the bottom
#                 labeltop=False,     # Hide tick labels on the top
#                 direction='out',  # Tick direction (inward, outward, or inout)
#                 length=4,           # Length of the tick marks in points
#                 width=2,            # Width of the tick marks in points
#                 color='black',        # Color of the tick marks
#                 pad=2,              # Padding between tick marks and tick labels in points
#                 labelsize=0.7*20,        # Font size of the tick labels
#                 labelcolor='black')  # Color of the tick labels

# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

# ax.set_xticks(np.arange(0, 0.8, 0.2))
# ax.set_yticks(np.arange(0, 11, 2))

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.show()

