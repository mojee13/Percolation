# %%
import numpy as np
import igraph as ig
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tqdm

def phi_psi(g, k):
    n = g.vcount()
    L = g.cliques(min=k, max=k)
    adj_kc = np.zeros((len(L), len(L)))

    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if len(set(L[i]).intersection(set(L[j]))) == k - 1:
                adj_kc[i, j] = 1
                adj_kc[j, i] = 1

    g = ig.Graph.Adjacency(adj_kc, mode='undirected')

    g.vs["id"] = L
    g.vs["label"] = L

    components = g.components()

    if len(components) == 0:
        phi = 0
        psi = 0
        return phi, psi

    giant_component = components.giant()

    num_vertices_giant = giant_component.vcount()
    num_vertices = g.vcount()
    psi = num_vertices_giant / num_vertices

    label = giant_component.vs["label"]
    n_star = len(set().union(*label))
    phi = n_star / n
    return phi, psi

# %%
marker_shapes = [['^', 'none'], ['s', 'none'], ['o', 'none'], ['^', 'black'], ['s', 'black'], ['o', 'black']]
N = [100, 200, 500]
k = 4
n_order = 80    # times that run for each num_iteration
m = 5  #avarage times
seed_base = 12345
order = np.linspace(0.2, 1.8, n_order)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for r in tqdm.tqdm(range(len(N))):
    n = N[r]
    p_c = 1 / ((k - 1) * n) ** (1 / (k - 1))
    p = p_c * order

    value = np.zeros((n_order, 3))
    for t in tqdm.tqdm(range(n_order)):
        A = []
        B = []
        for j in range(m):
            random.seed(seed_base + j)
            g = ig.Graph.Erdos_Renyi(n=n, p=p[t])
            g = g.simplify()
            g.vs["id"] = range(n)
            g.vs["label"] = [str(i) for i in range(n)]
            phi, psi = phi_psi(g, k)
            A.append(phi)
            B.append(psi)
        value[t, :] = [order[t], np.mean(A), np.mean(B)]

    ax1.plot(value[:, 0], value[:, 1], marker=marker_shapes[r][0], linestyle='-', linewidth=0.7,
             color='black', markersize=4, markerfacecolor=marker_shapes[r][1], markeredgecolor='black',
             label=f'n = {n}', markevery=1)

    ax2.plot(value[:, 0], value[:, 2], marker=marker_shapes[r][0], linestyle='-', linewidth=0.7,
             color='black', markersize=4, markerfacecolor=marker_shapes[r][1], markeredgecolor='black',
             label=f'n = {n}', markevery=1)


fontsize = 20
# For ax1
ax1.set_xlabel(r'$p / P_c(k)$', fontsize=fontsize, color='black')
ax1.set_ylabel(r'$\phi$', fontsize=fontsize, color='black', rotation='horizontal', labelpad=10)
ax1.tick_params(axis='both', which='both', direction='in', labelsize=0.7*fontsize)
ax1.tick_params(axis='both', which='both', top=True, right=True, labeltop=False, labelright=False)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

# For ax2
ax2.set_xlabel(r'$p / P_c(k)$', fontsize=fontsize, color='black')
ax2.set_ylabel(r'$\psi$', fontsize=fontsize, color='black', rotation='horizontal', labelpad=10)
ax2.tick_params(axis='both', which='both', direction='in', labelsize=0.7*fontsize)
ax2.tick_params(axis='both', which='both', top=True, right=True, labeltop=False, labelright=False)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

plt.savefig('phi_psi.pdf', bbox_inches = 'tight')

# %%
k_alpha_shift_text = [(3, 0.45, 0, 0.25), (4, 0.5, 0.4, 0.6), (5, 0.5, 0.8, 1)]
fig, ax3 = plt.subplots(1, 1, figsize=(10, 5))
for r in range(len(N)):
    n = N[r]
    for (z, x, c, v) in k_alpha_shift_text:

        p_c = 1 / ((z - 1) * n) ** (1 / (z - 1))
        p = p_c * order

        value = np.zeros((n_order, 2))
        for t in tqdm.tqdm(range(n_order)):
            A = []
            for j in range(m):
                random.seed(seed_base + j)
                g = ig.Graph.Erdos_Renyi(n=n, p=p[t])
                g = g.simplify()
                g.vs["id"] = range(n)
                g.vs["label"] = [str(i) for i in range(n)]
                phi, psi = phi_psi(g, z)
                A.append(phi)
            value[t, :] = [(order[t] - 1) * n ** x, np.mean(A)]

            ax3.plot(value[:, 0], value[:, 1] + c, marker=marker_shapes[r][0], linestyle='',
                     color='black', markersize=4, markerfacecolor=marker_shapes[r][1], markeredgecolor='black',
                     markevery=1)

            ax3.text(-3.8, v, r'$\alpha = {}$'.format(x) + r'$,\ k={}$'.format(z), fontsize=15, color='black')
            
# For ax3
ax3.set_xlabel(r'$(p$ / $P_c(k)-1)*N^\alpha$', fontsize=fontsize, color='black')
ax3.set_ylabel(r'$\phi$', fontsize=fontsize, color='black', rotation='horizontal', labelpad=10)
ax3.set_xlim(-4, 6)
ax3.tick_params(axis='both', which='both', direction='in', labelsize=0.7 * fontsize)
ax3.tick_params(axis='both', which='both', top=True, right=True, labeltop=False, labelright=False)
ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # Set tick locator for even fraction values
# ax3.legend(fontsize=fontsize)



plt.savefig('phi_alpha.pdf' , bbox_inches = 'tight' )

# %%
# fig, ax4 = plt.subplots(1, 1, figsize=(12, 7))
# N_c = [100,200 ,500 ,1000]
# K = [3,4,5]


# for k in tqdm.tqdm(K):
#     value = np.zeros((len(N_c), 3))
#     for r in range(len(N_c)):
#         n = N_c[r]
#         print(r , n)
#         p_c = 1 / ((k - 1) * n) ** (1 / (k - 1))
#         B = []
#         B_var = []
#         for j in range(m):
#             random.seed(seed_base + j)
#             g = ig.Graph.Erdos_Renyi(n=n, p=p_c)
#             g = g.simplify()
#             g.vs["id"] = range(n)
#             g.vs["label"] = [str(i) for i in range(n)]
#             phi, psi = phi_psi(g, k)
#             B.append(psi)
#             B_var.append(psi**2)
            
#         value[r, :] = [n , np.mean(B) , np.sqrt(np.mean(B_var)-np.mean(B)**2)]
    
#     ax4.plot(value[:, 0], value[:, 1] , marker=marker_shapes[K.index(k)][0], linestyle='',
#             color='black', markersize=4, markerfacecolor='black', markeredgecolor='black', markevery=1)
#     ax4.errorbar(value[:, 0], value[:, 1], yerr= value[:, 2], fmt=marker_shapes[K.index(k)][0], markersize=4, color='black',
#              markerfacecolor='black', markeredgecolor='black', ecolor='black', capsize=2, capthick=1, elinewidth=1,
#              linestyle='None' )        
    
#     # Perform linear regression
#     coefficients = np.polyfit(np.log(value[:, 0]), np.log(value[:, 1]), 1)
#     slope = coefficients[0]

#     # Generate x-values for the best-fit line
#     x_fit = np.linspace(min(value[:, 0]), max(value[:, 0]), 100)
#     y_fit = np.exp(coefficients[1]) * x_fit**coefficients[0]
#          c 
#     # Plot the best-fit line
#     ax4.plot(x_fit, y_fit, linestyle='-', color='black')

#     # Add text with K and slope information
#     text = r'$K = {}, \sim N^{{{:.2f}}}$'.format(k, slope)
#     # text = f'K = {k}, ~ N^{slope:.2f}'
#     ax4.text(0.6, 0.08*k+0.3, text , transform=ax4.transAxes, fontsize=15)


# fontsize= 20      
        
# # For ax4
# ax4.set_xlabel(r'$N$', fontsize=fontsize, color='black')
# ax4.set_ylabel(r'$\psi_c$', fontsize=fontsize, color='black', rotation='horizontal', labelpad=15)
# ax4.set_xscale('log')
# ax4.set_yscale('log')
# ax4.set_xlim(9.9*10**1, 10**4)
# ax4.set_ylim(10**-4, 10**0)
# ax4.tick_params(axis='both', which='both', direction='in', labelsize=0.7*fontsize)
# ax4.tick_params(axis='both', which='both', top=True, right=True, labeltop=False, labelright=False)
# # ax4.legend(fontsize=fontsize)
# plt.savefig('critical_psi.pdf' , bbox_inches = 'tight')

# %%



