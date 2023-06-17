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

    
deltas_pr=[]
ns=[1000,5000,10000,15000,20000,25000]
std_pr=[]
ens=5
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


plt.scatter(ns,deltas_pr,s=7,marker='+',color='k',label='PR')
plt.errorbar(ns,deltas_pr,std, capsize=10,fmt = 'o')

d = u"\u0394"  # Delta symbol
n_power = r"$n^{\frac{2}{3}}$"  # LaTeX formatting for n^2/3

x_label = f"{d}/{n_power}"

plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.ylabel(x_label)
plt.xlabel('n')
plt.legend()
plt.savefig("delta_pr.pdf")
df.to_csv('dataframe_delta_pr.csv', index=False)
