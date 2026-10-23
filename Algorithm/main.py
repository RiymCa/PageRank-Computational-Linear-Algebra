from utils import power_method_adv_csr, power_method_adv
from analysis import * 
from collections import Counter, defaultdict
import numpy as np 
import time

### ---------------- PAPER WEBS ----------------
separate()
print("PAPER WEBS")
### -------- Figure 2.1
A_21 = np.array([
        [0, 0, 1, 0.5],
        [1/3, 0, 0, 0],
        [1/3, 0.5, 0, 0.5],
        [1/3, 0.5, 0, 0]])

# pagerank computation
rnk_21 = power_method_adv(A_21)

# Output prints
separate()
print("PageRank for Figure 2.1")
for i, score in enumerate(rnk_21.flatten()):
        print(f"Page {i+1}: {score:.4f}")


### -------- Figure 2.2
A_22 = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0.5],
        [0, 0, 1, 0, 0.5],
        [0, 0, 0, 0, 0]])

# pagerank computation
rnk_22 = power_method_adv(A_22)

# Output prints
separate()
print("PageRank for Figure 2.2")
for i, score in enumerate(rnk_22.flatten()):
        print(f"Page {i+1}: {score:.4f}")




### ---------------- HOLLINS.DAT WEBS ----------------
separate()
print("HOLLINS.DAT WEB")
separate()
### -------- DATA LOADING 
with open("hollins.dat", "r") as f:
    all_lines = f.readlines()
    
    # getting total number of sites from the first line of the file
    total_sites = int(all_lines[0].split(" ")[0])
    
    # extracting all file identifiers
    sites = map(lambda x: int(x.split(" ")[0]),  all_lines[1:total_sites +1])
    
    # extracting all the (from, to) connections - come after total_sites + 1 in the file
    connections = map(lambda x: tuple(map(int, x.split(" "))), all_lines[total_sites +1: ])
    
    outgoing_links = Counter()
    incoming_links = defaultdict(list)
    
    for sender, reciever in list(connections):
        outgoing_links[sender] += 1
        incoming_links[reciever].append(sender)
    
    # values -- we're using CSR
    LV = []
    # column numbers
    JC = []
    # row numbers
    IA = []
    tot_ele = 0
    
    A = np.zeros(shape=(total_sites, total_sites), dtype=np.float64) 

    for reciever in range(1, total_sites+1):
        IA.append(tot_ele)
        for sender in sorted(incoming_links[reciever]):
            #populate dense matrix 
            A[reciever - 1, sender - 1] = 1 / outgoing_links[sender] 
            #populate sparse matrix 
            LV.append(1 / outgoing_links[sender])
            JC.append(sender - 1)
            tot_ele += 1
    IA.append(tot_ele)


### -------- PAGERANK COMPUTATION
# -------- DENSE VERSION 
    #contains indexes of dangling nodes 
    dangling_nodes = np.where(np.sum(A, axis=0)==0) 
    #filling columns corresponfing to dangling nodes
    A_no_dn = A.copy()
    A_no_dn[:, dangling_nodes] = 1 / A_no_dn.shape[0] 

    start_dense = time.time()
    rnk_dense = power_method_adv(A_no_dn) 
    end_dense = time.time()

# -------- SPARSE VERSION 
    n = len(IA) - 1
    #stores the number of non zero elements for each column
    nz_per_col = np.zeros(n) 
    for col in JC:
        nz_per_col[col] += 1
    dangling_nodes_csr = np.where(nz_per_col == 0)[0]
    
    start_csr = time.time()
    rnk, n_iter, convergence_rate = power_method_adv_csr(LV, JC, IA, total_sites, dangling_nodes_csr)
    end_csr = time.time()


### -------- ANALYSIS

## -------- OUTPUT PRINTS
display_subset_rank(rnk_dense, rnk, incoming_links, outgoing_links, 10)
separate()
display_timings(start_dense, end_dense, start_csr, end_csr)

# # ## -------- PLOTS
plot_dense_vs_sparse(rnk_dense, rnk, total_sites)
plot_nIterations_vs_nNodes(n_iter)
plot_nIterations_vs_nNodes_cumulative_(n_iter)
plot_score_vs_nIterations(n_iter, rnk)
plot_score_distribution(rnk)
plot_convergence_rate(convergence_rate)

 