import numpy as np
import matplotlib.pyplot as plt

# displays the first n links sorted by descending ranking showing for each of them
    # number of incoming links
    # number of outgoing links
    # ranking computed on dense representation
    # ranking computed on rparse representation
def display_subset_rank(rnk_dense, rnk_sparse, incoming_links, outgoing_links, n):
    # sorted version of the ranking (descending order to have highest first)
    sorted_indices = np.argsort(-rnk_sparse.flatten())  
    print("RANKING COMPARISON")
    print(f"{'Node ID':<8} {'Incoming':<10} {'Outgoing':<10} {'Dense rank':<12} {'Sparse rank':<12}")
    
    for idx in sorted_indices[:n]:
        node_id = idx + 1
        in_links = len(incoming_links[node_id])
        out_links = outgoing_links.get(node_id, 0)
        print(f"{node_id:<10} {in_links:<10} {out_links:<10} {rnk_dense[idx,0]:<12.8f} {rnk_sparse[idx,0]:<12.8f}")


# highlight advantages of using sparce representation
def display_timings(start_dense, end_dense, start_csr, end_csr):
    print("TIMING DENSE vs SPARSE")
    print(f"Time required by dense representation: {(end_dense - start_dense):<5.4f} sec")
    print(f"Time required by sparse representation: {(end_csr - start_csr):<5.4f} sec")


# plots dense and sparse rankings
# used to prove that both representations produce the same result 
def plot_nIterations_vs_nNodes(n_iter_no_dn):
    plt.figure(figsize=(10,5))
    plt.hist(n_iter_no_dn, bins=50, color='skyblue', edgecolor='black')
    plt.title("Number of iterations to converge")
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of nodes")
    plt.grid(True)
    plt.show()

# shows how many nodes converge in a certain number of iterations 
# for the sake of clarity in the plot nIterations are 
# collected in bins of 50
def plot_score_vs_nIterations(n_iter_no_dn, rnk_no_dn):
    plt.figure(figsize=(10,5))
    plt.scatter(n_iter_no_dn, rnk_no_dn.flatten(), s=20, alpha=0.7, color='blue')
    plt.title("PageRank score vs Convergence Iterations")
    plt.xlabel("Iterations to converge")
    plt.ylabel("Final PageRank value")
    plt.grid(True)
    plt.show()

# shows how many nodes converge within n iterations
# for the sake of clarity in the plot nIterations are 
# collected in bins of 50
def plot_nIterations_vs_nNodes_cumulative_(n_iter_no_dn):
    plt.figure(figsize=(10,5))
    plt.hist(n_iter_no_dn, bins=50, cumulative=True, color='skyblue', edgecolor='black')
    plt.title("Cumulative convergence of nodes")
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of nodes")
    plt.grid(True)
    plt.show()

# shows correlation between number of iterations to reach convergence
# and magnitude of dinal score
def plot_dense_vs_sparse(rnk_dense, rnk_no_dn, total_sites):
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(total_sites), rnk_dense.flatten(), 'o-', label="Dense PageRank")
    plt.plot(np.arange(total_sites), rnk_no_dn.flatten(), 'x-', label="Sparse PageRank")
    plt.xlabel("Node index")
    plt.ylabel("PageRank score")
    plt.title("Dense vs Sparse PageRank")
    plt.legend()
    plt.grid(True)
    plt.show()

# shows the distribution of rank values among pages
# for the sake of clarity in the plot scores are 
# collected in bins of 50
def plot_score_distribution(rnk_sparse):
    plt.figure(figsize=(10,5))
    plt.hist(rnk_sparse.flatten(), bins=50, color='skyblue', edgecolor='black')
    plt.xlabel("PageRank score")
    plt.ylabel("Number of nodes")
    plt.title("Distribution of PageRank scores")
    plt.grid(True)
    plt.show()

# displays convergence rate for csr representation
def plot_convergence_rate(rate):
    plt.figure(figsize=(10,5))
    plt.plot(rate, '-o')
    plt.yscale("log") # NEEDED otherwise plot doesn't show significative values
    plt.xlabel("Iteration count")
    plt.ylabel("||x_k - x_(k-1)||")
    plt.title("Convergence per iteration")
    plt.legend()
    plt.grid(True)
    plt.show()

def separate():
    print('\n------------------------------------------------------------------')