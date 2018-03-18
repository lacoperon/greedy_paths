'''
Elliot Williams
March 2018
Effect of graph perturbation on routing @ various levels of 'greed'
'''

import networkx as nx
import math
import random

'''
This function contains the simulation logic for the effect of perturbation on
routing at greed level num_lookahead, average degree k (if not grid)
'''
def perturb_sim(num_lookahead, k, graph_type, N=1000, dim=2, STEP=0.01,
                radius = 0.01):


        print("Running {} graph sim for N:{}, num_lookahead:{}, dim:{}".format(
                    graph_type, N, num_lookahead, dim))

        assert graph_type in ["geometric", "lattice", "hyberbolic"]
        if graph_type == "geometric":
            # create a geometric graph, G
            radius = math.sqrt(k/(math.pi*N))
            G = nx.random_geometric_graph(n=1000, radius=radius,
                                          dim=dim, pos=None)
            return G

        if graph_type == "lattice":
            print(">> Note that a grid graph's average degree isn't malleable")
            grid_input = [int(N ** (1. / float(dim)))] * dim
            actual_N = reduce(operator.mul, grid_input)
            G = nx.grid_graph(, periodic=False) # keep it undirected
            return G

        if graph_type == "hyperbolic":
            raise Exception("Hyperbolic graphs are not yet implemented")

        # also ,random hyperbolic graphs -- bc embed real-world networks well

        N = G.number_of_nodes()

        f = 0
        while f<1:
            # remove STEP nodes from the network according to some strategy

            # random removal
            G.remove_nodes_from(random.sample(G.nodes(),int(STEP*N)))

            # localized attack
            random_node = random.choice(G.nodes())
            nodes_removed = 0
            while nodes_removed < int(STEP*N):
                pass
                # remove random_node, its neighbors, their neighbors... until guard is False.

            # shortest path attack
            # iteratively pick two nodes and remove all nodes on the shortest path between them
            # note that we might remove slightly more than STEP*N nodes.
            # that's fine, just make sure to update f accordingly

            # betweenness centrality attack - remove nodes according to their BC rank
            # (non adaptively, i.e. rank is computed in the beginning and never changes)

            # another attack idea: move nodes around

            f = f + nodes_removed/float(N)

            # iteratively, randomly pick source and targer and record:
            # 1. average length of successful greedy(k) paths
            # 2. average sucess rate


if __name__ == "__main__":
    perturb_sim(num_lookahead=2, k=2, graph_type="geometric",
                N=1000, dim=2, STEP=0.01)
