'''
Elliot Williams
March 2018
Effect of graph perturbation on routing @ various levels of 'greed'
'''

import networkx as nx
import math
import random
import networkit as nk
from functools import reduce
import operator



def generateGraph(graph_type, N, k=2, dim=2):

    assert graph_type in ["geometric", "lattice", "hyberbolic"]

    # generating geometric networks is quite slow -- remember that
    if graph_type == "geometric":
        print("Generating {} graph with N={}, k={},".format(graph_type, N, k),
              "dim={}".format(dim))
        assert dim in [1, 2, 3]
        # create a geometric graph, G
        if dim == 1:
            radius = k / (2 * N)
        if dim == 2:
            radius = math.sqrt(k/(math.pi*N))
        if dim == 3:
            radius = (3 * k / (4 * N * math.pi)) ** (1. / 3)
        # if we wanted to generalize, we could by https://en.wikipedia.org/wiki/N-sphere
        # (but it's of little practical value to generate networks in N-spheres)
        print("N={}, ")
        G = nx.random_geometric_graph(n=N, radius=radius,
                                      dim=dim, pos=None)
        return G

    if graph_type == "lattice":
        print("Generating {} graph with N={}, dim={}".format(graph_type, N, dim))
        print(">> Note that a grid graph's average degree isn't malleable")
        grid_input = [int(N ** (1. / dim))] * dim
        print(grid_input)
        actual_N = reduce(operator.mul, grid_input)
        print("There are actually {} nodes".format(actual_N))
        G = nx.grid_graph(grid_input, periodic=False) # keep it undirected
        return G

    if graph_type == "hyperbolic":
        raise Exception("Hyperbolic graphs are not yet implemented")
        # http://parco.iti.kit.edu/looz/attachments/publications/HyperbolicGenerator.pdf
        # https://github.com/kit-parco/networkit/blob/19005a18180d227f3306b3d71c4ca8901b420a5b/networkit/cpp/generators/HyperbolicGenerator.cpp
        # (Having said that, it doesn't )
        # https://networkit.iti.kit.edu/


    # also ,random hyperbolic graphs -- bc embed real-world networks well

'''
This function contains the simulation logic for the effect of perturbation on
routing at greed level num_lookahead, average degree k (if not grid)
'''
def perturb_sim(num_lookahead, k, graph_type, perturb_strategy,
                N=1000, dim=2, STEP=0.01):

        print("Running {} graph sim for N:{}, num_lookahead:{}, dim:{}".format(
                    graph_type, N, num_lookahead, dim))

        G = generateGraph(graph_type, N, k, dim)

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

            # iteratively, randomly pick source and target and record:
            # 1. average length of successful greedy(k) paths
            # 2. average sucess rate


if __name__ == "__main__":
    perturb_sim(num_lookahead=2, k=2, graph_type="geometric",
                N=1000, dim=2, STEP=0.01)
