import networkx as nx
import random
import math

'''
Input:  Two nodes (where nodes are represented by tuples of dimension length)
Output: The Manhattan distance between the two nodes
'''
def lattice_dist(node1, node2):
    return sum((abs(b-a) for a,b in zip(node1,node2)))

'''
This function computes the 'greedy' route between two nodes
(ie like the Kleinberg (2000) navigable small world networks paper)

Input:   G    : networkx graph object,
         src  : source node in G (a dim length tuple),
         trg  : target node in G (a dim length tuple)

Output:  steps_count : int (number of steps in the greedy path computed),
         path        : node tuple (the nodes along the greedy path computed)
'''
def compute_greedy_route(G, src, trg):
    steps_count = 0
    path = []
    curr_node = src

    while curr_node != trg:
        path.append(curr_node)

        # find the neighbor with minimum grid distance from the target
        min_d, min_nei = -1 , -1
        for nei in G.neighbors(curr_node):
            curr_d = lattice_dist(nei,trg)
            if min_d == -1 or curr_d < min_d:
                min_d = curr_d
                min_nei = nei

        curr_node = min_nei
        steps_count += 1

    path.append(trg)
    return steps_count, path

def compute_not_so_greedy_route(G, src, trg):
    # here you are allowed to look at your neighbors' neighbors
    pass

if __name__ == '__main__':

    N = 100

    random.seed(1)

    # create 2D lattice
    G = nx.grid_graph([math.sqrt(N),math.sqrt(N)], periodic=False)
    print compute_greedy_route(G, G.nodes()[random.randint(0,N)], G.nodes()[random.randint(0,N)])

    # add shortcuts according to some rule
    # take a look at https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/generators/geometric.html#navigable_small_world_graph

    # see how greeedy paths are doing (compared to actual shortest paths)
    # compare paths generated to actual shortest paths (nx implements this)

    # see how not-so-greedy path ares doing

    # Start based on line graph; try to recover the results,
    # then try to generalize to more dimensions

    # i.e. optimal alpha should be 1 for linear,
    #                              2 for 2D lattice, etc

    # For each node, with probability p, add a shortcut based on the rules
    # we have n-2 nodes (linear case) we're not connected to, how do we choose?
    # get a bunch of distances, apply the ** - alpha, pick a shortcut proportional
    # to value in the rs_alpha array, then run this simulation for many alphas
    # and plot (Kleinberg has p=1, so try with this first)
