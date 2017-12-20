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
The compute_greedy_route function computes the 'greedy' route between two nodes,
where you can 'look ahead' to only your neighbors
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
                min_nei = [nei]

            if curr_d == min_d:
                min_nei.append(nei)

        # randomly selects greedy node to choose from
        # Note: this is naive; could lead to going in circles for certain
        #       edge cases (should deal with later -- w a dict, perhaps)
        curr_node = random.sample(min_nei,1)[0]
        steps_count += 1

    path.append(trg)
    return steps_count, path

'''
The compute_not_so greedy_route function computes the 'greedy' route between
two nodes, where you can 'look ahead' to num iterations of neighbors.
(ie for num=1, this should run identically to compute_greedy_route)

Input:   G    : networkx graph object,
         src  : source node in G (a dim length tuple),
         trg  : target node in G (a dim length tuple),
         num  : int (number of links to look out in the not_so_greedy search)

Output:  steps_count : int (number of steps in the greedy path computed),
         path        : node tuple (the nodes along the greedy path computed)
'''

# TODO: Fix bugs (currently, code doesn't work -- fix this tomorrow!)
#       (lo, we're getting close, I think...)
def compute_not_so_greedy_route(G, src, trg, num=2):

    assert num > 0 and isinstance(num, int)

    path = []
    curr_node = src

    while curr_node != trg:

        '''
        Part I: Generates all possible greedy num-neighbor paths, and puts
                them into the kth_paths list
        '''

        # Counter for number of neighborhoods looked out
        k = 0
        # list of all kth-step paths (stored in a tuple) considered
        kth_paths = [ [curr_node] ]
        # set with all previously considered/visited nodes
        already_visited = set(path)
        while k != num:
            # adds all of the previously greedily-visited nodes to the
            # already_visited set
            end_nodes = [x[-1] for x in kth_paths]
            print end_nodes
            already_visited.update(end_nodes)
            new_kth_paths = []
            for j in range(len(end_nodes)):
                current_path = kth_paths[j]
                current_node = end_nodes[j]
                current_neighbors = G.neighbors(current_node)

                # end condition
                if trg in current_neighbors:
                    current_path.append(trg)
                    path.append(current_path)
                    steps_count = len(path) - 1
                    return steps_count, path

                # List of neighbors, filtered to include only those not seen
                filt_neighbors = filter(lambda x : x not in already_visited,
                                    current_neighbors)

                # Goes through all of the possible neighbours, adds them to the
                # possible paths considered for the next round of greedy search,
                # or in the path-choosing in Part II.
                for nei in filt_neighbors:
                    print current_path
                    new_path = current_path.append(nei)
                    print current_path
                    new_kth_paths.append(new_path)
                    print new_kth_paths


            kth_paths = new_kth_paths
            k += 1
            print kth_paths

        '''
        Part II: Considers all possible num-step paths in kth_paths,
                 and chooses the one that has the lowest Manhattan distance to
                 trg
        '''

        # Note that, at this point, kth_paths is equivalent to all possible
        # greedy (or rather, 'not so greedy') paths under consideration
        ns_greedy_paths = kth_paths
        print ns_greedy_paths
        ns_greedy_path_end_nodes = [x[-1] for x in ns_greedy_paths]
        ns_greedy_path_dists = map(lambda x : lattice_dist(x, trg),
                                ns_greedy_path_end_nodes)
        min_dist = min(ns_greedy_path_dists)
        pos_path_indices = range(len(ns_greedy_paths_dists))

        # list of possible indices corresp. to best 'not so greedy' paths
        pos_path_indices = filter(lambda x : ns_greedy_path_dists[x]==min_dist,
                                  pos_path_indices)
        # randomly selects one such 'not so greedy' paths
        chosen_index = random.sample(pos_path_indices, 1)[0]
        path = ns_greedy_paths[chosen_index]

if __name__ == '__main__':

    N = 100

    random.seed(1)

    # create 2D lattice
    G = nx.grid_graph([int(math.sqrt(N)),int(math.sqrt(N))], periodic=False)
    src = random.randint(0,N)
    trg = random.randint(0,N)
    print compute_greedy_route(G, G.nodes()[src], G.nodes()[trg])
    print compute_not_so_greedy_route(G, G.nodes()[src], G.nodes()[trg],num=1)

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
