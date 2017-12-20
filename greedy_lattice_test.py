import networkx as nx
import random
import math

# TODO: Try to code up a way to check if we're 'stuck' in local regions,
#       although I don't think that's technically possible for unperturbed lattices
#       (it is, however, for perturbed lattices)

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
    cur_node = src

    while cur_node != trg:
        path.append(cur_node)

        # find the neighbor with minimum grid distance from the target
        min_d, min_nei = -1 , -1
        for nei in G.neighbors(cur_node):
            curr_d = lattice_dist(nei,trg)
            if min_d == -1 or curr_d < min_d:
                min_d = curr_d
                min_nei = [nei]

            if curr_d == min_d:
                min_nei.append(nei)

        # randomly selects greedy node to choose from
        # Note: this is naive; could lead to going in circles for certain
        #       edge cases (should deal with later -- w a set, perhaps)
        cur_node = random.sample(min_nei,1)[0]
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
def compute_not_so_greedy_route(G, src, trg, num=1):

    assert num > 0 and isinstance(num, int)
    assert src != trg

    path = [src]
    cur_node = src

    while cur_node != trg:
        pos_greedy_paths = get_pos_ns_greedy_paths(G, cur_node, trg, num)
        path_taken = select_ns_greedy_path(G, pos_greedy_paths, trg)
        cur_node = path_taken[-1]
        path += path_taken[1:]

    return len(path)-1, path

'''
The get_pos_ns_greedy_paths outputs all possible paths, and whether the path
contains trg. Helper function for compute_not_so_greedy_route.

Input:   G        : networkx graph object,
         cur_node : current node in G (a dim length tuple),
         trg      : target node in G  (a dim length tuple),
         num      : int (# of links to look out in the not_so_greedy search)

Output:  pos_paths : node tuple (possible paths num steps away,
                                 or if trg is encountered, path to trg),
'''
def get_pos_ns_greedy_paths(G, cur_node, trg, num):
    # Counter for number of neighborhoods looked out
    k = 0
    # list of all kth-step paths (stored in a tuple) considered
    kth_paths = [ [cur_node] ]
    # set with all previously considered/visited nodes
    already_visited = set()
    while k != num:
        # adds all of the previously greedily-visited nodes to the
        # already_visited set
        end_nodes = [x[-1] for x in kth_paths]

        already_visited.update(end_nodes)
        new_kth_paths = []
        for j in range(len(end_nodes)):
            current_path = kth_paths[j]
            current_node = end_nodes[j]
            current_neighbors = G.neighbors(current_node)

            # end condition
            if trg in current_neighbors:
                current_path.append(trg)
                return [current_path]

            # List of neighbors, filtered to include only those not seen
            # TODO: Parallelize
            filt_neighbors = filter(lambda x : x not in already_visited,
                                current_neighbors)

            new_paths = []
            # Goes through all of the possible neighbours, adds them to the
            # possible paths considered for the next round of greedy search
            for nei in filt_neighbors:
                new_path = current_path + [nei]
                new_kth_paths.append(new_path)

        kth_paths = new_kth_paths
        k += 1

    return kth_paths

'''
The select_ns_greedy_path is a helper function, which chooses the
'not so greedy' path to take (based on minimizing Manhattan distance to trg)
Input:   G        : networkx graph object,
         pos_paths: possible greedy paths towards the target
         trg      : target node in G (a dim length tuple),

Output:  path_taken : path from cur_node ... trg taken
'''
def select_ns_greedy_path(G, pos_paths, trg):
    pos_path_end_nodes = [x[-1] for x in pos_paths]
    # TODO: Parallelize
    pos_path_dists = map(lambda x : lattice_dist(x, trg),
                            pos_path_end_nodes)
    min_dist = min(pos_path_dists)
    pos_path_indices = range(len(pos_path_dists))

    # list of possible indices corresp. to best 'not so greedy' paths
    # TODO: Parallelize (although this is not critical)
    pos_path_indices = filter(lambda x : pos_path_dists[x]==min_dist,
                              pos_path_indices)
    # randomly selects one such 'not so greedy' paths
    chosen_index = random.sample(pos_path_indices, 1)[0]
    path = pos_paths[chosen_index]

    return path


'''
This function picks nodes to gain 'shortcuts', and picks the node which is on
the other side of that shortcut, according to some rules.

Input: G : networkx graph object we're trying to perturb with shortcuts
       p : probability that a given node gets a shortcut
       alpha : float, constant for generating node partners)
       mode  : str, what rules are followed for picking shortcut node partners
               (default is 'oneforall',
               (IE each node has probability p of having a shortcut))

Output: G' : updated (ie perturbed) networkx graph object
'''
# TODO: Test
def add_shortcuts(G, p=1, alpha=2., mode="oneforall"):
# Note: The linked function assumes directed shortcuts, and grid_graph does
#       generate directed shortcuts, so an assumption of directionality is made
# add shortcuts according to some rule
# take a look at https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/generators/geometric.html#navigable_small_world_graph
    assert mode in ["oneforall"]
    assert p <= 1 and p > 0 # p=0 is uninteresting

    if mode == "oneforall":
        nodes = G.nodes()
        # Each node is chosen (or not) to have a shortcut based on a
        # Bernoulli trial (TODO: Parallelize)
        chosen_nodes = filter(lambda x : random.random() < p, nodes)

        # TODO: Parallelize (have parallel map generate a list of new edges from
        #                    list of chosen nodes, and only then add to G )

        new_edges = map(lambda x: (x, choose_shortcut_partner(G,x,alpha)),
                                   chosen_nodes)
        print new_edges #currently contains NoneTypes, so we should fix this
        print len(new_edges)
        G.add_edges_from(new_edges)

    return G

'''
Helper function for add_shortcuts. Chooses shortcut partner according to
Kleinberg's rules
ie Pr(node u being shortcut) ~ dist(chosen_node, u) ** (- alpha)

Input : G     : networkx graph,
        node  : chosen_node, for which we're trying to find a shortcut partner,
        alpha : alpha value used for prob. calculation

Output: partner_node : node's partner for the shortcut
'''
# TODO: Test
def choose_shortcut_partner(G, node, alpha):
    nodes = G.nodes()
    nei_set = set(G.neighbors(node))
    nei_set.add(node) #note that we also don't want a self-loop shortcut
    # TODO: Parallelize
    # All nodes that are not the chosen node's neighbors
    not_neighbors = filter(lambda x : x not in nei_set, nodes)
    # Distance between the chosen node and its 'not neighbors'
    # TODO: Parallelize
    not_neighbor_dists = map(lambda x : lattice_dist(x, node),
                             not_neighbors)
    # Note that, according to Kleinberg (2000)'s logic, the probability
    # of connecting the chosen_node to a node in not neighbors should
    # be proportional to r ** - alpha
    # TODO: Parallelize
    prop_to_partner_prob = map(lambda x : x ** (- alpha),
                               not_neighbor_dists)
    total = sum(prop_to_partner_prob)
    rand_val = total * random.random()

    i = 0
    cdf_before_i = 0
    while i < len(prop_to_partner_prob):
        cdf_after_i = cdf_before_i + prop_to_partner_prob[i]
        if cdf_before_i <= rand_val and cdf_after_i <= rand_val:
            return not_neighbors[i]

        cdf_before_i = cdf_after_i
        i += 1


if __name__ == '__main__':

    # TODO: Generalize this code (maybe inside a function) to generate a
    #       lattice with an arbitrary dimension

    N = 100

    random.seed(1)

    # create 2D lattice
    G = nx.grid_graph([int(math.sqrt(N)),int(math.sqrt(N))], periodic=False)

    # randomly selects source, target nodes from G
    src_index = random.randint(0,N)
    trg_index = random.randint(0,N)

    random.seed(1)
    print "Greedy route is: "
    print compute_greedy_route(G, G.nodes()[src_index], G.nodes()[trg_index])
    random.seed(1)
    print "Not so greedy route is: "
    print compute_not_so_greedy_route(G, G.nodes()[src_index],
                                      G.nodes()[trg_index],num=2)

    add_shortcuts(G)


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
