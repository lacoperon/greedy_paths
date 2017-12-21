import networkx as nx
import random
import math
import operator

# TODO: Try to code up a way to check if we're 'stuck' in local regions,
#       although I don't think that's technically possible for unperturbed lattices
#       (it is, however, for perturbed lattices)

'''
Input:  Two nodes (where nodes are represented by tuples of dimension length)
Output: The Manhattan distance between the two nodes
'''
def lattice_dist(node1, node2):
    if isinstance(node1, int):
        assert isinstance(node2, int)
        return abs(node2 - node1)
    else:
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

        # cur_node = path_taken[1]
        # path    += path_taken[1] # I THINK -- TODO: Check this

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
    chosen_index = random.sample(pos_path_indices, 1)[0] # TODO: return them all, and then run on them all (maybe)
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
        print "There are " +str(len(new_edges))+ " edges added by add_shortcuts"
        G.add_edges_from(new_edges)
        print new_edges

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
        if cdf_before_i <= rand_val and cdf_after_i >= rand_val:
            return not_neighbors[i]

        cdf_before_i = cdf_after_i
        i += 1

    return not_neighbors[-1]


if __name__ == '__main__':

    N = 100
    dim = 1

    assert isinstance(dim, int) and dim > 0
    assert isinstance(N,   int)

    grid_input = [int(N ** (1. / float(dim)))] * dim
    actual_N = reduce(operator.mul, grid_input)

    assert actual_N <= N

    if actual_N != N:
        print("********\n The "+str(dim)+" root of N is not an int\n********")


    G = nx.grid_graph(grid_input, periodic=False)
    G = G.to_directed()

    random.seed(1)
    G = add_shortcuts(G)


    random.seed(1)
    # randomly selects source, target nodes from G
    src_index = random.randint(0,actual_N)
    trg_index = random.randint(0,actual_N)
    src = G.nodes()[src_index]
    trg = G.nodes()[trg_index]


    print "Routing from " + str(src) + " to " + str(trg)



    random.seed(1)
    print "Greedy route is: "
    print compute_not_so_greedy_route(G, src, trg, num=1)
    # print compute_greedy_route(G, src, trg)
    random.seed(1)
    print "Not so greedy route is: "
    print compute_not_so_greedy_route(G, src, trg, num=2)


    # see how greedy paths are doing (compared to actual shortest paths)
    # compare paths generated to actual shortest paths (nx implements this),
    # and then see that compared for various small n, alpha, p

    # see how the not-so-greedy path are doing

    # Start based on line graph; try to recover the results,
    # then try to generalize to more dimensions

    # i.e. optimal alpha should be 1 for linear,
    #                              2 for 2D lattice, etc

    # For each node, with probability p, add a shortcut based on the rules
    # we have n-2 nodes (linear case) we're not connected to, how do we choose?
    # get a bunch of distances, apply the ** - alpha, pick a shortcut proportional
    # to value in the rs_alpha array, then run this simulation for many alphas
    # and plot (Kleinberg has p=1, so try with this first)

    # TODO: (21st December, 2017)

    # Count to see the number of weird cases -- where the num=k case leads to a
    # shorter path when compared to the num=k+1 case, and insodoing see how the
    # graph will potentially change.

    # TODO: The greedy routing should only take one step at a time, even if
    #       it looks out to more steps -- but should only take one step.
    #       FIX THIS (potential fix is written down, but check this)

    # Two types of effects -- true effect (IE k=2 is slower, deterministically,
    # as compared to k=1) -- vs. "bug" effect, which is possiblity random choices
    # make 'large' (or any) differences in the shortest path length computed
    # (bug effect is only for d>1)

    # Thing to do: Test for true effect
    # Randomize graph (ie change seed to add shortcuts)
    # AND Randomize src and trg (ie change seed again)
    # AND Randomize seed to compute_not_so_greedy_route
    # (ie. change which random paths chosen at a given point )
    #
    # loop on build graph:
    #     loop on src, trg:
    #         loop on routing

    # for i in range(arbitrary_num): # maybe 25
    #     random.seed(i+1)
    #     G = buildGraph(N, alpha, p=1)
    #     for j in range( arbitrary_frac * O(N^2) ): #some sample of possible pairs (0.01)
    #         src, trg = chooseSrcTrg()
    #         for k in range(something? (maybe 2))
    #             route(G, src, trg, num)

    # Connection w how good is an embedding

    # try to count # of backwards steps in a path, for each path
    # also:   * # steps for k = 1
            # * # steps for k = 2
            # * ...
            # * # backwards steps for k > 1 (separated by case)
            # * shortest path length for the src, trg
            # * # shortcuts taken (k > 1)
            # * # shortcuts taken (k = 1)
            # * # shortcuts taken in shortest path

            # instead of sampling pairs, just sample src maybe and use all such
            # targets maybe



# curve w # steps, alphas, w avg + std
