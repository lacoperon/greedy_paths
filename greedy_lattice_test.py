import networkx as nx
import random
import math
import operator
import csv
import os
import glob

# TODO: Try to code up a way to check if we're 'stuck' in local regions,
#       although I don't think that's technically possible for unperturbed lattices
#       (it is, however, for perturbed lattices where edges are removed)
#       -- it won't be for lattices where edges are added, because it's not for
#       regular lattices

# TODO: Could probably make this faster using dynamic programming on each graph,
#       Saving each node's distance to trg, and maybe also each node's neighbors

# TODO: Parallelization didn't go so well with Pool.map, but there should be an
#       easy way to get that done...
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

# Test cases
assert lattice_dist(1,2) is 1
assert lattice_dist(13,13) is 0
assert lattice_dist((1,2,3),(4,5,6)) is 9
assert lattice_dist((1,2,3),(1,1,1)) is 3
assert lattice_dist((1,2,3,4,5),(1,2,3,4,5)) is 0

'''
Function to see if two nodes are, *de novo*, adjacent to each other in a lattice
(IE two nodes are connected prior to pertubation)
Input:
    node1 : node in a given lattice graph
    node2 : another node in a given lattice graph
Output:
    bool (whether or not the two nodes are de novo adjacent in a lattice)
'''
def are_denovo_adj(node1, node2):
    if node1 == node2:
        return False

    if isinstance(node1, int):
        assert isinstance(node2, int)
        return abs(node2 - node1) == 1
    else:
        assert len(node1) == len(node2)
        hasBeenDifferent = False
        for i in range(len(node1)):
            if node1[i] != node2[i]:
                if hasBeenDifferent:
                    return False
                else:
                    if abs(node1[i]-node2[i]) != 1:
                        return False
                    hasBeenDifferent = True
        return True

assert are_denovo_adj(1,2) is True
assert are_denovo_adj(1,1) is False
assert are_denovo_adj(13,15) is False
assert are_denovo_adj((1,1),(1,2))

'''
Function that counts the number of 'shortcuts' taken in a given path from src
to trg.
Input:
    path : node list (list of the nodes taken along the path from src to trg)
Output:
    int (the number of shortcuts taken)
'''
def shortcuts_taken(path):
    step_indices = range(len(path)-1)
    shortcut_vector = [not are_denovo_adj(path[i],path[i+1]) for i in step_indices]
    return sum(shortcut_vector)

path1 = [1,2,3,4,7,8,9]
assert shortcuts_taken(path1) is 1
path2 = [1,2,4,5,7,8,10,11]
assert shortcuts_taken(path2) is 3
path3 = [(1,1),(1,2),(2,2),(17,3),(17,4)]
assert shortcuts_taken(path3) is 1
path4 = []
assert shortcuts_taken(path4) is 0


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

    path = [src]
    cur_node = src

    while cur_node != trg:
        pos_greedy_paths = get_pos_ns_greedy_paths(G, cur_node, trg, num)
        path_taken = select_ns_greedy_path(G, pos_greedy_paths, trg)
        cur_node = path_taken[1]
        path += [path_taken[1]]

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

    pos_path_dists = map(lambda x : lattice_dist(x, trg),
                            pos_path_end_nodes)
    min_dist = min(pos_path_dists)
    pos_path_indices = range(len(pos_path_dists))

    # list of possible indices corresp. to best 'not so greedy' paths
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
def add_shortcuts(G, p=1, alpha=2., mode="oneforall", verbose=False):
# Note: The linked function assumes directed shortcuts, and grid_graph does
#       generate directed shortcuts, so an assumption of directionality is made
# add shortcuts according to some rule
# take a look at https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/generators/geometric.html#navigable_small_world_graph
    assert mode in ["oneforall"]
    assert p <= 1 and p >= 0 # p=0 is uninteresting

    if mode == "oneforall":
        nodes = G.nodes()
        # Each node is chosen (or not) to have a shortcut based on a
        # Bernoulli trial
        chosen_nodes = filter(lambda x : random.random() < p, nodes)
        new_edges = map(lambda x: (x, choose_shortcut_partner(G,x,alpha)),
                                   chosen_nodes)
        if verbose:
            print "There are " +str(len(new_edges))+ " edges added by add_shortcuts"
        G.add_edges_from(new_edges)
        if verbose:
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
    alpha = float(alpha)
    nodes = G.nodes()
    nei_set = set(G.neighbors(node))
    nei_set.add(node) #note that we also don't want a self-loop shortcut
    # All nodes that are not the chosen node's neighbors
    not_neighbors = filter(lambda x : x not in nei_set, nodes)
    # Distance between the chosen node and its 'not neighbors'

    not_neighbor_dists = map(lambda x : lattice_dist(x, node),
                             not_neighbors)
    # Note that, according to Kleinberg (2000)'s logic, the probability
    # of connecting the chosen_node to a node in not neighbors should
    # be proportional to r ** - alpha

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

'''
Function takes in a graph object, and a node list corresponding to a path
from a src node to a trg node. It then calculates the number of 'backwards'
steps taken in the path. I define a 'backwards' step as a step which increases
the Manhattan distance between the current node and the target.

Input:  G : networkx graph object,
        path : node list

Output:
        num_backwards_steps : int (the number of backwards steps taken)
'''
def calc_num_backwards_steps(G, path, trg):
    num_backwards_steps = 0

    for i in range(len(path)-1):
        dist1 = lattice_dist(path[i],trg)
        dist2 = lattice_dist(path[i+1],trg)
        if dist1 < dist2:
            num_backwards_steps += 1

    return num_backwards_steps

'''
Function which runs a number of simulations based on the various parameters
described below.

Input:  N   : int (# of nodes desired in lattice (upper bound)),
        dim : int (dimension of the lattice),
        num_graph_gen : int (number of distinct graphs we want to run sims on),
        pair_frac : float (frac of all node pairs we want to route to/from),
        num_tries : int (# tries for routing (to catch "bug" effect)),
        verbose : bool (whether or not to print out everything),
        alpha : float (what alpha value to use for the sims),
        p : float (probability of a given node having a shortcut added to it),
        SEED : int (seed for the run)
Output: TBD (should be void, need to write code to output various desired
             measures for each run to a .csv file)
'''
def runSimulation(N=100, dim=1, num_graph_gen=25, pair_frac=0.01, printDict=False,
                 num_tries=2, verbose=False, alpha=2., p=1, numMax=2, SEED=1):
    grid_input = [int(N ** (1. / float(dim)))] * dim
    actual_N = reduce(operator.mul, grid_input)
    assert isinstance(dim, int) and dim > 0
    assert isinstance(N,   int)
    assert actual_N <= N

    print "Running simulation on " + str(grid_input) + " lattice"

    if actual_N != N:
        print("********\n The "+str(dim)+" root of N is not an int\n********")

    random.seed(SEED)

    dcd = {} #Dict collecting results

    #initializing dict columns (maybe this should be its own function)
    dcd["shortcutstakenSP"] = []
    dcd["lengthOfShortestPath"] = []
    dcd["backsteps_SP"] = []
    dcd["graphNum"] = []
    dcd["attemptNum"] = []
    for num in range(1,numMax+1):
        dcd["lengthOfPathk="+str(num)] = []
        dcd["backsteps_k="+str(num)] = []
        dcd["shortcutsTakenk="+str(num)] = []

    # Running sim for a number of graphs
    for graph_number in range(num_graph_gen):
        G = nx.grid_graph(grid_input, periodic=False)
        G = G.to_directed()
        G = add_shortcuts(G, p=p, alpha=alpha, verbose=verbose)

        if verbose:
            print ("Running with "
                    + str(int(pair_frac * (actual_N * (actual_N - 1))))
                    + " pairs of nodes")
        for j in range( int(pair_frac * (actual_N * (actual_N - 1)))):
            # randomly selects src, trg from G.nodes() WITH REPLACEMENT
            src_index = random.randint(0,actual_N-1)
            trg_index = random.randint(0,actual_N-1)
            src = G.nodes()[src_index]
            trg = G.nodes()[trg_index]

            actualShortestPath = nx.shortest_path(G, source=src, target=trg)

            for attemptNum in range(num_tries):
                results_at_given_run = []

                # Maybe should add this to a separate function and refactor,
                # because this looks mighty funky

                # Data Collection (Part I)
                dcd["graphNum"] += [graph_number+1]
                dcd["shortcutstakenSP"] += [shortcuts_taken(actualShortestPath)]
                dcd["lengthOfShortestPath"] += [len(actualShortestPath)-1]
                dcd["backsteps_SP"] += [calc_num_backwards_steps(G,
                                        actualShortestPath, trg)]
                dcd["attemptNum"] += [attemptNum]

                for num in range(1,numMax+1):
                    result = compute_not_so_greedy_route(G,src,trg,num=num)

                    # Data collection (Part II)
                    dcd["lengthOfPathk="+str(num)] += [result[0]]
                    dcd["shortcutsTakenk="+str(num)] += [
                        shortcuts_taken(result[1])]
                    dcd["backsteps_k="+str(num)] += [
                        calc_num_backwards_steps(G, actualShortestPath, trg)
                    ]

                    results_at_given_run += [result]

                gr_result   = results_at_given_run[0]
                nsgr_result = results_at_given_run[1]

                if verbose:
                    print "-----------------"
                    print "Attempt Number " + str(attemptNum)
                    print "Routing from " + str(src) + " to " + str(trg)
                    print "Greedy route is: "
                    print gr_result
                    print "Not so greedy route is: "
                    print nsgr_result
                    print "Actual shortest path is"
                    print actualShortestPath
                    print "Actual shortest path is length:"
                    print (len(actualShortestPath) -1)

    if printDict:
        for key in dcd:
            print key
            print len(dcd[key])

    return dcd

'''
Function that generates a range of values, from xmin to xmax, with
steps values evenly between xmin and xmax.
Input:
    xlims : xmin * xmax (ie lower + upper bound of x values)
    steps : int (number of values in the returned values array)
Output:
    values : float array
'''
def generate_range(xlims, steps):
    xmin, xmax = xlims
    stepsize = (xmax - xmin) / (steps - 1.)
    return [stepsize * x for x in range(steps)]

'''
Function which takes in a data collection dictionary, and outputs the data
collected (for that particular run) to csv
Input:    dcd : dict (dict w keys as variable names,
                      lists as values corresponding to the vector representations
                      of those columns)
Output :  void (Should write to .csv though...)
'''
def write_dcd_to_csv(dcd, filename="test.csv"):
    keys = sorted(dcd.keys())
    with open(filename, "wb") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(keys)
        for index in range(len(dcd[keys[0]])):
            writer.writerow([dcd[x][index] for x in keys])

'''
~~~ THE ACTUAL SIMULATION RUN CODE STARTS HERE ~~~
'''
if __name__ == '__main__':

    # Removes leftover files from previous runs
    files = glob.glob('./data_output/*.csv')
    for f in files:
        os.remove(f)

    # Simulation Parameters
    # Please change them here! Otherwise the .csv files will be mislabelled...

    ns = [100, 1000]
    dim = 1
    alphas = generate_range([0,3],7)
    ps     = [1]
    num_lookahead = 2 # IE number of 'links' we look out (IE 1 is greedy)


    # TODO: This is an obvious (obvious) candidate for parallelization
    for N in ns:
        for alpha in alphas:
            print "Running for alpha equal to " + str(alpha)
            for p in ps:
                dcd = runSimulation(N=N, dim=dim, num_graph_gen=1, pair_frac=0.01,
                              printDict=False, num_tries=2, verbose=False,
                              numMax = num_lookahead,
                              alpha=alpha, p=p, SEED=1)
                # print dcd

                # TODO: Maybe output a file which details the simulation params,
                #       instead of storing them all in the filename (subject to change)
                filename = "./data_output/sim_"
                filename += "p_"+str(p)+"_alpha_"+str(alpha)
                filename += "_N_"+str(N)+"_dim_" + str(dim) + ".csv"
                write_dcd_to_csv(dcd, filename= filename)
