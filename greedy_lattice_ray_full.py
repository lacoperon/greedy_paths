import networkx as nx
import random
import math
import operator
import csv
import os
import glob
import threading
import time
import sys
import copy
import ray
import functools

# TODO: Could probably make this faster using dynamic programming on each graph,
#       Saving each node's distance to trg, and maybe also each node's neighbors

def node_dist(node1, node2, G=None, graph_type="lattice", positions=None):
    assert graph_type in ["lattice", "geometric", "hyperbolic"]
    if graph_type == "lattice":
        return manhattan_dist(node1, node2)
    if graph_type == "geometric":
        node1_loc = positions[node1]
        node2_loc = positions[node1]
        return manhattan_dist(node1_loc, node2_loc)

'''
Input:  Two nodes (where nodes are represented by tuples of dimension length)
Output: The Manhattan distance between the two nodes
'''
def manhattan_dist(node1, node2):
    if isinstance(node1, int):
        assert isinstance(node2, int)
        return abs(node2 - node1)
    else:
        return sum((abs(b-a) for a,b in zip(node1,node2)))

def initialize_dv_counter():
    dv_counter = {
        "e_x"  : 0,
        "e_x2" : 0,
        "n"    : 0, # number of successful routes
        "fail" : 0  # number of failed routes
    }
    return dv_counter

def update_dv_counter(datapoint, dv_counter):
    if datapoint is None:
        dv_counter["fail"] += 1

    if dv_counter["n"] is 0:
        dv_counter["e_x"] = datapoint
        dv_counter["e_x2"] = datapoint ** 2
    else:
        n = dv_counter["n"]
        # Updates the expected value of X
        curr_e_x = dv_counter["e_x"]
        dv_counter["e_x"] = curr_e_x * (float(n) / (n+1)) + (datapoint / float(n+1))
        # Updates the expected value of x ** 2
        curr_e_x2 = dv_counter["e_x2"]
        dv_counter["e_x2"] = curr_e_x2 * (float(n) / (n+1)) + ( (datapoint ** 2.) / (n+1))

    dv_counter["n"] += 1

    # print(dv_counter)
    return dv_counter

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

# '''
# The compute_greedy_route function computes the 'greedy' route between two nodes,
# where you can 'look ahead' to only your neighbors
# (ie like the Kleinberg (2000) navigable small world networks paper)
#
# Input:   G    : networkx graph object,
#          src  : source node in G (a dim length tuple),
#          trg  : target node in G (a dim length tuple)
#
# Output:  steps_count : int (number of steps in the greedy path computed),
#          path        : node tuple (the nodes along the greedy path computed)
# '''
# def compute_greedy_route(G, src, trg):
#     steps_count = 0
#     path = []
#     cur_node = src
#
#     while cur_node != trg:
#         path.append(cur_node)
#
#         # find the neighbor with minimum grid distance from the target
#         min_d, min_nei = -1 , -1
#         for nei in G.neighbors(cur_node):
#             curr_d = manhattan_dist(nei,trg)
#             if min_d == -1 or curr_d < min_d:
#                 min_d = curr_d
#                 min_nei = [nei]
#
#             if curr_d == min_d:
#                 min_nei.append(nei)
#
#         # randomly selects greedy node to choose from
#         # Note: this is naive; could lead to going in circles for certain
#         #       edge cases (should deal with later -- w a set, perhaps)
#         cur_node = random.sample(min_nei,1)[0]
#         steps_count += 1
#
#     path.append(trg)
#     return steps_count, path

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
def compute_not_so_greedy_route(G, src, trg, num=1, graph_type="lattice"):

    assert num > 0 and isinstance(num, int)

    path = [src]
    cur_node = src
    rep_count = 0

    if graph_type == "geometric":
        positions = nx.get_node_attributes(G, 'pos')
    else:
        positions = None

    while cur_node != trg:
        pos_greedy_paths = get_pos_ns_greedy_paths(G, cur_node, trg, num)
        path_taken = select_ns_greedy_path(G, pos_greedy_paths, trg, path, graph_type, positions)
        if not path_taken:
            return None, None
        cur_node = path_taken[1]
        if cur_node in path:
            # print("Failure to route!!!")
            # print(cur_node, path)
            return None, None

        path += [cur_node]

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
def select_ns_greedy_path(G, pos_paths, trg, path, graph_type, positions):
    pos_path_end_nodes = [x[-1] for x in pos_paths]
    # print(len(pos_path_end_nodes))

    pos_path_dists = list(map(lambda x : node_dist(x, trg, G, graph_type, positions),
                            pos_path_end_nodes))
    if len(pos_path_dists) == 0:
        return None
    min_dist = min(pos_path_dists)
    pos_path_indices = range(len(pos_path_dists))

    # list of possible indices corresp. to best 'not so greedy' paths
    pos_path_indices = list(filter(lambda x : pos_path_dists[x]==min_dist,
                              pos_path_indices))
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
            print("There are " +str(len(new_edges))+ " edges added by add_shortcuts")
        G.add_edges_from(new_edges)
        if verbose:
            print(new_edges)

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
# Just in case this becomes undirected -- be sure that shortcuts aren't
# duplicated, because that shouldn't be allowed
def choose_shortcut_partner(G, node, alpha):
    alpha = float(alpha)
    nodes = G.nodes()
    nei_set = set(G.neighbors(node))
    nei_set.add(node) #note that we also don't want a self-loop shortcut
    # All nodes that are not the chosen node's neighbors
    not_neighbors = list(filter(lambda x : x not in nei_set, nodes))
    # Distance between the chosen node and its 'not neighbors'

    not_neighbor_dists = map(lambda x : manhattan_dist(x, node),
                             not_neighbors)
    # Note that, according to Kleinberg (2000)'s logic, the probability
    # of connecting the chosen_node to a node in not neighbors should
    # be proportional to r ** - alpha

    prop_to_partner_prob = list(map(lambda x : x ** (- alpha),
                               not_neighbor_dists))
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
        dist1 = node_dist(path[i],trg)
        dist2 = node_dist(path[i+1],trg)
        if dist1 < dist2:
            num_backwards_steps += 1

    return num_backwards_steps

@ray.remote
def runRoute(graph_key, actual_N, numMax):

    G = nx.read_gpickle("ray_graphs/graph_{}".format(graph_key))

    # randomly selects src, trg from G.nodes() WITH REPLACEMENT
    src_index = random.randint(0,actual_N-1)
    trg_index = random.randint(0,actual_N-1)
    src = G.nodes()[src_index]
    trg = G.nodes()[trg_index]

    results = {}
    num_tries = 1

    for attemptNum in range(num_tries):
        for num in range(1,numMax+1):
            result = compute_not_so_greedy_route(G,src,trg,num=num)
            pathlength = result[0]
            num_shortcuts = shortcuts_taken(result[1])
            num_backsteps = calc_num_backwards_steps(G, result[1], trg)
            results[str(num)] = [pathlength, num_shortcuts, num_backsteps]

    return results


@ray.remote
def runSim(numMax, grid_input, p, alpha, pair_frac, actual_N, scale_val):
    #initializing dict columns (maybe this should be its own function)
    dvc_pathlength = [initialize_dv_counter() for i in range(numMax)]
    dvc_shortcuts  = [initialize_dv_counter() for i in range(numMax)]
    dvc_backsteps  = [initialize_dv_counter() for i in range(numMax)]

    verbose = False
    num_tries = 1

    G = nx.grid_graph(grid_input, periodic=False)
    G = G.to_directed()
    G = add_shortcuts(G, p=p, alpha=alpha, verbose=verbose)


    graph_key = random.getrandbits(100)
    graph_path = "ray_graphs/graph_{}".format(graph_key)
    nx.write_gpickle(G, graph_path)

    if verbose:
        print ("Running with "
                + str(int(pair_frac * actual_N))
                + " pairs of nodes")

    runs = ray.get([runRoute.remote(graph_key, actual_N, numMax) for i in range(int(pair_frac * scale_val))])
    print(runs[1:10])

    return

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
Output: TBD (should be void, need to write code to output various desired
             measures for each run to a .csv file)
'''

@ray.remote
def runSimulation(N=100, dim=1, num_graph_gen=25, pair_frac=0.01,
                 num_tries=1, verbose=False, alpha=2., p=1, numMax=2):

    print("Running sim for N:{}, alpha:{}, p:{}".format(N, alpha, p))
    grid_input = [int(N ** (1. / float(dim)))] * dim
    actual_N = functools.reduce(operator.mul, grid_input)
    assert isinstance(dim, int) and dim > 0
    assert isinstance(N,   int)
    assert actual_N <= N
    scale_val = grid_input[0]

    print("Running simulation on " + str(grid_input) + " lattice")

    if actual_N != N:
        print("********\n The "+str(dim)+" root of N is not an int\n********")

    # Running sim for a number of graphs
    results = ray.get([runSim.remote(numMax, grid_input, p, alpha, pair_frac, actual_N, scale_val) for i in range(num_graph_gen)])

    return

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
~~~ THE ACTUAL SIMULATION RUN CODE STARTS HERE ~~~
'''

if __name__ == '__main__':

    ray.init()

    # start = time.time()

    # Removes prior run files
    files = glob.glob('./data_output/*.csv')
    for f in files:
        os.remove(f)

    # Simulation Parameters
    random.seed(1)
    n = int(sys.argv[1])
    ns = [n]
    dim = int(sys.argv[2])
    # generates range of values
    # ie generate_range([0,3], 7) returns [0., 0.5, 1., 1.5, 2., 2.5, 3.]
    all_alphas = sys.argv[4]
    if all_alphas == "True":
        alphas = generate_range([0,3],10)
    else:
        alphas = [2.]
    ps     = [1]
    num_lookahead = int(sys.argv[3]) # IE number of 'links' we look out (IE 1 is greedy)
    num_graph_gen = 25

    result_values = []

    start = time.time()

    run_values = []
    for N in ns:
        for alpha in alphas:
            for p in ps:
                run_values.append([N, alpha, p])

    # Note: may just be faster to run in for loop -- test this
    results = ray.get([runSimulation.remote(N, dim, num_graph_gen, 0.1, 1, False, alpha, p, num_lookahead) for N, alpha, p in run_values])

    end = time.time()
    print("Ran in time: {} ".format(end-start))

    with open("timing_ray.csv", "a") as tfile:
        w = csv.writer(tfile)
        w.writerow([n, start-end])
                # # Prints out results to csv file
                # for i in range(len(result)):
                #     for k in range(num_lookahead):
                #         e_x = result[i][k]["e_x"]
                #         std = math.sqrt(result[i][k]["e_x2"] - (result[i][k]["e_x"])**2.)
                #         label = resultLabels[i]
                #         result_value += [e_x, std]
                #         print("Expected Value of {0}: {1} for {2}".format(label, e_x, k))
                #         print("Std Dev of {0}: {1} for {2}".format(label, std, k))
                # result_values.append(result_value)

    # with open('results_N_{}_dim_{}_k_{}.csv'.format(n, dim, num_lookahead), 'wb') as csvfile:
    #     w = csv.writer(csvfile)
    #     header_rows = ["N", "alpha"]
    #     k_invariant_prefixes = ['E<T>', 'S<T>', 'E(short)', 'S(short)', 'E(back)', 'S(back)']
    #     for num in range(1, num_lookahead+1):
    #         prefixes = list(map(lambda x : x + "k={}".format(num), k_invariant_prefixes))
    #         header_rows += prefixes
    #     w.writerow(header_rows)
    #     for row in result_values:
    #         w.writerow(row)
