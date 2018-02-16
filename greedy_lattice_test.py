import networkx as nx
import random
import math
import operator
import csv
import os
import glob
import threading
import time
from Queue import *
import copy

# TODO: Could probably make this faster using dynamic programming on each graph,
#       Saving each node's distance to trg, and maybe also each node's neighbors

# TODO: Add more coherent print logic, put it in a log file

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
# Just in case this becomes undirected -- be sure that shortcuts aren't
# duplicated, because that shouldn't be allowed
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

def runSimulationMultithread(N=100, dim=1, num_graph_gen=25, pair_frac=0.01, printDict=False,
                 num_tries=2, verbose=False, alpha=2., p=1, numMax=2,
                 NUM_MAX_THREADS = 1, graphInput=False):

    lock = threading.RLock() # lock for a given simulation's data file

    grid_input = [int(N ** (1. / float(dim)))] * dim
    actual_N = reduce(operator.mul, grid_input)
    assert isinstance(dim, int) and dim > 0
    assert isinstance(N,   int)
    assert actual_N <= N

    if verbose:
        print "Running simulation on " + str(grid_input) + " lattice"

    if actual_N != N:
        print("********\n The "+str(dim)+" root of N is not an int\n********")

    dcd = initialize_dcd(numMax)

    # Added ability to feed simulation graphs
    # (TODO: Implement this to test equality of methods)
    if graphInput:
        graph_list = graphInput
    else:
        graph_list = initialize_graphs(num_graph_gen, N, 1, alpha, NUM_MAX_THREADS)

    '''
    The following defines a Thread class that should contain everything required
    to run multithreaded pathfinding
    '''
    class pathfindThread (threading.Thread):
       def __init__(self, threadID, input_tuple):
          threading.Thread.__init__(self)
          self.threadID = threadID
          self.graph_num = input_tuple[0]
          self.G = graph_list[self.graph_num]
          src_index = input_tuple[1]
          self.src = self.G.nodes()[src_index]
          trg_index = input_tuple[2]
          self.trg = self.G.nodes()[trg_index]
          self.numMax = input_tuple[3]
          self.numAttempts = input_tuple[4]

       def run(self):
          if verbose:
               print ("Starting PathFind Thread-{}".format(self.threadID))
          results = []
          for attemptNum in range(self.numAttempts):
              for num in range(1,self.numMax+1):
                  results.append([compute_not_so_greedy_route(self.G,self.src,self.trg,num), num])

              with lock:
                 dcd["attemptNum"] += [attemptNum]
                 dcd["graphNum"] += [self.graph_num+1]
                 for entry in results:
                      result, knum = entry
                      dcd["lengthOfPathk="+str(knum)] += [result[0]]
                      dcd["shortcutsTakenk="+str(knum)] += [
                          shortcuts_taken(result[1])]
                      dcd["backsteps_k="+str(knum)] += [
                          calc_num_backwards_steps(self.G, result[1], self.trg)
                      ]
          if verbose:
              print ("Exiting Thread for Pathfind-{}".format(self.threadID))

    pathfind_queue = Queue()

    for graph_num in range(num_graph_gen):
        if verbose:
            print ("Running with "
                    + str(int(pair_frac * (actual_N * (actual_N - 1))))
                    + " pairs of nodes")
        for j in range( int(pair_frac * (actual_N * (actual_N - 1)))):
            # randomly selects src, trg from G.nodes() with replacement
            src_index = random.randint(0,actual_N-1)
            trg_index = random.randint(0,actual_N-1)
            for attemptNum in range(num_tries):
                pathfind_queue.put([graph_num, src_index, trg_index, numMax, attemptNum])

    i = 1
    while (pathfind_queue.qsize() != 0):
        num_threads = threading.active_count()
        if num_threads < NUM_MAX_THREADS:
            if pathfind_queue.qsize() != 0:
                input_tuple = pathfind_queue.get()
                newThread = pathfindThread(i, input_tuple)
                newThread.start()
                i += 1
                pathfind_queue.task_done()

    pathfind_queue.join()
    if verbose:
        for key in dcd:
            print key
            print len(dcd[key])
    return dcd


'''
Helper function to initialize a data-collecting dictionary
'''
def initialize_dcd(numMax):
    dcd = {}
    # dcd["shortcutstakenSP"] = []
    # dcd["lengthOfShortestPath"] = []
    # dcd["backsteps_SP"] = []
    dcd["graphNum"] = []
    dcd["attemptNum"] = []
    for num in range(1,numMax+1):
        dcd["lengthOfPathk="+str(num)] = []
        dcd["backsteps_k="+str(num)] = []
        dcd["shortcutsTakenk="+str(num)] = []
        # TODO: Add the Furthest we get from the trg in the path (relative to src)
        # IE 5->9 that touches 1 should yield 4
        # TODO: Add the max and sum of backsteps
    return dcd

'''
Helper function for multithreaded initialization of all graphs used in a sim
'''
def initialize_graphs(num_graph_gen, N, alpha, p=1, NUM_MAX_THREADS=1):
    '''
    The following defines a Thread class that should contain everything required
    to run multithreaded graph generation, in so doing allowing for multicore
    operations to succeed.
    '''
    class graphThread (threading.Thread):
       def __init__(self, threadID, input_tuple):
          threading.Thread.__init__(self)
          self.threadID = threadID
          self.graph_num = input_tuple[0]
          self.N = input_tuple[1]
          self.p = input_tuple[2]
          self.alpha = input_tuple[3]
          self.verbose = False

       def run(self):
          # print ("Starting Graph Gen Thread-{}".format(input_tuple))
          G = nx.grid_graph(grid_input, periodic=False)
          G = G.to_directed()
          G = add_shortcuts(G, p=1, alpha=self.alpha, verbose=self.verbose)
          graph_list[self.graph_num] = G
          graph_gen_queue.task_done()
          # print ("Exiting Thread for GraphGen-{}".format(self.graph_num))


    assert NUM_MAX_THREADS > 1 and isinstance(NUM_MAX_THREADS, int)
    grid_input = [int(N ** (1. / float(dim)))] * dim
    actual_N = reduce(operator.mul, grid_input)
    assert isinstance(dim, int) and dim > 0
    assert isinstance(N,   int) and actual_N <= N

    # print "Generating {} graphs for {} lattice, alpha : {}, p: {}".format(
    #     num_graph_gen, N, alpha, p
    #     )
    if actual_N != N:
        print("********\n The "+str(dim)+" root of N is not an int\n********")

    # Initializes list of generated graphs (lists are thread-safe)
    graph_list = [0] * num_graph_gen
    # Initializes Queue used to keep track of graph-generating threads
    graph_gen_queue = Queue()
    for graph_num in range(num_graph_gen):
        graph_gen_queue.put([graph_num, N, p, alpha])

    i = 1
    while (graph_gen_queue.qsize() != 0):
        num_threads = threading.active_count()
        if num_threads < NUM_MAX_THREADS:
            if graph_gen_queue.qsize() != 0:
                input_tuple = graph_gen_queue.get()
                newThread = graphThread(i, input_tuple)
                newThread.start()
                i += 1

    graph_gen_queue.join()
    return graph_list

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
# TODO: Implement this through buffering (or, even better, multithreaded
# row writing with threaded locking -- maybe in batches -- or perhaps
# writing to a database for increased efficiency for map reduce stuff)
def write_dcd_to_csv(dcd, filename="test.csv"):
    print("Touched for " + filename)
    keys = sorted(dcd.keys())
    with open(filename, "wb") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(keys)
        for index in range(len(dcd[keys[0]])):
            writer.writerow([dcd[x][index] for x in keys])

'''
Test simThread for the multithreaded runSimulation
'''
class simThreadTest (threading.Thread):
   def __init__(self, threadID, input_tuple):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.N = input_tuple[0]
      self.alpha = input_tuple[1]
      self.p = input_tuple[2]
      self.NUM_MAX_THREADS = 10
   def run(self):
      print ("Starting Thread-{} for {}".format(self.threadID, input_tuple))
      dcd = runSimulationMultithread(N=self.N, dim=dim, num_graph_gen=1, pair_frac=0.0001,
                                printDict=False, num_tries=2, verbose=False,
                                numMax = num_lookahead,
                                alpha=self.alpha, p=self.p, NUM_MAX_THREADS = self.NUM_MAX_THREADS)
      print ("Writing CSV for Thread-{}".format(self.threadID))
      filename = "./data_output/sim_"
      filename += "p_"+str(self.p)+"_alpha_"+str(self.alpha)
      filename += "_N_"+str(self.N)+"_dim_" + str(dim) + "_test.csv"
      write_dcd_to_csv(dcd, filename= filename)
      print ("Exiting  Thread-{}".format(self.threadID))
      thread_init_queue_test.task_done()

'''
~~~ THE ACTUAL SIMULATION RUN CODE STARTS HERE ~~~
'''
if __name__ == '__main__':

    # Removes prior run files
    files = glob.glob('./data_output/*.csv')
    for f in files:
        os.remove(f)

    # Simulation Parameters
    random.seed(1)
    ns = [10000]
    dim = 2
    # generates range of values
    # ie generate_range([0,3], 7) returns [0., 0.5, 1., 1.5, 2., 2.5, 3.]
    alphas = generate_range([0,3],10)
    ps     = [1]
    num_lookahead = 2 # IE number of 'links' we look out (IE 1 is greedy)
    NUM_MAX_THREADS = 8 # SHOULD OPTIMISE THIS -->
                    # https://stackoverflow.com/questions/481970/how-many-threads-is-too-many
    num_graph_gen = 1
    thread_init_queue_test = Queue()
    graph_list = []

    for N in ns:
        for alpha in alphas:
            for p in ps:
                thread_init_queue_test.put([N, alpha, p])
                graph_list.append([num_graph_gen, N, p, alpha, 5])

    print("Number of graphs is {}".format(len(graph_list)))

    # Use the same graphs for testing -- the graphs look similar, but we want
    # to ensure that the graphs are the same for the same graphs
    i = 1
    while (thread_init_queue_test.qsize() != 0):
        num_threads = threading.active_count()
        if num_threads < NUM_MAX_THREADS:
            if thread_init_queue_test.qsize() != 0:
                input_tuple = thread_init_queue_test.get()
                testThread = simThreadTest(i, input_tuple)
                testThread.start()
                i += 1

    thread_init_queue_test.join()
