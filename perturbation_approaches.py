'''
Elliot Williams
March 2018
Effect of graph perturbation on routing @ various levels of 'greed'
'''

import networkx as nx
import math
import random
import os
if not 'TRAVIS' in os.environ or not os.environ['TRAVIS']:
    import networkit as nk
from functools import reduce
import operator
import greedy_lattice_ray_full as gr
import ray
import matplotlib.pyplot as plt
import csv
import math
import sys


'''
This function generates a NetworkX graph (ie geometric, lattice, hyperbolic)
Input:
    graph_type - type of graph (as above)
    N - number of nodes
    k - average degree
    dim - dimension of graph

Output:
    NetworkX graph representing a graph with said qualities
'''
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
        G = nx.random_geometric_graph(n=N, radius=radius,
                                      dim=dim, pos=None)
        return G

    if graph_type == "lattice":
        print("Generating {} graph with N={}, dim={}".format(graph_type, N, dim))
        print(">> Note that a grid graph's average degree isn't malleable")
        grid_input = [int(N ** (1. / dim))] * dim
        actual_N = reduce(operator.mul, grid_input)
        print("There are actually {} nodes".format(actual_N))
        G = nx.grid_graph(grid_input, periodic=False) # keep it undirected
        return G

    # also ,random hyperbolic graphs -- bc embed real-world networks well
    if graph_type == "hyperbolic":
        raise Exception("Hyperbolic graphs are not yet implemented")
        # http://parco.iti.kit.edu/looz/attachments/publications/HyperbolicGenerator.pdf
        # https://github.com/kit-parco/networkit/blob/19005a18180d227f3306b3d71c4ca8901b420a5b/networkit/cpp/generators/HyperbolicGenerator.cpp
        # (The code to get coords should live in DynamicHyperbolicGenerator,
        # according to Looz (the dude who wrote the paper)
        # https://networkit.iti.kit.edu/

        # HOW MANY NODES DO WE ACTUALLY WANT TO USE?
        # (it might be easier to use this other tool and parse in the results),
        # but it's wayyyy slower if we want to work with hyperbolic graphs
        # in the future -- might be worthwhile doing the networkit work imo

        # https://en.wikipedia.org/wiki/Hyperbolic_geometric_graph

'''
This function plots a graph out using networkx's draw and matplotlib
Input:
    G - NetworkX graph object to graph
    graph_type - type of graph (ie lattice, geometric, hyperbolic)
'''
def plotGraph(G, graph_type):
    if graph_type == "lattice":
        lattice_pos = dict( [ (x,x) for x in G.nodes() ])
        nx.draw_networkx(G, lattice_pos)
        plt.show()

    if graph_type == "geometric":
        geom_pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx(G, geom_pos)
        plt.show()

    if graph_type == "hyperbolic":
        raise Exception("Plotting hyperbolic graphs is currently unimplemented")

@ray.remote
def performRoute(G, num_lookahead, src, trg, graph_type):
    G  = nx.read_gpickle(G)
    result = {}
    
    # Gets shortest path route between src and trg
    
    try:
        result["SP"] = len(nx.shortest_path(G, src, trg)) - 1
    except nx.exception.NetworkXNoPath:
        result["SP"] = None
    
    # Gets ns_greedy_path route \forall num \in num_lookahead 
    for num in range(1, num_lookahead+1):
        length, path = gr.compute_not_so_greedy_route(G, src, trg,
                                                      num, graph_type)
        result[num] = length
    return result


'''
This function removes STEP proportion of all nodes from the network according
to some strategy.

Input:
    G - NetworkX graph object
    N - Number of nodes originally in G
    perturb_strategy - perturbation strategy being used
    f - Counter for proportion of perturbation (f == 1 implies fully perturbed)
    STEP - step size for perturbation
Output:
    G - Perturbed NetworkX graph object
    f - Incremented perturbation counter
'''
def perturbGraph(G, N, perturb_strategy, f, STEP, state=None):
    assert perturb_strategy in ["none", "random", "localized", "localized_expanding_hole"]

    if perturb_strategy == "none":
        return(G, f+STEP, None)

    if perturb_strategy == "random":
        G.remove_nodes_from(random.sample(G.nodes(), int(STEP*N)))
        f += (STEP*N) / N
        return (G, f, None)

    if perturb_strategy == "localized":
        # localized attack
        random_node = random.choice(G.nodes())
        nodes_to_remove = [random_node]
        while len(nodes_to_remove) < int(STEP*N):
            a = len(nodes_to_remove)
            # We definitely want to remove nodes previously selected if
            # there are still nodes to be selected to remove
            nodes_to_def_remove = nodes_to_remove
            # Gets the neighbor nodes from all nodes we're currently removing
            neighbors = [G.neighbors(node) for node in nodes_to_remove]
            neighbors = set(reduce(lambda x,y : x + y, neighbors))
            
            # Gets the new possible total list of nodes to  remove
            nodes_to_remove = list(set(nodes_to_remove) | neighbors)
            
            # Gets the list of new nodes we might remove
            remaining_nodes = list(set(G.nodes()) - set(nodes_to_remove))
            
            # If there are no new neighbors to be had,
            # we break out of the loop
            if len(remaining_nodes) == 0:
                break
            
            # If there are new nodes to remove, but they're not connected to us locally,
            # we add a random node to the nodes to remove list
            if len(nodes_to_remove) == a:
                nodes_to_remove += [random.choice(remaining_nodes)]
                
        # These are the nodes we want to randomly choose from to complete our set of
        # nodes that we are going to remove at each localized attack step
        nodes_to_maybe_remove = set(nodes_to_remove) - set(nodes_to_def_remove)
        
        # ... but, we should make sure we actually can remove the number of nodes we want to
        num_nodes_random_remove = int(STEP*N) - len(nodes_to_def_remove)
        max_nodes_can_remove = G.number_of_nodes() - len(nodes_to_def_remove)
        num_nodes_random_remove = min(num_nodes_random_remove, max_nodes_can_remove)
        
        # ... and now, we remove the appropriate number of nodes from the graph
        nodes_to_def_remove += random.sample(nodes_to_maybe_remove, num_nodes_random_remove)
        G.remove_nodes_from(nodes_to_def_remove)
        f += len(nodes_to_def_remove) / float(N)
        return (G, f, None)
    
    if perturb_strategy == "localized_expanding_hole":
        #Note: This is currently only valid for lattice graphs
        # MUST IMPLEMENT A 'find center node' function for it to otherwise be valid
        
        #ALSO NOTE: Untested
        assert graph_type in ["lattice"]
        assert dim is 2
        
        if state == None:
            N = G.number_of_nodes()
            center_node = tuple([int(math.sqrt(N)/2)] * 2)
            state = [center_node]
            curr_nodes = [center_node]
            while len(set(G.nodes()) - set(state)) > 0:
                neighbors = [G.neighbors(node) for node in state]
                neighbors = reduce(lambda x,y: x+y, neighbors)
                neighbors = list(set(neighbors))
                neighbors = [x for x in neighbors if x not in state]
                state += neighbors
                curr_nodes = neighbors
        
        print(len(state))
        print(state)
        print(G.number_of_nodes())
        assert len(state) == G.number_of_nodes()
        nodes_to_remove = state[0:int(round(STEP*N))]
        G.remove_nodes_from(nodes_to_remove)
        f += len(nodes_to_remove) / float(N)
        state = state[int(round(STEP*N)):]
        return (G, f, state)

    if perturb_strategy == "SP":
        raise Exception("Shortest path perturbation is currently unimplemented")
        # shortest path attack
        # iteratively pick two nodes and remove all nodes on the shortest path between them
        # note that we might remove slightly more than STEP*N nodes.
        # that's fine, just make sure to update f accordingly

    if perturb_strategy == "BC":
        raise Exception("BC perturbation is unimplemented")
        
        
        
        # betweenness centrality attack - remove nodes according to their BC rank
        # (non adaptively, i.e. rank is computed in the beginning and never changes)

    if perturb_strategy == "motion":
        raise Exception("Coordinate motion perturbation is unimplemented")
        # another attack idea: move nodes around

'''
This function contains the simulation logic for the effect of perturbation on
routing at greed level num_lookahead, average degree k (if not grid)
'''
def perturb_sim(num_lookahead, k, graph_type, perturb_strategy,
                N=1000, dim=2, STEP=0.01, num_routes = 1000):
    
        rand_seed = random.getrandbits(15)

        assert graph_type in ["lattice", "geometric"] #hyperbolic doesn't currently work

        print("Running {} graph sim for N:{}, num_lookahead:{}, dim:{}".format(
                    graph_type, N, num_lookahead, dim))

        G = generateGraph(graph_type, N, k, dim)
        # plotGraph(G, graph_type)

        N = G.number_of_nodes()
        
        statistics_list = []

        f = 0
        i = 0
        state = None
        while not math.isclose(1, f) and f < 1:

            # Pickles graph to file (for quicker read by threads spawned by ray)
            graph_key = random.getrandbits(100)
            graph_path = "ray_graphs/graph_{}".format(graph_key)
            nx.write_gpickle(G, graph_path)

            print("Greedily routing.... for f={}".format(f))

            # Generates list of src, trg of length len(num_routes)
            
            if len(G.nodes()) == 0:
                break
            
            src_list = [random.choice(G.nodes()) for _ in range(num_routes)]
            
           #  if len([x for x in G.nodes() if x!=src_list[i] ]) for i in range(num_routes)}
            # THIS IS WHERE THE ISSUE OCCURS
            trg_list = [random.choice([x for x in G.nodes() if x!=src_list[i] ])
                        for i in range(num_routes)]

            # Feeds the appropriate arguments for each call of performRoute into
            # a list, which can then be used with ray
            arg_list = zip([graph_path]*num_routes, [num_lookahead]*num_routes,
                           src_list, trg_list, [graph_type]*num_routes)
            # Calls the ray remote for all argument lists in arg_list
            graph_results = ray.get([performRoute.remote(*args)
                                     for args in arg_list])

            # TODO: Maybe should implement the caching to same trg --> independence violated?
            
            # We should always have tried to route `num_routes` times by now
            assert len(graph_results) == num_routes
            
            
            results_num_lookahead = ["SP"] + list(range(1, num_lookahead+1))
            
            
            for num in results_num_lookahead:
                # Filters result for given ns_greedy lookahead
                filtered_results = [x[num] for x in graph_results]

                # Filters successful routes from all tried routes
                succ_results = list(filter(lambda x : x != None, filtered_results))
                succ_sq_results = [x ** 2 for x in succ_results]
                print("For num_lookahead={}".format(num))
                print("Successfully routed {} times".format(len(succ_results)))

                # Calculates the success rate (and std_dev)
                succ_rate = len(succ_results) / len(graph_results)
                succ_std = math.sqrt(succ_rate * (1 - succ_rate) / math.sqrt(num_routes)) # Note: Make sure this is correct (always)
                print("Success rate is {} +/- {}".format(succ_rate, succ_std))

                # Calculates the average path length, std_dev
                if len(succ_results) > 0:
                    avg_len = reduce(lambda x,y : x+y, succ_results)
                    avg_len = avg_len / len(succ_results)
                    avg_sq_len = reduce(lambda x,y : x+y, succ_sq_results) / len(succ_results)
                    std_dev = math.sqrt(avg_sq_len - (avg_len ** 2))
                    std_dev = std_dev / (2 * math.sqrt(len(succ_results)))
                    # Note: the below two lines are new as of 6th April 2018
                    efficiency = [1./x for x in succ_results] + [0] * (len(graph_results) - len(succ_results))
                    efficiency = reduce(lambda x,y : x + y, efficiency) / len(graph_results)
                else:
                    avg_len, std_dev = "NA", "NA"
                    efficiency = 0 # if no paths route, then the network is perfectly inefficient
                print("Average path length is {} +/- {} (2SD)".format(avg_len, 2*std_dev))
                print("Average routing efficiency is {}".format(efficiency))
                
                if len(list(nx.connected_component_subgraphs(G))) > 0:
                    giant_component_size = len(list(max(nx.connected_component_subgraphs(G), key=len))) / N
                else:
                    giant_component_size = 0
                    
                print("Size of giant component is {}".format(giant_component_size))
                
                statistics_list.append([num, succ_rate, succ_std, avg_len, std_dev, f, efficiency, giant_component_size])

            # plots the graph state every tenth frame
            if graph_type == "lattice" and i % 10 == 0:
                heatmap_title = "./data/heatmap_N_{}_strat_{}_graph_{}_STEP_{}_SEED_{}.csv".format(N, perturb_strategy, graph_type, STEP, rand_seed)
                with open(heatmap_title, "a") as file:
                    if len(G.nodes()) > 0:
                        x, y = zip(*G.nodes())
                        labelled_nodes = zip(x, y, [i] * len(x))
                        writer = csv.writer(file)
                        writer.writerows(labelled_nodes)
            i += 1
                
            # perturbs G according to perturb_strategy, by amount STEP
            G, f, state = perturbGraph(G, N, perturb_strategy, f, STEP, state)
            
                
            #plotGraph(G, graph_type)
            
        # plots the final graph state
        if graph_type == "lattice":
                with open(heatmap_title, "a") as file:
                    if len(G.nodes()) > 0:
                        x, y = zip(*G.nodes())
                        labelled_nodes = zip(x, y, [i] * len(x))
                        writer = csv.writer(file)
                        writer.writerows(labelled_nodes)
            
        file_title = "N_{}_strat_{}_STEP_{}_graph_{}_numroutes_{}_dim_{}_k_{}_numlookahead_{}_rand_{}.csv".format(
                     N, perturb_strategy, STEP, graph_type, num_routes, dim, k, num_lookahead+1, rand_seed)
        
        with open("./data/" + file_title, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["num_look", "succ_rate", "succ_std", "avg_len", "avg_std_dev", "f", "avg_efficiency", "giant_comp_size"])
            writer.writerows(statistics_list)
            
if __name__ == "__main__":
    
    ray.init()
    
    num_lookahead = int(sys.argv[1])
    k             = int(sys.argv[2])
    graph_type    = sys.argv[3]
    N             = int(sys.argv[4])
    dim           = int(sys.argv[5])
    STEP          = float(sys.argv[6])
    perturb_strat = sys.argv[7]
    num_routes    = int(sys.argv[8])
    
    
    perturb_sim(num_lookahead=num_lookahead, k=k, graph_type=graph_type,
                N=N, dim=dim, STEP=STEP, perturb_strategy=perturb_strat,
                num_routes=num_routes)