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
def perturbGraph(G, N, perturb_strategy, f, STEP):
    assert perturb_strategy in ["none", "random", "localized"]

    if perturb_strategy == "none":
        return(G, f+STEP)

    if perturb_strategy == "random":
        G.remove_nodes_from(random.sample(G.nodes(), int(STEP*N)))
        f += (STEP*N) / N
        return (G, f)

    if perturb_strategy == "localized":
        # localized attack
        random_node = random.choice(G.nodes())
        nodes_to_remove = [random_node]
        while len(nodes_to_remove) < int(STEP*N):
            a = len(nodes_to_remove)
            nodes_to_def_remove = nodes_to_remove
            neighbors = [G.neighbors(node) for node in nodes_to_remove]
            neighbors = set(reduce(lambda x,y : x + y, neighbors))
            nodes_to_remove = list(set(nodes_to_remove) | neighbors)
            if len(nodes_to_remove) == a:
                nodes_to_remove += [random.choice(list(set(G.nodes()) - set(nodes_to_remove)))]
                
        nodes_to_maybe_remove = set(nodes_to_remove) - set(nodes_to_def_remove)
        num_nodes_random_remove = int(STEP*N) - len(nodes_to_def_remove)
        nodes_to_def_remove = random.sample(nodes_to_maybe_remove, num_nodes_random_remove)
        G.remove_nodes_from(nodes_to_def_remove)
        f += int(STEP*N) / float(N)
        return (G, f)

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

        assert graph_type in ["lattice", "geometric"] #hyperbolic doesn't currently work

        print("Running {} graph sim for N:{}, num_lookahead:{}, dim:{}".format(
                    graph_type, N, num_lookahead, dim))

        G = generateGraph(graph_type, N, k, dim)
        # plotGraph(G, graph_type)

        N = G.number_of_nodes()
        
        statistics_list = []

        f = 0
        while not math.isclose(1, f):

            # Pickles graph to file (for quicker read by threads spawned by ray)
            graph_key = random.getrandbits(100)
            graph_path = "ray_graphs/graph_{}".format(graph_key)
            nx.write_gpickle(G, graph_path)

            print("Greedily routing.... for f={}".format(f))

            # Generates list of src, trg of length len(num_routes)
            src_list = [random.choice(G.nodes()) for _ in range(num_routes)]
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
            for num in range(1, num_lookahead+1):
                # Filters result for given ns_greedy lookahead
                filtered_results = [x[num] for x in graph_results]

                # Filters successful routes from all tried routes
                succ_results = list(filter(lambda x : x != None, filtered_results))
                succ_sq_results = [x ** 2 for x in succ_results]
                print("For num_lookahead={}".format(num))
                print("Successfully routed {} times".format(len(succ_results)))

                # Calculates the success rate (and std_dev)
                succ_rate = len(succ_results) / len(graph_results)
                succ_std = succ_rate * (1 - succ_rate) / math.sqrt(num_routes)
                print("Success rate is {} +/- {}".format(succ_rate, succ_std))

                # Calculates the average path length, std_dev
                if len(succ_results) > 0:
                    avg_len = reduce(lambda x,y : x+y, succ_results)
                    avg_len = avg_len / len(succ_results)
                    avg_sq_len = reduce(lambda x,y : x+y, succ_sq_results) / len(succ_results)
                    std_dev = math.sqrt(avg_sq_len - (avg_len ** 2))
                    std_dev = std_dev / (2 * math.sqrt(len(succ_results)))
                else:
                    avg_len, std_dev = "NA", "NA"
                print("Average path length is {} +/- {} (2SD)".format(avg_len, 2*std_dev))
                
                statistics_list.append([num, succ_rate, succ_std, avg_len, std_dev, f])

            # perturbs G according to perturb_strategy, by amount STEP
            G, f = perturbGraph(G, N, perturb_strategy, f, STEP)
            plotGraph(G, graph_type)
            
        file_title = "N_{}_strat_{}_STEP_{}_graph_{}_numroutes_{}_dim_{}_k_{}_numlookahead_{}.csv".format(
                     N, perturb_strategy, STEP, graph_type, num_routes, dim, k, num_lookahead+1)
        
        with open("./data/" + file_title, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["num_look", "succ_rate", "succ_std", "avg_len", "avg_std_dev", "f"])
            writer.writerows(statistics_list)


if __name__ == "__main__":
    
    ray.init()
    #perturb_sim(num_lookahead=3, k=50, graph_type="geometric",
    #           N=10000, dim=2, STEP=0.01, perturb_strategy="random",
    #          num_routes=10000)

    perturb_sim(num_lookahead=2, k=50, graph_type="geometric",
                N=1000, dim=2, STEP=0.01, perturb_strategy="localized",
                num_routes=100)