import greedy_lattice_test as gr
import ray
import time
import networkx as nx
import csv

ray.init()


grid_input = [10,10]
G2 = nx.grid_graph(grid_input, periodic=False)
G2 = G2.to_directed()
G2.add_edges_from([[(0,0), (3,3)]])

def f(G2):
    return gr.compute_not_so_greedy_route(G2, (1,1), (3,3), num=3)[1]

start = time.time()

# Execute f serially.
results = [f(G2) for i in range(5000)]
print()

middle = time.time()
print("Finished serial case in {} seconds".format(middle - start))

@ray.remote
def f2(G2):
    return gr.compute_not_so_greedy_route(G2, (1,1), (3,3), num=3)[1]

# Execute f in parallel.
object_ids = [f2.remote(G2) for i in range(5000)]
results = ray.get(object_ids)
end = time.time()
print("Finished parallel case in {} seconds".format(end - middle))

time_result = [middle-start, end-middle]
