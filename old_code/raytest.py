import greedy_lattice_test as gr
import ray
import time
import networkx as nx
import csv
import sys

ray.init()

many = sys.argv[1]



def f():
    grid_input = [100,100]
    G2 = nx.grid_graph(grid_input, periodic=False)
    G2 = G2.to_directed()
    G2.add_edges_from([[(0,0), (3,3)]])
    return gr.compute_not_so_greedy_route(G2, (1,1), (50,50), num=2)[1]

start = time.time()

# Execute f serially.
results = [f() for i in range(100)]
print()

middle = time.time()
print("Finished serial case in {} seconds".format(middle - start))

@ray.remote
def f2():
    grid_input = [100,100]
    G2 = nx.grid_graph(grid_input, periodic=False)
    G2 = G2.to_directed()
    G3 = G2
    G3.add_edges_from([[(0,0), (3,3)]])
    return gr.compute_not_so_greedy_route(G3, (1,1), (50,50), num=2)[1]

# Execute f in parallel.
object_ids = [f2.remote() for i in range(100)]
results = ray.get(object_ids)
end = time.time()
print("Finished parallel case in {} seconds".format(end - middle))

with open("ray.csv", "a") as file_open:
    w = csv.writer(file_open)
    time_result = [middle-start, end-middle]
    w.writerow(time_result)
