# create a geometric graph, G
radius = math.sqrt(k/(math.pi*N))
G = nx.random_geometric_graph(n=1000, radius=0.1, dim=2, pos=None)

G = nx.grid_graph(dim=(1000), periodic=False) # keep it undirected

# also ,random hyperbolic graphs -- bc embed real-world networks well

N = G.number_of_nodes()


STEP = 0.01
f = 0
while f<1:
    # remove STEP nodes from the network according to some strategy

    # random removal
    G.remove_nodes_from(random.sample(G.nodes(),int(STEP*N)))

    # localized attack
    random_node = random.choice(G.nodes())
    nodes_removed = 0
    while nodes_removed < int(STEP*N):
        # remove random_node, its neighbors, their neighbors... until guard is False.

    # shortest path attack
    # iteratively pick two nodes and remove all nodes on the shortest path between them
    # note that we might remove slightly more than STEP*N nodes.
    # that's fine, just make sure to update f accordingly

    # betweenness centrality attack - remove nodes according to their BC rank
    # (non adaptively, i.e. rank is computed in the beginning and never changes)

    # another attack idea: move nodes around

    f = f + nodes_removed/float(N)

    # iteratively, randomly pick source and targer and record:
    # 1. average length of successful greedy(k) paths
    # 2. average sucess rate
